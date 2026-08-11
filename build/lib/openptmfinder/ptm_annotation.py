"""
ptm_annotation.py — iPTMnet-based annotation and position rescue of PTM sites.

Runs BEFORE the statistical analysis (whole-proteome branch only):

  1. Unique modified sites are collected from the PSM table (annotation is
     per unique site, NOT per PSM — this is orders of magnitude faster than
     row-wise apply/explode on the full PSM table).
  2. iPTMnet substrate entries are fetched per protein (multithreaded).
     Proteins that return nothing (typically obsolete/secondary UniProt
     accessions, which iPTMnet does not know) are resolved through the
     SHARED UniProt ID-mapping cache (<output_dir>/uniprot_idmap.csv,
     computed once and reused by the dbPTM/SIGNOR annotations as well):
     iPTMnet is queried with the current accession, and the rows are
     relabelled with the ORIGINAL accession so they merge with the
     experimental table.
  3. Typed matching: a site is 'in_iPTM' if an iPTMnet entry of the SAME
     normalized PTM class sits at the exact position; 'perhapse_*' describes
     the nearest same-class entry within +/- window residues.
  4. Sequence-based rescue: the position is validated against the FASTA
     sequence using per-class allowed residues. Invalid positions are moved
     to the iPTMnet suggestion (if any) or to the nearest chemically
     plausible residue within the window.

Output columns (one row per unique site):
    id_prot, position_in_protein, modified_peptide_x, mods,
    in_iPTM, perhapse_in_iPTM, perhapse_position, perhapse_ptm_type,
    rescued_position, rescued_ptm_type, rescue_shift

Python 3.10 compatible.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

try:  # package import
    from .dbconnect import fetch_protein, apply_idmap, resolve_unmapped_ids
    from .sage_preprocessing import load_fasta_dict
except ImportError:  # standalone usage
    from dbconnect import fetch_protein, apply_idmap, resolve_unmapped_ids
    from sage_preprocessing import load_fasta_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modification-class normalization
# ---------------------------------------------------------------------------

MOD_MAPPING = {
    'Phospho': 'Phosphorylation',
    'Acetyl': 'Acetylation',
    'Methyl': 'Methylation',
    'di-Methylation': 'Methylation',
    'HexNAc@S': 'O-Glycosylation',
    'HexNAc2@S': 'O-Glycosylation',
    'Fuc@S': 'O-Glycosylation',
    'Hex(1)HexNAc(1)@S': 'O-Glycosylation',
    'Hex@S': 'O-Glycosylation',
    'HexNAc@T': 'O-Glycosylation',
    'HexNAc2@T': 'O-Glycosylation',
    'Fuc@T': 'O-Glycosylation',
    'Hex(1)HexNAc(1)@T': 'O-Glycosylation',
    'Hex@T@N': 'N-Glycosylation',
    'HexNAc@N': 'N-Glycosylation',
    'HexNAc2@N': 'N-Glycosylation',
    'Fuc@N': 'N-Glycosylation',
    'Hex(1)HexNAc(1)@N': 'N-Glycosylation',
    'Hex@N': 'N-Glycosylation',
    'pyrophospho': 'Phosphorylation',
    'Phospho+PL': 'Phosphorylation',
    'tri-Methylation': 'Methylation',
    'dihydroxy': 'Dihydroxylation',
    'SNO': 'S-Nitrosylation',
    'GlyGly': 'Ubiquitination',
    'Myristoyl': 'Myristoylation',
    'Farnesyl': 'Farnesylation',
}

# Substring matching is order-dependent: longer keys first so that e.g.
# 'di-Methylation' is checked before 'Methyl'.
_MOD_KEYS = sorted(MOD_MAPPING, key=len, reverse=True)

RESIDUE_RULES = {
    'Phosphorylation': frozenset('STY'),
    'O-Glycosylation': frozenset('ST'),
    'N-Glycosylation': frozenset('N'),
    'S-Nitrosylation': frozenset('C'),
    'Ubiquitination': frozenset('K'),
    'Myristoylation': frozenset('G'),
    'Farnesylation': frozenset('C'),
}


def normalize_mod(mod):
    """Map a raw modification label to a broad PTM class (substring match,
    longest key first). Unmapped labels are returned unchanged."""
    if not isinstance(mod, str):
        return None
    for key in _MOD_KEYS:
        if key in mod:
            return MOD_MAPPING[key]
    return mod


def get_allowed_residues(mod):
    """Allowed residues for a PTM class; None = no rule (position unchecked)."""
    return RESIDUE_RULES.get(normalize_mod(mod))


# ---------------------------------------------------------------------------
# iPTMnet fetching (multithreaded, reuses dbconnect.fetch_protein)
# ---------------------------------------------------------------------------

def _fetch_iptmnet_parallel(ids, session, max_workers, query_plan=None):
    """Query iPTMnet for each id in parallel.

    query_plan: {original_id: query_id}; when the query accession differs
    from the original one (UniProt-remapped obsolete ids), the returned
    rows are relabelled with the original id (sub_form column).

    Returns ({original_id: DataFrame}, [original ids without data]).
    """
    plan = query_plan or {p: p for p in ids}
    fetched = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_protein, plan[prot], session): prot
                   for prot in ids}
        for future in as_completed(futures):
            prot = futures[future]
            try:
                df = future.result()
            except Exception as e:
                logger.warning(f"iPTMnet fetch failed for {prot}: {e}")
                continue
            if df is not None and not df.empty:
                if plan[prot] != prot:
                    df['sub_form'] = prot  # relabel to the original accession
                fetched[prot] = df
    missing = [p for p in ids if p not in fetched]
    return fetched, missing


def fetch_iptmnet_long(protein_ids, max_workers=10, output_dir=None):
    """Fetch iPTMnet substrate entries for all proteins in parallel.

    Obsolete/secondary UniProt accessions (unknown to iPTMnet) go through
    the SHARED UniProt ID-mapping cache (<output_dir>/uniprot_idmap.csv,
    see dbconnect.apply_idmap / resolve_unmapped_ids): proteins already
    resolved by any previous database annotation are queried with the
    current accession right away; the remaining misses are resolved through
    the API once and retried. Rows fetched with a mapped accession are
    relabelled with the ORIGINAL id.

    Returns a long DataFrame: id_prot, iptm_position (int), iptm_type (str).
    """
    ids = [str(p) for p in pd.unique(pd.Series(list(protein_ids))) if pd.notna(p)]
    query_plan = apply_idmap(ids, output_dir)  # {original_id: query_id}
    remapped = {p: q for p, q in query_plan.items() if q != p}
    if remapped:
        logger.info(f"UniProt ID-mapping cache: querying iPTMnet with "
                    f"current accessions for {len(remapped)} proteins: {remapped}")
    with requests.Session() as session:
        session.headers.update({"Accept": "text/plain"})
        fetched, missing = _fetch_iptmnet_parallel(ids, session, max_workers,
                                                   query_plan)

        # Only ids queried with their OWN accession are candidates for
        # remapping (already-remapped misses would fail again identically).
        retry_src = [p for p in missing if query_plan[p] == p]
        if retry_src:
            mapping = resolve_unmapped_ids(retry_src, output_dir)
            if mapping:
                logger.info(f"UniProt ID mapping resolved {len(mapping)} of "
                            f"{len(retry_src)} proteins without iPTMnet data: "
                            f"{mapping}")
                refetched, still = _fetch_iptmnet_parallel(
                    sorted(mapping), session, max_workers, mapping)
                fetched.update(refetched)
                if still:
                    logger.info(f"No iPTMnet data even after ID remapping: "
                                f"{still}")
            else:
                logger.info(f"UniProt ID mapping found no current accession "
                            f"for {len(retry_src)} proteins without iPTMnet "
                            f"data (left unannotated).")

    df_list = list(fetched.values())
    if not df_list:
        logger.warning("No data fetched from iPTMnet.")
        return pd.DataFrame(columns=['id_prot', 'iptm_position', 'iptm_type'])

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.dropna(subset=['site'])
    out = pd.DataFrame({
        'id_prot': df_all['sub_form'].astype(str),
        'iptm_position': df_all['site'].astype(str).str[1:].astype(int),
    })
    out['iptm_type'] = (df_all['ptm_type'].astype(str)
                        if 'ptm_type' in df_all.columns else None)
    out['iptm_norm'] = out['iptm_type'].map(normalize_mod)
    return out


# ---------------------------------------------------------------------------
# Typed site matching (vectorized)
# ---------------------------------------------------------------------------

def _match_iptmnet(sites: pd.DataFrame, iptm: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add in_iPTM / perhapse_* columns to the unique-sites table.

    Typed matching: only iPTMnet entries whose normalized PTM class equals
    the site's normalized class are considered (an exact-position entry of a
    DIFFERENT PTM type does not make the site 'known').
    """
    sites = sites.copy()
    sites['in_iPTM'] = False
    sites['perhapse_in_iPTM'] = False
    sites['perhapse_position'] = np.nan
    sites['perhapse_ptm_type'] = None

    if iptm.empty:
        return sites

    # NB: merge resets the index, so carry the site row id explicitly.
    left = sites[['id_prot', 'position_in_protein', 'norm_mod']].copy()
    left['site_row'] = sites.index
    m = left.merge(iptm, on='id_prot', how='inner')
    # typed match only
    m = m[m['norm_mod'] == m['iptm_norm']]
    if m.empty:
        return sites

    m['dist'] = (m['iptm_position'] - m['position_in_protein'].astype(float)).abs()

    exact = m.groupby('site_row')['dist'].min()
    sites.loc[exact.index[exact == 0], 'in_iPTM'] = True

    near = m[m['dist'] <= window]
    if not near.empty:
        idx_min = near.groupby('site_row')['dist'].idxmin()
        closest = near.loc[idx_min]
        rows = closest['site_row'].to_numpy()
        sites.loc[rows, 'perhapse_in_iPTM'] = True
        sites.loc[rows, 'perhapse_position'] = \
            closest['iptm_position'].astype(float).to_numpy()
        sites.loc[rows, 'perhapse_ptm_type'] = closest['iptm_type'].to_numpy()
    return sites


# ---------------------------------------------------------------------------
# Sequence-based rescue (vectorized per protein)
# ---------------------------------------------------------------------------

def _rescue_from_sequence(sites: pd.DataFrame, fasta_dict: dict, window: int) -> pd.DataFrame:
    """Validate/rescue positions against FASTA sequences.

    Logic (preserved from the original localize_ptm_from_sequence):
      - protein missing or position out of range  -> NaN (caller fills original);
      - class without a residue rule              -> keep original position;
      - current residue valid                     -> keep original position;
      - invalid + iPTMnet suggestion              -> take the suggestion;
      - invalid + no suggestion                   -> nearest allowed residue
        within +/- window (if none: keep original).
    rescued_ptm_type follows the original semantics: normalized class for
    sequence-based decisions, raw iPTMnet type for the perhapse fallback.
    """
    sites = sites.copy()
    sites['rescued_position'] = sites['position_in_protein'].astype(float)
    sites['rescued_ptm_type'] = sites['norm_mod']

    pos0_all = pd.to_numeric(sites['position_in_protein'],
                             errors='coerce').to_numpy(dtype=float) - 1
    has_per = sites['perhapse_position'].notna().to_numpy()
    allowed = sites['norm_mod'].map(lambda m: RESIDUE_RULES.get(m))

    for prot, idx in sites.groupby('id_prot').groups.items():
        idx = np.asarray(list(idx))
        seq = fasta_dict.get(str(prot))
        if seq is None:
            sites.loc[idx, 'rescued_position'] = np.nan
            sites.loc[idx, 'rescued_ptm_type'] = None
            continue

        arr = np.fromiter(seq, dtype='U1', count=len(seq))
        n = len(arr)
        pos0 = pos0_all[sites.index.get_indexer(idx)]
        in_range = (pos0 >= 0) & (pos0 < n)

        # out-of-range positions -> NaN (original code returned None, None)
        bad = idx[~in_range]
        if len(bad):
            sites.loc[bad, 'rescued_position'] = np.nan
            sites.loc[bad, 'rescued_ptm_type'] = None

        # group the in-range rows by their residue rule
        sub_allowed = allowed.loc[idx[in_range]]
        for rkey, sub_idx in sub_allowed.groupby(sub_allowed).groups.items():
            if rkey is None:
                continue  # no residue rule -> keep original
            sub_idx = np.asarray(list(sub_idx))
            # NB: index into the FULL arrays — pos0 is the protein-level subset
            p0 = pos0_all[sites.index.get_indexer(sub_idx)].astype(int)
            residues = np.fromiter(rkey, dtype='U1', count=len(rkey))
            allowed_pos = np.flatnonzero(np.isin(arr, residues))

            cur_valid = np.isin(arr[p0], residues)
            per_here = has_per[sites.index.get_indexer(sub_idx)]

            # invalid current residue + iPTMnet suggestion -> take suggestion
            take_per = (~cur_valid) & per_here
            if take_per.any():
                rows = sub_idx[take_per]
                sites.loc[rows, 'rescued_position'] = \
                    sites.loc[rows, 'perhapse_position'].astype(float)
                sites.loc[rows, 'rescued_ptm_type'] = \
                    sites.loc[rows, 'perhapse_ptm_type']

            # invalid current residue + no suggestion -> nearest allowed residue
            need_near = (~cur_valid) & (~per_here)
            if need_near.any() and len(allowed_pos):
                rows = sub_idx[need_near]
                q = p0[need_near]
                ins = np.searchsorted(allowed_pos, q)
                right = np.clip(ins, 0, len(allowed_pos) - 1)
                left = np.clip(ins - 1, 0, len(allowed_pos) - 1)
                d_right = np.abs(allowed_pos[right] - q)
                d_left = np.abs(allowed_pos[left] - q)
                best = np.where(d_left <= d_right, allowed_pos[left], allowed_pos[right])
                dist = np.minimum(d_left, d_right)
                moved = dist <= window
                if moved.any():
                    sites.loc[rows[moved], 'rescued_position'] = \
                        (best[moved] + 1).astype(float)
    return sites


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def annotate_sites_with_iptmnet(psm_df: pd.DataFrame,
                                fasta_file: str,
                                window: int = 7,
                                max_workers: int = 10,
                                modification_col: str = 'Modification',
                                reference_label: str = 'reference',
                                output_dir: str = None) -> pd.DataFrame:
    """Annotate unique modified sites with iPTMnet + sequence rescue.

    Returns one row per unique (id_prot, position_in_protein,
    modified_peptide_x) site of a NON-reference modification.
    """
    keys = ['id_prot', 'position_in_protein', 'modified_peptide_x']
    missing = [c for c in keys + [modification_col] if c not in psm_df.columns]
    if missing:
        raise KeyError(f"Required columns missing for iPTMnet annotation: {missing}")

    mod_mask = psm_df[modification_col] != reference_label
    sites = (psm_df.loc[mod_mask, keys + [modification_col]]
             .dropna(subset=['position_in_protein'])
             .drop_duplicates(subset=keys)
             .reset_index(drop=True))
    if sites.empty:
        logger.warning("No modified sites to annotate.")
        return pd.DataFrame()

    sites['mods'] = sites[modification_col].astype(str).str.split('@').str[0]
    sites['norm_mod'] = sites['mods'].map(normalize_mod)
    unmapped = sorted(sites.loc[sites['norm_mod'] == sites['mods'], 'mods'].unique())
    if unmapped:
        logger.info(f"Modifications without a class mapping (kept as-is): {unmapped}")

    protein_ids = sites['id_prot'].unique()
    logger.info(f"iPTMnet annotation: {len(sites)} unique sites, "
                f"{len(protein_ids)} proteins, window=+/-{window}")

    iptm = fetch_iptmnet_long(protein_ids, max_workers=max_workers,
                              output_dir=output_dir)
    sites = _match_iptmnet(sites, iptm, window=window)

    fasta_dict = load_fasta_dict(fasta_file)
    sites = _rescue_from_sequence(sites, fasta_dict, window=window)

    sites['rescue_shift'] = (sites['rescued_position'] -
                             sites['position_in_protein'].astype(float))
    n_moved = int((sites['rescue_shift'].fillna(0) != 0).sum())
    logger.info(f"iPTMnet annotation done: in_iPTM={int(sites['in_iPTM'].sum())}, "
                f"perhapse={int(sites['perhapse_in_iPTM'].sum())}, "
                f"positions moved={n_moved}")

    return sites[['id_prot', 'position_in_protein', 'modified_peptide_x', 'mods',
                  'in_iPTM', 'perhapse_in_iPTM', 'perhapse_position',
                  'perhapse_ptm_type', 'rescued_position', 'rescued_ptm_type',
                  'rescue_shift']]
