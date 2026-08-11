"""
dbptm_annotation.py — dbPTM-based annotation of PTM sites.

dbPTM (https://biomics.lab.nycu.edu.tw/dbPTM/) has no public REST API and no
per-protein query endpoint in the classical sense, but every protein has a
server-rendered page

    https://biomics.lab.nycu.edu.tw/dbPTM/info.php?id=<UniProt entry name>

whose "Experimental Post-Translational Modification Sites" table (tab
<div id="exp">) contains exactly what the full-database download provides per
protein: modified position, PTM type, substrate peptide and PubMed reference.
Fetching these pages for the proteins of interest only is orders of magnitude
lighter than downloading the per-modication-type full dumps.

UniProt accession -> dbPTM entry name resolution:
  1. from the FASTA headers (sp|ACC|ENTRY_NAME / tr|ACC|ENTRY_NAME);
  2. fallback: dbPTM "search by database ID" endpoint
     (search_result.php?search_type=db_id with db_type=ac), which accepts
     UniProt accessions and returns the corresponding entry name.

Runs BEFORE the statistical analysis (whole-proteome branch only), after /
independently of the iPTMnet annotation:

  1. Unique modified sites are collected from the PSM table (annotation is
     per unique site, NOT per PSM).
  2. dbPTM experimental-site tables are fetched per protein (multithreaded).
  3. Typed matching: a site is 'in_dbPTM' if a dbPTM entry of the SAME
     normalized PTM class sits at the exact position; 'perhapse_*' describes
     the nearest same-class entry within +/- window residues (same semantics
     as the iPTMnet annotation in ptm_annotation.py).

Output columns (one row per unique site):
    id_prot, position_in_protein, modified_peptide_x, mods,
    in_dbPTM, dbptm_ptm_types, dbptm_pmids,
    perhapse_in_dbPTM, perhapse_position_dbptm, perhapse_ptm_type_dbptm

Python 3.10 compatible.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:  # package import
    from .ptm_annotation import normalize_mod
    from .dbconnect import apply_idmap, resolve_unmapped_ids
except ImportError:  # standalone usage
    from ptm_annotation import normalize_mod
    from dbconnect import apply_idmap, resolve_unmapped_ids

logger = logging.getLogger(__name__)

DBPTM_BASE = "https://biomics.lab.nycu.edu.tw/dbPTM"
DBPTM_INFO_URL = DBPTM_BASE + "/info.php"
DBPTM_SEARCH_URL = DBPTM_BASE + "/search_result.php"

# ---------------------------------------------------------------------------
# dbPTM PTM-type normalization
# ---------------------------------------------------------------------------
# dbPTM uses its own PTM-type vocabulary (e.g. 'O-linked Glycosylation',
# 'S-nitrosylation'). Map it onto the same broad classes that
# ptm_annotation.normalize_mod produces for the experimental mod labels, so
# that typed matching compares like with like. Exact (case-insensitive)
# matching is used on purpose: substring rules would misfire on
# 'Dephosphorylation', 'Carbamidation', 'Deamidation', 'Nitration', etc.
# Unknown types are returned unchanged (mirrors normalize_mod semantics).

DBPTM_TYPE_MAPPING = {
    'phosphorylation': 'Phosphorylation',
    'acetylation': 'Acetylation',
    'methylation': 'Methylation',
    'o-linked glycosylation': 'O-Glycosylation',
    'n-linked glycosylation': 'N-Glycosylation',
    'c-linked glycosylation': 'C-Glycosylation',
    's-nitrosylation': 'S-Nitrosylation',
    'nitrosylation': 'S-Nitrosylation',
    'nitration': 'Nitration',
    'ubiquitination': 'Ubiquitination',
    'sumoylation': 'SUMOylation',
    'neddylation': 'Neddylation',
    'myristoylation': 'Myristoylation',
    'farnesylation': 'Farnesylation',
    'geranylgeranylation': 'Geranylgeranylation',
    'palmitoylation': 'Palmitoylation',
    'sulfation': 'Sulfation',
    'hydroxylation': 'Hydroxylation',
    'dihydroxylation': 'Dihydroxylation',
    'adp-ribosylation': 'ADP-ribosylation',
    'citrullination': 'Citrullination',
    'oxidation': 'Oxidation',
    'amidation': 'Amidation',
    'glycation': 'Glycation',
    'glutathionylation': 'Glutathionylation',
    'crotonylation': 'Crotonylation',
    'succinylation': 'Succinylation',
    'lactylation': 'Lactylation',
    'malonylation': 'Malonylation',
    'butyrylation': 'Butyrylation',
    'glutarylation': 'Glutarylation',
    'formylation': 'Formylation',
    'carboxylation': 'Carboxylation',
    'disulfide bond': 'Disulfide bond',
    'gpi-anchor': 'GPI-anchor',
}


def normalize_dbptm_type(ptm_type):
    """Map a dbPTM modification type to the shared PTM-class vocabulary.
    Unmapped types are returned unchanged."""
    if not isinstance(ptm_type, str):
        return None
    return DBPTM_TYPE_MAPPING.get(ptm_type.strip().lower(), ptm_type.strip())


# ---------------------------------------------------------------------------
# UniProt accession -> dbPTM entry name
# ---------------------------------------------------------------------------

def load_fasta_entry_names(fasta_file: str) -> dict:
    """Parse UniProt-style FASTA headers into {accession: entry_name}.

    Handles 'sp|ACC|ENTRY_NAME ...' / 'tr|ACC|ENTRY_NAME ...' headers
    (including 'DECOY_sp|...' prefixes). Non-UniProt headers are skipped —
    their owners fall back to the dbPTM db_id search.
    """
    mapping = {}
    with open(fasta_file, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if not line.startswith('>'):
                continue
            header = line[1:].strip()
            m = re.match(r'^(?:DECOY_\|?|rev_)?(?:sp|tr)\|([^|]+)\|([^\s|]+)', header)
            if m:
                acc, entry = m.groups()
                mapping.setdefault(acc, entry)
    return mapping


def search_dbptm_entry_name(accession: str, session: requests.Session):
    """Resolve a UniProt accession to a dbPTM entry name via the dbPTM
    'search by database ID' endpoint (db_type=ac). Returns None if not found."""
    try:
        r = session.post(DBPTM_SEARCH_URL,
                         params={'search_type': 'db_id'},
                         data={'db_type': 'ac', 'db_value': accession},
                         timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"dbPTM db_id search failed for {accession}: {e}")
        return None

    soup = BeautifulSoup(r.text, 'html.parser')
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        for tr in rows:
            cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
            # Result rows look like: ID | UniProt AC(s) | Organism | PTM
            if len(cells) >= 2 and cells[0] and 'Oops' not in cells[0]:
                accs = re.split(r'[;\s]+', cells[1].replace('\xa0', ' '))
                if accession in accs:
                    return cells[0]
    return None


# ---------------------------------------------------------------------------
# Per-protein page fetching and parsing
# ---------------------------------------------------------------------------

def _parse_exp_table(html: str) -> pd.DataFrame:
    """Extract the 'Experimental Post-Translational Modification Sites' table
    (tab <div id="exp">) from a dbPTM info page. Returns an empty DataFrame
    if the page carries no experimental sites."""
    soup = BeautifulSoup(html, 'html.parser')
    div = soup.find('div', id='exp')
    if div is None:
        return pd.DataFrame()
    table = div.find('table')
    if table is None:
        return pd.DataFrame()
    try:
        df = pd.read_html(StringIO(str(table)))[0]
    except ValueError:
        return pd.DataFrame()
    if df.shape[1] < 5:
        return pd.DataFrame()

    # Column layout: Locations | Modification | Substrate peptide + SecStruct
    # | ASA | Reference (PMID) | (optional ortholog/cluster extras).
    df = df.iloc[:, :5]
    df.columns = ['location', 'dbptm_type', 'peptide_raw', 'asa', 'dbptm_pmid']

    # 'location' may carry isoform notes ('7 (in isoform 4)') — the leading
    # integer is the position on the canonical sequence.
    df['dbptm_position'] = pd.to_numeric(
        df['location'].astype(str).str.extract(r'^\s*(\d+)', expand=False),
        errors='coerce')
    df = df.dropna(subset=['dbptm_position'])
    if df.empty:
        return pd.DataFrame()
    df['dbptm_position'] = df['dbptm_position'].astype(int)

    # The peptide cell is glued to the secondary-structure string; the first
    # token is the substrate 15-mer ('-' padding at termini).
    pep = df['peptide_raw'].astype(str).str.split().str[0]
    df['dbptm_peptide'] = pep.where(pep.str.fullmatch(r'[A-Za-z-]+'), None)

    df['dbptm_pmid'] = (df['dbptm_pmid'].astype(str)
                        .str.extract(r'(\d+)', expand=False))
    return df[['dbptm_position', 'dbptm_type', 'dbptm_peptide', 'dbptm_pmid']]


def fetch_dbptm_protein(entry_name: str, session: requests.Session,
                        retries: int = 1, timeout: int = 60):
    """Fetch and parse the dbPTM info page of one protein (entry name).
    Returns a DataFrame or None (no record / fetch failed)."""
    url = f"{DBPTM_INFO_URL}?id={entry_name}"
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            if 'no any information' in r.text:
                return None
            df = _parse_exp_table(r.text)
            if df.empty:
                return None
            df['entry_name'] = entry_name
            return df
        except requests.exceptions.RequestException as e:
            if attempt >= retries:
                logger.warning(f"dbPTM fetch failed for {entry_name}: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Multithreaded fetch for a protein list
# ---------------------------------------------------------------------------

def fetch_dbptm_long(protein_ids, fasta_file: str, max_workers: int = 10,
                     output_dir: str = None) -> pd.DataFrame:
    """Fetch dbPTM experimental sites for all proteins in parallel.

    Obsolete/secondary UniProt accessions (no dbPTM entry under the old id)
    go through the SHARED UniProt ID-mapping cache
    (<output_dir>/uniprot_idmap.csv, see dbconnect.apply_idmap /
    resolve_unmapped_ids): proteins already resolved by any previous
    database annotation are searched with the current accession right away;
    the remaining misses are resolved through the API once and retried.
    Rows fetched via a mapped accession keep the ORIGINAL id_prot.

    Returns a long DataFrame: id_prot, dbptm_position (int), dbptm_type,
    dbptm_norm, dbptm_peptide, dbptm_pmid.
    """
    cols = ['id_prot', 'dbptm_position', 'dbptm_type', 'dbptm_norm',
            'dbptm_peptide', 'dbptm_pmid']
    ids = [str(p) for p in pd.unique(pd.Series(list(protein_ids))) if pd.notna(p)]
    if not ids:
        return pd.DataFrame(columns=cols)

    # Isoform accessions ('P04637-2') resolve through the canonical accession.
    native = {p: p.split('-')[0] for p in ids}

    # Shared UniProt ID-mapping cache: accession to use for dbPTM searches.
    acc_plan = apply_idmap(sorted(set(native.values())), output_dir)
    search_acc = {p: acc_plan[native[p]] for p in ids}
    remapped = {p: a for p, a in search_acc.items() if a != native[p]}
    if remapped:
        logger.info(f"UniProt ID-mapping cache: searching dbPTM with current "
                    f"accessions for {len(remapped)} proteins: {remapped}")

    fasta_entries = {}
    try:
        fasta_entries = load_fasta_entry_names(fasta_file)
    except OSError as e:
        logger.warning(f"Could not parse FASTA headers for entry names: {e}")

    id2entry = {p: fasta_entries[acc] for p, acc in native.items()
                if acc in fasta_entries}
    missing = [p for p in ids if p not in id2entry]
    logger.info(f"dbPTM: {len(id2entry)}/{len(ids)} entry names from FASTA headers, "
                f"{len(missing)} to resolve via dbPTM search.")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'OpenPtmFinder dbPTM annotation (academic research use)'})
    try:
        if missing:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(search_dbptm_entry_name, search_acc[p], session): p
                           for p in missing}
                for future in as_completed(futures):
                    p = futures[future]
                    try:
                        entry = future.result()
                    except Exception as e:
                        logger.warning(f"dbPTM entry-name search failed for {p}: {e}")
                        continue
                    if entry:
                        id2entry[p] = entry

        unresolved = [p for p in ids if p not in id2entry]
        if unresolved:
            logger.info(f"dbPTM: no entry name for {len(unresolved)} proteins "
                        f"(skipped): {unresolved[:10]}{'...' if len(unresolved) > 10 else ''}")

        entry2ids = {}
        for p, entry in id2entry.items():
            entry2ids.setdefault(entry, []).append(p)

        df_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_dbptm_protein, entry, session): entry
                       for entry in entry2ids}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    df = future.result()
                except Exception as e:
                    logger.warning(f"dbPTM parse failed for {entry}: {e}")
                    continue
                if df is None or df.empty:
                    continue
                for p in entry2ids[entry]:
                    sub = df.copy()
                    sub['id_prot'] = p
                    df_list.append(sub)

        # --- retry misses through the shared UniProt ID-mapping cache -----
        # Only proteins searched with their OWN accession are candidates
        # (already-remapped misses would fail again identically).
        covered = set()
        for sub in df_list:
            covered.update(sub['id_prot'].unique())
        retry_src = [p for p in ids
                     if p not in covered and search_acc[p] == native[p]]
        if retry_src:
            acc2orig = {}
            for p in retry_src:
                acc2orig.setdefault(native[p], []).append(p)
            mapping = resolve_unmapped_ids(sorted(acc2orig), output_dir)
            if mapping:
                logger.info(f"dbPTM: UniProt ID mapping resolved {len(mapping)} "
                            f"of {len(acc2orig)} proteins without dbPTM data: "
                            f"{mapping}")
                # new accession -> dbPTM entry name -> info page
                retry_entries = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(search_dbptm_entry_name, new, session): old
                               for old, new in mapping.items()}
                    for future in as_completed(futures):
                        old = futures[future]
                        try:
                            entry = future.result()
                        except Exception as e:
                            logger.warning(f"dbPTM entry-name search failed for "
                                           f"{mapping[old]}: {e}")
                            continue
                        if entry:
                            retry_entries.setdefault(entry, []).append(old)
                if retry_entries:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(fetch_dbptm_protein, entry, session): entry
                                   for entry in retry_entries}
                        for future in as_completed(futures):
                            entry = futures[future]
                            try:
                                df = future.result()
                            except Exception as e:
                                logger.warning(f"dbPTM parse failed for {entry}: {e}")
                                continue
                            if df is None or df.empty:
                                continue
                            for old in retry_entries[entry]:
                                for p in acc2orig[old]:
                                    sub = df.copy()
                                    sub['id_prot'] = p  # original accession
                                    df_list.append(sub)
    finally:
        session.close()

    if not df_list:
        logger.warning("No data fetched from dbPTM.")
        return pd.DataFrame(columns=cols)

    out = pd.concat(df_list, ignore_index=True)
    out['dbptm_norm'] = out['dbptm_type'].map(normalize_dbptm_type)
    return out[cols]


# ---------------------------------------------------------------------------
# Typed site matching (mirrors ptm_annotation._match_iptmnet)
# ---------------------------------------------------------------------------

def _match_dbptm(sites: pd.DataFrame, dbptm: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add in_dbPTM / dbptm_ptm_types / dbptm_pmids / perhapse_* columns to the
    unique-sites table.

    Typed matching: only dbPTM entries whose normalized PTM class equals the
    site's normalized class are considered for in_dbPTM / perhapse_*. The
    dbptm_ptm_types / dbptm_pmids columns aggregate ALL dbPTM entries (any
    class) at the exact position, as plain information.
    """
    sites = sites.copy()
    sites['in_dbPTM'] = False
    sites['dbptm_ptm_types'] = None
    sites['dbptm_pmids'] = None
    sites['perhapse_in_dbPTM'] = False
    sites['perhapse_position_dbptm'] = np.nan
    sites['perhapse_ptm_type_dbptm'] = None

    if dbptm.empty:
        return sites

    # --- aggregates at the exact position (any PTM class) ---
    agg = dbptm.groupby(['id_prot', 'dbptm_position']).agg(
        dbptm_ptm_types=('dbptm_type', lambda s: ';'.join(sorted(set(s.dropna().astype(str))))),
        dbptm_pmids=('dbptm_pmid', lambda s: ';'.join(sorted(set(s.dropna().astype(str))))),
    ).reset_index()
    exact_any = sites[['id_prot', 'position_in_protein']].copy()
    exact_any['site_row'] = sites.index
    exact_any = exact_any.merge(
        agg, left_on=['id_prot', 'position_in_protein'],
        right_on=['id_prot', 'dbptm_position'], how='inner')
    if not exact_any.empty:
        rows = exact_any['site_row'].to_numpy()
        sites.loc[rows, 'dbptm_ptm_types'] = exact_any['dbptm_ptm_types'].to_numpy()
        pm = exact_any['dbptm_pmids'].replace('', None)
        sites.loc[rows, 'dbptm_pmids'] = pm.to_numpy()

    # --- typed matching (same class only) ---
    left = sites[['id_prot', 'position_in_protein', 'norm_mod']].copy()
    left['site_row'] = sites.index
    m = left.merge(dbptm[['id_prot', 'dbptm_position', 'dbptm_type', 'dbptm_norm']],
                   on='id_prot', how='inner')
    m = m[m['norm_mod'] == m['dbptm_norm']]
    if m.empty:
        return sites

    m['dist'] = (m['dbptm_position'] - m['position_in_protein'].astype(float)).abs()

    exact = m.groupby('site_row')['dist'].min()
    sites.loc[exact.index[exact == 0], 'in_dbPTM'] = True

    # Same semantics as the iPTMnet annotation: exact matches (dist == 0)
    # also count as 'perhapse' (they are within the window by definition).
    near = m[m['dist'] <= window]
    if not near.empty:
        idx_min = near.groupby('site_row')['dist'].idxmin()
        closest = near.loc[idx_min]
        rows = closest['site_row'].to_numpy()
        sites.loc[rows, 'perhapse_in_dbPTM'] = True
        sites.loc[rows, 'perhapse_position_dbptm'] = \
            closest['dbptm_position'].astype(float).to_numpy()
        sites.loc[rows, 'perhapse_ptm_type_dbptm'] = closest['dbptm_type'].to_numpy()
    return sites


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def annotate_sites_with_dbptm(psm_df: pd.DataFrame,
                              fasta_file: str,
                              window: int = 7,
                              max_workers: int = 10,
                              modification_col: str = 'Modification',
                              reference_label: str = 'reference',
                              output_dir: str = None) -> pd.DataFrame:
    """Annotate unique modified sites with dbPTM experimental PTM sites.

    Returns one row per unique (id_prot, position_in_protein,
    modified_peptide_x) site of a NON-reference modification.
    """
    keys = ['id_prot', 'position_in_protein', 'modified_peptide_x']
    missing = [c for c in keys + [modification_col] if c not in psm_df.columns]
    if missing:
        raise KeyError(f"Required columns missing for dbPTM annotation: {missing}")

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

    protein_ids = sites['id_prot'].unique()
    logger.info(f"dbPTM annotation: {len(sites)} unique sites, "
                f"{len(protein_ids)} proteins, window=+/-{window}")

    dbptm = fetch_dbptm_long(protein_ids, fasta_file, max_workers=max_workers,
                             output_dir=output_dir)
    sites = _match_dbptm(sites, dbptm, window=window)

    logger.info(f"dbPTM annotation done: in_dbPTM={int(sites['in_dbPTM'].sum())}, "
                f"perhapse={int(sites['perhapse_in_dbPTM'].sum())}, "
                f"with any dbPTM record at position={int(sites['dbptm_ptm_types'].notna().sum())}")

    return sites[['id_prot', 'position_in_protein', 'modified_peptide_x', 'mods',
                  'in_dbPTM', 'dbptm_ptm_types', 'dbptm_pmids',
                  'perhapse_in_dbPTM', 'perhapse_position_dbptm',
                  'perhapse_ptm_type_dbptm']]
