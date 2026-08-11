"""
signor_annotation.py — SIGNOR causal-network annotation of PTM sites.

SIGNOR (https://signor.uniroma2.it/) curates causal relationships between
proteins with the underlying mechanism. For PTM studies the valuable part is
the "PTM effect on a protein": edges of the form

    regulator (enzyme) --[phosphorylation @ Ser15]--> target protein
    EFFECT = up-regulates / down-regulates ...

i.e. which modification of which residue activates or inhibits the protein,
which enzyme is responsible, and on which evidence (PMID, direct/indirect,
curator score).

Programmatic access (documented at https://signor.uniroma2.it/APIs.php):

    getData.php?organism=<tax_id>&id=<UniProtKB AC>

returns a headerless TSV (29 columns, see SIGNOR_COLUMNS) with all curated
interactions of one protein. Querying per protein avoids downloading the
full multi-organism dump (~100 MB); results are cached by the caller
(main.py) in <output_dir>/signor_edges.csv / signor_sites.csv.

Runs AFTER the statistical analysis (results-level annotation, does not
affect testing), gated by config option signor_annotation=True.

Outputs
-------
1. Per-site table (one row per unique site, mirrors the iPTMnet/dbPTM
   annotation pattern, merge keys: id_prot, position_in_protein,
   modified_peptide_x):
     in_SIGNOR                 site-exact causal PTM record exists
     signor_evidence           'site' (residue-exact) / 'protein' (mechanism
                               matches the PTM class but no residue is
                               annotated) / None
     signor_effect_on_protein  'activate' / 'inhibit' / 'conflicting' /
                               'unknown' — what the PTM at this site does to
                               the protein according to SIGNOR
     signor_regulations        'PLK1 phosphorylation->down-regulates; ...'
     signor_regulators         upstream enzymes, ';'-joined
     signor_pmids              supporting PMIDs, ';'-joined

2. signor_network.html — interactive, interpretation-oriented network:
   measured proteins colored by their experimental regulation, upstream
   SIGNOR regulators as satellite nodes, edges colored by effect
   (green = activation, red = inhibition, grey = unknown), mechanism and
   residue on the edge label, HTML legend, edge filters and a protein-focus
   selector. Works fully offline (vis-network is inlined).

Python 3.10 compatible.
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

try:  # package import
    from .ptm_annotation import normalize_mod
    from .dbconnect import apply_idmap, resolve_unmapped_ids
except ImportError:  # standalone usage
    from ptm_annotation import normalize_mod
    from dbconnect import apply_idmap, resolve_unmapped_ids

logger = logging.getLogger(__name__)

SIGNOR_API_URL = "https://signor.uniroma2.it/getData.php"

# Headerless TSV column order, as documented on https://signor.uniroma2.it/APIs.php
SIGNOR_COLUMNS = [
    'ENTITYA', 'TYPEA', 'IDA', 'DATABASEA',
    'ENTITYB', 'TYPEB', 'IDB', 'DATABASEB',
    'EFFECT', 'MECHANISM', 'RESIDUE', 'SEQUENCE', 'TAX_ID',
    'CELL_DATA', 'TISSUE_DATA', 'MODULATOR_COMPLEX', 'TARGET_COMPLEX',
    'MODIFICATIONA', 'MODASEQ', 'MODIFICATIONB', 'MODBSEQ',
    'PMID', 'DIRECT', 'NOTES', 'ANNOTATOR', 'SENTENCE', 'SIGNOR_ID', 'SCORE',
]

# Entity types accepted as interaction endpoints. Chemicals / phenotypes /
# stimuli never carry a PTM mechanism relevant here and only add noise.
_ALLOWED_TYPES = {'protein', 'protein family', 'fusion protein', 'complex'}

# ---------------------------------------------------------------------------
# Mechanism / effect normalization
# ---------------------------------------------------------------------------
# SIGNOR mechanisms are lowercase free-ish strings ('phosphorylation',
# 'polyubiquitination', 'dephosphorylation', 'binding', ...). PTM mechanisms
# are mapped onto the same broad classes that ptm_annotation.normalize_mod
# produces for the experimental mod labels, so typed matching compares like
# with like. Non-PTM mechanisms (binding, transcriptional regulation, ...)
# map to None but are kept in the edge table for the network context layer.

SIGNOR_MECHANISM_MAP = {
    'phosphorylation': 'Phosphorylation',
    'dephosphorylation': 'Phosphorylation',
    'acetylation': 'Acetylation',
    'deacetylation': 'Acetylation',
    'methylation': 'Methylation',
    'trimethylation': 'Methylation',
    'demethylation': 'Methylation',
    'ubiquitination': 'Ubiquitination',
    'polyubiquitination': 'Ubiquitination',
    'deubiquitination': 'Ubiquitination',
    'sumoylation': 'SUMOylation',
    'desumoylation': 'SUMOylation',
    'neddylation': 'Neddylation',
    'deneddylation': 'Neddylation',
    'glycosylation': 'Glycosylation',
    'o-linked glycosylation': 'O-Glycosylation',
    'n-linked glycosylation': 'N-Glycosylation',
    'deglycosylation': 'Glycosylation',
    'palmitoylation': 'Palmitoylation',
    'depalmitoylation': 'Palmitoylation',
    'myristoylation': 'Myristoylation',
    'farnesylation': 'Farnesylation',
    's-nitrosylation': 'S-Nitrosylation',
    'nitrosylation': 'S-Nitrosylation',
    'oxidation': 'Oxidation',
    'hydroxylation': 'Hydroxylation',
    'adp-ribosylation': 'ADP-ribosylation',
    'carboxylation': 'Carboxylation',
}

# Mechanisms that REMOVE a modification (effect then refers to the
# de-modified state — important for interpretation).
SIGNOR_DEMOD_MECHANISMS = {
    'dephosphorylation', 'deacetylation', 'deubiquitination', 'demethylation',
    'desumoylation', 'deneddylation', 'deglycosylation', 'depalmitoylation',
}

# A SIGNOR 'glycosylation' record does not specify the linkage; let it match
# any glycosylation site class from normalize_mod.
_CLASS_EQUIV = {
    'Glycosylation': {'Glycosylation', 'O-Glycosylation', 'N-Glycosylation',
                      'C-Glycosylation'},
}


def classes_compatible(signor_class, site_class):
    """True if a SIGNOR PTM class may annotate a site of the given class."""
    if not signor_class or not site_class:
        return False
    if signor_class == site_class:
        return True
    return site_class in _CLASS_EQUIV.get(signor_class, set())


def normalize_effect(effect_raw):
    """SIGNOR EFFECT -> 'activate' / 'inhibit' / 'unknown'.

    EFFECT values are fine-grained ('up-regulates activity',
    'down-regulates quantity by destabilization', ...); the direction is
    carried by the prefix."""
    s = str(effect_raw).strip().lower()
    if s.startswith('up-regulates'):
        return 'activate'
    if s.startswith('down-regulates'):
        return 'inhibit'
    return 'unknown'


# Short edge-label abbreviations for the network (mechanism + residue).
_MECH_ABBR = {
    'phosphorylation': 'phos', 'dephosphorylation': 'dephos',
    'acetylation': 'ac', 'deacetylation': 'deac',
    'methylation': 'me', 'demethylation': 'deme',
    'ubiquitination': 'ub', 'polyubiquitination': 'polyUb',
    'deubiquitination': 'deub',
    'sumoylation': 'SUMO', 'desumoylation': 'deSUMO',
    'neddylation': 'NEDD8', 'deneddylation': 'deNEDD8',
    'glycosylation': 'glyc', 'deglycosylation': 'deglyc',
    'palmitoylation': 'palm', 'depalmitoylation': 'depalm',
    's-nitrosylation': 'SNO', 'nitrosylation': 'SNO',
    'oxidation': 'ox', 'hydroxylation': 'OH',
    'adp-ribosylation': 'ADPr', 'myristoylation': 'myr',
    'farnesylation': 'farn', 'carboxylation': 'carb',
}

_RESIDUE_RE = re.compile(r'([A-Za-z]{3})[^0-9]{0,3}(\d+)')

# UniProt isoform suffix ('P04637-2' -> 'P04637'). Applied ONLY to
# UniProt-shaped accessions so complex ids like 'SIGNOR-C144' survive.
_ISOFORM_RE = re.compile(r'^([A-Z0-9]{6}(?:[A-Z0-9]{4})?)-\d+$')


def _base_accession(pid) -> str:
    m = _ISOFORM_RE.match(str(pid))
    return m.group(1) if m else str(pid)


def parse_residue(residue_str):
    """'Ser315' -> ('Ser', 315). Returns (None, None) if unparseable."""
    if not isinstance(residue_str, str) or not residue_str.strip():
        return None, None
    m = _RESIDUE_RE.search(residue_str)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


# ---------------------------------------------------------------------------
# Per-protein fetching (parallel)
# ---------------------------------------------------------------------------

def fetch_signor_protein(uniprot_id: str, session: requests.Session,
                         organism: int = 9606, retries: int = 2,
                         timeout: int = 60):
    """Fetch all curated interactions of one protein from SIGNOR.
    Returns a list of raw TSV rows (lists of 29 strings) or None."""
    params = {'organism': organism, 'id': uniprot_id}
    for attempt in range(retries + 1):
        try:
            r = session.get(SIGNOR_API_URL, params=params, timeout=timeout)
            r.raise_for_status()
            rows = [line.split('\t')
                    for line in r.text.splitlines() if line.strip()]
            # Data rows carry a trailing empty field (trailing tab), i.e.
            # len(SIGNOR_COLUMNS)+1 fields; accept and trim to the header.
            rows = [row[:len(SIGNOR_COLUMNS)] for row in rows
                    if len(row) >= len(SIGNOR_COLUMNS)]
            # The server occasionally drops the connection mid-body: the
            # response is then truncated and no complete rows survive —
            # treat that as a retryable failure, not as "no interactions".
            if not rows and len(r.text) > 2000 and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return rows if rows else None
        except requests.exceptions.RequestException as e:
            if attempt >= retries:
                logger.warning(f"SIGNOR fetch failed for {uniprot_id}: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _fetch_signor_parallel(query_ids, session, organism, max_workers):
    """Query SIGNOR for each id in parallel.

    Returns ({query_id: raw TSV rows}, [query ids without data]).
    """
    fetched = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_signor_protein, pid, session,
                                   organism=organism): pid
                   for pid in query_ids}
        done = 0
        for future in as_completed(futures):
            pid = futures[future]
            try:
                rows = future.result()
            except Exception as e:
                logger.warning(f"SIGNOR fetch failed for {pid}: {e}")
                rows = None
            if rows:
                fetched[pid] = rows
            done += 1
            if done % 50 == 0:
                logger.info(f"SIGNOR: fetched {done}/{len(query_ids)} proteins")
    missing = [p for p in query_ids if p not in fetched]
    return fetched, missing


def fetch_signor_edges(protein_ids, organism: int = 9606,
                       max_workers: int = 6, output_dir: str = None) -> pd.DataFrame:
    """Fetch and clean SIGNOR interactions for all proteins in parallel.

    Obsolete/secondary UniProt accessions (no SIGNOR record under the old
    id) go through the SHARED UniProt ID-mapping cache
    (<output_dir>/uniprot_idmap.csv, see dbconnect.apply_idmap /
    resolve_unmapped_ids): proteins already resolved by any previous
    database annotation are queried with the current accession right away;
    the remaining misses are resolved through the API once and retried. In
    the returned edge table the mapped protein is relabelled with the
    ORIGINAL accession (ida/idb and their *_base variants) so that matching
    against the experimental table and the network view stay consistent.

    Returns the edge table with normalized columns:
      entitya, typea, ida, entityb, typeb, idb,
      effect_raw, effect ('activate'/'inhibit'/'unknown'),
      mechanism, ptm_class (shared vocabulary or None),
      mod_action ('add'/'remove'/None), residue ('Ser315'), position (Int64),
      sequence, tax_id, pmid, direct (bool), score (float),
      signor_id, sentence
    """
    cols = ['entitya', 'typea', 'ida', 'entityb', 'typeb', 'idb',
            'effect_raw', 'effect', 'mechanism', 'ptm_class', 'mod_action',
            'residue', 'position', 'sequence', 'tax_id', 'pmid', 'direct',
            'score', 'signor_id', 'sentence']
    ids = [str(p) for p in pd.unique(pd.Series(list(protein_ids))) if pd.notna(p)]
    if not ids:
        return pd.DataFrame(columns=cols)

    # Isoform accessions ('P04637-2') are queried via the canonical accession.
    base_ids = sorted({_base_accession(p) for p in ids})

    # Shared UniProt ID-mapping cache: {original_base: query accession}.
    query_plan = apply_idmap(base_ids, output_dir)
    relabel = {q: b for b, q in query_plan.items() if q != b}
    if relabel:
        logger.info(f"UniProt ID-mapping cache: querying SIGNOR with current "
                    f"accessions for {len(relabel)} proteins: "
                    f"{ {b: q for q, b in relabel.items()} }")

    all_rows = []
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'OpenPtmFinder SIGNOR annotation (academic research use)'})
    try:
        query_ids = sorted(set(query_plan.values()))
        fetched, missing = _fetch_signor_parallel(query_ids, session,
                                                  organism, max_workers)
        for rows in fetched.values():
            all_rows.extend(rows)

        # Only proteins queried with their OWN accession are candidates for
        # remapping (already-remapped misses would fail again identically).
        missing = set(missing)
        retry_src = [b for b, q in query_plan.items() if q in missing and q == b]
        if retry_src:
            mapping = resolve_unmapped_ids(sorted(retry_src), output_dir)
            if mapping:
                logger.info(f"SIGNOR: UniProt ID mapping resolved {len(mapping)} "
                            f"of {len(retry_src)} proteins without SIGNOR data: "
                            f"{mapping}")
                refetched, still = _fetch_signor_parallel(
                    sorted(set(mapping.values())), session, organism, max_workers)
                for q, rows in refetched.items():
                    all_rows.extend(rows)
                for old, new in mapping.items():
                    if new in refetched:
                        relabel[new] = old  # relabel edges to the original id
                if still:
                    logger.info(f"SIGNOR: no data even after ID remapping for "
                                f"{[o for o, n in mapping.items() if n in still]}")
            else:
                logger.info(f"SIGNOR: UniProt ID mapping found no current "
                            f"accession for {len(retry_src)} proteins without "
                            f"SIGNOR data (left unannotated).")
    finally:
        session.close()

    if not all_rows:
        logger.warning("No data fetched from SIGNOR.")
        return pd.DataFrame(columns=cols)

    raw = pd.DataFrame(all_rows, columns=SIGNOR_COLUMNS)

    # The same edge appears in the ego-network of every queried endpoint —
    # deduplicate on the curated relation id.
    raw['SIGNOR_ID'] = raw['SIGNOR_ID'].replace('', np.nan)
    raw = (raw.drop_duplicates(subset=['SIGNOR_ID'], keep='first')
              if raw['SIGNOR_ID'].notna().any()
              else raw.drop_duplicates(keep='first'))

    edges = pd.DataFrame({
        'entitya': raw['ENTITYA'],
        'typea': raw['TYPEA'],
        'ida': raw['IDA'],
        'entityb': raw['ENTITYB'],
        'typeb': raw['TYPEB'],
        'idb': raw['IDB'],
        'effect_raw': raw['EFFECT'],
        'mechanism': raw['MECHANISM'].replace('', np.nan),
        'residue': raw['RESIDUE'].replace('', np.nan),
        'sequence': raw['SEQUENCE'].replace('', np.nan),
        'tax_id': raw['TAX_ID'],
        'pmid': raw['PMID'].replace('', np.nan),
        'direct': raw['DIRECT'].str.strip().str.lower().eq('t'),
        'score': pd.to_numeric(raw['SCORE'], errors='coerce'),
        'signor_id': raw['SIGNOR_ID'],
        'sentence': raw['SENTENCE'],
    })
    edges['effect'] = edges['effect_raw'].map(normalize_effect)
    mech_norm = edges['mechanism'].str.strip().str.lower()
    edges['ptm_class'] = mech_norm.map(SIGNOR_MECHANISM_MAP)
    edges['mod_action'] = np.where(
        edges['ptm_class'].isna(), None,
        np.where(mech_norm.isin(SIGNOR_DEMOD_MECHANISMS), 'remove', 'add'))
    parsed = edges['residue'].map(parse_residue)
    edges['position'] = pd.array([p[1] for p in parsed], dtype='Int64')

    # Endpoint-type filter: proteins/families/complexes only.
    keep = edges['typea'].isin(_ALLOWED_TYPES) & edges['typeb'].isin(_ALLOWED_TYPES)
    edges = edges[keep].reset_index(drop=True)

    # Base (canonical) accessions for matching against isoform-bearing ids.
    edges['ida_base'] = edges['ida'].map(_base_accession)
    edges['idb_base'] = edges['idb'].map(_base_accession)

    # UniProt-remapped queries: relabel the mapped protein back to the
    # ORIGINAL accession everywhere it appears as an edge endpoint.
    if relabel:
        for col in ('ida', 'idb', 'ida_base', 'idb_base'):
            edges[col] = edges[col].replace(relabel)

    logger.info(f"SIGNOR: {len(edges)} unique interactions for "
                f"{len(query_ids)} queried proteins (organism={organism}); "
                f"PTM-mechanism edges: {int(edges['ptm_class'].notna().sum())}")
    return edges[cols + ['ida_base', 'idb_base']]


# ---------------------------------------------------------------------------
# Site-level annotation: what does the PTM at this site do to the protein
# ---------------------------------------------------------------------------

def _aggregate_effect(effects: pd.Series) -> str:
    s = set(effects.dropna())
    if 'activate' in s and 'inhibit' in s:
        return 'conflicting'
    if 'activate' in s:
        return 'activate'
    if 'inhibit' in s:
        return 'inhibit'
    return 'unknown'


def _format_regulations(df: pd.DataFrame) -> pd.Series:
    """'PLK1 phosphorylation->down-regulates' per edge row."""
    direction = df['effect'].map({'activate': 'up-regulates',
                                  'inhibit': 'down-regulates'}).fillna('?')
    return (df['entitya'].astype(str) + ' ' +
            df['mechanism'].fillna('?').astype(str) + '->' + direction)


def _match_signor(sites: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Add SIGNOR annotation columns to the unique-sites table."""
    sites = sites.copy()
    for col, default in [('in_SIGNOR', False), ('signor_evidence', None),
                         ('signor_effect_on_protein', None),
                         ('signor_regulations', None), ('signor_regulators', None),
                         ('signor_pmids', None)]:
        sites[col] = default

    if edges.empty:
        return sites

    # Only causal PTM edges INTO the measured proteins can annotate a site.
    ptm_in = edges[edges['ptm_class'].notna()].copy()
    if ptm_in.empty:
        return sites

    left = sites[['id_prot_base', 'position_in_protein', 'norm_mod']].copy()
    left['site_row'] = sites.index
    m = left.merge(
        ptm_in[['idb_base', 'position', 'ptm_class', 'entitya', 'mechanism',
                'effect', 'pmid']],
        left_on='id_prot_base', right_on='idb_base', how='inner')
    # Typed matching: SIGNOR mechanism class vs site PTM class.
    compat = [classes_compatible(sc, mc)
              for sc, mc in zip(m['ptm_class'], m['norm_mod'])]
    m = m[pd.Series(compat, index=m.index)]
    if m.empty:
        return sites

    m = m.copy()
    m['regulation'] = _format_regulations(m)
    site_exact = m['position'].notna() & (
        m['position'].astype('Float64') == m['position_in_protein'].astype('Float64'))

    def _aggregate(sub: pd.DataFrame) -> pd.Series:
        return pd.Series({
            'effect': _aggregate_effect(sub['effect']),
            'regulations': ';'.join(sorted(set(sub['regulation']))),
            'regulators': ';'.join(sorted(set(sub['entitya'].astype(str)))),
            'pmids': ';'.join(sorted({str(p) for p in sub['pmid'].dropna()} - {''})) or None,
        })

    # 1) residue-exact matches -> signor_evidence='site'
    exact = m[site_exact]
    if not exact.empty:
        # groupby(external key) instead of groupby(col).apply(...,
        # include_groups=False): the include_groups kwarg only exists in
        # pandas >= 2.2, and this form warns on none of the versions.
        agg = (exact.drop(columns='site_row')
               .groupby(exact['site_row']).apply(_aggregate))
        rows = agg.index.to_numpy()
        sites.loc[rows, 'in_SIGNOR'] = True
        sites.loc[rows, 'signor_evidence'] = 'site'
        sites.loc[rows, 'signor_effect_on_protein'] = agg['effect'].to_numpy()
        sites.loc[rows, 'signor_regulations'] = agg['regulations'].to_numpy()
        sites.loc[rows, 'signor_regulators'] = agg['regulators'].to_numpy()
        sites.loc[rows, 'signor_pmids'] = agg['pmids'].to_numpy()

    # 2) protein-level matches (right PTM class, no/unknown residue) for the
    #    remaining sites -> signor_evidence='protein'
    rest = m[~m['site_row'].isin(sites.index[sites['in_SIGNOR']])]
    rest = rest[rest['position'].isna()]
    if not rest.empty:
        agg = (rest.drop(columns='site_row')
               .groupby(rest['site_row']).apply(_aggregate))
        rows = agg.index.to_numpy()
        sites.loc[rows, 'signor_evidence'] = 'protein'
        sites.loc[rows, 'signor_effect_on_protein'] = agg['effect'].to_numpy()
        sites.loc[rows, 'signor_regulations'] = agg['regulations'].to_numpy()
        sites.loc[rows, 'signor_regulators'] = agg['regulators'].to_numpy()
        sites.loc[rows, 'signor_pmids'] = agg['pmids'].to_numpy()

    return sites


def annotate_sites_with_signor(psm_df: pd.DataFrame,
                               organism: int = 9606,
                               max_workers: int = 6,
                               modification_col: str = 'Modification',
                               reference_label: str = 'reference',
                               output_dir: str = None):
    """Annotate unique modified sites with SIGNOR causal PTM interactions.

    Returns (sites_df, edges_df):
      sites_df — one row per unique (id_prot, position_in_protein,
                 modified_peptide_x) site with SIGNOR annotation columns;
      edges_df — the full cleaned SIGNOR edge table (for the network and
                 for caching in <output_dir>/signor_edges.csv).
    """
    keys = ['id_prot', 'position_in_protein', 'modified_peptide_x']
    missing = [c for c in keys + [modification_col] if c not in psm_df.columns]
    if missing:
        raise KeyError(f"Required columns missing for SIGNOR annotation: {missing}")

    mod_mask = psm_df[modification_col] != reference_label
    sites = (psm_df.loc[mod_mask, keys + [modification_col]]
             .dropna(subset=['position_in_protein'])
             .drop_duplicates(subset=keys)
             .reset_index(drop=True))
    if sites.empty:
        logger.warning("No modified sites to annotate with SIGNOR.")
        return pd.DataFrame(), pd.DataFrame()

    sites['mods'] = sites[modification_col].astype(str).str.split('@').str[0]
    sites['norm_mod'] = sites['mods'].map(normalize_mod)
    sites['id_prot_base'] = sites['id_prot'].map(_base_accession)

    protein_ids = sites['id_prot'].unique()
    logger.info(f"SIGNOR annotation: {len(sites)} unique sites, "
                f"{len(protein_ids)} proteins, organism={organism}")

    edges = fetch_signor_edges(protein_ids, organism=organism,
                               max_workers=max_workers, output_dir=output_dir)
    if edges.empty:
        # Total fetch failure — return empty so the caller does NOT cache
        # a fake all-negative annotation (next run will retry the fetch).
        logger.warning("SIGNOR returned no interactions; annotation skipped.")
        return pd.DataFrame(), edges

    sites = _match_signor(sites, edges)

    logger.info(f"SIGNOR annotation done: site-level={int(sites['in_SIGNOR'].sum())}, "
                f"protein-level={int((sites['signor_evidence'] == 'protein').sum())} "
                f"of {len(sites)} sites")

    out_cols = ['id_prot', 'position_in_protein', 'modified_peptide_x', 'mods',
                'in_SIGNOR', 'signor_evidence', 'signor_effect_on_protein',
                'signor_regulations', 'signor_regulators', 'signor_pmids']
    return sites[out_cols], edges


# ---------------------------------------------------------------------------
# Network visualization
# ---------------------------------------------------------------------------

_EFFECT_COLORS = {'activate': '#2fa36b', 'inhibit': '#d9534f', 'unknown': '#97a3ae'}
_CONTEXT_COLOR = 'rgba(151,163,174,0.45)'
_SEED_COLORS = {'up': '#e74c3c', 'down': '#3498db', 'ns': '#b8c2cc'}
_REGULATOR_COLOR = '#8d99a6'
_BORDER_COLOR = '#1b2a41'


def _protein_regulation(output_dir: str, alpha: float, logfc_thr: float) -> dict:
    """Best (min adj.P.Val) contrast per protein from the final statistical
    results: {protein_base: {'logfc', 'contrast', 'pval', 'sig'}}.
    Empty dict if results are unavailable."""
    if not output_dir:
        return {}
    try:
        try:
            from .report import load_stat_results
        except ImportError:
            from report import load_stat_results
        stats = load_stat_results(output_dir)
    except Exception as e:
        logger.warning(f"Could not load stat results for SIGNOR network: {e}")
        return {}
    if stats is None or stats.empty or 'protein' not in stats.columns:
        return {}
    sub = stats.dropna(subset=['adj.P.Val'])
    if sub.empty:
        return {}
    best = sub.loc[sub.groupby('protein')['adj.P.Val'].idxmin()]
    reg = {}
    for _, row in best.iterrows():
        logfc = row.get('logFC')
        pval = row.get('adj.P.Val')
        reg[_base_accession(row['protein'])] = {
            'logfc': float(logfc) if pd.notna(logfc) else None,
            'contrast': str(row.get('contrast', '')),
            'pval': float(pval),
            'sig': bool(pd.notna(logfc) and pval < alpha and abs(logfc) >= logfc_thr),
        }
    return reg


def _edge_label(mechanism, residue):
    abbr = _MECH_ABBR.get(str(mechanism).strip().lower(),
                          str(mechanism)[:8] if pd.notna(mechanism) else '')
    if pd.notna(residue) and str(residue).strip():
        return f"{abbr} {residue}"
    return abbr or None


def _edge_title(row: pd.Series) -> str:
    direct = 'direct' if row['direct'] else 'indirect'
    mech = row['mechanism'] if pd.notna(row['mechanism']) else '—'
    res = f" ({row['residue']})" if pd.notna(row['residue']) else ''
    pmid = f"PMID {row['pmid']}" if pd.notna(row['pmid']) else 'PMID —'
    score = f"{row['score']:.2f}" if pd.notna(row['score']) else '—'
    return (f"<b>{row['entitya']} → {row['entityb']}</b><br>"
            f"Mechanism: {mech}{res}<br>"
            f"Effect: {row['effect_raw']}<br>"
            f"{direct} edge · score {score} · {pmid}")


def build_signor_network(edges_df: pd.DataFrame, sites_df: pd.DataFrame,
                         output_html: str, output_dir: str = None,
                         alpha: float = 0.05, logfc_thr: float = 1.0,
                         max_neighbors: int = 25,
                         title: str = 'SIGNOR: PTM effects on proteins'):
    """Build the interactive SIGNOR PTM-effect network (fully offline HTML).

    Focused, interpretation-oriented graph (replaces the old full-dump
    "hairball"):
      - nodes: measured modified proteins (colored by their experimental
        regulation, sized by the number of PTM sites) + their upstream
        SIGNOR regulators (small grey diamonds);
      - edges: causal PTM interactions INTO the measured proteins, colored
        by effect (green = activation, red = inhibition, grey = unknown),
        solid = direct, dashed = indirect, mechanism+residue as the label;
      - context: non-PTM causal edges between measured proteins (thin grey,
        switchable off);
      - HTML legend, effect/direct/context edge filters and a protein-focus
        selector are injected as an overlay (no fake legend nodes).

    Returns the output path, or None if there is nothing to draw.
    """
    if edges_df is None or edges_df.empty or sites_df is None or sites_df.empty:
        logger.warning("SIGNOR network: no edges or sites to draw.")
        return None

    import networkx as nx
    from pyvis.network import Network

    def _to_bool(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin({'true', 't', '1'})

    edges_df = edges_df.copy()
    edges_df['direct'] = _to_bool(edges_df['direct'])
    edges_df['score'] = pd.to_numeric(edges_df['score'], errors='coerce')

    sites = sites_df.copy()
    sites['in_SIGNOR'] = _to_bool(sites['in_SIGNOR'])
    sites['id_prot_base'] = sites['id_prot'].map(_base_accession)
    seeds = set(sites['id_prot_base'])
    n_sites = sites.groupby('id_prot_base').size().to_dict()
    n_signor_sites = (sites[sites['in_SIGNOR']].groupby('id_prot_base')
                      .size().to_dict())
    reg = _protein_regulation(output_dir, alpha, logfc_thr)

    # --- edge selection -----------------------------------------------------
    ptm_edges = edges_df[edges_df['ptm_class'].notna()]
    core = ptm_edges[ptm_edges['idb_base'].isin(seeds)].copy()   # -> measured
    context = edges_df[edges_df['ptm_class'].isna()
                       & edges_df['ida_base'].isin(seeds)
                       & edges_df['idb_base'].isin(seeds)].copy()

    # Cap over-connected seeds: keep the strongest regulators per target.
    dropped = 0
    keep_idx = []
    for _, grp in core.groupby('idb'):
        if len(grp) <= max_neighbors:
            keep_idx.extend(grp.index)
            continue
        ranked = grp.sort_values(['direct', 'score'], ascending=[False, False])
        keep_idx.extend(ranked.index[:max_neighbors])
        dropped += len(grp) - max_neighbors
    if dropped:
        logger.info(f"SIGNOR network: {dropped} weakest regulator edges hidden "
                    f"(max_neighbors={max_neighbors} per protein).")
    core = core.loc[sorted(keep_idx)]

    if core.empty and context.empty:
        logger.warning("SIGNOR network: no causal edges between SIGNOR and "
                       "the measured proteins.")
        return None

    core['context'] = False
    context['context'] = True
    net_edges = pd.concat([core, context], ignore_index=True)

    # --- nodes ----------------------------------------------------------------
    node_ids = set(net_edges['ida_base']) | set(net_edges['idb_base'])
    graph_seeds = sorted(node_ids & seeds)
    regulators = sorted(node_ids - seeds)

    label_map = {}
    for _, row in net_edges.iterrows():
        label_map.setdefault(row['ida_base'], str(row['entitya']))
        label_map.setdefault(row['idb_base'], str(row['entityb']))

    # SIGNOR site-effect summaries for seed tooltips.
    tooltip_lines = {}
    ann = sites[sites['signor_evidence'].notna()]
    for prot, grp in ann.groupby('id_prot_base'):
        lines = []
        for _, srow in grp.head(8).iterrows():
            pos = srow['position_in_protein']
            pos_s = str(int(pos)) if pd.notna(pos) else '?'
            eff = {'activate': 'activation', 'inhibit': 'inhibition',
                   'conflicting': 'conflicting', 'unknown': 'unknown'}.get(
                       srow['signor_effect_on_protein'], 'unknown')
            lines.append(f"• {srow['mods']}{pos_s}: {eff} "
                         f"({srow['signor_regulators']})")
        tooltip_lines[prot] = lines

    # --- layout (deterministic, physics off) ----------------------------------
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(zip(net_edges['ida_base'], net_edges['idb_base']))
    n = len(node_ids)
    pos = nx.spring_layout(G, seed=42, k=2.2 / np.sqrt(max(n, 1)), scale=620)

    net = Network(height='820px', width='100%', bgcolor='#ffffff',
                  font_color='#1b2a41', directed=True, cdn_resources='in_line')
    net.toggle_physics(False)

    def _seed_title(prot):
        r = reg.get(prot)
        exp = 'no data'
        if r and r['logfc'] is not None:
            direction = 'up-regulated' if r['logfc'] > 0 else 'down-regulated'
            exp = (f"logFC={r['logfc']:+.2f} ({r['contrast']}, "
                   f"adj.P={r['pval']:.2g}) — {direction}"
                   + ('' if r['sig'] else ' (not significant)'))
        lines = tooltip_lines.get(prot, [])
        signor_part = ('<hr style="margin:4px 0"><b>PTM effects per SIGNOR:</b><br>'
                       + '<br>'.join(lines)) if lines else ''
        return (f"<div style='max-width:340px'><b>{label_map[prot]}</b> ({prot})<br>"
                f"Measured PTM sites: {n_sites.get(prot, 0)} "
                f"(with a SIGNOR record: {n_signor_sites.get(prot, 0)})<br>"
                f"Experiment: {exp}{signor_part}</div>")

    for prot in graph_seeds:
        r = reg.get(prot)
        if r and r['sig']:
            fill = _SEED_COLORS['up'] if r['logfc'] > 0 else _SEED_COLORS['down']
        else:
            fill = _SEED_COLORS['ns']
        size = min(30.0, 13.0 + 4.0 * np.sqrt(n_sites.get(prot, 1)))
        x, y = pos[prot]
        net.add_node(prot, label=label_map[prot], title=_seed_title(prot),
                     color={'background': fill, 'border': _BORDER_COLOR},
                     borderWidth=2.5, size=size, shape='dot',
                     x=float(x), y=float(y), physics=False)

    for reg_id in regulators:
        x, y = pos[reg_id]
        net.add_node(reg_id, label=label_map[reg_id],
                     title=(f"<b>{label_map[reg_id]}</b> ({reg_id})<br>"
                            "SIGNOR regulator (not measured in the experiment)"),
                     color={'background': _REGULATOR_COLOR, 'border': '#5d6b78'},
                     borderWidth=1.5, size=7, shape='diamond',
                     x=float(x), y=float(y), physics=False)

    # --- edges ------------------------------------------------------------------
    for _, row in net_edges.iterrows():
        u, v = row['ida_base'], row['idb_base']
        if row['context']:
            color, width, label = _CONTEXT_COLOR, 1.0, None
        else:
            color = _EFFECT_COLORS.get(row['effect'], _EFFECT_COLORS['unknown'])
            width = 1.6 + (2.4 * row['score'] if pd.notna(row['score']) else 0.0)
            label = _edge_label(row['mechanism'], row['residue'])
        kwargs = dict(color=color, width=width, dashes=not bool(row['direct']),
                      arrows='to', title=_edge_title(row),
                      effect=row['effect'], direct=bool(row['direct']),
                      context=bool(row['context']))
        if label:
            kwargs['label'] = label
            kwargs['font'] = {'size': 9, 'color': '#37424d',
                              'background': 'rgba(255,255,255,0.78)',
                              'strokeWidth': 0}
        net.add_edge(u, v, **kwargs)

    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
    # pyvis save_graph() -> write_html() opens the output in the LOCALE
    # default encoding; under an ASCII locale (LANG=C) it crashes on
    # non-ascii characters (e.g. the copyright sign in the bundled
    # vis-network license header). Generate the HTML and write UTF-8
    # explicitly instead.
    html_doc = net.generate_html()
    if not isinstance(html_doc, str) or not html_doc:
        html_doc = getattr(net, "html", "")
    with open(output_html, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    _inject_overlay(output_html, title=title, seeds=graph_seeds,
                    label_map=label_map,
                    stats=dict(seeds=len(graph_seeds), total_seeds=len(seeds),
                               regulators=len(regulators),
                               core=len(core), context=len(context),
                               dropped=dropped))

    logger.info(f"SIGNOR network saved to {output_html} "
                f"({len(graph_seeds)} measured proteins, {len(regulators)} "
                f"SIGNOR regulators, {len(core)} PTM edges, "
                f"{len(context)} context edges)")
    return output_html


# ---------------------------------------------------------------------------
# HTML overlay: legend, filters, protein focus (injected into the pyvis page)
# ---------------------------------------------------------------------------

_OVERLAY_CSS = """
<style>
.sg-panel{position:fixed;z-index:1000;background:rgba(255,255,255,.94);
  border:1px solid #d8dee6;border-radius:10px;padding:10px 14px;
  font:13px/1.45 "Segoe UI",Arial,sans-serif;color:#1b2a41;
  box-shadow:0 2px 10px rgba(27,42,65,.12);}
#sg_header{top:12px;left:50%;transform:translateX(-50%);text-align:center;}
#sg_header b{font-size:15px}
#sg_header span{color:#5d6b78;font-size:12px}
#sg_legend{top:12px;left:12px;max-width:295px}
#sg_legend h4,#sg_controls h4{margin:0 0 6px;font-size:13px}
#sg_legend .sw{display:inline-block;width:11px;height:11px;border-radius:50%;
  margin-right:6px;vertical-align:-1px;border:1px solid #1b2a41}
#sg_legend .dm{border-radius:2px}
#sg_legend .ln{display:inline-block;width:22px;height:0;border-top:3px solid;
  margin-right:6px;vertical-align:3px}
#sg_legend .dash{border-top-style:dashed}
#sg_controls{top:12px;right:12px;max-width:240px}
#sg_controls label{display:block;margin:3px 0;cursor:pointer;user-select:none}
#sg_controls select,#sg_controls button{width:100%;margin:4px 0;padding:4px 6px;
  border:1px solid #c3ccd6;border-radius:6px;background:#fff;font-size:13px}
#sg_controls button{cursor:pointer;background:#eef2f6}
#sg_controls button:hover{background:#dde6ee}
</style>
"""


def _inject_overlay(output_html: str, title: str, seeds, label_map: dict,
                    stats: dict):
    """Inject the legend / filter / focus overlay into the saved pyvis page."""
    with open(output_html, 'r', encoding='utf-8') as fh:
        html = fh.read()

    # The pyvis template pulls bootstrap from a CDN (purely cosmetic there);
    # strip it so the network page is fully self-contained / offline.
    html = re.sub(r'<link[^>]+bootstrap[^>]*>', '', html)
    html = re.sub(r'<script[^>]+bootstrap[^>]*></script>', '', html)

    options = ''.join(
        f'<option value="{p}">{label_map[p]} ({p})</option>' for p in seeds)
    hidden_note = (f" · weak edges hidden: {stats['dropped']}"
                   if stats['dropped'] else '')

    panel = f"""
<div id="sg_header" class="sg-panel">
  <b>{title}</b><br>
  <span>{stats['seeds']} of {stats['total_seeds']} measured proteins ·
  SIGNOR regulators: {stats['regulators']} ·
  PTM edges: {stats['core']} · context: {stats['context']}{hidden_note}</span>
</div>
<div id="sg_legend" class="sg-panel">
  <h4>Nodes</h4>
  <div><span class="sw" style="background:{_SEED_COLORS['up']}"></span>measured protein, up-regulated</div>
  <div><span class="sw" style="background:{_SEED_COLORS['down']}"></span>measured protein, down-regulated</div>
  <div><span class="sw" style="background:{_SEED_COLORS['ns']}"></span>measured, no significant change</div>
  <div><span class="sw dm" style="background:{_REGULATOR_COLOR}"></span>SIGNOR regulator (not measured)</div>
  <div style="color:#5d6b78;font-size:12px;margin-top:2px">node size ∝ number of measured PTM sites</div>
  <h4 style="margin-top:8px">Edges (PTM effects)</h4>
  <div><span class="ln" style="border-color:{_EFFECT_COLORS['activate']}"></span>protein activation</div>
  <div><span class="ln" style="border-color:{_EFFECT_COLORS['inhibit']}"></span>protein inhibition</div>
  <div><span class="ln" style="border-color:{_EFFECT_COLORS['unknown']}"></span>effect unknown</div>
  <div><span class="ln dash" style="border-color:#97a3ae"></span>indirect (solid = direct)</div>
  <div style="color:#5d6b78;font-size:12px;margin-top:4px">
  edge label — mechanism and site (phos Ser15);<br>
  hover an edge — effect, PMID, score;<br>
  nodes can be dragged.</div>
</div>
<div id="sg_controls" class="sg-panel">
  <h4>Edge filters</h4>
  <label><input type="checkbox" id="sg_act" checked> activation</label>
  <label><input type="checkbox" id="sg_inh" checked> inhibition</label>
  <label><input type="checkbox" id="sg_unk" checked> effect unknown</label>
  <label><input type="checkbox" id="sg_dir"> direct edges only</label>
  <label><input type="checkbox" id="sg_ctx" checked> context (non-PTM) edges</label>
  <h4 style="margin-top:8px">Focus on a protein</h4>
  <select id="sg_focus"><option value="">— select a protein —</option>{options}</select>
  <button id="sg_fit">show the whole graph</button>
</div>
"""

    script = """
<script>
(function(){
  function boot(){
    if (typeof network === 'undefined' || !network.body){ setTimeout(boot, 150); return; }
    var edgesDS = network.body.data.edges;
    var allEdges = edgesDS.get();
    function applyFilters(){
      var a = document.getElementById('sg_act').checked;
      var i = document.getElementById('sg_inh').checked;
      var u = document.getElementById('sg_unk').checked;
      var d = document.getElementById('sg_dir').checked;
      var c = document.getElementById('sg_ctx').checked;
      var visEdges = allEdges.filter(function(e){
        if (e.context && !c) return false;
        if (d && !e.direct) return false;
        if (e.effect === 'activate' && !a) return false;
        if (e.effect === 'inhibit' && !i) return false;
        if (e.effect !== 'activate' && e.effect !== 'inhibit' && !u) return false;
        return true;
      });
      edgesDS.clear();
      edgesDS.add(visEdges);
    }
    ['sg_act','sg_inh','sg_unk','sg_dir','sg_ctx'].forEach(function(id){
      document.getElementById(id).addEventListener('change', applyFilters);
    });
    document.getElementById('sg_focus').addEventListener('change', function(){
      if (!this.value) return;
      network.selectNodes([this.value]);
      network.focus(this.value, {scale: 1.05,
        animation: {duration: 600, easingFunction: 'easeInOutQuad'}});
    });
    document.getElementById('sg_fit').addEventListener('click', function(){
      network.fit({animation: {duration: 500}});
    });
    network.once('stabilized', function(){ network.fit(); });
    network.fit();
  }
  if (document.readyState === 'complete') boot();
  else window.addEventListener('load', boot);
})();
</script>
"""

    html = html.replace('</head>', _OVERLAY_CSS + '</head>')
    html = html.replace('</body>', panel + script + '</body>')
    with open(output_html, 'w', encoding='utf-8') as fh:
        fh.write(html)
