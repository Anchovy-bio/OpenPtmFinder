"""
dbconnect.py — low-level external-API access.

- get_protein_info_from_iptmnet / fetch_protein: per-protein iPTMnet
  substrate query used by ptm_annotation.py.
- map_uniprot_ids: UniProt ID-mapping REST API
  (https://www.uniprot.org/id-mapping) resolving obsolete/secondary
  accessions to current primary accessions, so that proteins whose ids
  changed since the experiment can still be annotated in iPTMnet.

The former full-dump helpers (dbPTM per-modification downloads and the
legacy SIGNOR graph code) were superseded by dbptm_annotation.py and
signor_annotation.py, which fetch per-protein pages instead of whole-database
dumps, and were removed.
"""

import logging
import os
import time
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

UNIPROT_IDMAPPING_API = "https://rest.uniprot.org/idmapping"


def get_protein_info_from_iptmnet(uniprot_id, session=None):
    """
    Fetch substrate PTM entries for one protein from the iPTMnet API.

    Args:
        uniprot_id (str): UniProt accession of the protein.
        session (requests.Session, optional): session for connection reuse.

    Returns:
        pd.DataFrame or None
    """
    base_url = "https://research.bioinformatics.udel.edu/iptmnet/api"
    endpoint = f"{uniprot_id}/substrate"
    url = f"{base_url}/{endpoint}"

    try:
        # Reuse the session if one was provided
        req = session.get(url, timeout=10) if session else requests.get(url, timeout=10)
        req.raise_for_status()

        data = req.text
        df = pd.read_csv(StringIO(data), sep=",")
        return df

    except requests.exceptions.RequestException as e:
        logger.warning(f"Request failed for {uniprot_id}: {e}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty response for {uniprot_id}")
        return None
    except Exception as e:
        logger.warning(f"Failed to process the response for {uniprot_id}: {e}")
        return None


def fetch_protein(prot, session):
    """Fetch iPTMnet substrate entries; isoform ids ('P04637-2') are resolved
    through the canonical accession."""
    prot_nat = prot.split('-')[0] if '-' in prot else prot
    df = get_protein_info_from_iptmnet(prot_nat, session=session)
    return df


# ---------------------------------------------------------------------------
# UniProt ID mapping (obsolete/secondary accessions -> current primary ones)
# ---------------------------------------------------------------------------

def _next_page_url(response):
    """Next-page URL of a paginated UniProt response (RFC 5988 Link header)."""
    link = response.headers.get("Link", "")
    for part in link.split(","):
        if 'rel="next"' in part:
            return part.split(";")[0].strip().strip("<>")
    return None


def map_uniprot_ids(uniprot_ids, session=None, poll_interval=2.0,
                    timeout=300.0):
    """Map UniProt accessions to current primary accessions.

    Uses the UniProt ID-mapping REST API (https://www.uniprot.org/id-mapping):
    one job is submitted for all ids (from=UniProtKB_AC-ID, to=UniProtKB),
    polled until finished, and the UniProtKB results are read (with
    pagination). This resolves secondary/obsolete accessions (merged or
    demerged entries) that services like iPTMnet do not know.

    Args:
        uniprot_ids (iterable of str): accessions to map.
        session (requests.Session, optional): session for connection reuse.
            NB: do NOT pass a session whose Accept header forces a non-JSON
            format — the results endpoint honours content negotiation.
        poll_interval (float): seconds between job-status polls.
        timeout (float): max seconds to wait for the mapping job.

    Returns:
        dict: {submitted_id: current_primary_accession}. Ids UniProt could
        not map (deleted entries) are absent. Demerged accessions mapping to
        several current entries keep the first hit (a warning is logged).
        Any network/API failure yields an empty dict so that annotation can
        proceed without remapping.
    """
    ids = sorted({str(i).strip() for i in uniprot_ids if str(i).strip()})
    if not ids:
        return {}
    http = session if session is not None else requests
    try:
        run = http.post(f"{UNIPROT_IDMAPPING_API}/run",
                        data={"from": "UniProtKB_AC-ID", "to": "UniProtKB",
                              "ids": ",".join(ids)},
                        timeout=30)
        run.raise_for_status()
        job_id = run.json()["jobId"]

        deadline = time.time() + timeout
        while True:
            # When the job finishes, the status endpoint redirects (303) to
            # the results payload; requests follows the redirect, so the
            # final JSON IS the results page (plus its Link header).
            status = http.get(f"{UNIPROT_IDMAPPING_API}/status/{job_id}",
                              timeout=30)
            status.raise_for_status()
            payload = status.json()
            if "jobStatus" in payload:  # NEW/RUNNING -> keep polling
                if time.time() > deadline:
                    raise TimeoutError(
                        f"UniProt ID-mapping job {job_id} timed out")
                time.sleep(poll_interval)
                continue
            break

        results = list(payload.get("results", []))
        next_url = _next_page_url(status)
        while next_url:  # follow pagination for large jobs
            page = http.get(next_url, timeout=30)
            page.raise_for_status()
            results.extend(page.json().get("results", []))
            next_url = _next_page_url(page)

        mapping, multi = {}, {}
        for item in results:
            to = item.get("to")
            new_id = (to.get("primaryAccession") if isinstance(to, dict)
                      else str(to)) if to else None
            if not new_id:
                continue
            old_id = item.get("from")
            if old_id in mapping and mapping[old_id] != new_id:
                multi.setdefault(old_id, {mapping[old_id]}).add(new_id)
            else:
                mapping[old_id] = new_id
        if multi:
            logger.warning(f"Demerged accessions with several current entries "
                           f"(first kept): {multi}")
        return mapping
    except Exception as e:
        logger.warning(f"UniProt ID mapping unavailable ({e}); "
                       f"proceeding without accession remapping.")
        return {}


# ---------------------------------------------------------------------------
# Shared accession-mapping cache (<output_dir>/uniprot_idmap.csv)
# ---------------------------------------------------------------------------
# All database annotations (iPTMnet, dbPTM, SIGNOR) share ONE mapping cache:
# the first annotation that meets an unknown (obsolete/secondary) accession
# resolves it through UniProt ID mapping and stores the pair; later
# annotations then query their databases with the CURRENT accession directly
# and relabel the results back to the original id. Ids that could not be
# resolved are cached as identity pairs (id -> id) so they are never
# re-submitted to the API.

IDMAP_CACHE_FILENAME = "uniprot_idmap.csv"


def load_idmap(output_dir):
    """Load the shared UniProt accession mapping
    (<output_dir>/uniprot_idmap.csv) as {original_id: current_id};
    {} when absent or unreadable."""
    if not output_dir:
        return {}
    path = os.path.join(output_dir, IDMAP_CACHE_FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        return dict(zip(df['original_id'], df['current_id']))
    except Exception as e:
        logger.warning(f"Could not read the UniProt ID-mapping cache {path}: {e}")
        return {}


def _save_idmap(output_dir, mapping: dict):
    """Persist the shared accession mapping (overwrites the cache file)."""
    path = os.path.join(output_dir, IDMAP_CACHE_FILENAME)
    df = pd.DataFrame({'original_id': list(mapping),
                       'current_id': list(mapping.values())})
    df.to_csv(path, index=False)


def apply_idmap(ids, output_dir=None):
    """{original_id: query_id} — replace accessions with their mapped
    current counterparts where the shared cache has one (identity
    otherwise). Use this BEFORE querying a database."""
    cached = load_idmap(output_dir)
    if not cached:
        return {str(i): str(i) for i in ids}
    return {str(i): cached.get(str(i), str(i)) for i in ids}


def resolve_unmapped_ids(missing_ids, output_dir=None):
    """Resolve database misses to current UniProt accessions.

    Cache-aware wrapper around map_uniprot_ids: ids already present in the
    shared cache are answered from disk (positives return the stored
    accession, negatives are skipped), only never-seen ids are submitted to
    the UniProt ID-mapping API — so the API is hit at most once per id
    across ALL database annotations and reruns. New results (positives AND
    negatives, the latter as identity pairs) are appended to the cache when
    output_dir is given.

    Returns {original_id: current_id} for resolvable ids only.
    """
    ids = list(dict.fromkeys(str(i) for i in missing_ids))  # dedupe, keep order
    if not ids:
        return {}
    cached = load_idmap(output_dir)
    out, todo = {}, []
    for i in ids:
        if i in cached:
            if cached[i] != i:      # cached positive
                out[i] = cached[i]
            # cached identity pair = known negative -> skip silently
        else:
            todo.append(i)
    if not todo:
        return out
    new_map = map_uniprot_ids(todo)
    resolved = {k: v for k, v in new_map.items() if v and v != k}
    out.update(resolved)
    if output_dir:
        for k in todo:  # negatives are cached as identity pairs
            cached[k] = resolved.get(k, k)
        try:
            _save_idmap(output_dir, cached)
            if resolved:
                logger.info(f"UniProt ID-mapping cache updated (+{len(resolved)} "
                            f"resolved): {os.path.join(output_dir, IDMAP_CACHE_FILENAME)}")
        except OSError as e:
            logger.warning(f"Could not write the UniProt ID-mapping cache: {e}")
    return out
