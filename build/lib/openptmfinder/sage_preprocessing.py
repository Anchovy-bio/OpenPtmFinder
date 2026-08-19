"""
Sage / enriched-PTM preprocessing for OpenPtmFinder.

This module converts Sage output (results.sage.tsv + tmt.tsv) into the same
annotated PSM table that the MSFragger/pepXML branch stores in
annotated_df.pickle. It is intended for enrichment designs (by default
phospho), where there are no unmodified reference PSMs and statistics must run
with type_experiment='phospho enrichment'.

Python 3.10 compatible.
"""

import ast
import glob
import logging
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from pyteomics import fasta as _pyteomics_fasta
except Exception:  # pragma: no cover - fallback for minimal environments
    _pyteomics_fasta = None

logger = logging.getLogger(__name__)


def _parse_channel_cell(x) -> List[str]:
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(z).strip() for z in v if str(z).strip()]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return [str(v).strip()]
        except Exception:
            pass
        s = s.strip("[](){}")
        return [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]
    return [str(x).strip()]


def _extract_protein_id(header: str) -> str:
    s = str(header).strip()
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return s.split()[0]


def _split_proteins(x) -> List[str]:
    if pd.isna(x):
        return []
    return [_extract_protein_id(p) for p in str(x).split(";") if str(p).strip()]


def load_fasta_dict(fasta_file: str) -> Dict[str, str]:
    """Load target protein sequences; supports UniProt and plain FASTA headers."""
    fasta_dict = {}
    if _pyteomics_fasta is not None:
        with _pyteomics_fasta.read(fasta_file) as db:
            for descr, seq in db:
                if re.search(r"DECOY_|rev_", str(descr), flags=re.IGNORECASE):
                    continue
                fasta_dict[_extract_protein_id(descr)] = seq
        return fasta_dict

    # Minimal FASTA fallback (no pyteomics): header lines start with '>'.
    descr = None
    chunks = []
    with open(fasta_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if descr is not None and not re.search(r"DECOY_|rev_", descr, flags=re.IGNORECASE):
                    fasta_dict[_extract_protein_id(descr)] = ''.join(chunks)
                descr = line[1:]
                chunks = []
            elif line:
                chunks.append(line)
    if descr is not None and not re.search(r"DECOY_|rev_", descr, flags=re.IGNORECASE):
        fasta_dict[_extract_protein_id(descr)] = ''.join(chunks)
    return fasta_dict


def target_decoy_filter(psm_df: pd.DataFrame,
                        fdr_threshold: float = 0.01,
                        score_column: str = "sage_discriminant_score",
                        decoy_regex: str = r"DECOY|rev_") -> pd.DataFrame:
    """Score-based target-decoy FDR for Sage discriminant scores."""
    df = psm_df.copy()
    if score_column not in df.columns:
        raise KeyError(f"Sage score column '{score_column}' not found.")
    df["is_decoy"] = df["proteins"].astype(str).str.contains(decoy_regex, case=False, regex=True)
    df = df.sort_values(score_column, ascending=False).reset_index(drop=True)
    is_decoy = df["is_decoy"].to_numpy(dtype=bool)
    df["decoy_cum"] = np.cumsum(is_decoy.astype(int))
    df["target_cum"] = np.cumsum((~is_decoy).astype(int))
    df["FDR"] = df["decoy_cum"] / df["target_cum"].clip(lower=1)
    df["qvalue"] = df["FDR"][::-1].cummin()[::-1]
    return df[(df["qvalue"] <= fdr_threshold) & (~df["is_decoy"])].copy()


def _strip_mods_except(peptide: str, keep_regex: str) -> str:
    """Keep amino-acid letters and only bracketed mods matching keep_regex."""
    s = str(peptide)
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if "A" <= ch <= "Z":
            out.append(ch)
            i += 1
        elif ch == "[":
            j = s.find("]", i + 1)
            if j == -1:
                i += 1
                continue
            content = s[i + 1:j]
            if re.search(keep_regex, content):
                out.append(s[i:j + 1])
            i = j + 1
        else:
            i += 1
    return "".join(out)


def add_sage_peptide_columns(df: pd.DataFrame,
                             peptide_col: str = "peptide",
                             mod_keep_regex: str = r"\+79\.") -> pd.DataFrame:
    out = df.copy()
    if peptide_col not in out.columns:
        raise KeyError(f"Sage peptide column '{peptide_col}' not found.")
    out["peptide_clean"] = out[peptide_col].astype(str).str.replace(r"[^A-Z]", "", regex=True)
    out["peptide_phospho"] = out[peptide_col].astype(str).map(
        lambda p: _strip_mods_except(p, mod_keep_regex))
    return out


def _kept_mod_offsets(peptide_phospho: str, keep_regex: str) -> List[int]:
    """
    Return offsets of kept modifications as the number of residues preceding
    the mod bracket. With 0-based protein start (seq.find), start + offset is
    the conventional 1-based protein position.
    """
    s = str(peptide_phospho)
    offsets = []
    aa_count = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if "A" <= ch <= "Z":
            aa_count += 1
            i += 1
        elif ch == "[":
            j = s.find("]", i + 1)
            if j == -1:
                break
            content = s[i + 1:j]
            if re.search(keep_regex, content):
                offsets.append(aa_count)
            i = j + 1
        else:
            i += 1
    return offsets


def map_mod_positions(results: pd.DataFrame,
                      fasta_dict: Dict[str, str],
                      mod_keep_regex: str = r"\+79\.",
                      all_proteins: bool = False,
                      mod_name: str = "Phospho") -> pd.DataFrame:
    """Map Sage modified peptides to protein positions without bracket-shift bugs."""
    rows = []
    for row in results.itertuples(index=False):
        clean_pep = getattr(row, "peptide_clean", None)
        pep_ph = getattr(row, "peptide_phospho", None)
        pep = getattr(row, "peptide", None)
        if not clean_pep or not pep_ph:
            continue
        offsets = _kept_mod_offsets(pep_ph, mod_keep_regex)
        if not offsets:
            continue
        prot_ids = _split_proteins(getattr(row, "proteins", ""))
        if not prot_ids:
            continue
        if not all_proteins:
            prot_ids = prot_ids[:1]
        base = row._asdict()
        base.pop("proteins", None)
        for prot_id in prot_ids:
            seq = fasta_dict.get(prot_id)
            if not seq:
                continue
            start = seq.find(clean_pep)
            if start == -1:
                continue
            for off in offsets:
                rec = dict(base)
                rec["protein"] = prot_id
                rec["position_in_protein"] = int(start + off)
                rec["peptide_phospho"] = pep_ph
                rec["Modification"] = mod_name
                rec["peptide"] = pep
                rows.append(rec)
    return pd.DataFrame(rows)


def samples_annotation_sage(full_df: pd.DataFrame, group_df_link: str) -> pd.DataFrame:
    """Sage variant of sample annotation with dynamic TMT_groupN parsing."""
    if full_df.empty:
        return full_df
    df = full_df.copy()
    if "file_name" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "file_name"})
    if "file_name" not in df.columns:
        raise KeyError("Sage results need a 'filename' or 'file_name' column.")
    df['file_name'] = df['file_name'].str.split('.').str[0]
    try:
        group_df = pd.read_csv(group_df_link, sep=None, engine="python")
    except Exception as e:
        raise RuntimeError(f"Error reading grouping file {group_df_link}: {e}")
    if "file_name" not in group_df.columns and "filename" in group_df.columns:
        group_df = group_df.rename(columns={"filename": "file_name"})
    if "file_name" not in group_df.columns or "batch" not in group_df.columns:
        raise KeyError("Grouping file must contain 'file_name' and 'batch' columns.")

    merged = df.merge(group_df, how="left", on="file_name")
    group_cols = [c for c in merged.columns if re.fullmatch(r"TMT_group\d+", str(c))]
    if not group_cols:
        raise KeyError("Grouping file must contain at least TMT_group1/TMT_group2 columns.")
    missing = merged.loc[merged[group_cols[0]].isna(), "file_name"].nunique()
    if missing:
        logger.warning(f"There are no annotations for {missing} files.")
    merged = merged[merged[group_cols[0]].notna()].copy()
    for c in group_cols + (["mix_channels"] if "mix_channels" in merged.columns else []):
        merged[c] = merged[c].map(_parse_channel_cell)
    merged["batch"] = merged["batch"].astype(int)
    return merged


def _sage_result_dirs(sage_dir: str, results_filename: str, tmt_filename: str) -> List[str]:
    if os.path.isfile(os.path.join(sage_dir, results_filename)) and \
       os.path.isfile(os.path.join(sage_dir, tmt_filename)):
        return [sage_dir]
    dirs = []
    for d in sorted(glob.glob(os.path.join(sage_dir, "*"))):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, results_filename)) and \
           os.path.isfile(os.path.join(d, tmt_filename)):
            dirs.append(d)
    return dirs


def prepare_sage_phospho(sage_dir: str,
                         grouping_file: str,
                         fasta_file: str,
                         fdr_threshold: float = 0.05,
                         results_filename: str = "results.sage.tsv",
                         tmt_filename: str = "tmt.tsv",
                         intensity_prefix: str = "tmt_",
                         fdr_method: str = "spectrum_peptide_q",
                         score_column: str = "sage_discriminant_score",
                         decoy_regex: str = r"DECOY|rev_",
                         mod_keep_regex: str = r"\+79\.",
                         mod_name: str = "Phospho",
                         require_mod: bool = True,
                         map_all_proteins: bool = False) -> pd.DataFrame:
    """
    Build annotated_df for the Sage/enriched-PTM branch.

    Output columns include: file_name, scannr, batch, TMT_group*, mix_channels,
    protein, position_in_protein, peptide (= modified peptide), peptide_clean,
    peptide_phospho, charge, isotope_error, Modification and raw tmt_* columns.
    """
    result_dirs = _sage_result_dirs(sage_dir, results_filename, tmt_filename)
    if not result_dirs:
        raise FileNotFoundError(
            f"No Sage result folders with {results_filename} and {tmt_filename} found in {sage_dir}")

    fasta_dict = load_fasta_dict(fasta_file)
    all_mapped = []
    fdr_method = str(fdr_method or "spectrum_peptide_q").strip().lower()

    for i, d in enumerate(result_dirs):
        logger.info(f"Sage preprocessing [{i + 1}/{len(result_dirs)}]: {d}")
        results = pd.read_csv(os.path.join(d, results_filename), sep="\t", low_memory=False)
        if "is_decoy" not in results.columns:
            results["is_decoy"] = results["proteins"].astype(str).str.contains(
                decoy_regex, case=False, regex=True)

        if fdr_method == "discriminant_score":
            filtered = target_decoy_filter(results, fdr_threshold=fdr_threshold,
                                           score_column=score_column, decoy_regex=decoy_regex)
        else:
            needed = {"spectrum_q", "peptide_q"}
            if not needed.issubset(results.columns):
                logger.warning(f"{needed} not found in {d}; falling back to discriminant_score FDR.")
                filtered = target_decoy_filter(results, fdr_threshold=fdr_threshold,
                                               score_column=score_column, decoy_regex=decoy_regex)
            else:
                filtered = results[(results["spectrum_q"] <= fdr_threshold) &
                                   (results["peptide_q"] <= fdr_threshold) &
                                   (~results["is_decoy"])].copy()

        filtered = add_sage_peptide_columns(filtered, peptide_col="peptide",
                                            mod_keep_regex=mod_keep_regex)
        if require_mod:
            filtered = filtered[filtered["peptide_phospho"].astype(str).str.contains(
                mod_keep_regex, regex=True, na=False)].copy()
        if filtered.empty:
            logger.warning(f"No PSMs left after filtering in {d}")
            continue

        annot_df = samples_annotation_sage(filtered, grouping_file)

        tmt = pd.read_csv(os.path.join(d, tmt_filename), sep="\t", low_memory=False)
        if "file_name" not in tmt.columns and "filename" in tmt.columns:
            tmt = tmt.rename(columns={"filename": "file_name"})
        intensity_cols = [c for c in tmt.columns if str(c).startswith(intensity_prefix)]
        if not intensity_cols:
            raise KeyError(f"No '{intensity_prefix}*' reporter columns found in {os.path.join(d, tmt_filename)}")
        for key in ("file_name", "scannr"):
            if key not in annot_df.columns or key not in tmt.columns:
                raise KeyError(f"Merge key '{key}' must be present in both Sage results and {tmt_filename}.")
        annot_df["scannr"] = annot_df["scannr"].astype(str)
        tmt["scannr"] = tmt["scannr"].astype(str)
        annot_df["file_name"] = annot_df["file_name"].astype(str)
        tmt["file_name"] = tmt["file_name"].astype(str)
        tmt['file_name'] = tmt['file_name'].str.split('.').str[0]
        tmt_small = tmt[["file_name", "scannr"] + intensity_cols].copy()
        merged = annot_df.merge(tmt_small, on=["file_name", "scannr"], how="inner")
        if merged.empty:
            logger.warning(f"No PSMs matched between results.sage.tsv and tmt.tsv in {d}")
            continue
        all_mapped.append(merged)

    if not all_mapped:
        return pd.DataFrame()
    full_results = pd.concat(all_mapped, ignore_index=True)
    if "isotope_error" not in full_results.columns:
        full_results["isotope_error"] = 0
    mapped = map_mod_positions(full_results, fasta_dict,
                               mod_keep_regex=mod_keep_regex,
                               all_proteins=map_all_proteins,
                               mod_name=mod_name)
    if mapped.empty:
        logger.warning("No modified positions could be mapped to the FASTA.")
    return mapped
