"""
TMT reporter-intensity normalization and PSM filtering helpers.

Drop-in replacement for the pasted tmt_normalization()/sorting_psms() code.
Python 3.10 compatible.

Key changes relative to the reviewed version
-------------------------------------------
1. Row count and order are preserved: normalized values are written back by
   index, not via a merge on spectrum keys (the old merge could multiply rows
   when duplicate_spectrum keys were not unique).
2. Zeros are treated as missing BEFORE min_fraction_valid filtering and log2.
3. normalize_target='auto' is type_experiment-aware:
   - whole proteome: per-batch median channel centering; GIS (if present) is
     used only for cross-batch alignment;
   - phospho enrichment: GIS-based scalar alignment when a global internal
     standard is available; median centering is only a fallback and is
     explicitly warned about, because it assumes most phosphosites are
     unchanged and can remove real global phosphorylation shifts.
4. The GIS branch no longer double-shifts GIS channels and no longer silently
   skips normalization when use_gis_for_batch=False.
5. sorting_psms() parses TMT_groupN annotations robustly, supports more than
   two groups, operates on normalized *_norm columns when they exist, and does
   NOT impute by default. Median imputation of missing TMT channels biases
   logFC toward zero and understates variance; the downstream WLS/limma code
   handles per-channel missingness explicitly.
"""

import ast
import logging
import re
from collections import defaultdict
from typing import List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

logger = logging.getLogger(__name__)


def _unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_channel_cell(x) -> List[str]:
    """Parse list-like annotation cells ("['126','127']", "126,127", [126, 127])."""
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
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]
    return [str(x).strip()]


def _raw_intensity_columns(df: pd.DataFrame, intensity_prefix: str, norm_suffix: str) -> List[str]:
    cols = []
    for c in df.columns:
        c = str(c)
        if not c.startswith(intensity_prefix):
            continue
        if c.endswith(norm_suffix) or c.endswith("_var") or c.endswith("_prec"):
            continue
        cols.append(c)
    return cols


def _map_to_raw_intensity(tokens: Sequence[str], intensity_cols: Sequence[str], intensity_prefix: str) -> List[str]:
    available = set(intensity_cols)
    mapped = []
    for tok in tokens:
        s = str(tok).strip()
        if not s:
            continue
        candidates = [s, f"{intensity_prefix}{s}"]
        for cand in candidates:
            if cand in available:
                mapped.append(cand)
                break
    return _unique_preserve_order(mapped)


def _map_for_sort(tokens: Sequence[str], df_cols: Sequence[str], intensity_prefix: str, norm_suffix: str) -> List[str]:
    """Map annotation tokens to analysis columns, preferring normalized *_norm columns."""
    cols = set(map(str, df_cols))
    mapped = []
    for tok in tokens:
        s = str(tok).strip()
        if not s:
            continue
        candidates = [s, f"{intensity_prefix}{s}", f"{s}{norm_suffix}", f"{intensity_prefix}{s}{norm_suffix}"]
        chosen = None
        for cand in candidates:
            if cand not in cols:
                continue
            if cand.endswith(norm_suffix):
                chosen = cand
                break
            if cand.startswith(intensity_prefix):
                norm = f"{cand}{norm_suffix}"
                chosen = norm if norm in cols else cand
                break
        if chosen is not None:
            mapped.append(chosen)
    return _unique_preserve_order(mapped)


def _batch_series(df: pd.DataFrame, batch_col: Optional[str] = None) -> pd.Series:
    if batch_col is not None and batch_col in df.columns:
        return df[batch_col]
    if "batch" in df.columns:
        return df["batch"]
    if "file_name" in df.columns:
        return df["file_name"]
    return pd.Series(["__all__"] * len(df), index=df.index)


def tmt_normalization(df: pd.DataFrame,
                      intensity_prefix: str = "intensity_",
                      min_fraction_valid: float = 0.5,
                      use_gis_for_batch: bool = True,
                      gis_column: str = "mix_channels",
                      normalize_target: Literal["auto", "median", "gis"] = "auto",
                      type_experiment: str = "whole proteome",
                      return_suffix: str = "_norm",
                      duplicate_spectrum: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Log2-transform and normalize TMT reporter intensities.

    Returns a copy of df with added `<intensity_*><return_suffix>` columns.
    Original intensity columns are left untouched. Rows failing
    min_fraction_valid receive NaN in normalized columns; the table shape is
    never changed.

    duplicate_spectrum is accepted for backward compatibility. Normalization
    factors are robust medians, so duplicate spectra do not need to be dropped
    to write values back safely by index.
    """
    del duplicate_spectrum  # kept only for API compatibility; no merge is used

    out = df.copy()
    intensity_cols = _raw_intensity_columns(out, intensity_prefix, return_suffix)
    norm_cols = [f"{c}{return_suffix}" for c in intensity_cols]
    for c in norm_cols:
        out[c] = np.nan

    if out.empty:
        logger.warning("Input DataFrame is empty. Skipping TMT normalization.")
        return out
    if not intensity_cols:
        logger.warning("No raw intensity columns found. Skipping TMT normalization.")
        return out

    # Zeros are missing reporter signals in TMT, not measured values.
    X = out[intensity_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    min_valid = max(1, int(np.ceil(len(intensity_cols) * float(min_fraction_valid))))
    valid_mask = X.notna().sum(axis=1) >= min_valid
    work_idx = out.index[valid_mask]
    if len(work_idx) == 0:
        logger.warning("All PSMs filtered by min_fraction_valid; normalized columns left as NaN.")
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        X_log = np.log2(X.loc[work_idx, intensity_cols])

    batches = _batch_series(out.loc[work_idx])
    batch_levels = list(pd.unique(batches))

    # GIS channels per batch, using the first non-empty annotation in the batch.
    batch_gis = {}
    if use_gis_for_batch and gis_column in out.columns:
        for b in batch_levels:
            idx = batches[batches == b].index
            cols = []
            for cell in out.loc[idx, gis_column].dropna():
                cols = _map_to_raw_intensity(_parse_channel_cell(cell), intensity_cols, intensity_prefix)
                if cols:
                    break
            batch_gis[b] = cols
    has_gis = any(len(v) > 0 for v in batch_gis.values())

    phospho = str(type_experiment).strip().lower() == "phospho enrichment"
    requested = str(normalize_target or "auto").strip().lower()
    if requested not in {"auto", "median", "gis"}:
        logger.warning(f"Unknown normalize_target={normalize_target!r}; using 'auto'.")
        requested = "auto"

    if requested == "auto":
        mode = "gis" if (phospho and has_gis) else "median"
    elif requested == "gis":
        mode = "gis" if has_gis else "median"
        if not has_gis:
            logger.warning("normalize_target='gis' requested but no GIS channels were found; falling back to median.")
    else:
        mode = "median"

    n_gis_batches = sum(1 for v in batch_gis.values() if len(v) > 0)
    if mode == "gis" and n_gis_batches < 2:
        logger.warning("GIS normalization needs GIS channels in at least two batches; falling back to median.")
        mode = "median"

    if phospho and mode == "median":
        logger.warning(
            "Phospho-enrichment data without usable GIS are median-normalized. "
            "This assumes most phosphosites are unchanged across channels and can "
            "remove real global phosphorylation differences. Prefer a global "
            "internal standard (mix_channels) for enrichment designs."
        )

    if mode == "median":
        # Within-plex channel centering: correct unequal loading/labeling while
        # keeping batch-specific scale separate before cross-batch alignment.
        for b in batch_levels:
            idx = batches[batches == b].index
            ch_med = X_log.loc[idx, intensity_cols].median(axis=0, skipna=True)
            if ch_med.notna().sum() == 0:
                continue
            center = float(np.nanmedian(ch_med.to_numpy(dtype=float)))
            X_log.loc[idx, intensity_cols] = X_log.loc[idx, intensity_cols].subtract(ch_med, axis=1).add(center)

        # Cross-batch alignment: prefer GIS scalar if available, otherwise the
        # all-channel median of the already within-batch-centered values.
        shifts = {}
        for b in batch_levels:
            idx = batches[batches == b].index
            gis_cols = batch_gis.get(b, []) if use_gis_for_batch else []
            source_cols = gis_cols if gis_cols else intensity_cols
            vals = X_log.loc[idx, source_cols].to_numpy(dtype=float)
            shift = np.nanmedian(vals)
            if np.isfinite(shift):
                shifts[b] = float(shift)
        if len(shifts) > 1:
            global_shift = float(np.nanmedian(list(shifts.values())))
            for b, shift in shifts.items():
                idx = batches[batches == b].index
                X_log.loc[idx, intensity_cols] = X_log.loc[idx, intensity_cols].subtract(shift - global_shift, axis=0)

    elif mode == "gis":
        # Enrichment design: align plexes by the global internal standard only.
        # No channel-wise median centering is applied, so global phosphorylation
        # changes within a plex are not normalized away.
        shifts = {}
        for b in batch_levels:
            gis_cols = batch_gis.get(b, [])
            if not gis_cols:
                continue
            idx = batches[batches == b].index
            vals = X_log.loc[idx, gis_cols].to_numpy(dtype=float)
            shift = np.nanmedian(vals)
            if np.isfinite(shift):
                shifts[b] = float(shift)
        if len(shifts) > 1:
            global_shift = float(np.nanmedian(list(shifts.values())))
            for b, shift in shifts.items():
                idx = batches[batches == b].index
                X_log.loc[idx, intensity_cols] = X_log.loc[idx, intensity_cols].subtract(shift - global_shift, axis=0)

    out.loc[work_idx, norm_cols] = X_log[intensity_cols].to_numpy(dtype=float)
    return out


def sorting_psms(df_copy: pd.DataFrame,
                 intensity_prefix: str = "intensity_",
                 normalized_suffix: str = "_norm",
                 group_prefix: str = "TMT_group",
                 batch_col: str = "batch",
                 max_missing_fraction: float = 0.5,
                 impute_missing: bool = False,
                 impute_low: bool = False,
                 low_factor: float = 0.5):
    """
    Filter PSMs by within-group channel completeness; optionally impute.

    Defaults are deliberately conservative: no imputation. The function only
    removes PSMs whose per-TMT-group missing fraction exceeds
    max_missing_fraction. Downstream site aggregation/WLS uses observed channels
    and per-channel precisions, so median imputation is not required and is
    generally harmful for differential testing.

    Returns (df, stat_dict, num_deleted, delete_indices) for backward
    compatibility.
    """
    if df_copy.empty:
        logger.warning("Input DataFrame is empty. Skipping PSM sorting.")
        return df_copy, {}, 0, []

    if batch_col in df_copy.columns:
        df = df_copy.sort_values(by=batch_col).reset_index(drop=True)
        batches = df[batch_col]
    else:
        df = df_copy.reset_index(drop=True)
        batches = pd.Series(["__all__"] * len(df), index=df.index)

    group_cols = [c for c in df.columns if re.fullmatch(rf"{re.escape(group_prefix)}\d+", str(c))]
    group_cols = sorted(group_cols, key=lambda c: int(re.search(r"\d+", str(c)).group()))

    raw_cols = _raw_intensity_columns(df, intensity_prefix, normalized_suffix)
    fallback_targets = [f"{c}{normalized_suffix}" if f"{c}{normalized_suffix}" in df.columns else c
                        for c in raw_cols]
    any_log_target = any(str(c).endswith(normalized_suffix) for c in fallback_targets)
    if impute_low and any_log_target:
        logger.warning(
            "impute_low was requested on log2-normalized columns; low-value "
            "replacement by median*low_factor is only defined for positive "
            "linear intensities. impute_low was disabled."
        )
        impute_low = False

    groups_cache = {}
    for b in pd.unique(batches):
        first_pos = batches[batches == b].index[0]
        row = df.loc[first_pos]
        per_group = []
        for gc in group_cols:
            cols = _map_for_sort(_parse_channel_cell(row.get(gc)), df.columns, intensity_prefix, normalized_suffix)
            if cols:
                per_group.append(cols)
        if not per_group and fallback_targets:
            per_group = [fallback_targets]
        groups_cache[b] = per_group

    stat = defaultdict(int)
    delete_indices = []
    num_deleted = 0
    max_missing_fraction = float(np.clip(max_missing_fraction, 0.0, 1.0))

    row_iter = range(len(df))
    if tqdm is not None:
        row_iter = tqdm(row_iter, total=len(df), desc="Processing sorting intensity")

    for row_idx in row_iter:
        b = batches.iloc[row_idx]
        per_group = groups_cache.get(b)
        if per_group is None:
            delete_indices.append(row_idx)
            num_deleted += 1
            continue

        drop_row = False
        for cols in per_group:
            if not cols:
                continue
            vals = df.loc[row_idx, cols].to_numpy(dtype=float, copy=True)
            vals = np.where(vals == 0, np.nan, vals)  # zeros are missing in TMT
            n = vals.size
            if n == 0:
                continue
            finite = np.isfinite(vals)
            n_obs = int(finite.sum())
            n_nan = int(n - n_obs)

            if n_obs == 0 or n_nan > max_missing_fraction * n:
                drop_row = True
                break

            if (impute_missing or impute_low) and n_obs > 0:
                med = float(np.nanmedian(vals))
                changed = False
                if impute_low and np.isfinite(med):
                    low = finite & (vals < med * float(low_factor))
                    if low.any():
                        vals[low] = med
                        for col in np.array(cols)[low]:
                            stat[col] += 1
                        changed = True
                if impute_missing:
                    miss = ~np.isfinite(vals)
                    if miss.any():
                        vals[miss] = med
                        for col in np.array(cols)[miss]:
                            stat[col] += 1
                        changed = True
                if changed:
                    df.loc[row_idx, cols] = vals

        if drop_row:
            delete_indices.append(row_idx)
            num_deleted += 1

    if delete_indices:
        df.drop(index=delete_indices, inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)

    return df, dict(stat), num_deleted, delete_indices


def impute_tmt_psms(df_copy: pd.DataFrame, **kwargs):
    """Backward-compatible alias for sorting_psms()."""
    return sorting_psms(df_copy, **kwargs)




#########################
'''
def tmt_normalization(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping TMT normalization.")
        return df
    intensity_cols = [c for c in df.columns if c.startswith("intensity_")]
    calc_norm = df.drop_duplicates(subset='spectrum_y')

    # Медианное центрирование внутри batch
    def median_centering(group):
        medians = group[intensity_cols].median(axis=0)
        group[intensity_cols] = group[intensity_cols] / medians
        
        batch_median = group[intensity_cols].stack().median()
        group[intensity_cols] = group[intensity_cols] / batch_median
        return group

    df_norm = calc_norm.groupby("batch", group_keys=False).apply(median_centering)
    final_df = df.merge(df_norm[intensity_cols + ['spectrum_y']], how = 'left', on = 'spectrum_y', suffixes=('', '_norm'))
    return final_df
'''
def tmt_normalization(df: pd.DataFrame,
                      intensity_prefix: str = "intensity_",
                      min_fraction_valid: float = 0.5,
                      use_gis_for_batch: bool = True,
                      gis_column: str = "mix_channels", 
                      normalize_target: Literal["median", "gis"] = "median",
                      return_suffix: str = "_norm",
                      duplicate_spectrum: list = ['spectrum_y']
                     ) -> pd.DataFrame:
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping TMT normalization.")
        return df
    def parse_mix_channels_cell(x) -> List[str]:
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                if isinstance(v, (list, tuple)):
                    return list(v)
            except Exception:
                # Fallback: comma separated
                parts = [p.strip() for p in x.split(",") if p.strip()]
                return parts
        return []
    intensity_cols = [c for c in df.columns if c.startswith("intensity_")]
    calc = df.drop_duplicates(subset=duplicate_spectrum)

    # ---------- remove duplicated scans ----------

    key_cols = duplicate_spectrum
    has_keys = all(k in df.columns for k in key_cols)
    working = df.copy()
    
    min_valid = int(max(1, round(len(intensity_cols) * float(min_fraction_valid))))
    valid_counts = working[intensity_cols].notna().sum(axis=1)
    mask = valid_counts >= min_valid
    if mask.sum() == 0:
        logger.warning("All PSMs filtered by min_fraction_valid threshold — returning original with no normalization.")
        return df.copy()
    working = working.loc[mask].reset_index(drop=True)

    # ---------- handle zeros / missing before log ----------
    X = working[intensity_cols].copy().astype(float)
    X = X.replace(0, np.nan)

    # ---------- log2 transform ----------
    with np.errstate(divide='ignore', invalid='ignore'):
        X_log = np.log2(X)

    # ---------- GIS (batch) correction (robust) ----------
    batch_shift = {}
    if use_gis_for_batch and gis_column in working.columns:
        batch_col = "batch" if "batch" in working.columns else "file_name"
        for batch, g in working.groupby(batch_col):
            mix_vals = g[gis_column].dropna()
            if mix_vals.empty:
                continue
            mix_info = mix_vals.iloc[0]
            gis_list = parse_mix_channels_cell(mix_info)
            gis_cols = []
            for c in gis_list:
                if isinstance(c, str) and c.startswith(intensity_prefix) and c in intensity_cols:
                    gis_cols.append(c)
                else:
                    try:
                        maybe = f"{intensity_prefix}{str(c)}"
                        if maybe in intensity_cols:
                            gis_cols.append(maybe)
                    except Exception:
                        pass
            if not gis_cols:
                continue

            # per-PSM median across GIS cols (robust per-PSM summary)
            per_psm_median = X_log.loc[g.index, gis_cols].median(axis=1, skipna=True)
            shift = per_psm_median.median(skipna=True)  # robust center for batch
            if pd.notna(shift):
                batch_shift[batch] = float(shift)

    # apply batch shifts if found
    if batch_shift:
        global_center = np.nanmedian(list(batch_shift.values()))
        batch_col = "batch" if "batch" in working.columns else "file_name"
        for batch, shift in batch_shift.items():
            idx = working[batch_col] == batch
            X_log.loc[idx, :] = X_log.loc[idx, :].subtract((shift - global_center), axis=0)

    # ---------- channel centering / reference normalization ----------
    # two options:
    #  - 'median' : median across all rows (robust global centering)
    #  - 'gis'    : use GIS channels (if present) as reference for scaling (recommended for phospho)
    if normalize_target == "gis" and use_gis_for_batch and gis_column in working.columns and batch_shift:
        all_gis_vals = []
        for batch, g in working.groupby("batch" if "batch" in working.columns else "file_name"):
            mix_vals = g[gis_column].dropna()
            if mix_vals.empty:
                continue
            mix_info = mix_vals.iloc[0]
            gis_list = parse_mix_channels_cell(mix_info)
            gis_cols = []
            for c in gis_list:
                if isinstance(c, str) and c.startswith(intensity_prefix) and c in intensity_cols:
                    gis_cols.append(c)
                else:
                    maybe = f"{intensity_prefix}{str(c)}"
                    if maybe in intensity_cols:
                        gis_cols.append(maybe)
            if not gis_cols:
                continue
            # collect values
            sub = X_log.loc[g.index, gis_cols]
            # per-channel medians (skipna)
            ch_meds = sub.median(axis=0, skipna=True)
            if not ch_meds.empty:
                all_gis_vals.append(ch_meds)
        if all_gis_vals:
            # DataFrame of per-batch GIS channel medians
            gis_meds_df = pd.DataFrame(all_gis_vals)
            # global median per GIS channel
            global_gis_meds = gis_meds_df.median(axis=0, skipna=True)
            # now for each batch compute its GIS per-channel medians and shift toward global
            batch_col = "batch" if "batch" in working.columns else "file_name"
            for batch, g in working.groupby(batch_col):
                mix_vals = g[gis_column].dropna()
                if mix_vals.empty:
                    continue
                mix_info = mix_vals.iloc[0]
                gis_list = parse_mix_channels_cell(mix_info)
                gis_cols = []
                for c in gis_list:
                    if isinstance(c, str) and c.startswith(intensity_prefix) and c in intensity_cols:
                        gis_cols.append(c)
                    else:
                        maybe = f"{intensity_prefix}{str(c)}"
                        if maybe in intensity_cols:
                            gis_cols.append(maybe)
                if not gis_cols:
                    continue
                batch_gis_meds = X_log.loc[g.index, gis_cols].median(axis=0, skipna=True)
                if batch_gis_meds.shape[0] == global_gis_meds.shape[0]:
                    per_channel_shift = batch_gis_meds - global_gis_meds
                    idx = (working[batch_col] == batch)
                    X_log.loc[idx, gis_cols] = X_log.loc[idx, gis_cols].subtract(per_channel_shift, axis=1)
                    scalar_shift = np.nanmedian(per_channel_shift.values)
                    X_log.loc[idx, :] = X_log.loc[idx, :].subtract(scalar_shift, axis=1)
                else:
                    pass
        else:
            # if no GIS medians assembled — fallback to median normalization
            normalize_target = "median"

    if normalize_target == "median":
        # simple robust column median centering
        channel_medians = X_log.median(axis=0, skipna=True)
        global_center = np.nanmedian(channel_medians.values)
        X_log = X_log.subtract(channel_medians, axis=1).add(global_center)

    # ---------- prepare final output: do not overwrite original intensities ----------
    # produce normalized columns and merge back to original df structure
    norm_df = X_log.copy()
    norm_df.columns = [f"{c}{return_suffix}" for c in norm_df.columns]

    merge_keys = duplicate_spectrum
    final = df.copy()
    to_merge = pd.concat([working[merge_keys].reset_index(drop=True), norm_df.reset_index(drop=True)], axis=1)
    final = final.merge(to_merge, on=merge_keys, how="left")

    return final



def sorting_psms(df_copy: pd.DataFrame):
    if df_copy.empty:
        logger.warning("Input DataFrame is empty. Skipping PSM sorting.")
        return df_copy, {}, 0, []

    df_copy = df_copy.sort_values(by="batch").reset_index(drop=True)

    #intensity_cols = [c for c in df_copy.columns if c.startswith("intensity_")]
    stat = defaultdict(int)
    delete_indices = []
    num_deleted = 0
    
    groups_cache = {}
    for batch, group_df in df_copy.groupby("batch"):
        first_row = group_df.iloc[0]
        try:
            ad_cols = [f"intensity_{x}" for x in first_row["TMT_group1"]]
            control_cols = [f"intensity_{x}" for x in first_row["TMT_group2"]]
        except Exception as e:
            logger.error(f"Error caching channel groups for batch {batch}: {e}")
            ad_cols, control_cols = [], []
        groups_cache[batch] = {"ad": ad_cols, "control": control_cols}

    for row_idx, row in tqdm(enumerate(df_copy.itertuples(index=False)), total=len(df_copy),
                             desc="Processing sorting intensity"):
        try:
            batch = getattr(row, "batch")
        except AttributeError:
            delete_indices.append(row_idx)
            num_deleted += 1
            continue

        groups = groups_cache.get(batch)
        if groups is None:
            logger.debug(f"Batch {batch} not found in cache; marking row {row_idx} for deletion.")
            delete_indices.append(row_idx)
            num_deleted += 1
            continue

        for group_cols in (groups["ad"], groups["control"]):
            if not group_cols:
                continue
            try:
                vals = df_copy.loc[row_idx, group_cols].to_numpy(dtype=float)
            except KeyError as e:
                logger.error(f"Missing intensity columns at row {row_idx}: {e}")
                delete_indices.append(row_idx)
                num_deleted += 1
                break

            row_median = np.nanmedian(vals)
            if np.isnan(row_median):
                delete_indices.append(row_idx)
                num_deleted += 1
                break

            low_mask = vals < (row_median * 0.5)
            nan_mask = np.isnan(vals)

            if low_mask.any():
                cols_low = np.array(group_cols)[low_mask]
                df_copy.loc[row_idx, cols_low] = row_median
                for col in cols_low:
                    stat[col] += 1

            if nan_mask.any():
                n_nan = nan_mask.sum()
                if n_nan <= len(vals) / 2:
                    cols_nan = np.array(group_cols)[nan_mask]
                    df_copy.loc[row_idx, cols_nan] = row_median
                else:
                    delete_indices.append(row_idx)
                    num_deleted += 1
                    break

    if delete_indices:
        df_copy.drop(index=delete_indices, inplace=True, errors="ignore")
        df_copy.reset_index(drop=True, inplace=True)

    return df_copy, dict(stat), num_deleted, delete_indices


def impute_tmt_psms(
    df: pd.DataFrame,
    min_valid_fraction: float = 0.5,
    downshift: float = 1.8,
    width: float = 0.3,
):
    """
    Biologically informed left-censored imputation for TMT PSMs.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return df, {}, 0, []

    df = df.sort_values("batch").reset_index(drop=True)

    stat = defaultdict(int)
    delete_indices = []

    # cache channel groups per batch
    groups_cache = {}
    for batch, g in df.groupby("batch"):
        row = g.iloc[0]
        groups_cache[batch] = {
            "ad": [f"intensity_{x}_norm" for x in row["TMT_group1"]],
            "control": [f"intensity_{x}_norm" for x in row["TMT_group2"]],
        }

    for idx in tqdm(range(len(df)), desc="Imputing TMT intensities"):
        batch = df.at[idx, "batch"]
        groups = groups_cache.get(batch)

        if groups is None:
            delete_indices.append(idx)
            continue

        for group_name, cols in groups.items():
            if not cols:
                continue

            vals = df.loc[idx, cols].astype(float).to_numpy()
            valid = ~np.isnan(vals)

            # too many missing → drop
            if valid.sum() < len(vals) * min_valid_fraction:
                delete_indices.append(idx)
                break

            # work in log2
            log_vals = np.log2(vals[valid])

            mu = np.mean(log_vals)
            sigma = np.std(log_vals, ddof=1)

            if sigma == 0 or np.isnan(sigma):
                delete_indices.append(idx)
                break

            mu_imp = mu - downshift * sigma
            sigma_imp = width * sigma

            nan_mask = np.isnan(vals)
            if nan_mask.any():
                n = nan_mask.sum()
                imputed = np.random.normal(mu_imp, sigma_imp, n)
                df.loc[idx, np.array(cols)[nan_mask]] = 2 ** imputed

                for c in np.array(cols)[nan_mask]:
                    stat[c] += 1

    if delete_indices:
        df.drop(index=delete_indices, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df, dict(stat), len(delete_indices), delete_indices
