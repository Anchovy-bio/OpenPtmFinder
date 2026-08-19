import itertools
import logging
import re
import warnings
from math import comb
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.optimize import brentq
from scipy.special import polygamma
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _pos_to_str(series: pd.Series) -> pd.Series:
    """Safe conversion of position column to string (NaN -> '<NA>')."""
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def _estimate_k_shrink(vars_, counts_local, floor=2.0, ceil=200.0, winsor_q=(0.05, 0.95)):
    """
    Estimate prior degrees of freedom k (analogous to limma's d0)
    from the distribution of log-variances using method of moments.

    Var(log s^2) = trigamma(k/2) + trigamma(df/2)  (Smyth 2004, fitFDist).
    """
    mask = np.isfinite(vars_) & (vars_ > 0) & (counts_local >= 3)
    if mask.sum() < 5:
        return floor

    log_var = np.log(vars_[mask])
    lo, hi = np.percentile(log_var, [100.0 * winsor_q[0], 100.0 * winsor_q[1]])
    log_var_w = np.clip(log_var, lo, hi)

    df_obs = counts_local[mask] - 1
    df_mean = float(np.mean(df_obs))

    var_log = float(np.var(log_var_w, ddof=1))

    def obj(k):
        if k <= 0:
            return np.inf
        return polygamma(1, k / 2.0) + polygamma(1, df_mean / 2.0) - var_log

    try:
        k_est = brentq(obj, floor, ceil)
    except ValueError:
        # var_log outside feasible range — clamp to boundary
        k_est = ceil if obj(floor) < 0 else floor
    return float(k_est)


def _estimate_icc_rough(grouped, channel_cols):
    """
    Rough ICC(1,1) estimate from one-way ANOVA across PSM groups.
    Returns median ICC across channels, clipped to [0, 0.95].
    """
    iccs = []
    # Materialize the groups once and replace tuple keys with int indices
    groups_list = list(grouped)

    for col in channel_cols:
        vals = []
        group_ids = []
        for grp_idx, (gid, g) in enumerate(groups_list):
            x = g[col].dropna().to_numpy()
            if len(x) >= 2:
                vals.extend(x.tolist())
                group_ids.extend([grp_idx] * len(x))
        if len(vals) < 10:
            continue

        vals = np.array(vals, dtype=float)
        group_ids = np.array(group_ids, dtype=int)   # guaranteed 1D
        grand_mean = np.mean(vals)

        unique_g = np.unique(group_ids)
        ss_between = 0.0
        n_total = len(vals)
        for ug in unique_g:
            mask = group_ids == ug
            xg = vals[mask]
            ss_between += xg.size * (np.mean(xg) - grand_mean) ** 2

        ss_total = np.sum((vals - grand_mean) ** 2)
        ss_within = ss_total - ss_between

        df_between = len(unique_g) - 1
        df_within = n_total - len(unique_g)
        if df_within <= 0 or df_between <= 0:
            continue

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        k_avg = n_total / len(unique_g)
        denom = ms_between + (k_avg - 1) * ms_within
        if denom <= 0:
            continue
        icc = (ms_between - ms_within) / denom
        if np.isfinite(icc):
            iccs.append(icc)

    if len(iccs) == 0:
        return 0.0
    raw_icc = np.median(iccs)
    return float(np.clip(raw_icc, 0.0, 0.95))


def bayesian_site_aggregation(
    group: pd.DataFrame,
    global_batch: dict,
    method: str = 'aggregate',
    type_experiment: str = 'phospho enrichment',
    k_shrink: float = None,
    channel_cols: list = None,
    icc_mode: str = 'estimate',   # 'estimate' | 'fixed' | 'sqrt' (legacy)
    fixed_icc: float = 0.30,
    huber_c: float = 1.345,
    huber_iters: int = 3,
    var_floor_pct: float = 10.0
) -> pd.Series:
    """
    Robust Bayesian aggregation of PSMs into site-level abundances.

      - Group location = mean (compatible with precision weights)
      - Adaptive std/MAD mixture: alpha = clip((n-3)/27, 0, 1)
      - Data-driven k_shrink via trigamma method-of-moments
      - Effective sample size n_eff = n / (1 + (n-1)*rho) with estimated ICC
      - Huber-IRLS robust aggregation
      - Aggregated variance: Var = 1 / sum(weights); precision kept
        consistent with the (floored) variance on output
    """
    if not channel_cols:
        return pd.Series(dtype=float)

    if 'batch' in group.columns:
        batch = int(group['batch'].iloc[0])
    elif method in ('median', 'protein'):
        # median: (batch, peptide[, protein]); protein: (batch, protein)
        batch = int(group.name[0])
    else:
        # aggregate: (..., batch)
        batch = int(group.name[-1])

    # --- grouping keys ---
    if method in ('aggregate', 'protein'):
        if type_experiment == 'phospho enrichment':
            psm_cols = ["peptide", "charge", "isotope_error"]
        else:
            psm_cols = ["peptide", "charge"]
    elif method == 'median':
        if type_experiment == 'phospho enrichment':
            psm_cols = ["charge", "isotope_error"]
        else:
            psm_cols = ["charge"]
    else:
        raise ValueError("Invalid method")

    # --- numeric matrix ---
    data = group[channel_cols].to_numpy(dtype=float, copy=True)
    data[data == 0] = np.nan
    df_numeric = pd.DataFrame(data, columns=channel_cols, index=group.index)

    group_for_calc = pd.concat([group[psm_cols], df_numeric], axis=1)
    grouped = group_for_calc.groupby(psm_cols, sort=False)

    # --- per-group stats ---
    counts = grouped[channel_cols].count().to_numpy(dtype=float)   # (n_groups, n_channels)
    means = grouped[channel_cols].mean().to_numpy(dtype=float)

    vars_list = []
    for _, g in grouped:
        X = g[channel_cols].to_numpy(dtype=float)
        counts_local = np.sum(~np.isnan(X), axis=0)
        var = np.full(len(channel_cols), np.nan)
        valid = counts_local >= 3

        if np.any(valid):
            Xv = X[:, valid]
            med = np.nanmedian(Xv, axis=0)
            mad = np.nanmedian(np.abs(Xv - med), axis=0)
            std = np.nanstd(Xv, axis=0, ddof=1)

            mad_floor = np.nanpercentile(mad[np.isfinite(mad)], 5) if np.any(np.isfinite(mad)) else 1e-6
            mad = np.maximum(mad, mad_floor)

            # Adaptive convex combination:
            #   n=3  -> pure MAD  (robustness dominates)
            #   n=30 -> pure std  (efficiency dominates)
            alpha = np.clip((counts_local[valid] - 3) / 27, 0, 1)
            var_valid = alpha * (std ** 2) + (1 - alpha) * ((1.4826 * mad) ** 2)
            var[valid] = var_valid

        vars_list.append(var)

    vars_ = np.vstack(vars_list)

    # --- ICC and effective sample size ---
    if icc_mode == 'estimate':
        rho = _estimate_icc_rough(grouped, channel_cols)
    elif icc_mode == 'fixed':
        rho = fixed_icc
    elif icc_mode == 'sqrt':
        rho = None
    else:
        raise ValueError("icc_mode must be 'estimate', 'fixed', or 'sqrt'")

    # --- local & global prior ---
    n_local = np.sum(np.isfinite(vars_), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        local_prior = np.nanmedian(vars_, axis=0)
    local_prior[n_local < 3] = np.nan

    global_prior = global_batch.get(batch, np.ones(len(channel_cols), dtype=float))
    var_prior = 0.5 * local_prior + 0.5 * global_prior
    var_prior = np.where(np.isfinite(var_prior), var_prior, global_prior)

    # --- estimate k_shrink from data if not provided ---
    if k_shrink is None:
        k_shrink = _estimate_k_shrink(vars_, counts, floor=2.0, ceil=200.0)

    # --- shrinkage with effective n ---
    if icc_mode == 'sqrt':
        effective_n = np.sqrt(counts)
    else:
        effective_n = counts / (1.0 + np.maximum(counts - 1, 0) * rho)

    w = effective_n / (effective_n + k_shrink)

    var_shrinked = np.where(
        np.isnan(vars_),
        var_prior[None, :],
        w * vars_ + (1.0 - w) * var_prior[None, :]
    )

    # --- floor per channel ---
    floor = np.nanpercentile(var_shrinked, var_floor_pct, axis=0)
    floor = np.where((floor <= 0) | (~np.isfinite(floor)), 1e-8, floor)
    var_shrinked = np.maximum(var_shrinked, floor[None, :])

    # --- precision ---
    prec_per_group = np.zeros_like(var_shrinked)
    mask = counts > 0
    prec_per_group[mask] = effective_n[mask] / var_shrinked[mask]

    # --- Huber-IRLS robust aggregation ---
    beta = np.full(len(channel_cols), np.nan)
    den = np.nansum(prec_per_group, axis=0)
    num = np.nansum(means * prec_per_group, axis=0)
    valid_init = den > 0
    beta[valid_init] = num[valid_init] / den[valid_init]

    for _ in range(max(int(huber_iters), 0)):
        resid = means - beta[None, :]          # (n_groups, n_channels)
        mad_resid = np.nanmedian(np.abs(resid), axis=0)
        mad_resid = np.maximum(mad_resid, 1e-12)

        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.abs(resid) / mad_resid[None, :]
            huber_w = np.where(r <= huber_c, 1.0, huber_c / r)
        huber_w = np.where(np.isfinite(resid), huber_w, 0.0)

        final_w = prec_per_group * huber_w
        den = np.nansum(final_w, axis=0)
        num = np.nansum(means * final_w, axis=0)
        valid = den > 0
        # Masked division: np.where(valid, num / den, nan) would still
        # evaluate num / den everywhere and flood RuntimeWarning on the
        # (normal) channels where no group has observations (den == 0).
        beta = np.full(len(channel_cols), np.nan)
        beta[valid] = num[valid] / den[valid]

    # --- aggregated variance & precision ---
    out_vals = beta.copy()
    var_agg = np.full(len(channel_cols), np.nan)

    valid = den > 0
    var_agg[valid] = 1.0 / den[valid]      # Var = 1 / sum(weights) — correct for WLS

    # floor for the aggregated variance
    var_floor = np.nanpercentile(var_agg[np.isfinite(var_agg)], 1) if np.any(np.isfinite(var_agg)) else 1e-8
    if not np.isfinite(var_floor) or var_floor <= 0:
        var_floor = 1e-8
    var_agg = np.maximum(var_agg, var_floor)

    # export precision consistent with the (floored) variance so that the
    # (var, prec) pair does not contradict itself in the output files
    prec_agg = np.where(np.isfinite(var_agg), 1.0 / var_agg, np.nan)

    # --- output ---
    out = pd.Series(out_vals, index=channel_cols)
    var_series = pd.Series(var_agg, index=[c.replace('_norm', '_var') for c in channel_cols])
    prec_series = pd.Series(prec_agg, index=[c.replace('_norm', '_prec') for c in channel_cols])

    return pd.concat([out, var_series, prec_series])


def build_expression_noagg(df: pd.DataFrame, method: str = "aggregate",
                           type_experiment: str = 'phospho enrichment',
                           channel_cols: list = None) -> pd.DataFrame:
    """
    Build site x sample matrix without Bayesian aggregation.
    """
    if not channel_cols:
        return pd.DataFrame()

    pos_str = _pos_to_str(df["position_in_protein"]) if "position_in_protein" in df.columns else None

    if method == "aggregate":
        if type_experiment == 'phospho enrichment':
            site = df["protein"].astype(str) + "_" + pos_str
        else:
            site = df["Modification"].astype(str) + "_" + df["protein"].astype(str) + "_" + pos_str
        df2 = df.assign(site=site)
    elif method == "median":
        site = df["protein"].astype(str) + "_" + df["peptide"].astype(str)
        df2 = df.assign(site=site)
    else:
        raise ValueError("invalid method")

    group_cols = ["site", "batch"]
    df_med = df2.groupby(group_cols)[channel_cols].median().reset_index()

    long_df = df_med.melt(id_vars=group_cols, var_name="channel", value_name="value")
    long_df["sample"] = long_df["channel"].str.split("_").str[1] + "_" + long_df["batch"].astype(str)

    expr_noagg = long_df.pivot_table(index="site", columns="sample", values="value", aggfunc="mean")
    return expr_noagg


def _estimate_d0_from_varlog(var_log: float,
                             d0_floor: float = 2.0,
                             d0_ceil: float = 200.0) -> float:
    """
    Solve trigamma(d0/2) = var_log by bisection (trigamma is monotone decreasing).
    """
    var_log = float(var_log)
    if not np.isfinite(var_log) or var_log <= 0:
        return d0_ceil

    lo, hi = d0_floor, d0_ceil
    target = var_log

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = special.polygamma(1, mid / 2.0)
        if val > target:
            lo = mid
        else:
            hi = mid
    return float(np.clip(0.5 * (lo + hi), d0_floor, d0_ceil))


def empirical_bayes_global(sigma2_in,
                           df_resid_in,
                           d0_floor: float = 2.0,
                           d0_ceil: float = 200.0,
                           winsor_q: Tuple[float, float] = (0.05, 0.95)):
    """
    limma-style empirical Bayes shrinkage of residual variances
    (Smyth 2004, fitFDist), generalized to heterogeneous residual df.

    For z = log(s^2), prior sigma^2 ~ scaled-inv-chi2(d0, s0^2):

        E[z]   = log(s0^2) + log(d0/2) - psi(d0/2) + psi(d/2) - log(d/2)
        Var(z) = trigamma(d0/2) + trigamma(d/2)

    We therefore work with  e = z - psi(d/2) + log(d/2),  for which

        E[e]   = log(s0^2) + log(d0/2) - psi(d0/2)
        Var(e) = trigamma(d0/2)

    so that  s0^2 = exp( mean(e) + psi(d0/2) - log(d0/2) )
    and d0 comes from Var(e) after subtracting the sampling part
    mean(trigamma(d/2)). If no residual heterogeneity is detected,
    the prior collapses to a point (d0 = Inf, as in limma).
    """
    sigma2 = np.asarray(sigma2_in, dtype=float).copy()
    df_resid = np.asarray(df_resid_in, dtype=float).copy()

    out_sigma2_post = np.full_like(sigma2, np.nan, dtype=float)
    out_df_total = np.full_like(df_resid, np.nan, dtype=float)

    mask = np.isfinite(sigma2) & np.isfinite(df_resid) & (df_resid > 0) & (sigma2 > 0)
    if mask.sum() < 3:
        out_sigma2_post[mask] = sigma2[mask]
        out_df_total[mask] = df_resid[mask]
        return out_sigma2_post, out_df_total

    s2 = np.clip(sigma2[mask], 1e-12, None)
    d = df_resid[mask]

    z = np.log(s2)
    e = z - special.digamma(d / 2.0) + np.log(d / 2.0)

    lo = np.nanpercentile(e, 100.0 * winsor_q[0])
    hi = np.nanpercentile(e, 100.0 * winsor_q[1])
    e_w = np.clip(e, lo, hi)

    emean = float(np.mean(e_w))
    # Var(z) = trigamma(d0/2) + trigamma(d/2)  ->  prior part only
    evar = float(np.var(e_w, ddof=1)) - float(np.mean(special.polygamma(1, d / 2.0)))

    if not np.isfinite(evar) or evar <= 0:
        # no detectable variance heterogeneity: d0 -> Inf, posterior = prior mean
        s0_sq = float(np.exp(emean))
        out_sigma2_post[mask] = s0_sq
        out_df_total[mask] = np.inf
        return out_sigma2_post, out_df_total

    d0 = _estimate_d0_from_varlog(evar, d0_floor=d0_floor, d0_ceil=d0_ceil)
    s0_sq = float(np.exp(emean + special.digamma(d0 / 2.0) - np.log(d0 / 2.0)))

    out_sigma2_post[mask] = (d0 * s0_sq + d * s2) / (d0 + d)
    out_df_total[mask] = d0 + d

    return out_sigma2_post, out_df_total


def _is_estimable_contrast(X: np.ndarray, c: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check whether contrast c is estimable in design matrix X.
    """
    if X.size == 0:
        return False

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        return False

    eps = np.finfo(float).eps
    sv_tol = max(X.shape) * s[0] * eps
    rank = int(np.sum(s > sv_tol))

    if rank == X.shape[1]:
        return True

    null_basis = Vt[rank:, :]
    if null_basis.size == 0:
        return True

    return np.all(np.abs(null_basis @ c) <= tol)


def lm_sitewise(y: np.ndarray,
                X: np.ndarray,
                coef_idx: int,
                w_obs: Optional[np.ndarray] = None,
                min_obs_over_coef: float = 1.0) -> Dict[str, Any]:
    """
    OLS/WLS per-site fit with explicit failure reasons.

    Stores only the diagonal element (X'X)^{-1}[coef, coef] needed for the
    standard error — full matrices are never retained (memory).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if w_obs is None:
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    else:
        w_obs = np.asarray(w_obs, dtype=float)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(w_obs) & (w_obs > 0)

    y = y[mask]
    X = X[mask]
    if w_obs is not None:
        w_obs = w_obs[mask]

    n_obs, n_coef = X.shape

    if n_obs == 0:
        return {"ok": False, "reason": "no_observations"}

    if w_obs is None:
        X_eff = X
        y_eff = y
    else:
        sqrtw = np.sqrt(w_obs)
        X_eff = X * sqrtw[:, None]
        y_eff = y * sqrtw

    rank = int(np.linalg.matrix_rank(X_eff))

    # Minimum effective sample size: need at least one residual df.
    min_required = max(rank + 1, int(np.ceil(min_obs_over_coef * rank)))
    if n_obs < min_required:
        return {
            "ok": False,
            "reason": "too_few_observations",
            "n_obs": n_obs,
            "rank": rank
        }

    c = np.zeros(n_coef, dtype=float)
    c[coef_idx] = 1.0

    if not _is_estimable_contrast(X_eff, c):
        return {
            "ok": False,
            "reason": "coef_not_estimable",
            "n_obs": n_obs,
            "rank": rank
        }

    XtX = X_eff.T @ X_eff
    XtX_inv = np.linalg.pinv(XtX)

    # Minimal-norm solution; for estimable contrasts the target coefficient is valid.
    beta = np.linalg.pinv(X_eff) @ y_eff

    resid = y_eff - X_eff @ beta
    df_resid = n_obs - rank
    if df_resid <= 0:
        return {
            "ok": False,
            "reason": "no_residual_df",
            "n_obs": n_obs,
            "rank": rank
        }

    sigma2 = float(np.dot(resid, resid) / df_resid)

    return {
        "ok": True,
        "reason": "ok",
        "beta": beta,
        "sigma2": sigma2,
        "XtX_cc": max(float(XtX_inv[coef_idx, coef_idx]), 0.0),
        "df_resid": float(df_resid),
        "rank": rank,
        "n_obs": n_obs
    }


def run_limma_py(expr: pd.DataFrame,
                 design: pd.DataFrame,
                 sample_weights: Optional[pd.DataFrame] = None,
                 coef_name: str = "Condition",
                 min_obs_over_coef: float = 1.0,
                 skip_eb: bool = False,
                 min_sites_eb: int = 30,
                 eb_d0_floor: float = 2.0,
                 eb_d0_ceil: float = 200.0) -> pd.DataFrame:
    """
    Limma-like site-wise linear model with global empirical Bayes variance moderation.

    Workflow:
      1. Fit all sites independently (WLS if sample_weights given).
      2. Estimate global variance prior from all residual variances.
      3. Apply empirical Bayes moderation.
      4. Moderated t-statistics.
      5. BH correction.

    EB moderation is skipped (ordinary t per site) when skip_eb=True or when
    fewer than min_sites_eb sites were successfully fitted: the moment
    estimator of (d0, s0^2) is unreliable on a handful of sites.
    """
    if coef_name not in design.columns:
        raise KeyError(f"coef_name='{coef_name}' not found in design.columns")

    # ---- align samples ----
    common_samples = [s for s in design.index if s in expr.columns]
    if len(common_samples) == 0:
        raise ValueError("No overlapping samples between expr and design")

    expr_al = expr.loc[:, common_samples].copy()
    design_al = design.loc[common_samples].copy()

    X_full = design_al.to_numpy(dtype=float)
    cols_full = design_al.columns
    coef_idx_global = cols_full.get_loc(coef_name)

    # ---- align weights ----
    if sample_weights is not None and not sample_weights.empty:
        w_al = sample_weights.reindex(index=expr_al.index, columns=common_samples)
    else:
        w_al = None

    # ---- first pass: collect beta, sigma2, df ----
    fit_results = []
    sigma2_all = []
    df_all = []

    for site, row in expr_al.iterrows():
        y = row.to_numpy(dtype=float)
        w = w_al.loc[site].to_numpy(dtype=float) if w_al is not None else None

        fit = lm_sitewise(y=y, X=X_full, coef_idx=coef_idx_global,
                          w_obs=w, min_obs_over_coef=min_obs_over_coef)

        if not fit["ok"]:
            fit_results.append({"site": site, "ok": False, "reason": fit["reason"]})
            continue

        fit_results.append({"site": site, "ok": True, "fit": fit})
        sigma2_all.append(fit["sigma2"])
        df_all.append(fit["df_resid"])

    # ---- global empirical Bayes ----
    sigma2_all = np.asarray(sigma2_all, dtype=float)
    df_all = np.asarray(df_all, dtype=float)

    if not skip_eb and len(sigma2_all) >= max(3, min_sites_eb):
        sigma2_post_all, df_total_all = empirical_bayes_global(
            sigma2_all, df_all, d0_floor=eb_d0_floor, d0_ceil=eb_d0_ceil)
    else:
        if not skip_eb and len(sigma2_all) >= 3:
            logger.info(
                f"Only {len(sigma2_all)} fitted sites (< min_sites_eb={min_sites_eb}); "
                "skipping EB moderation (ordinary t-statistics)."
            )
        sigma2_post_all = sigma2_all.copy()
        df_total_all = df_all.copy()

    # ---- second pass: moderated statistics ----
    records = []
    eb_index = 0

    for item in fit_results:
        site = item["site"]

        if not item["ok"]:
            records.append({
                "site": site, "logFC": np.nan, "t": np.nan, "P.Value": np.nan,
                "df": np.nan, "sigma2": np.nan, "sigma2_post": np.nan,
                "status": item["reason"], "n_obs": np.nan, "rank": np.nan
            })
            continue

        fit = item["fit"]
        beta = fit["beta"]
        sigma2 = fit["sigma2"]
        XtX_cc = fit["XtX_cc"]

        sigma2_post = sigma2_post_all[eb_index]
        df_total = df_total_all[eb_index]
        eb_index += 1

        beta_c = float(beta[coef_idx_global])
        se = np.sqrt(sigma2_post * XtX_cc)

        if not np.isfinite(se) or se <= 0:
            t_stat = np.nan
            pval = np.nan
        else:
            t_stat = beta_c / se
            pval = 2 * stats.t.sf(abs(t_stat), df_total)

        records.append({
            "site": site,
            "logFC": beta_c,
            "t": t_stat,
            "P.Value": pval,
            "df": df_total,
            "sigma2": sigma2,
            "sigma2_post": sigma2_post,
            "status": "ok",
            "n_obs": fit["n_obs"],
            "rank": fit["rank"]
        })

    res = pd.DataFrame.from_records(records).set_index("site")
    if res.empty:
        return pd.DataFrame()

    # ---- BH correction ----
    res["adj.P.Val"] = np.nan
    ok = np.isfinite(res["P.Value"])
    if ok.sum() > 0:
        res.loc[ok, "adj.P.Val"] = multipletests(res.loc[ok, "P.Value"], method="fdr_bh")[1]

    return res


def permutation_limma(expr_pw: pd.DataFrame, design: pd.DataFrame,
                      weights_df: Optional[pd.DataFrame] = None,
                      n_perm: int = 1000, alpha: float = 0.05,
                      logfc_thresh: float = 1.0,
                      exact_threshold: int = 5000,
                      seed: Optional[int] = None,
                      skip_eb: bool = False,
                      min_sites_eb: int = 30,
                      eb_d0_floor: float = 2.0,
                      eb_d0_ceil: float = 200.0,
                      coef_name: str = "Condition") -> Optional[Dict[str, Any]]:
    """
    Batch-stratified label-permutation validation of the testing procedure.

    Under the global null (no group effect anywhere) sample labels are
    exchangeable only WITHIN a TMT batch, so Condition labels are permuted
    within batches. The null distribution is generated by the SAME testing
    procedure (WLS + EB moderation + BH) as the observed analysis.

    Number of distinct labelings = prod over batches of C(n_b, n1_b).
    If it does not exceed `exact_threshold`, ALL labelings are enumerated
    exhaustively (no Monte-Carlo error; the observed labeling is among them,
    so the permutation p-value is a plain proportion). Otherwise `n_perm`
    random labelings are drawn and the +1 correction is applied
    (Phipson & Smyth 2010: permutation p-values should never be zero).

    Returns a dict with:
      obs_hits          - observed number of hits (adj.P.Val < alpha &
                          |logFC| >= logfc_thresh)
      perm_counts       - null distribution of the hit count (np.array)
      perm_pval         - P(N_null >= N_obs)
      empirical_fdr     - mean(N_null) / max(N_obs, 1), capped at 1
      plus summary statistics of the null distribution.
    """
    if coef_name not in design.columns:
        raise KeyError(f"coef_name='{coef_name}' not found in design.columns")

    lm_kwargs = dict(skip_eb=skip_eb, min_sites_eb=min_sites_eb,
                     eb_d0_floor=eb_d0_floor, eb_d0_ceil=eb_d0_ceil)

    def n_hits(res) -> int:
        if res is None or res.empty or 'adj.P.Val' not in res.columns:
            return 0
        return int(((res['adj.P.Val'] < alpha) &
                    (res['logFC'].abs() >= logfc_thresh)).sum())

    res_obs = run_limma_py(expr_pw, design, sample_weights=weights_df,
                           coef_name=coef_name, **lm_kwargs)
    obs_hits = n_hits(res_obs)

    # --- batch structure of the design ---
    batch_cols = [c for c in design.columns if str(c).startswith("Batch")]
    if batch_cols:
        batch_id = design[batch_cols].apply(
            lambda r: r.idxmax() if r.sum() > 0 else "Batch_first", axis=1)
    else:
        batch_id = pd.Series("all", index=design.index)

    cond = design[coef_name].to_numpy(dtype=float)
    batch_arr = batch_id.to_numpy()

    per_batch = []        # (sample indices, n_ones) per batch
    n_distinct = 1
    for b in pd.unique(batch_id):
        idx = np.where(batch_arr == b)[0]
        n_b = len(idx)
        n1 = int(round(cond[idx].sum()))
        if n1 == 0 or n1 == n_b:
            continue      # no permutable group structure in this batch
        per_batch.append((idx, n1))
        n_distinct *= comb(n_b, n1)

    if len(per_batch) == 0:
        logger.warning("permutation_limma: no batch contains both groups; "
                       "nothing to permute.")
        return None

    exact = n_distinct <= exact_threshold
    rng = np.random.default_rng(seed)

    def run_one(new_cond: np.ndarray) -> int:
        perm_design = design.copy()
        perm_design[coef_name] = new_cond
        res = run_limma_py(expr_pw, perm_design, sample_weights=weights_df,
                           coef_name=coef_name, **lm_kwargs)
        return n_hits(res)

    counts = []
    if exact:
        combo_lists = [list(itertools.combinations(idx, n1)) for idx, n1 in per_batch]
        for combo in itertools.product(*combo_lists):
            new_cond = np.zeros_like(cond)
            for chosen in combo:
                new_cond[list(chosen)] = 1.0
            counts.append(run_one(new_cond))
        n_eval = n_distinct
    else:
        for _ in range(n_perm):
            new_cond = cond.copy()
            for idx, _n1 in per_batch:
                new_cond[idx] = cond[rng.permutation(idx)]
            counts.append(run_one(new_cond))
        n_eval = n_perm

    counts = np.asarray(counts, dtype=int)
    if counts.size == 0:
        return None

    ge = int(np.sum(counts >= obs_hits))
    if exact:
        perm_pval = ge / n_eval
    else:
        perm_pval = (1 + ge) / (1 + n_eval)

    perm_mean = float(np.mean(counts))
    logger.info(f"permutation_limma: obs_hits={obs_hits}, "
                f"null mean={perm_mean:.1f} (n={n_eval}, exact={exact}, "
                f"distinct={n_distinct}), perm_pval={perm_pval:.4f}")

    return {
        'obs_hits': obs_hits,
        'n_perm': int(n_eval),
        'exact': bool(exact),
        'n_distinct_labelings': int(n_distinct),
        'perm_mean': perm_mean,
        'perm_median': float(np.median(counts)),
        'perm_p95': float(np.percentile(counts, 95)),
        'perm_max': int(np.max(counts)),
        'perm_pval': float(perm_pval),
        'empirical_fdr': float(min(1.0, perm_mean / max(obs_hits, 1))),
        'alpha': alpha,
        'logfc_thresh': logfc_thresh,
        'perm_counts': counts,
    }


def compute_batch_global_prior(batch_df: pd.DataFrame, channel_cols: list = None):
    """
    Robust global prior variance per channel for a batch.
    """
    if not channel_cols:
        return None

    # --- numeric matrix ---
    X = batch_df[channel_cols].to_numpy(dtype=float)
    X[X == 0] = np.nan  # treat zeros as missing

    # --- variance per channel over all PSMs of the batch ---
    counts_local = np.sum(~np.isnan(X), axis=0)
    valid = counts_local >= 2

    var_per_channel = np.full(X.shape[1], np.nan)

    if np.any(valid):
        Xv = X[:, valid]

        med = np.nanmedian(Xv, axis=0)
        mad = np.nanmedian(np.abs(Xv - med), axis=0)
        std = np.nanstd(Xv, axis=0, ddof=1)

        # adaptive floor for MAD
        mad_floor = np.nanpercentile(mad[np.isfinite(mad)], 5) if np.any(np.isfinite(mad)) else 1e-6
        mad = np.maximum(mad, mad_floor)

        var_per_channel[valid] = 0.5 * (std ** 2) + 0.5 * ((1.4826 * mad) ** 2)

    # --- fallback: median over channels ---
    fallback = np.nanmedian(var_per_channel) if np.any(np.isfinite(var_per_channel)) else 1.0
    var_prior = np.where(np.isfinite(var_per_channel), var_per_channel, fallback)

    return var_prior


def statistics(df: pd.DataFrame,
               min_group_for_stats: int = 1,
               min_batches: int = 1,
               method: str = 'aggregate',
               type_experiment: str = 'whole proteome',
               skip_eb: bool = False,
               min_sites_eb: int = 30,
               icc_mode: str = 'estimate',
               fixed_icc: float = 0.30,
               huber_c: float = 1.345,
               huber_iters: int = 3,
               var_floor_pct: float = 10.0,
               eb_d0_floor: float = 2.0,
               eb_d0_ceil: float = 200.0,
               run_permutation: bool = False,
               n_perm: int = 1000,
               perm_alpha: float = 0.05,
               perm_logfc_thresh: float = 1.0,
               perm_exact_threshold: int = 5000,
               perm_seed: Optional[int] = None):
    """
    Main statistics pipeline.

    ALWAYS returns an 8-tuple:
        (final_df, expr_all, expr_corrected, df_site, weights_df, design,
         noagg, perm_df)

    Contrasts are built dynamically from all TMT_group* columns
    (works for 2, 3 or more groups).

    Aggregation / EB hyperparameters are forwarded to
    bayesian_site_aggregation() and run_limma_py().

    If run_permutation=True, each contrast is additionally validated by
    batch-stratified label permutation (permutation_limma); the per-contrast
    summaries are returned in perm_df.
    """
    EMPTY8 = (pd.DataFrame(),) * 8

    if df is None or df.empty:
        logger.warning("Input DataFrame is empty. Skipping statistics.")
        return EMPTY8

    # --- dynamic group columns and pairwise contrasts ---
    group_cols = sorted(
        [c for c in df.columns if re.fullmatch(r'TMT_group\d+', str(c))],
        key=lambda s: int(str(s).replace('TMT_group', ''))
    )
    if len(group_cols) < 2:
        logger.error("Need at least two TMT_group* columns to build contrasts.")
        return EMPTY8
    pairwise = list(itertools.combinations(group_cols, 2))

    # batch -> groups mapping (elements normalized to lists of stripped strings)
    df['batch'] = df['batch'].astype(int)
    batch_groups = df.drop_duplicates("batch").set_index("batch")[group_cols]
    batch_groups = batch_groups.apply(lambda col: col.map(
        lambda xs: [str(x).strip() for x in (xs if isinstance(xs, list) else [xs])]
    ))

    channel_cols = [c for c in df.columns if c.endswith('_norm')]
    if not channel_cols:
        logger.error("No *_norm channel columns found.")
        return EMPTY8

    # drop rows with all channel columns NA
    df = df[~df[channel_cols].isna().all(axis=1)].copy()

    # Rename to canonical names; if the target name is already taken, the
    # source column (modified_peptide_x/spectrum_y/id_prot) wins, otherwise
    # we would get duplicated names and a "Grouper not 1-dimensional" error.
    for src, dst in {'id_prot': 'protein', 'spectrum_y': 'scannr',
                     'modified_peptide_x': 'peptide'}.items():
        if src in df.columns:
            if dst in df.columns and dst != src:
                df = df.drop(columns=[dst])
            df = df.rename(columns={src: dst})

    if {'protein', 'position_in_protein', 'scannr', 'peptide'}.issubset(df.columns):
        dedup_subset = ['protein', 'position_in_protein', 'scannr', 'peptide']
        # Sage scan numbers restart per file/batch; without file_name/batch the
        # deduplication below would discard legitimate PSMs from other plexes.
        dedup_subset += [c for c in ['file_name', 'batch'] if c in df.columns]
        df = df.drop_duplicates(subset=dedup_subset)
    else:
        df = df.drop_duplicates()

    def batch_prior(sub: pd.DataFrame) -> dict:
        prior = {}
        for b in sub['batch'].unique():
            prior[int(b)] = compute_batch_global_prior(
                sub[sub['batch'] == b], channel_cols=channel_cols)
        return prior

    # outputs initialized so every return path is well-defined
    df_site = pd.DataFrame()
    expr = pd.DataFrame()
    weights_df = pd.DataFrame()
    protein_abundance = None
    protein_var = None
    design = pd.DataFrame()
    noagg = pd.DataFrame()

    pos_str = _pos_to_str(df["position_in_protein"])

    # ==================================================================
    if method == 'aggregate':
        if type_experiment == 'phospho enrichment':
            psm_stats = (df.groupby(["protein", "position_in_protein"])
                         .agg(n_psm=("scannr", "count"), n_batches=("batch", pd.Series.nunique))
                         .reset_index())
            psm_stats["site"] = psm_stats["protein"].astype(str) + "_" + _pos_to_str(psm_stats["position_in_protein"])
            merge_cols = ['protein', 'position_in_protein']
        else:
            psm_stats = (df.groupby(['Modification', "protein", "position_in_protein"])
                         .agg(n_psm=("scannr", "count"), n_batches=("batch", pd.Series.nunique))
                         .reset_index())
            psm_stats["site"] = (psm_stats["Modification"].astype(str) + "_" +
                                 psm_stats["protein"].astype(str) + "_" +
                                 _pos_to_str(psm_stats["position_in_protein"]))
            merge_cols = ['protein', 'position_in_protein', 'Modification']

        df = df.merge(psm_stats[["n_psm", "n_batches"] + merge_cols].drop_duplicates(),
                      on=merge_cols, how="left")
        df = df[df["n_batches"] >= min_batches]
        df = df[df["n_psm"] >= min_group_for_stats]
        if df.empty:
            logger.warning("No rows left after n_psm/n_batches filtering.")
            return EMPTY8

        if type_experiment == 'phospho enrichment':
            global_prior = batch_prior(df)
            df_site = (df.groupby(["protein", "position_in_protein", "batch"])
                       .apply(lambda g: bayesian_site_aggregation(
                           g, global_prior, method=method, type_experiment=type_experiment,
                           channel_cols=channel_cols, icc_mode=icc_mode,
                           fixed_icc=fixed_icc, huber_c=huber_c,
                           huber_iters=huber_iters, var_floor_pct=var_floor_pct))
                       .reset_index())
            long_df = df_site.melt(id_vars=["protein", "position_in_protein", "batch"],
                                   var_name="channel", value_name="value")
            long_df["site"] = long_df["protein"].astype(str) + "_" + _pos_to_str(long_df["position_in_protein"])
            long_df["sample"] = (long_df["channel"].str.split("_").str[1] + "_" +
                                 long_df["batch"].astype(int).astype(str))
            stoich_df_long = long_df[~long_df['channel'].str.endswith(('_var', '_prec'))]
            prec_long = long_df[long_df['channel'].str.endswith('_prec')]

            expr = stoich_df_long.pivot_table(index="site", columns="sample",
                                              values="value", aggfunc="mean")
            weights_df = prec_long.pivot_table(index="site", columns="sample",
                                               values="value", aggfunc="mean").reindex(columns=expr.columns)

        else:  # whole proteome
            # ---- modified PSMs -> site level ----
            mod_df = df[df['Modification'] != 'reference'].copy()
            if mod_df.empty:
                logger.warning("No modified PSMs after filtering.")
                return EMPTY8
            global_prior = batch_prior(mod_df)

            df_site = (mod_df.groupby(['Modification', "protein", "position_in_protein", "batch"])
                       .apply(lambda g: bayesian_site_aggregation(
                           g, global_prior, method=method, type_experiment=type_experiment,
                           channel_cols=channel_cols, icc_mode=icc_mode,
                           fixed_icc=fixed_icc, huber_c=huber_c,
                           huber_iters=huber_iters, var_floor_pct=var_floor_pct))
                       .reset_index())

            long_df = df_site.melt(id_vars=['Modification', "protein", "position_in_protein", "batch"],
                                   var_name="channel", value_name="value")
            long_df["site"] = (long_df["Modification"].astype(str) + "_" +
                               long_df["protein"].astype(str) + "_" +
                               _pos_to_str(long_df["position_in_protein"]))
            long_df["sample"] = (long_df["channel"].str.split("_").str[1] + "_" +
                                 long_df["batch"].astype(int).astype(str))

            stoich_df_long = long_df[~long_df['channel'].str.endswith(('_var', '_prec'))]
            prec_long = long_df[long_df['channel'].str.endswith('_prec')]

            expr = stoich_df_long.pivot_table(index='site', columns='sample',
                                              values='value', aggfunc='first')
            weights_df = prec_long.pivot_table(index='site', columns='sample',
                                               values='value', aggfunc='first').reindex(columns=expr.columns)

            # ---- reference PSMs -> protein abundance ----
            unmod_df = df[df['Modification'] == 'reference'].copy()
            if not unmod_df.empty:
                global_prior_un = batch_prior(unmod_df)
                df_site_un = (unmod_df.groupby(["batch", 'protein'])
                              .apply(lambda g: bayesian_site_aggregation(
                                  g, global_prior_un, method='protein',
                                  type_experiment=type_experiment,
                                  channel_cols=channel_cols, icc_mode=icc_mode,
                           fixed_icc=fixed_icc, huber_c=huber_c,
                           huber_iters=huber_iters, var_floor_pct=var_floor_pct))
                              .reset_index())

                long_un = df_site_un.melt(id_vars=["protein", "batch"],
                                          var_name="channel", value_name="value")
                long_un["sample"] = (long_un["channel"].str.split("_").str[1] + "_" +
                                     long_un["batch"].astype(int).astype(str))
                protein_abundance_long = long_un[~long_un['channel'].str.endswith(('_var', '_prec'))]
                protein_abundance = protein_abundance_long.pivot(index='protein',
                                                                 columns='sample',
                                                                 values='value')
                # per-channel aggregation variance of the protein estimate
                # (needed for the variance of the corrected site values)
                protein_var_long = long_un[long_un['channel'].str.endswith('_var')]
                protein_var = protein_var_long.pivot(index='protein',
                                                     columns='sample',
                                                     values='value')
            else:
                logger.warning("No reference PSMs: protein-abundance correction disabled.")

    # ==================================================================
    elif method == 'median':
        if type_experiment == 'phospho enrichment':
            df = df.drop_duplicates(subset=["peptide", 'isotope_error', 'charge',
                                            'file_name', 'scannr']).reset_index(drop=True)
            psm_stats = (df.groupby(["peptide", 'protein'])
                         .agg(n_psm=("scannr", "count"), n_batches=("batch", pd.Series.nunique))
                         .reset_index())
            psm_stats['site'] = psm_stats["protein"].astype(str) + "_" + psm_stats["peptide"].astype(str)
            df = df.merge(psm_stats[["n_psm", "n_batches", "peptide", "protein",'site']],
                          on=["peptide", "protein"], how="left")
        else:
            df = df.drop_duplicates(subset=['peptide', 'charge', 'protein',
                                            'scannr', 'file_name']).reset_index(drop=True)
            psm_stats = (df[df['Modification'] != 'reference']
                         .groupby(["peptide", 'protein'])
                         .agg(n_psm=("scannr", "count"), n_batches=("batch", pd.Series.nunique))
                         .reset_index())
            psm_stats['site'] = psm_stats["protein"].astype(str) + "_" + psm_stats["peptide"].astype(str)
            df = df.merge(psm_stats[["n_psm", "n_batches", "peptide", 'protein', 'site']],
                          on=["peptide", 'protein'], how="left")

        # The n_psm/n_batches thresholds apply to modified (peptide-level)
        # rows only: reference rows must survive so that protein-abundance
        # correction remains possible in the whole-proteome branch. Their
        # n_psm/n_batches are NaN because psm_stats was computed from the
        # modified subset.
        keep = (df['Modification'] == 'reference') | (
            (df["n_batches"] >= min_batches) & (df["n_psm"] >= min_group_for_stats))
        df = df[keep]
        if df.empty:
            logger.warning("No rows left after n_psm/n_batches filtering.")
            return EMPTY8

        if type_experiment == 'phospho enrichment':
            global_prior = batch_prior(df)
            df_site = (df.groupby(["batch", "protein", "peptide"])
                       .apply(lambda g: bayesian_site_aggregation(
                           g, global_prior, method=method, type_experiment=type_experiment,
                           channel_cols=channel_cols, icc_mode=icc_mode,
                           fixed_icc=fixed_icc, huber_c=huber_c,
                           huber_iters=huber_iters, var_floor_pct=var_floor_pct))
                       .reset_index())

            long_df = df_site.melt(id_vars=["protein", "batch", 'peptide'],
                                   var_name="channel", value_name="value")
            long_df["site"] = long_df["protein"].astype(str) + "_" + long_df["peptide"].astype(str)
            long_df["sample"] = (long_df["channel"].str.split("_").str[1] + "_" +
                                 long_df["batch"].astype(int).astype(str))
            stoich_df_long = long_df[~long_df['channel'].str.endswith(('_var', '_prec'))]
            prec_long = long_df[long_df['channel'].str.endswith('_prec')]

            expr = stoich_df_long.pivot_table(index='site', columns='sample',
                                              values='value', aggfunc='first')
            weights_df = prec_long.pivot_table(index='site', columns='sample',
                                               values='value', aggfunc='first').reindex(columns=expr.columns)

        else:  # whole proteome
            mod_df = df[df['Modification'] != 'reference'].copy()
            if mod_df.empty:
                logger.warning("No modified PSMs after filtering.")
                return EMPTY8
            global_prior = batch_prior(mod_df)

            df_site = (mod_df.groupby(["batch", "peptide", 'protein'])
                       .apply(lambda g: bayesian_site_aggregation(
                           g, global_prior, method=method, type_experiment=type_experiment,
                           channel_cols=channel_cols, icc_mode=icc_mode,
                           fixed_icc=fixed_icc, huber_c=huber_c,
                           huber_iters=huber_iters, var_floor_pct=var_floor_pct))
                       .reset_index())

            long_df = df_site.melt(id_vars=["protein", "batch", 'peptide'],
                                   var_name="channel", value_name="value")
            long_df["site"] = long_df["protein"].astype(str) + "_" + long_df["peptide"].astype(str)
            long_df["sample"] = (long_df["channel"].str.split("_").str[1] + "_" +
                                 long_df["batch"].astype(int).astype(str))
            stoich_df_long = long_df[~long_df['channel'].str.endswith(('_var', '_prec'))]
            prec_long = long_df[long_df['channel'].str.endswith('_prec')]

            expr = stoich_df_long.pivot_table(index='site', columns='sample',
                                              values='value', aggfunc='first')
            weights_df = prec_long.pivot_table(index='site', columns='sample',
                                               values='value', aggfunc='first').reindex(columns=expr.columns)

            # ---- reference PSMs -> protein abundance ----
            unmod_df = df[df['Modification'] == 'reference'].copy()
            if not unmod_df.empty:
                global_prior_un = batch_prior(unmod_df)
                df_site_un = (unmod_df.groupby(["batch", 'protein'])
                              .apply(lambda g: bayesian_site_aggregation(
                                  g, global_prior_un, method='protein',
                                  type_experiment=type_experiment,
                                  channel_cols=channel_cols, icc_mode=icc_mode,
                           fixed_icc=fixed_icc, huber_c=huber_c,
                           huber_iters=huber_iters, var_floor_pct=var_floor_pct))
                              .reset_index())
                long_un = df_site_un.melt(id_vars=["protein", "batch"],
                                          var_name="channel", value_name="value")
                long_un["sample"] = (long_un["channel"].str.split("_").str[1] + "_" +
                                     long_un["batch"].astype(int).astype(str))
                protein_abundance_long = long_un[~long_un['channel'].str.endswith(('_var', '_prec'))]
                protein_abundance = protein_abundance_long.pivot(index='protein',
                                                                 columns='sample',
                                                                 values='value')
                # per-channel aggregation variance of the protein estimate
                protein_var_long = long_un[long_un['channel'].str.endswith('_var')]
                protein_var = protein_var_long.pivot(index='protein',
                                                     columns='sample',
                                                     values='value')
            else:
                logger.warning("No reference PSMs: protein-abundance correction disabled.")

    else:
        raise ValueError("Not valid method")

    # ==================================================================
    # site x sample matrices, protein-abundance correction
    # ==================================================================
    if expr.empty:
        logger.warning("Expression matrix is empty after aggregation.")
        return EMPTY8

    expr = expr.merge(psm_stats.set_index("site")[["n_psm", "n_batches"]],
                      left_index=True, right_index=True, how="left")
    expr_all = expr.copy()
    expr_data = expr.drop(columns=["n_psm", "n_batches"])

    if type_experiment == 'whole proteome' and protein_abundance is not None:
        # site = "Modification_protein_position" (aggregate) or "protein_peptide" (median)
        prot_idx = 1 if method == 'aggregate' else 0
        valid_sites = expr_data.index.to_series().map(
            lambda s: s.split('_')[prot_idx] in protein_abundance.index)
        expr_data = expr_data[valid_sites.to_numpy()]

        expr_corrected = expr_data.copy()
        for site in expr_corrected.index:
            prot = site.split('_')[prot_idx]
            expr_corrected.loc[site] = expr_data.loc[site] - protein_abundance.loc[prot, expr_data.columns]

        # The corrected value is a DIFFERENCE (site - protein), so its variance
        # is the sum of the two aggregation variances. Propagate this into the
        # WLS weights: prec_corr = 1 / (var_site + var_protein). Without this
        # step the standard errors after protein correction are too optimistic.
        if protein_var is not None and not weights_df.empty:
            site_prots = expr_corrected.index.to_series().map(
                lambda s: s.split('_')[prot_idx])
            var_prot = protein_var.reindex(index=site_prots.to_numpy(),
                                           columns=expr_corrected.columns)
            var_prot.index = expr_corrected.index
            with np.errstate(divide='ignore', invalid='ignore'):
                var_site = 1.0 / weights_df.reindex(index=expr_corrected.index,
                                                    columns=expr_corrected.columns)
                var_corr = var_site + var_prot
                weights_corr = 1.0 / var_corr
            weights_df.loc[expr_corrected.index, expr_corrected.columns] = \
                weights_corr.to_numpy(dtype=float)
    else:
        # phospho enrichment: no protein correction by design
        expr_corrected = expr_data.copy()

    # build sample annotation
    sample_df = expr_corrected.columns.to_series().str.rsplit("_", n=1, expand=True).dropna()
    sample_df.columns = ["s", "batch"]
    sample_df["batch"] = sample_df["batch"].astype(int)
    sample_df.index = expr_corrected.columns

    # noagg (diagnostic matrix without aggregation)
    try:
        noagg = build_expression_noagg(df, method=method,
                                       type_experiment=type_experiment,
                                       channel_cols=channel_cols)
        if not noagg.empty:
            noagg = noagg.reindex(index=expr_data.index)
    except Exception as e:
        logger.warning(f"build_expression_noagg failed: {e}")
        noagg = pd.DataFrame()

    # ==================================================================
    # pairwise contrasts
    # ==================================================================
    final_df_list = []
    perm_records = []

    for gA, gB in pairwise:
        def map_condition(row, gA=gA, gB=gB):
            b = int(row["batch"])
            s = row["s"]
            if s in batch_groups.loc[b, gB]:
                return 1
            if s in batch_groups.loc[b, gA]:
                return 0
            return np.nan

        condition = sample_df.apply(map_condition, axis=1)
        design = pd.DataFrame({"Intercept": 1, "Condition": condition,
                               "Batch": sample_df["batch"]}).dropna()
        design = pd.get_dummies(design, columns=["Batch"], drop_first=True)

        # per-group minimum sample requirement (NOT total-sample count)
        nA = int((design["Condition"] == 0).sum())
        nB = int((design["Condition"] == 1).sum())
        logger.info(f"=== Contrast {gB} vs {gA}: n({gA})={nA}, n({gB})={nB}, "
                    f"design={design.shape} ===")
        if min(nA, nB) < min_group_for_stats:
            logger.warning(f"Contrast {gB} vs {gA}: fewer than min_group_for_stats="
                           f"{min_group_for_stats} samples in a group - skipped.")
            continue

        expr_pw = expr_corrected.reindex(columns=design.index)
        weights_gr = weights_df.reindex(columns=design.index) if not weights_df.empty else None

        res = run_limma_py(expr_pw, design, sample_weights=weights_gr,
                           skip_eb=skip_eb, min_sites_eb=min_sites_eb,
                           eb_d0_floor=eb_d0_floor, eb_d0_ceil=eb_d0_ceil)
        res["contrast"] = f"{gB}_vs_{gA}"

        # diagnostic Welch t-test (unweighted, no batch adjustment!)
        grpA_cols = design.index[design["Condition"] == 0]
        grpB_cols = design.index[design["Condition"] == 1]

        if len(grpA_cols) >= 1 and len(grpB_cols) >= 1:
            A = expr_pw[grpA_cols].to_numpy(dtype=float)
            B = expr_pw[grpB_cols].to_numpy(dtype=float)
            pvals = []
            for i in range(A.shape[0]):
                xA = A[i, :]
                xB = B[i, :]
                xA = xA[~np.isnan(xA)]
                xB = xB[~np.isnan(xB)]
                if xA.size < 2 or xB.size < 2:
                    pvals.append(np.nan)
                else:
                    _, p = ttest_ind(xA, xB, equal_var=False)
                    pvals.append(p)
            pvals_series = pd.Series(pvals, index=expr_pw.index.to_list()).reindex(res.index.to_list())
            res['pval_ttest'] = pvals_series.values
        else:
            res['pval_ttest'] = np.nan

        final_df_list.append(res.reset_index())

        # --- optional batch-stratified permutation validation ---
        if run_permutation:
            perm_res = permutation_limma(
                expr_pw, design, weights_gr,
                n_perm=n_perm, alpha=perm_alpha, logfc_thresh=perm_logfc_thresh,
                exact_threshold=perm_exact_threshold, seed=perm_seed,
                skip_eb=skip_eb, min_sites_eb=min_sites_eb,
                eb_d0_floor=eb_d0_floor, eb_d0_ceil=eb_d0_ceil)
            if perm_res is not None:
                rec = {k: v for k, v in perm_res.items() if k != 'perm_counts'}
                rec['contrast'] = f"{gB}_vs_{gA}"
                rec['perm_counts'] = ';'.join(map(str, perm_res['perm_counts'].tolist()))
                perm_records.append(rec)

    perm_df = pd.DataFrame(perm_records) if perm_records else pd.DataFrame()

    if not final_df_list:
        logger.warning("No contrast could be computed.")
        return pd.DataFrame(), expr_all, expr_corrected, df_site, weights_df, design, noagg, perm_df

    final_df = pd.concat(final_df_list, ignore_index=True)

    if method == 'median':
        if type_experiment == 'phospho enrichment':
            try:
                final_df = final_df.merge(
                    df[["site", "n_psm", "n_batches", "peptide",'peptide_clean','protein']].drop_duplicates(),
                    left_on='site', right_on='site', how="left")
                final_df = final_df.rename(columns={'protein':'id_prot'})
            except:
                print(final_df.head())
        else:
            final_df = final_df.merge(psm_stats[["site", "n_psm", "n_batches"]],
                                      on="site", how="left")
    elif method == 'aggregate':
        final_df = final_df.merge(psm_stats[["site", "n_psm", "n_batches"]],
                                  on="site", how="left")

    return final_df, expr_all, expr_corrected, df_site, weights_df, design, noagg, perm_df
