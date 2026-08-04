import numpy as np
import pandas as pd
from statistics import median, mean, stdev
import glob
from pyteomics import fasta
from scipy import stats
from scipy import special
import re
import ast
import logging
from scipy.stats import rankdata, ttest_ind, norm
from statsmodels.robust.scale import mad
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import warnings
from typing import List, Optional, Literal
import numpy as np
import pandas as pd
from statistics import median, mean, stdev
from sklearn.mixture import GaussianMixture
import warnings
from typing import List, Optional, Literal
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)



def bayesian_site_aggregation(group, global_batch, method = 'aggregate', type_experiment = 'phospho enrichment', eps=1e-8, 
                              k_shrink=10, weight_clip=5.0):

    channel_cols = [c for c in group.columns if c.endswith("_norm")]
    batch = list(group['batch'].unique())[0]

    if not channel_cols:
        return pd.Series(dtype=float)
    if method=='aggregate':
        if type_experiment == 'phospho enrichment':
            psm_types = (
                group[["peptide", "charge", "isotope_error"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            psm_types = (
                group[["peptide", "charge"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
    elif method=='median':
        if type_experiment == 'phospho enrichment':
            psm_types = (
                group[["charge", "isotope_error"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            psm_types = (
                group[["charge"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
    else:
        print('Not valid method')

    n_types = len(psm_types)
    n_channels = len(channel_cols)

    means = np.full((n_types, n_channels), np.nan)
    vars_ = np.full((n_types, n_channels), np.nan)
    counts = np.zeros((n_types, n_channels))

    for i, row in psm_types.iterrows():
        if method=='aggregate':
            if type_experiment == 'phospho enrichment':
                psm_df = group[
                    (group["peptide"] == row["peptide"]) &
                    (group["charge"] == row["charge"]) &
                    (group["isotope_error"] == row["isotope_error"])
                ]
            else:
                psm_df = group[
                    (group["peptide"] == row["peptide"]) &
                    (group["charge"] == row["charge"])
                ]
        elif method=='median':
            if type_experiment == 'phospho enrichment':
                psm_df = group[
                    (group["charge"] == row["charge"]) &
                    (group["isotope_error"] == row["isotope_error"])
                ]
            else:
                psm_df = group[
                    (group["charge"] == row["charge"])
                ]

        if psm_df.empty:
            continue
            
        X = psm_df[channel_cols].replace(0, np.nan).to_numpy(float)
        counts[i, :] = np.isfinite(X).sum(axis=0)
        means[i, :] = np.nanmedian(X, axis=0)
        if X.shape[0] < 2:
            vars_[i, :] = global_batch[batch]
        else:
            resid = X - means[i, :]
            mad = np.nanmedian(np.abs(resid), axis=0)
            sigma = 1.4826 * mad
            var_psm = sigma ** 2
            vars_[i, :] = var_psm

    var_prior = np.nanmedian(vars_, axis=0)

    global_prior = np.nanmedian(var_prior[np.isfinite(var_prior)])
    if not np.isfinite(global_prior):
        global_prior = 1.0

    var_prior = np.where(np.isfinite(var_prior),
                         var_prior, global_prior)

    n_eff = counts
    w = n_eff / (n_eff + k_shrink)

    var_shrinked = np.where(
        np.isnan(vars_),
        var_prior[None, :],
        w * vars_ + (1 - w) * var_prior[None, :]
    )

    floor = np.nanpercentile(var_shrinked[np.isfinite(var_shrinked)], 10)
    if not np.isfinite(floor):
        floor = 1e-4

    var_shrinked = np.clip(var_shrinked, floor, None)

    weights = (n_eff / var_shrinked) * (2.0 / np.pi)
    weights[n_eff == 0] = 0

    med_w = np.nanmedian(weights[np.isfinite(weights)])
    if np.isfinite(med_w) and med_w > 0:
        weights = np.minimum(weights, med_w * weight_clip)

    num = np.nansum(means * weights, axis=0)
    den = np.nansum(weights, axis=0)

    out_vals = np.full(n_channels, np.nan)
    valid = den > 0
    out_vals[valid] = num[valid] / den[valid]

    out = pd.Series(out_vals, index=channel_cols)
    out_w = pd.Series(den,
                      index=[c.replace('_norm', '_weight')
                             for c in channel_cols])

    return pd.concat([out, out_w])



def build_expression_noagg(df, method="aggregate", type_experiment = 'phospho enrichment'):
    """
    Строит site × sample матрицу без bayesian aggregation.
    """
    channel_cols = [c for c in df.columns if c.endswith("_norm")]
    df = df.copy()
    if method == "aggregate":
        if type_experiment == 'phospho enrichment':
            df["site"] = (
                df["protein"].astype(str)
                + "_"
                + df["position_in_protein"].astype(int).astype(str)
            )
        else:
            df["site"] = (df["Modification"].astype(str)
                + "_" +
                df["protein"].astype(str)
                + "_"
                + df["position_in_protein"].astype(int).astype(str)
            )

    elif method == "median":
        df["site"] = df["peptide"].astype(str)

    else:
        raise ValueError("invalid method")
        
    group_cols = ["site", "batch"]
    df_med = (
        df.groupby(group_cols)[channel_cols]
          .median()
          .reset_index()
    )

    long_df = df_med.melt(
        id_vars=group_cols,
        var_name="channel",
        value_name="value"
    )

    long_df["sample"] = (
        long_df["channel"].str.split("_").str[1]
        + "_"
        + long_df["batch"].astype(str)
    )

    expr_noagg = long_df.pivot_table(
        index="site",
        columns="sample",
        values="value",
        aggfunc="mean"
    )

    return expr_noagg


def empirical_bayes_global(sigma2_in, df_resid_in, d0_floor=2.0, d0_ceil=200.0, winsor_q=(0.05, 0.95)):
    """
    Robust Limma-style EB shrinkage.
    Inputs can contain NaN; returns arrays aligned with inputs (NaN for masked entries).
    winsor_q: tuple of lower/upper quantiles for winsorization of log(sigma2).
    """
    sigma2 = np.asarray(sigma2_in, float).copy()
    df_resid = np.asarray(df_resid_in, float).copy()

    out_sigma2_post = np.full_like(sigma2, np.nan, dtype=float)
    out_df_total = np.full_like(df_resid, np.nan, dtype=float)

    mask = np.isfinite(sigma2) & np.isfinite(df_resid) & (df_resid > 0)
    if mask.sum() < 3:
        return out_sigma2_post, out_df_total

    sigma2_mask = sigma2[mask]
    df_mask = df_resid[mask]

    sigma2_mask = np.clip(sigma2_mask, 1e-12, None)
    log_s2 = np.log(sigma2_mask)

    lo = np.nanpercentile(log_s2, 100.0 * winsor_q[0])
    hi = np.nanpercentile(log_s2, 100.0 * winsor_q[1])
    log_s2_w = np.clip(log_s2, lo, hi)

    var_log = log_s2_w.var(ddof=1)
    if not np.isfinite(var_log) or var_log <= 1e-12:
        # no variability => minimal shrinkage (use large d0 to dominate)
        d0 = d0_ceil
        mean_log = np.nanmean(log_s2_w)
    else:
        mean_log = np.nanmean(log_s2_w)
        d0 = 2.0 / var_log
        d0 = np.clip(d0, d0_floor, d0_ceil)

    s0_sq = np.exp(mean_log - special.digamma(d0 / 2.0) + np.log(d0 / 2.0))

    sigma2_post_mask = (d0 * s0_sq + df_mask * sigma2_mask) / (d0 + df_mask)
    df_total_mask = d0 + df_mask

    out_sigma2_post[mask] = sigma2_post_mask
    out_df_total[mask] = df_total_mask

    return out_sigma2_post, out_df_total


def lm_sitewise(y, X, w_obs=None, min_obs_over_coef=1):
    """
    Ordinary least squares per site. Optionally weighted (WLS) if w_obs provided.
    y: (n_samples,)
    X: (n_samples, n_coef)
    w_obs: (n_samples,) weights (precision: larger -> more weight). Can contain NaN; those rows will be removed.
    Returns (beta, sigma2, XtX_inv, df_resid) or None if not enough data / rank-deficient.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)

    if w_obs is None:
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    else:
        w_obs = np.asarray(w_obs, float)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(w_obs) & (w_obs > 0)

    y = y[mask]
    X = X[mask]
    if w_obs is not None:
        w_obs = w_obs[mask]

    n_obs, n_coef = X.shape
    if n_obs <= n_coef:
        return None
    if np.linalg.matrix_rank(X) < n_coef:
        return None

    if w_obs is None:
        # OLS
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ X.T @ y
        resid = y - X @ beta
    else:
        # WLS: transform by sqrt(weights)
        sqrtw = np.sqrt(w_obs)
        Xw = X * sqrtw[:, None]
        yw = y * sqrtw
        XtX = Xw.T @ Xw
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ Xw.T @ yw
        resid = yw - Xw @ beta
        # BUT residuals are in weighted space; to get sigma2 unbiased estimate on original scale:
        # sigma2 = sum((y - X@beta)^2 * w_obs) / df_resid
        resid_original = y - X @ beta
        sigma2 = np.sum((resid_original ** 2) * w_obs) / (n_obs - n_coef)
        return beta, sigma2, XtX_inv, (n_obs - n_coef)

    df_resid = n_obs - n_coef
    sigma2 = np.sum(resid**2) / df_resid

    return beta, sigma2, XtX_inv, df_resid


def run_limma_py(expr: pd.DataFrame,
                 design: pd.DataFrame,
                 sample_weights: pd.DataFrame = None,
                 coef_name: str = "Condition"):
    """
    expr: DataFrame (site × sample) with log2 intensities
    design: DataFrame (sample × covariates) with same index as expr.columns (samples)
    sample_weights: optional DataFrame (site × sample) or callable(site)->vector
      If DataFrame, index matches expr.index (sites), columns match samples.
    """
    X_full = design.values
    coef_idx = design.columns.get_loc(coef_name)

    records = []
    sites = []
    for site, row in expr.iterrows():
        y = row.reindex(design.index).values.astype(float)
        w_obs = None
        if sample_weights is not None:
            if isinstance(sample_weights, pd.DataFrame):
                if site in sample_weights.index:
                    w_obs = sample_weights.loc[site].reindex(design.index).values
                    med = np.nanmedian(w_obs[np.isfinite(w_obs)])
                    if np.isfinite(med) and med > 0:
                        w_obs = w_obs / med
            elif callable(sample_weights):
                w_obs = sample_weights(site)

        fit = lm_sitewise(y, X_full, w_obs=w_obs)
        if fit is None:
            continue
        beta, sigma2, XtX_inv, df_resid = fit
        records.append({
            "site": site,
            "beta": beta,
            "sigma2": sigma2,
            "XtX_inv": XtX_inv,
            "df_resid": df_resid
        })
        sites.append(site)

    if len(records) < 3:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # empirical Bayes (robust)
    sigma2_post, df_total = empirical_bayes_global(df["sigma2"].values, df["df_resid"].values)

    # ensure ordering aligns
    beta_c = np.array([b[coef_idx] for b in df["beta"]])
    XtX_cc = np.array([m[coef_idx, coef_idx] for m in df["XtX_inv"]])

    se = np.sqrt(sigma2_post * XtX_cc)
    t_stat = beta_c / se
    pval = 2 * stats.t.sf(np.abs(t_stat), df_total)

    res = pd.DataFrame({
        "site": df["site"].values,
        "logFC": beta_c,
        "t": t_stat,
        "P.Value": pval,
        "df": df_total,
        "sigma2": df["sigma2"].values,
        "sigma2_post": sigma2_post
    })
    # BH FDR
    res["adj.P.Val"] = multipletests(res["P.Value"].values, method="fdr_bh")[1]
    res = res.set_index("site")
    return res

def permutation_limma(expr_pw, design, weights_df,
                       n_perm=100, alpha=0.05):
    perm_counts = []

    for _ in range(n_perm):

        perm_design = design.copy()
        perm_design["Condition"] = np.random.permutation(
            perm_design["Condition"].values
        )

        res_perm = run_limma_py(
            expr_pw,
            perm_design,
            sample_weights=weights_df
        )

        if res_perm is None or res_perm.empty:
            continue

        n_sig = ((res_perm["adj.P.Val"] < alpha) & (abs(res_perm['logFC'])>=1)).sum()
        perm_counts.append(n_sig)

    if len(perm_counts) == 0:
        return None

    return np.array(perm_counts)



def compute_batch_global_prior(batch_df, method="median", type_experiment = 'phospho enrichment'):
    """
    Вычисляет robust global prior variance по всему батчу.
    Возвращает вектор длиной n_channels.
    """

    channel_cols = [c for c in batch_df.columns if c.endswith("_norm")]
    if not channel_cols:
        return None

    # Определяем PSM-типы
    if method == "aggregate":
        if type_experiment == 'phospho enrichment':
            group_cols = ["protein", "position_in_protein","peptide", "charge", "isotope_error"]
        else:
            group_cols = ['Modification',"protein", "position_in_protein","peptide", "charge"]
            batch_df = batch_df[batch_df['Modification']!='reference']
    elif method == "median":
        if type_experiment == 'phospho enrichment':
            group_cols = ['peptide',"charge", "isotope_error"]
        else:
            batch_df = batch_df[batch_df['Modification']!='reference']
            group_cols = ['peptide',"charge"]  
    else:
        raise ValueError("Invalid method")

    vars_list = []

    for _, g in batch_df.groupby(group_cols):

        X = g[channel_cols].replace(0, np.nan).to_numpy(float)

        if X.shape[0] < 2:
            continue

        med = np.nanmedian(X, axis=0)
        resid = X - med
        mad = np.nanmedian(np.abs(resid), axis=0)
        sigma = 1.4826 * mad
        var = sigma ** 2

        vars_list.append(var)

    if not vars_list:
        # если вообще нет variance
        return np.ones(len(channel_cols))

    vars_arr = np.vstack(vars_list)

    var_prior = np.nanmedian(vars_arr, axis=0)

    # fallback если где-то NaN
    global_scalar = np.nanmedian(var_prior[np.isfinite(var_prior)])
    if not np.isfinite(global_scalar):
        global_scalar = 1.0

    var_prior = np.where(np.isfinite(var_prior),
                         var_prior,
                         global_scalar)

    return var_prior


def chemo_coef(df_site):
    df_site['site'] = df_site['protein'].astype(str) + "_" + df_site['position_in_protein'].astype(str) + "_" + df_site['batch'].astype(str)
    df_zero = df_site[df_site['Modification']=='reference']
    df_mod = df_site[df_site['Modification']!='reference']
    df_stoich = df_mod.merge(df_zero, how = 'inner', on = ['site','protein','position_in_protein', 'batch'], suffixes = ('_mod','_unmod'))
    cols_mod = [c for c in df_stoich.columns if c.endswith("_stoich_mod")]
    cols_unmod = [c for c in df_stoich.columns if c.endswith("_stoich_unmod")]
    #stoich_df = pd.DataFrame(index=df_stoich.index)
    stoich_df = df_stoich[['protein','position_in_protein', 'batch', 'Modification_mod','TMT_group1_mod','TMT_group2_mod',
                          'TMT_group3_mod']].copy()
    for mod_col, ref_col in zip(cols_mod, cols_unmod):
        stoich_df[mod_col.replace("_mod","")] = df_stoich[mod_col] / (df_stoich[mod_col] + df_stoich[ref_col])
    return stoich_df.rename(columns={'Modification_mod':'Modification','TMT_group1_mod': 'TMT_group1',
                                     'TMT_group2_mod':'TMT_group2', 'TMT_group3_mod':'TMT_group3'})



def statistics(df, min_group_for_stats = 1, method = 'aggregate', variance_filter=0.25, type_experiment = 'whole proteome'):
    """
    Main statistics pipeline:
    1. PSM -> site aggregation (Bayesian)
    2. Long-format table with safe log2 stoichiometry
    3. Sample metadata / batch-group mapping
    4. Pairwise limma contrasts
    5. Returns: final stats, full expression table, site-level table
    """

    if df.empty:
        print("Input DataFrame is empty. Skipping statistics.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---------- PSM summary ----------
    pairwise = [
        ("TMT_group1", "TMT_group2"),
        ("TMT_group1", "TMT_group3"),
        ("TMT_group2", "TMT_group3"),
    ]
    batch_groups = (
        df.drop_duplicates("batch")
          .assign(batch=lambda x: x["batch"].astype(int))
          .set_index("batch")[["TMT_group1", "TMT_group2", "TMT_group3"]]
    )
    batch_groups = batch_groups.apply(
        lambda col: col.map(
            lambda xs: [str(x) for x in xs] if isinstance(xs, list) else [str(xs)]
        )
    )

    final_df = []
    # ---------- SITE-LEVEL AGGREGATION (Bayesian) ----------
    channel_cols = [c for c in df.columns if c.endswith("_norm")]
    df.dropna(subset=channel_cols, how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if 'id_prot' and 'spectrum_y' in df.columns.tolist():
        df = df.rename(columns = {'id_prot':'protein','spectrum_y':'scannr', 'modified_peptide_x' : 'peptide'})
    df.drop_duplicates(subset=['protein','position_in_protein','scannr'], inplace=True)
    
    if method == 'aggregate':
        global_prior = {}
        for batch in df['batch'].unique():
            var_ = compute_batch_global_prior(df[df['batch']==batch], method = 'aggregate')
            global_prior[batch] = var_
        if type_experiment == 'phospho enrichment':
            psm_stats = (
                df.groupby(["protein", "position_in_protein"])
                  .agg(n_psm=("scannr", "count"),
                       n_batches=("batch", pd.Series.nunique))
                  .reset_index()
            )
        else:
            psm_stats = (
                df.groupby(['Modification',"protein", "position_in_protein"])
                  .agg(n_psm=("scannr", "count"),
                       n_batches=("batch", pd.Series.nunique))
                  .reset_index()
            )
        psm_stats["site"] = (
            psm_stats["protein"].astype(str) + "_" + psm_stats["position_in_protein"].astype(str)
        )
        if type_experiment == 'phospho enrichment':
            df = df.merge(
                psm_stats[["n_psm", "n_batches",'protein','position_in_protein']],
                on =['protein','position_in_protein'],
                how="left"
            )
        else:
            df = df.merge(
                psm_stats[["n_psm", "n_batches",'protein','position_in_protein', 'Modification']],
                on =['protein','position_in_protein','Modification'],
                how="left"
            )
        
        eps = 1e-8
        df = df[df["n_batches"] >= 1]
        df = df[df["n_psm"] >= min_group_for_stats]

        if type_experiment == 'phospho enrichment':
            df_site = (
                df.groupby(["protein", "position_in_protein", "batch"])
                    .apply(lambda g: bayesian_site_aggregation(g, global_prior, method = method, type_experiment = type_experiment))
                    .reset_index()
            )

            long_df = df_site.melt(
                id_vars=["protein", "position_in_protein", "batch"],
                var_name="channel",
                value_name="stoich"
            )
            long_df["site"] = (
                long_df["protein"].astype(str)
                + "_" + long_df["position_in_protein"].astype(int).astype(str)
        )
        else:
            df_site = (
                df.groupby(['Modification', "protein", "position_in_protein", "batch"])
                    .apply(lambda g: bayesian_site_aggregation(g, global_prior, method = method))
                    .reset_index()
            )
            df_chem = chemo_coef(df_site)

            long_df = df_chem.melt(
                id_vars=['Modification',"protein", "position_in_protein", "batch"],
                var_name="channel",
                value_name="stoich"
            )
            long_df["site"] = (long_df["Modification"].astype(str)
                + "_" + long_df["protein"].astype(str)
                + "_" + long_df["position_in_protein"].astype(int).astype(str)
        )
        
        long_df["sample"] = (
            long_df["channel"].str.split("_").str[1]
            + "_" + long_df["batch"].astype(str)
        )
    
        
        long_df["is_weight"] = long_df["channel"].str.endswith("_weight")

        expr = (
            long_df[~long_df["is_weight"]]
            .pivot_table(
                index="site",
                columns="sample",
                values="stoich",
                aggfunc="mean"
            )
        )
        
        weights_df = (
            long_df[long_df["is_weight"]]
            .pivot_table(
                index="site",
                columns="sample",
                values="stoich",
                aggfunc="mean"
            )
        )
        
        weights_df = weights_df.merge(
            psm_stats.set_index("site")[["n_psm", "n_batches"]],
            left_index=True,
            right_index=True,
            how="left"
        )
    
        if expr.shape[0] < 50:
            return pd.DataFrame(), expr, df_site, weights_df
    
        # ---------- merge PSM stats ----------
        expr = expr.merge(
            psm_stats.set_index("site")[["n_psm", "n_batches"]],
            left_index=True,
            right_index=True,
            how="left"
        )
        expr_all = expr.copy() 
    
        expr_data = expr.drop(columns=["n_psm", "n_batches"])
        #expr_data = filter_by_variance(expr_data, percentile=variance_filter)
    
        # ---------- sample metadata ----------
        sample_df = expr_data.columns.to_series().str.rsplit("_", n=1, expand=True)
        sample_df.columns = ["s", "batch"]
        sample_df["batch"] = sample_df["batch"].astype(int)
        sample_df.index = expr_data.columns
        weights_df = weights_df.reindex(columns=expr.columns, fill_value=np.nan).drop(columns=["n_psm", "n_batches"])

    elif method == 'median':
        global_prior = {}
        for batch in df['batch'].unique():
            var_ = compute_batch_global_prior(df[df['batch']==batch], method = 'median', type_experiment = type_experiment)
            global_prior[batch] = var_
        
        if type_experiment == 'phospho enrichment':
            df = df.drop_duplicates(subset=["peptide",'isotope_error','charge','file_name', 'scannr']).reset_index(drop=True)
        else: 
            df = df.drop_duplicates(subset=["Modification",'peptide','charge', 'scannr']).reset_index(drop=True)
            
        psm_stats = (
            df.groupby(["peptide"])
              .agg(n_psm=("scannr", "count"),
                   n_batches=("batch", pd.Series.nunique))
              .reset_index()
        )

        df = df.merge(
            psm_stats[["n_psm", "n_batches","peptide"]],
            on =["peptide"],
            how="left"
        )
        
        eps = 1e-8
        df = df[df["n_batches"] >= 1]
        df = df[df["n_psm"] >= min_group_for_stats]

        df_site = (
            df.groupby(["batch", "peptide"])
                .apply(lambda g: bayesian_site_aggregation(g, global_prior, method='median', type_experiment = type_experiment))
                .reset_index()
        )
        if type_experiment != 'phospho enrichment':
            df_site = chemo_coef(df_site)

        long_df = df_site.melt(
            id_vars=["batch","peptide"],
            var_name="channel",
            value_name="value"
        )
        
        long_df["sample"] = (
            long_df["channel"].str.split("_").str[1]
            + "_" + long_df["batch"].astype(str)
        )
        
        long_df["is_weight"] = long_df["channel"].str.endswith("_weight")

        expr_data = (
            long_df[~long_df["is_weight"]]
            .pivot_table(
                index=["peptide",],
                columns="sample",
                values="value",
                aggfunc="mean"
            )
        )
        
        weights_df = (
            long_df[long_df["is_weight"]]
            .pivot_table(
                index=["peptide"],
                columns="sample",
                values="value",
                aggfunc="mean"
            )
        )
        weights_df = weights_df.reindex_like(expr_data)

        #expr_data[expr_data.index=='']
        #expr_data = filter_by_variance(expr_data, percentile=variance_filter)
        expr_all = expr_data.copy()

        sample_df = expr_data.columns.to_series().str.rsplit("_", n=1, expand=True)
        sample_df.columns = ["s", "batch"]
        sample_df["batch"] = sample_df["batch"].astype(int)
        sample_df.index = expr_data.columns

    else:
        raise ValueError("Not valid method")
        
    expr_noagg = build_expression_noagg(df, method=method, type_experiment = type_experiment)
    expr_noagg = expr_noagg.reindex(expr_data.index)
    noagg = []
    perm_df = pd.DataFrame()
    # ---------- run pairwise limma ----------
    for gA, gB in pairwise:
        condition = sample_df.apply(
            lambda r: 1 if r.s in batch_groups.loc[r.batch, gB]
            else 0 if r.s in batch_groups.loc[r.batch, gA]
            else np.nan,
            axis=1
        )
    
        design = pd.DataFrame({
            "Intercept": 1,
            "Condition": condition,
            "Batch": sample_df["batch"]
        }).dropna()

        design = pd.get_dummies(design, columns=["Batch"], drop_first=True)

    
        if design.shape[0] < min_group_for_stats:
            continue
    
        expr_pw = expr_data.loc[:, design.index]
    
        # ---------- LIMMA ----------
        res = run_limma_py(expr_pw, design, sample_weights=weights_df)
        if res is None or res.empty:
            continue
    
        res["contrast"] = f"{gB}_vs_{gA}"
        '''
        # ---------- permutation test ----------
        perm_counts = permutation_limma(
            expr_pw,
            design,
            weights_df,
            n_perm=100,
            alpha=0.05
        )
        
        if perm_counts is not None:
            real_sig = ((res["adj.P.Val"] < 0.05) & (abs(res['logFC'])>=1)).sum()
            emp_fdr = perm_counts.mean() / max(real_sig, 1)
        
            res["perm_mean_sig"] = perm_counts.mean()
            res["perm_empirical_fdr"] = emp_fdr
            perm_df[f'{gB}_vs_{gA}'] = perm_counts
            
        
            print(
                f"{gB}_vs_{gA}: real={real_sig}, "
                f"perm_mean={perm_counts.mean():.1f}, "
                f"empFDR={emp_fdr:.3f}"
            )
        '''
        # ---------- t-test ----------
        grpA_cols = design.index[design["Condition"] == 0]
        grpB_cols = design.index[design["Condition"] == 1]

        AB_cols = [*grpA_cols, *grpB_cols]
        
        def row_ttest(row):
            xA = row[grpA_cols].to_numpy(dtype=float)
            xB = row[grpB_cols].to_numpy(dtype=float)
        
            xA = xA[~np.isnan(xA)]
            xB = xB[~np.isnan(xB)]
        
            if len(xA) < 2 or len(xB) < 2:
                return np.nan
        
            _, pval = ttest_ind(xA, xB, equal_var=False)
            return pval
        
        # считаем только для строк limma
        site_order = res.index.to_list()
        pvals = expr_pw.loc[site_order].apply(row_ttest, axis=1)
        res['pval_ttest'] = pvals.values
        res = res.reset_index()
        final_df.append(res)

        res_noagg = run_limma_py(expr_noagg, design, sample_weights = pd.DataFrame())
        res_noagg["contrast"] = f"{gB}_vs_{gA}"
        mean_int = expr_noagg[AB_cols].median(axis=1)
        res_noagg["mean_intensity"] = mean_int.loc[res_noagg.index]
        res_noagg = res_noagg.reset_index()
        noagg.append(res_noagg)

    if not final_df:
        return pd.DataFrame(), expr_all, df_site, weights_df

    final_df = pd.concat(final_df, ignore_index=True)
    noagg =  pd.concat(noagg, ignore_index=True)
        
    if method=='median':
        final_df[['peptide']] = pd.DataFrame(
            final_df['site'].tolist(),
            index=final_df.index
        )
        del final_df['site']
        if type_experiment == 'phospho enrichment': 
            final_df = final_df.merge(
                df[["site", "n_psm", "n_batches","peptide",'isotope_error','charge','peptide_clean']],
                on=['peptide'],
                how="left"
            )
            final_df = final_df.groupby(['peptide','peptide_clean','logFC',
                                         't','P.Value', 'df', 'adj.P.Val', 'contrast','pval_ttest', 'n_psm', 'n_batches',
                                        'sigma2','sigma2_post']).agg(list).reset_index()
        else:
            final_df = final_df.merge(
                df[['Modification', "site", "n_psm", "n_batches","peptide",'charge','peptide_y']],
                on=['peptide'],
                how="left"
            )
            final_df = final_df.groupby(['Modification','peptide','peptide_y','logFC',
                                         't','P.Value', 'df', 'adj.P.Val', 'contrast','pval_ttest', 'n_psm', 'n_batches',
                                        'sigma2','sigma2_post']).agg(list).reset_index()
        noagg.rename(columns ={'site':'peptide'},inplace=True)
        noagg = noagg.merge(
            psm_stats[["peptide", "n_psm", "n_batches"]],
            on="peptide",
            how="left"
        )
    if method=='aggregate':
        final_df = final_df.merge(
            psm_stats[["site", "n_psm", "n_batches"]],
            on="site",
            how="left"
        )
        noagg = noagg.merge(
            psm_stats[["site", "n_psm", "n_batches"]],
            on="site",
            how="left"
        )

    return final_df, expr_all, df_site, weights_df, design, noagg, perm_df

