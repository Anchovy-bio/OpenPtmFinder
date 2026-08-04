import numpy as np
import pandas as pd
from scipy import stats
from scipy import special
import logging
from scipy.stats import rankdata
from statsmodels.robust.scale import mad
from statsmodels.stats.multitest import multipletests


logger = logging.getLogger(__name__)



def estimate_sigma_psm(df, channel_cols, eps=1e-8):
    """
    Robust sigma per PSM across TMT channels (non-log space)
    """
    X = df[channel_cols].values

    median = np.nanmedian(X, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(X - median), axis=1)

    # MAD → sigma for Gaussian
    sigma = 1.4826 * mad
    sigma[sigma < eps] = eps

    return sigma


def bayes_group_intensity(df):
    """
    Bayesian aggregation for TMT group intensities (weighted mean with precision)
    """
    cols = [c for c in df.columns if c.endswith("_norm")]
    
    if not cols:
        return pd.Series(dtype=float, index=[])

    X = df[cols].values
    sigma = estimate_sigma_psm(df, cols)

    precision = 1.0 / (sigma[:, None] ** 2 + 1e-12)  # небольшая стабилизация

    num = np.nansum(X * precision, axis=0)
    den = np.nansum(precision, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = num / den
        out[~np.isfinite(out)] = np.nan   # или 0 — по вашему желанию

    return pd.Series(out, index=cols)



def weighted_mean(group):
    # уникальные каналы в каждой группе
    group1 = sorted(set(np.concatenate(group['TMT_group1'].values)))
    group2 = sorted(set(np.concatenate(group['TMT_group2'].values)))

    intens = bayes_group_intensity(group)
    
    #cols1 = [f'intensity_{ch}_norm' for ch in group1 if f'intensity_{ch}_norm' in intens.index]
    #cols2 = [f'intensity_{ch}_norm' for ch in group2 if f'intensity_{ch}_norm' in intens.index]

    # коэффициенты: сумма по PSM
    #coef1 = group[cols1].values if cols1 else np.array([])
    #coef2 = group[cols2].values if cols2 else np.array([])

    # создаём словарь для столбцов со стохиометрией
    stoich_dict = {f'{col}_stoich': [val] for col, val in zip(intens.index, intens.values)}

    return pd.DataFrame({
        'peptide_y': [group['peptide_y'].tolist()],
        'charge_y': [group['charge'].tolist()],
        'peptide_x': [[i for sub in group['peptide_x'] for i in sub]],
        'modified_peptide_x': [[i for sub in group['modified_peptide_x'] for i in sub]],
        'spectrum_y': [[i for sub in group['spectrum_y'] for i in sub]],
        'spectrum_x': [[i for sub in group['spectrum_x'] for i in sub]],
        'TMT_group1': [group1],
        'TMT_group2': [group2],
        **stoich_dict,  # добавляем столбцы стохиометрии
        #'coef1': [coef1.tolist()],
        #'coef2': [coef2.tolist()]
    })


def chemo_coef(df_site):
    df_site['site'] = df_site['id_prot'].astype(str) + "_" + df_site['position_in_protein'].astype(str) + "_" + df_site['batch'].astype(str)
    df_zero = df_site[df_site['Modification']=='reference']
    df_mod = df_site[df_site['Modification']!='reference']
    df_stoich = df_mod.merge(df_zero, how = 'inner', on = ['site','id_prot','position_in_protein', 'batch'], suffixes = ('_mod','_unmod'))
    cols_mod = [c for c in df_stoich.columns if c.endswith("_stoich_mod")]
    cols_unmod = [c for c in df_stoich.columns if c.endswith("_stoich_unmod")]
    #stoich_df = pd.DataFrame(index=df_stoich.index)
    stoich_df = df_stoich[['id_prot','position_in_protein', 'batch', 'Modification_mod','TMT_group1_mod','TMT_group2_mod']].copy()
    for mod_col, ref_col in zip(cols_mod, cols_unmod):
        stoich_df[mod_col.replace("_mod","")] = df_stoich[mod_col] / (df_stoich[mod_col] + df_stoich[ref_col])
    return stoich_df.rename(columns={'Modification_mod':'Modification','TMT_group1_mod': 'TMT_group1','TMT_group2_mod':'TMT_group2'})


def logit_transform(x, eps=1e-5):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x) - np.log1p(-x)


def empirical_bayes_global(sigma2, df_resid):
    """
    sigma2: array (n_sites,)
    df_resid: array (n_sites,)
    """

    sigma2 = np.clip(sigma2, 1e-8, np.inf)

    log_s2 = np.log(sigma2)
    mean_log = log_s2.mean()
    var_log = log_s2.var(ddof=1)

    if var_log <= 0:
        return sigma2, df_resid

    d0 = 2.0 / var_log
    s0_sq = np.exp(
        mean_log
        - special.digamma(d0 / 2)
        + np.log(d0 / 2)
    )

    sigma2_post = (d0 * s0_sq + df_resid * sigma2) / (d0 + df_resid)
    df_total = d0 + df_resid

    return sigma2_post, df_total


def lm_sitewise(y, X):
    mask = ~np.isnan(y)
    y = y[mask]
    X = X[mask]

    n_obs, n_coef = X.shape
    if n_obs <= n_coef:
        return None

    if np.linalg.matrix_rank(X) < n_coef:
        return None

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    df_resid = n_obs - n_coef
    sigma2 = (resid @ resid) / df_resid

    return beta, sigma2, XtX_inv, df_resid



def run_limma_py(expr: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:

    X = design.values
    coef_idx = design.columns.get_loc("Condition")

    records = []

    for site, row in expr.iterrows():
        fit = lm_sitewise(row.values.astype(float), X)
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

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # -------- GLOBAL EB --------
    sigma2_post, df_total = empirical_bayes_global(
        df["sigma2"].values,
        df["df_resid"].values
    )

    # -------- moderated t --------
    beta_c = np.array([b[coef_idx] for b in df["beta"]])
    XtX_cc = np.array([m[coef_idx, coef_idx] for m in df["XtX_inv"]])

    se = np.sqrt(sigma2_post * XtX_cc)
    t_stat = beta_c / se
    pval = 2 * stats.t.sf(np.abs(t_stat), df_total)

    res = pd.DataFrame({
        "site": df["site"],
        "logFC": beta_c,
        "t": t_stat,
        "P.Value": pval,
        "df": df_total
    })

    res["adj.P.Val"] = multipletests(
        res["P.Value"], method="fdr_bh"
    )[1]

    return res


def filter_by_variance(expr_df: pd.DataFrame, percentile: float = 0.2) -> pd.DataFrame:
    """
    Независимая фильтрация сайтов по общей вариативности.
    Удаляет сайты с наименьшим стандартным отклонением.
    """
    # Рассчитываем стандартное отклонение для каждой строки (сайта)
    row_stdev = expr_df.std(axis=1)
    
    # Определяем порог отсечения (например, нижние 20%)
    cutoff = row_stdev.quantile(percentile)
    
    # Оставляем только те сайты, чья вариативность выше порога
    filtered_df = expr_df[row_stdev > cutoff]
    
    logger.info(f"Filtering: removed {len(expr_df) - len(filtered_df)} sites "
                f"with stdev <= {cutoff:.4f} (bottom {percentile*100}%).")
    
    return filtered_df


def new_stats(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return None

    stat = (
        df.drop_duplicates(
            subset=['Modification','id_prot','position_in_protein',
                    'batch','peptide_y','charge','spectrum_y']
        )
    )

    df_site = (
        stat.groupby(['Modification','id_prot','position_in_protein','batch'])
            .apply(weighted_mean)
            .reset_index()
    )

    df_site = chemo_coef(df_site)
    df_site = df_site[df_site['Modification'] != 'reference']

    intensity_cols = [c for c in df_site.columns if c.endswith("stoich")]

    long_df = df_site.melt(
        id_vars=["Modification", "id_prot", "position_in_protein", "batch"],
        value_vars=intensity_cols,
        var_name="channel",
        value_name="stoich"
    )

    long_df["stoich_logit"] = logit_transform(long_df["stoich"])

    long_df["sample"] = (
        long_df["channel"].str.split("_").str[1] + "_" + long_df["batch"].astype(int).astype(str)
    )

    long_df["site"] = (
        long_df["Modification"].astype(str) + "_" +
        long_df["id_prot"].astype(str) + "_" +
        long_df["position_in_protein"].astype(str)
    )
                   
    batch_groups = (
        df.drop_duplicates("batch")
        .assign(batch=lambda x: x["batch"].astype(float).astype(int))
        .set_index("batch")[['TMT_group1', 'TMT_group2']]
    )

    batch_groups["TMT_group1"] = batch_groups["TMT_group1"].apply(lambda x: [str(i) for i in x])
    batch_groups["TMT_group2"] = batch_groups["TMT_group2"].apply(lambda x: [str(i) for i in x])
    
    final_df = pd.DataFrame()
    dop_df = pd.DataFrame()
    
    long_df['mods'] = [x.split('@')[0] for x in long_df["Modification"].tolist()]
    n_mod = len(long_df['mods'].unique())
    logger.info(f'There`re was found {n_mod} types of modification')
    
    for mod in long_df['mods'].unique():
        logger.info(f'Calculating: {mod}')
        expr = long_df[long_df['mods']==mod]
        n_sites = expr['site'].nunique()
        
        if n_sites <= 50:
            dop_df = pd.concat([dop_df, expr], ignore_index=True)
            logger.warning(f"Group {mod} is too small ({n_sites} sites).Using global parameters...")
            continue
            
        expr = expr.pivot_table(
            index="site",
            columns="sample",
            values="stoich_logit",
            aggfunc="mean"
        )
        print(expr.head())
        expr = filter_by_variance(expr, percentile=0.25)

        sample_df = (
            pd.Series(expr.columns, name="sample")
            .str.split("_", expand=True)
            .rename(columns={0: "s", 1: "batch"})
        )

        sample_df["s"] = sample_df["s"].astype(str)
        sample_df["batch"] = sample_df["batch"].astype(float).astype(int)

        sample_df.index = expr.columns

        sample_df["Condition"] = sample_df.apply(
            lambda r: 1 if r.s in batch_groups.loc[r.batch, 'TMT_group2']
            else 0 if r.s in batch_groups.loc[r.batch, 'TMT_group1']
            else np.nan,
            axis=1
        )

        design = pd.DataFrame({
            "Intercept": 1,
            "Condition": sample_df["Condition"]
        }).dropna()

        if design.empty:
            logger.error("Design matrix is empty — check TMT_group1 / TMT_group2 mapping.")
            return None

        expr = expr.loc[:, design.index]

        res_limma = run_limma_py(expr, design)

        res_limma["site"] = expr.index.values
        final_df = pd.concat([final_df, res_limma], ignore_index=True)
    
    if dop_df.empty or dop_df['site'].nunique() <= 50:
            logger.warning("Not enough samples for global statistical testing")
            return final_df, df_site
    else:
        expr = dop_df.pivot_table(
            index="site",
            columns="sample",
            values="stoich_logit",
            aggfunc="mean"
        )
            
        expr = filter_by_variance(expr, percentile=0.25)
        print(expr.head())
        sample_df = (
            pd.Series(expr.columns, name="sample")
            .str.split("_", expand=True)
            .rename(columns={0: "s", 1: "batch"})
        )

        sample_df["s"] = sample_df["s"].astype(str)
        sample_df["batch"] = sample_df["batch"].astype(float).astype(int)

        sample_df.index = expr.columns

        sample_df["Condition"] = sample_df.apply(
            lambda r: 1 if r.s in batch_groups.loc[r.batch, 'TMT_group2']
            else 0 if r.s in batch_groups.loc[r.batch, 'TMT_group1']
            else np.nan,
            axis=1
        )

        design = pd.DataFrame({
            "Intercept": 1,
            "Condition": sample_df["Condition"]
        }).dropna()

        if design.empty:
            logger.error("Design matrix is empty — check TMT_group1 / TMT_group2 mapping.")
            return None

        expr = expr.loc[:, design.index]

        res_limma = run_limma_py(expr, design)

        res_limma["site"] = expr.index.values
        final_df = pd.concat([final_df, res_limma], ignore_index=True)
        
        return final_df, df_site

