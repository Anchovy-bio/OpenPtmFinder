"""
report.py — data layer and self-contained interactive HTML report for
OpenPtmFinder results.

Data layer
----------
`export_report_data(output_dir)` consolidates the pipeline outputs into a
parquet layer under `<output_dir>/report_data/`:

    annotated_df.parquet    — PSM-level annotation (from annotated_df.pickle)
    stat_results.parquet    — all final_stat_result_*.csv with a mod_family tag
    expr_corrected.parquet  — all expr_all_corrected_*.csv (site x sample)

Both this module and the Dash interactive server (dash_server.py) read the
parquet files first and fall back to the legacy CSV/pickle outputs, so the
report can also be generated for older runs. The parquet files can equally be
queried directly with DuckDB, e.g.:

    SELECT * FROM 'report_data/stat_results.parquet' WHERE "adj.P.Val" < 0.05;

Report
------
`generate_report(output_dir, ...)` renders ONE portable .html file (plotly.js
and all table/volcano logic embedded — no CDN, no server) with:

  - run summary cards (proteins / sites / modification types / hits);
  - PTM landscape: identified (donut) and statistically tested (bar) sites
    per modification type, with a consistent per-modification color map;
  - differential analysis: volcano plot with CLIENT-SIDE controls
    (modification family, ONE pairwise contrast at a time, adj.P and
    |logFC| thresholds, live hit counter) plus a table of the
    differentially expressed sites only, kept in sync with the volcano
    cutoffs (search / sort / pagination / CSV export, vanilla JS);
  - heatmap of the top differential sites (row z-scored);
  - per-protein modification landscape for the top significant proteins
    (lollipop plot + peptide-coverage track from FASTA);
  - QC section: permutation validation table, n_PSM and precision
    distributions.

Usage (standalone):
    python report.py -o <output_dir> [-f <protein_db.fasta>]
                     [--plotlyjs inline|cdn] [--top-proteins 10]
                     [--alpha 0.05] [--logfc 1.0]

Python 3.10+ compatible.
"""

import argparse
import glob
import json
import logging
import os
import re
from datetime import datetime
from html import escape as _esc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

try:  # plotly >= 5/6
    from plotly.offline import get_plotlyjs as _get_plotlyjs
except ImportError:  # very old plotly
    def _get_plotlyjs():
        return pio.get_plotlyjs()

logger = logging.getLogger(__name__)

STAT_GLOB = "final_stat_result_*.csv"
EXPR_GLOB = "expr_all_corrected_*.csv"
WEIGHTS_GLOB = "weights_df_*.csv"
PERM_GLOB = "permutation_*.csv"

REPORT_DATA_DIR = "report_data"

# Consistent, colorful qualitative palette for modification types
_MOD_PALETTE = (
    ["#e63946", "#2a9d8f", "#ffb703", "#4361ee", "#7b2cbf", "#f3722c",
     "#00b4d8", "#90be6d", "#e07a5f", "#6a4c93", "#43aa8b", "#f94144"]
)

PLOTLY_CDN = ('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" '
              'charset="utf-8"></script>')


# ---------------------------------------------------------------------------
# Loading (parquet-first, CSV/pickle fallback)
# ---------------------------------------------------------------------------

def _tag_from_path(path: str, prefix: str) -> str:
    base = os.path.basename(path)
    m = re.match(re.escape(prefix) + r"(?P<method>.+?)_(?P<tag>.+)\.csv$", base)
    return m.group("tag") if m else base[:-4]


def _parse_stat_results(df: pd.DataFrame) -> pd.DataFrame:
    """Robust site parsing: split from the right so modification names or
    peptide sequences containing '_' do not corrupt the protein id."""
    out = df
    parts = out["site"].astype(str).str.rsplit("_", n=2, expand=True)
    if parts.shape[1] == 3:                      # aggregate: Mod_protein_pos
        out["modification"] = parts[0]
        out["protein"] = parts[1]
        out["position"] = pd.to_numeric(parts[2], errors="coerce")
    elif parts.shape[1] == 2:                    # median: protein_peptide
        out["modification"] = np.nan
        out["protein"] = parts[0]
        out["position"] = np.nan
    else:
        out["modification"] = np.nan
        out["protein"] = out["site"]
        out["position"] = np.nan
    return out


def _load_stat_results_csv(output_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(output_dir, STAT_GLOB))):
        # final_stat_result_with_dbs.csv is the PSM-level export enriched with
        # DB-annotation columns — not a per-family statistics table.
        if os.path.basename(path) == "final_stat_result_with_dbs.csv":
            continue
        df = pd.read_csv(path)
        df["mod_family"] = _tag_from_path(path, "final_stat_result_")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _parse_stat_results(pd.concat(frames, ignore_index=True))


def load_stat_results(output_dir: str) -> pd.DataFrame:
    pq = os.path.join(output_dir, REPORT_DATA_DIR, "stat_results.parquet")
    if os.path.isfile(pq):
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            logger.warning(f"Could not read {pq} ({e}); falling back to CSV.")
    return _load_stat_results_csv(output_dir)


def _load_expression_csv(output_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(output_dir, EXPR_GLOB))):
        df = pd.read_csv(path, index_col=0)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    expr = pd.concat(frames)
    expr = expr[~expr.index.duplicated(keep="first")]
    for aux in ("n_psm", "n_batches"):
        if aux in expr.columns:
            expr = expr.drop(columns=aux)
    return expr.apply(pd.to_numeric, errors="coerce")


def load_expression(output_dir: str) -> pd.DataFrame:
    pq = os.path.join(output_dir, REPORT_DATA_DIR, "expr_corrected.parquet")
    if os.path.isfile(pq):
        try:
            df = pd.read_parquet(pq)
            if "site" in df.columns:  # restore the matrix shape (site x sample)
                df = df.set_index("site")
            return df
        except Exception as e:
            logger.warning(f"Could not read {pq} ({e}); falling back to CSV.")
    return _load_expression_csv(output_dir)


def load_weights(output_dir: str) -> pd.DataFrame:
    frames = [pd.read_csv(p, index_col=0)
              for p in sorted(glob.glob(os.path.join(output_dir, WEIGHTS_GLOB)))]
    if not frames:
        return pd.DataFrame()
    w = pd.concat(frames)
    return w[~w.index.duplicated(keep="first")].apply(pd.to_numeric, errors="coerce")


def load_permutations(output_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(output_dir, PERM_GLOB))):
        df = pd.read_csv(path)
        keep = [c for c in df.columns if c != "perm_counts"]
        df = df[keep]
        if "mod_family" not in df.columns:
            # permutation_{method}_{tag}.csv -> tag (method may be absent in
            # older outputs, so strip only a known method prefix)
            tag = os.path.basename(path)[len("permutation_"):-4]
            for meth in ("aggregate_", "median_"):
                if tag.startswith(meth):
                    tag = tag[len(meth):]
            df["mod_family"] = tag
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Site-level caches written by the database-annotation stage (one row per
# unique protein-position-modification site).
DB_ANNOTATION_FILES = {
    "iPTMnet": ("iptmnet_positions.csv", "in_iPTM", "perhapse_in_iPTM"),
    "dbPTM":   ("dbptm_positions.csv", "in_dbPTM", "perhapse_in_dbPTM"),
    "SIGNOR":  ("signor_sites.csv", "in_SIGNOR", None),
}


def load_db_annotation(output_dir: str) -> dict:
    """Load the per-database site-annotation caches that exist in output_dir.

    Returns {db_name: DataFrame}; databases whose cache file is missing or
    unreadable are simply absent.
    """
    out = {}
    for db, (fname, _, _) in DB_ANNOTATION_FILES.items():
        path = os.path.join(output_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            continue
        if not df.empty:
            out[db] = df
    return out


def _truthy(series: pd.Series) -> pd.Series:
    """Robust bool interpretation of a column that may hold bools, 0/1 or
    strings ('True'/'False') after a CSV round-trip."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        ("true", "1", "yes"))


def _load_annotated_pickle(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "annotated_df.pickle")
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_pickle(path)
    except Exception as e:
        logger.warning(f"Could not read annotated_df.pickle: {e}")
        return pd.DataFrame()


def load_annotated(output_dir: str) -> pd.DataFrame:
    pq = os.path.join(output_dir, REPORT_DATA_DIR, "annotated_df.parquet")
    if os.path.isfile(pq):
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            logger.warning(f"Could not read {pq} ({e}); falling back to pickle.")
    return _load_annotated_pickle(output_dir)


def load_fasta_dict(fasta_file: str) -> dict:
    seqs = {}
    if not fasta_file or not os.path.isfile(fasta_file):
        return seqs
    descr, chunks = None, []
    with open(fasta_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if descr is not None and not re.search(r"DECOY_|rev_", descr, re.I):
                    pid = descr.split("|")[1] if "|" in descr else descr.split()[0]
                    seqs[pid] = "".join(chunks)
                descr, chunks = line[1:], []
            elif line:
                chunks.append(line)
    if descr is not None and not re.search(r"DECOY_|rev_", descr, re.I):
        pid = descr.split("|")[1] if "|" in descr else descr.split()[0]
        seqs[pid] = "".join(chunks)
    return seqs


# ---------------------------------------------------------------------------
# Parquet data layer
# ---------------------------------------------------------------------------

def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame parquet-safe for pyarrow.

    CSV reloads produce object columns that mix str values with float('nan')
    (e.g. the DB-annotation columns dbptm_pmids, signor_pmids,
    perhapse_ptm_type). pyarrow infers a string array from the str values
    and then fails on the floats ("Expected bytes, got a 'float' object").
    Object columns whose non-null values are all strings are cast to the
    pandas 'string' dtype, where missing values are pd.NA — which pyarrow
    handles natively. Columns with genuinely non-string content are left
    untouched.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna()
        if non_null.empty or non_null.map(lambda v: isinstance(v, str)).all():
            df[col] = df[col].astype('string')
    return df


def export_report_data(output_dir: str) -> dict:
    """Consolidate pipeline outputs into the parquet data layer used by the
    HTML report and the Dash interactive server.

    Returns a dict of written paths. Requires pyarrow; without it the report
    and server transparently fall back to the CSV/pickle outputs, so this is
    an optimization/consolidation step, not a hard dependency.
    """
    output_dir = os.path.abspath(output_dir)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        logger.warning("pyarrow is not installed; skipping the parquet data "
                       "layer (report/interactive mode will read CSV/pickle). "
                       "Install with: pip install pyarrow")
        return {}

    rd = os.path.join(output_dir, REPORT_DATA_DIR)
    os.makedirs(rd, exist_ok=True)
    written = {}

    stats = _load_stat_results_csv(output_dir)
    if not stats.empty:
        path = os.path.join(rd, "stat_results.parquet")
        try:
            _sanitize_for_parquet(stats).to_parquet(path, index=False)
            written["stat_results"] = path
        except Exception as e:
            logger.warning(f"Could not write stat_results.parquet ({e}); "
                           "CSV remains the source for the statistics table.")

    expr = _load_expression_csv(output_dir)
    if not expr.empty:
        path = os.path.join(rd, "expr_corrected.parquet")
        expr.reset_index(names="site").to_parquet(path, index=False)
        written["expr_corrected"] = path

    annot = _load_annotated_pickle(output_dir)
    if not annot.empty:
        path = os.path.join(rd, "annotated_df.parquet")
        try:
            _sanitize_for_parquet(annot).to_parquet(path, index=False)
            written["annotated_df"] = path
        except Exception as e:
            logger.warning(f"Could not write annotated_df.parquet ({e}); "
                           "pickle remains the source for the PSM table.")

    logger.info(f"Report data layer written to {rd} "
                f"({', '.join(written) if written else 'nothing to export'})")
    return written


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#334155"),
    margin=dict(t=60, b=50, l=70, r=30),
    title_font=dict(size=15, color="#0f2540"),
)


def mod_color_map(mods) -> dict:
    """Deterministic modification -> color mapping shared by all figures."""
    mods = sorted({str(m) for m in mods if pd.notna(m)})
    return {m: _MOD_PALETTE[i % len(_MOD_PALETTE)] for i, m in enumerate(mods)}


def ptm_landscape_figures(stats: pd.DataFrame, annot: pd.DataFrame,
                          colors: dict):
    """Donut of identified mod types (unique sites) + bar of tested sites."""
    figs = {}

    if not annot.empty and "Modification" in annot.columns:
        idf = annot[annot["Modification"] != "reference"].copy()
        idf["Mods"] = idf["Modification"].astype(str).str.split("@").str[0]
        if "position_in_protein" in idf.columns:
            idf = idf.dropna(subset=["position_in_protein"])
            idf = idf.drop_duplicates(subset=["id_prot", "position_in_protein", "Mods"])
        counts = idf["Mods"].value_counts()
        top, other = counts.head(8), counts.iloc[8:]
        labels, values = list(top.index), list(top.values)
        pie_colors = [colors.get(l, "#94a3b8") for l in labels]
        if not other.empty:
            labels.append("Other")
            values.append(int(other.sum()))
            pie_colors.append("#94a3b8")
        figs["pie"] = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.45,
            marker=dict(colors=pie_colors, line=dict(color="white", width=2)),
            textinfo="percent", textposition="inside",
            insidetextorientation="horizontal",
            hovertemplate="%{label}: %{value} sites (%{percent})<extra></extra>"))
        figs["pie"].update_layout(
            title="Identified modification sites by type<br>"
                  "<sup>unique protein-position sites</sup>",
            showlegend=True,
            legend=dict(orientation="v", x=1.0, y=0.5,
                        xanchor="left", yanchor="middle"),
            uniformtext=dict(minsize=11, mode="hide"),
            **{**_BASE_LAYOUT, "margin": dict(t=60, b=40, l=30, r=110)})

    if not stats.empty:
        tested = (stats.dropna(subset=["modification"])
                  .drop_duplicates(subset=["mod_family", "site"])["modification"]
                  .value_counts())
        if not tested.empty:
            figs["bar"] = go.Figure(go.Bar(
                x=tested.index, y=tested.values,
                marker_color=[colors.get(m, "#4361ee") for m in tested.index],
                marker_line=dict(color="white", width=1),
                text=tested.values, textposition="outside",
                hovertemplate="%{x}: %{y} tested sites<extra></extra>"))
            figs["bar"].update_layout(
                title="Statistically tested sites per modification type",
                xaxis_title="Modification", yaxis_title="Tested sites",
                **_BASE_LAYOUT)
    return figs


def de_payload(stats: pd.DataFrame) -> str:
    """Serialize per-site differential results to compact JSON for the
    client-side volcano plot AND the significant-sites table (one shared
    payload keeps both views in sync; short keys keep the report small).

    Keys: s=site, g=protein, m=modification, f=mod_family, c=contrast,
    x=logFC, a=adj.P.Val, p=P.Value, t=t-statistic, o=position, n=n_psm,
    u=status. Missing numeric values are null.
    """
    df = stats.dropna(subset=["logFC"]).copy()
    if df.empty:
        return "[]"

    def col(name, default=np.nan):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    def fnum(v, nd=None):
        if pd.isna(v) or not np.isfinite(v):
            return None
        return round(float(v), nd) if nd else float(v)

    def inum(v):
        return None if pd.isna(v) else int(v)

    payload = [{
        "s": str(s), "g": str(g),
        "m": ("NA" if pd.isna(m) else str(m)),
        "f": str(f), "c": (str(c) if pd.notna(c) else "all"),
        "x": fnum(x, 4), "a": fnum(a), "p": fnum(p), "t": fnum(t, 3),
        "o": inum(o), "n": inum(n),
        "u": ("" if pd.isna(u) else str(u)),
    } for s, g, m, f, c, x, a, p, t, o, n, u in zip(
        df["site"], df["protein"], col("modification"), col("mod_family", ""),
        col("contrast", ""), df["logFC"], col("adj.P.Val"), col("P.Value"),
        col("t"), col("position"), col("n_psm"), col("status", ""))]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def heatmap_figure(stats: pd.DataFrame, expr: pd.DataFrame, top_n: int = 30):
    if expr.empty or stats.empty:
        return None
    best = (stats.dropna(subset=["adj.P.Val"])
            .groupby("site")["adj.P.Val"].min().nsmallest(top_n))
    mat = expr.reindex(best.index).dropna(how="all")
    if mat.empty:
        return None
    mat = mat.loc[best.index.intersection(mat.index)]
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(z.columns), y=list(z.index),
        colorscale="RdBu", reversescale=True, zmid=0,
        colorbar=dict(title="z-score")))
    fig.update_layout(title=f"Top {len(mat)} differential sites "
                            "(row z-score of corrected abundance)",
                      height=max(400, 22 * len(mat)),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                      xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                      **{**_BASE_LAYOUT, "margin": dict(t=60, b=130, l=170, r=40)})
    return fig


def _peptide_coverage(seq: str, peptides: list) -> np.ndarray:
    """Per-residue covering-peptide count.

    ``peptides`` is expected WITH multiplicities (one entry per PSM), as in
    the legacy web-server ``peptide_density_distribution``: a residue covered
    by two distinct peptides or by two PSMs of the same peptide gets count 2.
    """
    cov = np.zeros(len(seq), dtype=float)
    for pep in peptides:
        start = 0
        while True:
            i = seq.find(pep, start)
            if i == -1:
                break
            cov[i:i + len(pep)] += 1
            start = i + 1
    return cov


# Coverage colorscale: zero coverage must stay visibly light grey (not the
# pale pink of "Reds" at z=0) so uncovered stretches read as "no data".
_COVERAGE_SCALE = [
    [0.0, "#e8e8e8"], [0.02, "#fee5d9"], [0.25, "#fcae91"],
    [0.5, "#fb6a4a"], [0.75, "#de2d26"], [1.0, "#a50f15"],
]


def single_protein_figure(prot: str, annot: pd.DataFrame, fasta_seqs: dict,
                          colors: dict = None, stats: pd.DataFrame = None,
                          alpha: float = 0.05, logfc_thr: float = 1.0):
    """Per-protein modification landscape (two stacked panels, shared x):

    Top panel — lollipop plot: PSM count per modified position, one color per
    modification type; marker size grows with the PSM count, and sites that
    pass the differential cutoffs (adj.P < alpha, |logFC| >= logfc_thr in at
    least one contrast, taken from ``stats``) get a red rim plus per-contrast
    statistics in the hover tooltip.

    Bottom panel — peptide-coverage strip: one cell per residue of the FASTA
    sequence, colored (Reds) by the fraction of distinct identified peptides
    covering that residue; hover shows the amino-acid letter, position and
    raw covering-peptide count. Empty when no FASTA is available.

    Shared by the static report (top-K dropdown) and the Dash server
    (arbitrary protein queried on the fly).
    """
    sub = annot[(annot["id_prot"].astype(str) == str(prot)) &
                (annot["Modification"] != "reference")].copy()
    sub = sub.dropna(subset=["position_in_protein"])
    if sub.empty:
        return None
    sub["Mods"] = sub["Modification"].astype(str).str.split("@").str[0]
    grp = sub.groupby(["Mods", "position_in_protein"]).size().reset_index(name="count")
    ymax = max(1, int(grp["count"].max()))
    if colors is None:
        colors = mod_color_map(grp["Mods"].unique())

    seq = fasta_seqs.get(str(prot)) if fasta_seqs else None

    # ---- differential statistics per position (for rims + hover) ----
    pos_stat, sig_pos = {}, set()
    if stats is not None and not stats.empty and \
            {"protein", "position", "contrast", "logFC", "adj.P.Val"}.issubset(stats.columns):
        st = stats[stats["protein"].astype(str) == str(prot)]
        for pos, g in st.groupby("position"):
            lines, hit = [], False
            for _, r in g.iterrows():
                if pd.isna(r["adj.P.Val"]) or pd.isna(r["logFC"]):
                    continue
                sig = (r["adj.P.Val"] < alpha) and (abs(r["logFC"]) >= logfc_thr)
                hit |= sig
                lines.append(f"{r['contrast']}: logFC={r['logFC']:.2f}, "
                             f"adj.P={r['adj.P.Val']:.2g}" + (" *" if sig else ""))
            if lines:
                pos_stat[int(pos)] = "<br>".join(lines)
                if hit:
                    sig_pos.add(int(pos))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.05)

    for mod, g in grp.groupby("Mods"):
        color = colors.get(mod, "#4361ee")
        xs, ys = [], []
        for _, r in g.iterrows():
            xs += [r["position_in_protein"], r["position_in_protein"], None]
            ys += [0, r["count"], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                 legendgroup=str(mod),
                                 line=dict(width=1.5, color=color),
                                 hoverinfo="skip"), row=1, col=1)

        texts, rim = [], []
        for _, r in g.iterrows():
            pos = int(r["position_in_protein"])
            aa = seq[pos - 1] if seq and 0 < pos <= len(seq) else ""
            t = (f"{prot} — {mod} {aa}{pos}<br>Position in protein: {pos}"
                 f"<br>PSMs: {int(r['count'])}")
            if pos in pos_stat:
                t += "<br>" + pos_stat[pos]
            texts.append(t)
            rim.append(pos in sig_pos)
        rim = np.array(rim, dtype=bool)
        # marker area proportional to the PSM count
        sizes = 9 + 9 * np.sqrt(g["count"] / ymax)
        fig.add_trace(go.Scatter(
            x=g["position_in_protein"], y=g["count"], mode="markers",
            name=str(mod), legendgroup=str(mod), text=texts,
            marker=dict(size=sizes, color=color,
                        symbol="circle",
                        line=dict(color=np.where(rim, "#d62828", "white"),
                                  width=np.where(rim, 2.5, 1.0))),
            hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    if sig_pos:  # legend entry explaining the red rim
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="significant site",
            legendgroup="__sig__",
            marker=dict(size=11, color="rgba(0,0,0,0)",
                        line=dict(color="#d62828", width=2.5)),
            hoverinfo="skip"), row=1, col=1)

    # ---- full-length peptide-coverage bar (one cell per residue) ----
    # Multiplicity coverage as in the legacy web server: peptides enter WITH
    # duplicates (one entry per PSM); density = covering PSM count / total PSMs.
    pep_col = "peptide_y" if "peptide_y" in sub.columns else None
    if seq and pep_col:
        peps = [re.sub(r"[^A-Z]", "", str(p)) for p in sub[pep_col].dropna()]
        peps = [p for p in peps if p]
        if peps:
            cov = _peptide_coverage(seq, peps)
            dens = cov / max(len(peps), 1)
            fig.add_trace(go.Heatmap(
                z=[dens.tolist()], x=list(range(1, len(seq) + 1)),
                y=["coverage"], colorscale=_COVERAGE_SCALE, zmin=0,
                zmax=max(float(dens.max()), 1e-9), showscale=True,
                colorbar=dict(title=dict(text="Degree of<br>protein<br>coverage",
                                         font=dict(size=11)),
                              thickness=12, len=0.26, y=0.13,
                              tickfont=dict(size=10)),
                customdata=[[[aa, int(c)] for aa, c in zip(seq, cov)]],
                hovertemplate=("Residue: %{customdata[0]}%{x}<br>"
                               "Covering PSMs: %{customdata[1]}<br>"
                               "Coverage: %{z:.3f}<extra></extra>"),
            ), row=2, col=1)
            # outline of the full-length protein bar (phosphosite.org style)
            fig.add_shape(type="rect", xref="x2", yref="y2 domain",
                          x0=0.5, x1=len(seq) + 0.5, y0=0, y1=1,
                          line=dict(color="#334155", width=1.2),
                          fillcolor="rgba(0,0,0,0)", layer="above")
        else:
            fig.update_xaxes(visible=False, row=2, col=1)
            fig.update_yaxes(visible=False, row=2, col=1)
    else:
        fig.update_xaxes(visible=False, row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)

    fig.update_layout(title=f"Modification landscape — {prot}",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                  xanchor="center", x=0.5),
                      **{**_BASE_LAYOUT,
                         "margin": dict(t=90, b=50, l=70, r=90)})
    fig.update_xaxes(title_text="Position in protein", row=2, col=1)
    fig.update_yaxes(title_text="PSM count", row=1, col=1)
    return fig


def protein_site_payload(annot: pd.DataFrame, stats: pd.DataFrame,
                         fasta_seqs: dict, colors: dict,
                         alpha: float = 0.05, logfc_thr: float = 1.0):
    """Serialize the per-protein landscape data for ALL identified proteins
    to compact JSON; the report renders any selected protein client-side
    (search box + top-significant quick list), so the static report is no
    longer limited to a pre-rendered top-K.

    Payload size is roughly 1-2 KB per protein (sequence + site list +
    peptide intervals), so even a few thousand proteins add only a few MB —
    acceptable for a single-file report.

    Per protein: L=length, seq=amino-acid sequence ("" without FASTA),
    sites=[[pos, modIdx, psmCount, significant, statsHoverHtml], ...],
    peps=[[start, end, psmCount], ...] (peptide occurrences WITH
    multiplicities; the JS side derives the multiplicity coverage profile).
    """
    if annot.empty or "id_prot" not in annot.columns:
        return None
    need = {"id_prot", "position_in_protein", "Modification"}
    if not need.issubset(annot.columns):
        return None
    d = annot[annot["Modification"] != "reference"].copy()
    d = d.dropna(subset=["position_in_protein"])
    if d.empty:
        return None
    d["Mods"] = d["Modification"].astype(str).str.split("@").str[0]
    d["pos"] = pd.to_numeric(d["position_in_protein"], errors="coerce")
    d = d.dropna(subset=["pos"])
    d["pos"] = d["pos"].astype(int)

    mods = sorted(d["Mods"].unique())
    mod_idx = {m: i for i, m in enumerate(mods)}

    # differential stats per (protein, position): significance flag + hover
    stat_map = {}
    stat_cols = {"protein", "position", "contrast", "logFC", "adj.P.Val"}
    if stats is not None and not stats.empty and stat_cols.issubset(stats.columns):
        st = stats.dropna(subset=["protein", "position"])
        for (prot, pos), g in st.groupby(
                [st["protein"].astype(str), st["position"].astype(int)]):
            lines, hit = [], False
            for _, r in g.iterrows():
                if pd.isna(r["adj.P.Val"]) or pd.isna(r["logFC"]):
                    continue
                sig = (r["adj.P.Val"] < alpha) and (abs(r["logFC"]) >= logfc_thr)
                hit |= bool(sig)
                lines.append(f"{r['contrast']}: logFC={r['logFC']:.2f}, "
                             f"adj.P={r['adj.P.Val']:.2g}" + (" *" if sig else ""))
            if lines:
                stat_map[(prot, pos)] = (hit, "<br>".join(lines))

    # ranking: most significant proteins first (for the default view)
    rank = {}
    if stats is not None and not stats.empty and "adj.P.Val" in stats.columns:
        s = stats.dropna(subset=["adj.P.Val"])
        rank = s.groupby(s["protein"].astype(str))["adj.P.Val"].min().to_dict()

    fasta_seqs = fasta_seqs or {}
    pep_col = "peptide_y" if "peptide_y" in d.columns else None
    proteins = {}
    for prot, g in d.groupby(d["id_prot"].astype(str)):
        cnt = g.groupby(["pos", "Mods"]).size()
        sites = []
        for (pos, mod), c in cnt.items():
            hit, stat_txt = stat_map.get((prot, pos), (False, ""))
            sites.append([int(pos), mod_idx[mod], int(c), bool(hit), stat_txt])
        sites.sort(key=lambda r: r[0])

        seq = fasta_seqs.get(prot, "")
        peps = []
        if seq and pep_col:
            pep_series = g[pep_col].dropna().astype(str).map(
                lambda p: re.sub(r"[^A-Z]", "", p))
            pep_series = pep_series[pep_series.str.len() > 0]
            for pep, c in pep_series.value_counts().items():
                start = 0
                while True:
                    i = seq.find(pep, start)
                    if i == -1:
                        break
                    peps.append([i + 1, i + len(pep), int(c)])
                    start = i + 1
        L = len(seq) if seq else int(max([s[0] for s in sites] + [1]))
        proteins[prot] = {"L": L, "seq": seq, "sites": sites, "peps": peps}

    if not proteins:
        return None
    order = sorted(proteins.keys(),
                   key=lambda p: (rank.get(p, float("inf")), p))
    payload = {"mods": mods,
               "colors": {m: colors.get(m, "#4361ee") for m in mods},
               "order": order, "proteins": proteins}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def qc_figures(stats: pd.DataFrame, weights: pd.DataFrame):
    figs = {}
    if not stats.empty and "n_psm" in stats.columns:
        n = pd.to_numeric(stats["n_psm"], errors="coerce").dropna()
        if not n.empty:
            figs["n_psm"] = go.Figure(go.Histogram(
                x=n.clip(upper=n.quantile(0.99)), nbinsx=50,
                marker_color="#2a9d8f", marker_line=dict(color="white", width=0.5)))
            figs["n_psm"].update_layout(title="PSMs per site (99th percentile cut)",
                                        xaxis_title="n_PSM", yaxis_title="Sites",
                                        **_BASE_LAYOUT)
    if not weights.empty:
        vals = weights.to_numpy().ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            figs["prec"] = go.Figure(go.Histogram(
                x=np.log10(vals), nbinsx=60, marker_color="#7b2cbf",
                marker_line=dict(color="white", width=0.5)))
            figs["prec"].update_layout(title="Aggregation precision per site-sample (log10)",
                                       xaxis_title="log10 precision", yaxis_title="Count",
                                       **_BASE_LAYOUT)
    return figs


def pvalue_histogram_figure(stats: pd.DataFrame):
    """P-value distribution per contrast — a calibration check: under the
    null the histogram is flat; a spike near 0 marks true signal, a spike
    near 1 or a U-shape hints at model problems."""
    if stats.empty or "P.Value" not in stats.columns:
        return None
    d = stats.drop_duplicates(subset=["mod_family", "contrast", "site"])
    d = d.dropna(subset=["P.Value"])
    if d.empty:
        return None
    contrasts = sorted(d["contrast"].astype(str).unique())
    n = len(contrasts)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=contrasts,
                        horizontal_spacing=0.07, vertical_spacing=0.12)
    palette = ["#1d4e89", "#2a9d8f", "#e07a5f", "#7b2cbf", "#00b4d8",
               "#f3722c", "#4361ee", "#90be6d"]
    for i, con in enumerate(contrasts):
        sub = d[d["contrast"].astype(str) == con]
        fig.add_trace(go.Histogram(
            x=sub["P.Value"], xbins=dict(start=0, end=1, size=0.05),
            marker_color=palette[i % len(palette)],
            marker_line=dict(color="white", width=0.5),
            hovertemplate="p ∈ [%{x:.2f}, %{x:.2f}+0.05): %{y} sites<extra></extra>"),
            row=i // cols + 1, col=i % cols + 1)
        fig.update_xaxes(range=[0, 1], row=i // cols + 1, col=i % cols + 1)
    fig.update_layout(title="P-value distribution per contrast "
                            "<sup>flat = null-like, spike at 0 = signal</sup>",
                      showlegend=False,
                      height=max(300, 280 * rows),
                      **_BASE_LAYOUT)
    fig.update_annotations(font=dict(size=12, color="#1d4e89"))
    for i in range(n):
        fig.update_xaxes(title_text="p-value", row=i // cols + 1, col=i % cols + 1)
        fig.update_yaxes(title_text="sites", row=i // cols + 1, col=1)
    return fig


def ma_figure(stats: pd.DataFrame, expr: pd.DataFrame,
              alpha: float = 0.05, logfc_thr: float = 1.0):
    """MA-style plots: logFC vs mean log2 abundance of the site, significant
    sites highlighted. Low-abundance-driven hits show up as a red cloud on
    the left and deserve extra caution."""
    if stats.empty or "logFC" not in stats.columns:
        return None
    d = stats.drop_duplicates(subset=["mod_family", "contrast", "site"])
    d = d.dropna(subset=["logFC"])
    if d.empty:
        return None
    abund = expr.mean(axis=1) if not expr.empty else pd.Series(dtype=float)
    d["abund"] = d["site"].map(abund)
    d = d.dropna(subset=["abund"])
    if d.empty:
        return None
    contrasts = sorted(d["contrast"].astype(str).unique())
    n = len(contrasts)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=contrasts,
                        horizontal_spacing=0.07, vertical_spacing=0.12)
    for i, con in enumerate(contrasts):
        sub = d[d["contrast"].astype(str) == con]
        sig = ((sub["adj.P.Val"] < alpha) &
               (sub["logFC"].abs() >= logfc_thr)).fillna(False)
        r, c = i // cols + 1, i % cols + 1
        for mask, name, color, size in (
                (~sig, "not significant", "rgba(148,163,184,0.5)", 5),
                (sig, "significant", "rgba(230,57,70,0.85)", 7)):
            ss = sub[mask]
            if ss.empty:
                continue
            fig.add_trace(go.Scatter(
                x=ss["abund"], y=ss["logFC"], mode="markers", name=name,
                legendgroup=name, showlegend=(i == 0),
                marker=dict(size=size, color=color),
                text=ss["site"],
                hovertemplate="%{text}<br>mean log2 abundance: %{x:.2f}"
                              "<br>log2FC: %{y:.2f}<extra></extra>"),
                row=r, col=c)
    fig.update_layout(title="MA plots — logFC vs mean abundance "
                            "<sup>red = passing the report cutoffs</sup>",
                      height=max(320, 300 * rows),
                      legend=dict(orientation="h", y=1.06, x=0.5,
                                  xanchor="center"),
                      **_BASE_LAYOUT)
    fig.update_annotations(font=dict(size=12, color="#1d4e89"))
    for i in range(n):
        fig.update_xaxes(title_text="mean log2 abundance",
                         row=i // cols + 1, col=i % cols + 1)
        fig.update_yaxes(title_text="log2FC", row=i // cols + 1, col=1)
    return fig


def permutation_bar_figure(perm: pd.DataFrame):
    """Observed vs permutation-null hit counts per family × contrast."""
    if perm.empty or "obs_hits" not in perm.columns or "perm_mean" not in perm.columns:
        return None
    d = perm.copy()
    if "mod_family" in d.columns:
        d["label"] = d["mod_family"].astype(str) + ": " + d["contrast"].astype(str)
    else:
        d["label"] = d["contrast"].astype(str)
    d = d.sort_values("label")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["label"], y=d["obs_hits"], name="observed hits",
                         marker_color="#e63946",
                         hovertemplate="%{x}<br>observed: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(x=d["label"], y=d["perm_mean"],
                         name="permutation null (mean)",
                         marker_color="#94a3b8",
                         hovertemplate="%{x}<br>null mean: %{y:.1f}<extra></extra>"))
    for _, r in d.iterrows():
        if "perm_pval" in d.columns and pd.notna(r.get("perm_pval")):
            fig.add_annotation(x=r["label"], y=max(r["obs_hits"], 1),
                               text=f"p={r['perm_pval']:.3g}",
                               showarrow=False, yanchor="bottom",
                               font=dict(size=10, color="#0f2540"))
    fig.update_layout(title="Permutation validation — observed hits vs null",
                      barmode="group",
                      xaxis_title="", yaxis_title="hits",
                      legend=dict(orientation="h", y=1.08, x=0.5,
                                  xanchor="center"),
                      **{**_BASE_LAYOUT,
                         "margin": dict(t=90, b=110, l=70, r=30)})
    fig.update_xaxes(tickangle=-35)
    return fig


# ---------------------------------------------------------------------------
# Database annotation figures (iPTMnet / dbPTM / SIGNOR site-level caches)
# ---------------------------------------------------------------------------

_DB_COLORS = {"iPTMnet": "#4361ee", "dbPTM": "#2a9d8f", "SIGNOR": "#e07a5f"}


def db_annotation_figures(db_ann: dict, colors: dict):
    """Figures summarizing how the identified sites map to the PTM databases.

    Returns an ordered dict of figures:
      coverage — per-DB bars: exact matches and window (±7 aa) matches;
      by_mod   — per modification type, % of sites known to each DB;
      signor   — SIGNOR effect-on-protein distribution (site- vs
                 protein-level evidence);
      regulators — top SIGNOR regulators acting on the measured sites;
      dbptm_evidence — dbPTM literature depth (PMIDs per site).
    """
    figs = {}
    present = [db for db in ("iPTMnet", "dbPTM", "SIGNOR") if db in db_ann]
    if not present:
        return figs

    # ---- per-DB coverage ----
    labels, exact, window, totals = [], [], [], []
    for db in present:
        df = db_ann[db]
        _, in_col, perh_col = DB_ANNOTATION_FILES[db]
        if in_col not in df.columns:
            continue
        n = len(df)
        ex = int(_truthy(df[in_col]).sum())
        win = int(_truthy(df[perh_col]).sum()) if perh_col and perh_col in df.columns else 0
        labels.append(db)
        exact.append(ex)
        window.append(win)
        totals.append(n)
    if labels:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=exact, name="exact site match",
            marker_color="#1d4e89",
            text=[f"{e} ({e / max(t, 1):.0%})" for e, t in zip(exact, totals)],
            textposition="outside",
            hovertemplate="%{x}: %{y} exact matches<extra></extra>"))
        if any(window):
            fig.add_trace(go.Bar(
                x=labels, y=window, name="window match (±7 aa)",
                marker_color="#90be6d",
                text=[f"{w} ({w / max(t, 1):.0%})" for w, t in zip(window, totals)],
                textposition="outside",
                hovertemplate="%{x}: %{y} window matches<extra></extra>"))
        fig.update_layout(
            title="Identified sites found in the PTM databases<br>"
                  "<sup>percentages are relative to all tested unique sites; "
                  "window match = known PTM within ±7 residues</sup>",
            barmode="group", yaxis_title="sites",
            legend=dict(orientation="h", y=1.14, x=0.5, xanchor="center"),
            **{**_BASE_LAYOUT, "margin": dict(t=110, b=50, l=70, r=30)})
        figs["coverage"] = fig

    # ---- per modification type ----
    mod_rates = {}
    for db in present:
        df = db_ann[db]
        _, in_col, _ = DB_ANNOTATION_FILES[db]
        if "mods" not in df.columns or in_col not in df.columns:
            continue
        g = df.groupby("mods")[in_col].agg(["count", lambda s: _truthy(s).mean()])
        g.columns = ["n", "rate"]
        mod_rates[db] = g
    if mod_rates:
        all_mods = pd.concat([g["n"] for g in mod_rates.values()])
        top_mods = all_mods.groupby(level=0).sum().nlargest(8).index.tolist()
        fig = go.Figure()
        for db, g in mod_rates.items():
            g = g.reindex(top_mods)
            fig.add_trace(go.Bar(
                x=top_mods, y=(g["rate"] * 100).round(1), name=db,
                marker_color=_DB_COLORS.get(db, "#4361ee"),
                customdata=g["n"].fillna(0).astype(int),
                hovertemplate=("%{x} — " + db + ": %{y:.1f}% of %{customdata} "
                               "sites<extra></extra>")))
        fig.update_layout(
            title="Database coverage by modification type<br>"
                  "<sup>% of tested sites with an exact DB record</sup>",
            barmode="group", yaxis_title="sites in DB (%)",
            xaxis_title="Modification",
            legend=dict(orientation="h", y=1.14, x=0.5, xanchor="center"),
            **{**_BASE_LAYOUT, "margin": dict(t=110, b=50, l=70, r=30)})
        figs["by_mod"] = fig

    # ---- SIGNOR: effect direction ----
    if "SIGNOR" in db_ann:
        sdf = db_ann["SIGNOR"]
        if "signor_effect_on_protein" in sdf.columns:
            ann = sdf[_truthy(sdf["in_SIGNOR"]) &
                      sdf["signor_effect_on_protein"].notna()].copy()
            if not ann.empty:
                eff_map = {"activate": "activates the protein",
                           "inhibit": "inhibits the protein",
                           "conflicting": "conflicting evidence",
                           "unknown": "unknown direction"}
                ann["effect"] = ann["signor_effect_on_protein"].map(eff_map).fillna(
                    ann["signor_effect_on_protein"])
                ev = (ann["signor_evidence"].map(
                    {"site": "site-level", "protein": "protein-level"})
                    .fillna("site-level"))
                ann["evidence"] = ev
                cnt = ann.groupby(["effect", "evidence"]).size().reset_index(name="n")
                eff_colors = {"activates the protein": "#2fa36b",
                              "inhibits the protein": "#d9534f",
                              "conflicting evidence": "#ffb703",
                              "unknown direction": "#97a3ae"}
                fig = go.Figure()
                for evi, hatch in (("site-level", ""), ("protein-level", "/")):
                    sub = cnt[cnt["evidence"] == evi]
                    if sub.empty:
                        continue
                    fig.add_trace(go.Bar(
                        x=sub["effect"], y=sub["n"], name=evi,
                        marker_color=[eff_colors.get(e, "#97a3ae")
                                      for e in sub["effect"]],
                        marker_pattern_shape=hatch,
                        marker_line=dict(color="white", width=1),
                        text=sub["n"], textposition="outside",
                        hovertemplate="%{x} (%{y} sites, " + evi +
                                      " evidence)<extra></extra>"))
                fig.update_layout(
                    title="SIGNOR: effect of the measured PTMs on their proteins<br>"
                          "<sup>site-level = the exact modified residue is curated; "
                          "protein-level = another site of the same protein</sup>",
                    barmode="stack", yaxis_title="sites",
                    legend=dict(orientation="h", y=1.16, x=0.5, xanchor="center"),
                    **{**_BASE_LAYOUT, "margin": dict(t=115, b=50, l=70, r=30)})
                figs["signor"] = fig

        # ---- SIGNOR: top regulators ----
        if "signor_regulators" in sdf.columns:
            regs = (sdf["signor_regulators"].dropna().astype(str)
                    .str.split(";").explode().str.strip())
            regs = regs[regs != ""]
            if not regs.empty:
                top = regs.value_counts().head(15).iloc[::-1]
                fig = go.Figure(go.Bar(
                    y=top.index, x=top.values, orientation="h",
                    marker_color="#e07a5f",
                    marker_line=dict(color="white", width=1),
                    hovertemplate="%{y}: acts on %{x} measured sites<extra></extra>"))
                fig.update_layout(
                    title="SIGNOR: top regulators of the measured sites",
                    xaxis_title="regulated measured sites",
                    **{**_BASE_LAYOUT,
                       "margin": dict(t=60, b=50, l=130, r=30)})
                figs["regulators"] = fig

    # ---- dbPTM: literature depth ----
    if "dbPTM" in db_ann:
        ddf = db_ann["dbPTM"]
        if "dbptm_pmids" in ddf.columns:
            n_pmids = (ddf.loc[_truthy(ddf.get("in_dbPTM", pd.Series(False, index=ddf.index))),
                             "dbptm_pmids"]
                         .dropna().astype(str)
                         .map(lambda s: len([p for p in s.split(";") if p.strip()])))
            n_pmids = n_pmids[n_pmids > 0]
            if not n_pmids.empty:
                fig = go.Figure(go.Histogram(
                    x=n_pmids.clip(upper=n_pmids.quantile(0.99)),
                    nbinsx=30, marker_color="#2a9d8f",
                    marker_line=dict(color="white", width=0.5),
                    hovertemplate="%{x} PMIDs: %{y} sites<extra></extra>"))
                fig.update_layout(
                    title="dbPTM evidence depth — publications per site "
                          "<sup>99th percentile cut</sup>",
                    xaxis_title="PMIDs supporting the site",
                    yaxis_title="sites",
                    **_BASE_LAYOUT)
                figs["dbptm_evidence"] = fig

    return figs


def signor_network_iframe(output_dir: str):
    """Embed the fully offline SIGNOR network page into the report via an
    srcdoc iframe (keeps the report a single portable file)."""
    path = os.path.join(output_dir, "signor_network.html")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = fh.read()
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return None
    if "<html" not in doc.lower():
        return None
    return ('<iframe class="signor-frame" srcdoc="' + _esc(doc) + '" '
            'title="SIGNOR causal PTM network" loading="lazy"></iframe>')


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
:root { --navy:#0f2540; --blue:#1d4e89; --teal:#00b4d8; --amber:#ffb703;
        --red:#e63946; --green:#2a9d8f; --purple:#7b2cbf; --bg:#f1f5f9;
        --ink:#334155; --line:#e2e8f0; }
* { box-sizing:border-box; }
body { font-family:'Segoe UI', Arial, sans-serif; margin:0; background:var(--bg);
       color:var(--ink); }
header { background:linear-gradient(120deg, var(--navy) 0%, var(--blue) 55%,
         var(--teal) 130%); color:#fff; padding:26px 34px 22px; }
header h1 { margin:0; font-size:1.55em; letter-spacing:.3px; }
header h1 .accent { color:var(--amber); }
header p { color:#cbd5e1; margin:6px 0 0; font-size:.85em; }
nav { position:sticky; top:0; z-index:50; background:#fff; box-shadow:0 1px 6px
      rgba(15,37,64,.15); padding:9px 30px; display:flex; gap:18px; flex-wrap:wrap; }
nav a { color:var(--blue); text-decoration:none; font-size:.86em; font-weight:600;
        padding:3px 2px; border-bottom:2px solid transparent; }
nav a:hover { border-bottom-color:var(--amber); color:var(--navy); }
section { background:#fff; margin:20px auto; padding:20px 26px 24px; max-width:1240px;
          border-radius:14px; box-shadow:0 2px 10px rgba(15,37,64,.08); }
h2 { margin:2px 0 16px; font-size:1.18em; color:var(--navy); padding-left:12px;
     border-left:5px solid var(--amber); }
h3 { color:var(--blue); font-size:1em; margin:24px 0 6px; }
.cards { display:flex; gap:14px; flex-wrap:wrap; }
.card { flex:1; min-width:160px; border-radius:12px; padding:16px 14px; text-align:center;
        color:#fff; box-shadow:0 2px 8px rgba(15,37,64,.18); }
.card .num { font-size:2em; font-weight:700; }
.card .lbl { font-size:.78em; opacity:.92; text-transform:uppercase; letter-spacing:.5px; }
.card.c0 { background:linear-gradient(140deg,#0f2540,#1d4e89); }
.card.c1 { background:linear-gradient(140deg,#0077b6,#00b4d8); }
.card.c2 { background:linear-gradient(140deg,#2a9d8f,#43aa8b); }
.card.c3 { background:linear-gradient(140deg,#7b2cbf,#9d4edd); }
.card.c4 { background:linear-gradient(140deg,#d62828,#f3722c); }
.badges { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.badge { display:inline-block; padding:3px 11px; border-radius:999px; font-size:.78em;
         font-weight:600; background:#eef2f7; color:var(--navy); border:1px solid var(--line); }
.badge-red { background:#fde8e9; color:var(--red); border-color:#f5c2c6; }
.controls { display:flex; gap:22px; flex-wrap:wrap; align-items:center; margin:4px 0 12px;
            padding:10px 14px; background:#f8fafc; border:1px solid var(--line);
            border-radius:10px; }
.controls label { font-size:.84em; font-weight:600; color:var(--navy); display:flex;
                  gap:7px; align-items:center; }
.controls select, .controls input { padding:4px 8px; border:1px solid #cbd5e1;
        border-radius:7px; font-size:.95em; background:#fff; color:var(--ink); }
.controls input[type=number] { width:86px; }
.plot { width:100%; }
.fig-row { display:flex; gap:18px; flex-wrap:wrap; margin-top:14px; }
.fig-row > div { flex:1; min-width:min(420px, 100%); }
.note { color:#64748b; font-size:.85em; }
.table-wrap { overflow-x:auto; max-width:100%; }
table.dtable { border-collapse:collapse; width:100%; font-size:.86em; }
table.dtable thead th { background:var(--navy); color:#fff; padding:8px 10px;
     position:sticky; top:0; cursor:pointer; user-select:none; white-space:nowrap; }
table.dtable thead th.asc::after  { content:' \\25B2'; font-size:.7em; color:var(--amber); }
table.dtable thead th.desc::after { content:' \\25BC'; font-size:.7em; color:var(--amber); }
table.dtable tbody td { padding:6px 10px; border-bottom:1px solid var(--line);
     white-space:nowrap; }
table.dtable tbody tr:nth-child(even) { background:#f8fafc; }
table.dtable tbody tr:hover { background:#fff7e0; }
.table-controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap;
                  margin-bottom:10px; }
.table-controls input[type=search] { padding:6px 12px; border:1px solid #cbd5e1;
     border-radius:8px; width:260px; font-size:.95em; }
.table-controls select { padding:5px 8px; border:1px solid #cbd5e1; border-radius:7px; }
.table-controls button, .pager button { padding:5px 12px; border:1px solid #cbd5e1;
     background:#fff; border-radius:7px; cursor:pointer; font-size:.85em; color:var(--navy); }
.table-controls button:hover, .pager button:hover:not(:disabled) { background:#eef2f7;
     border-color:var(--teal); }
.pager { display:flex; gap:6px; margin-top:10px; }
.pager button:disabled { opacity:.4; cursor:default; }
.signor-frame { width:100%; height:880px; border:1px solid var(--line);
     border-radius:10px; background:#fff; }
#protSearch { width:240px; }
footer { text-align:center; color:#94a3b8; font-size:.8em; padding:18px 0 26px; }
"""

_DIFF_JS = """
const DE_DATA = __DE_JSON__;
(function () {
  // One shared dataset drives BOTH the volcano plot and the significant-sites
  // table, so the table always reflects the cutoffs chosen for the volcano.
  const COLS = [
    ["s", "site"], ["m", "modification"], ["g", "protein"], ["o", "position"],
    ["c", "contrast"], ["x", "logFC"], ["t", "t"], ["p", "P.Value"],
    ["a", "adj.P.Val"], ["n", "n_psm"], ["u", "status"]];
  const NUM = new Set(["o", "x", "t", "p", "a", "n"]);
  const HOVER = "Site: %{customdata[0]}<br>Protein: %{customdata[1]}<br>Mod: %{customdata[2]}"
              + "<br>log2FC: %{x:.3f}<br>-log10(adj.P): %{y:.3f}<extra></extra>";

  const tbody = document.querySelector("#deTable tbody");
  const heads = Array.from(document.querySelectorAll("#deTable thead th"));
  const search = document.getElementById("deSearch");
  const sizeSel = document.getElementById("dePageSize");
  const csvBtn = document.getElementById("deCsv");
  const info = document.getElementById("deInfo");
  const pager = document.getElementById("dePager");
  let sigRows = [], sortIdx = -1, sortDir = 1, page = 0,
      pageSize = parseInt(sizeSel.value, 10), query = "";

  function fmt(v, k) {
    if (v === null || v === undefined) return "";
    if (!NUM.has(k) || k === "o" || k === "n") return String(v);
    return String(parseFloat(Number(v).toPrecision(4)));
  }

  function selection() {
    return {
      fam: document.getElementById("vc_family").value,
      con: document.getElementById("vc_contrast").value,
      alpha: parseFloat(document.getElementById("vc_alpha").value) || 1,
      fc: parseFloat(document.getElementById("vc_fc").value) || 0};
  }

  function renderTable() {
    let rows = sigRows;
    if (query) rows = rows.filter(r =>
      (r.s + " " + r.g + " " + r.m + " " + r.u).toLowerCase().includes(query));
    if (sortIdx >= 0) {
      const k = COLS[sortIdx][0], numeric = NUM.has(k);
      rows = rows.slice().sort((a, b) => {
        const av = a[k], bv = b[k];
        if (av === null && bv === null) return 0;
        if (av === null) return 1;   // missing values always last
        if (bv === null) return -1;
        const c = numeric ? av - bv : String(av).localeCompare(String(bv));
        return c * sortDir;
      });
    }
    const total = rows.length;
    const pages = Math.max(1, Math.ceil(total / pageSize));
    page = Math.min(Math.max(page, 0), pages - 1);
    const frag = document.createDocumentFragment();
    rows.slice(page * pageSize, (page + 1) * pageSize).forEach(r => {
      const tr = document.createElement("tr");
      COLS.forEach(([k]) => {
        const td = document.createElement("td");
        td.textContent = fmt(r[k], k);
        tr.appendChild(td);
      });
      frag.appendChild(tr);
    });
    tbody.innerHTML = ""; tbody.appendChild(frag);
    info.textContent = total + " DE sites \u00b7 page " + (page + 1) + "/" + pages;
    pager.innerHTML = "";
    const mk = (t, fn, dis) => {
      const b = document.createElement("button");
      b.textContent = t; b.disabled = !!dis; b.onclick = fn; pager.appendChild(b);
    };
    mk("\u00ab First", () => { page = 0; renderTable(); }, page === 0);
    mk("\u2039 Prev", () => { page--; renderTable(); }, page === 0);
    mk("Next \u203a", () => { page++; renderTable(); }, page >= pages - 1);
    mk("Last \u00bb", () => { page = pages - 1; renderTable(); }, page >= pages - 1);
  }

  function update() {
    const sel = selection();
    const ythr = -Math.log10(Math.max(sel.alpha, 1e-300));
    const ns = {x: [], y: [], cd: []}, sg = {x: [], y: [], cd: []};
    sigRows = [];
    let nTested = 0;
    for (const r of DE_DATA) {
      if (sel.fam !== "__ALL__" && r.f !== sel.fam) continue;
      if (r.c !== sel.con) continue;                 // ONE pairwise comparison
      if (r.a === null || r.a <= 0 || r.x === null) continue;
      nTested++;
      const y = -Math.log10(Math.max(r.a, 1e-300));
      const hit = (r.a < sel.alpha && Math.abs(r.x) >= sel.fc);
      const tr = hit ? sg : ns;
      tr.x.push(r.x); tr.y.push(y); tr.cd.push([r.s, r.g, r.m]);
      if (hit) sigRows.push(r);
    }
    const traces = [
      {x: ns.x, y: ns.y, customdata: ns.cd, type: "scatter", mode: "markers",
       name: "not significant", marker: {size: 6, color: "rgba(148,163,184,0.45)"},
       hovertemplate: HOVER},
      {x: sg.x, y: sg.y, customdata: sg.cd, type: "scatter", mode: "markers",
       name: "significant", marker: {size: 7, color: "rgba(230,57,70,0.85)"},
       hovertemplate: HOVER}];
    const line = {type: "line", line: {dash: "dash", color: "#94a3b8", width: 1}};
    const layout = {
      template: "plotly_white", height: 560,
      title: {text: "Volcano plot — " + sel.con, font: {size: 15, color: "#0f2540"},
              y: 0.985, yref: "container", yanchor: "top", x: 0.5, xanchor: "center"},
      xaxis: {title: "log2 fold change"}, yaxis: {title: "-log10(adj. p-value)"},
      legend: {orientation: "h", y: 1.10, x: 0.5, xanchor: "center"},
      margin: {t: 110, b: 50, l: 70, r: 30},
      shapes: [
        Object.assign({x0: -sel.fc, x1: -sel.fc, yref: "paper", y0: 0, y1: 1}, line),
        Object.assign({x0: sel.fc, x1: sel.fc, yref: "paper", y0: 0, y1: 1}, line),
        Object.assign({y0: ythr, y1: ythr, xref: "paper", x0: 0, x1: 1}, line)]};
    Plotly.react("volcano_plot", traces, layout,
                 {responsive: true, displaylogo: false});
    document.getElementById("vc_hits").textContent =
      sg.x.length + " significant of " + nTested + " sites";
    page = 0;
    renderTable();
  }

  ["vc_family", "vc_contrast", "vc_alpha", "vc_fc"].forEach(
    id => document.getElementById(id).addEventListener("change", update));
  heads.forEach((th, i) => {
    th.addEventListener("click", () => {
      sortDir = (sortIdx === i) ? -sortDir : 1; sortIdx = i;
      heads.forEach(h => h.classList.remove("asc", "desc"));
      th.classList.add(sortDir > 0 ? "asc" : "desc");
      renderTable();
    });
  });
  search.addEventListener("input", e => {
    query = e.target.value.toLowerCase(); page = 0; renderTable(); });
  sizeSel.addEventListener("change", e => {
    pageSize = parseInt(e.target.value, 10); page = 0; renderTable(); });
  csvBtn.addEventListener("click", () => {
    const esc = s => '"' + String(s).replace(/"/g, '""') + '"';
    const hdr = COLS.map(c => esc(c[1])).join(",");
    const lines = sigRows.map(r => COLS.map(([k]) => esc(fmt(r[k], k))).join(","));
    const blob = new Blob([hdr + "\\n" + lines.join("\\n")], {type: "text/csv"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "openptmfinder_DE_sites.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  update();
})();
"""

_PROTEIN_JS = """
const PROT_DATA = __PROT_JSON__;
(function () {
  // Client-side renderer for the per-protein modification landscape: ALL
  // identified proteins are embedded in PROT_DATA, the figure for the
  // selected protein is built on the fly (lollipop of PSM counts per site +
  // full-length coverage bar colored by per-residue PSM multiplicity).
  const SCALE = [[0, "#e8e8e8"], [0.02, "#fee5d9"], [0.25, "#fcae91"],
                 [0.5, "#fb6a4a"], [0.75, "#de2d26"], [1, "#a50f15"]];
  const input = document.getElementById("protSearch");
  const quick = document.getElementById("protQuick");
  const info = document.getElementById("protInfo");

  const dl = document.getElementById("protList");
  for (const p of PROT_DATA.order) {
    const o = document.createElement("option");
    o.value = p; dl.appendChild(o);
  }
  for (const p of PROT_DATA.order.slice(0, 10)) {
    const o = document.createElement("option");
    o.value = p; o.textContent = p; quick.appendChild(o);
  }

  function covProfile(P) {
    // multiplicity coverage (legacy web-server method): peptides enter WITH
    // duplicates, density = covering PSM count / total PSM count
    const L = P.L, cov = new Float64Array(L + 1);
    let total = 0;
    for (const iv of P.peps) {
      total += iv[2];
      for (let i = iv[0]; i <= iv[1] && i <= L; i++) cov[i] += iv[2];
    }
    const dens = new Array(L), covA = new Array(L);
    for (let i = 0; i < L; i++) {
      covA[i] = cov[i + 1];
      dens[i] = total > 0 ? cov[i + 1] / total : 0;
    }
    return {cov: covA, dens: dens, total: total};
  }

  function render(prot) {
    const P = PROT_DATA.proteins[prot];
    if (!P) {
      info.textContent = "protein not found";
      return;
    }
    const mods = PROT_DATA.mods, colors = PROT_DATA.colors;
    let nSig = 0, ymax = 1;
    for (const s of P.sites) { ymax = Math.max(ymax, s[2]); if (s[3]) nSig++; }
    const byMod = new Map();
    for (const s of P.sites) {
      const m = mods[s[1]];
      if (!byMod.has(m)) byMod.set(m, []);
      byMod.get(m).push(s);
    }
    const traces = [];
    for (const entry of byMod) {
      const m = entry[0], arr = entry[1];
      const col = colors[m] || "#4361ee";
      const sx = [], sy = [], mx = [], my = [], text = [], sizes = [],
            rimC = [], rimW = [];
      for (const s of arr) {
        const pos = s[0], count = s[2], sig = s[3], stat = s[4];
        sx.push(pos, pos, null); sy.push(0, count, null);
        mx.push(pos); my.push(count);
        const aa = (P.seq && pos <= P.seq.length) ? P.seq[pos - 1] : "";
        let t = prot + " \u2014 " + m + " " + aa + pos
              + "<br>Position in protein: " + pos + "<br>PSMs: " + count;
        if (stat) t += "<br>" + stat;
        text.push(t);
        sizes.push(9 + 9 * Math.sqrt(count / ymax));
        rimC.push(sig ? "#d62828" : "white");
        rimW.push(sig ? 2.5 : 1);
      }
      traces.push({x: sx, y: sy, mode: "lines", showlegend: false,
                   legendgroup: m, line: {width: 1.5, color: col},
                   hoverinfo: "skip", xaxis: "x", yaxis: "y"});
      traces.push({x: mx, y: my, mode: "markers", name: m, legendgroup: m,
                   text: text,
                   marker: {size: sizes, color: col,
                            line: {color: rimC, width: rimW}},
                   hovertemplate: "%{text}<extra></extra>",
                   xaxis: "x", yaxis: "y"});
    }
    if (nSig > 0) {
      traces.push({x: [null], y: [null], mode: "markers",
                   name: "significant site", legendgroup: "__sig__",
                   marker: {size: 11, color: "rgba(0,0,0,0)",
                            line: {color: "#d62828", width: 2.5}},
                   hoverinfo: "skip", xaxis: "x", yaxis: "y"});
    }

    const hasCov = P.seq && P.peps.length > 0;
    const shapes = [];
    if (hasCov) {
      const prof = covProfile(P);
      const cd = [];
      for (let i = 0; i < P.L; i++) cd.push([P.seq[i], prof.cov[i]]);
      let zmax = 1e-9;
      for (const v of prof.dens) zmax = Math.max(zmax, v);
      traces.push({type: "heatmap", z: [prof.dens],
                   x: Array.from({length: P.L}, (_, i) => i + 1),
                   y: ["coverage"], colorscale: SCALE, zmin: 0, zmax: zmax,
                   showscale: true,
                   colorbar: {title: {text: "Degree of<br>protein<br>coverage",
                                      font: {size: 11}},
                              thickness: 12, len: 0.26, y: 0.13,
                              tickfont: {size: 10}},
                   customdata: [cd],
                   hovertemplate: "Residue: %{customdata[0]}%{x}<br>"
                                + "Covering PSMs: %{customdata[1]}<br>"
                                + "Coverage: %{z:.3f}<extra></extra>",
                   xaxis: "x2", yaxis: "y2"});
      shapes.push({type: "rect", xref: "x2", yref: "y2 domain",
                   x0: 0.5, x1: P.L + 0.5, y0: 0, y1: 1,
                   line: {color: "#334155", width: 1.2},
                   fillcolor: "rgba(0,0,0,0)", layer: "above"});
    }

    const layout = {
      template: "plotly_white", height: 640,
      title: {text: "Modification landscape \u2014 " + prot + " (" + P.sites.length
                  + " modified sites, " + nSig + " significant)",
              font: {size: 15, color: "#0f2540"}},
      xaxis: {anchor: "y", range: [0.5, P.L + 0.5], showticklabels: !hasCov,
              zeroline: false,
              title: hasCov ? "" : "Position in protein"},
      yaxis: {domain: hasCov ? [0.32, 1] : [0, 1], anchor: "x",
              title: "PSM count", rangemode: "tozero"},
      legend: {orientation: "h", y: 1.09, x: 0.5, xanchor: "center"},
      margin: {t: 110, b: 50, l: 70, r: 90},
      shapes: shapes};
    if (hasCov) {
      layout.xaxis2 = {anchor: "y2", matches: "x", range: [0.5, P.L + 0.5],
                       title: "Position in protein"};
      layout.yaxis2 = {domain: [0, 0.18], anchor: "x2",
                       showticklabels: false};
    }
    Plotly.react("protein_plot", traces, layout,
                 {responsive: true, displaylogo: false});
    info.textContent = P.sites.length + " modified sites \u00b7 " + nSig
        + " significant \u00b7 protein length " + P.L + " aa";
  }

  quick.addEventListener("change", () => {
    input.value = quick.value; render(quick.value); });
  input.addEventListener("change", () => {
    const v = input.value.trim();
    if (PROT_DATA.proteins[v]) { quick.value = ""; render(v); }
    else info.textContent = "protein not found"; });
  input.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); input.blur();
                             input.dispatchEvent(new Event("change")); } });

  const first = PROT_DATA.order[0];
  if (first) { quick.value = first; input.value = first; render(first); }
})();
"""

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__PLOTLYJS__
<style>__CSS__</style>
</head>
<body>
<header>
  <h1>OpenPtmFinder <span class="accent">report</span></h1>
  <p>Generated __DATE__ &middot; source directory: __OUTDIR__</p>
</header>
__NAV__
__SECTIONS__
<footer>OpenPtmFinder &middot; self-contained interactive report &middot;
no server or internet connection required</footer>
<script>
// Inline plot scripts run before later flex siblings exist, so each Plotly
// graph is first sized to the FULL row width. Once the page has settled,
// emit a resize so every responsive plot redraws at its final flex width
// (otherwise legends/labels land outside the visible area).
window.addEventListener("load", function () {
  setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 60);
});
</script>
</body>
</html>
"""


def _fig_div(fig, div_id: str) -> str:
    return '<div class="plot">' + pio.to_html(
        fig, full_html=False, include_plotlyjs=False, div_id=div_id,
        config={"responsive": True, "displaylogo": False}) + "</div>"


def _section(sec_id: str, title: str, body: str) -> str:
    return f'<section id="{sec_id}"><h2>{title}</h2>{body}</section>'


def generate_report(output_dir: str,
                    report_path: str = None,
                    fasta_file: str = None,
                    plotlyjs: str = "inline",
                    top_proteins: int = 10,
                    alpha: float = 0.05,
                    logfc_thr: float = 1.0) -> str:
    """Render the self-contained interactive HTML report. Returns its path."""
    output_dir = os.path.abspath(output_dir)
    report_path = report_path or os.path.join(output_dir, "openptmfinder_report.html")

    stats = load_stat_results(output_dir)
    expr = load_expression(output_dir)
    weights = load_weights(output_dir)
    perm = load_permutations(output_dir)
    annot = load_annotated(output_dir)
    fasta_seqs = load_fasta_dict(fasta_file)

    if stats.empty:
        raise FileNotFoundError(
            f"No {STAT_GLOB} files found in {output_dir}; run the statistics step first.")

    colors = mod_color_map(pd.concat([
        stats["modification"].dropna(),
        annot["Modification"].dropna().astype(str).str.split("@").str[0]
        if not annot.empty and "Modification" in annot.columns
        else pd.Series(dtype=str)]).unique())

    # ---- summary cards ----
    n_prot = stats["protein"].nunique()
    n_sites = stats["site"].nunique()
    n_mods = stats["modification"].dropna().nunique()
    sig_hits = int(((stats["adj.P.Val"] < alpha) &
                    (stats["logFC"].abs() >= logfc_thr)).sum())
    n_contr = stats["contrast"].nunique()
    card_data = [("Proteins", n_prot), ("Tested sites", n_sites),
                 ("Modification types", n_mods), ("Contrasts", n_contr),
                 (f"Hits (adj.P&lt;{alpha}, |logFC|&ge;{logfc_thr})", sig_hits)]
    cards = "".join(
        f'<div class="card c{i % 5}"><div class="num">{v}</div>'
        f'<div class="lbl">{k}</div></div>' for i, (k, v) in enumerate(card_data))
    fam_badges = "".join(f'<span class="badge">{f}</span>'
                         for f in sorted(stats["mod_family"].unique()))
    sections, nav = [], []

    def add(sec_id, title, body):
        if body:
            sections.append(_section(sec_id, title, body))
            nav.append((sec_id, title.split(" (")[0]))

    add("summary", "Summary",
        f'<div class="cards">{cards}</div>'
        f'<div class="badges">{fam_badges}</div>')

    # ---- PTM landscape ----
    land_figs = ptm_landscape_figures(stats, annot, colors)
    if land_figs:
        body = '<div class="fig-row">' + "".join(
            _fig_div(f, f"land_{k}") for k, f in land_figs.items()) + "</div>"
        add("landscape", "PTM landscape", body)

    # ---- differential analysis: volcano + synced DE-sites table ----
    families = sorted(stats["mod_family"].dropna().unique())
    contrasts = sorted(stats["contrast"].dropna().astype(str).unique())
    fam_opts = ('<option value="__ALL__" selected>All families</option>' +
                "".join(f'<option value="{_esc(f)}">{_esc(f)}</option>'
                        for f in families))
    # Exactly ONE pairwise comparison at a time: no "all contrasts" option.
    con_opts = "".join(f'<option value="{_esc(c)}">{_esc(c)}</option>'
                       for c in contrasts)
    de_head = "".join(f"<th>{h}</th>" for h in
                      ["site", "modification", "protein", "position", "contrast",
                       "logFC", "t", "P.Value", "adj.P.Val", "n_psm", "status"])
    volcano_body = f"""
<div class="controls">
  <label>Family <select id="vc_family">{fam_opts}</select></label>
  <label>Contrast <select id="vc_contrast">{con_opts}</select></label>
  <label>adj.P &lt; <input id="vc_alpha" type="number" value="{alpha}"
        min="0" max="1" step="0.005"></label>
  <label>|log2FC| &ge; <input id="vc_fc" type="number" value="{logfc_thr}"
        min="0" step="0.1"></label>
  <span class="badge badge-red" id="vc_hits"></span>
</div>
<div id="volcano_plot" class="plot"></div>
<h3>Differentially expressed sites</h3>
<p class="note">Only sites passing the cutoffs above (the red volcano points)
are listed — the table stays in sync with the family / contrast / adj.P /
|logFC| selection. Click a column header to sort. Full statistics for all
tested sites remain in the final_stat_result_*.csv files.</p>
<div class="table-controls">
  <input id="deSearch" type="search" placeholder="Search in DE sites&hellip;">
  <select id="dePageSize" title="Rows per page"><option>10</option>
    <option selected>25</option><option>50</option><option>100</option>
    <option>500</option></select>
  <button id="deCsv">Download CSV (listed sites)</button>
  <span class="note" id="deInfo"></span>
</div>
<div class="table-wrap">
<table class="dtable" id="deTable"><thead><tr>{de_head}</tr></thead>
<tbody></tbody></table>
</div>
<div class="pager" id="dePager"></div>
<script>{_DIFF_JS.replace("__DE_JSON__", de_payload(stats))}</script>"""
    add("volcano", "Differential analysis", volcano_body)

    # ---- heatmap ----
    hm = heatmap_figure(stats, expr)
    if hm is not None:
        # NB: the Plotly div id must NOT collide with the <section> anchor id
        # ("heatmap") — getElementById returns the <section> first and Plotly
        # would render outside its container (detached/empty section).
        add("heatmap", "Top differential sites", _fig_div(hm, "heatmap_plot"))

    # ---- protein landscapes: ALL proteins, rendered client-side ----
    prot_json = protein_site_payload(annot, stats, fasta_seqs, colors,
                                     alpha=alpha, logfc_thr=logfc_thr)
    if prot_json is not None:
        n_prots = int(annot["id_prot"].astype(str).nunique())
        prot_note = "" if fasta_seqs else (
            '<p class="note">Peptide-coverage bar disabled: no FASTA provided '
            '(--fasta).</p>')
        prot_body = f"""
<p class="note">Lollipop plot: PSM count per modified position (color =
modification type, red rim = passes adj.P&lt;{alpha} and
|log2FC|&ge;{logfc_thr} in at least one contrast; hover shows per-contrast
statistics). Full-length bar below: every residue of the protein colored by
peptide coverage — the fraction of PSMs whose peptide covers that residue
(grey = not covered). All <b>{n_prots}</b> identified proteins are embedded:
pick one of the top-significant or type any protein id.</p>
{prot_note}
<div class="controls">
  <label>Top significant <select id="protQuick"></select></label>
  <label>Protein search <input id="protSearch" list="protList"
        placeholder="type a protein id&hellip;"><datalist id="protList"></datalist></label>
  <span class="badge" id="protInfo"></span>
</div>
<div id="protein_plot" class="plot"></div>
<script>{_PROTEIN_JS.replace("__PROT_JSON__", prot_json)}</script>"""
        add("proteins", "Protein modification landscape", prot_body)

    # ---- database annotation (iPTMnet / dbPTM / SIGNOR caches) ----
    db_ann = load_db_annotation(output_dir)
    db_figs = db_annotation_figures(db_ann, colors)
    if db_figs:
        missing = [db for db in DB_ANNOTATION_FILES if db not in db_ann]
        miss_note = (f'<p class="note">No site-level cache found for: '
                     f'{", ".join(missing)} — the annotation step was not run '
                     f'or produced no results for these databases.</p>'
                     if missing else "")
        rows = []
        for chunk in (("coverage", "by_mod"), ("signor", "regulators"),
                      ("dbptm_evidence",)):
            divs = [_fig_div(db_figs[k], f"db_{k}") for k in chunk if k in db_figs]
            if divs:
                rows.append('<div class="fig-row">' + "".join(divs) + "</div>")
        add("dbs", "Database annotation", miss_note + "".join(rows))

    # ---- SIGNOR causal network (embedded, fully offline) ----
    signor_iframe = signor_network_iframe(output_dir)
    if signor_iframe:
        add("signor", "SIGNOR causal network",
            '<p class="note">Causal PTM-centric view from SIGNOR: large round '
            'nodes are proteins measured in the experiment (green/red = '
            'significantly up-/down-regulated, grey = not significant), small '
            'diamonds are SIGNOR regulators; green/red edges = '
            'activation/inhibition (solid = direct on a measured site, dashed '
            '= indirect). Hover nodes and edges for details. The same network '
            'is saved as <b>signor_network.html</b> in the output directory '
            '(open it for a full-screen view).</p>' + signor_iframe)

    # ---- QC ----
    qc_parts = []
    if not perm.empty:
        qc_parts.append("<h3>Permutation validation</h3><div class=\"table-wrap\">" +
                        perm.to_html(index=False, classes="dtable", border=0,
                                     float_format=lambda x: f"{x:.4g}") + "</div>")
    pb = permutation_bar_figure(perm)
    if pb is not None:
        qc_parts.append(_fig_div(pb, "qc_perm_bar"))
    ph = pvalue_histogram_figure(stats)
    if ph is not None:
        qc_parts.append(_fig_div(ph, "qc_pval"))
    ma = ma_figure(stats, expr, alpha=alpha, logfc_thr=logfc_thr)
    if ma is not None:
        qc_parts.append(_fig_div(ma, "qc_ma"))
    qc_figs = qc_figures(stats, weights)
    if qc_figs:
        qc_parts.append('<div class="fig-row">' + "".join(
            _fig_div(f, f"qc_{k}") for k, f in qc_figs.items()) + "</div>")
    if qc_parts:
        add("qc", "QC", "".join(qc_parts))

    nav_html = ("<nav>" + "".join(f'<a href="#{i}">{t}</a>' for i, t in nav) + "</nav>")
    plotlyjs_head = ("<script>" + _get_plotlyjs() + "</script>"
                     if plotlyjs == "inline" else PLOTLY_CDN)

    html = (_TEMPLATE
            .replace("__TITLE__", "OpenPtmFinder report")
            .replace("__PLOTLYJS__", plotlyjs_head)
            .replace("__CSS__", _CSS)
            .replace("__DATE__", datetime.now().strftime("%Y-%m-%d %H:%M"))
            .replace("__OUTDIR__", output_dir)
            .replace("__NAV__", nav_html)
            .replace("__SECTIONS__", "".join(sections)))

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    # machine-readable manifest alongside the report
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir,
        "report": report_path,
        "alpha": alpha,
        "logfc_threshold": logfc_thr,
        "n_stat_files": len([p for p in glob.glob(os.path.join(output_dir, STAT_GLOB))
                             if os.path.basename(p) != "final_stat_result_with_dbs.csv"]),
        "mod_families": sorted(stats["mod_family"].unique().tolist()),
        "contrasts": sorted(stats["contrast"].dropna().unique().tolist()),
        "n_tested_sites": int(n_sites),
        "n_significant_hits": sig_hits,
    }
    with open(os.path.join(output_dir, "report_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    logger.info(f"Report written to {report_path}")
    return report_path


def main():
    ap = argparse.ArgumentParser(description="Static HTML report for OpenPtmFinder results.")
    ap.add_argument("-o", "--output_dir", required=True,
                    help="Pipeline output directory with final_stat_result_*.csv etc.")
    ap.add_argument("-r", "--report", default=None, help="Report file path "
                    "(default: <output_dir>/openptmfinder_report.html)")
    ap.add_argument("-f", "--fasta", default=None,
                    help="Protein FASTA for the peptide-coverage track.")
    ap.add_argument("--plotlyjs", choices=["inline", "cdn"], default="inline",
                    help="'inline' embeds plotly.js for a fully offline report "
                    "(larger file); 'cdn' needs internet but is compact.")
    ap.add_argument("--top-proteins", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--logfc", type=float, default=1.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    path = generate_report(args.output_dir, report_path=args.report,
                           fasta_file=args.fasta, plotlyjs=args.plotlyjs,
                           top_proteins=args.top_proteins,
                           alpha=args.alpha, logfc_thr=args.logfc)
    print(path)


if __name__ == "__main__":
    main()
