"""
dash_server.py — interactive Dash web app for OpenPtmFinder results.

Optional interactive mode (the static self-contained HTML report produced by
report.py remains the primary output). Under the hood Dash is still Flask,
but with real server-side callbacks instead of hand-rolled JSON/fetch, and it
is served by waitress when available (production WSGI server).

Data comes from the parquet layer (<output_dir>/report_data/) written by
report.export_report_data, with automatic fallback to the CSV/pickle outputs.

Pages:
  - Overview:  summary cards, PTM landscape (donut + tested-sites bar), heatmap;
  - Differential analysis: family selector, ONE pairwise contrast at a time,
               adj.P / |logFC| threshold inputs, live hit counter, and a table
               of the differentially expressed sites only — fed by the same
               callback as the volcano, so it always reflects the cutoffs;
  - Proteins:  arbitrary protein queried ON THE FLY (searchable dropdown) —
               lollipop landscape with peptide-coverage track from FASTA.

Launched by main.py (--interactive or interactive=True in config.ini), or
standalone:
    OUTPUT_DIR=<dir> fasta=<proteins.fasta> port_n=10030 python dash_server.py

Requires: pip install dash pyarrow (waitress recommended).
"""

import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

try:  # package import
    from .report import (load_stat_results, load_annotated, load_expression,
                         load_weights, load_permutations, load_db_annotation,
                         load_fasta_dict, mod_color_map, single_protein_figure,
                         ptm_landscape_figures, heatmap_figure, qc_figures,
                         pvalue_histogram_figure, ma_figure,
                         permutation_bar_figure, db_annotation_figures,
                         DB_ANNOTATION_FILES, _BASE_LAYOUT)
except ImportError:  # standalone usage
    from report import (load_stat_results, load_annotated, load_expression,
                        load_weights, load_permutations, load_db_annotation,
                        load_fasta_dict, mod_color_map, single_protein_figure,
                        ptm_landscape_figures, heatmap_figure, qc_figures,
                        pvalue_histogram_figure, ma_figure,
                        permutation_bar_figure, db_annotation_figures,
                        DB_ANNOTATION_FILES, _BASE_LAYOUT)

CARD_COLORS = ["linear-gradient(140deg,#0f2540,#1d4e89)",
               "linear-gradient(140deg,#0077b6,#00b4d8)",
               "linear-gradient(140deg,#2a9d8f,#43aa8b)",
               "linear-gradient(140deg,#7b2cbf,#9d4edd)",
               "linear-gradient(140deg,#d62828,#f3722c)"]

_S = dict(
    body=dict(fontFamily="Segoe UI, Arial, sans-serif", margin=0,
              backgroundColor="#f1f5f9", color="#334155"),
    header=dict(background="linear-gradient(120deg,#0f2540 0%,#1d4e89 55%,#00b4d8 130%)",
                color="#fff", padding="22px 34px 18px"),
    h1=dict(margin=0, fontSize="1.5em"),
    sub=dict(color="#cbd5e1", margin="6px 0 0", fontSize=".85em"),
    section=dict(background="#fff", margin="18px auto", padding="18px 24px",
                 maxWidth="1240px", borderRadius="14px",
                 boxShadow="0 2px 10px rgba(15,37,64,.08)"),
    h2=dict(margin="2px 0 14px", fontSize="1.15em", color="#0f2540",
            paddingLeft="12px", borderLeft="5px solid #ffb703"),
    cards=dict(display="flex", gap="14px", flexWrap="wrap"),
    card=dict(flex="1", minWidth="150px", borderRadius="12px", padding="14px",
              textAlign="center", color="#fff",
              boxShadow="0 2px 8px rgba(15,37,64,.18)"),
    num=dict(fontSize="1.9em", fontWeight=700),
    lbl=dict(fontSize=".78em", textTransform="uppercase", letterSpacing=".5px",
             opacity=.92),
    controls=dict(display="flex", gap="20px", flexWrap="wrap", alignItems="center",
                  margin="4px 0 12px", padding="10px 14px", background="#f8fafc",
                  border="1px solid #e2e8f0", borderRadius="10px"),
    label=dict(fontSize=".85em", fontWeight=600, color="#0f2540"),
    badge=dict(padding="4px 12px", borderRadius="999px", fontSize=".82em",
               fontWeight=600, background="#fde8e9", color="#e63946",
               border="1px solid #f5c2c6"),
    note=dict(color="#64748b", fontSize=".85em"),
)


def volcano_dash_figure(stats: pd.DataFrame, family: str, contrast: str,
                        alpha: float, logfc_thr: float):
    df = stats.dropna(subset=["logFC", "adj.P.Val"]).copy()
    df = df[df["adj.P.Val"] > 0]
    if family and family != "__ALL__":
        df = df[df["mod_family"] == family]
    if contrast and contrast != "__ALL__":
        df = df[df["contrast"].astype(str) == contrast]

    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="No data for the current selection", **_BASE_LAYOUT)
        return fig, df, 0

    y = -np.log10(df["adj.P.Val"].clip(lower=1e-300))
    sig = (df["adj.P.Val"] < alpha) & (df["logFC"].abs() >= logfc_thr)
    cd = np.stack([df["site"].astype(str), df["protein"].astype(str),
                   df["modification"].fillna("NA").astype(str)], axis=-1)
    hover = ("Site: %{customdata[0]}<br>Protein: %{customdata[1]}<br>Mod: %{customdata[2]}"
             "<br>log2FC: %{x:.3f}<br>-log10(adj.P): %{y:.3f}<extra></extra>")
    for mask, name, color, size in ((~sig, "not significant", "rgba(148,163,184,0.45)", 7),
                                    (sig, "significant", "rgba(230,57,70,0.85)", 8)):
        fig.add_trace(go.Scattergl(
            x=df.loc[mask, "logFC"], y=y[mask], mode="markers", name=name,
            marker=dict(size=size, color=color),
            customdata=cd[mask.to_numpy()], hovertemplate=hover))

    ythr = -np.log10(max(alpha, 1e-300))
    line = dict(dash="dash", color="#94a3b8", width=1)
    fig.update_layout(
        xaxis_title="log2 fold change", yaxis_title="-log10(adj. p-value)",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        shapes=[dict(type="line", x0=-logfc_thr, x1=-logfc_thr, yref="paper",
                     y0=0, y1=1, line=line),
                dict(type="line", x0=logfc_thr, x1=logfc_thr, yref="paper",
                     y0=0, y1=1, line=line),
                dict(type="line", y0=ythr, y1=ythr, xref="paper", x0=0, x1=1,
                     line=line)],
        **_BASE_LAYOUT)
    return fig, df.loc[sig], len(df)


def create_app(output_dir: str, fasta_file: str = None):
    from dash import Dash, dcc, html, dash_table, Input, Output

    stats = load_stat_results(output_dir)
    annot = load_annotated(output_dir)
    expr = load_expression(output_dir)
    weights = load_weights(output_dir)
    perm = load_permutations(output_dir)
    db_ann = load_db_annotation(output_dir)
    fasta_seqs = load_fasta_dict(fasta_file)

    if stats.empty:
        raise FileNotFoundError(
            f"No final_stat_result_*.csv / stat_results.parquet in {output_dir}; "
            "run the statistics step first.")

    colors = mod_color_map(pd.concat([
        stats["modification"].dropna(),
        annot["Modification"].dropna().astype(str).str.split("@").str[0]
        if not annot.empty and "Modification" in annot.columns
        else pd.Series(dtype=str)]).unique())

    families = sorted(stats["mod_family"].dropna().unique())
    contrasts = sorted(stats["contrast"].dropna().astype(str).unique())
    proteins = sorted(set(stats["protein"].dropna().astype(str)) |
                      (set(annot["id_prot"].astype(str))
                       if not annot.empty and "id_prot" in annot.columns else set()))

    app = Dash(__name__, title="OpenPtmFinder")
    app.index_string = ("<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>"
                        "{%favicon%}{%css%}</head><body "
                        "style='margin:0;font-family:Segoe UI,Arial,sans-serif;"
                        "background:#f1f5f9'>{%app_entry%}<footer>{%config%}"
                        "{%scripts%}{%renderer%}</footer></body></html>")

    # ---- summary cards ----
    card_data = [("Proteins", stats["protein"].nunique()),
                 ("Tested sites", stats["site"].nunique()),
                 ("Modification types", stats["modification"].dropna().nunique()),
                 ("Contrasts", stats["contrast"].nunique()),
                 ("Hits (adj.P<0.05, |logFC|≥1)",
                  int(((stats["adj.P.Val"] < 0.05) & (stats["logFC"].abs() >= 1)).sum()))]
    cards = [html.Div(style={**_S["card"], "background": CARD_COLORS[i % 5]}, children=[
        html.Div(str(v), style=_S["num"]), html.Div(k, style=_S["lbl"])])
        for i, (k, v) in enumerate(card_data)]

    # ---- overview figures ----
    land_figs = ptm_landscape_figures(stats, annot, colors)
    hm = heatmap_figure(stats, expr)

    # ---- database annotation figures + SIGNOR network ----
    db_figs = db_annotation_figures(db_ann, colors)
    signor_net_path = os.path.join(output_dir, "signor_network.html")
    signor_doc = None
    if os.path.isfile(signor_net_path):
        try:
            with open(signor_net_path, "r", encoding="utf-8") as fh:
                signor_doc = fh.read()
            if "<html" not in signor_doc.lower():
                signor_doc = None
        except OSError:
            signor_doc = None

    # ---- QC figures ----
    qc_stat_figs = []
    for f in (permutation_bar_figure(perm),
              pvalue_histogram_figure(stats),
              ma_figure(stats, expr, alpha=0.05, logfc_thr=1.0)):
        if f is not None:
            qc_stat_figs.append(f)
    qc_stat_figs.extend(qc_figures(stats, weights).values())

    de_cols = [c for c in ["site", "modification", "protein", "position",
                           "contrast", "logFC", "t", "P.Value", "adj.P.Val",
                           "n_psm", "status"]
               if c in stats.columns]
    from dash.dash_table.Format import Format, Scheme
    _num = {"type": "numeric",
            "format": Format(precision=4, scheme=Scheme.decimal_or_exponent)}
    de_columns = [{"name": c, "id": c,
                   **(_num if c in ("logFC", "t", "P.Value", "adj.P.Val") else {})}
                  for c in de_cols]

    app.layout = html.Div([
        html.Div(style=_S["header"], children=[
            html.H1(style=_S["h1"], children=["OpenPtmFinder ",
                    html.Span("interactive", style={"color": "#ffb703"})]),
            html.P(f"Directory: {os.path.abspath(output_dir)}", style=_S["sub"])]),

        dcc.Tabs(style={"maxWidth": "1240px", "margin": "18px auto 0"}, children=[

            dcc.Tab(label="Overview", children=[
                html.Div(style=_S["section"], children=[
                    html.H2("Summary", style=_S["h2"]),
                    html.Div(cards, style=_S["cards"])]),
                html.Div(style=_S["section"], children=[
                    html.H2("PTM landscape", style=_S["h2"]),
                    html.Div(style={"display": "flex", "gap": "18px", "flexWrap": "wrap"},
                             children=[dcc.Graph(figure=f, style={"flex": "1", "minWidth": "420px"})
                                       for f in land_figs.values()])]),
            ] + ([html.Div(style=_S["section"], children=[
                    html.H2("Top differential sites", style=_S["h2"]),
                    dcc.Graph(figure=hm)])] if hm is not None else [])),

            dcc.Tab(label="Differential analysis", children=[
                html.Div(style=_S["section"], children=[
                    html.H2("Differential analysis — volcano", style=_S["h2"]),
                    html.Div(style=_S["controls"], children=[
                        html.Label(["Family ", dcc.Dropdown(
                            id="vc_family",
                            options=[{"label": "All families", "value": "__ALL__"}] +
                                    [{"label": f, "value": f} for f in families],
                            value="__ALL__", clearable=False,
                            style={"width": "220px"})], style=_S["label"]),
                        # Exactly ONE pairwise comparison at a time:
                        # no "all contrasts" option.
                        html.Label(["Contrast ", dcc.Dropdown(
                            id="vc_contrast",
                            options=[{"label": c, "value": c} for c in contrasts],
                            value=(contrasts[0] if contrasts else None),
                            clearable=False,
                            style={"width": "260px"})], style=_S["label"]),
                        html.Label(["adj.P < ", dcc.Input(
                            id="vc_alpha", type="number", value=0.05, min=0, max=1,
                            step=0.005, style={"width": "90px"})], style=_S["label"]),
                        html.Label(["|log2FC| ≥ ", dcc.Input(
                            id="vc_fc", type="number", value=1.0, min=0, step=0.1,
                            style={"width": "80px"})], style=_S["label"]),
                        html.Span(id="vc_hits", style=_S["badge"]),
                    ]),
                    dcc.Graph(id="volcano_graph"),
                    html.H3("Differentially expressed sites",
                            style={"color": "#1d4e89", "margin": "22px 0 6px"}),
                    html.P("Only sites passing the cutoffs above (the red volcano "
                           "points) are listed — the table is fed by the same "
                           "callback as the volcano and stays in sync with it. "
                           "Full statistics for all tested sites remain in the "
                           "final_stat_result_*.csv files.", style=_S["note"]),
                    dash_table.DataTable(
                        id="de_table",
                        columns=de_columns,
                        data=[],
                        page_size=25, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto", "maxWidth": "100%"},
                        style_header={"backgroundColor": "#0f2540", "color": "#fff",
                                      "fontWeight": 600},
                        style_cell={"fontFamily": "Segoe UI, Arial, sans-serif",
                                    "fontSize": 13, "padding": "6px 10px",
                                    "whiteSpace": "nowrap"},
                        style_data_conditional=[
                            {"if": {"row_index": "odd"},
                             "backgroundColor": "#f8fafc"}],
                        export_format="csv", export_headers="display",
                    ),
                ])]),

            dcc.Tab(label="Proteins", children=[
                html.Div(style=_S["section"], children=[
                    html.H2("Protein modification landscape", style=_S["h2"]),
                    html.Div(style=_S["controls"], children=[
                        html.Label(["Protein ", dcc.Dropdown(
                            id="prot_select",
                            options=[{"label": p, "value": p} for p in proteins],
                            placeholder="Search a protein…",
                            style={"width": "320px"})], style=_S["label"]),
                        html.Span("Queried on the fly from the parquet data layer.",
                                  style=_S["note"]),
                    ]),
                    dcc.Graph(id="protein_graph"),
                    html.Div(id="protein_msg", style=_S["note"]),
                ])]),

            dcc.Tab(label="Database annotation", children=(
                [html.Div(style=_S["section"], children=[
                    html.H2("Database annotation (iPTMnet / dbPTM / SIGNOR)",
                            style=_S["h2"]),
                    html.Div(style={"display": "flex", "gap": "18px",
                                    "flexWrap": "wrap"},
                             children=[dcc.Graph(figure=f,
                                                 style={"flex": "1",
                                                        "minWidth": "420px"})
                                       for f in db_figs.values()])])]
                if db_figs else
                [html.Div(style=_S["section"], children=[
                    html.H2("Database annotation", style=_S["h2"]),
                    html.P("No site-level annotation caches found "
                           "(iptmnet_positions.csv / dbptm_positions.csv / "
                           "signor_sites.csv) — run the pipeline with the "
                           "corresponding annotation options.",
                           style=_S["note"])])]
            ) + ([html.Div(style=_S["section"], children=[
                    html.H2("SIGNOR causal network", style=_S["h2"]),
                    html.P("Green/red nodes: significantly up-/down-regulated "
                           "measured proteins; diamonds: SIGNOR regulators; "
                           "green/red edges: activation/inhibition (solid = "
                           "direct, dashed = indirect).", style=_S["note"]),
                    html.Iframe(srcDoc=signor_doc,
                                style={"width": "100%", "height": "880px",
                                       "border": "1px solid #e2e8f0",
                                       "borderRadius": "10px"})])]
                 if signor_doc else [])),

            dcc.Tab(label="QC", children=[
                html.Div(style=_S["section"], children=[
                    html.H2("QC statistics", style=_S["h2"]),
                    html.P("P-value histograms should be flat under the null "
                           "with a spike near 0 for real signal; MA plots show "
                           "whether hits are driven by low-abundance sites; "
                           "the permutation panel compares observed hits with "
                           "the label-permutation null.", style=_S["note"]),
                    html.Div(children=[dcc.Graph(figure=f)
                                       for f in qc_stat_figs]),
                ])] if qc_stat_figs else
                [html.Div(style=_S["section"], children=[
                    html.H2("QC statistics", style=_S["h2"]),
                    html.P("No QC data available.", style=_S["note"])])]),

        ]),
    ])

    @app.callback(Output("volcano_graph", "figure"),
                  Output("vc_hits", "children"),
                  Output("de_table", "data"),
                  Input("vc_family", "value"), Input("vc_contrast", "value"),
                  Input("vc_alpha", "value"), Input("vc_fc", "value"))
    def _volcano(family, contrast, alpha, fc):
        alpha = float(alpha) if alpha else 1.0
        fc = float(fc) if fc is not None else 0.0
        fig, sig_df, n_tot = volcano_dash_figure(stats, family, contrast, alpha, fc)
        records = ([] if sig_df.empty else
                   sig_df[de_cols].sort_values("adj.P.Val").to_dict("records"))
        return fig, f"{len(sig_df)} significant of {n_tot} sites", records

    @app.callback(Output("protein_graph", "figure"),
                  Output("protein_msg", "children"),
                  Input("prot_select", "value"))
    def _protein(prot):
        if not prot:
            return go.Figure(layout={**_BASE_LAYOUT,
                                     "title": "Select a protein above"}), ""
        if annot.empty or "id_prot" not in annot.columns:
            return go.Figure(), "No PSM-level annotation available."
        fig = single_protein_figure(prot, annot, fasta_seqs, colors, stats=stats)
        if fig is None:
            return go.Figure(), f"No modified sites found for {prot}."
        return fig, ""

    return app


def run_server(output_dir: str, fasta_file: str = None, port: int = 10030):
    output_dir = os.path.abspath(output_dir)
    app = create_app(output_dir, fasta_file=fasta_file)
    try:
        from waitress import serve
        logger.info(f"OpenPtmFinder interactive server (waitress): "
                    f"http://0.0.0.0:{port}")
        serve(app.server, host="0.0.0.0", port=port)
    except ImportError:
        logger.warning("waitress is not installed; using the Dash development "
                       "server (pip install waitress for production use).")
        app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run_server(os.environ.get("OUTPUT_DIR", os.getcwd()),
               fasta_file=os.environ.get("fasta"),
               port=int(os.environ.get("port_n", 10030)))
