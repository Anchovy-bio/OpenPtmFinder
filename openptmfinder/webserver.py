from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import plotly
import json
import numpy as np
import random
import re
import plotly.utils
import os
from ast import literal_eval
from pyteomics import fasta

template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)
output = os.environ.get('OUTPUT_DIR', '')
port_n = os.environ.get('port_n', '')
fasta_file = os.environ.get('fasta', '')
full_df = pd.read_pickle(os.path.join(output, 'annotated_df.pickle'))
diff_resuls=pd.read_csv(os.path.join(output,'final_stat_result.csv'))
pr=[]
se=[]
try:
    with fasta.read(fasta_file) as db:
        for descr, seq in db:
            pr.append(descr)
            se.append(seq)
except FileNotFoundError:
    logger.error(f"FASTA file not found at {fasta_file}. Skipping FASTA concat.")

fasta_df=pd.DataFrame()
fasta_df['protein']=pr
fasta_df['sequence']=se
fasta_df['id_prot']=fasta_df['protein'].str.split('|').str[1]


@app.route("/")
def index():
    chart_df=full_df[full_df['position_in_protein'].notna()]
    chart_df = chart_df[chart_df['Modification'] != 'reference']
    chart_df=chart_df.drop_duplicates(subset=['position_in_protein','id_prot','Modification'])
    chart_df['Mods']=chart_df['Modification'].str.split('@').str[0]
    mods_count = chart_df['Mods'].value_counts()
    top_mods = mods_count.head(8)
    other_mods = mods_count.iloc[8:]

    pie_labels = list(top_mods.index)
    pie_values = list(top_mods.values)

    if not other_mods.empty:
        pie_labels.append("Other")
        pie_values.append(other_mods.sum())

    pie_labels_bold = [f"<b>{label}</b>" for label in pie_labels]

    fig_pie = go.Figure(data=[go.Pie(
        labels=pie_labels_bold,
        values=pie_values,
        hole=0.4,
        marker=dict(line=dict(color='#000000', width=2)),
        textinfo='label+value+percent',
        texttemplate = "%{label}: %{value:s} <br>(%{percent})",
        insidetextorientation='radial'
    )])

    # Текстовый блок со списком "Other"
    other_text = ""
    if not other_mods.empty:
        mods_list = list(other_mods.index)
        chunks = [", ".join(mods_list[i:i+10]) for i in range(0, len(mods_list), 10)]
        other_text = "Other includes:<br>" + "<br>".join(chunks)

    fig_pie.update_layout(
        title='Number of modification sites found in filtered mass-spectra',
        title_x=0.5,
        font=dict(family="Khula, sans-serif", size=16, color="black"),
        paper_bgcolor='white',
        height=800,
        margin=dict(t=100, b=200),
        annotations=[
            dict(
                text=other_text,
                showarrow=False,
                x=0.5,
                y=-0.25,
                xref="paper",
                yref="paper",
                font=dict(size=14, color="black"),
                align="center"
            )
        ]
    )
    path_fig=os.path.join(output, f'pie_plot.html')
    fig_pie.write_html(path_fig)
    pie_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    
    group1=diff_resuls['intensity_TMT_group1'].str.split(',').str.len()
    group2=diff_resuls['intensity_TMT_group2'].str.split(',').str.len()
    diff_resuls['number of samples\n group1/group2'] = group1.astype(str) + '/' + group2.astype(str)
    diff_resuls['number of spectra']=diff_resuls['spectrum_y'].str.split(',').str.len()
    columns=['id_prot','position_in_protein','Modification','number of samples\n group1/group2','number of spectra','stoich1_median','stoich2_median','FC_coef','FC_median_abs','T_test_p_value_coef','pvalue_Ttest_correct']
    stats_table_html = diff_resuls[columns].to_html(index=False, classes='centered-table', table_id="modTable")
    protein_count = diff_resuls['id_prot'].nunique()
    modsite_count = diff_resuls['position_in_protein'].nunique()
    modtype_count = diff_resuls['Modification'].nunique()

    return render_template(
        "index.html",
        stats_table=stats_table_html,
        pie_json=pie_json,
        protein_count=protein_count,
        modsite_count=modsite_count,
        modtype_count=modtype_count
    )

@app.route("/", methods=['POST'])
def get_protein_plot():
    data = request.get_json()
    protein = data.get('proteinId', '').strip()
    print(protein)
    
    if full_df.empty:
        return jsonify({'success': False, 'error': 'Dataframe is empty.'})
    
    df_local = full_df[(full_df['id_prot'] == protein)].drop_duplicates(
        subset=['spectrum_y','position_in_protein','peptide_y','Modification']
    )
    
    if df_local.empty:
        return jsonify({'success': False, 'error': 'Data for the specified protein was not found. Please check that the entered ID is correct.'})

    def peptide_density_distribution(protein_sequence, peptides):
        peptide_positions = []
        for peptide in peptides:
            for i in range(len(protein_sequence) - len(peptide) + 1):
                if protein_sequence[i:i+len(peptide)] == peptide:
                    peptide_positions.append([peptide, i, i + len(peptide)])

        df = pd.DataFrame(peptide_positions, columns=['Peptide', 'Start', 'End'])

        # Группируем данные по позициям начала пептидов
        grouped_df = df.groupby(['Start','End']).size().reset_index(name='Count')
        # Создаем список позиций на белке
        positions = list(range(1, len(protein_sequence) + 1))

        # Вычисляем плотность пептидов в каждой позиции
        densities = []
        for position in positions:
            try:
                count = grouped_df[(grouped_df['Start'] <= position) & (grouped_df['End'] >= position)]['Count'].values.sum()
                density = count / len(peptides) if len(peptides) > 0 else 0
            except: 
                density=0
            densities.append(density)

        return pd.DataFrame({'Position': positions, 'Density': densities})
    
    seq = fasta_df.loc[fasta_df['id_prot'] == protein, 'sequence'].iloc[0]
    density_df = peptide_density_distribution(
        seq, df_local['peptide_y'].tolist()
    )
    df = df_local[df_local['position_in_protein'].notna()]
    df = df[df['Modification']!='reference']
    df = df.groupby(['Modification','position_in_protein']).agg(list).reset_index()
    df['count'] = [len(x) for x in df.spectrum_y]

    # Получаем список уникальных модификаций
    df['Mods'] = df['Modification'].apply(lambda x: x.split('@')[0])
    df = df[df['Mods']!='Sulfation']
    unique_mods = df['Mods'].unique()
    no_of_colors = len(unique_mods)

    # Выбираем палитру (повторяем, если не хватает цветов)
    base_colors = pc.qualitative.Set2
    colors = (base_colors * ((no_of_colors // len(base_colors)) + 1))[:no_of_colors]

    # Создаем отображение "модификация → цвет"
    mod_to_color = dict(zip(unique_mods, colors))
    df['color'] = df['Mods'].map(mod_to_color)
    print(mod_to_color)

    # Создание графика
    fig = go.Figure()

    for modification in unique_mods:
        subset = df[df['Mods'] == modification]

        # Точки
        fig.add_trace(go.Scatter(
            x=subset['position_in_protein'].tolist(),
            y=subset['count'].tolist(),
            mode='markers',
            marker=dict(
                size=10,
                color=mod_to_color[modification],
                line=dict(color='white', width=0.7)
            ),
            hovertemplate="Position: %{x}<br>Count PSMs: %{y}<extra></extra>",
            name=modification,
            legendgroup=modification,
            showlegend=True
        ))

        # Линии вверх
        for line in subset.index:
            fig.add_trace(go.Scatter(
                x=[subset.position_in_protein[line], subset.position_in_protein[line]],
                y=[0, subset['count'][line]],
                mode='lines',
                line=dict(color=mod_to_color[modification], width=1),
                name=f"{modification}_line",
                legendgroup=modification,
                showlegend=False,
                hoverinfo="skip"
            ))

    # Максимумы
    xmax = df['position_in_protein'].max()
    ymax = df['count'].max()    

    # Прямоугольники для плотности покрытия
    amino_acids = list(seq)
    heights = density_df['Density'].tolist()
    bottom = np.arange(len(amino_acids)) + 0.5

    fig.add_trace(go.Bar(
        x=[1] * len(amino_acids),
        y=[-round(ymax * 0.1, 1)] * len(amino_acids),
        base=bottom,
        width=[round(ymax * 0.1, 1) * 2] * len(amino_acids),
        marker=dict(
            color=heights,
            colorscale="Reds",
            colorbar=dict(
                title=dict(text="Degree of<br>protein coverage",font=dict(family="Arial", size=14)),
                tickfont=dict(size=12),
                thickness=15,
                x=1.01,
                bgcolor="rgba(240,240,240,0.9)",
                bordercolor="white",
                borderwidth=1,
            ),
            line=dict(color='black', width=0.5)
        ),
        text=[f"<b>{aa}</b>" for aa in amino_acids],
        hovertemplate=[
            f"Aminoacid: {aa}<br>Position: {i+1}<br>Density: {d:.3f}<extra></extra>"
            for i, (aa, d) in enumerate(zip(amino_acids, heights))
        ],
        textposition="inside",
        textfont=dict(family="Arial Black", size=26),
        orientation="h",
        showlegend=False
    ))

    # Единое оформление
    fig.update_layout(
        title=dict(
            #text=f"<b>{protein}</b>",
            x=0.95,
            font=dict(family="Arial", size=20, color="black")
        ),
        xaxis=dict(
            title=dict(text="Position of modification",font=dict(size=16)),
            linecolor="black",
            showgrid=False,
            ticks="outside",
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=dict(text="Count PSMs",font=dict(size=16)),
            linecolor="black",
            showgrid=False,
            ticks="outside",
            tickfont=dict(size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=14, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(240,240,240,0.9)",
            bordercolor="black",
            borderwidth=1,
            title=dict(text="<b>Modification Types</b>", font=dict(size=14, family="Arial"))
        ),
        margin=dict(t=20, b=60, l=80, r=100)
    )
    fig.update_layout(xaxis_autorange=True, yaxis_autorange=True)
    path_fig=os.path.join(output, f'{protein}.html')
    fig.write_html(path_fig)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'success': True, 'graphJSON': graphJSON})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(port_n), debug=True, use_reloader=False)
