from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import plotly
import random
import json
import numpy as np
import random
import re
import plotly.utils
import os
from ast import literal_eval

template_dir = os.path.join(os.path.dirname(__file__), 'templates')
print("TEMPLATE DIR:", template_dir)
print("FILES:", os.listdir(template_dir))
app = Flask(__name__, template_folder=template_dir)
output = os.environ.get('OUTPUT_DIR', '')
full_df = pd.read_pickle(output+'annotated_df.pickle')
diff_resuls=pd.read_csv(output+'final_stat_result.csv')

@app.route("/")
def index():
    diff_resuls['Modification'] = diff_resuls['Modification'].apply(lambda x: list(set(sum(literal_eval(x),[])))[0])
    diff_resuls['spectrum_x'] = diff_resuls['spectrum_x'].apply(lambda x: list(set(sum(literal_eval(x),[]))))
    chart_df=full_df[full_df['position_in_protein'].notna()]
    chart_df=chart_df.drop_duplicates(subset=['position_in_protein','id_prot','Modification'])
    pie_labels=chart_df['Modification'].value_counts().keys()
    pie_values=chart_df['Modification'].value_counts().values
    if len(pie_labels)==1:
        pie_labels=list(pie_labels)
        pie_values=list(pie_values)
    print(pie_values)

    fig_pie = go.Figure(data=[go.Pie(
        labels=pie_labels,
        values=pie_values.tolist(),
        hole=0.4,
        marker=dict(line=dict(color='#000000', width=2)),
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    fig_pie.update_layout(title='Number of modification sites found in filtered mass-spectra',
        title_x=0.5,
        font=dict(family="Khula, sans-serif", size=16, color="black"),
        paper_bgcolor='white',
        height=700
    )
    pie_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    
    group1=diff_resuls['intensity_TMT_group1'].apply(lambda x: str(len(x.split(','))))
    group2=diff_resuls['intensity_TMT_group2'].apply(lambda x: str(len(x.split(','))))
    diff_resuls['number of samples group1/group2']=group1+'/'+group2
    diff_resuls['number of spectra']=diff_resuls['spectrum_x'].apply(len)
    columns=['id_prot','position_in_protein','Modification','number of samples group1/group2','number of spectra','pvalue_correct']
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

    uniprot_pattern = r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]{5}$|^A0A[A-Z0-9]{7}$"
    if not protein or not re.match(uniprot_pattern, protein):
        return jsonify({'success': False, 'error': 'Invalid UniProt ID'}), 400
    
    if full_df.empty:
        return jsonify({'success': False, 'error': 'Dataframe is empty.'})
    
    df_local = full_df[(full_df['id_prot'] == protein)]
    if df_local.empty:
        return jsonify({'success': False, 'error': 'Data for the specified protein was not found.'})

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

    density_df = peptide_density_distribution(df_local.iloc[0]['sequence'], df_local['peptide'].tolist())
    df=df_local[df_local['position_in_protein'].notna()]
    df=df.groupby(['Modification','position_in_protein']).agg(list).reset_index()
    df['count']=[len(x) for x in df.spectrum_x]

    # Получаем список уникальных модификаций
    unique_mods = df.Modification.unique()
    no_of_colors = len(unique_mods)

    # Выбираем палитру (повторяем, если не хватает цветов)
    base_colors = pc.qualitative.Set2
    colors = (base_colors * ((no_of_colors // len(base_colors)) + 1))[:no_of_colors]

    # Создаем отображение "модификация → цвет"
    mod_to_color = dict(zip(unique_mods, colors))
    df['color'] = df['Modification'].map(mod_to_color)

    # Создание графика
    fig = go.Figure()
    leg = []
    for modification in unique_mods:
        subset = df[df['Modification'] == modification]
        fig.add_trace(go.Scatter(
            x=subset['position_in_protein'].tolist(),
            y=subset['count'].tolist(),
            mode='markers',
            marker=dict(
                size=10,
                color=mod_to_color[modification],
                line=dict(color='white', width=0.7)
            ),
            hoverinfo="all",
            hovertemplate="Position mod: %{x}<br>Count PSMs: %{y}<extra></extra>",
            hoverlabel=dict(
                font_family='Khula, sans-serif',
                font_size=14,
                bgcolor="#2d313c",
                bordercolor="#ffcc00",
            ),
            name=modification,
            legendgroup=modification,
            showlegend=True
        ))
        for line in subset.index:
            fig.add_trace(go.Scatter(
                x=[subset.position_in_protein[line], subset.position_in_protein[line]],
                y=[0, subset['count'][line]],
                mode='lines',
                name=modification,
                legendgroup=modification,
                showlegend=False,
                hoverinfo="skip",
                line=dict(color=mod_to_color[modification])
            ))

    fig.update_layout(title=f"Modifications in the protein {protein}", xaxis_title="position of modification", yaxis_title="count PSMs",
                      title_x=0.5, font=dict(family="Khula, sans-serif", size=16, color="black"), 
                      plot_bgcolor='white',paper_bgcolor='lightgrey',xaxis_linecolor='black',yaxis_linecolor='black')
    fig.update_layout(margin=dict(t=80, b=40)) #посмотреть что измениться
    xmax = df['position_in_protein'].max()
    ymax = df['count'].max()    

    amino_acids = list(df_local.iloc[0]['sequence'])
    heights = density_df['Density'].tolist()
    
    bottom = np.arange(len(amino_acids))+0.5
    colors = [f'rgba(255,0,0,{h})' for h in heights]

    fig.add_trace(go.Bar(
        x=[1] * len(amino_acids),
        y=[-round(ymax * 0.1, 1)] * len(amino_acids),
        base=bottom,
        width=[round(ymax * 0.1, 1) * 2] * len(amino_acids),
        marker_color=colors,
        text=[f"<b>{aa}</b>" for aa in amino_acids],
        hovertemplate=[
            f"Aminoacid: {aa}<br>Position: {i+1}<br>Density: {d}<extra></extra>"
            for i, (aa, d) in enumerate(zip(amino_acids, heights))
        ],
        textposition="inside",
        textfont=dict(family="Arial Black", size=30),
        orientation="h",
        showlegend=False,
        marker_line_color='black'
    ))
    fig.update_layout(xaxis_autorange=True, yaxis_autorange=True)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'success': True, 'graphJSON': graphJSON})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10005, debug=True, use_reloader=False)
