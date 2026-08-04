import requests
import pandas as pd
import numpy as np
from io import StringIO
import json
from pyvis.network import Network
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def get_protein_info_from_iptmnet(uniprot_id, session=None):
    """
    Получает информацию о белке по UniProt ID из API iPTMnet.
    
    Args:
        uniprot_id (str): UniProt ID белка.
        session (requests.Session, optional): Сессия для повторного использования соединений.
        
    Returns:
        pd.DataFrame или None
    """
    base_url = "https://research.bioinformatics.udel.edu/iptmnet/api"
    endpoint = f"{uniprot_id}/substrate"
    url = f"{base_url}/{endpoint}"
    
    try:
        # Если сессия передана, используем её
        req = session.get(url, timeout=10) if session else requests.get(url, timeout=10)
        req.raise_for_status()
        
        data = req.text
        df = pd.read_csv(StringIO(data), sep=",")
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса для {uniprot_id}: {e}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Пустой ответ для {uniprot_id}")
        return None
    except Exception as e:
        print(f"Ошибка обработки ответа для {uniprot_id}: {e}")
        return None

def fetch_protein(prot, session):
    prot_nat = prot.split('-')[0] if '-' in prot else prot
    df = get_protein_info_from_iptmnet(prot_nat, session=session)
    return df

def fetch_iptmnet_data(protein_ids, max_workers=10):
    """
    Загружает данные для списка UniProt ID параллельно.
    
    Args:
        protein_ids (list): Список UniProt ID.
        max_workers (int): Число параллельных запросов.
        
    Returns:
        pd.DataFrame: объединенный результат
    """
    df_list = []
    with requests.Session() as session:
        session.headers.update({"Accept": "text/plain"})
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_protein, prot, session): prot for prot in protein_ids}
            for future in as_completed(futures):
                prot = futures[future]
                df = future.result()
                if df is not None:
                    df_list.append(df)
    
    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
        df_all = df_all.dropna(subset='site')
        df_all['site'] = df_all['site'].str[1:].astype(int)
        df_full = df_all.groupby(['sub_form','site']).agg(list).reset_index()
        return df_full
    else:
        return pd.DataFrame()

    
    
def fetch_and_filter(url, result_ids, column_names):
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), sep="\t", names=column_names)
    df = df[df['id_prot'].isin(result_ids)]
    df = df.dropna(subset=['id_prot','position_in_protein'])
    return df

def get_dbptm_download_links(result, base_url="https://biomics.lab.nycu.edu.tw/dbPTM/download.php", target="Windows", n_threads=4):
    dbptm_column_names = [
        "Protein_Name",     
        "id_prot",      
        "position_in_protein",         
        "Modification_Type", 
        "PubMed_ID",        
        "Sequence_Context"
    ]
    
    result_ids = set(result['id_prot'])
    
    r = requests.get(base_url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    urls = [
        urllib.parse.urljoin(base_url, a["href"])
        for a in soup.find_all("a", href=True)
        if target in a.get_text() or target.lower() in a["href"].lower()
    ]

    urls = [u for u in urls if 'experiment' in u]

    all_dfs = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(fetch_and_filter, url, result_ids, dbptm_column_names) for url in urls]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading dbPTM files"):
            df = future.result()
            all_dfs.append(df)
    
    if all_dfs:
        all_ptms = pd.concat(all_dfs, ignore_index=True)
        all_ptms = all_ptms.groupby(['id_prot','position_in_protein']).agg(list).reset_index()
        result_df = result.merge(all_ptms, how='left', on=['id_prot','position_in_protein'])
    else:
        result_df = result.copy()
    
    return result_df

    
    
def get_protein_info_from_signor(df_result: list):
    """
    Получает информацию о белке по его UniProt ID из API SIGNOR
    и возвращает её в виде DataFrame.
    """
    col=['ENTITYA', 'TYPEA', 'IDA', 'DATABASEA', 'ENTITYB', 'TYPEB', 'IDB', 'DATABASEB', 'EFFECT', 
         'MECHANISM', 'RESIDUE', 'SEQUENCE', 'TAX_ID', 'CELL_DATA', 'TISSUE_DATA', 'MODULATOR_COMPLEX', 
         'TARGET_COMPLEX', 'MODIFICATIONA', 'MODASEQ', 'MODIFICATIONB', 'MODBSEQ', 'PMID',
         'DIRECT', 'NOTES', 'ANNOTATOR', 'SENTENCE', 'SIGNOR_ID', 'SCORE']
    url = f"https://signor.uniroma2.it/getData.php"
    
    try:
        print(f"Выполняется запрос к URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        text = response.text.strip()

        # Определяем разделитель автоматически
        if "\t" in text:
            sep = "\t"
        else:
            sep = ","

        dataframe = pd.read_csv(StringIO(text), sep=sep, header=0, names=col, index_col=False)

        if dataframe.empty:
            print("DataFrame пуст —  ID нет.")
            return None
        
        allowed_types = ['protein', 'protein family', 'fusion protein']
        protein_data = dataframe[
            dataframe['TYPEA'].isin(allowed_types) & 
            dataframe['TYPEB'].isin(allowed_types)
        ]
        protein_data['effect'] = None
        conditions = [
            protein_data['EFFECT'].str.contains('up-regulates', na=False),
            protein_data['EFFECT'].str.contains('down-regulates', na=False),
            protein_data['EFFECT'].str.contains('unknown', na=False)
        ]

        choices = [
            'activate',
            'inhibit',
            'unknown'
        ]
        mechanism_map = {'Phospho' : 'phosphorylation', 'Acetyl' : 'acetylation', 'Methyl' : 'methylation', 
                         'Carboxy':'carboxylation','Palmitoyl':'palmitoylation',
                         'Hydroxylation':'hydroxylation', 'ADP-Ribosyl':'ADP-ribosylation',
                         'Trimethyl':'trimethylation','Nitrosyl':'s-nitrosylation', 'Oxidation':'oxidation',
                        'Hex':'glycosylation','Fuc': 'glycosylation'}
        #'':'glycosylation', 'GG' : 'ubiquitination', '' : 'polyubiquitination','' : 'sumoylation','': 'monoubiquitination','':'deglycosylation'

        # Заполнение новой колонки 'effect'
        protein_data['effect'] = np.select(conditions, choices, default='unknown')
        protein_data['position_in_protein'] = protein_data['RESIDUE'].str.findall(r'\d+').str[0]
        #protein_data['position_in_protein'] = pd.to_numeric(protein_data['position_in_protein'], errors='coerce').astype('Int64')
        protein_data['position_in_protein'] = protein_data['position_in_protein'].astype('Int64')
        df_result['position_in_protein'] = df_result['position_in_protein'].astype('Int64')
        df_result['mechanism'] = None
        df_result['mod_name']=df_result['Modification'].apply(lambda x: x.split('@')[0])
        df_result['mechanism'] = df_result['mod_name'].replace(
            {pat: mech for pat, mech in mechanism_map.items()},
            regex=True
        )

        df_result.loc[:,'disease_effect'] = None
        df_result.loc[df_result['median_group1_coef'] > df_result['median_group2_coef'],'disease_effect'] = 'increase'
        df_result.loc[df_result['median_group1_coef'] < df_result['median_group2_coef'],'disease_effect'] = 'decrease'
        df_result.loc[df_result['median_group1_coef'] == df_result['median_group2_coef'],'disease_effect'] = 'no_change'
        protein_data=protein_data.rename(columns = {'MECHANISM':'mechanism'})
        common_df_target = df_result.merge(protein_data, left_on=['id_prot','mechanism','position_in_protein'],
                                      right_on=['IDB','mechanism','position_in_protein'], how='inner')
        df_detarget = df_result.copy()
        df_detarget['mechanism'] = df_detarget['mechanism'].apply(
            lambda x: 'de' + x if pd.notna(x) else x
        )
        common_df_detarget = df_detarget.merge(protein_data, left_on=['id_prot','mechanism','position_in_protein'],
                                      right_on=['IDB','mechanism','position_in_protein'], how='inner')
        
        common_df_ensyme = df_result.merge(protein_data, left_on=['id_prot','mechanism','position_in_protein'],
                                      right_on=['IDA','mechanism','position_in_protein'], how='inner')
        common_df_deensyme = df_detarget.merge(protein_data, left_on=['id_prot','mechanism','position_in_protein'],
                                      right_on=['IDA','mechanism','position_in_protein'], how='inner')
        df = pd.concat(
            [common_df_target, common_df_detarget, common_df_ensyme, common_df_deensyme],
            ignore_index=True
        )
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Error: {e}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Error: {e}")
        return None



def grafs(common_df, id_prot, output_dir):
    G = nx.DiGraph()

    for _, r in common_df.iterrows():
        G.add_edge(
            r.IDA, r.IDB,
            kind=r.mechanism,
            site=r.position_in_protein,
            effect=r.effect,
            weight=r.SCORE
        )

        if r.IDA in id_prot:
            G.nodes[r.IDA]["disease_effect"] = r.disease_effect
        else:
            G.nodes[r.IDA]["disease_effect"] = "no findings"

        if r.IDB in id_prot:
            G.nodes[r.IDB]["disease_effect"] = r.disease_effect
        else:
            G.nodes[r.IDB]["disease_effect"] = "no findings"

    net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white", directed=True)

    for n, attrs in G.nodes(data=True):
        disease_effect = attrs.get("disease_effect")

        if disease_effect == "increase":
            color = "red"
            shape = "triangle"
        elif disease_effect == "decrease":
            color = "blue"
            shape = "box"
        elif disease_effect == "no_change":
            color = "green"
            shape = "dot"
        else:
            color = "gray"
            shape = "dot"

        node_label = n
        node_size = 20

        net.add_node(
            n, label=node_label, title=f"{n}, disease_effect={disease_effect}",
            color=color, size=node_size, shape=shape
        )


    for u, v, attrs in G.edges(data=True):
        edge_label = ""
        edge_color = ""

        if attrs.get("effect") == "activate":
            edge_color = "red"
            edge_label = "+"
        elif attrs.get("effect") == "inhibit":
            edge_color = "blue"
            edge_label = "-"
        else:
            edge_color = "gray"
            edge_label = "?"

        net.add_edge(u, v, title=str(attrs), label=edge_label, color=edge_color, arrows="to")

    # ===== legent =====
    legend_opts = dict(physics=False, fixed=True)
    net.add_node(
        'legend_bg',
        label='',  # без текста
        shape='box',
        color={'background': '#333333', 'border': '#333333'},
        x=-1270, y=-750,
        widthConstraint=200, heightConstraint=550,
        physics=False, fixed=True
    )

    net.add_node('legend_edges', label='Edge legend:', shape='text',
                 x=-1300, y=-1000, font={'color': 'yellow', 'size': 25}, **legend_opts)
    net.add_node('legend_nodes', label='Node legend:', shape='text',
                 x=-1300, y=-720, font={'color': 'yellow', 'size': 25}, **legend_opts)

    net.add_node('legend_edge1', label='+ (активация)', color='red', shape='dot',
                 x=-1280, y=-960, **legend_opts)
    net.add_node('legend_edge2', label='- (ингибирование)', color='blue', shape='dot',
                 x=-1280, y=-880, **legend_opts)
    net.add_node('legend_edge3', label='? (неизвестно)', color='gray', shape='dot',
                 x=-1280, y=-800, **legend_opts)

    net.add_node('legend_node1', label='increase PTM in AD', color='red', shape='triangle',
                 x=-1280, y=-680, **legend_opts)
    net.add_node('legend_node2', label='decrease PTM in AD', color='blue', shape='box',
                 x=-1280, y=-600, **legend_opts)
    net.add_node('legend_node3', label='no changes', color='green', shape='dot',
                 x=-1280, y=-540, **legend_opts)

    net.save_graph(os.path.join(output_dir,"network.html"))
    
    