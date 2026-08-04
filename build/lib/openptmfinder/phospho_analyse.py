from os import listdir
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stat
from statistics import median, mean, stdev
import glob
from pyteomics import mgf, mass, fasta, pepxml
from scipy import stats as scipy_stats


def load_fasta_dict(fasta_file):
    fasta_dict = {}
    with fasta.read(fasta_file) as db:
        for descr, seq in db:
            if 'DECOY_' in descr or 'rev_' in descr:
                continue
            prot_id = descr.split('|')[1]
            fasta_dict[prot_id] = seq
    return fasta_dict


def target_decoy_filter(psm_df, fdr_threshold=0.01):
    df = psm_df.copy()

    df['is_decoy'] = df['proteins'].astype(str).str.contains(r'DECOY|rev_', case=False, regex=True)
    is_decoy_bool = df['is_decoy'].astype(bool)
    
    df = df.sort_values('sage_discriminant_score', ascending=False).reset_index(drop=True)

    df['decoy_cum'] = is_decoy_bool.astype(int).cumsum()
    df['target_cum'] = (~is_decoy_bool).astype(int).cumsum()
    
    df['FDR'] = df['decoy_cum'] / df['target_cum']
    df['qvalue'] = df['FDR'][::-1].cummin()[::-1]
    
    # Фильтруем по заданному порогу FDR
    df_filtered = df[(df['qvalue'] <= fdr_threshold) & (~df['is_decoy'])]
    
    return df_filtered

    
def map_mod_positions(results, fasta_link):
    fasta_dict = load_fasta_dict(fasta_link)
    df = results.copy()
    df['proteins'] = df['proteins'].str.split(';')
    rows = []

    for row in df.itertuples(index=False):
        clean_pep = row.peptide_clean
        mod_positions_in_pep = [m.start() for m in re.finditer(r'\[\+79\.', row.peptide_phospho)]
        if not mod_positions_in_pep:
            continue

        for prot in row.proteins:
            prot_id = prot.split('|')[1]
            seq = fasta_dict.get(prot_id)
            if not seq:
                continue

            start = seq.find(clean_pep)
            if start == -1:
                continue

            for pos in mod_positions_in_pep:
                rows.append({
                    **row._asdict(),
                    'protein': prot_id,
                    'position_in_protein': start + pos
                })
            
    df = pd.DataFrame(rows)
    return df

