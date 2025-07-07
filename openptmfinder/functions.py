import re
import json
import logging
import pandas as pd
import numpy as np
from pyteomics import pepxml, mzml, fasta
from statsmodels.stats.multitest import multipletests
from xml.etree import ElementTree as ET
import glob
import os
from deeplc import DeepLC, FeatExtractor
from scipy import stats as scipy_stats


#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_unimod_interpretations(interpretation_file):
    """Loads data from a Unimod interpretation file."""
    unimod = {}
    isotope = {}
    try:
        with open(interpretation_file, 'r') as f:
            fcc_data = json.load(f)
            for section, commands in fcc_data.items():
                for mass in commands:
                    if mass['type'] == 'unimod':
                        unimod[section] = commands
                    elif mass['type'] == 'isotope':
                        isotope[section] = commands
    except Exception as e:
        logger.error(f"Error loading file {interpretation_file}: {e}")
    return unimod, isotope, fcc_data

def unimod_name(n, xml_text):
    """Extracts the modification name and corresponding data from XML."""
    try:
        subline = re.search(f'modifications_row.*record_id="{n}".*?>', xml_text).group(0)
        mod=subline.split(' code_name=')[1].split(' ')[0][1:-1]
        mod = mod.replace('&gt;', '')
        
        t = {}
        for match in re.finditer(f'specificity_row.*mod_key="{n}".*?>', xml_text, re.IGNORECASE):
            classifications_key = re.sub(r'\D', '', match.group().split('classifications_key=')[1].split(' ')[0])
            one_letter = re.sub(r'[^a-zA-Z]', '', match.group().split('one_letter=')[1].split(' ')[0])
            t[one_letter] = classifications_key
        
        return mod, t
    except AttributeError as e:
        logger.warning(f"Unable to find information about unimod with record_id={n}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error retrieving data for unimod with record_id={n}: {e}")
        return None, None

def unimod_reads(xml_file):
    """Reads an XML file and returns its contents as a string."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode')
    except Exception as e:
        logger.error(f"Error reading XML file {xml_file}: {e}")
        return ""

def create_unimod_dataframe(interpretation_file, xml_file):
    """Creates a DataFrame with unimod information."""
    unimod, isotope, fcc_data = load_unimod_interpretations(interpretation_file)
    line = unimod_reads(xml_file)

    # Инициализация списков для хранения данных
    name_mod = []
    type_mod = []
    common = []
    massmod = []
    accession = []

    # Обработка unimod
    for section, commands in unimod.items():
        m = []
        acc = []
        tt = []
        for mass in commands:
            if mass['type'] == 'unimod':
                n = mass['label'].split('=')[2].split('"')[0]
                mod_data = unimod_name(n, line)
                if mod_data is not None:
                    mod, t = mod_data
                    acc.append(n)
                    m.append(mod)
                    tt.append(t)
                else:
                    logger.warning(f"Skipped unimod annotation with record_id={n} due to error retrieving information.")
        name_mod.append(m)
        type_mod.append(tt)
        massmod.append(section)
        common.append(commands)
        accession.append(acc)

    # Обработка isotope
    for mass, interpret in isotope.items():
        for inter in interpret:
            if inter['type']=='isotope':
                iso_list = fcc_data[str(inter['ref'][0])]
                m = []
                tt = []
                p = []
                for val in iso_list:
                    if val['type'] == 'unimod':
                        n = val['label'].split('=')[2].split('"')[0]
                        try:
                            mod, t = unimod_name(n, line)
                            if mod and t:
                                m.append(mod)
                                tt.append(t)
                                p.append(n)
                        except Exception as e:
                            logger.warning(f"Error processing isotope for {n}: {e}")

                name_mod.append(m)
                type_mod.append(tt)
                massmod.append(mass)
                common.append(interpret)
                accession.append(p)

    # Создание DataFrame
    df = pd.DataFrame({
        'unimod_name': name_mod,
        'type_modification': type_mod,
        'accession': accession,
        'massmod': massmod,
        'interpretations': common
    })

    # Фильтрация строк, где unimod_name пустое или содержит только пустые строки
    df = df[df['unimod_name'].apply(lambda x: len(x) > 0 and x != [''])]

    logger.info(f"A DataFrame with {len(df)} rows is created.")
    return df

def dataframe_start(mod1, file, name_modifications, link_data, localization_score_threshold):
    """Creates a DataFrame for a catalog based on localization rating and modifications."""
    try:
        cataloque = pd.DataFrame()
        file_reader = pd.read_csv(link_data + str(file), sep="\t", engine='python')  # engine='python' явно указан
        file_reader = file_reader[file_reader['localization score'] > localization_score_threshold]
        file_reader['select'] = file_reader['top isoform'].apply(lambda x: re.findall(mod1, x))
        file_reader = file_reader[file_reader['select'].apply(len) > 0]
        file_reader = file_reader.reset_index(drop=True)

        cataloque = file_reader.copy()
        cataloque['Modification'] = name_modifications

        # Попытка преобразования имени файла в число
        try:
            cataloque['file_mass'] = float(file[:-4])  # Преобразуем часть имени файла в число
        except ValueError:
            logger.error(f"Error: Cannot convert file name '{file}' to a number. Check the file name format.")
            raise  # Перебрасываем исключение для остановки выполнения

        #logger.info(f'Каталог сформирован с {len(cataloque)} записями.')
        return cataloque
    except Exception as e:
        logger.error(f"Error creating directory for file {file}: {e}")
        raise

def cataloque_create(unimod, name_of_modification, type_of_modification, link_data, localization_score_threshold):
    """Создает каталог с модификациями и аминокислотами."""
    cataloque = pd.DataFrame()
    unimod_search=pd.DataFrame(columns=['modifiction', 'aminoacid','mass_shift','type','accession_unimod'])
    name_of_modification = re.split(r'\s*,\s*',name_of_modification)
    type_of_modification = re.split(r'\s*,\s*',type_of_modification)
    try:
        for line in unimod.itertuples():
            a = line.massmod
            file = f'{a}.csv'
            for ind, mod in enumerate(line.unimod_name):
                name_modifications = mod if '"' not in mod else mod[:-1]
                for name_mod in name_of_modification:
                    if name_modifications == name_mod:
                        for amino, number in line.type_modification[ind].items(): #учитывать номер и тип модификации
                            if number=='2':
                                full = int(round(float(a), 0))
                                sign = '+' if full > 0 else ''
                                if amino == 'Cterm':
                                    mod1 = rf"\[\{sign}{full}\]\.-"
                                elif amino == 'Nterm':
                                    mod1 = rf"-\.[A-Z]\[\{sign}{full}\]"
                                else:
                                    mod1 = rf"{amino}\[\{sign}{full}\]"
                                new_row = {'modifiction': name_mod, 'aminoacid': mod1, 'mass_shift': a, 
                                           'type': number, 'accession_unimod':line.accession[ind]}
                                unimod_search = unimod_search.append(new_row, ignore_index=True)

                                df = dataframe_start(mod1, file, name_modifications + '@' + amino, link_data, localization_score_threshold)
                                df['accession_unimod'] = line.accession[ind]
                                cataloque = pd.concat([cataloque, df])
                            #logger.info(f"Добавлено {len(df)} записей в каталог.")

        cataloque['peptide'] = cataloque['top isoform'].apply(
            lambda x: re.sub(r'[^A-Z]', '', x[1:-1]) if x[-3] == 'R' or x[-3] == 'K' else re.sub(r'[^A-Z]', '', x[1:])
        )
        
        file_reader = pd.read_csv(link_data + str('+0.0000.csv'), sep="	")
        file_reader['peptide'] = file_reader['peptide'].apply(
            lambda x: re.sub(r'[^A-Z]', '', x[1:-1]) if x[-3] == 'R' or x[-3] == 'K' else re.sub(r'[^A-Z]', '', x[1:])
        )
        file_reader = file_reader[file_reader['peptide'].isin(list(cataloque['peptide']))]
        file_reader['Modification'] = 'reference'
        file_reader['file_mass'] = 0

        cataloque.rename(columns={'top isoform': 'modified_peptide'}, inplace=True)
        cataloque = pd.concat([file_reader, cataloque], ignore_index=True)

        # Расчет позиций
        for ind, line in enumerate(cataloque.itertuples()):
            pos = 0
            if line.file_mass != 0:
                if 'Nterm' in line.Modification:
                    pos = 1
                elif 'Cterm' in line.Modification:
                    pos = len(line.peptide)
                else:
                    pos = line.modified_peptide.find(line.select[0]) - 1    #находит только первую модификацию на пептиде???
                cataloque.loc[ind, 'position_mod'] = pos
                cataloque.loc[ind, 'for_prediction'] = str(pos) + '|' + line.Modification.split('@')[0]

        logger.info(f"Catalog with modifications created, total entries: {len(cataloque)}.")
        return cataloque, unimod_search

    except Exception as e:
        logger.error(f"Error creating modifications directory: {e}")
        raise


def intensity(link_mzml, cataloque, logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    unique_files = cataloque['file_name'].unique()
    logger.info(f"Found {len(unique_files)} unique mzML files.")
    cataloque['intensity']=None 
    cataloque['m/z']=None

    for ind, f in enumerate(unique_files):
        mzml_path = os.path.join(link_mzml, f + '.mzML')

        if not os.path.isfile(mzml_path):
            logger.warning(f"[{ind}] File not found: {mzml_path}, skipping.")
            continue

        logger.info(f"[{ind}] Reading mzML file: {f}")
        try:
            with mzml.read(mzml_path, use_index=True) as file:
                sub_df = cataloque[cataloque['file_name'] == f]

                for idx, row in sub_df.iterrows():
                    spectrum_index = int(row['index spectrum']) - 1

                    try:
                        spectrum = file[spectrum_index]
                        
                        cataloque.at[idx, 'intensity'] = spectrum['intensity array']
                        cataloque.at[idx, 'm/z'] = spectrum['m/z array']
                    except IndexError:
                        logger.error(f"Spectrum index {spectrum_index} out of range in file {f}. Skipping.")
                        continue

        except Exception as e:
            logger.exception(f"Error processing file {mzml_path}: {e}")

    logger.info("Processing of all mzML files completed.")
    return cataloque


def process_pepxml_files(cataloque, pepxml_dir, mass_tolerance=0.012, fdr_threshold=0.05,sorting_pepxml=True):
    """Обрабатывает pepXML файлы и фильтрует по массе и FDR."""
    try:
        modmass = cataloque['file_mass'].unique()
        
        peptides_by_mass = {
            mod: set(cataloque.loc[cataloque['file_mass'] == mod, 'peptide'])
            for mod in modmass
        }
        if len(pepxml_dir)==1 and '.pepXML' not in pepxml[0]:
            xml_files = glob.glob(f"{pepxml_dir}/*.pepXML")
        else:
            xml_files=pepxml_dir
            
        if xml_files:
            logger.info(f'Total {len(xml_files)} pepxml files found.')
        else:
            logger.warning(f"No pepXML files found in directory: {pepxml_dir}")
            return pd.DataFrame()

        dfs = []
        for i, file in enumerate(xml_files):
            try:
                ftf = pepxml.DataFrame(file)
                logger.info(f"{i}, File processing: {file}")

                # Предфильтрация по всем modmass сразу
                mask = np.zeros(len(ftf), dtype=bool)
                for mod in modmass:
                    mask |= (np.abs(ftf['massdiff'] - mod) <= mass_tolerance)

                filtered_ftf = ftf[mask]

                for mod in modmass:
                    # Узкое окно по mod
                    mod_mask = (np.abs(filtered_ftf['massdiff'] - mod) <= mass_tolerance)
                    df1 = pepxml.filter_df(filtered_ftf[mod_mask], fdr=fdr_threshold)
                    if sorting_pepxml==True:
                        df1=df1[(df1['hyperscore']>=20) & (df1['expect']<=0.05)]
                    if df1.empty:
                        continue

                    df1 = df1[df1['peptide'].isin(peptides_by_mass[mod])]
                    if not df1.empty:
                        df1['file_mass'] = mod
                        dfs.append(df1)

            except Exception as e:
                logger.error(f"Error processing file {file}: {e}", exc_info=True)
                continue

        if not dfs:
            logger.warning("No matching records found after processing all files.")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        df['index spectrum'] = df['spectrum'].str.split('.').str[1].astype(int)
        df['file_name'] = df['spectrum'].str.split('.').str[0]
        df_full=df.merge(cataloque,how='left',on=['peptide','file_mass'])

        return df_full

    except Exception as e:
        logger.error(f"Error processing pepXML files: {e}", exc_info=True)
        raise


def prediction_rt(pepxml_psms: pd.DataFrame) -> pd.DataFrame:
    calibration_set = pepxml_psms[(pepxml_psms['file_mass'] != 0) & (pepxml_psms['spectrum_x']==pepxml_psms['spectrum_y'])][
        ['peptide', 'for_prediction', 'retention_time_sec']
    ]

    df_for_calib = pd.DataFrame({
        'seq': calibration_set['peptide'],
        'modifications': calibration_set['for_prediction'],
        'tr': calibration_set['retention_time_sec']
    }).drop_duplicates(subset=['seq', 'modifications'])
    logger.info(f'Create a dataframe for calibration {len(df_for_calib)}')
    
    if len(df_for_calib)>=50:
        dlc = DeepLC(verbose=False, pygam_calibration=False)
        dlc.calibrate_preds(seq_df=df_for_calib)
        logger.info(f'The model is calibrated')

        predict_set = pepxml_psms[
            (pepxml_psms['file_mass'] != 0) &
            (pepxml_psms['spectrum_x']!=pepxml_psms['spectrum_y'])
        ]

        df_for_predict = pd.DataFrame({
            'seq': predict_set['peptide'],
            'modifications': predict_set['for_prediction']
        })

        df_for_predict['predicted_RT'] = dlc.make_preds(seq_df=df_for_predict)
        return df_for_predict
    else:
        logger.warning("Not enough psm for calibration. Step skipped.")
        return None


def name_tmt(value: float, type_tmt: str) -> str:
    c = float(value)
    
    if 126 <= c < 127:
        return 'tmt_126'
    elif 127.1225 <= c <= 127.1275:
        return 'tmt_127N'
    elif 127.1275 < c <= 127.135:
        return 'tmt_127C'
    elif 128.125 <= c <= 128.131:
        return 'tmt_128N'
    elif 128.131 < c <= 128.138:
        return 'tmt_128C'
    elif 129.127 <= c <= 129.134:
        return 'tmt_129N'
    elif 129.134 < c <= 129.141:
        return 'tmt_129C'
    elif 130.130 <= c <= 130.138:
        return 'tmt_130N'
    elif 130.138 < c <= 130.148:
        return 'tmt_130C'
    elif 131 <= c < 132:
        if type_tmt == 'TMT10plex':
            return 'tmt_131'
        elif type_tmt == 'TMT11plex':
            if 131.12 <= c <= 131.14:
                return 'tmt_131N'
            elif 131.141 <= c <= 131.2:
                return 'tmt_131C'
    return 'None'


def tags_annotation(cataloque: pd.DataFrame, type_tmt: str) -> pd.DataFrame:
    r = 4  # rounding precision
    for i in range(len(cataloque)):
        try:
            mz_values = cataloque.loc[i, 'm/z']
            intensities = cataloque.loc[i, 'intensity']

            for j, mass_str in enumerate(mz_values):
                if not mass_str:
                    continue
                mass = float(round(mass_str,r))
                if 126 < mass < 132 and j <= len(intensities):
                    key = name_tmt(mass, type_tmt)
                    if key != 'None':
                        cataloque.at[i, f'intensity_{key}'] = intensities[j]
        except Exception as e:
            print(f"Error in TMT annotation in line {i}: {e}")
            continue
    return cataloque

def noisygaus(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b

def calibrate_RT_gaus(bwidth, mass_left, mass_right, true_md):

    bbins = np.arange(-mass_left, mass_right, bwidth)
    H1, b1 = np.histogram(true_md, bins=bbins)
    b1 = b1 + bwidth
    b1 = b1[:-1]


    popt, pcov = curve_fit(noisygaus, b1, H1, p0=[1, np.median(true_md), bwidth * 5, 1])
    mass_shift, mass_sigma = popt[1], abs(popt[2])
    return mass_shift, mass_sigma, pcov[0][0]

def calibrate_RT_gaus_full(rt_diff_tmp):
    RT_left = -min(rt_diff_tmp)
    RT_right = max(rt_diff_tmp)

    try:
        start_width = (stats.scoreatpercentile(rt_diff_tmp, 95) - stats.scoreatpercentile(rt_diff_tmp, 5)) / 100
        XRT_shift, XRT_sigma, covvalue = calibrate_RT_gaus(start_width, RT_left, RT_right, rt_diff_tmp)
    except:
        start_width = (stats.scoreatpercentile(rt_diff_tmp, 95) - stats.scoreatpercentile(rt_diff_tmp, 5)) / 50
        XRT_shift, XRT_sigma, covvalue = calibrate_RT_gaus(start_width, RT_left, RT_right, rt_diff_tmp)
    if np.isinf(covvalue):
        XRT_shift, XRT_sigma, covvalue = calibrate_RT_gaus(0.1, RT_left, RT_right, rt_diff_tmp)
    if np.isinf(covvalue):
        XRT_shift, XRT_sigma, covvalue = calibrate_RT_gaus(1.0, RT_left, RT_right, rt_diff_tmp)
    return XRT_shift, XRT_sigma, covvalue

#нужно создать функцию по опредленияю образцов
def samples_annotation(full_df: pd.DataFrame, group_df_link: str) -> pd.DataFrame:
    try:
        group_df = pd.read_csv(group_df_link)
        full_df_group = full_df.merge(group_df, how='left', on='file_name')
    except:
        group_df = pd.read_csv(group_df_link,sep=';')
        full_df_group = full_df.merge(group_df, how='left', on='file_name')
    logger.info(f"There are no annotations for {full_df_group['TMT_group1'].isna().sum()} files." )
    full_df_group=full_df_group[full_df_group['TMT_group1'].notna()]
    full_df_group['TMT_group1'] = full_df_group['TMT_group1'].apply(
        lambda x: re.split(r'\s*,\s*', re.sub(r"[\'\[\]]", "", x)))
    full_df_group['TMT_group2'] = full_df_group['TMT_group2'].apply(
        lambda x: re.split(r'\s*,\s*', re.sub(r"[\'\[\]]", "", x)))
    full_df_group['mix_channels'] = full_df_group['mix_channels'].apply(
        lambda x: re.split(r'\s*,\s*', re.sub(r"[\'\[\]]", "", x)))

    return full_df_group


def statistics(stat: pd.DataFrame, calc_pval: bool = True, min_group_for_stats: int = 3) -> pd.DataFrame:
    columns = ['TMT_group1', 'TMT_group2']
    stat['position_in_protein'] = stat['position_in_protein'].astype(str)
    #stat=stat[,'position_in_protein']=

    stat = stat.groupby(['id_prot', 'position_in_protein', 'batch']).agg(list).reset_index()

    for column in columns:
        stat[f'intensity_{column}'] = pd.NA
        for ind, line in enumerate(stat[column]):
            intens = []
            for dis in line[0]:  # предполагается, что внутри column хранятся вложенные списки
                intens.append(max(stat.at[ind, f'norm_intens_{dis}']))
            stat.at[ind, f'intensity_{column}'] = intens

    if calc_pval:
        stat = stat.groupby(['id_prot', 'position_in_protein']).agg(list).reset_index()

        for column in columns:
            stat[f'intensity_{column}'] = stat[f'intensity_{column}'].apply(lambda x: sum(x, []))

        for ind in range(len(stat)):
            group1 = stat.loc[ind, 'intensity_TMT_group1']
            group2 = stat.loc[ind, 'intensity_TMT_group2']

            group1 = np.log2([x for x in group1 if not pd.isna(x)])
            group2 = np.log2([x for x in group2 if not pd.isna(x)])

            if len(group1) >= min_group_for_stats and len(group2) >= min_group_for_stats:
                try:
                    res = scipy_stats.permutation_test((group1, group2), statistic=lambda x, y: np.mean(x) - np.mean(y),
                       permutation_type='independent', alternative='two-sided',
                       n_resamples=10000, random_state=42)
                    pval = res.pvalue
                    _, p_log = scipy_stats.ttest_ind(group1, group2, nan_policy='omit', equal_var=False)
                    _, p_mn = scipy_stats.mannwhitneyu(group1, group2, nan_policy='omit',alternative='two-sided')
                    fc_median = np.median(group1) / np.median(group2)
                    
                    stat.at[ind, 'T_test_p_value'] = p_log
                    stat.at[ind, 'mannwhitneyu_p_value'] = p_mn
                    stat.at[ind, 'permutation_p_value'] = pval
                    stat.at[ind, 'FC_median'] = fc_median
                except Exception as e:
                    print(f"Statistics error in line {ind}: {e}")
                    
        stat=stat[stat['permutation_p_value']>0].sort_values('permutation_p_value')
        flag,pvalue,alfa1,alfa2=multipletests(list(stat['permutation_p_value'].dropna()), alpha=0.05, method='fdr_bh', is_sorted = False)
        stat['pvalue_correct']=list(pvalue)

    return stat



def tmt_normalization(df1: pd.DataFrame) -> pd.DataFrame:
    sp_tmt = sum(df1[['TMT_group1', 'TMT_group2', 'mix_channels']].iloc[0].tolist(),[])
    columns=[f'intensity_tmt_{x}' for x in sp_tmt]
    dop_columns=['Modification','modified_peptide_y','spectrum_x','batch','TMT_group1',
                 'TMT_group2', 'mix_channels','id_prot','position_in_protein']
    df=df1[columns+dop_columns].copy()

    for ind, line in enumerate(df['modified_peptide_y']):
        mix = [f'intensity_tmt_{x}' for x in df.loc[ind, 'mix_channels']]
        for tmt in sp_tmt:
            df.loc[ind,f'intensity_tmt_{tmt}'] = df.loc[ind,f'intensity_tmt_{tmt}']/(df.loc[ind,mix].mean())
    #df.replace(np.inf, 0)

    for batch in df['batch'].unique():
        df_batch = df[df['batch'] == batch]
        for tmt in sp_tmt:
            median = df_batch[f'intensity_tmt_{tmt}'].median()
            df.loc[df['batch'] == batch, f'norm_intens_{tmt}'] = df[f'intensity_tmt_{tmt}'] / median
    for tmt in sp_tmt:
        del df[f'intensity_tmt_{tmt}']

    return df


def sorting_psms(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int, list]:
    df = df[df['Modification']!='reference'].copy()
    cols = sum(df[['TMT_group1', 'TMT_group2']].iloc[0].tolist(), [])
    stat = {f'intensity_tmt_{col}': 0 for col in cols}
    num = 0
    delete_indices = []

    for index, row in df.iterrows():
        ad = [f'intensity_tmt_{x}' for x in row['TMT_group1']]
        control = [f'intensity_tmt_{x}' for x in row['TMT_group2']]
        mix = [f'intensity_tmt_{x}' for x in row['mix_channels']]
        
        mean_mix = row[mix].mean()
        median_all = np.median(row[[f'intensity_tmt_{x}' for x in cols]])

        if mean_mix < median_all * 0.5:
            df.loc[index, mix] = pd.NA
            continue

        for group in [ad, control]:
            row_median = np.median(row[group])
            mask = (row[group] < row_median * 0.5) & (row[group] < mean_mix * 0.5)
            mask_nan = row[group].isna()

            if mask.any():
                affected = [col for col, val in zip(group, mask) if val]
                try:
                    df.loc[index, affected] = row_median
                    for tag in affected:
                        stat[tag] += 1
                    num += 1
                except Exception as e:
                    print(f"Mask replacement error: {e}, индекс: {index}")

            if mask_nan.any():
                if mask_nan.sum() <= len(mask_nan) / 2:
                    affected_nan = [col for col, val in zip(group, mask_nan) if val]
                    try:
                        df.loc[index, affected_nan] = row_median
                    except Exception as e:
                        print(f"NaN substitution error: {e}, индекс: {index}")
                else:
                    delete_indices.append(index)

    df.drop(index=delete_indices, inplace=True, errors='ignore')
    df.reset_index(drop=True, inplace=True)
    if 'level_0' in df.columns:
        df.drop(columns=['level_0'], inplace=True)

    return df, stat, num, delete_indices


def fasta_concat(df,fasta_file):
    pr=[]
    se=[]
    df['id_prot']=df['protein'].apply(lambda x: str(x).split('|')[1])
    with fasta.read(fasta_file) as db:
        for descr, seq in db:
            pr.append(descr)
            se.append(seq)
    fasta_df=pd.DataFrame()
    fasta_df['protein']=pr
    fasta_df['sequence']=se
    fasta_df['id_prot']=fasta_df['protein'].apply(lambda x: x.split('|')[1])
    df=pd.merge(df,fasta_df,on='id_prot',how='left')
    if 'sequence' in df.columns and 'peptide' in df.columns:
        df['position_in_protein'] = df.apply(
            lambda row: row['sequence'].find(row['peptide']) if pd.notna(row['sequence']) and pd.notna(row['peptide']) else pd.NA,
            axis=1
        )
        df['position_in_protein'] = df.apply(
            lambda row: row['position_mod'] + row['position_in_protein'] if pd.notna(row['position_mod']) and pd.notna(row['position_in_protein']) else pd.NA,
            axis=1
        )
    del df['protein_x']
    del df['protein_descr']
    return df