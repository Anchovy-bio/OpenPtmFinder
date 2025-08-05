import re
import json
import logging
import pandas as pd
import numpy as np
import os
import glob
from pyteomics import pepxml, mzml, fasta
from statsmodels.stats.multitest import multipletests
from xml.etree import ElementTree as ET
from deeplc import DeepLC, FeatExtractor
from scipy import stats as scipy_stats
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit


logger = logging.getLogger(__name__)


def load_unimod_interpretations(interpretation_file):
    """Loads data from a Unimod interpretation file."""
    unimod = {}
    isotope = {}
    fcc_data = {}
    try:
        with open(interpretation_file, 'r', encoding='utf-8') as f:
            fcc_data = json.load(f)
            for section, commands in fcc_data.items():
                for mass in commands:
                    if mass['type'] == 'unimod':
                        unimod[section] = commands
                    elif mass['type'] == 'isotope':
                        isotope[section] = commands
    except FileNotFoundError:
        logger.error(f"Error: File not found at {interpretation_file}.")
        return None, None, None
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON in file {interpretation_file}.")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading file {interpretation_file}: {e}")
        return None, None, None
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
    except AttributeError:
        # Это может произойти, если регулярное выражение не находит совпадение
        logger.warning(f"Unable to find information about unimod with record_id={n}. Skipped.")
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
    if unimod is None or isotope is None or fcc_data is None:
        return pd.DataFrame()
        
    line = unimod_reads(xml_file)
    if not line:
        return pd.DataFrame()

    rows_data = []

    def process_modifications(mass_dict, mod_type):
        for section, commands in mass_dict.items():
            for command in commands:
                if command['type'] == 'unimod':
                    n = command['label'].split('=')[2].split('"')[0]
                    mod_data = unimod_name(n, line)
                    if mod_data is not None:
                        mod, t = mod_data
                        rows_data.append({
                            'unimod_name': [mod],
                            'type_modification': [t],
                            'accession': [n],
                            'massmod': section,
                            'interpretations': [commands]
                        })

    def process_isotopes(isotope_dict):
        for mass, interpret in isotope_dict.items():
            for inter in interpret:
                if inter['type'] == 'isotope' and inter['ref']:
                    iso_list = fcc_data.get(str(inter['ref'][0]))
                    if iso_list:
                        mods = []
                        types = []
                        accessions = []
                        for val in iso_list:
                            if val['type'] == 'unimod':
                                n = val['label'].split('=')[2].split('"')[0]
                                mod_data = unimod_name(n, line)
                                if mod_data:
                                    mod, t = mod_data
                                    mods.append(mod)
                                    types.append(t)
                                    accessions.append(n)
                        if mods:
                            rows_data.append({
                                'unimod_name': mods,
                                'type_modification': types,
                                'accession': accessions,
                                'massmod': mass,
                                'interpretations': [interpret]
                            })
    
    process_modifications(unimod, 'unimod')
    process_isotopes(isotope)

    df = pd.DataFrame(rows_data)
    
    # Фильтрация строк, где unimod_name пустое
    df = df[df['unimod_name'].apply(lambda x: len(x) > 0 and x != [''])]

    logger.info(f"A DataFrame with {len(df)} rows is created.")
    return df

def dataframe_start(mod1, file, name_modifications, link_data, localization_score_threshold):
    """Creates a DataFrame for a catalog based on localization rating and modifications."""
    try:
        file_path = os.path.join(link_data, file)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_reader = pd.read_csv(file_path, sep="\t")  
        file_reader = file_reader[file_reader['localization score'] > localization_score_threshold]
        file_reader['select'] = file_reader['top isoform'].apply(lambda x: re.findall(mod1, x))
        file_reader = file_reader[file_reader['select'].apply(len) > 0].reset_index(drop=True)

        cataloque = file_reader.copy()
        cataloque['Modification'] = name_modifications

        try:
            cataloque['file_mass'] = float(file.replace('.csv', ''))  
        except ValueError:
            logger.error(f"Error: Cannot convert file name '{file}' to a number. Check the file name format.")
            return None

        return cataloque
    except Exception as e:
        logger.error(f"Error creating directory for file {file}: {e}")
        return None

def cataloque_create(unimod, name_of_modification, type_of_modification, link_data, localization_score_threshold):
    """Создает каталог с модификациями и аминокислотами."""
    cataloque = pd.DataFrame()
    unimod_search_list = []
    
    if not isinstance(name_of_modification, list):
        name_of_modification = re.split(r'\s*,\s*',name_of_modification)
    if not isinstance(type_of_modification, list):
        type_of_modification = re.split(r'\s*,\s*',type_of_modification)

    try:
        for line in unimod.itertuples():
            a = line.massmod
            file = f'{a}.csv'
            for ind, mod in enumerate(line.unimod_name):
                name_modifications = mod if '"' not in mod else mod[:-1]
                if name_modifications in name_of_modification:
                    for amino, number in line.type_modification[ind].items():
                        if number in type_of_modification:
                            full = int(round(float(a), 0))
                            sign = '+' if full > 0 else ''
                            if amino == 'Cterm':
                                mod1 = rf"\[\{sign}{full}\]\.-"
                            elif amino == 'Nterm':
                                mod1 = rf"-\.[A-Z]\[\{sign}{full}\]"
                            else:
                                mod1 = rf"{amino}\[\{sign}{full}\]"
                            
                            new_row = {'modifiction': name_modifications, 'aminoacid': mod1, 'mass_shift': a, 
                                       'type': number, 'accession_unimod':line.accession[ind]}
                            unimod_search_list.append(new_row)

                            df = dataframe_start(mod1, file, name_modifications + '@' + amino, link_data, localization_score_threshold)
                            if df is not None and not df.empty:
                                df['accession_unimod'] = line.accession[ind]
                                cataloque = pd.concat([cataloque, df], ignore_index=True)
                                #logger.info(f"Добавлено {len(df)} записей в каталог.")
        
        unimod_search = pd.DataFrame(unimod_search_list)
        
        # Обработка референсного файла
        ref_file = os.path.join(link_data, '+0.0000.csv')
        file_reader = pd.read_csv(ref_file, sep="\t")
        file_reader['peptide'] = file_reader['peptide'].apply(
            lambda x: re.sub(r'[^A-Z]', '', x[1:-1]) if x[-3] == 'R' or x[-3] == 'K' else re.sub(r'[^A-Z]', '', x[1:])
        )
        cataloque['peptide'] = cataloque['top isoform'].apply(
            lambda x: re.sub(r'[^A-Z]', '', x[1:-1]) if x[-3] == 'R' or x[-3] == 'K' else re.sub(r'[^A-Z]', '', x[1:])
        )
        file_reader = file_reader[file_reader['peptide'].isin(cataloque['peptide'].unique())]
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
                    # Потенциальная проблема: find() находит только первое совпадение.
                    # Если на пептиде несколько одинаковых модификаций, может быть ошибка.
                    find_pep=re.sub(r'\{[^}]*\}', '', line.modified_peptide)
                    pos = find_pep.find(line.select[0]) - 1
                cataloque.loc[ind, 'position_mod'] = pos
                cataloque.loc[ind, 'for_prediction'] = str(pos) + '|' + line.Modification.split('@')[0]

        logger.info(f"Catalog with modifications created, total entries: {len(cataloque)}.")
        return cataloque, unimod_search

    except Exception as e:
        logger.error(f"Error creating modifications directory: {e}", exc_info=True)
        return None, None

def process_single_mzml(file_path, sub_df):
    result_rows = []
    error_message = None
    try:
        with mzml.read(file_path, use_index=True) as file:
            for _, row in sub_df.iterrows():
                spectrum_index = int(row['index spectrum']) - 1
                try:
                    spectrum = file[spectrum_index]
                    result_rows.append({
                        'index': row.name,
                        'intensity': spectrum.get('intensity array'),
                        'm/z': spectrum.get('m/z array')
                    })
                except IndexError:
                    error_message = f"Spectrum index {spectrum_index} out of range in file {os.path.basename(file_path)}."
    except Exception as e:
        error_message = f"Error processing file {file_path}: {e}"
    return result_rows, error_message


def intensity(link_mzml, cataloque, n_processes=None):
    if cataloque.empty:
        logger.warning("Input DataFrame is empty. Skipping intensity calculation.")
        return cataloque
        
    unique_files = cataloque['file_name'].unique()
    logger.info(f"Found {len(unique_files)} unique mzML files.")
    
    cataloque['intensity'] = None
    cataloque['m/z'] = None

    tasks = []
    for f in unique_files:
        mzml_path = os.path.join(link_mzml, f + '.mzML')
        if not os.path.isfile(mzml_path):
            logger.warning(f"File not found: {mzml_path}, skipping.")
            continue
        sub_df = cataloque[cataloque['file_name'] == f].copy()
        tasks.append((mzml_path, sub_df))

    results = []
    n_proc = n_processes or os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        futures = {executor.submit(process_single_mzml, path, df): path for path, df in tasks}
        for future in as_completed(futures):
            path = futures[future]
            res, error = future.result()
            if error:
                logger.error(f"Error from child process for file {path}: {error}")
            if res:
                results.extend(res)

    for r in results:
        cataloque.loc[r['index'], 'intensity'] = r['intensity']
        cataloque.loc[r['index'], 'm/z'] = r['m/z']

    logger.info("Processing of all mzML files completed.")
    return cataloque

def process_single_pepxml(file, modmass, peptides_by_mass, mass_tolerance, fdr_threshold, sorting_pepxml):
    error_message = None
    try:
        ftf = pepxml.DataFrame(file)
        #logger.info(f"Reading file: {file}") # Логирование в дочернем процессе убрано.

        mask = np.zeros(len(ftf), dtype=bool)
        for mod in modmass:
            mask |= (np.abs(ftf['massdiff'] - mod) <= mass_tolerance)
        filtered_ftf = ftf[mask]

        dfs = []
        for mod in modmass:
            mod_mask = (np.abs(filtered_ftf['massdiff'] - mod) <= mass_tolerance)
            try:
                df1 = pepxml.filter_df(filtered_ftf[mod_mask], fdr=fdr_threshold)
            except ZeroDivisionError:
                error_message = f"ZeroDivisionError during FDR filtering in file: {os.path.basename(file)}, mod: {mod}"
                continue
            except Exception as e:
                error_message = f"Unexpected error during FDR filtering in file {os.path.basename(file)}: {e}"
                continue
            
            if sorting_pepxml:
                df1 = df1[(df1['hyperscore'] >= 20) & (df1['expect'] <= 0.05)]
            
            if df1.empty:
                continue
            
            df1 = df1[df1['peptide'].isin(peptides_by_mass[mod])]
            if not df1.empty:
                df1['file_mass'] = mod
                dfs.append(df1)
        
        if dfs:
            return pd.concat(dfs), error_message
        else:
            return None, error_message


    except Exception as e:
        error_message = f"Error processing file {file}: {e}"
        return None, error_message


def process_pepxml_files(cataloque, pepxml_dir, mass_tolerance=0.012, fdr_threshold=0.05, sorting_pepxml=True, n_processes=None):
    try:
        if cataloque.empty:
            logger.warning("Input DataFrame is empty. Skipping pepXML processing.")
            return pd.DataFrame()

        modmass = cataloque['file_mass'].unique()
        peptides_by_mass = {
            mod: set(cataloque.loc[cataloque['file_mass'] == mod, 'peptide'])
            for mod in modmass
        }

        if len(pepxml_dir) == 1 and '.pepXML' not in pepxml_dir[0]:
            xml_files = glob.glob(os.path.join(pepxml_dir[0], '*.pepXML'))
        else:
            xml_files = pepxml_dir

        if not xml_files:
            logger.warning(f"No pepXML files found in directory: {pepxml_dir}")
            return pd.DataFrame()
        logger.info(f"Total {len(xml_files)} pepxml files found.")

        n_proc = n_processes or os.cpu_count()
        results = []
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            futures = {
                executor.submit(process_single_pepxml, file, modmass, peptides_by_mass,
                                mass_tolerance, fdr_threshold, sorting_pepxml): file
                for file in xml_files
            }
            for future in as_completed(futures):
                file = futures[future]
                df, error = future.result()
                if error:
                    logger.error(f"Error from child process for file {file}: {error}")
                if df is not None and not df.empty:
                    results.append(df)

        if not results:
            logger.warning("No matching records found after processing all files.")
            return pd.DataFrame()

        df = pd.concat(results, ignore_index=True)
        df['index spectrum'] = df['spectrum'].str.split('.').str[1].astype(int)
        df['file_name'] = df['spectrum'].str.split('.').str[0]
        df_full = df.merge(cataloque, how='left', on=['peptide', 'file_mass'])

        return df_full

    except Exception as e:
        logger.error(f"Error processing pepXML files: {e}", exc_info=True)
        return None


def prediction_rt(pepxml_psms: pd.DataFrame) -> pd.DataFrame:
    if pepxml_psms.empty:
        logger.warning("Input DataFrame is empty. Skipping RT prediction.")
        return None

    # Фильтруем данные для калибровки и предсказания
    calibration_set = pepxml_psms[
        (pepxml_psms['file_mass'] != 0) & 
        (pepxml_psms['spectrum_x'] == pepxml_psms['spectrum_y'])
    ][['peptide', 'for_prediction', 'retention_time_sec']]

    if len(calibration_set) < 50:
        logger.warning("Not enough PSMs for calibration. Skipping RT prediction.")
        return None

    df_for_calib = pd.DataFrame({
        'seq': calibration_set['peptide'],
        'modifications': calibration_set['for_prediction'],
        'tr': calibration_set['retention_time_sec']
    }).drop_duplicates(subset=['seq', 'modifications'])

    logger.info(f'Create a dataframe for calibration {len(df_for_calib)}')

    dlc = DeepLC(verbose=False, pygam_calibration=False)
    dlc.calibrate_preds(seq_df=df_for_calib)
    logger.info('The model is calibrated.')

    pepxml_psms['predicted_RT'] = dlc.make_preds(seq_df=pd.DataFrame({
        'seq': pepxml_psms['peptide'],
        'modifications': pepxml_psms['for_prediction']
    }))
    
    rt_diff_df = pepxml_psms[['for_prediction', 'retention_time_sec', 'predicted_RT']].copy()
    rt_diff_df['rt_diff'] = rt_diff_df['predicted_RT'] - rt_diff_df['retention_time_sec']
    
    calibration_params = {}
    
    # Калибровка и фильтрация для каждой модификации отдельно
    mod_types = rt_diff_df['for_prediction'].unique()
    print(mod_types)
    for mod in mod_types:
        mod_df = rt_diff_df[rt_diff_df['for_prediction'] == mod]
        if len(mod_df) < 50:
            logger.warning(f"Not enough data for modification '{mod}' ({len(mod_df)} entries). Skipping specific RT calibration and filtering.")
            continue
        
        rt_diff_tmp = mod_df['rt_diff'].dropna().values
        if len(rt_diff_tmp) < 50:
            logger.warning(f"Not enough non-NA RT differences for modification '{mod}'. Skipping specific RT calibration.")
            continue

        try:
            XRT_shift, XRT_sigma, _ = _calibrate_single_mod_rt_gaus(rt_diff_tmp)
            calibration_params[mod] = {'shift': XRT_shift, 'sigma': XRT_sigma}
        except Exception as e:
            logger.error(f"Error during RT calibration for modification '{mod}': {e}. Skipping specific calibration.")
            continue
    
    outlier_indices = []
    
    # Применение фильтрации по модификациям
    for mod, params in calibration_params.items():
        mod_df = rt_diff_df[rt_diff_df['for_prediction'] == mod].copy()
        rt_diff_col = mod_df['rt_diff']
        outliers = rt_diff_col[abs(rt_diff_col - params['shift']) >= 3 * params['sigma']].index
        outlier_indices.extend(outliers.tolist())
        logger.info(f"Modification '{mod}': Found {len(outliers)} outliers.")

    if outlier_indices:
        pepxml_psms_filtered = pepxml_psms.drop(index=outlier_indices).reset_index(drop=True)
        logger.info(f"Total number of filtered PSMs: {len(outlier_indices)}. Retained {len(pepxml_psms_filtered)} PSMs.")
        return pepxml_psms_filtered
    else:
        logger.info("No outliers found or not enough data to calibrate. Returning original DataFrame.")
        return pepxml_psms


def _calibrate_single_mod_rt_gaus(rt_diff_tmp):
    RT_left = -min(rt_diff_tmp)
    RT_right = max(rt_diff_tmp)
    try:
        start_width = (scipy_stats.scoreatpercentile(rt_diff_tmp, 95) - scipy_stats.scoreatpercentile(rt_diff_tmp, 5)) / 100
        XRT_shift, XRT_sigma, covvalue = calibrate_RT_gaus(start_width, RT_left, RT_right, rt_diff_tmp)
    except:
        start_width = (scipy_stats.scoreatpercentile(rt_diff_tmp, 95) - scipy_stats.scoreatpercentile(rt_diff_tmp, 5)) / 50
        XRT_shift, XRT_sigma, covvalue = calibrate_RT_gaus(start_width, RT_left, RT_right, rt_diff_tmp)
    if np.isinf(covvalue):
        XRT_shift, XRT_sigma, covvalue = calibrate_RT_gaus(0.1, RT_left, RT_right, rt_diff_tmp)
    if np.isinf(covvalue):
        XRT_shift, XRT_sigma, covvalue = calibrate_RT_gaus(1.0, RT_left, RT_right, rt_diff_tmp)
    return XRT_shift, XRT_sigma, covvalue


def fast_name_tmt(mz: float, type_tmt: str) -> str:
    if 126 <= mz < 127:
        return 'tmt_126'
    elif 127.1225 <= mz <= 127.1275:
        return 'tmt_127N'
    elif 127.1275 < mz <= 127.135:
        return 'tmt_127C'
    elif 128.125 <= mz <= 128.131:
        return 'tmt_128N'
    elif 128.131 < mz <= 128.138:
        return 'tmt_128C'
    elif 129.127 <= mz <= 129.134:
        return 'tmt_129N'
    elif 129.134 < mz <= 129.141:
        return 'tmt_129C'
    elif 130.130 <= mz <= 130.138:
        return 'tmt_130N'
    elif 130.138 < mz <= 130.148:
        return 'tmt_130C'
    elif 131 <= mz < 132:
        if type_tmt == 'TMT10plex':
            return 'tmt_131'
        elif type_tmt == 'TMT11plex':
            if 131.12 <= mz <= 131.14:
                return 'tmt_131N'
            elif 131.141 <= mz <= 131.2:
                return 'tmt_131C'
    return None

def annotate_tmt_chunk(chunk: pd.DataFrame, type_tmt: str, r: int = 4) -> pd.DataFrame:
    tmt_keys = ['tmt_126', 'tmt_127N', 'tmt_127C', 'tmt_128N', 'tmt_128C',
                'tmt_129N', 'tmt_129C', 'tmt_130N', 'tmt_130C', 'tmt_131', 'tmt_131N', 'tmt_131C']
    
    for key in tmt_keys:
        chunk[f'intensity_{key}'] = np.nan

    for i, row in chunk.iterrows():
        try:
            mzs = row['m/z']
            intensities = row['intensity']
            if mzs is None or intensities is None:
                continue

            for mz, intensity in zip(mzs, intensities):
                if mz is None:
                    continue
                mass = round(mz, r)
                if 126 < mass < 132:
                    tag = fast_name_tmt(mass, type_tmt)
                    if tag:
                        chunk.at[i, f'intensity_{tag}'] = intensity
        except Exception:
            continue
    return chunk


def tags_annotation(cataloque: pd.DataFrame, type_tmt: str, n_proc: int = None) -> pd.DataFrame:
    if cataloque.empty:
        logger.warning("Input DataFrame is empty. Skipping TMT annotation.")
        return cataloque
        
    logger.info('Annotation start.')
    cpu_count = n_proc or os.cpu_count()
    chunk_size = max(1, len(cataloque) // cpu_count)
    chunks = [cataloque.iloc[i:i + chunk_size].copy() for i in range(0, len(cataloque), chunk_size)]

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [executor.submit(annotate_tmt_chunk, chunk, type_tmt) for chunk in chunks]
        results = [f.result() for f in as_completed(futures)]

    result_df = pd.concat(results, ignore_index=True)
    logger.info('TMT annotation completed.')
    return result_df


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


def samples_annotation(full_df: pd.DataFrame, group_df_link: str) -> pd.DataFrame:
    if full_df.empty:
        logger.warning("Input DataFrame is empty. Skipping samples annotation.")
        return full_df
        
    try:
        # Пытаемся прочитать с разделителем по умолчанию (',')
        group_df = pd.read_csv(group_df_link)
    except Exception:
        # Если не получилось, пробуем с разделителем ';'
        try:
            group_df = pd.read_csv(group_df_link, sep=';')
        except Exception as e:
            logger.error(f"Error reading grouping file {group_df_link}: {e}")
            return full_df.copy()
            
    full_df_group = full_df.merge(group_df, how='left', on='file_name')
    
    missing_annotations_count = full_df_group['TMT_group1'].isna().sum()
    if missing_annotations_count > 0:
        logger.warning(f"There are no annotations for {missing_annotations_count} files.")
    
    full_df_group = full_df_group[full_df_group['TMT_group1'].notna()].copy()
    
    full_df_group['TMT_group1'] = full_df_group['TMT_group1'].apply(
        lambda x: re.split(r'\s*,\s*', re.sub(r"[\'\[\]]", "", str(x))))
    full_df_group['TMT_group2'] = full_df_group['TMT_group2'].apply(
        lambda x: re.split(r'\s*,\s*', re.sub(r"[\'\[\]]", "", str(x))))
    full_df_group['mix_channels'] = full_df_group['mix_channels'].apply(
        lambda x: re.split(r'\s*,\s*', re.sub(r"[\'\[\]]", "", str(x))))

    logger.info("Samples annotation completed.")
    return full_df_group


def statistics(stat: pd.DataFrame, calc_pval: bool = True, min_group_for_stats: int = 3) -> pd.DataFrame:
    if stat.empty:
        logger.warning("Input DataFrame is empty. Skipping statistics.")
        return stat
        
    columns = ['TMT_group1', 'TMT_group2']
    stat = stat.copy()
    stat['position_in_protein'] = stat['position_in_protein'].astype(str)

    # Группировка для подсчета интенсивности по батчам
    stat_batch_agg = stat.groupby(['Modification','id_prot', 'position_in_protein', 'batch']).agg(list).reset_index()

    for column in columns:
        stat_batch_agg[f'intensity_{column}'] = pd.NA
        for ind, line in stat_batch_agg[column].items():
            intens = []
            if isinstance(line, list) and line: # Проверка, что line - это список и он не пуст
                # line[0] так как в groupby agg list получается [[['tmt_127C', 'tmt_128N']]]
                if isinstance(line[0], list):
                    for dis in line[0]:
                        intens.append(max(stat_batch_agg.at[ind, f'norm_intens_{dis}']))
            stat_batch_agg.at[ind, f'intensity_{column}'] = intens
            
    if not calc_pval:
        return stat_batch_agg
        
    # Повторная группировка для вычисления p-value
    stat = stat_batch_agg.groupby(['Modification','id_prot', 'position_in_protein']).agg(list).reset_index()

    for column in columns:
        stat[f'intensity_{column}'] = stat[f'intensity_{column}'].apply(lambda x: sum(x, []))
    
    p_values = []
    fc_medians = []
    t_test_p_values = []
    mannwhitneyu_p_values = []
    
    for ind in range(len(stat)):
        group1 = stat.loc[ind, 'intensity_TMT_group1']
        group2 = stat.loc[ind, 'intensity_TMT_group2']

        group1 = np.log2([x for x in group1 if pd.notna(x)])
        group2 = np.log2([x for x in group2 if pd.notna(x)])
        
        pval, t_pval, mn_pval, fc_median = np.nan, np.nan, np.nan, np.nan
        
        if len(group1) >= min_group_for_stats and len(group2) >= min_group_for_stats:
            try:
                res = scipy_stats.permutation_test((group1, group2), statistic=lambda x, y: np.mean(x) - np.mean(y),
                   permutation_type='independent', alternative='two-sided',
                   n_resamples=10000, random_state=42)
                pval = res.pvalue
                _, t_pval = scipy_stats.ttest_ind(group1, group2, nan_policy='omit', equal_var=False)
                _, mn_pval = scipy_stats.mannwhitneyu(group1, group2, nan_policy='omit', alternative='two-sided')
                fc_median = np.median(group1) / np.median(group2)
            except Exception as e:
                logger.error(f"Statistics error in line {ind}: {e}")
        
        p_values.append(pval)
        t_test_p_values.append(t_pval)
        mannwhitneyu_p_values.append(mn_pval)
        fc_medians.append(fc_median)

    stat['permutation_p_value'] = p_values
    stat['T_test_p_value'] = t_test_p_values
    stat['mannwhitneyu_p_value'] = mannwhitneyu_p_values
    stat['FC_median'] = fc_medians
    
    stat = stat[stat['permutation_p_value'].notna()].sort_values('permutation_p_value').reset_index(drop=True)
    
    if not stat.empty:
        flag, pvalue_corrected, _, _ = multipletests(stat['permutation_p_value'], alpha=0.05, method='fdr_bh', is_sorted = False)
        stat['pvalue_correct'] = pvalue_corrected
        
    return stat


def tmt_normalization(df1: pd.DataFrame) -> pd.DataFrame:
    if df1.empty:
        logger.warning("Input DataFrame is empty. Skipping TMT normalization.")
        return df1
        
    sp_tmt = sum(df1[['TMT_group1', 'TMT_group2', 'mix_channels']].iloc[0].tolist(),[])
    columns_to_keep = [f'intensity_{x}' for x in sp_tmt]
    dop_columns=['Modification','modified_peptide_y','spectrum_x','batch','TMT_group1',
                 'TMT_group2', 'mix_channels','id_prot','position_in_protein']
    
    df=df1[columns_to_keep + dop_columns].copy()

    for ind, row in df.iterrows():
        mix = [f'intensity_{x}' for x in row['mix_channels']]
        mean_mix = row[mix].mean()
        
        if mean_mix == 0:
            continue
            
        for tmt in sp_tmt:
            df.loc[ind, f'intensity_{tmt}'] = df.loc[ind, f'intensity_{tmt}'] / mean_mix

    for batch in df['batch'].unique():
        df_batch_indices = df['batch'] == batch
        df_batch = df.loc[df_batch_indices].copy()
        
        for tmt in sp_tmt:
            median = df_batch[f'intensity_{tmt}'].median()
            if median == 0:
                continue
            df.loc[df_batch_indices, f'norm_intens_{tmt}'] = df.loc[df_batch_indices, f'intensity_{tmt}'] / median

    for tmt in sp_tmt:
        del df[f'intensity_{tmt}']

    return df


def sorting_psms(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int, list]:
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping PSM sorting.")
        return df, {}, 0, []
        
    df_copy = df[df['Modification'] != 'reference'].copy()
    
    if df_copy.empty:
        logger.warning("DataFrame contains only 'reference' modifications. Skipping PSM sorting.")
        return df, {}, 0, []
        
    cols = sum(df_copy[['TMT_group1', 'TMT_group2']].iloc[0].tolist(), [])
    stat = {f'intensity_{col}': 0 for col in cols}
    num = 0
    delete_indices = []

    for index, row in df_copy.iterrows():
        ad = [f'intensity_{x}' for x in row['TMT_group1']]
        control = [f'intensity_{x}' for x in row['TMT_group2']]
        mix = [f'intensity_{x}' for x in row['mix_channels']]
        
        mean_mix = row[mix].mean()
        median_all = np.median(row[[f'intensity_{x}' for x in cols]])
        
        if pd.isna(mean_mix) or mean_mix == 0:
            df_copy.loc[index, mix] = pd.NA
            continue

        if mean_mix < median_all * 0.5:
            df_copy.loc[index, mix] = pd.NA
            continue

        for group in [ad, control]:
            row_median = np.median(row[group])
            mask = (row[group] < row_median * 0.5) & (row[group] < mean_mix * 0.5)
            mask_nan = row[group].isna()

            if mask.any():
                affected = [col for col, val in zip(group, mask) if val]
                df_copy.loc[index, affected] = row_median
                for tag in affected:
                    stat[tag] += 1
                num += 1

            if mask_nan.any():
                if mask_nan.sum() <= len(mask_nan) / 2:
                    affected_nan = [col for col, val in zip(group, mask_nan) if val]
                    df_copy.loc[index, affected_nan] = row_median
                else:
                    delete_indices.append(index)

    df_copy.drop(index=delete_indices, inplace=True, errors='ignore')
    df_copy.reset_index(drop=True, inplace=True)
    if 'level_0' in df_copy.columns:
        df_copy.drop(columns=['level_0'], inplace=True)

    return df_copy, stat, num, delete_indices


def fasta_concat(df,fasta_file):
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping FASTA concat.")
        return df
        
    pr=[]
    se=[]
    df['id_prot']=df['protein'].apply(lambda x: str(x).split('|')[1])
    try:
        with fasta.read(fasta_file) as db:
            for descr, seq in db:
                pr.append(descr)
                se.append(seq)
    except FileNotFoundError:
        logger.error(f"FASTA file not found at {fasta_file}. Skipping FASTA concat.")
        return df

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
    
    df.drop(columns=['protein_x', 'protein_descr'], errors='ignore', inplace=True)
    return df