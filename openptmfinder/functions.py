import re
import json
import logging
import pandas as pd
import numpy as np
import os
import zipfile
import io
import gzip
import glob
import importlib
import contextlib
from pyteomics import pepxml, mzml, fasta
from statsmodels.stats.multitest import multipletests
from xml.etree import ElementTree as ET
from deeplc import DeepLC, FeatExtractor
from scipy import stats as scipy_stats
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit
from tqdm import tqdm
#from Bio import pairwise2
from Bio.Align import PairwiseAligner
import warnings
warnings.filterwarnings("ignore")


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

    try:
        for line in unimod.itertuples():
            a = line.massmod
            file = f'{a}.csv'
            for ind, mod in enumerate(line.unimod_name):
                if mod is None or pd.isna(mod):
                    continue
                name_modifications = mod if '"' not in mod else mod[:-1]
                if (name_modifications in name_of_modification) or (name_of_modification == ['all']):
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
                    pos = 0
                elif 'Cterm' in line.Modification:
                    pos = -1
                else:
                    find_pep=re.sub(r'\{[^}]*\}', '', line.modified_peptide)
                    pos = find_pep.find(line.select[0]) - 1
                    while pos>len(line.peptide):
                        pos= pos - (find_pep.find(']') -  find_pep.find('['))
                cataloque.loc[ind, 'position_mod'] = pos
                #cataloque.loc[ind, 'for_prediction'] = str(pos) + '|' + line.Modification.split('@')[0]
        #delite dublicates
        cataloque['mod_name'] = cataloque['Modification'].apply(lambda x: x.split('@')[0])
        cataloque['mod_name'] = cataloque['mod_name'].astype('category')
        cataloque = (cataloque.sort_values("localization score", ascending=False)
                       .drop_duplicates(subset=["spectrum", "peptide","file_mass",'mod_name'], keep="first")
                       .reset_index(drop=True)
                      )
        logger.info(f"Catalog with modifications created, total entries: {len(cataloque)}.")
        return cataloque, unimod_search

    except Exception as e:
        logger.error(f"Error creating modifications directory: {e}", exc_info=True)
        return None, None

    
def process_single_mzml(file_path, sub_df, output_temp_file):
    result_rows = []
    error_message = None
    try:
        with mzml.read(file_path, use_index=True) as file:
            spectrum_indices = (sub_df['index spectrum'] - 1).astype(int).tolist()
            for original_index, spectrum_index in zip(sub_df.index, spectrum_indices):
                try:
                    spectrum = file[spectrum_index]
                    result_rows.append({
                        'file_name': sub_df['file_name'].unique()[0],
                        'index spectrum': spectrum_index + 1,
                        'intensity': spectrum.get('intensity array'),
                        'm/z': spectrum.get('m/z array')
                    })
                except IndexError:
                    error_message = f"Spectrum index {spectrum_index} out of range in file {os.path.basename(file_path)}."
                    logger.error(error_message)
    except Exception as e:
        error_message = f"Error processing file {file_path}: {e}"
        logger.error(error_message)
        return None

    if result_rows:
        try:
            results_df = pd.DataFrame(result_rows)#.set_index('index')
            results_df.to_pickle(output_temp_file)
            return output_temp_file
        except Exception as e:
            logger.error(f"Error saving temporary file {output_temp_file}: {e}")
            return None
    return None


def intensity(link_mzml, cataloque, output_dir, n_processes=None):
    if cataloque.empty:
        logger.warning("Input DataFrame is empty. Skipping intensity calculation.")
        return cataloque
        
    unique_files = cataloque['file_name'].unique()
    
    # Создаем временную директорию для результатов
    temp_dir = os.path.join(output_dir, 'temp_intensity_results')
    os.makedirs(temp_dir, exist_ok=True)

    tasks = []
    for f in unique_files:
        mzml_path = os.path.join(link_mzml, f + '.mzML')
        if not os.path.isfile(mzml_path):
            logger.warning(f"File not found: {mzml_path}, skipping.")
            continue
        sub_df = cataloque[cataloque['file_name'] == f].copy()
        temp_file_path = os.path.join(temp_dir, f"{f}.pkl") # Имя временного файла
        tasks.append((mzml_path, sub_df, temp_file_path))

    results_files = []
    n_proc = n_processes or os.cpu_count()
    executor = ProcessPoolExecutor(max_workers=n_proc)
    try:
        futures = {executor.submit(process_single_mzml, path, df, temp_path): path for path, df, temp_path in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing mzML files"):
            path = futures[future]
            try:
                temp_file_path = future.result()
                if temp_file_path:
                    results_files.append(temp_file_path)
            except Exception as e:
                logger.error(f"An unexpected error occurred for file {path}: {e}")
    finally:
        # shutdown() гарантирует, что все worker-процессы будут завершены
        executor.shutdown(wait=True, cancel_futures=True)
    
    # Объединяем результаты из временных файлов
    if not results_files:
        logger.warning("No results were generated.")
        return pd.DataFrame()
        
    logger.info("Combining results from temporary files...")
    all_results_df = pd.DataFrame()
    for file in results_files:
        try:
            df_chunk = pd.read_pickle(file)
            all_results_df = pd.concat([all_results_df, df_chunk])
            os.remove(file) # Удаляем временный файл
        except Exception as e:
            logger.error(f"Error reading temporary file {file}: {e}")
            
    # Удаляем временную директорию, если она пуста
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)
        
    logger.info("Processing of all mzML files completed.")
    return all_results_df


def process_single_pepxml(file, modmass, spectra_map, peptide, mass_tolerance, fdr_threshold, sorting_pepxml,
                          min_hits_for_fdr_calc, default_hyperscore_threshold, default_expect_threshold):
    error_message = None
    try:
        ftf = pepxml.DataFrame(file)
        ftf['is_decoy'] = ftf['protein'].astype(str).str.contains('DECOY_', case=False, na=False)
        
        #updating cataloque dataframe (add protein name)
        if file in spectra_map.keys():
            catal_df=ftf[ftf['start_scan'].isin(list(spectra_map[file]))]
            
        #search different mass shifts
        df_mods_unique = ftf[ftf['peptide'].isin(list(peptide))]
        df_mods_unique_filter = pepxml.filter_df(df_mods_unique, fdr=fdr_threshold)
        
        mask = np.zeros(len(ftf), dtype=bool)
        for mod in modmass:
            tol = 0.05 if mod == 0 else mass_tolerance
            mask |= (np.abs(ftf['massdiff'] - mod) <= tol)
        filtered_ftf = ftf[mask]

        dfs = []
        for mod in modmass:
            mod_mask = (np.abs(filtered_ftf['massdiff'] - mod) <= mass_tolerance)
            current_mod_df = filtered_ftf[mod_mask].copy()

            if current_mod_df.empty:
                continue
            
            # Existing column 'is decoy'
            if 'is_decoy' not in current_mod_df.columns:
                logger.warning(f"File {os.path.basename(file)} for modification {mod}: 'is_decoy' column not found. Applying score-based filtering.")
                df1 = current_mod_df[(current_mod_df['hyperscore'] >= default_hyperscore_threshold) &
                                     (current_mod_df['expect'] <= default_expect_threshold)]
            else:
                num_targets = current_mod_df[current_mod_df['is_decoy'] == False].shape[0]
                num_decoys = current_mod_df[current_mod_df['is_decoy'] == True].shape[0]

                #if num_targets < min_hits_for_fdr_calc or num_decoys == 0:
                if num_decoys == 0:
                    logger.warning(f"File {os.path.basename(file)} for modification {mod}: Insufficient targets ({num_targets}) or decoys ({num_decoys}) for reliable FDR calculation. Applying score-based filtering.")
                    df1 = current_mod_df[(current_mod_df['hyperscore'] >= default_hyperscore_threshold) &
                                         (current_mod_df['expect'] <= default_expect_threshold)]
                else:
                    try:
                        df1 = pepxml.filter_df(current_mod_df, fdr=fdr_threshold)
                    except ZeroDivisionError:
                        error_message = f"ZeroDivisionError during FDR filtering in file: {os.path.basename(file)}, mod: {mod}. Applying score-based filtering."
                        logger.error(error_message)
                        df1 = current_mod_df[(current_mod_df['hyperscore'] >= default_hyperscore_threshold) &
                                             (current_mod_df['expect'] <= default_expect_threshold)]
                    except Exception as e:
                        error_message = f"Unexpected error during FDR filtering in file {os.path.basename(file)}: {e}. Applying score-based filtering."
                        logger.error(error_message)
                        df1 = current_mod_df[(current_mod_df['hyperscore'] >= default_hyperscore_threshold) &
                                             (current_mod_df['expect'] <= default_expect_threshold)]
            
            if df1.empty:
                continue
            
            if not df1.empty:
                df1['file_mass'] = mod
                dfs.append(df1)
        
        if dfs:
            return pd.concat(dfs), error_message, catal_df, df_mods_unique_filter
        else:
            return None, error_message, catal_df, df_mods_unique_filter

    except Exception as e:
        error_message = f"Error processing file {file}: {e}"
        return None, error_message, None, None

    
def process_pepxml_files(cataloque, pepxml_dir, mass_tolerance=0.012, fdr_threshold=0.05, sorting_pepxml=True, n_processes=1,
                         min_hits_for_fdr_calc=20, default_hyperscore_threshold=20.0, default_expect_threshold=0.05):
    if cataloque.empty:
        logger.warning("Input DataFrame is empty. Skipping pepXML processing.")
        return pd.DataFrame()
    
    spectra_map = {}
    modmass = cataloque['file_mass'].unique()
    peptide = cataloque['peptide'].unique()
    for spec in cataloque['spectrum'].unique():
        parts = spec.split(".")
        filename, scan = parts[0], parts[1]
        filepath = os.path.join(pepxml_dir[0],f"{filename}.pepXML")
        if not os.path.isfile(filepath):
            logger.warning(f"File {filepath} was not found.")
            continue
        spectra_map.setdefault(str(filepath), set()).add(int(scan))

    if len(pepxml_dir) == 1 and '.pepXML' not in pepxml_dir[0]:
        xml_files = glob.glob(os.path.join(pepxml_dir[0], '*.pepXML'))
    else:
        xml_files = pepxml_dir

    if not xml_files:
        logger.warning(f"No pepXML files found in directory: {pepxml_dir}")
        return pd.DataFrame()
    logger.info(f"Total {len(xml_files)} pepxml files found.")
    
    n_proc = n_processes
    results = []
    results_catal = pd.DataFrame()
    results_unique_mods = pd.DataFrame()
    executor = ProcessPoolExecutor(max_workers=n_proc)
    try:
        futures = {
            executor.submit(process_single_pepxml, file, modmass, spectra_map, peptide,
                            mass_tolerance, fdr_threshold, sorting_pepxml,
                            min_hits_for_fdr_calc, default_hyperscore_threshold, default_expect_threshold): file
            for file in xml_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pepXML files"):
            file = futures[future]
            try:
                df, error, catal_df, df_mods_unique = future.result()
                if error:
                    logger.error(f"Error from child process for file {file}: {error}")
                if df is not None and not df.empty:
                    results.append(df)
                if catal_df is not None and not catal_df.empty:
                    results_catal = pd.concat([catal_df,results_catal], ignore_index=True)
                if df_mods_unique is not None and not df_mods_unique.empty:
                    results_unique_mods = pd.concat([df_mods_unique,results_unique_mods], ignore_index=True)
            except Exception as e:
                logger.error(f"An unexpected error occurred for file {file}: {e}")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
                
    if not results:
        logger.warning("No matching records found after processing all files.")
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    df['index spectrum'] = df['spectrum'].str.split('.').str[1].astype(int)
    df['file_name'] = df['spectrum'].str.split('.').str[0]
    
    cataloque_pos = cataloque.merge(results_catal[['spectrum','retention_time_sec','massdiff','protein']], how='inner', on='spectrum')

    return df, cataloque_pos, results_unique_mods

def spectra_merge(cataloque, all_psms_df):
    
    psms_without_zero = all_psms_df[all_psms_df['file_mass'] != 0]
    psms_zero = all_psms_df[all_psms_df['file_mass'] == 0]
    del all_psms_df
    columns = ['Modification','id_prot','modified_peptide_x','position_in_protein',
               'peptide_x','peptide_y','spectrum_x','spectrum_y',
               'file_name','charge','sequence_y','index spectrum']
    
    merged = cataloque.merge(psms_without_zero, 
                            on=["id_prot","file_mass"], 
                            how="inner"
                           )
    mask = (merged["position_in_protein"] >= merged["peptide_start_y"]) & (merged["position_in_protein"] <= merged["peptide_end_y"])
    filtered = merged[mask]
    psms_zero['charge'] = psms_zero['spectrum'].str.split('.').str[3]
    filtered['charge'] = filtered['spectrum_y'].str.split('.').str[3]
    filtered['index spectrum'] = filtered['index spectrum'].astype('int32')
    psms_zero = psms_zero.drop_duplicates(subset='spectrum')
    
    psms_zero_filtered = psms_zero.merge(filtered[['Modification','id_prot','peptide_y','position_in_protein',
                                                   'spectrum_x','peptide_x','modified_peptide_x','charge']], 
                                         left_on = ['peptide','id_prot','charge'], 
                                         right_on = ['peptide_y','id_prot','charge'],
                                         how = 'inner'
                                        )
    del psms_zero_filtered['peptide']
    psms_zero_filtered.rename(columns = {'spectrum':'spectrum_y','modified_peptide':'modified_peptide_y',
                                        'sequence':'sequence_y'}, inplace = True)
    psms_zero_filtered['Modification'] = 'reference'
    psms_zero_filtered['charge'] = psms_zero_filtered['charge'].astype('int8')
    psms_zero_filtered['index spectrum'] = psms_zero_filtered['index spectrum'].astype('int32')
    psms_zero_filtered['position_in_protein'] = psms_zero_filtered['position_in_protein'].astype('int32')
    psms_zero_filtered['id_prot'] = psms_zero_filtered['id_prot'].astype('category')
    psms_zero_filtered['Modification'] = psms_zero_filtered['Modification'].astype('category')
    
    #psms_zero_filtered = psms_zero_filtered.drop_duplicates(subset=['Modification','id_prot','position_in_protein',
                                                                    #'peptide_y','spectrum_y','peptide_x','spectrum_x'])
    return filtered, psms_zero_filtered[columns].drop_duplicates()


def map_mod_position(peptide1: str, mod_position1: int, peptide2: str) -> int:
    
    mod_position2 = 0
    if mod_position1 == -1:
        mod_position2 == -1
    elif mod_position1 == 0:
        mod_position2 == 0
    else:
        aln1, aln2, *_ = pairwise2.align.globalxx(peptide1, peptide2)[0]

        pos_in_aln1 = 0
        aa_counter = 0
        for i, aa in enumerate(aln1):
            if aa != "-":
                aa_counter += 1
            if aa_counter == mod_position1:
                pos_in_aln1 = i
                break

        aa_counter2 = 0
        for i, aa in enumerate(aln2):
            if aa != "-":
                aa_counter2 += 1
            if i == pos_in_aln1:
                mod_position2 = aa_counter2
                break
    return mod_position2


def prediction_rt(pepxml_psms: pd.DataFrame) -> pd.DataFrame:
    if pepxml_psms.empty:
        logger.warning("Input DataFrame is empty. Skipping RT prediction.")
        return None

    # Фильтруем данные для калибровки и предсказания
    calibration_set = pepxml_psms[
        (pepxml_psms['file_mass'] != 0) & 
        (pepxml_psms['spectrum_x'] == pepxml_psms['spectrum_y'])
    ][['peptide_x', 'for_prediction', 'retention_time_sec_x']]

    if len(calibration_set) < 50:
        logger.warning("Not enough PSMs for calibration. Skipping RT prediction.")
        return None

    df_for_calib = pd.DataFrame({
        'seq': calibration_set['peptide_x'],
        'modifications': calibration_set['for_prediction'],
        'tr': calibration_set['retention_time_sec_x']
    }).drop_duplicates()

    logger.info(f'Create a dataframe for calibration {len(df_for_calib)}')

    dlc = DeepLC(verbose=False, pygam_calibration=False)
    dlc.calibrate_preds(seq_df=df_for_calib)
    logger.info('The model is calibrated.')
    
    predict_set = pepxml_psms[(pepxml_psms['file_mass'] != 0) & (pepxml_psms['spectrum_x']!=pepxml_psms['spectrum_y'])]
    
    predict_set['predicted_RT'] = dlc.make_preds(seq_df=pd.DataFrame({
        'seq': predict_set['peptide_y'],
        'modifications': predict_set['for_prediction']
    }))
    
    rt_diff_df = predict_set[['for_prediction', 'retention_time_sec_y', 'predicted_RT','file_mass']].copy()
    rt_diff_df['rt_diff'] = rt_diff_df['predicted_RT'] - rt_diff_df['retention_time_sec_y']
    
    calibration_params = {}
    
    # Калибровка и фильтрация для каждой модификации отдельно
    mod_types = rt_diff_df['file_mass'].unique()
    for mod in mod_types:
        mod_df = rt_diff_df[rt_diff_df['file_mass'] == mod]
        rt_diff_tmp = mod_df['rt_diff'].dropna().values
        try:
            XRT_shift, XRT_sigma, _ = _calibrate_single_mod_rt_gaus(rt_diff_tmp)
            calibration_params[mod] = {'shift': XRT_shift, 'sigma': XRT_sigma}
        except Exception as e:
            logger.error(f"Error during RT calibration for modification '{mod}': {e}. Skipping specific calibration.")
            continue
    
    outlier_indices = []
    
    for mod, params in calibration_params.items():
        mod_df = rt_diff_df[rt_diff_df['file_mass'] == mod].copy()
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


def fast_name_tmt(mz: float, type_tmt: str) -> str:
    if 126.125226 <= mz <= 126.130226:
        return '126'
    elif 127.122261 <= mz <= 127.127261:
        return '127N'
    elif 127.128581 <= mz <= 127.133581:
        return '127C'
    elif 128.125616 <= mz <= 128.130616:
        return '128N'
    elif 128.131936 <= mz <= 128.136936:
        return '128C'
    elif 129.128971 <= mz <= 129.133971:
        return '129N'
    elif 129.13529 <= mz <= 129.14029:
        return '129C'
    elif 130.132325 <= mz <= 130.137325:
        return '130N'
    elif 130.138645 <= mz <= 130.143645:
        return '130C'
    elif 131 <= mz <= 132:
        if type_tmt == 'TMT10plex' and 131.13568 <= mz <= 131.14068:
            return '131'
        elif type_tmt == 'TMT11plex':
            if 131.13568 <= mz <= 131.14068:
                return '131N'
            elif 131.141999 <= mz <= 131.146999:
                return '131C'
    return None

def annotate_tmt_chunk(chunk: pd.DataFrame, type_tmt: str, output_temp_file: str, r: int = 4) -> str:
    tmt_keys = ['126', '127N', '127C', '128N', '128C',
                '129N', '129C', '130N', '130C', '131', '131N', '131C']
    
    for key in tmt_keys:
        chunk[f'intensity_{key}'] = np.nan
        
    for i, row in tqdm(chunk.iterrows(), total=len(chunk),desc="Processing tag annotation:"):
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
                        chunk.loc[i, f'intensity_{tag}'] = intensity
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            continue

    # Сохраняем результат в файл и возвращаем путь
    chunk.to_pickle(output_temp_file)
    return output_temp_file


def tags_annotation(cataloque: pd.DataFrame, type_tmt: str, output, n_proc: int = None) -> pd.DataFrame:
    if cataloque.empty:
        logger.warning("Input DataFrame is empty. Skipping TMT annotation.")
        return cataloque
        
    temp_dir=os.path.join(output, 'temp_tmt_results')
    os.makedirs(temp_dir, exist_ok=True)
    
    chunk_size = max(1, len(cataloque) // (n_proc or cpu_count()))
    chunks = [cataloque.iloc[i:i + chunk_size].copy() for i in range(0, len(cataloque), chunk_size)]
    
    executor = ProcessPoolExecutor(max_workers=n_proc)
    results_files = []
    
    try:
        futures = []
        for i, chunk in enumerate(chunks):
            temp_file_path = os.path.join(temp_dir, f"chunk_{i}.pkl")
            futures.append(executor.submit(annotate_tmt_chunk, chunk, type_tmt, temp_file_path))

        for future in as_completed(futures):
            try:
                temp_file_path = future.result()
                if temp_file_path:
                    results_files.append(temp_file_path)
            except Exception as e:
                logger.error(f"An error occurred in a worker process: {e}")
                
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    # Объединяем результаты из временных файлов
    if not results_files:
        logger.warning("No results were generated.")
        return pd.DataFrame()
        
    logger.info("Combining results from temporary files...")
    all_results_df = pd.DataFrame()
    for file in results_files:
        try:
            df_chunk = pd.read_pickle(file)
            all_results_df = pd.concat([all_results_df, df_chunk])
            os.remove(file) # Удаляем временный файл
        except Exception as e:
            logger.error(f"Error reading temporary file {file}: {e}")

    # Удаляем временную директорию, если она пуста
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
        
    logger.info('TMT annotation completed.')
    return all_results_df


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




def calc_weight_per_group(row, group_channels):
    intensities_in_group = [row[f'intensity_{ch}_norm'] for ch in group_channels if f'intensity_{ch}_norm' in row]
    if len(intensities_in_group) == 0:
        return 0
    return sum(intensities_in_group)


def weighted_mean(group):
    group1_channels = list(set(i for sub in group['TMT_group1'] for i in (sub if isinstance(sub, list) else [sub])))
    group2_channels = list(set(i for sub in group['TMT_group2'] for i in (sub if isinstance(sub, list) else [sub])))

    w_group1 = group.apply(lambda row: calc_weight_per_group(row, row['TMT_group1']), axis=1)
    w_group2 = group.apply(lambda row: calc_weight_per_group(row, row['TMT_group2']), axis=1)

    intensity_TMT_group1 = [(group[f'intensity_{ch}_norm'] * w_group1).sum() / w_group1.sum() if w_group1.sum()>0 and f'intensity_{ch}_norm' in group else 0 for ch in group1_channels]
    intensity_TMT_group2 = [(group[f'intensity_{ch}_norm'] * w_group2).sum() / w_group2.sum() if w_group2.sum()>0 and f'intensity_{ch}_norm' in group else 0 for ch in group2_channels]
    
    coef1 = [(group[f'intensity_{ch}_norm'].sum()) if f'intensity_{ch}_norm' in group else 0 for ch in group1_channels]
    coef2 = [(group[f'intensity_{ch}_norm'].sum()) if f'intensity_{ch}_norm' in group else 0 for ch in group2_channels]

    data = {
        'peptide_y': [list(group['peptide_y'])],
        'charge_y': [list(group['charge'])],
        'peptide_x': [[item for sublist in group['peptide_x'] for item in sublist]],
        'modified_peptide_x': [[item for sublist in group['modified_peptide_x'] for item in sublist]],
        'TMT_group1': [group1_channels],
        'TMT_group2': [group2_channels],
        'spectrum_y': [[item for sublist in group['spectrum_y'] for item in sublist]],
        'spectrum_x': [[item for sublist in group['spectrum_x'] for item in sublist]],
        'intensity_TMT_group1': [intensity_TMT_group1],
        'intensity_TMT_group2': [intensity_TMT_group2],
        'coef1' : [coef1],
        'coef2' : [coef2]
    }
    return pd.DataFrame(data)


def explode_group(df, group_col, intensity_col):
    rows = []
    for _, row in df.iterrows():
        channels = row[group_col]
        intensities = row[intensity_col]
        for ch, inten in zip(channels, intensities):
            new_row = row.copy()
            new_row['channel'] = ch
            new_row['intensity'] = inten
            rows.append(new_row)
    return pd.DataFrame(rows)


def chemo_coef(df_site):
    df_zero = df_site[df_site['Modification']=='reference']
    df_mod = df_site[df_site['Modification']!='reference']

    df_exp1 = explode_group(df_mod, 'TMT_group1', 'coef1')
    df_exp2 = explode_group(df_mod, 'TMT_group2', 'coef2')
    df_exp_mod = pd.concat([df_exp1, df_exp2], ignore_index=True)
    df_exp3 = explode_group(df_zero, 'TMT_group1', 'coef1')
    df_exp4 = explode_group(df_zero, 'TMT_group2', 'coef2')
    df_exp_zero = pd.concat([df_exp3, df_exp4], ignore_index=True)

    df_exp_zero['site'] = df_exp_zero['id_prot'].astype(str) + "_" + df_exp_zero['position_in_protein'].astype(str)
    df_exp_zero['sample'] = df_exp_zero['batch'].astype(str) + "_" + df_exp_zero['channel'].astype(str)
    site_matrix_zero = df_exp_zero.pivot_table(index='site', columns='sample', values='intensity', aggfunc='sum')

    df_exp_mod['site_base'] = df_exp_mod['id_prot'].astype(str) + "_" + df_exp_mod['position_in_protein'].astype(str)
    df_exp_mod['site_mod'] = df_exp_mod['Modification'].astype(str) + "_" + df_exp_mod['site_base']
    df_exp_mod['sample'] = df_exp_mod['batch'].astype(str) + "_" + df_exp_mod['channel'].astype(str)

    site_matrix_mod = df_exp_mod.pivot_table(index=['site_base','Modification'], 
                                             columns='sample', values='intensity', aggfunc='sum')

    stoich = site_matrix_mod.copy()
    for site_base in stoich.index.get_level_values('site_base').unique():
        if site_base not in site_matrix_zero.index:
            denom = 0
        else:
            denom = site_matrix_zero.loc[site_base]
        stoich.loc[(site_base, slice(None)), :] = (
            stoich.loc[(site_base, slice(None)), :] /
            (stoich.loc[(site_base, slice(None)), :] + denom)
        )

            
    valid_cols1 = stoich.columns.intersection(list(set(df_exp1['batch'].astype(str) + "_" + df_exp1['channel'].astype(str))))
    valid_cols2 = stoich.columns.intersection(list(set(df_exp2['batch'].astype(str) + "_" + df_exp2['channel'].astype(str))))

    stoich['stoichiometry_TMT_group1'] = stoich[valid_cols1].apply(
        lambda row: [x for x in row if pd.notna(x)] if any(pd.notna(row)) else pd.NA,
        axis=1
    )
    stoich['stoichiometry_TMT_group2'] = stoich[valid_cols2].apply(
        lambda row: [x for x in row if pd.notna(x)] if any(pd.notna(row)) else pd.NA,
        axis=1
    )         
        
    stoich['stoich1_median'] = stoich['stoichiometry_TMT_group1'].apply(
        lambda x: np.median([v for v in x if pd.notna(v)]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else pd.NA
    )
    stoich['stoich2_median'] = stoich['stoichiometry_TMT_group2'].apply(
        lambda x: np.median([v for v in x if pd.notna(v)]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else pd.NA
    )
    stoich['stoich1_median'] = pd.to_numeric(stoich['stoich1_median'], errors='coerce')
    stoich['stoich2_median'] = pd.to_numeric(stoich['stoich2_median'], errors='coerce')
    stoich['FC_coef'] = np.log2(stoich['stoich1_median'] / stoich['stoich2_median'])
    
    return stoich[['stoichiometry_TMT_group1','stoichiometry_TMT_group2','stoich1_median','stoich2_median','FC_coef']].reset_index()

def compute_stats(row,  col, min_group = 5):
    if col=='stoichiometry_TMT_group':
        suffix = 'coef'
    else:
        suffix = 'abs'

    result = {
        f'T_test_p_value_{suffix}': np.nan,
        f'Mann_p_value_{suffix}': np.nan,
        f'FC_median_{suffix}': np.nan,
        f'median_group1_{suffix}': np.nan,
        f'median_group2_{suffix}': np.nan
    }

    group1 = np.array([x for x in (row[col+'1'] if isinstance(row[col+'1'], list) else []) if pd.notna(x)])
    group2 = np.array([x for x in (row[col+'2'] if isinstance(row[col+'1'], list) else []) if pd.notna(x)])

    if len(group1) >= min_group and len(group2) >= min_group:
        try:
            _, result[f'Mann_p_value_{suffix}'] = scipy_stats.mannwhitneyu(
                group1, group2, nan_policy='omit', method ='auto'
            )

            # t-test
            _, result[f'T_test_p_value_{suffix}'] = scipy_stats.ttest_ind(
                group1, group2, nan_policy='omit', equal_var=False
            )
            if col!='stoichiometry_TMT_group':
                med1 = np.median(group1)
                med2 = np.median(group2)

                result[f'median_group1_{suffix}'] = med1
                result[f'median_group2_{suffix}'] = med2
                result[f'FC_median_{suffix}'] = np.log2(med1/med2) if med2 != 0 else np.nan

        except Exception as e:
            logger.error(f"Statistics error for index {row.name}: {e}")

    return pd.Series(result)


def statistics(df: pd.DataFrame, calc_pval: bool = True, min_group_for_stats: int = 5) -> pd.DataFrame:
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping statistics.")
        return stat
        
    columns = ['TMT_group1', 'TMT_group2']
    stat = df.copy()
    
    intensity_cols = [c for c in stat.columns if c.endswith("_norm")]
    stat = stat.drop_duplicates(subset=['Modification','id_prot','position_in_protein','batch','peptide_y','charge','spectrum_y'])
    
    agg_df = (
        stat
        .groupby(['Modification','id_prot','position_in_protein','batch','peptide_y','charge'], as_index=False)
        .agg(
            {**{col: 'sum' for col in intensity_cols},
             **{'modified_peptide_x': lambda x: list(x)},
             **{'peptide_x': lambda x: list(x)},
             **{'spectrum_x': lambda x: list(x)},
             **{'spectrum_y': lambda x: list(x)},
             **{'TMT_group1': lambda x: list(set([i for sub in x for i in sub]))},
             **{'TMT_group2': lambda x: list(set([i for sub in x for i in sub]))},
             **{'mix_channels': lambda x: list(set([i for sub in x for i in sub]))}}
        )
    )
    
    df_site = (
        agg_df
        .groupby(['Modification','id_prot','position_in_protein','batch'])
        .apply(weighted_mean).reset_index()
    )
    logger.info('Start calculate chemocoeficient')
    stoich = chemo_coef(df_site)
    df_site = df_site[df_site['Modification']!='reference']

    stat = (
        df_site
        .groupby(['Modification','id_prot','position_in_protein'], as_index=False)
        .agg({
            'intensity_TMT_group1': lambda x: [i for sub in x for i in sub],
            'intensity_TMT_group2': lambda x: [i for sub in x for i in sub],
            'spectrum_y': lambda x: [i for sub in x for i in sub],
            'peptide_y': list,
            'charge_y':list,
            'spectrum_x': list,
            'peptide_x': list,
            'modified_peptide_x':list,
            'TMT_group1': list,
            'TMT_group2': list
        })
    )

    stat['modified_peptide_x'] = stat['modified_peptide_x'].apply(lambda x: x[0][0])
    stat['spectrum_x'] = stat['spectrum_x'].apply(lambda x: x[0][0])
    stat['peptide_x'] = stat['peptide_x'].apply(lambda x: x[0][0])
    stat['site_base'] = stat['id_prot'].astype(str) + "_" + stat['position_in_protein'].astype(str)
    stat = stat.merge(stoich, on=['site_base','Modification'], how = 'left')
    del stat['site_base']
    
    logger.info('Start calculate chemocoeficient')
    
    stat_stats = stat.apply(compute_stats, axis=1, col ='stoichiometry_TMT_group', min_group = min_group_for_stats)
    res_coef = pd.concat([stat.reset_index(drop=True), stat_stats], axis=1)
    stat_stats = res_coef.apply(compute_stats, axis=1, col ='intensity_TMT_group', min_group = min_group_for_stats)
    stat_final = pd.concat([res_coef.reset_index(drop=True), stat_stats], axis=1)

    # Объединяем с исходным датафреймом
    stat_final = stat_final[stat_final['T_test_p_value_coef'].notna()].sort_values('T_test_p_value_coef').reset_index(drop=True)
    del stat_final['FC_median_coef']
    del stat_final['median_group1_coef']
    del stat_final['median_group2_coef']
    
    if not stat_final.empty:
        flag, pvalue_corrected, _, _ = multipletests(stat_final['T_test_p_value_coef'], alpha=0.05, method='fdr_bh', is_sorted = False)
        stat_final['pvalue_Ttest_correct'] = pvalue_corrected
        
    return stat_final, stoich


def tmt_normalization(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping TMT normalization.")
        return df
    intensity_cols = [c for c in df.columns if c.startswith("intensity_")]
    calc_norm = df.drop_duplicates(subset='spectrum_y')

    # Медианное центрирование внутри batch
    def median_centering(group):
        medians = group[intensity_cols].median(axis=0)
        group[intensity_cols] = group[intensity_cols] / medians
        
        batch_median = group[intensity_cols].stack().median()
        group[intensity_cols] = group[intensity_cols] / batch_median
        return group

    df_norm = calc_norm.groupby("batch", group_keys=False).apply(median_centering)
    final_df = df.merge(df_norm[intensity_cols + ['spectrum_y']], how = 'left', on = 'spectrum_y', suffixes=('', '_norm'))
    return final_df
"""
#batch normalisation
    batch_factors = {}
    for batch, group in calc_norm.groupby('batch'):
        reference_channels = group.iloc[0]['mix_channels']  
        ref_cols = [f'intensity_{r}' for r in reference_channels]
        batch_ref_median = group[ref_cols].median().median()
        batch_factors[batch] = batch_ref_median

    global_ref = np.median(list(batch_factors.values()))
    for batch in batch_factors:
        batch_factors[batch] = global_ref / batch_factors[batch]

    for tmt in sp_tmt:
        col = f'intensity_{tmt}'
        norm_col = f'norm_intens_{tmt}'
        calc_norm[norm_col] = calc_norm[col] * calc_norm['batch'].map(batch_factors)
    norm_intensity_cols = [c for c in calc_norm.columns if c.startswith("norm_intens_")] + ['spectrum_y']
    final_df = df.merge(calc_norm[norm_intensity_cols], how = 'left', on = 'spectrum_y') 
    return final_df
 """ 

def sorting_psms(df_copy: pd.DataFrame) -> tuple[pd.DataFrame, dict, int, list]:
    if df_copy.empty:
        logger.warning("Input DataFrame is empty. Skipping PSM sorting.")
        return df, {}, 0, []
    
    cols = sum(df_copy[['TMT_group1', 'TMT_group2']].iloc[0].tolist(), [])
    stat = {f'intensity_{col}': 0 for col in cols}
    num = 0
    delete_indices = []
    for index, row in tqdm(df_copy.iterrows(), total=len(df_copy),desc="Processing sorting intensity:"):
        try:
            ad = [f'intensity_{x}' for x in row['TMT_group1']]
            control = [f'intensity_{x}' for x in row['TMT_group2']]
            mix = [f'intensity_{x}' for x in row['mix_channels']]
            
            mean_mix = row[mix].median()
            median_all = np.median(row[[f'intensity_{x}' for x in cols]])
        except KeyError as e:
            logger.error(f"KeyError in row with index {index}: {e}. Skipping row.")
            delete_indices.append(index)
            continue
            
        #if pd.isna(mean_mix) or mean_mix == 0:
        #   df_copy.loc[index, mix] = pd.NA
        #   continue
        #if mean_mix < median_all * 0.5:
        #    df_copy.loc[index, mix] = pd.NA
        #    continue
        for group in [ad, control]:
            row_median = np.median(row[group])
            mask = (row[group] < row_median * 0.5)# & (row[group] < mean_mix * 0.5)
            mask_nan = row[group].isna()
            if mask.any():
                affected = [col for col, val in zip(group, mask) if val]
                df_copy.loc[index, affected] = row_median
                for tag in affected:
                    stat[tag] += 1
            if mask_nan.any():
                if mask_nan.sum() <= len(mask_nan) / 2:
                    affected_nan = [col for col, val in zip(group, mask_nan) if val]
                    df_copy.loc[index, affected_nan] = row_median
                else:
                    delete_indices.append(index)
                    num += 1

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
                if ('DECOY_' not in descr) and (('rev_' not in descr)):
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
        df['peptide_start'] = df.apply(
            lambda row: (row['sequence'].find(row['peptide']) + 1) if pd.notna(row['sequence']) and pd.notna(row['peptide']) else pd.NA,
            axis=1
        )
        df['peptide_end'] = df['peptide_start'] + df['peptide'].apply(lambda x: len(x)) - 1
        try:
            df['position_in_protein'] = df.apply(
                lambda row: (
                    row['peptide_start'] if pd.notna(row['peptide_start']) and row['position_mod'] == 0
                    else row['peptide_end'] if pd.notna(row['peptide_end']) and row['position_mod'] == -1
                    else (row['position_mod'] + row['peptide_start'] - 1) if pd.notna(row['position_mod']) and pd.notna(row['peptide_start'])
                    else pd.NA
                ),
                axis=1
            )

        except Exception as e:
            logger.warning(f"Positions weren't calculated: {e}")
    try:
        df.drop(columns=['protein_x', 'protein_descr','Unnamed: 0'], errors='ignore', inplace=True)
        if 'protein_y' in df.columns:
            df.rename(columns={'protein_y':'protein'}, inplace=True)
    except Exception as e:
        logger.warning(f"Columns weren't deleted or renamed: {e}")
    return df

