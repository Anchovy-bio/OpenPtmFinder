import ast
import re
import json
import logging
import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import cpu_count
from pyteomics import pepxml, mzml, fasta
from xml.etree import ElementTree as ET
from deeplc import DeepLC
from scipy import stats as scipy_stats
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit
from tqdm import tqdm
from Bio import pairwise2
from collections import defaultdict
from typing import List, Literal, Optional, Sequence


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
        # raised when the regular expression finds no match
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
    
    # Drop rows with an empty unimod_name
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
        if  localization_score_threshold != 0:
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
    """Build the catalogue of modification candidates and amino-acid patterns."""
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

        unimod_search = pd.DataFrame(unimod_search_list)

        # Process the reference (zero mass shift) file
        ref_file = os.path.join(link_data, '+0.0000.csv')
        file_reader = pd.read_csv(ref_file, sep="\t")
        file_reader['peptide'] = file_reader['peptide'].apply(
            lambda x: re.sub(r'[^A-Z]', '', x[1:-1]) if (x[-3] == 'R' or x[-3] == 'K' or x[-3] == ']') else re.sub(r'[^A-Z]', '', x[1:])
        )
        cataloque['peptide'] = cataloque['top isoform'].apply(
            lambda x: re.sub(r'[^A-Z]', '', x[1:-1]) if (x[-3] == 'R' or x[-3] == 'K' or x[-3] == ']') else re.sub(r'[^A-Z]', '', x[1:])
        )
        file_reader = file_reader[file_reader['peptide'].isin(cataloque['peptide'].unique())]
        file_reader['Modification'] = 'reference'
        file_reader['file_mass'] = 0

        cataloque.rename(columns={'top isoform': 'modified_peptide'}, inplace=True)
        cataloque = pd.concat([file_reader, cataloque], ignore_index=True)

        # Compute modification positions within the peptide
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
        # drop duplicates
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

    # Temporary directory for per-file results
    temp_dir = os.path.join(output_dir, 'temp_intensity_results')
    os.makedirs(temp_dir, exist_ok=True)

    tasks = []
    for f in unique_files:
        mzml_path = os.path.join(link_mzml, f + '.mzML')
        if not os.path.isfile(mzml_path):
            logger.warning(f"File not found: {mzml_path}, skipping.")
            continue
        sub_df = cataloque[cataloque['file_name'] == f].copy()
        temp_file_path = os.path.join(temp_dir, f"{f}.pkl")  # temporary file name
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
        # shutdown() guarantees that all worker processes are terminated
        executor.shutdown(wait=True, cancel_futures=True)

    # Combine results from the temporary files
    if not results_files:
        logger.warning("No results were generated.")
        return pd.DataFrame()

    logger.info("Combining results from temporary files...")
    all_results_df = pd.DataFrame()
    for file in results_files:
        try:
            df_chunk = pd.read_pickle(file)
            all_results_df = pd.concat([all_results_df, df_chunk])
            os.remove(file)  # remove the temporary file
        except Exception as e:
            logger.error(f"Error reading temporary file {file}: {e}")

    # Remove the temporary directory if it is empty
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    logger.info("Processing of all mzML files completed.")
    return all_results_df


def process_single_pepxml(file, modmass, spectra_map, peptide, mass_tolerance, fdr_threshold, sorting_pepxml,
                          min_hits_for_fdr_calc, default_hyperscore_threshold, default_expect_threshold):
    error_message = None
    catal_df = pd.DataFrame()
    df_mods_unique_filter = pd.DataFrame()
    try:
        ftf = pepxml.DataFrame(file)
        ftf['is_decoy'] = ftf['protein'].astype(str).str.contains('DECOY_', case=False, na=False)

        # updating catalogue dataframe (add protein name);
        # files without catalogue spectra (or outside spectra_map) keep an
        # empty catal_df instead of failing with an unbound variable
        if file in spectra_map.keys():
            catal_df = ftf[ftf['start_scan'].isin(list(spectra_map[file]))]

        # search different mass shifts
        df_mods_unique = ftf[ftf['peptide'].isin(list(peptide))]
        if not df_mods_unique.empty:
            try:
                df_mods_unique_filter = pepxml.filter_df(df_mods_unique, fdr=fdr_threshold)
            except Exception as e:
                logger.warning(f"FDR filtering of unique-modification PSMs failed "
                               f"in {os.path.basename(file)}: {e}")

        def _tol(mod):
            # the unmodified (zero) shift is allowed a wider window because
            # systematic calibration offsets affect all PSMs alike
            return 0.05 if mod == 0 else mass_tolerance

        mask = np.zeros(len(ftf), dtype=bool)
        for mod in modmass:
            mask |= (np.abs(ftf['massdiff'] - mod) <= _tol(mod))
        filtered_ftf = ftf[mask]

        dfs = []
        for mod in modmass:
            mod_mask = (np.abs(filtered_ftf['massdiff'] - mod) <= _tol(mod))
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

                # With no decoys the per-mod FDR is undefined; fall back to
                # the default score thresholds (min_hits_for_fdr_calc is kept
                # for API compatibility with the catalogue-based caller).
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

def spectra_merge(cataloque, all_psms_df, unimod):
    
    psms_without_zero = all_psms_df[all_psms_df['file_mass'] != 0]
    psms_zero = all_psms_df[all_psms_df['file_mass'] == 0]
    del all_psms_df
    columns = ['Modification','id_prot','modified_peptide_x','position_in_protein',
               'peptide_x','peptide_y','spectrum_x','spectrum_y',
               'file_name','charge','sequence_y','index spectrum']
    unimod.rename(columns={
        'modifiction': 'mod_name',
        'mass_shift': 'file_mass'
    }, inplace=True)
    
    unimod['file_mass'] = unimod['file_mass'].astype('float32')
    psms_without_zero['file_mass'] = psms_without_zero['file_mass'].astype('float32')
    mod_map = dict(zip(unimod['file_mass'].values,
                       unimod['mod_name'].values))
    psms_without_zero['mod_name'] = psms_without_zero['file_mass'].map(mod_map)
    
    merged = cataloque.merge(psms_without_zero, 
                            on=["id_prot","mod_name"], 
                            how="inner"
                           )
    logger.debug("spectra_merge: catalogue merged with modified PSMs")
    mask = (merged["position_in_protein"] >= merged["peptide_start_y"]) & (merged["position_in_protein"] <= merged["peptide_end_y"])
    filtered = merged[mask]
    del merged
    del psms_without_zero
    psms_zero['charge'] = psms_zero['spectrum'].str.split('.').str[3].astype('int8')
    filtered['charge'] = filtered['spectrum_y'].str.split('.').str[3].astype('int8')
    filtered['index spectrum'] = filtered['index spectrum'].astype('int32')
    psms_zero = psms_zero.drop_duplicates(subset='spectrum')
    logger.debug("spectra_merge: position filtering done")
    psms_zero_filtered = psms_zero.merge(filtered[['Modification','id_prot','peptide_y','position_in_protein',
                                                   'spectrum_x','peptide_x','modified_peptide_x','charge']].drop_duplicates(), 
                                         left_on = ['peptide','id_prot','charge'], 
                                         right_on = ['peptide_y','id_prot','charge'],
                                         how = 'inner'
                                        )
    del psms_zero_filtered['peptide']
    del psms_zero
    logger.debug("spectra_merge: reference (zero-shift) PSMs matched")
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
        mod_position2 = -1
    elif mod_position1 == 0:
        mod_position2 = 0
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

    # Select the data used for calibration and prediction
    pepxml_psms.rename(columns={'file_mass_y':'file_mass'}, inplace=True)
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
    
    # Calibrate and filter each modification separately
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

    # Save the result to a file and return its path
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

    # Combine results from the temporary files
    if not results_files:
        logger.warning("No results were generated.")
        return pd.DataFrame()

    logger.info("Combining results from temporary files...")
    all_results_df = pd.DataFrame()
    for file in results_files:
        try:
            df_chunk = pd.read_pickle(file)
            all_results_df = pd.concat([all_results_df, df_chunk])
            os.remove(file)  # remove the temporary file
        except Exception as e:
            logger.error(f"Error reading temporary file {file}: {e}")

    # Remove the temporary directory if it is empty
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
        
    logger.info('TMT annotation completed.')
    return all_results_df


def samples_annotation(full_df: pd.DataFrame, group_df_link: str) -> pd.DataFrame:
    if full_df.empty:
        logger.warning("Input DataFrame is empty. Skipping samples annotation.")
        return full_df
        
    try:
        # Try the default separator (',') first
        group_df = pd.read_csv(group_df_link)
    except Exception:
        # Fall back to the ';' separator
        try:
            group_df = pd.read_csv(group_df_link, sep=';')
        except Exception as e:
            logger.error(f"Error reading grouping file {group_df_link}: {e}")
            return full_df.copy()
        
    group_cols = sorted(
        [c for c in group_df.columns if re.fullmatch(r"TMT_group\d+", str(c))],
        key=lambda c: int(re.search(r"\d+", str(c)).group())
    )

    if not group_cols:
        logger.error(f"Grouping file {group_df_link} has no TMT_groupN columns.")
        return full_df.copy()
    parse_cols = group_cols + (["mix_channels"] if "mix_channels" in group_df.columns else [])

    full_df = full_df.drop_duplicates(subset=['id_prot','position_in_protein','modified_peptide_x','Modification','spectrum_y'])
    full_df['batch'] = full_df['file_name'].str.split('_').str[1].str[1:]
    full_df['batch'] = full_df['batch'].astype('int')
    group_df['batch'] = group_df['batch'].astype('int')
    
    if 'TMT_group1'in full_df.columns.tolist():
        del full_df['TMT_group1']
        
    if 'TMT_group2'in full_df.columns.tolist():
        del full_df['TMT_group2']
        
    if 'TMT_group3'in full_df.columns.tolist():
        del full_df['TMT_group3']
        
    if 'mix_channels'in full_df.columns.tolist():
        del full_df['mix_channels']
        
    full_df_group = full_df.merge(group_df, how='left', on=['file_name','batch'])
    missing_annotations_count = len(set(full_df_group['file_name'][full_df_group[group_cols[0]].isna()]))
    if missing_annotations_count > 0:
        logger.warning(f"There are no annotations for {missing_annotations_count} files.")

    full_df_group = full_df_group[full_df_group[group_cols[0]].notna()].copy()

    for c in parse_cols:
        full_df_group[c] = full_df_group[c].apply(
            lambda x: re.split(r'\s*,\s*', re.sub(r"[\'\[\]]", "", str(x))))

    logger.info(f"Samples annotation completed ({len(group_cols)} experimental groups).")

    return full_df_group


def _unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_channel_cell(x) -> List[str]:
    """Parse list-like annotation cells ("['126','127']", "126,127", [126, 127])."""
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(z).strip() for z in v if str(z).strip()]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return [str(v).strip()]
        except Exception:
            pass
        s = s.strip("[](){}")
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]
    return [str(x).strip()]


def _raw_intensity_columns(df: pd.DataFrame, intensity_prefix: str, norm_suffix: str) -> List[str]:
    cols = []
    for c in df.columns:
        c = str(c)
        if not c.startswith(intensity_prefix):
            continue
        if c.endswith(norm_suffix) or c.endswith("_var") or c.endswith("_prec"):
            continue
        cols.append(c)
    return cols


def _map_to_raw_intensity(
    tokens: Sequence[str],
    intensity_cols: Sequence[str],
    intensity_prefix: str
) -> List[str]:
    available = set(intensity_cols)
    mapped = []
    for tok in tokens:
        s = str(tok).strip()
        if not s:
            continue
        candidates = [s, f"{intensity_prefix}{s}"]
        for cand in candidates:
            if cand in available:
                mapped.append(cand)
                break
    return _unique_preserve_order(mapped)


def _map_for_sort(tokens: Sequence[str], df_cols: Sequence[str], intensity_prefix: str, norm_suffix: str) -> List[str]:
    """Map annotation tokens to analysis columns, preferring normalized *_norm columns."""
    cols = set(map(str, df_cols))
    mapped = []
    for tok in tokens:
        s = str(tok).strip()
        if not s:
            continue
        candidates = [s, f"{intensity_prefix}{s}", f"{s}{norm_suffix}", f"{intensity_prefix}{s}{norm_suffix}"]
        chosen = None
        for cand in candidates:
            if cand not in cols:
                continue
            if cand.endswith(norm_suffix):
                chosen = cand
                break
            if cand.startswith(intensity_prefix):
                norm = f"{cand}{norm_suffix}"
                chosen = norm if norm in cols else cand
                break
        if chosen is not None:
            mapped.append(chosen)
    return _unique_preserve_order(mapped)


def _batch_series(df: pd.DataFrame, batch_col: Optional[str] = None) -> pd.Series:
    if batch_col is not None and batch_col in df.columns:
        return df[batch_col]
    if "batch" in df.columns:
        return df["batch"]
    if "file_name" in df.columns:
        return df["file_name"]
    return pd.Series(["__all__"] * len(df), index=df.index)


def tmt_normalization(df: pd.DataFrame,
                      intensity_prefix: str = "intensity_",
                      min_fraction_valid: float = 0.5,
                      use_gis_for_batch: bool = True,
                      gis_column: str = "mix_channels",
                      normalize_target: Literal["auto", "median", "gis"] = "auto",
                      type_experiment: str = "whole proteome",
                      return_suffix: str = "_norm",
                      duplicate_spectrum: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Log2-transform and normalize TMT reporter intensities.

    Returns a copy of df with added `<intensity_*><return_suffix>` columns.
    Original intensity columns are left untouched. Rows failing
    min_fraction_valid receive NaN in normalized columns; the table shape is
    never changed.

    duplicate_spectrum is accepted for backward compatibility. Normalization
    factors are robust medians, so duplicate spectra do not need to be dropped
    to write values back safely by index.
    """
    del duplicate_spectrum  # kept only for API compatibility; no merge is used

    out = df.copy()
    if intensity_prefix=='tmt_':
        _SAGE_TMT_MAP = {
            "tmt_1":  "126",
            "tmt_2":  "127N",
            "tmt_3":  "127C",
            "tmt_4":  "128N",
            "tmt_5":  "128C",
            "tmt_6":  "129N",
            "tmt_7":  "129C",
            "tmt_8":  "130N",
            "tmt_9":  "130C",
            "tmt_10": "131",
            "tmt_11": "131C",
        }
        rename_map = {k: f"tmt_{v}" for k, v in _SAGE_TMT_MAP.items()}
        out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
        
    intensity_cols = _raw_intensity_columns(out, intensity_prefix, return_suffix)
    norm_cols = [f"{c}{return_suffix}" for c in intensity_cols]
    for c in norm_cols:
        out[c] = np.nan

    if out.empty:
        logger.warning("Input DataFrame is empty. Skipping TMT normalization.")
        return out
    if not intensity_cols:
        logger.warning("No raw intensity columns found. Skipping TMT normalization.")
        return out

    # Zeros are missing reporter signals in TMT, not measured values.
    X = out[intensity_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    min_valid = max(1, int(np.ceil(len(intensity_cols) * float(min_fraction_valid))))
    valid_mask = X.notna().sum(axis=1) >= min_valid
    work_idx = out.index[valid_mask]
    if len(work_idx) == 0:
        logger.warning("All PSMs filtered by min_fraction_valid; normalized columns left as NaN.")
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        X_log = np.log2(X.loc[work_idx, intensity_cols])

    batches = _batch_series(out.loc[work_idx])
    batch_levels = list(pd.unique(batches))        

    # GIS channels per batch, using the first non-empty annotation in the batch.
    batch_gis = {}
    if use_gis_for_batch and gis_column in out.columns:
        for b in batch_levels:
            idx = batches[batches == b].index
            cols = []
            for cell in out.loc[idx, gis_column].dropna():
                cols = _map_to_raw_intensity(_parse_channel_cell(cell), intensity_cols, intensity_prefix)
                if cols:
                    break
            batch_gis[b] = cols
    has_gis = any(len(v) > 0 for v in batch_gis.values())

    phospho = str(type_experiment).strip().lower() == "phospho enrichment"
    requested = str(normalize_target or "auto").strip().lower()
    if requested not in {"auto", "median", "gis"}:
        logger.warning(f"Unknown normalize_target={normalize_target!r}; using 'auto'.")
        requested = "auto"

    if requested == "auto":
        mode = "gis" if (phospho and has_gis) else "median"
    elif requested == "gis":
        mode = "gis" if has_gis else "median"
        if not has_gis:
            logger.warning("normalize_target='gis' requested but no GIS channels were found; falling back to median.")
    else:
        mode = "median"

    n_gis_batches = sum(1 for v in batch_gis.values() if len(v) > 0)
    if mode == "gis" and n_gis_batches < 2:
        logger.warning("GIS normalization needs GIS channels in at least two batches; falling back to median.")
        mode = "median"

    if phospho and mode == "median":
        logger.warning(
            "Phospho-enrichment data without usable GIS are median-normalized. "
            "This assumes most phosphosites are unchanged across channels and can "
            "remove real global phosphorylation differences. Prefer a global "
            "internal standard (mix_channels) for enrichment designs."
        )

    if mode == "median":
        # Within-plex channel centering: correct unequal loading/labeling while
        # keeping batch-specific scale separate before cross-batch alignment.
        for b in batch_levels:
            idx = batches[batches == b].index
            ch_med = X_log.loc[idx, intensity_cols].median(axis=0, skipna=True)
            if ch_med.notna().sum() == 0:
                continue
            center = float(np.nanmedian(ch_med.to_numpy(dtype=float)))
            X_log.loc[idx, intensity_cols] = X_log.loc[idx, intensity_cols].subtract(ch_med, axis=1).add(center)

        # Cross-batch alignment: prefer GIS scalar if available, otherwise the
        # all-channel median of the already within-batch-centered values.
        shifts = {}
        for b in batch_levels:
            idx = batches[batches == b].index
            gis_cols = batch_gis.get(b, []) if use_gis_for_batch else []
            source_cols = gis_cols if gis_cols else intensity_cols
            vals = X_log.loc[idx, source_cols].to_numpy(dtype=float)
            shift = np.nanmedian(vals)
            if np.isfinite(shift):
                shifts[b] = float(shift)
        if len(shifts) > 1:
            global_shift = float(np.nanmedian(list(shifts.values())))
            for b, shift in shifts.items():
                idx = batches[batches == b].index
                X_log.loc[idx, intensity_cols] = X_log.loc[idx, intensity_cols].subtract(shift - global_shift, axis=0)

    elif mode == "gis":
        # Enrichment design: align plexes by the global internal standard only.
        # No channel-wise median centering is applied, so global phosphorylation
        # changes within a plex are not normalized away.
        shifts = {}
        for b in batch_levels:
            gis_cols = batch_gis.get(b, [])
            if not gis_cols:
                continue
            idx = batches[batches == b].index
            vals = X_log.loc[idx, gis_cols].to_numpy(dtype=float)
            shift = np.nanmedian(vals)
            if np.isfinite(shift):
                shifts[b] = float(shift)
        if len(shifts) > 1:
            global_shift = float(np.nanmedian(list(shifts.values())))
            for b, shift in shifts.items():
                idx = batches[batches == b].index
                X_log.loc[idx, intensity_cols] = X_log.loc[idx, intensity_cols].subtract(shift - global_shift, axis=0)

    out.loc[work_idx, norm_cols] = X_log[intensity_cols].to_numpy(dtype=float)
    return out



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