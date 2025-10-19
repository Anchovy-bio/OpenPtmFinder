import subprocess
import sys
import csv
import numpy as np
import pandas as pd
import argparse
import logging
import os
import re
import json
import plotly
import configparser
from xml.etree import ElementTree as ET
from pyteomics import pepxml, mzml, fasta
from scipy import stats
import sys
import io
import multiprocessing as mp
from Bio import pairwise2
import warnings
warnings.filterwarnings("ignore")

from .functions import (
    create_unimod_dataframe,
    cataloque_create,
    process_pepxml_files,
    intensity,
    prediction_rt,
    tags_annotation,
    samples_annotation,
    fasta_concat,
    sorting_psms,
    tmt_normalization,
    statistics,
    spectra_merge,
    map_mod_position
)
from .dbconnect import (
    get_protein_info_from_signor,
    fetch_iptmnet_data,
    get_dbptm_download_links,
    grafs
)

# --- Константы для настроек по умолчанию ---
DEFAULT_LOG_FILE = 'openptmfinder.log'
DEFAULT_NPROC = 1
DEFAULT_PORT = 10030
DEFAULT_VERBOSITY = 'INFO'

def setup_logger(log_file_path, verbosity=DEFAULT_VERBOSITY):
    log_level = getattr(logging, verbosity.upper(), logging.INFO)
    
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='[%H:%M:%S]')
    
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Добавляем обработчик для записи в файл
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Добавляем обработчик для вывода в консоль
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def find_default_file(file):
    candidates = [
        os.path.join(os.getcwd(), file),
        os.path.expanduser(f'~/.config/OpenPtmFinder/{file}'),
        f'/etc/OpenPtmFinder/{file}',
        os.path.join(os.path.dirname(__file__), file),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            return abs_path
    return None

def parse_command():
    default_config = find_default_file('config.ini')

    parser = argparse.ArgumentParser(description="PTM Annotation Tool Based on Open strategy search", prog='OpenPtmFinder', epilog='See more information at https://github.com/Anchovy-bio/OpenPtmFinder/')
    parser.add_argument('-c','--config', default=default_config, help='Pathway to config.ini file with parameters. If there is no file, OpenPtmFinder uses default one.')
    parser.add_argument('-o','--output_dir', help='Directory to store the results. Default value is current directory.')
    parser.add_argument('-p','--pepxml', nargs='+', help='Directory or separate files include pepxml search from MSFragger. Default value is current directory.')
    parser.add_argument('-m','--mzml', help='Directory includes mzml files. Default value is current directory.')
    parser.add_argument('-a','--AAstat_dir', help='Directory with AA_stat search results (.csv and interpretations.json). Default value is current directory.')
    parser.add_argument('-d','--protein_db', help='Directory with .fasta file with proteins. If there is no file, OpenPtmFinder uses default one.')
    parser.add_argument('-u','--unimod_db', help='Directory with .xml UNIMOD database. If there is no file, OpenPtmFinder uses default one (version from 2025).')
    parser.add_argument('-g','--grouping_file', help='Directory with annotation file of samples by TMT groups. An example can be found at https://github.com/Anchovy-bio/OpenPtmFinder/')
    
    parser.add_argument('--run_server', action='store_true', help='Start web server after processing', default=False)
    parser.add_argument('--recalc_results', action='store_true', help='Recalculate results', default=False)
    parser.add_argument('-n', '--nproc', type=int,
                        help=f'Number of processes to use.')
    parser.add_argument('-pr', '--port', type=int, help=f'Port')
    parser.add_argument('-v','--verbosity', default=DEFAULT_VERBOSITY, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help=f'Logging verbosity level. Choose from:DEBUG, INFO, WARNING, ERROR, CRITICAL. Default {DEFAULT_VERBOSITY}.')

    args = parser.parse_args()

    if not args.config and not find_default_file('config.ini'):
        parser.error("Default config file was not found. Please specify a path with --config.")
        
    return args

def get_final_paths(args, config):
    def cfg(section, key, fallback=None):
        return config.get(section, key, fallback=fallback)

    return {
        'output_dir': args.output_dir or cfg('PATHS', 'output_dir', fallback=os.getcwd()),
        'pepxml_dir': args.pepxml or cfg('PATHS', 'pepxml_dir').split(),
        'mzml_dir': args.mzml or cfg('PATHS', 'mzml_dir'),
        'aa_stat_dir': args.AAstat_dir or cfg('PATHS', 'aa-stat_dir'),
        'protein_db': args.protein_db or cfg('PATHS', 'protein_db'),
        'unimod_db': args.unimod_db or cfg('PATHS', 'unimod_db'),
        'grouping_file': args.grouping_file or cfg('PATHS', 'grouping_file'),
        'nproc': args.nproc if args.nproc is not None else cfg('PARAMETERS', 'nproc', fallback=DEFAULT_NPROC),
        'port': args.port if args.port is not None else cfg('PARAMETERS', 'port', fallback=DEFAULT_PORT),
    }

def safe_execute(logger, description, func, *args, **kwargs):
    logger.info(f"Starting: {description}...")
    try:
        result = func(*args, **kwargs)
        logger.info(f"Successfully completed: {description}")
        return result
    except Exception as e:
        logger.error(f"Error while {description}: {e}", exc_info=True)
        return None

def main():
    args = parse_command()
    config = configparser.ConfigParser()
    
    if args.config:
        config.read(os.path.abspath(args.config))
    else:
        # If no config is specified and default is not found, parser.error handles it.
        # This part will be reached if a default config was found.
        config.read(find_default_file('config.ini'))
        
    paths = get_final_paths(args, config)
    output_dir = os.path.abspath(paths['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    nproc = int(paths['nproc'])
    port_n = int(paths['port'])

    logger = setup_logger(os.path.join(output_dir, DEFAULT_LOG_FILE), verbosity=args.verbosity)
    logger.info("Starting OpenPtmFinder...")
    logger.info(f'Config file was found in {args.config}')
    logger.info(f"Using {nproc} process(es)")
    
    fasta_file = paths['protein_db']
    xml_file = paths['unimod_db']
    group_df_link = paths['grouping_file']
    data_dir = paths['aa_stat_dir']
    mzml_dir = paths['mzml_dir']
    pepxml_dir = paths['pepxml_dir']
    
    logger.info(f"Paths:\n  - output_dir: {output_dir}\n  - protein_db: {fasta_file}\n  - unimod_db: {xml_file}\n  - grouping_file: {group_df_link}\n  - aa_stat_dir: {data_dir}\n  - mzml_dir: {mzml_dir}\n  - pepxml_dir: {pepxml_dir}")

    # Проверка обязательных файлов
    required_files = {
        'interpretation_file': os.path.join(data_dir, 'interpretations.json'),
    }
    for name, path in required_files.items():
        if not os.path.isfile(path):
            logger.error(f"Required file {name} not found at {path}. Exiting.")
            sys.exit(1)
            
    interpretation_file = required_files['interpretation_file']
    logger.info('File interpretations.json was found.')

    type_of_modification = re.split(r'\s*,\s*', config.get('PARAMETERS', 'type_of_modifications', fallback=''))
    name_of_modification = re.split(r'\s*,\s*', config.get('PARAMETERS', 'name_of_modifications', fallback=''))
    localization_score_threshold = float(config.get('PARAMETERS', 'localization_score_threshold', fallback=0.75))
    mass_tolerance = float(config.get('PARAMETERS', 'mass_tolerance', fallback=10))
    fdr_threshold = float(config.get('PARAMETERS', 'fdr_threshold', fallback=0.01))
    type_tmt = config.get('PARAMETERS', 'type_tmt', fallback='tmt10plex')
    calculation_pval = config.getboolean('PARAMETERS', 'calculation_pval', fallback=True)
    min_group_for_stats = int(config.get('PARAMETERS', 'min_group_for_stats', fallback=2))
    sorting_pepxml = config.get('PARAMETERS', 'sorting_pepxml', fallback='False')
    min_hits_for_fdr_calc=int(config.get('PARAMETERS', 'min_hits_for_fdr_calc', fallback=20))
    default_hyperscore_threshold=int(config.get('PARAMETERS', 'default_hyperscore_threshold', fallback=20))
    default_expect_threshold=float(config.get('PARAMETERS', 'default_expect_threshold', fallback=0.05))

    logger.info(f'PARAMETERS:\n  - type_of_modifications: {type_of_modification}\n  - name_of_modification: {name_of_modification}\n  - localization_score_threshold: {localization_score_threshold}\n  - mass_tolerance: {mass_tolerance}\n  - fdr_threshold: {fdr_threshold}\n  - type_tmt: {type_tmt}\n  - calculation_pval: {calculation_pval}\n  - min_group_for_stats: {min_group_for_stats}\n  - sorting_pepxml: {sorting_pepxml}\n  - port: {port_n}\n  - min_hits_for_fdr_calc: {min_hits_for_fdr_calc}\n  - default_hyperscore_threshold: {default_hyperscore_threshold}\n  - default_expect_threshold: {default_expect_threshold}')

    stats_df_path = os.path.join(output_dir, 'final_stat_result.csv')

    filtered_df_with_intens = None

    if not args.recalc_results and os.path.exists(stats_df_path):
        logger.info(f"Found existing final results at {stats_df_path}. Loading data.")
        try:
            stats_df = pd.read_csv(stats_df_path)
            logger.info("Successfully loaded final results. Skipping recalculation.")
        except Exception as e:
            logger.warning(f"Could not load final results from {stats_df_path}: {e}. Proceeding with full recalculation.")
            args.recalc_results = True
            
    if args.recalc_results or not os.path.exists(stats_df_path):
        logger.info("Starting full recalculation of results...")
        
        # Step 1: unimod_df
        unimod_csv_path = os.path.join(output_dir, 'unimod.csv')
        unimod_df = safe_execute(logger, "Processing AA_stat results", create_unimod_dataframe, interpretation_file, xml_file)
        if unimod_df is not None:
            unimod_df.to_csv(unimod_csv_path, index=False)
            logger.info(f"Unimod shift annotation saved in {unimod_csv_path}")
        else:
            sys.exit(1)

        # Шаг 2: cataloque
        cataloque_csv_path = os.path.join(output_dir, 'cataloque.csv')
        unimod_search_csv_path = os.path.join(output_dir, 'unimod_search.csv')
        if os.path.exists(cataloque_csv_path) and os.path.exists(unimod_search_csv_path):
            logger.info(f"Cataloque and unimod's files already exist in {cataloque_csv_path}")
        else:
            cataloque, unimod_search = safe_execute(
                logger, "Generate a table with peptide identifications based on AA_stat results",
                cataloque_create, unimod_df, name_of_modification, type_of_modification,
                data_dir, localization_score_threshold)
            if cataloque is not None and unimod_search is not None:
                cataloque.to_csv(cataloque_csv_path, index=False)
                logger.info(f"The catalog is saved in {cataloque_csv_path}.")
                unimod_search.to_csv(unimod_search_csv_path, index=False)
                del unimod_df
                del unimod_search
            else:
                sys.exit(1)

        # Шаг 3: all_psms_df
        all_psms_pickle_path = os.path.join(output_dir, 'pepxml_psms.pickle')
        unique_mass_psms_pickle_path = os.path.join(output_dir, 'unique_mass_psms.pickle')
        if os.path.exists(all_psms_pickle_path):
            logger.info(f"PSMs files already exist in {all_psms_pickle_path}")
        else:
            cataloque = pd.read_csv(cataloque_csv_path)
            unimod_search = pd.read_csv(unimod_search_csv_path)
            all_psms_df, cataloque, results_unique_mods = safe_execute(
                logger, "processing pepXML files",
                process_pepxml_files, cataloque, pepxml_dir,
                mass_tolerance, fdr_threshold, sorting_pepxml, nproc,
                min_hits_for_fdr_calc, default_hyperscore_threshold, default_expect_threshold)
            all_psms_df = safe_execute(logger, "FASTA concat", fasta_concat, all_psms_df, fasta_file)
            cataloque = safe_execute(logger, "FASTA concat", fasta_concat, cataloque, fasta_file)            
            if all_psms_df is not None:
                all_psms_df.to_pickle(all_psms_pickle_path)
                logger.info(f"Final save to file: {all_psms_pickle_path}")
                logger.info(f"Found {len(all_psms_df)} PSMs from pepXML files.")
            else:
                sys.exit(1)
            if cataloque is not None:
                cataloque.to_csv(cataloque_csv_path, index=False)
            else:
                logger.error('Was not updated cataloque.')
            if results_unique_mods is not None:
                results_unique_mods.to_pickle(unique_mass_psms_pickle_path)
                del results_unique_mods
            else:
                logger.error('Was not created unique_mass_df.')
                
        psms_zero_path = os.path.join(output_dir, 'psms_zero.pickle')
        psm_filtered_path = os.path.join(output_dir, 'psm_filtered.pickle')
        if os.path.exists(psm_filtered_path) and os.path.exists(psms_zero_path):
            logger.info(f"PSMs filtered files already exist in {psm_filtered_path}")
        else:
            all_psms_df = pd.read_pickle(all_psms_pickle_path)
            cataloque = pd.read_csv(cataloque_csv_path)
            psm_filtered, psms_zero = safe_execute(
                logger, "merge cataloque and pepxml dfs",
                spectra_merge, cataloque, all_psms_df
            )
            del cataloque
            del all_psms_df
            if psm_filtered is not None:
                psm_filtered.to_pickle(psm_filtered_path)
                logger.info(f"Found {len(psm_filtered)} filtered PSMs from pepXML files.")
                del psm_filtered
            else:
                sys.exit(1)
            if psms_zero is not None:
                psms_zero.to_pickle(psms_zero_path)
                del psms_zero
        
        # Step 4: filtered_df
        filtered_psms_pickle_path = os.path.join(output_dir, 'filtered_psms.pickle')
        if os.path.exists(filtered_psms_pickle_path):
            logger.info(f"Loading existing filtered PSMs from {filtered_psms_pickle_path}")
        else:
            all_psms_df = pd.read_pickle(psm_filtered_path)
            all_psms_df['mod_name'] = all_psms_df['mod_name'].apply(lambda x: ('Glu') if '-Glu-' in x else x)
            all_psms_df["position_mod2"] = all_psms_df.apply(
                lambda row: map_mod_position(row["peptide_x"], row["position_mod"], row["peptide_y"]),
                axis=1
            )
            all_psms_df['for_prediction'] = all_psms_df.apply(
                lambda row: str(row['position_mod2']) + '|' + row['mod_name'],
                axis=1
            )
            filtered_df = safe_execute(logger, "prediction RT", prediction_rt, all_psms_df)
            if filtered_df is not None:
                filtered_df.to_pickle(filtered_psms_pickle_path)
                logger.info(f"Predicted RT saved in {filtered_psms_pickle_path}")
                del filtered_df
                del all_psms_df


        # Step 5: filtered_df_with_intens
        intens_pickle_path = os.path.join(output_dir, 'filtered_pms_intens.pickle')
        intens_zero_path = os.path.join(output_dir, 'pms_zero_intens.pickle')
        if os.path.exists(intens_pickle_path):
            logger.info(f"Intensities data already exist in {intens_pickle_path}")
        else:
            filtered_df = pd.read_pickle(filtered_psms_pickle_path)
            psms_zero = pd.read_pickle(psms_zero_path)
            cataloque = pd.concat([filtered_df[['file_name','index spectrum']].copy(),
                                   psms_zero[['file_name','index spectrum']].copy()], ignore_index = True
                                 )
            del psms_zero
            results_df = safe_execute(logger, "intensities from mzML", intensity, mzml_dir, cataloque.drop_duplicates(), output_dir, nproc)
            if results_df is not None:
                filtered_df = filtered_df.merge(results_df, how = 'left',on=['file_name','index spectrum'])
                filtered_df.to_pickle(intens_pickle_path)
                del filtered_df
                del cataloque
                psms_zero = pd.read_pickle(psms_zero_path)
                psms_zero = psms_zero.merge(results_df, how = 'left', on=['file_name','index spectrum'])
                psms_zero.to_pickle(intens_zero_path)
                logger.info(f"The dataframe with intensities is saved in {intens_pickle_path}")
                del psms_zero
            else:
                sys.exit(1)

        # Step 6: Annotation
        annotated_pickle_path = os.path.join(output_dir, 'annotated_df.pickle')
        if os.path.exists(annotated_pickle_path):
            logger.info(f"Loading existing annotated data from {annotated_pickle_path}")
        else:
            filtered_df_with_intens = pd.read_pickle(intens_pickle_path)
            psms_zero = pd.read_pickle(intens_zero_path)
            if filtered_df_with_intens is not None:
                columns = ['Modification','id_prot','modified_peptide_x','position_in_protein',
                                                  'peptide_x','peptide_y','spectrum_x',
                                                  'spectrum_y','file_name','intensity','m/z','charge','sequence_y']
                annot_df = filtered_df_with_intens[columns].copy()
                final_df = pd.concat([psms_zero[columns],
                                      annot_df], ignore_index = True
                                    )
                del filtered_df_with_intens
                del psms_zero
                del annot_df
                final_df = final_df[~final_df['m/z'].isna()]
                annot_df = safe_execute(logger, "annotation of TMT labels", tags_annotation, final_df, 
                                                       type_tmt, output_dir, nproc)
                if annot_df is None: sys.exit(1)
                annot_df = safe_execute(logger, "annotation of samples", samples_annotation, annot_df, group_df_link)
                if annot_df is None: sys.exit(1)
                del annot_df['m/z']
                del annot_df['intensity']
                if 'Unnamed: 0' in annot_df.columns:
                    del annot_df['Unnamed: 0']
                annot_df.to_pickle(annotated_pickle_path)
                logger.info(f"The dataframe with annotation is saved in {annotated_pickle_path}")
            else:
                sys.exit(1)
        
        # Step 7: Statistics
        sort_df_path=os.path.join(output_dir, "sorted_df.pickle")
        norm_df_path=os.path.join(output_dir, "normalization_df.pickle")

        if (sorting_pepxml == 'True') and (os.path.exists(annotated_pickle_path)==True) and (os.path.exists(sort_df_path) == False):
            if 'annot_df' not in globals():
                annot_df = pd.read_pickle(annotated_pickle_path)
            logger.info(f"Sorting dataframe was starting calculate.")
            stats_df, stat, num, delete_indices = safe_execute(logger, "sorting PSM", sorting_psms, annot_df)
            if stats_df is None: sys.exit(1)
            stats_df.to_pickle(sort_df_path)
            logger.info(f"TMT intensities were replaced: {stat}")
            logger.info(f"TMT intensities were removed: {num}")
            del annot_df
        if os.path.exists(norm_df_path) == False:
            logger.info(f"Start normalisation.")
            if ('stats_df' not in globals()) and (os.path.exists(sort_df_path) == True):
                stats_df = pd.read_pickle(sort_df_path)
            elif ('annot_df' not in globals()) and (sorting_pepxml == 'False'):
                stats_df = pd.read_pickle(annotated_pickle_path)
            stats_df = safe_execute(logger, "normalization", tmt_normalization, stats_df)
            stats_df.to_pickle(norm_df_path)
            del stats_df
            if 'annot_df' in globals():
                del annot_df
            
        logger.info(f"Start calculate statistics.")
        if os.path.exists(norm_df_path) == True:
            stats_df = pd.read_pickle(norm_df_path)
        else:
            sys.exit(1)
        stats_df, stoich = safe_execute(logger, "calculate statistics", statistics, stats_df, calculation_pval, min_group_for_stats)
        stats_df_path = os.path.join(output_dir, "final_stat_result.csv")
        stats_df.to_csv(stats_df_path)
        stoich_path = os.path.join(output_dir, "stoich.pickle")
        stoich.to_pickle(stoich_path)
        logger.info(f"The final statistical result is saved in {stats_df_path}")

        try:    
            logger.info("Start annotation PTMs with db.")
            if stats_df is None:
                stats_df = pd.read_csv(stats_df_path)

            # iPTMnet
            protein_ids = stats_df['id_prot'].unique()
            dbPTNnet = safe_execute(
                logger, "db iPTMnet", fetch_iptmnet_data, protein_ids, max_workers=nproc
            )
            if dbPTNnet is None or dbPTNnet.empty:
                logger.warning("No data fetched from iPTMnet.")
            else:
                df_right_renamed = dbPTNnet.rename(
                    columns={c: f"{c}_iPTMnet" for c in dbPTNnet.columns}
                )
                df_right_renamed = df_right_renamed.rename(
                    columns={'sub_form_iPTMnet': 'id_prot', 'site_iPTMnet': 'position_in_protein'}
                )
                stats_df_db = stats_df.merge(
                    df_right_renamed, how='left', on=['id_prot', 'position_in_protein']
                )
                logger.info("Merged stats_df with iPTMnet annotations. Starts merge dbPTM.")
                '''
                dbPTM = safe_execute(
                    logger, "dbPTM", get_dbptm_download_links, stats_df_db, n_threads = nproc
                )
                '''
                stats_df_path1 = os.path.join(output_dir, "final_stat_result_with_dbs.csv")
                stats_df_db.to_csv(stats_df_path1)
                logger.info(f"The final result is saved in {stats_df_path1}")

            # Signor
            common_df = safe_execute(
                logger, "db Signor", get_protein_info_from_signor, stats_df
            )

            if common_df is not None:
                grafs_df_path = os.path.join(output_dir, "grafs_df.csv")
                common_df.to_csv(grafs_df_path)
                safe_execute(logger, "db Signor", grafs, common_df, stats_df['id_prot'].unique(), output_dir)

        except KeyError as e:
            logger.error(f"KeyError during db annotation: {e}")
        except ValueError as e:
            logger.error(f"ValueError during db annotation: {e}")
        except Exception as e:
            logger.exception("Unexpected error during db annotation.")
        
        
    logger.info("The program is complete.")
    
    if args.run_server:
        logger.info("Launching web server for interactive exploration...")
        os.environ['OUTPUT_DIR'] = output_dir
        os.environ['port_n'] = str(port_n)
        os.environ['fasta'] = fasta_file
        subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'webserver.py')])
    else:
        logger.info("Web server was not requested (--run_server) or have not valid data. Shutting down.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()