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
    statistics
)

# --- Константы для настроек по умолчанию ---
DEFAULT_LOG_FILE = 'openptmfinder.log'
DEFAULT_NPROC = max(1, os.cpu_count() - 1)
DEFAULT_PORT = 10030
DEFAULT_VERBOSITY = 'INFO'


def setup_logger(log_file_path, verbosity=DEFAULT_VERBOSITY):
    log_level = getattr(logging, verbosity.upper(), logging.INFO)
    logger = logging.getLogger("OpenPtmFinder")
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='[%H:%M:%S]')
    
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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
    parser.add_argument('-n', '--nproc', type=int, default=DEFAULT_NPROC,
                        help=f'Number of processes to use. Default: {DEFAULT_NPROC}')
    parser.add_argument('-pr', '--port', type=int, default=DEFAULT_PORT,
                        help=f'Port')
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
        'nproc': args.nproc if args.nproc is not None else cfg('PATHS', 'nproc', fallback=DEFAULT_NPROC),
        'port': args.port if args.port is not None else cfg('PATHS', 'port', fallback=DEFAULT_PORT),
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

    logger.info(f'PARAMETERS:\n  - type_of_modifications: {type_of_modification}\n  - name_of_modification: {name_of_modification}\n  - localization_score_threshold: {localization_score_threshold}\n  - mass_tolerance: {mass_tolerance}\n  - fdr_threshold: {fdr_threshold}\n  - type_tmt: {type_tmt}\n  - calculation_pval: {calculation_pval}\n  - min_group_for_stats: {min_group_for_stats}\n  - sorting_pepxml: {sorting_pepxml}\n  - port: {port_n}')

    stats_df_path = os.path.join(output_dir, 'final_stat_result.csv')

    filtered_df_with_intens = None

    if not args.recalc_results and os.path.exists(stats_df_path):
        logger.info(f"Found existing final results at {stats_df_path}. Loading data.")
        try:
            filtered_df_with_intens = pd.read_csv(stats_df_path)
            logger.info("Successfully loaded final results. Skipping recalculation.")
        except Exception as e:
            logger.warning(f"Could not load final results from {stats_df_path}: {e}. Proceeding with full recalculation.")
            args.recalc_results = True
            
    if args.recalc_results or not os.path.exists(stats_df_path):
        logger.info("Starting full recalculation of results...")
        
        # Шаг 1: unimod_df
        unimod_csv_path = os.path.join(output_dir, 'unimod.csv')
        if os.path.exists(unimod_csv_path):
            logger.info(f"Loading existing unimod data from {unimod_csv_path}")
            unimod_df = pd.read_csv(unimod_csv_path)
        else:
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
            logger.info(f"Loading existing cataloque from {cataloque_csv_path}")
            cataloque = pd.read_csv(cataloque_csv_path)
            unimod_search = pd.read_csv(unimod_search_csv_path)
        else:
            cataloque, unimod_search = safe_execute(
                logger, "Generate a table with peptide identifications based on AA_stat results",
                cataloque_create, unimod_df, name_of_modification, type_of_modification,
                data_dir, localization_score_threshold)
            if cataloque is not None and unimod_search is not None:
                cataloque.to_csv(cataloque_csv_path, index=False)
                logger.info(f"The catalog is saved in {cataloque_csv_path}.")
                unimod_search.to_csv(unimod_search_csv_path, index=False)
            else:
                sys.exit(1)

        # Шаг 3: all_psms_df
        all_psms_pickle_path = os.path.join(output_dir, 'pepxml_psms.pickle')
        if os.path.exists(all_psms_pickle_path):
            logger.info(f"Loading existing PSMs from {all_psms_pickle_path}")
            all_psms_df = pd.read_pickle(all_psms_pickle_path)
        else:
            all_psms_df = safe_execute(
                logger, "processing pepXML files",
                process_pepxml_files, cataloque, pepxml_dir,
                mass_tolerance, fdr_threshold, sorting_pepxml, nproc)
            if all_psms_df is not None:
                all_psms_df.to_pickle(all_psms_pickle_path)
                all_psms_df.to_csv(os.path.join(output_dir, "pepxml_psms.csv"))
                logger.info(f"Final save to file: {all_psms_pickle_path}")
                logger.info(f"Found {len(all_psms_df)} PSMs from pepXML files.")
            else:
                sys.exit(1)
        
        # Шаг 4: filtered_df
        filtered_psms_pickle_path = os.path.join(output_dir, 'filtered_psms.pickle')
        if os.path.exists(filtered_psms_pickle_path):
            logger.info(f"Loading existing filtered PSMs from {filtered_psms_pickle_path}")
            filtered_df = pd.read_pickle(filtered_psms_pickle_path)
        else:
            filtered_df = safe_execute(logger, "prediction RT", prediction_rt, all_psms_df)
            if filtered_df is not None:
                rt_csv_path = os.path.join(output_dir, "filtered_psms.pickle")
                filtered_df.to_pickle(rt_csv_path, index=False)
                logger.info(f"Predicted RT saved in {rt_csv_path}")


        # Шаг 5: filtered_df_with_intens
        intens_pickle_path = os.path.join(output_dir, 'filtered_pms_intens.pickle')
        if os.path.exists(intens_pickle_path):
            logger.info(f"Loading existing intensities data from {intens_pickle_path}")
            filtered_df_with_intens = pd.read_pickle(intens_pickle_path)
        else:
            if filtered_df is not None:
                filtered_df_with_intens = safe_execute(logger, "intensities from mzML", intensity, mzml_dir, filtered_df, nproc)
                if filtered_df_with_intens is not None:
                    filtered_df_with_intens.to_pickle(intens_pickle_path)
                    logger.info(f"The data frame with intensities is saved in {intens_pickle_path}")
                else:
                    sys.exit(1)

        # Шаг 6: Аннотация
        annotated_pickle_path = os.path.join(output_dir, 'annotated_df.pickle')
        if os.path.exists(annotated_pickle_path):
            logger.info(f"Loading existing annotated data from {annotated_pickle_path}")
            filtered_df_with_intens = pd.read_pickle(annotated_pickle_path)
        else:
            if filtered_df_with_intens is not None:
                filtered_df_with_intens = safe_execute(logger, "annotation of TMT labels", tags_annotation, filtered_df_with_intens, type_tmt, nproc)
                if filtered_df_with_intens is None: sys.exit(1)
                filtered_df_with_intens = safe_execute(logger, "annotation of samples", samples_annotation, filtered_df_with_intens, group_df_link)
                if filtered_df_with_intens is None: sys.exit(1)
                filtered_df_with_intens = safe_execute(logger, "FASTA concat", fasta_concat, filtered_df_with_intens, fasta_file)
                if filtered_df_with_intens is None: sys.exit(1)
                filtered_df_with_intens.to_pickle(annotated_pickle_path)
                logger.info(f"The dataframe with annotation is saved in {annotated_pickle_path}")
            else:
                sys.exit(1)
        
        # Шаг 7: Статистика
        if filtered_df_with_intens is not None:
            stats_df, stat, num, delete_indices = safe_execute(logger, "sorting PSM", sorting_psms, filtered_df_with_intens)
            if stats_df is None: sys.exit(1)
            
            logger.info(f"TMT intensities were replaced or removed: {stat}")
            
            stats_df = safe_execute(logger, "normalization", tmt_normalization, stats_df)
            if stats_df is None: sys.exit(1)

            stats_df = safe_execute(logger, "calculate statistics", statistics, stats_df, calculation_pval, min_group_for_stats)
            if stats_df is not None:
                stats_df.to_csv(stats_df_path, index=False)
                logger.info(f"The final statistical result is saved in {stats_df_path}")
            else:
                sys.exit(1)
    
    if filtered_df_with_intens is None:
        logger.error("Could not find or calculate final results. Exiting.")
        sys.exit(1)
        
    logger.info("The program is complete.")
    
    if args.run_server:
        logger.info("Launching web server for interactive exploration...")
        os.environ['OUTPUT_DIR'] = output_dir
        os.environ['port_n'] = str(port_n) # Передаем порт как строку
        subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'webserver.py')])
    else:
        logger.info("Web server was not requested (--run_server). Shutting down.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()