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
    calibrate_RT_gaus_full
)


def setup_logger(log_file_path, verbosity='INFO'):

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
        os.path.join(os.path.dirname(__file__), '..', file),
        os.path.join(os.getcwd(), file),
        os.path.expanduser(f'~/.config/OpenPtmFinder/{file}'),
        f'/etc/OpenPtmFinder/{file}',
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            return abs_path
    return None  


def parse_args():
    default_config = find_default_file('config.ini')

    parser = argparse.ArgumentParser(description="PTM Annotation Tool Based on Open strategy search", prog='OpenPtmFinder')
    parser.add_argument('--config', default=default_config, help='.ini file with parameters. If there is no file, OpenPtmFinder uses default one.')
    parser.add_argument('--output_dir', help='Directory to store the results. Default value is current directory.')
    parser.add_argument('--pepxml_dir', nargs='+', help='Directory with pepxml search files or separate files. Default value is current directory.')
    parser.add_argument('--mzml_dir', help='Directory with mzml search files. Default value is current directory.')
    parser.add_argument('--AAstat_dir', help='Directory with AA_stat search results (.csv and interpretations.json). Default value is current directory.')
    
    parser.add_argument('--protein_db', help='Directory with .fasta file with proteins. If there is no file, OpenPtmFinder uses default one.')
    parser.add_argument('--unimod_db', help='Directory with .xml UNIMOD database. An example can be found at https://github.com/Anchovy-bio/OpenPtmFinder/tree/main/data/unimod')
    parser.add_argument('--grouping_file', help='Directory with annotation file of samples by TMT groups. An example can be found at https://github.com/Anchovy-bio/OpenPtmFinder/blob/main/config.ini')
    
    parser.add_argument('--run-server', action='store_true', help='Start web server after processing')
    parser.add_argument('-n', type=int, help='Maximum number of processes to use.', default=1)
    parser.add_argument('--verbosity', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging verbosity level')

    args = parser.parse_args()

    if not args.config or not os.path.isfile(default_config):
        parser.error("No valid config file found. Please specify with --config.")

    return args


def get_final_paths(args, config):
    def cfg(section, key, fallback='.'):
        return config.get(section, key, fallback=fallback)

    return {
        'output_dir': args.output_dir or cfg('PATHS', 'output_dir'),
        'pepxml_dir': args.pepxml_dir or cfg('PATHS', 'pepxml_dir').split(),
        'mzml_dir': args.mzml_dir or cfg('PATHS', 'mzml_dir').split(),
        'aa_stat_dir': args.AAstat_dir or cfg('PATHS', 'aa-stat_dir'),
        'protein_db': args.protein_db or cfg('PATHS', 'protein_db'),
        'unimod_db': args.unimod_db or cfg('PATHS', 'unimod_db'),
        'grouping_file': args.grouping_file or cfg('PATHS', 'grouping_file'),
    }



def safe_execute(logger, description, func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        logger.info(f"{description} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error while {description}: {e}", exc_info=True)
        return None
    
    
def existing_file(output_path, logger):
    if os.path.exists(output_path):
        logger.info(f"File {output_path} already exists, step skipped.")
        return True
    else:
        logger.error(f"File does not exist at {output_path}", exc_info=True)
        return False

    
def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    paths = get_final_paths(args, config)

    output_dir = paths['output_dir']
    logger = setup_logger(os.path.join(output_dir, 'openptmfinder.log'), verbosity=args.verbosity)
    logger.info("Starting OpenPtmFinder...")

    fasta_file = paths['protein_db']
    xml_file = paths['unimod_db']
    group_df_link = paths['grouping_file']
    data_dir = paths['aa_stat_dir']
    mzml_dir = paths['mzml_dir']
    pepxml_dir = paths['pepxml_dir']
    print(data_dir)
    print(os.path.join(data_dir,'interpretations.json'))

    if existing_file(os.path.join(data_dir,'interpretations.json'),logger):
        interpretation_file = os.path.join(data_dir,'interpretations.json')

    type_of_modification = config['PARAMETERS']['type_of_modifications']
    name_of_modification = config['PARAMETERS']['name_of_modifications']
    localization_score_threshold = float(config['PARAMETERS']['localization_score_threshold'])
    mass_tolerance = float(config['PARAMETERS']['mass_tolerance'])
    fdr_threshold = float(config['PARAMETERS']['fdr_threshold'])
    type_tmt = config['PARAMETERS']['type_tmt']
    calculation_pval = config.getboolean('PARAMETERS', 'calculation_pval')
    min_group_for_stats = int(config['PARAMETERS']['min_group_for_stats'])
    sorting_pepxml=config['PARAMETERS']['sorting_pepxml']

    cataloque = None
    filtered_df = None
    filtered_df_with_intens = None
    stats_df = None
    
    if existing_file(output_dir+'unimod.csv', logger):
        unimod_df=pd.read_csv(output_dir+'unimod.csv')
    else:
        unimod_df = safe_execute(logger, "Processing AA_stat results", create_unimod_dataframe, interpretation_file, xml_file)
        unimod_csv = os.path.join(output_dir, "unimod.csv")
        unimod_df.to_csv(unimod_csv, index=False)
        logger.info(f"Unimod shift annotation saved in {unimod_csv}")

        
    if unimod_df is not None:
        if existing_file(output_dir+'cataloque.csv', logger)==True and existing_file(output_dir+'unimod_search.csv', logger)==True:
            cataloque=pd.read_csv(output_dir+'cataloque.csv')
            unimod_search=pd.read_csv(output_dir+'unimod_search.csv')
        else:
            cataloque, unimod_search = safe_execute(
                logger, "Generate a table with peptide identifications based on AA_stat results",
                cataloque_create, unimod_df, name_of_modification, type_of_modification,
                data_dir, localization_score_threshold)
            if cataloque is not None:
                cataloque_csv = os.path.join(output_dir, "cataloque.csv")
                #cataloque_pickle = os.path.join(output_dir, "cataloque.pickle")
                cataloque.to_csv(cataloque_csv, index=False)
                #cataloque.to_pickle(cataloque_pickle)
                logger.info(f"The catalog is saved in {cataloque_csv}.")
            if unimod_search is not None:
                unimod_search_csv = os.path.join(output_dir, "unimod_search.csv")
                unimod_search.to_csv(unimod_search_csv, index=False)
            
            
    if existing_file(output_dir+'pepxml_psms.pickle', logger):
        all_psms_df=pd.read_pickle(output_dir+'pepxml_psms.pickle')
    else:
        all_psms_df = safe_execute(
            logger, "processing pepXML files",
            process_pepxml_files, cataloque, pepxml_dir,
            mass_tolerance, fdr_threshold, sorting_pepxml)
        if all_psms_df is not None:
            output_df = os.path.join(output_dir, "pepxml_psms.pickle")
            output_df_csv = os.path.join(output_dir, "pepxml_psms.csv")
            all_psms_df.to_pickle(output_df)
            all_psms_df.to_csv(output_df_csv)
            logger.info(f"Final save to file: {output_df} and {output_df_csv}")
        if all_psms_df is not None and cataloque is not None:
            logger.info(f"Found {len(all_psms_df)} PSM pepXML files.")
            
            
    if existing_file(output_dir+'predicted_rt.csv', logger):
        predicted_rt_df=pd.read_csv(output_dir+'predicted_rt.csv')
    else:
        if all_psms_df is not None:
            predicted_rt_df = safe_execute(logger, "prediction RT", prediction_rt, all_psms_df)
            if predicted_rt_df is not None:
                rt_csv = os.path.join(output_dir, "predicted_rt.csv")
                predicted_rt_df.to_csv(rt_csv, index=False)
                logger.info(f"Predicted RT saved in{rt_csv}")
                
                XRT_shift, XRT_sigma, covvalues = safe_execute(logger, "selection of values", calibrate_RT_gaus_full, predicted_rt_df['predicted_RT'] - all_psms_df.loc[predicted_rt_df.index, 'retention_time_sec'] )

                rt_diff = predicted_rt_df['predicted_RT'] - all_psms_df.loc[predicted_rt_df.index, 'retention_time_sec']
                outliers = rt_diff[abs(rt_diff) >= 3 * XRT_sigma].index.tolist()
                logger.info(f"Number of emissions: {len(outliers)}")
                filtered_df = predicted_rt_df.drop(index=outliers).reset_index(drop=True)
                filtered_path = os.path.join(output_dir, "filtered_psms.pickle")
                filtered_df.to_pickle(filtered_path)
                logger.info(f"Filtered RTs are saved in {filtered_path} ({len(filtered_df)} rows)")
            else:
                filtered_df=all_psms_df.copy()
            
            
    if existing_file(output_dir+'filtered_psms.pickle', logger):
        filtered_df_with_intens=pd.read_pickle(output_dir+'filtered_psms.pickle')
    else:
        if filtered_df is not None:
            filtered_df_with_intens = safe_execute(logger, "intensities from mzML", intensity, mzml_dir, filtered_df)
            if filtered_df_with_intens is not None:
                filtered_path = os.path.join(output_dir, "filtered_psms.pickle")
                filtered_df_with_intens.to_pickle(filtered_path)
                logger.info(f"The data frame with intensities is saved in {filtered_path}")

    if filtered_df_with_intens is not None:
        filtered_df_with_intens = safe_execute(logger, "annotation of TMT lables", tags_annotation, filtered_df_with_intens, type_tmt)
        filtered_df_with_intens = safe_execute(logger, "annotation of samples", samples_annotation, filtered_df_with_intens, group_df_link)
        filtered_df_with_intens = safe_execute(logger, "FASTA concat", fasta_concat, filtered_df_with_intens, fasta_file)
        if filtered_df_with_intens is not None:
            filtered_path = os.path.join(output_dir, "annotated_df.pickle")
            filtered_df_with_intens.to_pickle(filtered_path)
            logger.info(f"The dataframe with annotation is saved in {filtered_path}")

    if filtered_df_with_intens is not None:
        stats_df,stat, num, delete_indices = safe_execute(logger, "sorting PSM", sorting_psms, filtered_df_with_intens)
        logger.info(f"TMT intensities were replaced or removed {stat}")
        stats_df = safe_execute(logger, "normalization", tmt_normalization, stats_df)
        stats_csv = os.path.join(output_dir, "final_stat_result.csv")
        stats_df.to_csv(stats_csv, index=False)
        stats_df = safe_execute(logger, "calculate statistics", statistics, stats_df, calculation_pval, min_group_for_stats)
        if stats_df is not None:
            stats_csv = os.path.join(output_dir, "final_stat_result.csv")
            stats_df.to_csv(stats_csv, index=False)
            logger.info(f"The final statistical result is saved in {stats_csv}")

    logger.info("The program is complete.")
    logger.info("Launching web server for interactive exploration...")
    if args.run_server:
        os.environ['OUTPUT_DIR'] = output_dir
        subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'webserver.py')])


if __name__ == "__main__":
    main()