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

def setup_logger(log_file):
    logger = logging.getLogger("proteomics")
    if not logger.handlers:  # ← предотвращает повторное добавление
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

import argparse
import os

def find_default_file(file):
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', file),
        os.path.join(os.getcwd(), file),
        os.path.expanduser(f'~/.config/OpenPtmFinder/{file}'),
        f'/etc/OpenPtmFinder/{file}',
    ]
    for path in candidates:
        print(path)
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            return abs_path
    return None  


def parse_args():
    default_config = find_default_file('config.ini')

    parser = argparse.ArgumentParser(description="PTM Annotation Tool Based on Open strategy search")
    parser.add_argument('--config', default=default_config, help='Path to config file')
    parser.add_argument('--run-server', action='store_true', help='Start web server after processing')

    args = parser.parse_args()

    if not args.config or not os.path.isfile(args.config):
        parser.error("No valid config file found. Please specify with --config.")

    return args



def safe_execute(logger, description, func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        logger.info(f"{description} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error while {description}: {e}", exc_info=True)
        return None
    
def existing_file(output_path):
    if os.path.exists(output_path):
        print(f"File {output_path} already exists, step skipped.")
        return True
    else: 
        return False

def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    output_dir = config['PATHS']['output_dir']
    logger = setup_logger(output_dir+'openptmfinder.log')
    logger.info("Launching the program")

    fasta_file = config['PATHS']['fasta_file']
    data_dir = config['PATHS']['data_dir']
    mzml_dir = config['PATHS']['mzml_dir']
    pepxml_dir = config['PATHS']['pepxml_dir']
    xml_file = config['PATHS']['unimod_xml_file']
    interpretation_file = config['PATHS']['interpretation_file']

    type_of_modification = config['PARAMETERS']['type_of_modifications']
    name_of_modification = config['PARAMETERS']['name_of_modifications']
    localization_score_threshold = float(config['PARAMETERS']['localization_score_threshold'])
    mass_tolerance = float(config['PARAMETERS']['mass_tolerance'])
    fdr_threshold = float(config['PARAMETERS']['fdr_threshold'])
    type_tmt = config['PARAMETERS']['type_tmt']
    group_df_link = config['PATHS']['group_df_link']
    calculation_pval = config.getboolean('PARAMETERS', 'calculation_pval')
    min_group_for_stats = int(config['PARAMETERS']['min_group_for_stats'])
    sorting_pepxml=config['PARAMETERS']['sorting_pepxml']

    cataloque = None
    filtered_df = None
    filtered_df_with_intens = None
    stats_df = None
    
    if existing_file(output_dir+'unimod.csv')==True:
        unimod_df=pd.read_csv(output_dir+'unimod.csv')
    else:
        unimod_df = safe_execute(logger, "Processing AA_stat results", create_unimod_dataframe, interpretation_file, xml_file)
        unimod_csv = os.path.join(output_dir, "unimod.csv")
        unimod_df.to_csv(unimod_csv, index=False)
        logger.info(f"Unimod shift annotation saved in {unimod_csv}")

        
    if unimod_df is not None:
        if existing_file(output_dir+'cataloque.csv')==True and existing_file(output_dir+'unimod_search.csv')==True:
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
            
            
    if existing_file(output_dir+'pepxml_psms.pickle')==True:
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
            
            
    if existing_file(output_dir+'predicted_rt.csv')==True:
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
            
            
    if existing_file(output_dir+'filtered_psms.pickle')==True:
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