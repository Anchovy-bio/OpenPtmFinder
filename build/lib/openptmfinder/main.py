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
import io
import multiprocessing as mp
from Bio import pairwise2
import warnings
warnings.filterwarnings("ignore")

from .calc_stats import (
    statistics)

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
    impute_tmt_psms,
    tmt_normalization,
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
        'iptmnet_positions_file': cfg('PATHS', 'iptmnet_positions_file', fallback=None),
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
        config.read(os.path.abspath(args.config), encoding='utf-8')
    else:
        # If no config is specified and default is not found, parser.error handles it.
        # This part will be reached if a default config was found.
        config.read(find_default_file('config.ini'), encoding='utf-8')

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
    iptmnet_positions_file = paths['iptmnet_positions_file']

    logger.info(f"Paths:\n  - output_dir: {output_dir}\n  - protein_db: {fasta_file}\n  - unimod_db: {xml_file}\n  - grouping_file: {group_df_link}\n  - aa_stat_dir: {data_dir}\n  - mzml_dir: {mzml_dir}\n  - pepxml_dir: {pepxml_dir}\n  - iptmnet_positions_file: {iptmnet_positions_file}")

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
    min_group_for_stats = int(config.get('STATISTICS', 'min_group_for_stats', fallback=2))
    sorting_pepxml = config.get('PARAMETERS', 'sorting_pepxml', fallback='False')
    min_hits_for_fdr_calc = int(config.get('PARAMETERS', 'min_hits_for_fdr_calc', fallback=20))
    default_hyperscore_threshold = int(config.get('PARAMETERS', 'default_hyperscore_threshold', fallback=20))
    default_expect_threshold = float(config.get('PARAMETERS', 'default_expect_threshold', fallback=0.05))

    # --- параметры статистического модуля (секция [STATISTICS]) ---
    method = config.get('STATISTICS', 'calculating_method', fallback='aggregate')
    type_experiment = config.get('STATISTICS', 'type_experiment', fallback='whole proteome')
    min_sites_mod = int(config.get('STATISTICS', 'min_sites_mod', fallback=100))
    min_ref = int(config.get('STATISTICS', 'min_ref', fallback=100))
    min_obs_per_site = float(config.get('STATISTICS', 'min_obs_per_site', fallback=3))
    # 0 = критерий отключен (см. комментарий в блоке статистики)
    min_pairs_for_stoich = float(config.get('STATISTICS', 'min_pairs_for_stoich', fallback=0))
    min_sites_for_common = int(config.get('STATISTICS', 'min_sites_for_common', fallback=20))
    min_sites_eb = int(config.get('STATISTICS', 'min_sites_eb', fallback=30))

    # --- гиперпараметры агрегации и EB ---
    icc_mode = config.get('STATISTICS', 'icc_mode', fallback='estimate')
    fixed_icc = float(config.get('STATISTICS', 'fixed_icc', fallback=0.30))
    huber_c = float(config.get('STATISTICS', 'huber_c', fallback=1.345))
    huber_iters = int(config.get('STATISTICS', 'huber_iters', fallback=3))
    var_floor_pct = float(config.get('STATISTICS', 'var_floor_pct', fallback=10.0))
    eb_d0_floor = float(config.get('STATISTICS', 'eb_d0_floor', fallback=2.0))
    eb_d0_ceil = float(config.get('STATISTICS', 'eb_d0_ceil', fallback=200.0))

    # --- пермутационная валидация (опционально) ---
    run_permutation = config.getboolean('STATISTICS', 'run_permutation', fallback=False)
    n_perm = int(config.get('STATISTICS', 'n_perm', fallback=1000))
    perm_alpha = float(config.get('STATISTICS', 'perm_alpha', fallback=0.05))
    perm_logfc_thresh = float(config.get('STATISTICS', 'perm_logfc_thresh', fallback=1.0))
    perm_exact_threshold = int(config.get('STATISTICS', 'perm_exact_threshold', fallback=5000))
    perm_seed = int(config.get('STATISTICS', 'perm_seed', fallback=42))

    # --- фильтры и алиасы модификаций ---
    exclude_modifications = [s for s in re.split(
        r'\s*,\s*', config.get('PARAMETERS', 'exclude_modifications', fallback='')) if s]
    modification_aliases = config.get('PARAMETERS', 'modification_aliases', fallback='')

    # --- TMT normalization and optional PSM completeness filter ---
    norm_min_fraction_valid = float(config.get('PARAMETERS', 'norm_min_fraction_valid', fallback=0.5))
    norm_use_gis_for_batch = config.getboolean('PARAMETERS', 'norm_use_gis_for_batch', fallback=True)
    norm_target = config.get('PARAMETERS', 'norm_target', fallback='auto')
    norm_max_missing_fraction = float(config.get('PARAMETERS', 'norm_max_missing_fraction', fallback=0.5))
    norm_impute_missing = config.getboolean('PARAMETERS', 'norm_impute_missing', fallback=False)
    norm_impute_low = config.getboolean('PARAMETERS', 'norm_impute_low', fallback=False)

    stats_kwargs = dict(
        min_group_for_stats=min_group_for_stats,
        method=method,
        type_experiment=type_experiment,
        min_sites_eb=min_sites_eb,
        icc_mode=icc_mode,
        fixed_icc=fixed_icc,
        huber_c=huber_c,
        huber_iters=huber_iters,
        var_floor_pct=var_floor_pct,
        eb_d0_floor=eb_d0_floor,
        eb_d0_ceil=eb_d0_ceil,
        run_permutation=run_permutation,
        n_perm=n_perm,
        perm_alpha=perm_alpha,
        perm_logfc_thresh=perm_logfc_thresh,
        perm_exact_threshold=perm_exact_threshold,
        perm_seed=perm_seed,
    )

    logger.info(f'PARAMETERS:\n  - type_of_modifications: {type_of_modification}\n  - name_of_modification: {name_of_modification}\n  - localization_score_threshold: {localization_score_threshold}\n  - mass_tolerance: {mass_tolerance}\n  - fdr_threshold: {fdr_threshold}\n  - type_tmt: {type_tmt}\n  - calculation_pval: {calculation_pval}\n  - min_group_for_stats: {min_group_for_stats}\n  - sorting_pepxml: {sorting_pepxml}\n  - port: {port_n}\n  - min_hits_for_fdr_calc: {min_hits_for_fdr_calc}\n  - default_hyperscore_threshold: {default_hyperscore_threshold}\n  - default_expect_threshold: {default_expect_threshold}')
    logger.info(f'STATS PARAMETERS:\n  - calculating_method: {method}\n  - type_experiment: {type_experiment}\n  - min_sites_mod: {min_sites_mod}\n  - min_ref: {min_ref}\n  - min_obs_per_site: {min_obs_per_site}\n  - min_pairs_for_stoich: {min_pairs_for_stoich}\n  - min_sites_for_common: {min_sites_for_common}\n  - min_sites_eb: {min_sites_eb}\n  - icc_mode: {icc_mode}\n  - fixed_icc: {fixed_icc}\n  - huber_c: {huber_c}\n  - huber_iters: {huber_iters}\n  - var_floor_pct: {var_floor_pct}\n  - eb_d0_floor: {eb_d0_floor}\n  - eb_d0_ceil: {eb_d0_ceil}\n  - exclude_modifications: {exclude_modifications}\n  - modification_aliases: {modification_aliases!r}\n  - run_permutation: {run_permutation}\n  - n_perm: {n_perm}\n  - perm_alpha: {perm_alpha}\n  - perm_logfc_thresh: {perm_logfc_thresh}\n  - perm_exact_threshold: {perm_exact_threshold}\n  - perm_seed: {perm_seed}')
    logger.info(f'NORMALIZATION PARAMETERS:\n  - norm_target: {norm_target}\n  - norm_min_fraction_valid: {norm_min_fraction_valid}\n  - norm_use_gis_for_batch: {norm_use_gis_for_batch}\n  - norm_max_missing_fraction: {norm_max_missing_fraction}\n  - norm_impute_missing: {norm_impute_missing}\n  - norm_impute_low: {norm_impute_low}')

    # Маркер успешно завершенного статистического расчета
    # (per-mod CSV-файлы имеют суффиксы и не подходят как единый кэш-файл)
    stats_done_marker = os.path.join(output_dir, '.stats_complete')
    run_stats = args.recalc_results or not os.path.exists(stats_done_marker)

    if not run_stats:
        logger.info(f"Found statistics completion marker at {stats_done_marker}. "
                    "Skipping recalculation (use --recalc_results to force).")

    if run_stats:
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
            pepxml_res = safe_execute(
                logger, "processing pepXML files",
                process_pepxml_files, cataloque, pepxml_dir,
                mass_tolerance, fdr_threshold, sorting_pepxml, nproc,
                min_hits_for_fdr_calc, default_hyperscore_threshold, default_expect_threshold)
            if pepxml_res is None:
                sys.exit(1)
            all_psms_df, cataloque, results_unique_mods = pepxml_res
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
        unimod_search = pd.read_csv(unimod_search_csv_path)
        if os.path.exists(psm_filtered_path) and os.path.exists(psms_zero_path):
            logger.info(f"PSMs filtered files already exist in {psm_filtered_path}")
        else:
            all_psms_df = pd.read_pickle(all_psms_pickle_path)
            cataloque = pd.read_csv(cataloque_csv_path)
            merge_res = safe_execute(
                logger, "merge cataloque and pepxml dfs",
                spectra_merge, cataloque, all_psms_df, unimod_search
            )
            del cataloque
            del all_psms_df
            if merge_res is None:
                sys.exit(1)
            psm_filtered, psms_zero = merge_res
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
            all_psms_df['mod_name'] = all_psms_df['mod_name'].apply(
                lambda x: 'Glu' if isinstance(x, str) and '-Glu-' in x else x)
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
                for c in ['m/z', 'intensity', 'Unnamed: 0']:
                    if c in annot_df.columns:
                        del annot_df[c]
                annot_df.to_pickle(annotated_pickle_path)
                logger.info(f"The dataframe with annotation is saved in {annotated_pickle_path}")
            else:
                sys.exit(1)

        # Step 7: Statistics
        sort_df_path = os.path.join(output_dir, "sorted_df.pickle")
        norm_df_path = os.path.join(output_dir, "normalization_df.pickle")

        if os.path.exists(norm_df_path) == False:
            logger.info(f"Start normalisation.")
            if 'annot_df' not in globals():
                annot_df = pd.read_pickle(annotated_pickle_path)
            # модификации, исключаемые из анализа (из конфига, без regex)
            for ex_mod in exclude_modifications:
                annot_df = annot_df[~annot_df['Modification'].str.contains(ex_mod, regex=False, na=False)]
            annot_df.reset_index(drop=True, inplace=True)
            stats_df = safe_execute(logger, "normalization", tmt_normalization, annot_df,
                                   intensity_prefix="intensity_",
                                   min_fraction_valid=norm_min_fraction_valid,
                                   use_gis_for_batch=norm_use_gis_for_batch,
                                   normalize_target=norm_target,
                                   type_experiment=type_experiment,
                                   duplicate_spectrum=['spectrum_y'])
            if stats_df is None: sys.exit(1)
            stats_df.to_pickle(norm_df_path)
            if 'annot_df' in globals():
                del annot_df

        if (sorting_pepxml == 'True') and (os.path.exists(norm_df_path)==True) and (os.path.exists(sort_df_path)==False):
            if 'stats_df' not in globals():
                stats_df = pd.read_pickle(norm_df_path)
            logger.info(f"Sorting dataframe was starting calculate.")
            sort_res = safe_execute(logger, "sorting PSM", impute_tmt_psms, stats_df,
                                    max_missing_fraction=norm_max_missing_fraction,
                                    impute_missing=norm_impute_missing,
                                    impute_low=norm_impute_low)
            if sort_res is None: sys.exit(1)
            stats_df, stat, num, delete_indices = sort_res
            stats_df.to_pickle(sort_df_path)
            logger.info(f"TMT intensities were replaced: {stat}")
            logger.info(f"TMT intensities were removed: {num}")

        logger.info(f"Start calculate statistics.")
        if sorting_pepxml == 'False' and os.path.exists(norm_df_path):
            stats_df = pd.read_pickle(norm_df_path)
        elif sorting_pepxml == 'True'  and os.path.exists(sort_df_path):
            stats_df = pd.read_pickle(sort_df_path)
            sys.exit(1)
        else:
            logger.error(f"Normalized data not found at {norm_df_path}.")
            sys.exit(1)

        for c in ['TMT_group1', 'TMT_group2', 'TMT_group3', 'mix_channels']:
            if c in stats_df.columns:
                del stats_df[c]
        stats_df = safe_execute(logger, "annotation of samples", samples_annotation, stats_df, group_df_link)
        if stats_df is None: sys.exit(1)

        # алиасы модификаций из конфига: "pattern=replacement" через запятую (regex)
        for alias in re.split(r'\s*,\s*', modification_aliases):
            if '=' in alias:
                pattern, repl = alias.split('=', 1)
                stats_df['Modification'] = stats_df['Modification'].str.replace(
                    pattern.strip(), repl.strip(), regex=True)

        # --- опциональный rescue позиций через таблицу iPTMnet ---
        if iptmnet_positions_file and os.path.isfile(iptmnet_positions_file):
            logger.info(f"Rescuing PTM positions via {iptmnet_positions_file}")
            cres = pd.read_csv(iptmnet_positions_file)
            cres['position_in_protein'] = cres['position_in_protein'].astype('int')
            cres['rescued_position'] = cres['rescued_position'].astype('int')
            stats_df['position_in_protein'] = stats_df['position_in_protein'].astype('int')
            stats_df = stats_df.merge(cres[['id_prot','position_in_protein','modified_peptide_x','in_iPTM',
                                            'perhapse_in_iPTM', 'perhapse_position','perhapse_ptm_type',
                                            'rescued_position', 'rescued_ptm_type']],
                                      how='left', on = ['id_prot','position_in_protein','modified_peptide_x'])

            stats_df = stats_df.drop_duplicates(subset=['id_prot','position_in_protein','modified_peptide_x','Modification','spectrum_y',
                                                       'rescued_position','rescued_ptm_type'])

            mask = stats_df['rescued_position'].isna()
            stats_df.loc[mask, 'rescued_position'] = stats_df.loc[mask, 'position_in_protein']
            mask = stats_df['rescued_ptm_type'].isna()
            stats_df.loc[mask, 'rescued_ptm_type'] = stats_df.loc[mask, 'Modification']
            stats_df.loc[stats_df['Modification']=='reference', 'rescued_ptm_type'] = 'reference'
            del stats_df['position_in_protein']
            del stats_df['Modification']
            stats_df.rename(columns={'rescued_position':'position_in_protein','rescued_ptm_type':'Modification'}, inplace=True)
        else:
            logger.info("iptmnet_positions_file is not set or not found; using original positions.")
            stats_df = stats_df.drop_duplicates(subset=['id_prot','position_in_protein','modified_peptide_x',
                                                        'Modification','spectrum_y'])

        mods = sorted(m for m in stats_df['Modification'].dropna().unique() if m != 'reference')
        logger.info(f"Modifications to test ({len(mods)}): {mods}")

        ref_mask_all = stats_df['Modification'] == 'reference'
        num_ref_total = int(ref_mask_all.sum())
        channel_cols = [c for c in stats_df.columns if c.endswith('_norm')]
        if len(channel_cols) == 0:
            logger.error("No *_norm channel columns in the normalized data. Exiting.")
            sys.exit(1)

        def median_obs_per_site(sites_df):
            """Median number of non-NA channel observations per PSM row, per site."""
            if sites_df.empty:
                return 0.0
            obs = (sites_df.groupby(['id_prot', 'position_in_protein'])[channel_cols]
                   .apply(lambda g: g.notna().sum().sum() / max(len(g), 1)))
            return float(obs.median()) if len(obs) else 0.0

        def median_mod_ref_pairs(sub, mod):
            """
            Per site: number of channels where BOTH modified and reference rows
            are observed. Used only when min_pairs_for_stoich > 0.

            NOTE: meaningful only if reference rows share (id_prot,
            position_in_protein) keys with modified rows; if references carry
            peptide-level positions, this criterion is uninformative — keep
            min_pairs_for_stoich = 0 in that case.
            """
            pairs = []
            for _, site_df in sub.groupby(['id_prot', 'position_in_protein']):
                mod_rows = site_df[site_df['Modification'] == mod]
                ref_rows = site_df[site_df['Modification'] == 'reference']
                if mod_rows.empty or ref_rows.empty:
                    continue
                both = int((mod_rows[channel_cols].notna().any(axis=0) &
                            ref_rows[channel_cols].notna().any(axis=0)).sum())
                pairs.append(both)
            return float(np.median(pairs)) if pairs else 0.0

        def save_stats_results(tag, res_tuple):
            """Save the 8-tuple returned by statistics(); guards None/empty."""
            (stats_df_res, expr_all, expr_corrected, df_site,
             weights_df, design, noagg, perm_df) = res_tuple
            if stats_df_res is None or stats_df_res.empty:
                logger.warning(f"No testable sites for '{tag}' — nothing to save.")
                return False
            stats_df_res.to_csv(os.path.join(output_dir, f"final_stat_result_{method}_{tag}.csv"))
            expr_corrected.to_csv(os.path.join(output_dir, f"expr_all_corrected_{method}_{tag}.csv"))
            expr_all.to_csv(os.path.join(output_dir, f"expr_all_{method}_{tag}.csv"))
            df_site.to_csv(os.path.join(output_dir, f"final_annot_result_{method}_{tag}.csv"))
            weights_df.to_csv(os.path.join(output_dir, f"weights_df_{method}_{tag}.csv"))
            design.to_csv(os.path.join(output_dir, f"design_{method}_{tag}.csv"))
            if perm_df is not None and not perm_df.empty:
                perm_df.to_csv(os.path.join(output_dir, f"permutation_{method}_{tag}.csv"), index=False)
                for _, pr in perm_df.iterrows():
                    logger.info(f"Permutation validation [{tag}, {pr['contrast']}]: "
                                f"obs_hits={pr['obs_hits']}, null mean={pr['perm_mean']:.1f}, "
                                f"perm_pval={pr['perm_pval']:.4f}, "
                                f"empirical_fdr={pr['empirical_fdr']:.3f}, "
                                f"exact={pr['exact']} (n={pr['n_perm']})")
            logger.info(f"The final statistical result of '{tag}' is saved "
                        f"(final_stat_result_{method}_{tag}.csv)")
            return True

        # ==============================================================
        # Сортировка модификаций:
        #   - "достаточно представленные" моды -> отдельный расчет с EB;
        #   - редкие моды -> общий пул БЕЗ EB-модерации (skip_eb=True):
        #     приор дисперсии на малом числе сайтов ненадежен, а смешивать
        #     распределения разных модов в один EB-приор некорректно, т.к.
        #     у каждого типа модификации свое распределение дисперсий.
        #     В пуле сайты тестируются обычной t-статистикой (WLS), а BH
        #     считается внутри пула как единого семейства гипотез.
        # Точное совпадение по строке 'Modification' (без regex) — имена
        # модов могут содержать спецсимволы ('+', '(', ...).
        # ==============================================================
        common_mod = []

        for mod in mods:
            mod_mask = stats_df['Modification'] == mod
            stats_mod = stats_df[mod_mask | ref_mask_all]
            sites_mod = stats_df[mod_mask]
            num_sites_mod = sites_mod[['id_prot', 'position_in_protein']].drop_duplicates().shape[0]
            med_obs = median_obs_per_site(sites_mod)

            enough = (num_sites_mod >= min_sites_mod) and \
                     (num_ref_total >= min_ref) and \
                     (med_obs >= min_obs_per_site)

            if enough and min_pairs_for_stoich > 0:
                med_pairs = median_mod_ref_pairs(stats_mod, mod)
                enough = med_pairs >= min_pairs_for_stoich
            else:
                med_pairs = np.nan

            if not enough:
                logger.warning(f"Not enough data for {mod}: sites={num_sites_mod}, "
                               f"refs={num_ref_total}, median_obs/site={med_obs:.1f}, "
                               f"median_mod_ref_pairs={med_pairs}")
                common_mod.append(mod)
                continue

            logger.info(f"Running stats for mod {mod}: sites={num_sites_mod}, "
                        f"refs={num_ref_total}, median_obs/site={med_obs:.1f}")
            res = safe_execute(logger, f"calculate statistics ({mod})", statistics,
                               stats_mod.reset_index(drop=True),
                               skip_eb=False, **stats_kwargs)
            if res is None:
                logger.error(f"Statistics failed for {mod}; modification skipped.")
                continue
            save_stats_results(mod, res)

        # --- общий пул редких модификаций (без EB-модерации) ---
        if common_mod:
            logger.info(f"Pooling {len(common_mod)} rare modifications "
                        f"(no EB moderation): {common_mod}")
            pool_mask = stats_df['Modification'].isin(common_mod) | ref_mask_all
            common_df = stats_df[pool_mask].reset_index(drop=True)
            n_common_sites = (common_df.loc[common_df['Modification'] != 'reference',
                                            ['Modification', 'id_prot', 'position_in_protein']]
                              .drop_duplicates().shape[0])

            if n_common_sites >= min_sites_for_common:
                res = safe_execute(logger, "calculate statistics (common pool)", statistics,
                                   common_df,
                                   skip_eb=True, **stats_kwargs)
                if res is not None:
                    save_stats_results('common', res)
            else:
                logger.warning(f"Common pool too small ({n_common_sites} sites < "
                               f"min_sites_for_common={min_sites_for_common}); "
                               "statistical testing skipped for rare modifications.")

        # статистика успешно завершена — ставим маркер кэша
        with open(stats_done_marker, 'w') as fh:
            fh.write(pd.Timestamp.now().isoformat())
        logger.info("The statistical calculation is complete")

        # --- аннотация результатов по базам данных ---
        try:
            logger.info("Start annotation PTMs with db.")
            if 'stats_df' not in globals() or stats_df is None:
                logger.warning("stats_df is not available; skipping db annotation.")
            else:
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
        except Exception:
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
