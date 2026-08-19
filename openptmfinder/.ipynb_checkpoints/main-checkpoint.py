import subprocess
import sys
import numpy as np
import pandas as pd
import argparse
import json
import logging
import os
import re
import configparser
import multiprocessing as mp

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
    tmt_normalization,
    spectra_merge,
    map_mod_position
)
from .sage_preprocessing import prepare_sage_phospho
from .ptm_annotation import annotate_sites_with_iptmnet
from .dbptm_annotation import annotate_sites_with_dbptm
from .report import export_report_data, generate_report
from .signor_annotation import annotate_sites_with_signor, build_signor_network

# --- Default settings ---
DEFAULT_LOG_FILE = 'openptmfinder.log'
DEFAULT_NPROC = 1
DEFAULT_PORT = 10030
DEFAULT_VERBOSITY = 'INFO'


def _grouping_signature(grouping_file: str) -> dict:
    """Signature of the grouping file, used to detect edits between runs.

    The normalized table (normalization_df.pickle) embeds the TMT_group*
    sample annotation, so a grouping-file edit (e.g. adding a third
    experimental group) must invalidate it — otherwise the statistics step
    silently keeps the old group definitions and builds the wrong contrasts.
    """
    sig = {"path": os.path.abspath(str(grouping_file))}
    try:
        st = os.stat(grouping_file)
        sig["mtime"] = st.st_mtime
        sig["size"] = st.st_size
    except (OSError, TypeError):
        sig["mtime"] = None
        sig["size"] = None
    return sig


def _norm_cache_reusable(norm_df_path: str, norm_meta_path: str,
                         grouping_file: str,
                         force_recalc: bool = False) -> bool:
    """Decide whether the cached normalized table matches the CURRENT
    grouping file. False (recompute) when: recalculation is forced, the
    cache or its meta file is missing, the meta is unreadable, or the
    grouping file changed since the cache was written. A cache without meta
    cannot prove it matches and is recomputed once (the meta is then
    written alongside the new cache)."""
    if force_recalc or not os.path.exists(norm_df_path):
        return False
    if not os.path.exists(norm_meta_path):
        return False
    try:
        with open(norm_meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return False
    return meta.get("grouping") == _grouping_signature(grouping_file)


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

    # File handler: always UTF-8 so the log file is readable regardless of
    # the system locale and never raises UnicodeEncodeError.
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8',
                                       errors='replace')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler. On misconfigured locales (e.g. LC_ALL=C) sys.stdout
    # may be ASCII-only and would crash on characters like '±' or '—';
    # errors='replace' degrades such characters to '?' instead of raising.
    try:
        sys.stdout.reconfigure(errors='replace')  # Python >= 3.7
    except (AttributeError, ValueError):
        pass  # not a real text stream (e.g. redirected/replaced stdout)
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
    parser.add_argument('-s','--sage_dir', help='Directory with Sage result folders (each containing results.sage.tsv and tmt.tsv) or a single Sage result folder. Used when search_engine=sage.')

    parser.add_argument('--run_server', action='store_true', help='Start the interactive Dash web app after processing (alias of --interactive)', default=False)
    parser.add_argument('--interactive', action='store_true', help='Start the interactive Dash web app after processing (requires dash and pyarrow)', default=False)
    parser.add_argument('--no_report', action='store_true', help='Do not generate the self-contained HTML report at the end of the run', default=False)
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
        'pepxml_dir': args.pepxml or (cfg('PATHS', 'pepxml_dir', fallback='') or '').split(),
        'sage_dir': args.sage_dir or cfg('PATHS', 'sage_dir', fallback=None),
        'mzml_dir': args.mzml or cfg('PATHS', 'mzml_dir', fallback=None),
        'aa_stat_dir': args.AAstat_dir or cfg('PATHS', 'aa-stat_dir', fallback=None),
        'protein_db': args.protein_db or cfg('PATHS', 'protein_db'),
        'unimod_db': args.unimod_db or cfg('PATHS', 'unimod_db', fallback=None),
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
    sage_dir = paths['sage_dir']

    logger.info(f"Paths:\n  - output_dir: {output_dir}\n  - protein_db: {fasta_file}\n  - unimod_db: {xml_file}\n  - grouping_file: {group_df_link}\n  - aa_stat_dir: {data_dir}\n  - mzml_dir: {mzml_dir}\n  - pepxml_dir: {pepxml_dir}\n  - sage_dir: {sage_dir}")

    search_engine = config.get('PARAMETERS', 'search_engine', fallback='msfragger').strip().lower()
    if search_engine not in {'msfragger', 'sage'}:
        logger.warning(f"Unknown search_engine={search_engine!r}; falling back to 'msfragger'.")
        search_engine = 'msfragger'

    # Check required input files
    required_files = {}
    if search_engine == 'msfragger':
        required_files['interpretation_file'] = os.path.join(data_dir, 'interpretations.json')
    else:
        for name, path in {'grouping_file': group_df_link, 'protein_db': fasta_file, 'sage_dir': sage_dir}.items():
            if not path or not os.path.exists(path):
                logger.error(f"Required Sage input {name} not found at {path}. Exiting.")
                sys.exit(1)
    for name, path in required_files.items():
        if not os.path.isfile(path):
            logger.error(f"Required file {name} not found at {path}. Exiting.")
            sys.exit(1)

    if search_engine == 'msfragger':
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
    min_batches = int(config.get('STATISTICS', 'min_batches', fallback=1))

    # --- database site annotation (before statistics) ---
    db_annotation = config.getboolean('PARAMETERS', 'db_annotation', fallback=True)
    iptmnet_rescue = config.getboolean('PARAMETERS', 'iptmnet_rescue', fallback=False)
    iptmnet_window = int(config.get('PARAMETERS', 'iptmnet_window', fallback=7))
    # dbPTM annotation (optional, independent of iPTMnet, before statistics)
    dbptm_annotation = config.getboolean('PARAMETERS', 'dbptm_annotation', fallback=False)
    dbptm_window = int(config.get('PARAMETERS', 'dbptm_window',
                                  fallback=config.get('PARAMETERS', 'iptmnet_window', fallback=7)))
    # SIGNOR annotation: effect of PTMs on the protein (optional, after statistics)
    signor_annotation = config.getboolean('PARAMETERS', 'signor_annotation', fallback=False)
    signor_organism = int(config.get('PARAMETERS', 'signor_organism', fallback=9606))
    signor_network = config.getboolean('PARAMETERS', 'signor_network', fallback=True)
    signor_max_workers = int(config.get('PARAMETERS', 'signor_max_workers', fallback=6))
    # --- HTML report and interactive mode ---
    html_report = config.getboolean('PARAMETERS', 'html_report', fallback=True) and not args.no_report
    report_plotlyjs = config.get('PARAMETERS', 'report_plotlyjs', fallback='inline').strip().lower()
    report_top_proteins = int(config.get('PARAMETERS', 'report_top_proteins', fallback=10))
    report_alpha = float(config.get('PARAMETERS', 'report_alpha', fallback=0.05))
    report_logfc = float(config.get('PARAMETERS', 'report_logfc', fallback=1.0))
    interactive_cfg = config.getboolean('PARAMETERS', 'interactive', fallback=False)
    sorting_pepxml = config.get('PARAMETERS', 'sorting_pepxml', fallback='False')
    min_hits_for_fdr_calc = int(config.get('PARAMETERS', 'min_hits_for_fdr_calc', fallback=20))
    default_hyperscore_threshold = int(config.get('PARAMETERS', 'default_hyperscore_threshold', fallback=20))
    default_expect_threshold = float(config.get('PARAMETERS', 'default_expect_threshold', fallback=0.05))

    # --- statistics module parameters ([STATISTICS] section) ---
    method = config.get('STATISTICS', 'calculating_method', fallback='aggregate')
    type_experiment = config.get('STATISTICS', 'type_experiment', fallback='whole proteome')
    if search_engine == 'sage' and type_experiment.strip().lower() != 'phospho enrichment':
        logger.warning("search_engine='sage' is implemented for enriched PTM data without reference PSMs; "
                       "using type_experiment='phospho enrichment' for normalization and statistics.")
        type_experiment = 'phospho enrichment'
    min_sites_mod = int(config.get('STATISTICS', 'min_sites_mod', fallback=100))
    min_ref = int(config.get('STATISTICS', 'min_ref', fallback=100))
    min_obs_per_site = float(config.get('STATISTICS', 'min_obs_per_site', fallback=3))
    # 0 = criterion disabled (see the comment in the statistics block)
    min_pairs_for_stoich = float(config.get('STATISTICS', 'min_pairs_for_stoich', fallback=0))
    min_sites_for_common = int(config.get('STATISTICS', 'min_sites_for_common', fallback=20))
    min_sites_eb = int(config.get('STATISTICS', 'min_sites_eb', fallback=30))

    # --- aggregation and EB hyperparameters ---
    icc_mode = config.get('STATISTICS', 'icc_mode', fallback='estimate')
    fixed_icc = float(config.get('STATISTICS', 'fixed_icc', fallback=0.30))
    huber_c = float(config.get('STATISTICS', 'huber_c', fallback=1.345))
    huber_iters = int(config.get('STATISTICS', 'huber_iters', fallback=3))
    var_floor_pct = float(config.get('STATISTICS', 'var_floor_pct', fallback=10.0))
    eb_d0_floor = float(config.get('STATISTICS', 'eb_d0_floor', fallback=2.0))
    eb_d0_ceil = float(config.get('STATISTICS', 'eb_d0_ceil', fallback=200.0))

    # --- permutation validation (optional) ---
    run_permutation = config.getboolean('STATISTICS', 'run_permutation', fallback=False)
    n_perm = int(config.get('STATISTICS', 'n_perm', fallback=1000))
    perm_alpha = float(config.get('STATISTICS', 'perm_alpha', fallback=0.05))
    perm_logfc_thresh = float(config.get('STATISTICS', 'perm_logfc_thresh', fallback=1.0))
    perm_exact_threshold = int(config.get('STATISTICS', 'perm_exact_threshold', fallback=5000))
    perm_seed = int(config.get('STATISTICS', 'perm_seed', fallback=42))
    perm_calib_perms = int(config.get('STATISTICS', 'perm_calib_perms', fallback=20))
    run_spikein = config.getboolean('STATISTICS', 'run_spikein', fallback=False)
    spike_effects = tuple(float(s) for s in re.split(
        r'\s*,\s*', config.get('STATISTICS', 'spike_effects',
                                fallback='0.5,0.75,1,1.5,2')) if s)
    spike_fraction = float(config.get('STATISTICS', 'spike_fraction', fallback=0.05))
    spike_reps = int(config.get('STATISTICS', 'spike_reps', fallback=3))

    # --- modification filters and aliases ---
    exclude_modifications = [s for s in re.split(
        r'\s*,\s*', config.get('PARAMETERS', 'exclude_modifications', fallback='')) if s]
    modification_aliases = config.get('PARAMETERS', 'modification_aliases', fallback='')

    # --- TMT normalization ---
    norm_min_fraction_valid = float(config.get('PARAMETERS', 'norm_min_fraction_valid', fallback=0.5))
    norm_use_gis_for_batch = config.getboolean('PARAMETERS', 'norm_use_gis_for_batch', fallback=True)
    norm_target = config.get('PARAMETERS', 'norm_target', fallback='auto')

    # --- Sage / enriched-PTM preprocessing (used when search_engine='sage') ---
    sage_results_filename = config.get('PARAMETERS', 'sage_results_filename', fallback='results.sage.tsv')
    sage_tmt_filename = config.get('PARAMETERS', 'sage_tmt_filename', fallback='tmt.tsv')
    sage_intensity_prefix = config.get('PARAMETERS', 'sage_intensity_prefix', fallback='tmt_')
    sage_fdr_method = config.get('PARAMETERS', 'sage_fdr_method', fallback='spectrum_peptide_q')
    sage_score_column = config.get('PARAMETERS', 'sage_score_column', fallback='sage_discriminant_score')
    sage_decoy_regex = config.get('PARAMETERS', 'sage_decoy_regex', fallback=r'DECOY|rev_')
    sage_mod_keep_regex = config.get('PARAMETERS', 'sage_mod_keep_regex', fallback=r'\+79\.')
    sage_mod_name = config.get('PARAMETERS', 'sage_mod_name', fallback='Phospho')
    sage_require_mod = config.getboolean('PARAMETERS', 'sage_require_mod', fallback=True)
    sage_map_all_proteins = config.getboolean('PARAMETERS', 'sage_map_all_proteins', fallback=False)
    norm_intensity_prefix = sage_intensity_prefix if search_engine == 'sage' else 'intensity_'

    stats_kwargs = dict(
        min_group_for_stats=min_group_for_stats,
        min_batches=min_batches,
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
        perm_calib_perms=perm_calib_perms,
        run_spikein=run_spikein,
        spike_effects=spike_effects,
        spike_fraction=spike_fraction,
        spike_reps=spike_reps,
    )

    logger.info(f'PARAMETERS:\n  - type_of_modifications: {type_of_modification}\n  - name_of_modification: {name_of_modification}\n  - localization_score_threshold: {localization_score_threshold}\n  - mass_tolerance: {mass_tolerance}\n  - fdr_threshold: {fdr_threshold}\n  - type_tmt: {type_tmt}\n  - calculation_pval: {calculation_pval}\n  - min_group_for_stats: {min_group_for_stats}\n  - sorting_pepxml: {sorting_pepxml}\n  - port: {port_n}\n  - min_hits_for_fdr_calc: {min_hits_for_fdr_calc}\n  - default_hyperscore_threshold: {default_hyperscore_threshold}\n  - default_expect_threshold: {default_expect_threshold}')
    logger.info(f'STATS PARAMETERS:\n  - calculating_method: {method}\n  - type_experiment: {type_experiment}\n  - min_batches: {min_batches}\n  - min_sites_mod: {min_sites_mod}\n  - min_ref: {min_ref}\n  - min_obs_per_site: {min_obs_per_site}\n  - min_pairs_for_stoich: {min_pairs_for_stoich}\n  - min_sites_for_common: {min_sites_for_common}\n  - min_sites_eb: {min_sites_eb}\n  - icc_mode: {icc_mode}\n  - fixed_icc: {fixed_icc}\n  - huber_c: {huber_c}\n  - huber_iters: {huber_iters}\n  - var_floor_pct: {var_floor_pct}\n  - eb_d0_floor: {eb_d0_floor}\n  - eb_d0_ceil: {eb_d0_ceil}\n  - exclude_modifications: {exclude_modifications}\n  - modification_aliases: {modification_aliases!r}\n  - run_permutation: {run_permutation}\n  - n_perm: {n_perm}\n  - perm_alpha: {perm_alpha}\n  - perm_logfc_thresh: {perm_logfc_thresh}\n  - perm_exact_threshold: {perm_exact_threshold}\n  - perm_seed: {perm_seed}\n  - perm_calib_perms: {perm_calib_perms}\n  - run_spikein: {run_spikein}\n  - spike_effects: {spike_effects}\n  - spike_fraction: {spike_fraction}\n  - spike_reps: {spike_reps}')
    logger.info(f'NORMALIZATION PARAMETERS:\n  - norm_target: {norm_target}\n  - norm_min_fraction_valid: {norm_min_fraction_valid}\n  - norm_use_gis_for_batch: {norm_use_gis_for_batch}')
    logger.info(f'INPUT/PREPROCESSING:\n  - search_engine: {search_engine}\n  - sage_results_filename: {sage_results_filename}\n  - sage_tmt_filename: {sage_tmt_filename}\n  - sage_intensity_prefix: {sage_intensity_prefix}\n  - sage_fdr_method: {sage_fdr_method}\n  - sage_mod_keep_regex: {sage_mod_keep_regex}\n  - sage_mod_name: {sage_mod_name}\n  - sage_require_mod: {sage_require_mod}\n  - sage_map_all_proteins: {sage_map_all_proteins}')
    logger.info(f'DB ANNOTATION:\n  - db_annotation: {db_annotation}\n  - iptmnet_rescue: {iptmnet_rescue}\n  - iptmnet_window: {iptmnet_window}\n  - dbptm_annotation: {dbptm_annotation}\n  - dbptm_window: {dbptm_window}\n  - signor_annotation: {signor_annotation}\n  - signor_organism: {signor_organism}\n  - signor_network: {signor_network}\n  - signor_max_workers: {signor_max_workers}')

    # Marker of a successfully completed statistical calculation
    # (per-mod CSV files carry suffixes and cannot serve as a single cache file)
    stats_done_marker = os.path.join(output_dir, '.stats_complete')
    run_stats = args.recalc_results or not os.path.exists(stats_done_marker)

    # A grouping-file edit must also invalidate the cached statistics even
    # without --recalc_results: the TMT_group* sample annotation (and thus
    # the set of contrasts) is baked into normalization_df.pickle.
    if not run_stats:
        norm_meta_probe = os.path.join(output_dir,
                                       "normalization_df.pickle.meta.json")
        if os.path.isfile(norm_meta_probe):
            try:
                with open(norm_meta_probe, "r", encoding="utf-8") as fh:
                    _norm_meta = json.load(fh)
                if _norm_meta.get("grouping") != _grouping_signature(group_df_link):
                    logger.info("Grouping file changed since the last run — "
                                "forcing recalculation of statistics.")
                    run_stats = True
            except Exception:
                pass

    if not run_stats:
        logger.info(f"Found statistics completion marker at {stats_done_marker}. "
                    "Skipping recalculation (use --recalc_results to force).")

    if run_stats:
        logger.info("Starting full recalculation of results...")

        annotated_pickle_path = os.path.join(output_dir, 'annotated_df.pickle')
        if search_engine == 'sage':
            logger.info("search_engine='sage': using Sage/enriched-PTM preprocessing; AA_stat, pepXML and mzML steps are skipped.")
            annot_df = safe_execute(logger, "Sage preprocessing", prepare_sage_phospho,
                                    sage_dir=sage_dir, grouping_file=group_df_link, fasta_file=fasta_file,
                                    fdr_threshold=fdr_threshold, results_filename=sage_results_filename,
                                    tmt_filename=sage_tmt_filename, intensity_prefix=sage_intensity_prefix,
                                    fdr_method=sage_fdr_method, score_column=sage_score_column,
                                    decoy_regex=sage_decoy_regex, mod_keep_regex=sage_mod_keep_regex,
                                    mod_name=sage_mod_name, require_mod=sage_require_mod,
                                    map_all_proteins=sage_map_all_proteins)
            if annot_df is None or annot_df.empty:
                logger.error('Sage preprocessing produced no annotated PSMs. Exiting.')
                sys.exit(1)
            annot_df.to_pickle(annotated_pickle_path)
            logger.info(f'The dataframe with Sage annotation is saved in {annotated_pickle_path}')
        else:
            # Step 1: unimod_df
            unimod_csv_path = os.path.join(output_dir, 'unimod.csv')
            unimod_df = safe_execute(logger, "Processing AA_stat results", create_unimod_dataframe, interpretation_file, xml_file)
            if unimod_df is not None:
                unimod_df.to_csv(unimod_csv_path, index=False)
                logger.info(f"Unimod shift annotation saved in {unimod_csv_path}")
            else:
                sys.exit(1)

            # Step 2: catalogue
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

            # Step 3: all_psms_df
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
        norm_df_path = os.path.join(output_dir, "normalization_df.pickle")
        norm_meta_path = norm_df_path + ".meta.json"

        # The cached normalized table embeds the TMT_group* annotation of the
        # grouping file. Reuse it only when the grouping file is unchanged AND
        # recalculation was not forced — otherwise a grouping-file edit (e.g.
        # a third experimental group added) would silently produce the old,
        # incomplete set of contrasts.
        if not _norm_cache_reusable(norm_df_path, norm_meta_path, group_df_link,
                                    force_recalc=args.recalc_results):
            if os.path.exists(norm_df_path):
                logger.info("Grouping file changed or recalculation forced — "
                            "re-running sample annotation and normalization.")
            logger.info(f"Start normalisation.")
            annot_df = pd.read_pickle(annotated_pickle_path)
            # modifications excluded from the analysis (from config, plain
            # substring match, no regex)
            if 'Modification' in annot_df.columns:
                for ex_mod in exclude_modifications:
                    annot_df = annot_df[~annot_df['Modification'].str.contains(ex_mod, regex=False, na=False)]
            annot_df.reset_index(drop=True, inplace=True)

            # Sample annotation must run BEFORE normalization: it creates the
            # 'batch' column that tmt_normalization uses for within-plex channel
            # centering and cross-batch alignment (without it each file would be
            # treated as its own plex).
            if search_engine == 'sage':
                # Sage preprocessing is already sample-annotated; keep TMT_group*/mix_channels.
                if 'batch' not in annot_df.columns:
                    logger.error("Sage preprocessing did not produce a 'batch' column. Exiting.")
                    sys.exit(1)
                annot_df['batch'] = annot_df['batch'].astype(int)
            else:
                annot_df = safe_execute(logger, "annotation of samples", samples_annotation,
                                        annot_df, group_df_link)
                if annot_df is None: sys.exit(1)

            stats_df = safe_execute(logger, "normalization", tmt_normalization, annot_df,
                                   intensity_prefix=norm_intensity_prefix,
                                   min_fraction_valid=norm_min_fraction_valid,
                                   use_gis_for_batch=norm_use_gis_for_batch,
                                   normalize_target=norm_target,
                                   type_experiment=type_experiment,
                                   duplicate_spectrum=['spectrum_y'])
            if stats_df is None: sys.exit(1)
            stats_df.to_pickle(norm_df_path)
            with open(norm_meta_path, "w", encoding="utf-8") as fh:
                json.dump({"grouping": _grouping_signature(group_df_link)}, fh)
            del annot_df
            del stats_df

        if sorting_pepxml == 'True':
            logger.error("sorting_pepxml='True' is not supported by the current "
                         "statistics pipeline (the WLS/limma model handles "
                         "per-channel missingness directly). Set sorting_pepxml=False.")
            sys.exit(1)

        logger.info(f"Start calculate statistics.")
        if os.path.exists(norm_df_path):
            stats_df = pd.read_pickle(norm_df_path)
        else:
            logger.error(f"Normalized data not found at {norm_df_path}.")
            sys.exit(1)

        # modification aliases from the config: comma-separated
        # "regex_pattern=replacement" pairs
        if 'Modification' in stats_df.columns and modification_aliases:
            for alias in re.split(r'\s*,\s*', modification_aliases):
                if '=' in alias:
                    pattern, repl = alias.split('=', 1)
                    stats_df['Modification'] = stats_df['Modification'].str.replace(
                        pattern.strip(), repl.strip(), regex=True)

        is_phospho_experiment = type_experiment.strip().lower() == 'phospho enrichment'

        # --- shared UniProt accession-mapping cache -----------------------
        # Obsolete/secondary UniProt accessions (unknown to iPTMnet/dbPTM/
        # SIGNOR) are resolved ONCE through the UniProt ID-mapping API by
        # whichever database annotation meets them first and are stored in
        # output_dir/uniprot_idmap.csv; all later annotations then query
        # their databases with the current accession directly and relabel
        # the results back to the original ids.
        idmap_path = os.path.join(output_dir, 'uniprot_idmap.csv')
        if args.recalc_results and os.path.isfile(idmap_path):
            os.remove(idmap_path)  # force re-resolution of obsolete accessions

        # --- site annotation via iPTMnet (runs BEFORE statistics) ---
        # db_annotation=True: unique sites are annotated against iPTMnet and
        # the FASTA; the result is cached in output_dir/iptmnet_positions.csv
        # and reused on reruns (reset with --recalc_results).
        # iptmnet_rescue=True: rescued positions replace the original
        # position_in_protein/Modification; False — the annotation columns
        # (in_iPTM, perhapse_*, rescued_*) are added for information only.
        if (not is_phospho_experiment) and db_annotation:
            annotation_path = os.path.join(output_dir, 'iptmnet_positions.csv')
            if os.path.isfile(annotation_path) and not args.recalc_results:
                logger.info(f"Loading cached iPTMnet annotation from {annotation_path}")
                cres = pd.read_csv(annotation_path)
            else:
                cres = safe_execute(logger, "iPTMnet site annotation", annotate_sites_with_iptmnet,
                                    stats_df, fasta_file=fasta_file, window=iptmnet_window,
                                    max_workers=nproc, output_dir=output_dir)
                if cres is None or cres.empty:
                    logger.warning("iPTMnet annotation produced no results; using original positions.")
                    cres = None
                else:
                    cres.to_csv(annotation_path, index=False)
                    logger.info(f"iPTMnet annotation is saved in {annotation_path}")

            if cres is not None:
                cres['position_in_protein'] = cres['position_in_protein'].astype('int')
                stats_df['position_in_protein'] = stats_df['position_in_protein'].astype('int')
                stats_df = stats_df.merge(cres[['id_prot', 'position_in_protein', 'modified_peptide_x',
                                                'in_iPTM', 'perhapse_in_iPTM', 'perhapse_position',
                                                'perhapse_ptm_type', 'rescued_position', 'rescued_ptm_type']],
                                          how='left',
                                          on=['id_prot', 'position_in_protein', 'modified_peptide_x'])

                stats_df = stats_df.drop_duplicates(
                    subset=['id_prot', 'position_in_protein', 'modified_peptide_x', 'Modification',
                            'spectrum_y', 'rescued_position', 'rescued_ptm_type'])

                if iptmnet_rescue:
                    logger.info("iptmnet_rescue=True: rescued positions replace the original ones.")
                    mask = stats_df['rescued_position'].isna()
                    stats_df.loc[mask, 'rescued_position'] = stats_df.loc[mask, 'position_in_protein']
                    mask = stats_df['rescued_ptm_type'].isna()
                    stats_df.loc[mask, 'rescued_ptm_type'] = stats_df.loc[mask, 'Modification']
                    stats_df.loc[stats_df['Modification'] == 'reference', 'rescued_ptm_type'] = 'reference'
                    del stats_df['position_in_protein']
                    del stats_df['Modification']
                    stats_df.rename(columns={'rescued_position': 'position_in_protein',
                                             'rescued_ptm_type': 'Modification'}, inplace=True)
                    stats_df['position_in_protein'] = stats_df['position_in_protein'].astype('int')
            else:
                stats_df = stats_df.drop_duplicates(
                    subset=['id_prot', 'position_in_protein', 'modified_peptide_x',
                            'Modification', 'spectrum_y'])
        elif not is_phospho_experiment:
            logger.info("db_annotation=False; using original positions.")
            stats_df = stats_df.drop_duplicates(subset=['id_prot','position_in_protein','modified_peptide_x',
                                                        'Modification','spectrum_y'])
        else:
            logger.info("Phospho-enrichment branch: skipping iPTMnet rescue and reference-based filtering.")
            if 'Modification' not in stats_df.columns:
                stats_df['Modification'] = sage_mod_name if search_engine == 'sage' else 'Phospho'
            if 'peptide_clean' not in stats_df.columns and 'peptide' in stats_df.columns:
                stats_df['peptide_clean'] = stats_df['peptide'].astype(str).str.replace(r'[^A-Z]', '', regex=True)
            if 'isotope_error' not in stats_df.columns:
                stats_df['isotope_error'] = 0
            stats_df = stats_df.drop_duplicates(
                    subset=['file_name', 'scannr','protein', 'position_in_protein','peptide'])

        # --- site annotation via dbPTM (optional, before statistics) ---
        # dbptm_annotation=True: unique sites are additionally annotated
        # against dbPTM (per-protein info.php page requests, without
        # downloading the whole database). The columns are purely
        # informational (in_dbPTM, dbptm_ptm_types, dbptm_pmids, perhapse_*):
        # original and rescued positions are left unchanged.
        # The result is cached in output_dir/dbptm_positions.csv
        # (reset with --recalc_results). Independent of db_annotation (iPTMnet).
        if (not is_phospho_experiment) and dbptm_annotation:
            dbptm_path = os.path.join(output_dir, 'dbptm_positions.csv')
            if os.path.isfile(dbptm_path) and not args.recalc_results:
                logger.info(f"Loading cached dbPTM annotation from {dbptm_path}")
                # dtype='string': the PMID column must not become a float
                dres = pd.read_csv(dbptm_path,
                                   dtype={'dbptm_ptm_types': 'string',
                                          'dbptm_pmids': 'string'})
            else:
                dres = safe_execute(logger, "dbPTM site annotation", annotate_sites_with_dbptm,
                                    stats_df, fasta_file=fasta_file, window=dbptm_window,
                                    max_workers=nproc, output_dir=output_dir)
                if dres is None or dres.empty:
                    logger.warning("dbPTM annotation produced no results.")
                    dres = None
                else:
                    dres.to_csv(dbptm_path, index=False)
                    logger.info(f"dbPTM annotation is saved in {dbptm_path}")

            if dres is not None:
                # Int64 (nullable) in case of NaN positions in stats_df
                dres['position_in_protein'] = pd.to_numeric(
                    dres['position_in_protein'], errors='coerce').astype('Int64')
                stats_df['position_in_protein'] = pd.to_numeric(
                    stats_df['position_in_protein'], errors='coerce').astype('Int64')
                stats_df = stats_df.merge(
                    dres[['id_prot', 'position_in_protein', 'modified_peptide_x',
                          'in_dbPTM', 'dbptm_ptm_types', 'dbptm_pmids',
                          'perhapse_in_dbPTM', 'perhapse_position_dbptm',
                          'perhapse_ptm_type_dbptm']],
                    how='left',
                    on=['id_prot', 'position_in_protein', 'modified_peptide_x'])

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
            """Save the 10-tuple returned by statistics(); guards None/empty.

            The design matrix is intentionally NOT saved: it is fully
            determined by the grouping file and the contrast definitions, and
            only the last contrast's design would be kept anyway.
            """
            (stats_df_res, expr_all, expr_corrected, df_site,
             weights_df, design, noagg, perm_df,
             perm_pvals_df, spikein_df) = res_tuple
            if stats_df_res is None or stats_df_res.empty:
                logger.warning(f"No testable sites for '{tag}' - nothing to save.")
                return False
            stats_df_res.to_csv(os.path.join(output_dir, f"final_stat_result_{method}_{tag}.csv"),
                                index=False)
            expr_corrected.to_csv(os.path.join(output_dir, f"expr_all_corrected_{method}_{tag}.csv"))
            expr_all.to_csv(os.path.join(output_dir, f"expr_all_{method}_{tag}.csv"))
            df_site.to_csv(os.path.join(output_dir, f"final_annot_result_{method}_{tag}.csv"),
                           index=False)
            weights_df.to_csv(os.path.join(output_dir, f"weights_df_{method}_{tag}.csv"))
            if noagg is not None and not noagg.empty:
                noagg.to_csv(os.path.join(output_dir, f"expr_noagg_{method}_{tag}.csv"))
            if perm_df is not None and not perm_df.empty:
                perm_df.to_csv(os.path.join(output_dir, f"permutation_{method}_{tag}.csv"), index=False)
                for _, pr in perm_df.iterrows():
                    logger.info(f"Permutation validation [{tag}, {pr['contrast']}]: "
                                f"obs_hits={pr['obs_hits']}, null mean={pr['perm_mean']:.1f}, "
                                f"perm_pval={pr['perm_pval']:.4f}, "
                                f"empirical_fdr={pr['empirical_fdr']:.3f}, "
                                f"exact={pr['exact']} (n={pr['n_perm']})")
            if perm_pvals_df is not None and not perm_pvals_df.empty:
                perm_pvals_df.to_csv(os.path.join(
                    output_dir, f"permutation_pvalues_{method}_{tag}.csv"), index=False)
                logger.info(f"Null p-value calibration [{tag}] saved "
                            f"(permutation_pvalues_{method}_{tag}.csv, "
                            f"{len(perm_pvals_df)} rows)")
            if spikein_df is not None and not spikein_df.empty:
                spikein_df.to_csv(os.path.join(output_dir, f"spikein_{method}_{tag}.csv"),
                                  index=False)
                for (con, eff), grp in spikein_df.groupby(['contrast', 'effect_size']):
                    logger.info(f"Spike-in [{tag}, {con}]: |log2FC|={eff:g} -> "
                                f"TPR={grp['tpr'].mean():.2f}, "
                                f"FP={grp['n_false_pos'].mean():.1f}")
            logger.info(f"The final statistical result of '{tag}' is saved "
                        f"(final_stat_result_{method}_{tag}.csv)")
            return True

        if is_phospho_experiment:
            tag = sage_mod_name if search_engine == 'sage' else 'phospho'
            logger.info(f"Running enrichment statistics without reference-based modification sorting (tag='{tag}').")
            res = safe_execute(logger, 'calculate statistics (phospho enrichment)', statistics,
                               stats_df.reset_index(drop=True), skip_eb=False, **stats_kwargs)
            if res is None:
                sys.exit(1)
            save_stats_results(tag, res)
        else:
            # ==============================================================
            # Modification sorting:
            #   - "sufficiently represented" mods -> separate run with EB;
            #   - rare mods -> a common pool WITHOUT EB moderation
            #     (skip_eb=True): the variance prior is unreliable on a
            #     small number of sites, and pooling the variance
            #     distributions of different mods into one EB prior is
            #     incorrect, since each modification type has its own
            #     variance distribution.
            #     In the pool, sites are tested with an ordinary
            #     t-statistic (WLS), and BH is computed within the pool as
            #     a single family of hypotheses.
            # Exact string match on 'Modification' (no regex) — mod names
            # may contain special characters ('+', '(', ...).
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

            # --- common pool of rare modifications (no EB moderation) ---
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

        # statistics finished successfully — write the cache marker
        with open(stats_done_marker, 'w') as fh:
            fh.write(pd.Timestamp.now().isoformat())
        logger.info("The statistical calculation is complete")

        # --- annotation of results against the databases ---
        # iPTMnet/dbPTM site annotation was already done before the
        # statistics (columns in_iPTM, perhapse_*, in_dbPTM, ... in
        # stats_df); SIGNOR annotation is performed here (after the
        # statistics, which it does not affect) and is likewise added to
        # stats_df; then the final file is saved.
        if (db_annotation or dbptm_annotation or signor_annotation) and not is_phospho_experiment:
            try:
                logger.info("Saving db annotation results.")
                if stats_df is None or 'id_prot' not in stats_df.columns:
                    logger.warning("stats_df is not available or has no id_prot column; skipping db annotation.")
                else:
                    # --- SIGNOR: PTM effects on proteins (optional) ---
                    # getData.php?id=<UniProtAC> requests per protein in the
                    # list (without downloading the whole database), typed
                    # matching by PTM class and position.
                    # Cache: signor_sites.csv + signor_edges.csv
                    # (reset with --recalc_results).
                    if signor_annotation:
                        signor_sites_path = os.path.join(output_dir, 'signor_sites.csv')
                        signor_edges_path = os.path.join(output_dir, 'signor_edges.csv')
                        sres = None
                        if (os.path.isfile(signor_sites_path)
                                and os.path.isfile(signor_edges_path)
                                and not args.recalc_results):
                            logger.info(f"Loading cached SIGNOR annotation from {signor_sites_path}")
                            try:
                                sres = (pd.read_csv(signor_sites_path,
                                                    dtype={'signor_regulations': 'string',
                                                           'signor_regulators': 'string',
                                                           'signor_pmids': 'string'}),
                                        pd.read_csv(signor_edges_path))
                            except (OSError, pd.errors.ParserError) as e:
                                logger.warning(f"Could not read cached SIGNOR files ({e}); re-fetching.")
                                sres = None
                        if sres is None:
                            sres = safe_execute(logger, "SIGNOR site annotation",
                                                annotate_sites_with_signor, stats_df,
                                                organism=signor_organism,
                                                max_workers=signor_max_workers,
                                                output_dir=output_dir)
                            if sres is None or sres[0] is None or sres[0].empty:
                                logger.warning("SIGNOR annotation produced no results.")
                                sres = None
                            else:
                                sres[0].to_csv(signor_sites_path, index=False)
                                sres[1].to_csv(signor_edges_path, index=False)
                                logger.info(f"SIGNOR annotation is saved in {signor_sites_path} "
                                            f"(edge table: {signor_edges_path})")

                        if sres is not None:
                            signor_sites_df, signor_edges_df = sres
                            # Int64 (nullable) in case of NaN positions in stats_df
                            signor_sites_df['position_in_protein'] = pd.to_numeric(
                                signor_sites_df['position_in_protein'], errors='coerce').astype('Int64')
                            stats_df['position_in_protein'] = pd.to_numeric(
                                stats_df['position_in_protein'], errors='coerce').astype('Int64')
                            stats_df = stats_df.merge(
                                signor_sites_df[['id_prot', 'position_in_protein',
                                                 'modified_peptide_x', 'in_SIGNOR',
                                                 'signor_evidence', 'signor_effect_on_protein',
                                                 'signor_regulations', 'signor_regulators',
                                                 'signor_pmids']],
                                how='left',
                                on=['id_prot', 'position_in_protein', 'modified_peptide_x'])

                            # --- interpretable "PTM effect on protein" network ---
                            if signor_network:
                                network_path = os.path.join(output_dir, 'signor_network.html')
                                safe_execute(logger, "SIGNOR network", build_signor_network,
                                             signor_edges_df, signor_sites_df, network_path,
                                             output_dir=output_dir,
                                             alpha=report_alpha, logfc_thr=report_logfc)

                    if any(c in stats_df.columns for c in ('in_iPTM', 'in_dbPTM', 'in_SIGNOR')):
                        stats_df_path1 = os.path.join(output_dir, "final_stat_result_with_dbs.csv")
                        stats_df.to_csv(stats_df_path1, index=False)
                        logger.info(f"The final result with db annotation (iPTMnet/dbPTM/SIGNOR) is saved in {stats_df_path1}")
                    else:
                        logger.warning("db annotation columns are missing in stats_df; "
                                       "final_stat_result_with_dbs.csv was not created.")

            except KeyError as e:
                logger.error(f"KeyError during db annotation: {e}")
            except ValueError as e:
                logger.error(f"ValueError during db annotation: {e}")
            except Exception:
                logger.exception("Unexpected error during db annotation.")


    # --- data layer (parquet), HTML report and interactive mode -----------
    # Runs both on a fresh calculation and on a rerun with a statistics
    # cache: the parquet layer (<output_dir>/report_data/) is the shared
    # data source for the report and the Dash app (with CSV/pickle
    # fallback).
    wants_interactive = args.interactive or interactive_cfg or args.run_server
    if html_report or wants_interactive:
        safe_execute(logger, "export report data layer (parquet)",
                     export_report_data, output_dir)
    if html_report:
        safe_execute(logger, "HTML report generation", generate_report,
                     output_dir, fasta_file=fasta_file, plotlyjs=report_plotlyjs,
                     top_proteins=report_top_proteins,
                     alpha=report_alpha, logfc_thr=report_logfc)

    logger.info("The program is complete.")

    if wants_interactive:
        logger.info("Launching Dash interactive server (--interactive)...")
        os.environ['OUTPUT_DIR'] = output_dir
        os.environ['port_n'] = str(port_n)
        os.environ['fasta'] = fasta_file
        subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'dash_server.py')])
    else:
        logger.info("Interactive mode was not requested (--interactive / interactive=True). Shutting down.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
