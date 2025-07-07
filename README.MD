

OpenPtmFinder
==============
PTM Annotation Tool Based on Open strategy search in proteomics

# Overview

**OpenPtmFinder** is a tool for shotgun labled proteomics for annotated peptide post-translation modifications, 
in the results of open strategy analysis.
**A.** The program searches for peptides with a given modification type in all spectra based on mass shifts from the results of the open AA_stat search. 
**B.** The program selects post-translational modifications based on the Unimod database and calculates the position of the modification on the protein.
**C.** The program calculates the differential expression of the modifications found between the two groups.
**D.** At the output, you get a local web application with calculated statistics and graphs.


The program includes search and normalization of different TMT tags, group-specific filtering of peptides based on FDR, and validation of modifications based on DeepLC predictions.


## How to cite

A preprint of this article is available on biorXiv: ...


## Installation

OpenPtmFinder requires Python 3.7 or newer. You can create microenvironments for the program to run.
```
pyenv virtualenv 3.10 OpenPtmFinder
```
Install from PyPI (all platforms):

```
pip install OpenPtmFinder
```

Alternatively, you can install directly from GitHub:

```
pip install git+https://github.com/Anchovy-bio/OpenPtmFinder
```


## Quickstart

### Command line

For OpenPtmFinder to work, the results of an open search after running the MSFragger and AA_stat programs are required.
Input data includes directories of raw .mzML, .csv files from AA_stat, processed pepXML files. 
In its simplest form, the program searches for the above files in the working directory, and uses the default parameters from the standard config file, the protein base from Uniprot from 2022, and the Unimod base from 2025.
You can go to the directory with the above files and run OpenPtmFinder with the following command with the run-server flag to create a web page of results.
```
OpenPtmFinder --run-server
```
To calculate differential expression, you must specify a file with annotation of tags and images for two groups in the config file:
```
OpenPtmFinder --config ./config.ini --run-server
```
or command line options:
```
OpenPtmFinder --grouping_file ./groups.csv --output_dir ./results --run-server
```


### Open search results and spectrum files

AA_stat deals with open search results in pepXML or CSV formats and is compatible with the search results obtained using
most existing search engines.
By default, it is recommended to use MSFragger search engine,
available from [Nesvizhskii lab](http://www.nesvilab.org/software.html) website.
For details of its operation, see
[MSFragger User manual](https://media.nature.com/original/nature-assets/nmeth/journal/v14/n5/extref/nmeth.4256-S2.pdf).

Most new features of AA_stat require that you also provide spectrum files. AA_stat supports MGF and mzML formats.
Note that you must provide the same files that are used for open search. **Note: If you run MSFragger on RAW files,
you can convert RAW files to mzML and specify those for AA_stat.**


### Examples

An example of the open search parameters file can be found in the repository [here](OpenPtmFinder/config.ini).

Example of MSFragger usage:

```
java -Xmx8G -jar MSFragger.jar open_search.params HeLa_run1.mzML HeLa_run2.mzML
```

Example of using AA_stat:

```
AA_stat --pepxml *.pepXML --mzml *.mzML
```

# User manual

### Command line options

```
usage: OpenPtmFinder [-h] [--config CONFIG] [--output_dir OUTPUT_DIR] [--pepxml_dir PEPXML_DIR [PEPXML_DIR ...]] [--mzml_dir MZML_DIR] [--AAstat_dir AASTAT_DIR]
                     [--protein_db PROTEIN_DB] [--unimod_db UNIMOD_DB] [--grouping_file GROUPING_FILE] [--run-server] [-n PROCESSES]
                     [--verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

PTM Annotation Tool Based on Open strategy search

options:
  -h, --help            show this help message and exit
  --config CONFIG       .ini file with parameters. If there is no file, OpenPtmFinder uses default one.
  --output_dir OUTPUT_DIR
                        Directory to store the results. Default value is current directory.
  --pepxml_dir PEPXML_DIR [PEPXML_DIR ...]
                        Directory with pepxml search files or separate files. Default value is current directory.
  --mzml_dir MZML_DIR   Directory with mzml search files. Default value is current directory.
  --AAstat_dir AASTAT_DIR
                        Directory with AA_stat search results (.csv and interpretations.json). Default value is current directory.
  --protein_db PROTEIN_DB
                        Directory with .fasta file with proteins. If there is no file, OpenPtmFinder uses default one.
  --unimod_db UNIMOD_DB
                        Directory with .xml UNIMOD database. 
                        An example can be found at https://github.com/Anchovy-bio/OpenPtmFinder/tree/main/data/unimod
  --grouping_file GROUPING_FILE
                        Directory with annotation file of samples by TMT groups. 
                        An example can be found at https://github.com/Anchovy-bio/OpenPtmFinder/blob/main/config.ini
  --run-server          Start web server after processing
  -n PROCESSES, --processes PROCESSES
                        Maximum number of processes to use.
  --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging verbosity level
```
### Configuration file

Configuration parameters can be set in a **config** file (default values and comments are in [config.ini](OpenPtmFinder/config.ini)).

OpenPtmFinder supports the following parameters:


| Name                             | Default value                            | Description                                                                |
| ---------------------------------|------------------------------------------|----------------------------------------------------------------------------|
|                                  | **[PATH]**                               |                                                                            |
| protein_db                       |  ./data                                  | Path to .fasta file with proteins. If there is no file, OpenPtmFinder uses default one (SwitssProt 2022). |
| unimod_db                        |                                          | Directory with .xml UNIMOD database. An example can be found at https://github.com/Anchovy-bio/OpenPtmFinder/tree/main/data/unimod |
| grouping_file                    |                                          | Directory with annotation file of samples by two groups. An example can be found at https://github.com/Anchovy-bio/OpenPtmFinder/blob/main/config.ini |
| aa-stat_dir                      |                                          | Directory with AA_stat search results (.csv and interpretations.json). Default value is current directory. |
| mzml_dir                         |                                          | Directory with mzml search files. Default value is pepxml directory. |
| pepxml_dir                       |                                          | Directory with pepxml search files or separate files. Default value is current directory. |
| output_dir                       |                                          | Directory to store the results. Default value is current directory. |
|                                  | **[PARAMETERS]**                         |                                                                            |
| delimiter                        | , (comma)                                | Delimiter used in CSV input files.                                         |
| type_of_modifications            | 2                                        | Type of modification based on Unimod accsecions.                                           |
| name_of_modifications            | Phospho, Oxidation                                  | Modification types for searching by Unimod.                                        |
| localization_score_threshold     | 0.3                                      | Selection of peptides based on the reliability of modification localization.                                           |
| mass_tolerance                   | 0.012                                    | Peptide mass error.                                        |
| fdr_threshold                    | 0.05                                     | Error rate in peptide filtering.                                           |
| type_tmt                         | TMT11plex                                | Design of the TMT experiment.                                         |
| min_group_for_stats              | 5                                        | Minimum amount of data to calculate statistics by group.                                           |
| calculation_pval                 | True                                     | To calculate statistics or not.                                         |
| sorting_pepxml                   | True                                     | Sort files or not.                                           |


### Output files

An example of OpenPtmFinder output can be found [here](OpenPtmFinder/output/).

OpenPtmFinder produces the following files:

**A.** 

