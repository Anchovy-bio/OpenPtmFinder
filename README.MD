

OpenPtmFinder
==============
PTM Annotation Tool Based on Open strategy search in proteomics

# Overview

**OpenPtmFinder** is a tool for shotgun labled proteomics for annotated peptide post-translation modifications, 
in the results of open strategy analysis.
> The program searches for peptides with a given modification type in all spectra based on mass shifts from the results of the open AA_stat search. 
> The program selects post-translational modifications based on the Unimod database and calculates the position of the modification on the protein.
> The program calculates the differential expression of the modifications found between the two groups.
> At the output, you get a local web application with calculated statistics and graphs.


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
![terminal animation](manual_pic/tty.gif)

To check results, open the file called **report.html** in your browser.
Example reports are shown [here](https://gorshkovlab.github.io/aa_stat_reports/).


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

Configuration parameters can be set in a **config** file (default values and comments are in [default.cfg](AA_stat/default.cfg)).

AA_stat supports the following parameters:


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
| type_of_modifications            | 2                                        | Type of modification baesd on Unimod accsecions.                                           |
| name_of_modifications            | Phospho                                  | Delimiter used in CSV input files.                                         |
| localization_score_threshold     | 0.3                                      | Type of modification baesd on Unimod accsecions.                                           |
| mass_tolerance                   | 0.012                                    | Delimiter used in CSV input files.                                         |
| fdr_threshold                    | 0.05                                     | Type of modification baesd on Unimod accsecions.                                           |
| type_tmt                         | TMT11plex                                | Delimiter used in CSV input files.                                         |
| min_group_for_stats              | 5                                        | Type of modification baesd on Unimod accsecions.                                           |
| calculation_pval                 | True                                     | Delimiter used in CSV input files.                                         |
| sorting_pepxml                   | True                                     | Type of modification baesd on Unimod accsecions.                                           |


### Output files

An example of AA_stat output can be found [here](https://gorshkovlab.github.io/aa_stat_reports/).

AA_stat produces the following files:

**A.** Gaussian fit report (gauss_fit.pdf).

**B.** Summary histogram (summary.png).

**C.** Charts (PNG and SVG files) of normalized frequencies for each significant mass shift interval. If MGF or mzML files are provided, tables with modifies peptide sequences and localization scores are generated.

**D.** Summary table (aa_statistics_table.csv) of amino acid frequencies for all mass shifts with Unimod.org links for possible modifications.

**E.** Summary table (p_values.csv) with p-values for each amino acid frequencies in each mass shift.

**F.** HTML file (report.html) aggregates and illustrates all results.

**G.** If MGF or mzML files are provided, a table with localization results is created (localization_statistics.csv).

**A.** Gaussian fit file shows PSM distributions in intervals that were considered as mass shift peaks.
Subplot titles correspond to mass shifts (interval center).
Peaks are classified into 3 groups:
PASSED - mass shift with good fit, which are considered for subsequent analysis;
NO FIT - mass shifts for which the algorithm could not find Gaussian function;
FAILED - mass shift with a fit not passing the configured filtering criteria.

|![img1](manual_pic/gauss.png )|
|------------------------------|
| **Figure 1.** Examples of Gaussian fit results in gauss_fit.pdf.|

**B.** Summary histogram shows numbers of filtered PSMs in all mass shift intervals.
Number on top of the bin indicates the percentage of all identified PSMs.
Each mass shift interval is filtered separately to the user-specified FDR level, using target-decoy approach.

|![img2](manual_pic/summary.png)|
|-------------------------------|
| **Figure 2.** Example of Summary histogram.|

**C.** Charts of normalized frequencies for each significant mass shift.
Each chart is named according to the mass shift.
Each bar in the chart denotes the normalized occurrence frequency
of a specific amino acid residue in the given mass shift interval.
The normalized frequency is calculated by:

1. Counting all amino acids in all non-redundant peptides identified with the given mass shifts;

2. Dividing the count for the given residue by the total amino acid count for the interval
   to obtain the occurrence frequency of the residue;

3. Normalizing the occurrence frequency of the residue
   by the occurrence frequency of the same residue for the zero mass shift interval.

If the normalized frequency of a residue significantly exceeds 1,
that means that this residue is "enriched" in the peptides identified with the corresponding mass shift,
suggesting that there is a connection between this residue and the cause of the mass shift.
In the simplest case, this residue is modified:


|![img3](manual_pic/ms.png)|
|--------------------------|
| **Figure 3.** Example of a normalized frequency chart for 15.9943 mass shift (green). Orange -- the percentage of peptides that contain at least one AA residue of a certain kind. Blue -- successful localizations of mass shifts at each AA (only if MS/MS spectra are provided). The counts are not normalized.|


**D.** Summary table (aa_statistics_table.csv) of amino acid frequencies for all mass shifts with peptide counts and Unimod.org links for possible modifications.

**E.** Summary table (p_values.csv) with p-values for all amino acid frequencies in all mass shifts that indicates the significant deviation amino acid frequency from zero mass shift peak.

**F.** HTML file (report.html) aggregates and illustrates all results. Examples can be found [here](https://gorshkovlab.github.io/aa_stat_reports/).

**G.** A summary of localization shows the number of peptides in each bin for which a modification was successfully localized in MS/MS.
Localization is done by generating theoretical spectra of possible isoforms and scoring them against the experimental spectrum.
If there is a clear winner in terms of score, the spectrum is considered localized.

|Column name|Description|
|------------|-----------|
|mass shift|Considered mass shift mass.|
|# peptides in bin| Number of peptides that passed all filtering procedures.|
|is isotope| Boolean. True if mass shift is a potential isotope of some other mass shift.|
|isotope index| If 'is isotope' is True, this column contains the mass of monoisotopic peak.|
|sum of mass shifts| Shows all possible pairs of mass shifts that produce the given mass shift.|
|unimod candidates| Localization candidates retrieved from Unimod.org database for given mass shift.|
|aa_stat candidates| Localization candidates from AA_stat statistics.|
|candicates for loc| Combination of amino acid candidates from all sources: Unimod, AA_stat results, isotope clusters, sum of modifications, considered for localization of mass shift using MS/MS spectra.|
|localization| Localization statistics for given mass shift using MS/MS spectra provided in a format "AminoAcid_MassShift:number-of-peptides-with-this-localization". If there is no clear leader for a specific peptide's mass shift localization, it counts as 'non-localized'.
