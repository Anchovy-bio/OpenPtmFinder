# OpenPtmFinder
OpenPtmFinder is a powerful tool designed for the comprehensive analysis of post-translational modifications (PTMs) in tandem mass spectrometry-based proteomics data. It utilizes an open search strategy to accurately identify, annotate, and perform differential expression analysis of modifications, presenting the results in an interactive web application.

## 🚀 Key Features
🔍 **Open Search:** Identifies peptides with modifications based on mass shifts obtained from the results of an AA_stat open search.
📚 **Unimod Integration:** Annotates found modifications by matching them against the standard Unimod database.
📊 **Statistical Analysis:** Calculates the differential expression of modifications between specified sample groups.
🌐 **Interactive Web Application:** Presents all results, including statistics and visualizations, in a user-friendly local web interface.
🧪 **TMT Support:** Includes normalization and statistical analysis for TMT-labeled data.

## 💻 Installation
OpenPtmFinder requires Python 3.7 or newer. It is recommended to install it in a separate virtual environment.

### Using pip
You can install the package directly from PyPI (if published) or from the GitHub repository.

```bash
# Recommended: Create and activate a virtual environment
python -m venv openptm_env
source openptm_env/bin/activate  # For Windows use: openptm_env\Scripts\activate

# Install from PyPI
pip install OpenPtmFinder

# Alternatively, install directly from GitHub
pip install git+[https://github.com/Anchovy-bio/OpenPtmFinder.git](https://github.com/Anchovy-bio/OpenPtmFinder.git)
```

## ⚡ Quick Start Guide
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Anchovy-bio/OpenPtmFinder.git](https://github.com/Anchovy-bio/OpenPtmFinder.git)
   cd OpenPtmFinder
   ```
2. **Configure `config.ini`:**
   Edit the `config.ini` file to specify the paths to your input data and to set up analysis parameters.
   ```ini
   # Example config.ini
   [PATHS]
   protein_db = /path/to/your/db.fasta
   unimod_db = /path/to/your/unimod_tables.xml
   grouping_file = /path/to/your/group_df.csv
   # ... and so on

   [PARAMETERS]
   type_of_modifications = Phospho,Deamidation
   # ... other parameters
   ```
3. **Run the application:**
   Execute the main script from your terminal. The `-c` flag points to your `config.ini` file.
   ```bash
   OpenPtmFinder -c config.ini --run_server
   ```
   The program will process the data, perform statistical analysis, and then automatically launch a local web server.

## ⚙️ Configuration Parameters (config.ini)
The behavior of OpenPtmFinder is controlled by the `config.ini` file.

### [PATHS] Section
| Parameter | Description |
| :--- | :--- |
| `protein_db` | **Required.** Path to the FASTA database file with protein sequences. |
| `unimod_db` | **Required.** Path to the UniMod XML file (`unimod_tables.xml`) for modification annotation. |
| `grouping_file` | **Required.** Path to the CSV file defining sample groups for statistical analysis. |
| `aa-stat_dir` | **Required.** Directory containing the results of your AA_stat search. |
| `mzml_dir` | **Required.** Directory with the source `.mzML` files for intensity extraction. |
| `pepxml_dir` | **Required.** Directory with the `.pepXML` files from your open search. |
| `output_dir` | **Required.** Directory where all output files and results will be saved. |

### [PARAMETERS] Section
| Parameter | Description |
| :--- | :--- |
| `type_of_modifications`| A comma-separated list of modification names to search for (e.g., `Phospho, Oxidation`). |
| `name_of_modifications`| Deprecated. Use `type_of_modifications` for clarity. |
| `localization_score_threshold` | The minimum score for reliable localization of a modification site (e.g., `0.3`). |
| `mass_tolerance` | The mass error tolerance for matching peptides (e.g., `0.012`). |
| `fdr_threshold` | The False Discovery Rate (FDR) threshold for filtering peptides (e.g., `0.05`). |
| `type_tmt` | Specifies the type of TMT label (e.g., `TMT10plex`, `TMT11plex`). |
| `min_group_for_stats`| The minimum number of samples in a group to perform statistical tests (e.g., `5`). |
| `calculation_pval` | `True` or `False`. Set to `True` to enable p-value calculation. |
| `sorting_pepxml` | `True` or `False`. Set to `True` to enable sorting of `pepXML` files. |
| `nproc` | The number of processors for parallel processing. |
| `port` | The port number for the local web server (e.g., `10005`). |

## 📊 How it works: Workflow
1.  **Input Data:** The program reads raw data from the `mzml`, `pepxml`, and `aa-stat` directories you specify.
2.  **Peptide Processing:** Filters peptides based on FDR, localization scores, and specified modification types.
3.  **Intensity Extraction:** Quantifies TMT reporter ion intensities from the `.mzml` files.
4.  **Data Normalization:** Normalizes TMT intensities across all samples to correct for technical variations.
5.  **Statistical Analysis:** Compares modification levels between groups and calculates statistical significance (p-values).
6.  **Web Server Generation:** Compiles the results into a dynamic, interactive web application for data analysis.

## 📁 Output Files
The results are saved in the directory specified by `output_dir` in the `config.ini` file. Key output files include:

- `final_stat_result.csv`: A CSV file containing all peptides, their modifications, normalized intensities, statistical results (p-values, fold change), and protein information. This is the main output file.
- `filtered_df_with_intens.csv`: An intermediate file containing filtered peptides with extracted TMT intensities before normalization and statistical analysis.
- `unimod_search.csv`: 
- `unimod.csv`: 
- `cataloque.csv`: 
- `pepxml_psms.csv`: 
- `annotated_df.pickle`: 

## 🖼️ Visualizations
The web application provides interactive plots to visualize your data:

-   **Bar Chart:** Shows the landscape of found modifications for a specific protein, as well as the protein coverage by peptides.
-   **Pie Chart:** Illustrates the distribution of all identified modification types across the entire dataset.

## 🤝 Contribution and Support
We welcome your contributions to the project. Please feel free to open issues or submit pull requests.

## 📝 License
This project is distributed under the MIT License. See the `LICENSE` file for details.

## 📖 How to Cite
The preprint of this article is available on biorXiv: ...