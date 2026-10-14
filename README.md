# Can Cancer-Cell Transcriptomes Predict Compound Response? A Reproducible Benchmark

## Overview and Motivation
This project evaluates whether baseline gene-expression profiles can accurately predict cancer cell-line responses to compounds. In computational oncology, machine learning models often report artificially high predictive accuracy due to undetected data leakage (e.g., tissue lineage overlap) or uncorrected experimental batch effects. 

This repository establishes a highly rigorous, reproducible baseline benchmark prioritizing experimental data integrity over algorithm complexity. It demonstrates that for a high-variance feasibility target, complex non-linear models (Random Forest) trained on baseline transcriptomics fail to outperform a simple mean-prediction dummy baseline when generalizing to unseen tissue lineages.

## Data Provenance
* **Transcriptomics:** DepMap Public 25Q2 (`OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `Model.csv`).
* **Drug Response:** PRISM Repurposing 19Q4 (`secondary-screen-dose-response-curve-parameters.csv`).
* **Identifier Mapping:** Identifiers map directly via the Broad Institute's standardized `ModelID` to the PRISM `depmap_id`. The intersection yields exactly 477 overlapping cell lines.

## Preprocessing and Experimental Quality Control
The PRISM 19Q4 dataset contains ~45k technical/biological replicate observations. Blindly aggregating these replicates introduces batch-effect noise.
* **MTS010 Priority Rule:** Per PRISM documentation, `MTS010` screens represent corrected "technical redos." Our processing pipeline rigorously prioritizes `MTS010` records over older screens, rescuing 37,474 high-quality pairs.
* **Ambiguity Quarantine:** 7,515 duplicated pairs (solely collisions between older `HTS002` and `MTS006` screens) lacked a documented priority rule. These were explicitly dropped from the target matrix to ensure absolute experimental integrity.

## Target Selection
To establish a benchmark, a single compound was selected dynamically using the following criteria:
1. Maximum number of valid, non-null tested cell lines (to maximize training data).
2. Highest variance in Area Under the Curve (AUC) response (to ensure the model has a dynamic range to predict).

**Selected Target:** `BRD-K13662825-001-07-5` (477 valid responses). 
*Note: Because maximizing variance favors a specific type of prediction problem, this compound serves as a "feasibility target" to test the pipeline's mechanics, rather than a universally representative drug.*

## Cross-Validation Design & Leakage Prevention
To ensure predictions generalize to novel biology rather than memorizing tissue-specific signatures, the pipeline utilizes a **GroupKFold (n=5)** cross-validation strategy grouped by `OncotreeLineage` (22 distinct tissue types).
* **Zero Lineage Leakage:** Audits confirm the intersection of training and testing lineages across all 5 folds is mathematically zero.
* **Nested Feature Selection:** Variance thresholding (top 1,000 highly variable genes) is calculated strictly *inside* the training fold before transforming the test fold.

## Benchmark Results (Average across 5 folds)
| Model | Mean Absolute Error (MAE) |
| :--- | :--- |
| **Dummy Baseline (Train Mean)** | `0.0929 ± 0.0090` |
| **Ridge Regression (alpha=1.0)** | `0.1246 ± 0.0098` |
| **Random Forest (100 estimators)** | `0.0957 ± 0.0097` |

## Limitations and Conclusion
The strict lineage-grouped evaluation confirms that baseline transcriptomics alone (specifically the top 1,000 highly variable genes) cannot reliably predict this compound's response better than guessing the mean of the training fold. The Random Forest's inability to beat the dummy baseline is a genuine biological limitation of predicting complex compound efficacy without incorporating multi-omics, molecular descriptors, or non-linear batch effect corrections.

## Reproduction Instructions
Due to file size limits, raw datasets are excluded from this repository.
1. Clone this repository and install dependencies: `pip install -r requirements.txt`
2. Download the DepMap 25Q2 and PRISM 19Q4 datasets linked in `data/README.md` and place them in the `data/raw/` directory.
3. Resolve replicates and generate the target matrix: `python scripts/resolve_prism_replicates.py`
4. Run the cross-validation benchmark: `python scripts/group_cv_benchmark.py`