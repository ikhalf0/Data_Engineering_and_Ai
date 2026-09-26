# Diabetes Risk Prediction, Data Engineering & Learning Classifier Systems

ENGE707 group project. Group iRobot.

## Project Overview

This project builds an end-to-end data engineering and machine learning pipeline
to predict diabetes/prediabetes risk from self-reported health, lifestyle, and
demographic indicators. Phase I established the data foundation. Phase II
develops a Learning Classifier System (LCS) based solution and compares it with
conventional machine learning models.

## Problem Statement

Type 2 diabetes often goes undiagnosed until complications appear, and clinical
testing isn't always accessible. This project explores whether a low-cost
screening tool based on self-reported survey data could help flag at-risk
individuals earlier. An LCS is well suited to this problem because it produces
human-readable IF-THEN rules, which matter in a health screening context where
predictions need to be explainable rather than only accurate.

## Dataset

- **Name:** CDC Diabetes Health Indicators
- **Source:** UCI Machine Learning Repository (Dataset ID 891)
- **Link:** https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- **Origin:** CDC's 2015 Behavioral Risk Factor Surveillance System (BRFSS)
- **Size:** 253,680 records, 21 features
- **Target variable:** `Diabetes_binary` (0 = no diabetes, 1 = prediabetes/diabetes)
- **Class balance:** 86.07% class 0, 13.93% class 1
- **License:** See UCI dataset page for terms of use

## Repository Structure

```
data/
  raw/                  Raw download from UCI (not committed; see Setup)
  processed/
    diabetes_cleaned.csv              Phase I cleaned dataset
    elcs/
      diabetes_elcs_training_balanced.csv   Balanced training set (Task 3)
      diabetes_elcs_test_unchanged.csv      Held-out test set (Task 3)
notebooks/              Analysis pipeline, run in numerical order
results/                Exported metrics, fold scores and predictions
reports/                Report, pipeline diagram, contribution statement
src/
  load_data.py          Downloads the dataset from UCI
  evaluation.py         Shared split, metrics, cross-validation, statistical tests
requirements.txt        Python environment
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/load_data.py         # downloads data/raw/diabetes_raw.csv
```

The LCS work additionally requires `scikit-eLCS`, which needs an older Python
version (3.9) than the rest of the pipeline:

```bash
pip install scikit-eLCS
```

## Pipeline

Notebooks are run in order. Each reads the output of the previous stage.

### Phase I : Data Engineering

| Notebook | Contents |
|---|---|
| `01_data_inspection.ipynb` | Structure, data types, missing values, duplicates, class balance |
| `02_data_cleaning.ipynb` | Validation, range checks, type correction, cleaned dataset export |
| `03_exploratory_data_analysis.ipynb` | Distributions, correlations, target relationships |

### Phase II : Learning Classifier Systems

| Notebook | Task | Contents |
|---|---|---|
| `04_original_elcs_baseline.ipynb` | 2 | Unmodified eLCS on the minimally processed dataset and baseline evaluation |
| `05_data_preprocessing_feature_engineering.ipynb` | 3 | Leakage-free preprocessing, class balancing, feature preparation and SMOTENC comparison |
| `06_conventional_models.ipynb` | 6 | Logistic Regression, Random Forest and Gaussian Naive Bayes evaluation |
| `07_improved_elcs_system.ipynb` | 4 | Improved eLCS configuration, validation-based tuning and final evaluation |
| `08_lcs_evaluation_and_comparison.ipynb` | 5, 6 | Common evaluation, full model comparison, statistical testing and precision–recall analysis |
| `09_interpretation_of_results.ipynb` | 7 | eLCS rule analysis, explainability and trustworthiness discussion |

## Experimental Protocol

All models are evaluated through `src/evaluation.py` so that results are
directly comparable. The protocol is fixed and must not be changed without
agreement from the whole group.

- **Split:** stratified 80/20 train-test split, `random_state=42`
  (202,944 training records, 50,736 test records).
- **Cross-validation:** stratified 10-fold within the training set, used to
  obtain per-fold scores for statistical testing. Ten folds rather than five
  because the Wilcoxon signed-rank test with five paired observations cannot
  reach p < 0.05.
- **Metrics:** accuracy, balanced accuracy, precision, recall, F1,
  specificity, ROC-AUC, PR-AUC and the confusion matrix. Balanced accuracy and
  PR-AUC are the primary metrics, because the class imbalance makes plain
  accuracy misleading.
- **Statistical tests:** Friedman test across models on cross-validation
  scores, Wilcoxon signed-rank tests on pairs with Holm-Bonferroni correction,
  and McNemar's test on held-out test predictions.
- **Imbalance handling:** random undersampling of the majority class in the
  training data only. SMOTE was rejected because almost all features are binary
  or ordinal codes, so interpolation would produce impossible values and distort
  the LCS rules that Task 7 must interpret.
- **Leakage control:** the test set is held out before any preprocessing.
  Scaling and balancing are applied inside `Pipeline` objects or through
  `BalancedUndersampler`, so they are refitted on each training fold and never
  see validation or test data.

## Reproducibility

- All random operations use `random_state=42`.
- Metrics, fold scores and test-set predictions are exported to `results/`, so
  the comparison and statistical tests can be reproduced without refitting any
  model.
- eLCS runs are slow. Expect several minutes per fit, and roughly 15-20 minutes
  for a 10-fold cross-validation of one LCS configuration.

## Team and Contributions

| Member | Phase I | Phase II |
|---|---|---|
| Simon | Data acquisition and inspection | Original eLCS baseline (Task 2, 4, 7) |
| Khlaf | Data cleaning and transformation | Experimental design and model comparison (Tasks 5, 6, 7) |
| Aman | Exploratory data analysis | Preprocessing and feature engineering (Task 1, 3, 6, 8) |

Work is developed on individual branches (`Simon-branch`, `khalf-branch`,
`Aman-branch`) and merged into `main`. All code required for the final
submission lives on `main`.

