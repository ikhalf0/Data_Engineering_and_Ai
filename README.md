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

## Key Results

All twelve models were evaluated on the same held-out test set of 50,736 records.

| Model | Balanced accuracy | Recall |
|---|---|---|
| Random Forest (undersampled) | 0.7486 | 0.7905 |
| Improved eLCS | 0.7309 | 0.7621 |
| Original eLCS (preprocessed data) | 0.7302 | 0.7640 |
| Original eLCS (minimally processed data) | 0.5000 | 0.0000 |

The baseline eLCS reached 86.06% accuracy while detecting none of the 7,069
positive cases, which is why balanced accuracy is used as the primary metric.
Almost all of the improvement came from balancing the training data rather than
from the changes to the LCS configuration: the difference between the improved
eLCS and the original eLCS on preprocessed data was not statistically
significant (McNemar, p = 0.0511).

## Repository Structure

```
data/
  raw/                  Raw download from UCI (not committed; see Setup)
  processed/
    diabetes_cleaned.csv                   Phase I cleaned dataset
    elcs/
      diabetes_elcs_training_balanced.csv  Undersampled training set (Task 3)
      diabetes_elcs_training_smotenc.csv   SMOTENC oversampled training set (Task 3)
      diabetes_elcs_test_unchanged.csv     Held-out test set (Task 3)
notebooks/              Analysis pipeline, run in numerical order
results/                Exported metrics, predictions and rules (see results/README.md)
reports/                Report, pipeline diagram, contribution statement
src/
  load_data.py          Downloads the dataset from UCI
  evaluation.py         Shared split, metrics, cross-validation, statistical tests
requirements.txt        Analysis environment (Python 3.12+)
requirements-elcs.txt   eLCS environment (Python 3.9), runs every notebook
```

## Setup

Two environments are provided because `scikit-eLCS` does not run on recent
Python versions. The Python 3.9 environment runs every notebook in the project
and is the one used to produce the committed results.

```bash
python3.9 -m venv venv39
source venv39/bin/activate          # Windows: venv39\Scripts\activate
pip install -r requirements-elcs.txt

python src/load_data.py             # downloads data/raw/diabetes_raw.csv
```

`requirements.txt` sets up a Python 3.12+ environment with current library
versions. It runs notebooks 01 to 03 and 05 but cannot run the LCS notebooks.

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
  (202,944 training records, 50,736 test records). The test set is used only
  for final reporting.
- **Cross-validation:** stratified 10-fold within the training set, used to
  obtain per-fold scores for statistical testing. Ten folds rather than five
  because the Wilcoxon signed-rank test with five paired observations cannot
  produce a two-sided p-value below 0.0625 and so could never reach
  significance. The conventional models are cross-validated; the LCS models
  are evaluated on the held-out test set only, because a single improved-eLCS
  run takes around 16 minutes to predict and ten-fold cross-validation of
  three configurations would take approximately eight hours.
- **Metrics:** accuracy, balanced accuracy, precision, recall, F1,
  specificity, ROC-AUC, PR-AUC and the confusion matrix. Balanced accuracy and
  PR-AUC are the primary metrics, because the class imbalance makes plain
  accuracy misleading. Records left unclassified by an LCS, where no rule
  matches, are counted as negative predictions and reported separately.
- **Statistical tests:** the Friedman test across models on cross-validation
  scores; Wilcoxon signed-rank tests against the best-performing model as a
  control, with Holm-Bonferroni correction; and McNemar's test on the held-out
  test predictions. All-pairs Wilcoxon testing was run first and retained in
  notebook 06, but it cannot detect a difference in this design: with 10 folds
  the smallest possible p-value is 0.00195, so Holm correction across 36 pairs
  raises it to 0.070 and no comparison can reach significance even where one
  model wins on every fold. Comparing against a single control divides the
  correction by 8 rather than 36 and is the standard post-hoc procedure for
  this situation.
- **Imbalance handling:** random undersampling of the majority class in the
  training data only, which outperformed both class weighting and SMOTENC for
  all three conventional models. SMOTE was rejected because almost all
  features are binary or ordinal codes, so interpolation would produce
  impossible values and distort the LCS rules that Task 7 must interpret.
  SMOTENC, which samples categorical features rather than interpolating them,
  was evaluated as a secondary comparison and performed worse.
- **Leakage control:** the test set is held out before any preprocessing.
  Scaling and resampling are applied inside `Pipeline` objects or through the
  `BalancedUndersampler` and `SMOTENCResampler` wrappers in
  `src/evaluation.py`, so they are refitted on each training fold and never
  see validation or test data.

## Reproducibility

- All random operations use `random_state=42`.
- Metrics, fold scores and test-set predictions are exported to `results/`, so
  the comparison and statistical tests can be reproduced without refitting any
  model. See `results/README.md` for what each file contains.
- eLCS runs are slow. Prediction dominates, because every test record is
  matched against the full rule population: the improved eLCS takes around
  75 seconds to train and 16 minutes to predict on 50,736 records. Notebook 08
  takes roughly 25 to 45 minutes in total; notebook 06 takes 30 to 60 minutes,
  mostly for the SMOTENC cross-validation.

## Team and Contributions

| Member | Phase I | Phase II |
|---|---|---|
| Simon Aung | Data acquisition and inspection | Original eLCS baseline and improved eLCS system (Tasks 2, 4) |
| Khlaf Alshammari | Data cleaning and transformation | Shared evaluation module, experimental design, model comparison and rule interpretation (Tasks 5, 6, 7) |
| Aman Mohammed | Exploratory data analysis | Phase I summary, preprocessing and feature engineering, discussion (Tasks 1, 3, 8) |

Work is developed on individual branches (`Simon-branch`, `khalf-branch`,
`Aman-branch`) and merged into `main`. All code required for the final
submission lives on `main`.
