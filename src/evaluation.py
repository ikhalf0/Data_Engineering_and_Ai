"""
Shared evaluation module for ENGE707 Project Phase 2 (Tasks 5 and 6).

Every model in the project (original eLCS, improved eLCS, and the
conventional ML models) must be evaluated through this module so that
all comparisons use the same data split, the same folds, and the same
metrics.

Validation strategy
-------------------
1. A single stratified 80/20 train-test split (random_state=42).
   The 20% test set is held out and used only for final reporting.
2. Stratified 10-fold cross-validation inside the 80% training set.
   This gives ten scores per model for the Friedman and Wilcoxon tests.
3. McNemar's test on the held-out test predictions for pairwise
   comparisons (e.g. improved eLCS vs original eLCS).

Usage (from a notebook in notebooks/):
    import sys; sys.path.append("..")
    from src.evaluation import load_raw_data, get_train_test_split, evaluate_on_test
"""

import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Shared settings - do not change without telling the whole group
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 10
TARGET = "Diabetes_binary"

RAW_PATH = "../data/raw/diabetes_raw.csv"
CLEANED_PATH = "../data/processed/diabetes_cleaned.csv"

# Main metric for cross-validation and statistical tests. Balanced accuracy
# is used because plain accuracy is misleading with an 86/14 class split.
PRIMARY_METRIC = "balanced_accuracy"


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------
def load_raw_data(path=RAW_PATH):
    """Load the UCI raw dataset saved by src/load_data.py."""
    return pd.read_csv(path)


def load_cleaned_data(path=CLEANED_PATH):
    """Load the Phase 1 cleaned dataset."""
    return pd.read_csv(path)


def get_train_test_split(df, target=TARGET):
    """
    Return the project's shared stratified 80/20 split.

    Because the rows and random_state are fixed, the same row indices
    end up in the test set every time, for every team member.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_cv_folds(X_train, y_train):
    """Return the shared list of (train_idx, val_idx) folds."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    return list(skf.split(X_train, y_train))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _clean_predictions(y_pred):
    """
    eLCS can leave a record unclassified (NaN) if no rule matches it.
    Those records are counted as predicted class 0 (the majority class),
    and the count is returned so it can be reported.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    n_unclassified = int(np.isnan(y_pred).sum())
    y_pred = np.where(np.isnan(y_pred), 0, y_pred).astype(int)
    return y_pred, n_unclassified


def _positive_class_scores(model, X):
    """Probability of class 1, or None if the model cannot provide it."""
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = np.asarray(model.predict_proba(X), dtype=float)
    except Exception:
        return None
    if proba.ndim != 2 or proba.shape[1] < 2:
        return None
    return np.nan_to_num(proba[:, 1], nan=0.0)


def compute_metrics(y_true, y_pred, y_score=None):
    """All classification metrics required by the brief, as a dict."""
    y_true = np.asarray(y_true).astype(int)
    y_pred, n_unclassified = _clean_predictions(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "unclassified": n_unclassified,
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    return metrics


def _to_input(X, as_numpy):
    """eLCS needs NumPy arrays; sklearn pipelines can use DataFrames."""
    if as_numpy and hasattr(X, "to_numpy"):
        return X.to_numpy()
    return X


# ---------------------------------------------------------------------------
# Held-out test evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test(model, name, X_train, y_train, X_test, y_test,
                     as_numpy=False):
    """
    Fit `model` on the training set and score it on the held-out test set.

    Returns (metrics_dict, y_pred). Keep y_pred - it is needed for
    McNemar's test. Set as_numpy=True for eLCS models.
    """
    Xtr, Xte = _to_input(X_train, as_numpy), _to_input(X_test, as_numpy)
    ytr = np.asarray(y_train).astype(int)

    start = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(Xte)
    y_score = _positive_class_scores(model, Xte)
    predict_time = time.perf_counter() - start

    metrics = compute_metrics(y_test, y_pred, y_score)
    metrics.update({"model": name, "fit_time_s": fit_time,
                    "predict_time_s": predict_time})
    y_pred_clean, _ = _clean_predictions(y_pred)
    return metrics, y_pred_clean


def results_table(results):
    """Turn a list of metrics dicts into a tidy comparison table."""
    cols = ["model", "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "specificity", "roc_auc", "pr_auc",
            "tn", "fp", "fn", "tp", "unclassified",
            "fit_time_s", "predict_time_s"]
    table = pd.DataFrame(results)
    return table[[c for c in cols if c in table.columns]].set_index("model")


# ---------------------------------------------------------------------------
# Cross-validation (for statistical tests)
# ---------------------------------------------------------------------------
def cross_validate_model(model, name, X_train, y_train, folds=None,
                         as_numpy=False, verbose=True):
    """
    Run the shared 10-fold CV on the training set.

    A fresh copy of the model is fitted on each fold, so any preprocessing
    or resampling inside a Pipeline is re-fitted on each training fold
    only (no data leakage). Returns one row per fold.
    """
    if folds is None:
        folds = get_cv_folds(X_train, y_train)
    y_arr = np.asarray(y_train).astype(int)

    rows = []
    for i, (tr_idx, val_idx) in enumerate(folds, start=1):
        X_tr = X_train.iloc[tr_idx] if hasattr(X_train, "iloc") else X_train[tr_idx]
        X_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
        X_tr, X_val = _to_input(X_tr, as_numpy), _to_input(X_val, as_numpy)

        fold_model = clone(model)
        fold_model.fit(X_tr, y_arr[tr_idx])
        y_pred = fold_model.predict(X_val)
        y_score = _positive_class_scores(fold_model, X_val)

        m = compute_metrics(y_arr[val_idx], y_pred, y_score)
        m.update({"model": name, "fold": i})
        rows.append(m)
        if verbose:
            print(f"{name} - fold {i}/{len(folds)}: "
                  f"{PRIMARY_METRIC}={m[PRIMARY_METRIC]:.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test on two models' predictions for the same test set.

    b = records model A got right and model B got wrong
    c = records model B got right and model A got wrong
    Uses the exact binomial test when b + c < 25, otherwise the
    chi-square version with continuity correction.
    """
    y_true = np.asarray(y_true).astype(int)
    a_right = np.asarray(y_pred_a) == y_true
    b_right = np.asarray(y_pred_b) == y_true
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0,
                "method": "no disagreements"}
    if b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(b), "p_value": p,
                "method": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(stat, df=1)
    return {"b": b, "c": c, "statistic": stat, "p_value": p,
            "method": "chi-square (continuity corrected)"}


# ---------------------------------------------------------------------------
# Class-imbalance handling
# ---------------------------------------------------------------------------
class BalancedUndersampler(BaseEstimator, ClassifierMixin):
    """
    Wraps any classifier so that the majority class is randomly
    undersampled to the minority-class size before fitting.

    This reproduces the balancing used in notebook 05, but does it inside
    fit(), which means that during cross-validation the undersampling is
    re-done on each training fold and never touches the validation fold.
    Predictions are made on the untouched data.

        model = BalancedUndersampler(RandomForestClassifier(random_state=42))
    """

    def __init__(self, estimator=None, random_state=RANDOM_STATE):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        rng = np.random.RandomState(self.random_state)

        classes, counts = np.unique(y, return_counts=True)
        n_keep = counts.min()

        keep = []
        for cls in classes:
            idx = np.flatnonzero(y == cls)
            if len(idx) > n_keep:
                idx = rng.choice(idx, size=n_keep, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))

        X_bal = X.iloc[keep] if hasattr(X, "iloc") else np.asarray(X)[keep]

        self.classes_ = classes
        self.n_training_rows_ = len(keep)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_bal, y[keep])
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def _fold_matrix(cv_results, metric):
    """Folds as rows, models as columns."""
    return cv_results.pivot(index="fold", columns="model", values=metric)


def friedman_test(cv_results, metric=PRIMARY_METRIC):
    """
    Friedman test: do any of the models differ on `metric` across folds?
    `cv_results` is the concatenation of cross_validate_model() outputs.
    Needs at least 3 models.
    """
    matrix = _fold_matrix(cv_results, metric)
    stat, p = stats.friedmanchisquare(*[matrix[c] for c in matrix.columns])
    mean_ranks = matrix.rank(axis=1, ascending=False).mean().sort_values()
    return {"statistic": stat, "p_value": p, "mean_ranks": mean_ranks}


def pairwise_wilcoxon(cv_results, metric=PRIMARY_METRIC, alpha=0.05):
    """
    Wilcoxon signed-rank tests between every pair of models, with Holm
    correction for multiple comparisons.

    10 folds are used because with 5 folds the smallest possible
    two-sided Wilcoxon p-value is 0.0625, so no pair could ever reach
    p < 0.05. With 10 folds the minimum is about 0.002.
    """
    matrix = _fold_matrix(cv_results, metric)
    rows = []
    for a, b in combinations(matrix.columns, 2):
        diff = matrix[a] - matrix[b]
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = stats.wilcoxon(matrix[a], matrix[b])
        rows.append({"model_a": a, "model_b": b,
                     "mean_diff": diff.mean(), "statistic": stat,
                     "p_value": p})
    table = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    # Holm-Bonferroni correction
    m = len(table)
    adjusted = (table["p_value"] * (m - np.arange(m))).cummax().clip(upper=1)
    table["p_holm"] = adjusted
    table["significant"] = table["p_holm"] < alpha
    return table