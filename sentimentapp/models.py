"""Candidate models, cross-validated comparison and light hyper-parameter tuning.

Every learned candidate is wrapped in the SAME pipeline (text features → classifier), so the comparison is fair and
nothing learned in one validation fold leaks into another. Folds keep the class shares equal (stratified).
Next to the learned models sit two that learn nothing: "always the most common class" and a rule-based word list.
They show how much the learning adds.

Models are ranked by macro-F1: the average of the F1 scores of the three classes, so the small Neutral class
counts as much as the big ones. Accuracy alone would reward ignoring it.
"""

import time
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import preprocessing as P
from .lexicon import LexiconClassifier

BASELINE = "Baseline (most common class)"
LEXICON = "Rule-based word list"
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy",
           "log_loss": "neg_log_loss"}

TUNING_GRIDS = {
    "Naive Bayes": {"model__alpha": [0.03, 0.1, 0.3, 1.0, 3.0]},
    "Logistic Regression": {"model__C": [1, 3, 10, 30, 100]},
    "Linear SVM": {"model__estimator__C": [0.1, 0.3, 1, 3]},
}


def build_pipeline(estimator):
    if getattr(estimator, "needs_raw_text", False):          # the rule-based scorer reads the text itself
        return Pipeline([("model", estimator)])
    return Pipeline([("features", P.build_features()), ("model", estimator)])


def candidates(seed=42, fast=False):
    """name -> unfitted estimator (the pipeline is built around it)."""
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        LEXICON: LexiconClassifier(),
        "Naive Bayes": MultinomialNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(C=10, max_iter=3000),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3, method="sigmoid"),
    }


def _summarise(name, scores, seconds):
    out = {"model": name, "seconds": round(seconds, 1)}
    for key in SCORING:
        value = float(scores[f"test_{key}"].mean())
        out[key] = -value if key == "log_loss" else value
    out["f1_macro_std"] = float(scores["test_f1_macro"].std())
    return out


def compare(X, y, cv=5, seed=42, fast=False, progress=None):
    """Cross-validate every candidate on the TRAINING data only. Best (by macro-F1) first."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    results = []
    for name, estimator in candidates(seed, fast).items():
        if progress:
            progress(f"Cross-validating {name} ...")
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(build_pipeline(estimator), X, y, cv=folds, scoring=SCORING)
        results.append(_summarise(name, scores, time.perf_counter() - started))
    return sorted(results, key=lambda r: (r["f1_macro"], r["accuracy"]), reverse=True)


def best_candidate(results):
    """The best-scoring LEARNED model (not the baseline and not the rule-based list)."""
    return next(r for r in results if r["model"] not in (BASELINE, LEXICON))


def tune(name, X, y, seed=42, cv=3, iterations=5, fast=False, progress=None):
    """Randomised search over a small grid. Returns (search, tuned_params) or (None, {}) if there is no grid."""
    grid = TUNING_GRIDS.get(name)
    if not grid:
        return None, {}
    if progress:
        progress(f"Tuning {name} ({iterations} settings × {cv} folds) ...")
    search = RandomizedSearchCV(build_pipeline(candidates(seed, fast)[name]), grid,
                                n_iter=int(min(iterations, np.prod([len(v) for v in grid.values()]))),
                                cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                                scoring="f1_macro", random_state=seed, n_jobs=1, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    return search, {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}


def pipeline_with_params(name, params, seed=42, fast=False):
    """A fresh pipeline for `name` with tuned parameters applied."""
    estimator = candidates(seed, fast)[name]
    estimator.set_params(**params)
    return build_pipeline(estimator)
