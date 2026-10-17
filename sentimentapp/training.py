"""The training workflow, step by step.

    raw data → clean → split (train/test, same class shares) → compare models by cross-validation (train only)
             → tune the best learned model → refit → out-of-fold probabilities on the training set
             → choose the confidence threshold → evaluate ONCE on the untouched test set → explain → save

The test set is used exactly once, for the final report; every choice (which model, which settings, which confidence
threshold) is made with cross-validation on the training reviews.
"""

import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from . import __version__
from . import evaluation as E
from . import models as M
from . import preprocessing as P
from . import schema as S
from .exceptions import DataError

log = logging.getLogger("sentimentapp")


@dataclass
class TrainingResult:
    pipeline: object
    metadata: dict
    comparison: list
    cleaning: P.CleaningReport
    test_frame: pd.DataFrame          # text / actual / predicted / confidence / class probabilities for every test review
    reference: pd.DataFrame           # cleaned TRAINING reviews (used to find similar reviews)
    errors: list = field(default_factory=list)


def train(df_raw, seed=42, fast=False, tune=True, test_size=0.2, cv=5, progress=None, source="dataset"):
    """Run the whole workflow. `progress(message)` is called as each stage starts."""
    say = progress or (lambda message: None)
    started = datetime.now()

    say("Cleaning the data ...")
    df, report = P.clean(df_raw)
    X, y = P.split_xy(df)

    say("Splitting into training and test sets ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if y_train.value_counts().min() < cv:
        raise DataError(f"Too few reviews of one kind to cross-validate ({y_train.value_counts().min()} in the smallest class).")

    comparison = M.compare(X_train, y_train, cv=cv, seed=seed, fast=fast, progress=say)
    best = M.best_candidate(comparison)
    chosen, params = best["model"], {}

    tuned_note = ""
    if tune and chosen in M.TUNING_GRIDS:
        _, params = M.tune(chosen, X_train, y_train, seed=seed, fast=fast, progress=say)
        if params:
            say("Checking the tuned settings ...")
            scores = cross_validate(M.pipeline_with_params(chosen, params, seed, fast), X_train, y_train,
                                    cv=StratifiedKFold(cv, shuffle=True, random_state=seed), scoring=M.SCORING)
            tuned = M._summarise(f"{chosen} (tuned)", scores, 0.0)
            if tuned["f1_macro"] > best["f1_macro"]:
                comparison = sorted(comparison + [tuned], key=lambda r: (r["f1_macro"], r["accuracy"]), reverse=True)
                best, tuned_note = tuned, "tuned"
            else:
                params = {}
                tuned_note = "tuning did not improve cross-validation, kept defaults"

    say("Training the final model ...")
    pipeline = (M.pipeline_with_params(chosen, params, seed, fast) if params
                else M.build_pipeline(M.candidates(seed, fast)[chosen]))
    pipeline.fit(X_train, y_train)

    say("Choosing the confidence threshold (from training data only) ...")
    oof = cross_val_predict(clone(pipeline), X_train, y_train, cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                            method="predict_proba")
    confidence = E.choose_confidence_threshold(y_train, oof)

    say("Evaluating on the untouched test set ...")
    test_p, train_p = pipeline.predict_proba(X_test), pipeline.predict_proba(X_train)
    test_metrics = E.overall_metrics(y_test, test_p)
    test_metrics["per_class"] = E.per_class(y_test, test_p)
    test_metrics["selective"] = E.selective_metrics(y_test, test_p, confidence["confidence"])
    test_metrics["majority_accuracy"] = float(100 * y_test.value_counts(normalize=True).max())
    train_metrics = E.overall_metrics(y_train, train_p)
    baseline_row = next(r for r in comparison if r["model"] == M.BASELINE)
    lexicon_row = next(r for r in comparison if r["model"] == M.LEXICON)

    test_frame = X_test.copy()
    test_frame["actual"] = y_test.to_numpy()
    test_frame["predicted"] = test_p.argmax(axis=1)
    test_frame["confidence"] = test_p.max(axis=1)
    for i, key in enumerate(S.KEYS):
        test_frame[f"p_{key}"] = test_p[:, i]

    say("Finding the most telling words and the mistakes ...")
    words = E.top_words(X_train["text"], y_train)
    errors = E.error_examples(test_frame)

    reference = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    metadata = {
        "project_version": __version__,
        "trained_at": started.isoformat(timespec="seconds"),
        "training_seconds": round((datetime.now() - started).total_seconds(), 1),
        "source": source,
        "seed": seed,
        "model": chosen,
        "model_note": tuned_note,
        "model_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v)) for k, v in params.items()},
        "rows": {"raw": report.rows_in, "clean": report.rows_out, "train": int(len(X_train)), "test": int(len(X_test))},
        "class_counts": report.class_counts,
        "metrics": {"test": test_metrics, "train": train_metrics, "cv": best, "baseline_cv": baseline_row,
                    "lexicon_cv": lexicon_row},
        "confidence": confidence,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "platform": sys.platform},
        "comparison": comparison,
        "cleaning": report.to_dict(),
        "confusion": E.confusion(y_test, test_p),
        "top_words": words,
        "errors": errors,
        "segments": E.summarise_frame(test_frame),
        "coverage_curve": E.coverage_curve(y_test, test_p),
        "confidence_bands": E.accuracy_by_confidence(y_test, test_p),
        "ceiling": E.noise_ceiling(0.03) if str(source).startswith("synthetic") else None,
    }
    log.info("Trained %s: test accuracy=%.1f macro-F1=%.1f", chosen, test_metrics["accuracy"], test_metrics["f1_macro"])
    return TrainingResult(pipeline, metadata, comparison, report, test_frame, reference, errors)
