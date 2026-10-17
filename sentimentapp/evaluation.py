"""Model evaluation for a three-class problem where some mistakes are worse than others.

Calling a Positive review Neutral is a small slip; calling it Negative is a serious one. So besides accuracy and
macro-F1 this module counts those "polarity flips", builds the confusion matrix, chooses a confidence threshold below
which a prediction is left to a person, and lists the actual mistakes.
"""

import math
import re
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support

from . import schema as S
from .textproc import NEGATORS, normalise_text, pretty_token

LABELS = [S.NEGATIVE, S.NEUTRAL, S.POSITIVE]
HIDDEN_TOKENS = {"lexpos", "lexneg"}      # shared polarity markers: they would crowd out the actual words in the lists


def overall_metrics(actual, probabilities):
    """Accuracy, macro-F1, balanced accuracy, log loss, average ordinal error and the share of polarity flips."""
    actual, probabilities = np.asarray(actual, int), np.asarray(probabilities, float)
    predicted = probabilities.argmax(axis=1)
    flips = ((actual == S.POSITIVE) & (predicted == S.NEGATIVE)) | ((actual == S.NEGATIVE) & (predicted == S.POSITIVE))
    return {
        "n": int(len(actual)),
        "accuracy": 100 * float(accuracy_score(actual, predicted)),
        "f1_macro": 100 * float(f1_score(actual, predicted, average="macro", labels=LABELS, zero_division=0)),
        "balanced_accuracy": 100 * float(balanced_accuracy_score(actual, predicted)),
        "log_loss": float(log_loss(actual, probabilities, labels=LABELS)),
        "ordinal_error": float(np.abs(predicted - actual).mean()),
        "serious_errors": int(flips.sum()),
        "serious_error_rate": 100 * float(flips.mean()),
    }


def per_class(actual, probabilities):
    """Precision, recall, F1 and support for each class."""
    actual, predicted = np.asarray(actual, int), np.asarray(probabilities, float).argmax(axis=1)
    p, r, f, support = precision_recall_fscore_support(actual, predicted, labels=LABELS, zero_division=0)
    return [{"class": S.CLASSES[i], "precision": 100 * float(p[i]), "recall": 100 * float(r[i]), "f1": 100 * float(f[i]),
             "support": int(support[i])} for i in range(3)]


def confusion(actual, probabilities):
    """3×3 counts: rows are the true class, columns the predicted class (Negative, Neutral, Positive)."""
    predicted = np.asarray(probabilities, float).argmax(axis=1)
    return confusion_matrix(np.asarray(actual, int), predicted, labels=LABELS).tolist()


def confidence_of(probabilities):
    return np.asarray(probabilities, float).max(axis=1)


def choose_confidence_threshold(actual, probabilities, target=S.CONFIDENCE_TARGET):
    """The lowest confidence at which predictions at or above it are right at least `target` of the time.

    Measured on (out-of-fold) training predictions. Predictions below it are marked "Not sure" and left to a person.
    If even the most confident predictions miss the target, the threshold is set just above the most confident one
    (nothing is trusted) and `target_met` is False.
    """
    actual, probabilities = np.asarray(actual, int), np.asarray(probabilities, float)
    confidence, correct = confidence_of(probabilities), probabilities.argmax(axis=1) == actual
    order = np.argsort(-confidence, kind="stable")
    levels = np.unique(confidence)[::-1]
    count = np.searchsorted(-confidence[order], -levels, side="right")
    accuracy = np.cumsum(correct[order])[count - 1] / count
    good = np.flatnonzero(accuracy >= target)
    if not len(good):
        return {"confidence": float(confidence.max()) + 1e-9, "target": target, "target_met": False}
    return {"confidence": float(levels[good[-1]]), "target": target, "target_met": True,
            "training_coverage": float(count[good[-1]] / len(actual)), "training_accuracy": float(accuracy[good[-1]])}


def selective_metrics(actual, probabilities, threshold):
    """How many predictions are made automatically at `threshold`, and how often those are right."""
    actual, probabilities = np.asarray(actual, int), np.asarray(probabilities, float)
    sure = confidence_of(probabilities) >= threshold
    correct = probabilities.argmax(axis=1) == actual
    return {"coverage": 100 * float(sure.mean()), "automatic": int(sure.sum()), "not_sure": int((~sure).sum()),
            "accuracy_when_sure": 100 * float(correct[sure].mean()) if sure.any() else 0.0,
            "accuracy_when_not_sure": 100 * float(correct[~sure].mean()) if (~sure).any() else 0.0}


def coverage_curve(actual, probabilities, steps=20):
    """[(share of reviews labelled automatically, accuracy of those)] as the confidence threshold rises."""
    actual, probabilities = np.asarray(actual, int), np.asarray(probabilities, float)
    confidence, correct = confidence_of(probabilities), probabilities.argmax(axis=1) == actual
    points = []
    for t in np.unique(np.quantile(confidence, np.linspace(0, 0.95, steps))):
        keep = confidence >= t
        points.append((round(float(keep.mean()), 4), round(float(correct[keep].mean()), 4)))
    return points


def accuracy_by_confidence(actual, probabilities, edges=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)):
    """Accuracy and count per confidence band (the model should be right more often when it is more sure)."""
    actual, probabilities = np.asarray(actual, int), np.asarray(probabilities, float)
    confidence, correct = confidence_of(probabilities), probabilities.argmax(axis=1) == actual
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high)
        label = f"{low * 100:.0f}-{min(high, 1) * 100:.0f}%"
        rows.append({"band": label, "n": int(mask.sum()), "accuracy": 100 * float(correct[mask].mean()) if mask.any() else None})
    return rows


def length_band(text):
    n = len(text)
    return "Short (under 40 characters)" if n < 40 else "Medium (40-120)" if n <= 120 else "Long (over 120)"


def segment_errors(frame, by):
    """Mistakes per group of reviews (e.g. per length band), most error-prone first."""
    rows = []
    for key, group in frame.groupby(by, observed=True):
        wrong = int((group["actual"] != group["predicted"]).sum())
        rows.append({"group": str(key), "n": int(len(group)), "errors": wrong, "error_rate": 100 * wrong / len(group)})
    return sorted(rows, key=lambda r: (-r["error_rate"], r["group"]))


def summarise_frame(frame):
    """A DataFrame of test reviews (text, actual, predicted) → the segment tables for the report."""
    frame = frame.copy()
    frame["length"] = frame["text"].map(length_band)
    lowered = frame["text"].str.lower()
    frame["negation"] = lowered.map(lambda t: "Has negation (not, never, n't ...)" if (
        re.search(r"n't\b", t) or any(w in NEGATORS for w in re.findall(r"[a-z']+", t))) else "No negation")
    frame["contrast"] = lowered.map(lambda t: "Has but / however" if re.search(r"\b(?:but|however|although)\b", t) else "No contrast")
    return {"length": segment_errors(frame, "length"), "negation": segment_errors(frame, "negation"),
            "contrast": segment_errors(frame, "contrast")}


def top_words(texts, labels, k=10, min_df=5):
    """The most telling words per class: smoothed log-odds of appearing in that class rather than in the others.

    Counted over the TRAINING texts only, once per text (after the negation-aware normalisation, so 'not good'
    shows up as its own feature). Independent of the chosen model.
    """
    labels = np.asarray(labels, int)
    counts = {c: Counter() for c in LABELS}
    for text, label in zip(texts, labels):
        counts[int(label)].update(set(re.findall(r"[a-z_]+", normalise_text(text))) - HIDDEN_TOKENS)
    sizes = {c: max(int((labels == c).sum()), 1) for c in LABELS}
    out = {}
    for c in LABELS:
        others = [o for o in LABELS if o != c]
        n_other = sum(sizes[o] for o in others)
        rows = []
        for word in counts[c]:
            inside, outside = counts[c][word], sum(counts[o].get(word, 0) for o in others)
            if inside + outside < min_df:
                continue
            odds = math.log((inside + 1) / (sizes[c] + 2)) - math.log((outside + 1) / (n_other + 2))
            rows.append({"word": pretty_token(word), "in_class": inside, "elsewhere": outside, "log_odds": round(odds, 3)})
        rows.sort(key=lambda r: r["log_odds"], reverse=True)
        out[S.KEYS[c]] = rows[:k]
    return out


def error_examples(frame, k=8, width=170):
    """The reviews the model got wrong on the test set, most confident mistakes first."""
    wrong = frame[frame["actual"] != frame["predicted"]].sort_values("confidence", ascending=False).head(k)
    return [{"text": t if len(t) <= width else t[:width - 1] + "…", "actual": S.CLASSES[int(a)], "predicted": S.CLASSES[int(p)],
             "confidence": round(float(c), 3)} for t, a, p, c in zip(wrong["text"], wrong["actual"], wrong["predicted"], wrong["confidence"])]


def noise_ceiling(label_noise):
    """For SYNTHETIC data only: with `label_noise` of labels shifted at random, no model can beat this accuracy."""
    return 100 * (1 - label_noise)
