"""A rule-based sentiment scorer: the baseline that does NOT learn from data.

A hand-made word list gives each word a score; negation flips the next few words, intensifiers strengthen
them, shouting and exclamation marks add emphasis, and "but" gives the second half more weight. It is wrapped
in a scikit-learn style class so it can be cross-validated next to the trained models. Comparing it with
them shows what learning from examples adds.

The list is deliberately small and general (about 130 words, written separately from the data generator), like a real
off-the-shelf lexicon: it knows common words but not every domain-specific phrase.
"""

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from . import schema as S
from .textproc import CAPS, NEGATIVE_EMOJI, NEGATORS, POSITIVE_EMOJI, STRETCHED, TOKEN, CONTRACTIONS
from .wordlist import CONTRAST, DAMPEN, INTENSIFY, LEXICON

NEGATION_FACTOR = -0.74


def lexicon_score(text):
    """A score between -1 (very negative) and +1 (very positive) for one text."""
    original = str(text)
    shouted = len(CAPS.findall(original))
    t = original.lower()
    for pattern, replacement in CONTRACTIONS:
        t = pattern.sub(replacement, t)
    t = STRETCHED.sub(r"\1\1", t)
    tokens = TOKEN.findall(t)
    total, weight_after_contrast, scope, boost = 0.0, 1.0, 0, 1.0
    for token in tokens:
        if token in ".,;:!?":
            scope, boost = 0, 1.0
            continue
        if token in CONTRAST:
            weight_after_contrast, scope, boost = 1.5, 0, 1.0
            continue
        if token in NEGATORS:
            scope = 3
            continue
        if token in INTENSIFY:
            boost = INTENSIFY[token]
            continue
        if token in DAMPEN:
            boost = DAMPEN[token]
            continue
        value = LEXICON.get(token)
        if value is not None:
            value *= boost
            if scope:
                value *= NEGATION_FACTOR
            total += value * weight_after_contrast
            boost = 1.0
        if scope:
            scope -= 1
    total += sum(1.5 for ch in original if ch in POSITIVE_EMOJI) - sum(1.5 for ch in original if ch in NEGATIVE_EMOJI)
    if total:
        total += np.sign(total) * min(original.count("!"), 3) * 0.3 + np.sign(total) * min(shouted, 2) * 0.4
    return total / np.sqrt(total * total + 8.0)           # squash into (-1, 1) like VADER's compound score


class LexiconClassifier(BaseEstimator, ClassifierMixin):
    """Scores each text with the word list; needs raw text (a DataFrame with a 'text' column)."""

    needs_raw_text = True

    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.classes_ = np.array([S.NEGATIVE, S.NEUTRAL, S.POSITIVE])
        return self

    def _scores(self, X):
        texts = X["text"] if hasattr(X, "columns") else X
        return np.array([lexicon_score(t) for t in texts])

    def predict_proba(self, X):
        c = self._scores(X)
        logits = np.column_stack([-3.2 * c, 1.1 - 3.2 * np.abs(c), 3.2 * c])
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        c = self._scores(X)
        return np.where(c > self.threshold, S.POSITIVE, np.where(c < -self.threshold, S.NEGATIVE, S.NEUTRAL))
