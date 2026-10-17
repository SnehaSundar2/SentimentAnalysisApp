"""Analysing texts: sentiment, confidence, a score, the words behind it, the parts of a mixed review,
similar known reviews and batch files."""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import persistence
from . import schema as S
from . import textproc
from .exceptions import DataError, NotTrainedError, ValidationError
from .preprocessing import normalise_headers
from .validators import validate_text

MAX_BATCH_ROWS = 5000
EFFECT_SHOWN = 0.25       # a piece must change the log-odds of the predicted sentiment by at least this much (odds ×1.3)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class Analysis:
    text: str
    probabilities: dict            # {"Negative": p, "Neutral": p, "Positive": p}
    label: str                     # the most likely sentiment
    confidence: float
    score: float                   # -100 .. +100
    tone: str
    needs_review: bool             # confidence below the threshold: "Not sure"
    confidence_threshold: float
    warnings: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    supporting: list = field(default_factory=list)      # [{"word", "effect", "odds"}] pieces that support the label
    opposing: list = field(default_factory=list)        # pieces that argue against it
    effects: dict = field(default_factory=dict)         # lower-case piece -> change in log-odds (used to highlight text)
    parts: list = field(default_factory=list)           # [{"text", "label", "confidence"}] for sentences / "but" clauses
    similar: list = field(default_factory=list)

    @property
    def verdict(self):
        return "Not sure" if self.needs_review else self.label

    def to_dict(self):
        return {"sentiment": self.verdict, "most_likely": self.label, "confidence": round(self.confidence, 4),
                "score": round(self.score, 1), "tone": self.tone, "needs_review": self.needs_review,
                "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
                "confidence_threshold": round(self.confidence_threshold, 4), "warnings": self.warnings,
                "signals": self.signals, "words_supporting": self.supporting, "words_opposing": self.opposing,
                "parts": self.parts, "similar_reviews": self.similar}


class Predictor:
    def __init__(self, loaded):
        self.model = loaded.pipeline
        self.meta = loaded.metadata
        self.reference = loaded.reference
        self.load_warnings = list(loaded.warnings)
        self.threshold = self.meta["confidence"]["confidence"]
        self._reference_matrix = None

    @classmethod
    def from_directory(cls, directory):
        return cls(persistence.load(directory))

    # ------------------------------------------------------------------ core
    def _proba(self, texts):
        return np.asarray(self.model.predict_proba(pd.DataFrame({"text": list(texts)})), float)

    def analyze(self, raw, explain=True, parts=True, similar=True):
        """Analyse one text. Raises ValidationError for an unusable text."""
        text, warnings = validate_text(raw)
        p = self._proba([text])[0]
        label_index = int(p.argmax())
        score = S.score_from_probabilities(p)
        result = Analysis(text, {S.CLASSES[i]: float(p[i]) for i in range(3)}, S.CLASSES[label_index], float(p.max()), score,
                          S.tone(score), bool(p.max() < self.threshold), self.threshold, warnings, textproc.signals(text))
        if explain:
            self._explain(result, label_index, p)
        if parts:
            result.parts = self._parts(text)
        if similar:
            result.similar = self._similar(text)
        return result

    # ------------------------------------------------------------------ explanations
    def _explain(self, result, label_index, p, top=6):
        """Which parts of the text push towards (or against) the predicted sentiment.

        Each piece (a word, an emoji, a link) is taken out in turn and the text is re-scored. The effect is the change
        in the LOG-ODDS of the predicted sentiment, so it still shows up when the model is already 99% sure.
        exp(effect) is the factor by which that piece multiplies the odds of that sentiment.
        """
        pieces = textproc.units(result.text)
        if not pieces:
            return
        variants = [textproc.remove_unit(result.text, kind, piece) for kind, piece in pieces]
        without = self._proba(variants)[:, label_index]
        effects = _logit(p[label_index]) - _logit(without)
        rows = []
        for (kind, piece), effect in zip(pieces, effects):
            rows.append({"word": ("[link] " + piece) if kind == "link" else piece, "effect": round(float(effect), 3),
                         "odds": round(float(np.exp(effect)), 2)})
            result.effects[piece.lower()] = float(effect)
        result.supporting = sorted((r for r in rows if r["effect"] >= EFFECT_SHOWN), key=lambda r: -r["effect"])[:top]
        result.opposing = sorted((r for r in rows if r["effect"] <= -EFFECT_SHOWN), key=lambda r: r["effect"])[:top]

    def _parts(self, text):
        """The sentiment of each sentence or 'but' clause, so mixed reviews show their two sides."""
        pieces = textproc.split_parts(text)
        if len(pieces) < 2:
            return []
        probabilities = self._proba(pieces)
        return [{"text": piece, "label": S.CLASSES[int(p.argmax())], "confidence": round(float(p.max()), 3)}
                for piece, p in zip(pieces, probabilities)]

    def _reference_vectors(self):
        if self._reference_matrix is None:
            vectorizer = self.model.named_steps["features"].named_transformers_["word"]
            self._reference_matrix = vectorizer.transform(self.reference["text"])
        return self._reference_matrix

    def _similar(self, text, n=3):
        """The known training reviews that use the most similar words (none when the model has no word features)."""
        if "features" not in self.model.named_steps:
            return []
        vectorizer = self.model.named_steps["features"].named_transformers_["word"]
        sims = (self._reference_vectors() @ vectorizer.transform([text]).T).toarray().ravel()
        rows = []
        for idx in np.argsort(-sims)[:n]:
            if sims[idx] < 0.05:
                break
            known = self.reference.iloc[int(idx)]
            shown = known["text"] if len(known["text"]) <= 140 else known["text"][:139] + "…"
            rows.append({"text": shown, "label": S.CLASSES[int(known["label"])], "similarity": round(float(sims[idx]), 2)})
        return rows

    # ------------------------------------------------------------------ batch
    def predict_csv(self, in_path, out_path):
        """Analyse every row of a CSV. Bad rows are reported (with line numbers), good rows still get a result.

        Returns (ok_count, errors, summary). The output CSV has the original columns plus sentiment, confidence, score,
        p_negative, p_neutral, p_positive and a status column ('Not sure' marks low-confidence rows for a person).
        """
        in_path, out_path = Path(str(in_path).strip().strip('"')), Path(str(out_path).strip().strip('"'))
        if not in_path.exists():
            raise DataError(f"File not found: {in_path}")
        if out_path.exists():
            raise DataError(f"{out_path} already exists; choose a new file name.")
        try:
            df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            raise DataError(f"{in_path.name} is empty.") from None
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            raise DataError(f"Could not read {in_path.name}: {exc}") from None
        df.columns = normalise_headers(df.columns)
        if len(df) > MAX_BATCH_ROWS:
            raise DataError(f"{in_path.name} has {len(df)} rows; the limit is {MAX_BATCH_ROWS}.")
        column = next((c for c in S.TEXT_COLUMNS if c in df.columns), None)
        if column is None:
            raise DataError(f"{in_path.name} needs a text column (any of: {', '.join(S.TEXT_COLUMNS)}).")

        good, errors, status = [], [], []
        for position, value in enumerate(df[column]):
            line = position + 2   # header is line 1
            try:
                text, warnings = validate_text(value)
                good.append((position, text))
                status.append("ok" + (f" ({'; '.join(w.split(';')[0].split('.')[0].lower() for w in warnings)})" if warnings else ""))
            except ValidationError as exc:
                errors.append(f"Line {line}: {exc}")
                status.append(f"error: {exc}")
        out = df.copy()
        for col in ("sentiment", "confidence", "score", "p_negative", "p_neutral", "p_positive"):
            out[col] = ""
        summary = {"analysed": len(good), "negative": 0, "neutral": 0, "positive": 0, "not_sure": 0, "average_score": 0.0}
        if good:
            probabilities = self._proba([text for _, text in good])
            scores = []
            for (i, _), p in zip(good, probabilities):
                label = S.CLASSES[int(p.argmax())]
                unsure = bool(p.max() < self.threshold)
                score = S.score_from_probabilities(p)
                scores.append(score)
                out.loc[i, "sentiment"] = "Not sure" if unsure else label
                out.loc[i, "confidence"] = f"{p.max():.3f}"
                out.loc[i, "score"] = f"{score:.0f}"
                for k, key in enumerate(S.KEYS):
                    out.loc[i, f"p_{key}"] = f"{p[k]:.3f}"
                if unsure:
                    summary["not_sure"] += 1
                    status[i] = status[i] + f" (leaning {label.lower()}: please read)"
                else:
                    summary[label.lower()] += 1
            summary["average_score"] = float(np.mean(scores))
        out["status"] = status
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        except OSError as exc:
            raise DataError(f"Could not write {out_path}: {exc.strerror or exc}") from exc
        return len(good), errors, summary


def highlight_pieces(text, effects, minimum=EFFECT_SHOWN):
    """Split a text into [(chunk, effect_or_None)] so a page can mark the words that support the predicted sentiment.

    Links (keys that are not a plain word) are matched whole; everything else word by word; emoji one by one.
    """
    long_keys = sorted((k for k in effects if " " in k or "/" in k or "." in k), key=len, reverse=True)
    pattern = "|".join([re.escape(k) for k in long_keys] + [r"[A-Za-z0-9']+", r"."])
    chunks = []
    for match in re.finditer(pattern, text, flags=re.I | re.S):
        chunk = match.group(0)
        effect = effects.get(chunk.lower().strip("'"))
        chunks.append((chunk, effect if effect is not None and effect >= minimum else None))
    return chunks


def require_predictor(directory):
    """Load the saved model or raise NotTrainedError with instructions."""
    if not persistence.is_trained(directory):
        raise NotTrainedError(f"No trained model found in '{directory}'. Train one first "
                              "(menu option 'Train a model', or: python main.py --train).")
    return Predictor.from_directory(directory)
