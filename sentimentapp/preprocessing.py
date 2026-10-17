"""Data preprocessing: loading, cleaning and the scikit-learn feature transformer.

Two kinds of preprocessing, kept deliberately separate:

* clean()           — fixes or removes BAD DATA (missing text, unreadable labels, leftover HTML, stray characters,
                      duplicates, contradictions) before the split. Every rule looks at one text or at exact repeats;
                      none learns statistics from other rows.
* build_features()  — LEARNS from data (which words exist, how rare each is). It lives inside the model pipeline and
                      is fitted on the TRAINING texts only, so nothing from the test set leaks into training.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from . import schema as S
from .exceptions import DataError
from .textproc import meta_features, normalise_text, strip_markup

log = logging.getLogger("sentimentapp")

MAX_FILE_BYTES = 50_000_000
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


# ====================================================================== loading
def normalise_headers(columns):
    return [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in columns]


def load_dataset(path):
    """Read a CSV with a label (or star rating) column and a text column; several common names are understood."""
    path = Path(str(path).strip().strip('"'))
    if not path.exists() or not path.is_file():
        raise DataError(f"Dataset file not found: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DataError(f"{path.name} is larger than {MAX_FILE_BYTES // 1_000_000} MB.")
    note = None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise DataError(f"{path.name} is empty.") from None
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="cp1252", dtype=str, keep_default_na=False)
        except (UnicodeDecodeError, pd.errors.ParserError, OSError):
            df = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)
        note = "File was not UTF-8 text; it was read as Windows-1252/Latin-1 (accented characters may differ)."
    except (pd.errors.ParserError, OSError) as exc:
        raise DataError(f"Could not read {path.name}: {exc}") from None
    df.columns = normalise_headers(df.columns)
    label_col = next((c for c in S.LABEL_COLUMNS if c in df.columns), None)
    rating_col = None if label_col else next((c for c in S.RATING_COLUMNS if c in df.columns), None)
    text_col = next((c for c in S.TEXT_COLUMNS if c in df.columns and c != label_col), None)
    if text_col is None or (label_col is None and rating_col is None):
        needs = ("a text column" if text_col is None else "") + (" and " if text_col is None and label_col is None and rating_col is None else "") \
            + ("a label column (or star ratings)" if label_col is None and rating_col is None else "")
        raise DataError(f"{path.name} needs {needs}. Text column names understood: {', '.join(S.TEXT_COLUMNS)}; "
                        f"label column names understood: {', '.join(S.LABEL_COLUMNS)}; "
                        f"rating column names understood: {', '.join(S.RATING_COLUMNS)}.")
    if len(df) < S.MIN_ROWS:
        raise DataError(f"{path.name} has only {len(df)} rows; at least {S.MIN_ROWS} are needed to train a "
                        "reliable model.")
    if rating_col:
        out = pd.DataFrame({"label": df[rating_col].map(lambda v: S.CLASSES[s] if (s := S.stars_to_label(v)) is not None else None),
                            "text": df[text_col]})
        note = ((note + " ") if note else "") + f"Labels were taken from the '{rating_col}' column (1-2 stars negative, 3 neutral, 4-5 positive)."
    else:
        out = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"}).copy()
    out.attrs["note"] = note
    return out


# ====================================================================== cleaning
@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    steps: list = field(default_factory=list)        # (label, count, outcome)
    notes: list = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)

    def add(self, label, count, outcome):
        if count:
            self.steps.append((label, int(count), outcome))

    @property
    def rows_dropped(self):
        return self.rows_in - self.rows_out

    def lines(self):
        out = [f"{self.rows_in} rows read → {self.rows_out} kept ({self.rows_dropped} removed)"]
        out += [f"  • {label}: {count} — {outcome}" for label, count, outcome in self.steps]
        out += [f"  ! {note}" for note in self.notes]
        if self.class_counts:
            out.append("  Kept: " + ", ".join(f"{self.class_counts[k]} {k}" for k in S.KEYS))
        return out

    def to_dict(self):
        return {"rows_in": self.rows_in, "rows_out": self.rows_out,
                "steps": [{"label": a, "count": b, "outcome": c} for a, b, c in self.steps],
                "notes": self.notes, "class_counts": self.class_counts}


def tidy_text(value):
    """HTML removed, control characters removed, whitespace collapsed. Anything that is not text gives None."""
    if not isinstance(value, str):
        return None
    return " ".join(CONTROL.sub("", strip_markup(value)).split())


def clean(df):
    """Return (clean_df with int `label` 0/1/2 and str `text`, CleaningReport). The input is not modified."""
    note = df.attrs.get("note") if hasattr(df, "attrs") else None
    df = df[["label", "text"]].copy()
    report = CleaningReport(rows_in=len(df))
    if note:
        report.notes.append(note)

    # 1. the text
    plain = df["text"].map(lambda v: " ".join(CONTROL.sub("", v).split()) if isinstance(v, str) else None)
    tidy = df["text"].map(tidy_text)
    missing = tidy.isna()
    report.add("Rows with a missing or non-text review", missing.sum(), "removed")
    df, tidy, plain = df[~missing], tidy[~missing], plain[~missing]
    report.add("Reviews with leftover HTML tags or entities", (tidy != plain).sum(), "markup removed")
    report.add("Reviews with stray whitespace or control characters", ((plain != df["text"]) & (tidy == plain)).sum(), "tidied")
    df["text"] = tidy
    tiny = df["text"].map(lambda t: sum(c.isalnum() for c in t) < S.MIN_ALNUM)
    report.add("Reviews empty or too short to judge", tiny.sum(), f"removed (need at least {S.MIN_ALNUM} letters or digits)")
    df = df[~tiny]
    long = df["text"].str.len() > S.MAX_TEXT
    report.add(f"Reviews longer than {S.MAX_TEXT} characters", long.sum(), "truncated")
    df.loc[long, "text"] = df.loc[long, "text"].str[:S.MAX_TEXT]

    # 2. the label
    original = df["label"]
    label = original.map(S.canonical_label)
    standard = label.map(lambda v: S.CLASSES[int(v)] if v == v and v is not None else None)
    report.add("Labels written in another style", (label.notna() & (original.astype(str) != standard)).sum(),
               "standardised to Negative / Neutral / Positive (e.g. 'pos', '+1', 'bad')")
    unknown = label.isna()
    report.add("Rows with a missing or unrecognised label", unknown.sum(), "removed (e.g. 'mixed', 'unsure')")
    df = df[~unknown].copy()
    df["label"] = label[~unknown].astype(int)

    # 3. repeats: the same text (ignoring capitals) must not appear in both training and test data
    key = df["text"].str.lower()
    contradictory = df.groupby(key)["label"].transform("nunique") > 1
    report.add("Same review given different labels", contradictory.sum(), "removed (no label can be trusted)")
    df, key = df[~contradictory], key[~contradictory]
    repeat = key.duplicated()
    report.add("Duplicate reviews", repeat.sum(), "removed (kept the first copy, ignoring capitals)")
    df = df[~repeat].reset_index(drop=True)

    counts = df["label"].value_counts()
    report.class_counts = {k: int(counts.get(i, 0)) for i, k in enumerate(S.KEYS)}
    report.rows_out = len(df)
    log.info("Cleaning: %d → %d rows", report.rows_in, report.rows_out)
    if len(df) < S.MIN_ROWS:
        raise DataError(f"Only {len(df)} usable rows remain after cleaning (need {S.MIN_ROWS}). "
                        "Check that the label and text columns are the right way round.")
    for key_name, count in report.class_counts.items():
        if count < S.MIN_PER_CLASS:
            raise DataError(f"Only {count} {key_name} reviews remain after cleaning; at least {S.MIN_PER_CLASS} of each "
                            "kind are needed to train and test a model.")
    return df, report


# ====================================================================== features
def build_features():
    """Word and character TF-IDF features plus eleven text-shape numbers, all non-negative."""
    word = TfidfVectorizer(preprocessor=normalise_text, ngram_range=(1, 2), min_df=2, max_features=30000,
                           sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=30000, sublinear_tf=True)
    return ColumnTransformer([
        ("word", word, "text"),
        ("char", char, "text"),
        ("meta", FunctionTransformer(meta_features), ["text"]),
    ])


def split_xy(df):
    return df[["text"]].copy(), df["label"].astype(int).copy()
