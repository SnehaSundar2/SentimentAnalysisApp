"""Text processing shared by training and prediction.

* normalise_text()  — turns a raw review into a cleaner string for the word-level TF-IDF features: links and
                      @mentions become marker words, emoji become `emopos` / `emoneg`, stretched words ("sooo")
                      are shortened, SHOUTED words are flagged, "n't" becomes "not", and **negation is carried
                      forward**: after "not" the next few words are prefixed (`not good` → `not not_good`),
                      so "not good" and "good" become different features.
* meta_features()   — a few numbers describing the text's shape (length, shouting, exclamations, emoji, negation ...).
* units(), signals() — pieces used to explain a prediction to a person.

Everything here is a plain module-level function so a trained pipeline can be saved and loaded.
"""

import html
import re

import numpy as np

from .wordlist import LEXICON

URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
MENTION = re.compile(r"@\w+")
TAG = re.compile(r"<[^>]{0,200}>")
POSITIVE_EMOJI = set("😊😍👍🎉😀😁🥰👏🙂😃🤩😄💯✨❤")
NEGATIVE_EMOJI = set("😡👎😠😞🤮😢😭🤬😤😒💔😖😩🙁☹")
NEGATORS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "without", "hardly", "barely"}
SCOPE_ENDERS = {"but", "however", "although", "though", "yet", "and"}
INTENSIFIERS = {"very", "really", "extremely", "absolutely", "totally", "incredibly", "so", "super", "utterly", "highly"}
CONTRACTIONS = [(re.compile(r"\bcan't\b", re.I), "can not"), (re.compile(r"\bwon't\b", re.I), "will not"),
                (re.compile(r"\bshan't\b", re.I), "shall not"), (re.compile(r"n't\b", re.I), " not")]
TOKEN = re.compile(r"[a-z0-9_']+|[.,;:!?]")
WORD = re.compile(r"[A-Za-z0-9']+")
CAPS = re.compile(r"\b[A-Z]{3,}\b")
STRETCHED = re.compile(r"(\w)\1{2,}")
SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+|\s+(?=but\b|however\b|although\b)|,\s+(?=but\b|however\b)", re.I)


def pretty_token(token):
    names = {"urltoken": "[link]", "usertoken": "[@mention]", "emopos": "[happy emoji]", "emoneg": "[unhappy emoji]",
             "capsword": "[SHOUTED word]", "elongated": "[stretched word]", "num": "[number]",
             "lexpos": "[positive word]", "lexneg": "[negative word]", "flippos": "[negated positive word]",
             "flipneg": "[negated negative word]"}
    if token in names:
        return names[token]
    return token.replace("not_", "not ") if token.startswith("not_") else token


def strip_markup(text):
    """HTML tags removed and entities (&amp; &#39;) turned back into characters."""
    return html.unescape(TAG.sub(" ", str(text)))


def normalise_text(text):
    """The preprocessing used by the word-level features (see module docstring)."""
    t = str(text)
    t = URL.sub(" urltoken ", t)
    t = MENTION.sub(" usertoken ", t)
    shouted = len(CAPS.findall(t))
    t = t.replace("#", " ").lower()
    for pattern, replacement in CONTRACTIONS:
        t = pattern.sub(replacement, t)
    stretched = bool(STRETCHED.search(t))
    t = STRETCHED.sub(r"\1\1", t)
    pieces = []
    for ch in t:
        pieces.append(" emopos " if ch in POSITIVE_EMOJI else " emoneg " if ch in NEGATIVE_EMOJI else ch)
    t = "".join(pieces)
    out, scope = [], 0
    for token in TOKEN.findall(t):
        if token in ".,;:!?":
            scope = 0
            continue
        if token in NEGATORS:
            out.append(token)
            scope = 3
        elif token in SCOPE_ENDERS:
            out.append(token)
            scope = 0
        elif scope and token.isalpha() and token not in INTENSIFIERS:
            out.append("not_" + token)
            scope -= 1
            if token in LEXICON:                       # a negated praise word is criticism, and the other way round
                out.append("flippos" if LEXICON[token] > 0 else "flipneg")
        else:
            out.append("num" if token.isdigit() else token)
            if token in LEXICON:                       # one shared feature for every known praise / criticism word
                out.append("lexpos" if LEXICON[token] > 0 else "lexneg")
    out += ["capsword"] * min(shouted, 2) + (["elongated"] if stretched else [])
    return " ".join(out)


META_NAMES = ["length", "exclamations", "questions", "capitals", "caps_words", "happy_emoji", "unhappy_emoji",
              "negations", "stretched", "contrast", "words", "praise_words", "criticism_words", "negated_praise",
              "negated_criticism"]


def meta_features(frame):
    """Fifteen numbers between 0 and 1 describing the shape of each text (input: DataFrame with a 'text' column)."""
    rows = []
    for text in frame["text"].astype(str):
        normalised = normalise_text(text).split()
        letters = [c for c in text if c.isalpha()]
        words = text.split()
        lowered = " " + " ".join(re.findall(r"[a-z']+", text.lower())) + " "
        negations = sum(f" {w} " in lowered for w in NEGATORS) + len(re.findall(r"n't\b", text.lower()))
        rows.append([
            min(len(text) / 500, 1.0),
            min(text.count("!") / 5, 1.0),
            min(text.count("?") / 3, 1.0),
            sum(c.isupper() for c in letters) / max(len(letters), 1),
            min(len(CAPS.findall(text)) / 3, 1.0),
            min(sum(c in POSITIVE_EMOJI for c in text) / 2, 1.0),
            min(sum(c in NEGATIVE_EMOJI for c in text) / 2, 1.0),
            min(negations / 3, 1.0),
            1.0 if STRETCHED.search(text.lower()) else 0.0,
            1.0 if any(f" {w} " in lowered for w in ("but", "however", "although")) else 0.0,
            min(len(words) / 80, 1.0),
            min(normalised.count("lexpos") / 3, 1.0),
            min(normalised.count("lexneg") / 3, 1.0),
            min(normalised.count("flippos") / 2, 1.0),
            min(normalised.count("flipneg") / 2, 1.0),
        ])
    return np.asarray(rows, float)


def units(text, limit=40):
    """The pieces of a text that can be removed one at a time to see their effect: links, words and emoji."""
    found, seen = [], set()
    for match in URL.finditer(text):
        piece = match.group(0).rstrip(".,;:!?)")
        if piece and piece.lower() not in seen:
            seen.add(piece.lower())
            found.append(("link", piece))
    remainder = text
    for _, piece in found:
        remainder = remainder.replace(piece, " ")
    for match in re.finditer(r"[A-Za-z0-9']+|[^\sA-Za-z0-9'.,;:!?\"()\-]", remainder):
        piece = match.group(0).strip("'")
        if len(piece) >= 1 and (len(piece) >= 2 or not piece.isascii()) and piece.lower() not in seen:
            seen.add(piece.lower())
            found.append(("word", piece))
    return found[:limit]


def remove_unit(text, kind, piece):
    """The text with one piece taken out."""
    if kind == "word" and piece.isascii():
        out = re.sub(r"(?<![\w'])" + re.escape(piece) + r"(?![\w'])", " ", text, flags=re.I)
    else:
        out = text.replace(piece, " ")
    return " ".join(out.split())


def split_parts(text, minimum_words=2):
    """Split a text into its sentences and 'but' clauses, dropping fragments that are too short to judge."""
    parts = [p.strip(" ,") for p in SENTENCE_BREAK.split(text) if p and p.strip(" ,")]
    return [p for p in parts if len(WORD.findall(p)) >= minimum_words]


def signals(text):
    """Plain-English observations about the text's style (only those that apply)."""
    out = []
    negated = re.findall(r"\b(?:not|never|no|cannot|can't|won't|don't|didn't|isn't|wasn't|aren't|weren't|doesn't|couldn't|wouldn't)\s+\w+", text, re.I)
    if negated:
        out.append("Negation found: " + ", ".join(f"“{n}”" for n in negated[:3]))
    if re.search(r"\b(?:but|however|although)\b", text, re.I):
        out.append("Contrast (but / however): mixed feelings are likely")
    happy = sum(c in POSITIVE_EMOJI for c in text)
    unhappy = sum(c in NEGATIVE_EMOJI for c in text)
    if happy:
        out.append(f"Happy emoji ({happy})")
    if unhappy:
        out.append(f"Unhappy emoji ({unhappy})")
    caps = CAPS.findall(text)
    if len(caps) >= 2:
        out.append(f"Several SHOUTED words ({len(caps)})")
    if text.count("!") >= 3:
        out.append("Many exclamation marks")
    if STRETCHED.search(text.lower()):
        out.append("Stretched words (sooo, reaaally)")
    if text.count("?") >= 2:
        out.append("Several questions")
    return out
