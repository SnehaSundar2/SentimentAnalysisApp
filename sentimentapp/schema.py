"""Shared constants: classes, column names, limits, confidence target and example texts.

Training-data cleaning and prediction-input validation both read from here, so the two can
never disagree about what a valid text or label looks like.
"""

NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2
CLASSES = ["Negative", "Neutral", "Positive"]
KEYS = ["negative", "neutral", "positive"]

# Column names accepted in a dataset (after lower-casing); the first match wins.
LABEL_COLUMNS = ["label", "sentiment", "class", "category", "polarity", "target", "v1"]
TEXT_COLUMNS = ["text", "review", "comment", "message", "tweet", "content", "body", "sentence", "v2"]
RATING_COLUMNS = ["rating", "stars", "star_rating", "score"]     # used only when there is no label column

LABEL_WORDS = {
    POSITIVE: {"positive", "pos", "+", "+1", "1", "good", "happy", "favorable", "favourable", "praise"},
    NEUTRAL: {"neutral", "neu", "0", "ok", "okay", "indifferent"},
    NEGATIVE: {"negative", "neg", "-", "-1", "bad", "unhappy", "unfavorable", "unfavourable", "complaint"},
}

MAX_TEXT = 5000          # longest text accepted or kept
MIN_ALNUM = 2            # a text needs at least this many letters or digits to be judged
MIN_ROWS = 150
MIN_PER_CLASS = 25       # each class needs at least this many reviews to train and test on

# Predictions the model is less sure about than this are marked "Not sure" for a person to read. The threshold
# is chosen on training data so that the predictions above it are right at least this often.
CONFIDENCE_TARGET = 0.95

# Messages offered as one-click examples in the web form (made up for this project)
EXAMPLES = {
    "positive": ("Positive", "Absolutely loved the new update, the app is so much faster now! \U0001F60D Would recommend."),
    "negative": ("Negative", "Terrible experience. The delivery was late and the packaging was damaged, never buying again."),
    "neutral": ("Neutral", "I ordered the blue one on Monday and it arrived after three days."),
    "mixed": ("Mixed", "The food was great but the service was painfully slow."),
    "negation": ("Negation", "The battery life is not good at all and I'm not impressed with the screen."),
    "sarcasm": ("Sarcasm", "Oh great, another crash right when I needed it."),
}


def canonical_label(value):
    """'Positive ' / 'POS' / '+1' -> 2;  'neu' / 0 -> 1;  'bad' / -1 -> 0;  anything else (or missing) -> None."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value == int(value):
        value = int(value)
    key = " ".join(str(value).split()).lower()
    if isinstance(value, int) and value > 0:
        key = "+1"
    for label, words in LABEL_WORDS.items():
        if key in words:
            return label
    return None


def stars_to_label(value):
    """1-2 stars -> Negative, 3 -> Neutral, 4-5 -> Positive. Anything else (or missing) -> None."""
    try:
        stars = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if stars != stars or not 1 <= stars <= 5:
        return None
    return NEGATIVE if stars < 2.5 else NEUTRAL if stars < 3.5 else POSITIVE


def score_from_probabilities(probabilities):
    """A single number from -100 (certainly negative) to +100 (certainly positive)."""
    return 100.0 * (float(probabilities[POSITIVE]) - float(probabilities[NEGATIVE]))


def tone(score):
    """Words for a score: 'Strongly negative' ... 'Strongly positive'."""
    if score <= -60:
        return "Strongly negative"
    if score <= -20:
        return "Negative"
    if score < 20:
        return "Neutral or mixed"
    if score < 60:
        return "Positive"
    return "Strongly positive"
