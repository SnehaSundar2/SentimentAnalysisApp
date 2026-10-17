"""Validation of the text to analyse (prediction input).

validate_text() returns a tidy text plus any warnings, or raises ValidationError with a clear reason.
"""

from . import schema as S
from .exceptions import ValidationError
from .preprocessing import tidy_text


def validate_text(value):
    """Return (tidy_text, warnings). Raises ValidationError(message, errors={'text': message})."""
    if value is None or (isinstance(value, float) and value != value):
        raise ValidationError("Enter some text to analyse.", {"text": "Enter some text to analyse."})
    if not isinstance(value, str):
        raise ValidationError("The text must be a string of characters.", {"text": "The text must be a string of characters."})
    text = tidy_text(value)
    if not text:
        raise ValidationError("Enter some text to analyse.", {"text": "Enter some text to analyse."})
    if len(text) > S.MAX_TEXT:
        problem = f"The text is too long ({len(text):,} characters; the limit is {S.MAX_TEXT:,})."
        raise ValidationError(problem, {"text": problem})
    warnings = []
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    if len(words) < 3:
        warnings.append("Very short texts are hard to judge; treat this result with care.")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(ord(c) > 127 for c in letters) / len(letters) > 0.3:
        warnings.append("Most of this text is not in English letters; the model was trained on English reviews, "
                        "so the result is unreliable.")
    return text, warnings


def validate_port(value):
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError("Port must be a whole number.") from None
    if not 1024 <= port <= 65535:
        raise ValidationError("Port must be between 1024 and 65535.")
    return port


def validate_int(value, field, low, high):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a whole number.") from None
    if not low <= number <= high:
        raise ValidationError(f"{field} must be between {low} and {high}.")
    return number
