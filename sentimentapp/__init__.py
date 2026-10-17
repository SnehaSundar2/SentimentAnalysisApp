"""Sentiment Analysis App.

An end-to-end machine-learning project for text: data generation and cleaning, text normalisation that
understands negation, emoji and shouting, a rule-based baseline, model comparison with cross-validation,
a confidence threshold that sends doubtful reviews to a person, honest evaluation on a held-out test set,
word-level explanations, and prediction interfaces (command-line menu, local web form, JSON API, batch CSV).
"""

import logging

__version__ = "1.0.0"

logging.getLogger("sentimentapp").addHandler(logging.NullHandler())
