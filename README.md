# Sentiment Analysis App

A complete machine-learning project that **reads a review or comment and says how the writer feels**:
Negative, Neutral or Positive, with a score from −100 to +100, the words that led to the answer, a verdict
for each part of a mixed review, and an honest **"Not sure"** when it is not confident.

It covers the whole workflow the internship brief asks for:

| Brief | Where it lives |
|---|---|
| **Data preprocessing** | `sentimentapp/textproc.py`, `preprocessing.py` – text normalisation that understands negation, emoji, shouting and stretched words; HTML stripping; dataset loading (labels or star ratings) and cleaning with an audit log; TF-IDF features |
| **Model training** | `sentimentapp/models.py`, `training.py` – three learned models, a rule-based word list and a baseline compared by stratified cross-validation, tuning, a leak-free split, a confidence threshold chosen on training data |
| **Evaluation** | `sentimentapp/evaluation.py`, `report.py` – accuracy, macro-F1, per-class scores, confusion matrix, serious (positive ↔ negative) errors, confidence analysis, the actual mistakes, HTML report |
| **Prediction interface** | `cli.py` (menu), `web.py` (web form + JSON API), `main.py` (command line), batch CSV |

Developed and submitted by **Sneha S** (Intern ID SKX-2026-6425) · Skyrovix virtual internship ·
AI & Machine Learning · due 15 Nov 2026.

> **Data note.** Real review collections are licensed and carry real people's words, so the project generates
> realistic but **artificial** reviews (restaurants, hotels, products, apps, films, services) with negation, sarcasm,
> mixed feelings, emoji, shouting, typos and a few percent of labels where "annotators disagree". Results show the
> *workflow* works; real reviews are messier and real accuracy would be lower. To use a real dataset, train with
> `--data your_file.csv` (columns such as `text,label` or `review,rating`).

## Features

- **Text preprocessing that understands how people write.** "n't" becomes "not", and negation is *carried forward*
  (`not good` → `not not_good flippos`), so "not good" and "good" are different features. A small word list marks praise and
  criticism words (and their negated forms) so that "not decent" is understood even if that exact phrase was never seen.
  Emoji, @mentions, links, SHOUTED words and stretched words ("sooo") are handled too.
- **Cleaning with an audit trail.** Standardises label spellings (`pos`, `+1`, `bad`), strips leftover HTML and entities, tidies
  whitespace and control characters, removes empty/too-short texts, duplicates and texts given different labels, and says
  exactly what it did. Star ratings (1–2 negative, 3 neutral, 4–5 positive) are accepted when there is no label column.
- **Fair model comparison.** Baseline (most common class), a **rule-based word list that does no learning**, Naive Bayes,
  Logistic Regression and Linear SVM, ranked by cross-validated **macro-F1** so the small Neutral class counts as much as the others.
- **No data leakage.** The test set is used once; features are fitted on training reviews only; no review appears on both sides of the
  split; the confidence threshold is chosen on out-of-fold *training* predictions.
- **"Not sure" instead of a wild guess.** Predictions below a confidence threshold, chosen so that the ones above it are right 95% of the
  time, are marked for a person to read.
- **Explanations.** Every word, emoji and link is removed in turn and the text re-scored (in log-odds, so it still works at 99% confidence);
  the page marks the words that support the answer and shows how much (`odds ×24`). Plus the sentiment of each sentence / "but" clause,
  plain-English style notes, and the three most similar known reviews.
- **Safe by design.** Input is validated with a reason per problem; saved models are verified by SHA-256 before loading; the web server only
  answers to localhost, limits request size and escapes everything, including the text itself.

## Results (synthetic data, 576 unseen test reviews)

| Model (5-fold CV on training reviews) | Macro-F1 | Accuracy | Log loss (lower is better) |
|---|---|---|---|
| **Linear SVM (tuned)** ← chosen | **88.1%** | 89.8% | 0.371 |
| Linear SVM | 88.0% | 89.9% | 0.371 |
| Logistic Regression | 85.6% | 87.7% | 0.390 |
| Naive Bayes | 77.4% | 80.2% | 0.668 |
| Rule-based word list (no learning) | 74.9% | 77.0% | 0.660 |
| Baseline (most common class) | 19.2% | 40.6% | 1.055 |

On the test set: **accuracy 88.5%**, **macro-F1 86.2%** (always the most common class: 40.6% accuracy). Per class: Negative F1 92.3%,
Positive 91.3%, Neutral 75.0%. Only **10 of 576 (1.7%)** are serious mistakes (positive called negative or the reverse); most errors are
between a polar class and Neutral.

| Real ↓ / predicted → | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 210 | 11 | 6 |
| Neutral | 14 | 84 | 17 |
| Positive | 4 | 14 | 216 |

Predictions with at least 65% confidence are made automatically for 84% of reviews and are right 94% of the time; the other 91 are marked
"Not sure". Reviews containing "but / however" are the hardest (39.6% wrong against 6.2% without): "great food but slow service" has no single
right answer. Learning from examples adds about 11 points of macro-F1 over the rule-based word list. Training accuracy is 94%, so there is no
serious overfitting.

## Project structure

```
SentimentAnalysisApp/
├── main.py                  entry point (menu or command line)
├── requirements.txt
├── sentimentapp/
│   ├── schema.py            classes, column names, limits, score and tone, example texts
│   ├── textproc.py          normalisation (negation, emoji ...), text-shape features, explanation helpers
│   ├── wordlist.py          the small hand-made praise / criticism word list
│   ├── lexicon.py           the rule-based scorer (no learning), usable as a model
│   ├── datagen.py           synthetic reviews with negation, sarcasm, mixed feelings, label noise and data problems
│   ├── preprocessing.py     loading, cleaning, TF-IDF + text-shape features
│   ├── models.py            candidate models, cross-validation, tuning
│   ├── evaluation.py        metrics, confidence threshold, curves, top words, mistakes
│   ├── training.py          the workflow end to end
│   ├── persistence.py       save / load with SHA-256 checks
│   ├── validators.py        validation of the text to analyse
│   ├── predictor.py         sentiment, score, explanation, parts, similar reviews, batch CSV
│   ├── report.py            text and HTML (SVG charts) reports
│   ├── web.py               local web form and JSON API
│   ├── cli.py               interactive menu
│   └── exceptions.py        custom errors
├── sample_data/             reviews_dirty.csv (practice cleaning), reviews_to_analyze.csv (batch example)
├── tests/                   105 automated tests
└── docs/PROJECT_REPORT.md   full project report
```

## How to run

Python 3.10+ is needed.

```bash
cd SentimentAnalysisApp
pip install -r requirements.txt
python main.py --train        # about 30-45 seconds; creates models/
python main.py                # interactive menu
python main.py --serve        # web form at http://127.0.0.1:8040/
```

Other commands:

```bash
python main.py --analyze "The food was great but the service was painfully slow"
python main.py --report reports/model_report.html
python main.py --train --data sample_data/reviews_dirty.csv
python main.py --generate sample_data/more_reviews.csv --rows 5000
```

JSON API (while the web server runs):

```bash
curl -X POST http://127.0.0.1:8040/api/predict -H "Content-Type: application/json" \
  -d '{"text": "Loved the new update, so much faster now!"}'
```

Your own data: a CSV with a text column (`text`, `review`, `comment`, `message`, `tweet`, `content`, `body` ...) and either a label column
(`label`, `sentiment`, `class` ... with values like positive/neutral/negative, pos/neu/neg or +1/0/−1) or a star-rating column
(`rating`, `stars`, `score`). Open `sample_data/reviews_dirty.csv` for an example.

Batch: menu option 2, with a CSV that has a text column. The output keeps your columns and adds the sentiment, confidence, score and the three
probabilities; "Not sure" rows say what they lean towards. Bad rows are reported by line number; good rows are still analysed.

## Concepts demonstrated

Text classification · text normalisation and negation handling · TF-IDF (word and character n-grams) · `Pipeline` and `ColumnTransformer` ·
multi-class evaluation (macro-F1, confusion matrix, per-class precision/recall) · class imbalance · stratified cross-validation ·
hyper-parameter search · probability calibration (Linear SVM) · selective prediction (a confidence threshold) · rule-based versus learned
models · ordinal errors (a serious mistake versus a small one) · overfitting check · model-agnostic explanation by occlusion ·
model persistence with integrity checks · input validation · responsible use of ML on text.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

105 tests (about 1–1.5 minutes) cover text normalisation (negation, emoji, contractions, scope), the rule-based scorer, label and rating
parsing, the data generator (including checks that no word belongs to one class by accident), file loading, every cleaning rule, metric maths
worked by hand, the confidence threshold, model quality and reproducibility, persistence and tamper detection, validation, sentiment for clear
and hard texts, negation, explanations, part-by-part analysis, batch files, the web server (HTML and JSON, malformed requests, escaping) and
the menu. A small model is trained once and shared.

## Limitations

- The data is synthetic and template-based, so it is easier than real reviews; real accuracy would be lower and the model should be retrained on
  local data. (For example, "love" appears among the negative words only because the generator uses sarcasm such as "Love how the screen never works".)
- English only. Texts in other languages get a warning and an unreliable score.
- Sarcasm and mixed reviews are hard even for people; "Not sure" exists for exactly that reason.
- Short idioms the training data never contained ("not bad at all") can be misjudged.
- A helper for reading feedback, not a judge of people. Check the "Not sure" and serious-mistake lists before acting on aggregate numbers.
- Pickle-based model files must only be loaded if you trained them yourself.
