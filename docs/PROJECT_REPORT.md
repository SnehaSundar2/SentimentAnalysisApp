# Project Report: Sentiment Analysis App

**Program:** Virtual Internship – Skyrovix IT Solutions Pvt. Ltd. (AI & Machine Learning project)
**Project:** Sentiment Analysis App
**Developed and submitted by:** Sneha S (Intern ID: SKX-2026-6425)
**Due date:** 15 November 2026

---

## 1. Introduction

Companies receive thousands of reviews, comments and messages. Reading them all is impossible, yet they say what customers love and what
makes them leave. *Sentiment analysis* teaches a computer to judge how a text feels (positive, neutral or negative) so that people can focus
on the feedback that matters.

Language is full of traps. "The battery was **not** good" is the opposite of "The battery was good". "Oh great, another crash" uses a
positive word for a complaint. "The food was great but the service was slow" has two feelings in one sentence. This project builds a
sentiment analyser that deals with these traps, tells the user how sure it is, and **says "Not sure" instead of guessing** when it is not.
It cleans messy data, compares several models (including a rule-based word list that learns nothing), evaluates the winner honestly on
reviews it has never seen, and offers prediction interfaces: a menu, a command line, a web form, a JSON API and batch files. Every answer
comes with a score, the words behind it, a verdict for each part of a mixed text and similar known reviews.

**About the data.** Real review collections are licensed and carry real people's words, so the project generates realistic but *artificial*
reviews. Each review is built from clauses with a known polarity; negation flips a clause, a clause after "but" counts one and a half times, a
tie is Neutral, and a few percent of labels are shifted the way human annotators disagree. The results therefore describe the workflow, not
real-world accuracy. A real dataset can be used with `--data` (text plus a label or a star rating).

## 2. Objectives

1. **Data preprocessing:** load, clean and normalise text (negation, emoji, shouting, HTML) and labels, leaving a record of every change.
2. **Model training:** compare several algorithms fairly, including a no-learning baseline, tune the best, and avoid leaking test data into training.
3. **Evaluation:** measure what matters for three classes (macro-F1, per-class scores, serious mistakes), study where the model is weak, and decide when it should not answer.
4. **Prediction interface:** make the model usable through easy interfaces, with strict input validation and clear explanations.
5. **Quality:** handle errors safely, document the workflow, and verify the behaviour with automated tests.

## 3. Tools and technologies

| Item | Used |
|---|---|
| Language | Python 3 (tested on 3.13) |
| ML libraries | scikit-learn (TF-IDF, pipelines, classifiers, cross-validation, tuning, calibration), pandas, numpy, joblib |
| Standard library | `http.server` (web interface), `re`, `html`, `hashlib` (checksums), `json`, `csv`, `unittest` |
| Output | Terminal reports, self-contained HTML report with SVG charts, JSON API |

## 4. System design

### 4.1 Workflow

1. **Load** the CSV (a text column plus a label or star-rating column; many common names understood; non-UTF-8 files are read and noted).
2. **Clean** the data and record every action.
3. **Split** 80% / 20% into training and test sets with the same class shares in both. The test set is not touched again until the end.
4. **Compare** a baseline, a rule-based word list and three learned models by 5-fold stratified cross-validation on the training reviews.
5. **Tune** the best learned model with a randomised search; keep the tuned version only if cross-validation says it is better.
6. **Choose the confidence threshold** from out-of-fold predictions on the training reviews.
7. **Evaluate once** on the test set; list the mistakes, the most telling words and the error rate per kind of text.
8. **Save** the model with checksums, then **analyse text** through any interface.

### 4.2 Text preprocessing and features

| Step | What it does |
|---|---|
| Markup and whitespace | HTML tags and entities removed (`<br />`, `&amp;`), control characters removed, spaces collapsed |
| Normalisation | Lower-case; links → `urltoken`, @mentions → `usertoken`; emoji → `emopos` / `emoneg`; stretched words shortened (`sooo` → `soo`, flagged); SHOUTED words flagged; `n't` → `not` |
| Negation carried forward | After a negator, the next three words (until punctuation or "but") are prefixed: `not good` → `not not_good` |
| Polarity features | A 130-word praise/criticism list adds shared markers: `lexpos` / `lexneg`, and `flippos` for a negated praise word, `flipneg` for a negated criticism word. This lets the model understand "not decent" without having seen it |
| Word TF-IDF | Single words and word pairs of the normalised text |
| Character TF-IDF | 3–5 letter pieces: robust to typos |
| Text shape | Fifteen numbers between 0 and 1: length, exclamations, questions, capitals, shouted words, happy/unhappy emoji, negations, stretched words, contrast words, word count, praise and criticism words, negated praise and criticism |

### 4.3 Modules

| Module | Responsibility |
|---|---|
| `schema.py` | Classes, column names, limits, score and tone, example texts |
| `textproc.py` | Normalisation, text-shape features, pieces and signals used for explanations |
| `wordlist.py`, `lexicon.py` | The small word list, and the rule-based scorer built on it (no learning) |
| `datagen.py` | Synthetic reviews with negation, sarcasm, mixed feelings, label noise and injected problems |
| `preprocessing.py` | Loading, cleaning with a report, feature transformer |
| `models.py` | Candidate models, cross-validated comparison, tuning |
| `evaluation.py` | Metrics, confidence threshold, curves, top words, mistakes |
| `training.py` | The workflow end to end |
| `persistence.py` | Saving and loading with SHA-256 integrity checks |
| `validators.py` | Validation of the text to analyse |
| `predictor.py` | Sentiment, score, explanation, parts, similar reviews, batch CSV |
| `report.py` | Text and HTML reports |
| `web.py`, `cli.py`, `main.py` | Web form and API, menu, command line |
| `exceptions.py` | Custom error types |

## 5. Implementation highlights

**Cleaning with an audit trail.** The generator deliberately produces about 10% problem rows. The cleaner found and handled all of them, and said what it did:

```
3079 rows read → 2878 kept (201 removed)
  • Rows with a missing or non-text review: 8 — removed
  • Reviews with leftover HTML tags or entities: 105 — markup removed
  • Reviews with stray whitespace or control characters: 107 — tidied
  • Reviews empty or too short to judge: 10 — removed (need at least 2 letters or digits)
  • Reviews longer than 5000 characters: 2 — truncated
  • Labels written in another style: 467 — standardised to Negative / Neutral / Positive (e.g. 'pos', '+1', 'bad')
  • Rows with a missing or unrecognised label: 10 — removed (e.g. 'mixed', 'unsure')
  • Same review given different labels: 37 — removed (no label can be trusted)
  • Duplicate reviews: 136 — removed (kept the first copy, ignoring capitals)
  Kept: 1133 negative, 577 neutral, 1168 positive
```

**No leakage.** Repeats (ignoring capitals) and contradictory labels are removed *before* the split, so the same review cannot be in training and test; a test checks
there is zero overlap. Features are fitted on training texts only. The confidence threshold uses out-of-fold training predictions, leaving the test set as an independent check.

**Why macro-F1, and why a rule-based baseline.** Neutral is the smallest class (about 20% of reviews), and accuracy would reward a model that neglects it. Macro-F1 averages the F1 of
the three classes, so Neutral counts as much as the others. A model that always says "Positive" scores 41% accuracy and 19% macro-F1. Next to the learned models sits a rule-based scorer
(a small word list with negation, intensifiers, emoji and "but" weighting) that learns nothing. It is a fair yardstick: it shows what learning from examples adds.

**Negation was a real bug, found by looking.** The first version treated `not_good`, `not_decent` ... as unrelated features. With only ten "was not good" examples in the data, the model
could not carry "not" over to other words, and "The battery was not good" was called positive. Marking negated praise and criticism words with *shared* polarity features fixed it (every
model now puts that sentence on Negative with 80–98% probability), and a test keeps it fixed. An earlier mistake of the same kind was in the data generator: the word "great" appeared only in
sarcasm, so the model learned that "great" means negative. The generator now uses it in genuine praise too, and a test checks that no word belongs to one class by accident.

**Saying "Not sure".** A wrong confident answer is worse than an honest doubt. The confidence threshold is the lowest confidence at which the predictions above it are right at least 95% of the time on
out-of-fold training predictions. Anything below it is marked "Not sure" and left to a person, with the likely sentiment shown. In the web page and batch output these rows say what they lean towards.

**Explanations anyone can follow.** Each word, emoji and link is removed in turn and the text re-scored. The effect is measured as a change in log-odds, so it still shows up when the model is 99% sure; it is shown
as a factor ("this word multiplies the odds of *negative* by 24"). The page marks those words in the text itself (underline plus highlight, so colour is not the only signal). A text with two or more sentences
or "but" clauses is also judged part by part, so "great food, painfully slow service" shows Positive and Neutral/Negative side by side. Plain-English style notes and the three most similar known reviews sit beside it.

**A safe model file.** `metadata.json` stores the SHA-256 of the model and reference files. Loading verifies them first, so a damaged or modified file is refused with a clear message instead of being unpickled.

## 6. Results

| Model (5-fold CV on training reviews) | Macro-F1 | Accuracy | Log loss |
|---|---|---|---|
| **Linear SVM (tuned)** ← chosen | **88.1%** | 89.8% | 0.371 |
| Linear SVM | 88.0% | 89.9% | 0.371 |
| Logistic Regression | 85.6% | 87.7% | 0.390 |
| Naive Bayes | 77.4% | 80.2% | 0.668 |
| Rule-based word list (no learning) | 74.9% | 77.0% | 0.660 |
| Baseline (most common class) | 19.2% | 40.6% | 1.055 |

**On the 576 unseen test reviews:** accuracy **88.5%**, macro-F1 **86.2%**, balanced accuracy 86.0%, log loss 0.365. Training accuracy is 94%, so there is no serious overfitting.

| Class | Precision | Recall | F1 | Reviews |
|---|---|---|---|---|
| Negative | 92.1% | 92.5% | 92.3% | 227 |
| Neutral | 77.1% | 73.0% | 75.0% | 115 |
| Positive | 90.4% | 92.3% | 91.3% | 234 |

| Real ↓ / predicted → | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 210 | 11 | 6 |
| Neutral | 14 | 84 | 17 |
| Positive | 4 | 14 | 216 |

**Interpretation.**
- Only 10 of 576 mistakes (1.7%) are *serious* (positive called negative or the reverse). Almost every other error is between a polar class and Neutral, the class with the least clear-cut wording (F1 75%).
- Learning from examples adds about 11 points of macro-F1 over the rule-based word list, and about 67 over the baseline.
- Predictions with at least 65% confidence are made automatically for 84% of reviews and are right 94% of the time; the other 91 reviews are marked "Not sure". The model is right more often when it is more sure: 99% in the top confidence band against 38% in the lowest.
- Texts with "but / however" are the hardest: 39.6% wrong against 6.2% without. "Great food but slow service" has no single right answer, which is why the part-by-part view exists.
- Texts with negation are wrong 12.9% of the time against 10.8% without: the negation features make them nearly as easy as the rest.
- The most telling words match intuition (`enjoyed`, `fabulous`, `outstanding`, happy emoji for positive; `waste`, `rubbish`, unhappy emoji for negative; `tried`, `arrived`, `visited` for neutral). "love" appears among the negative words only because the generator uses sarcasm such as "Love how the screen never works".
- The synthetic reviews are built from templates, so they are easier than real reviews. Real accuracy would be lower, and a real deployment needs retraining on local data.

## 7. Testing

105 automated tests were written and **all pass** (about 1–1.5 minutes). One small model is trained once and shared by the tests.

| # | Test case | Expected | Result |
|---|---|---|---|
| 1 | `not good`, `wasn't good`, `can't complain`, `won't refund` | Negation marked on the following words | Pass |
| 2 | Negation scope: comma, "but", period, three-word limit, intensifiers | Scope ends or skips as designed | Pass |
| 3 | Emoji, links, @mentions, hashtags, stretched and SHOUTED words, numbers | Marker tokens; word kept in lower case | Pass |
| 4 | Known praise/criticism words, negated and not | Shared polarity markers (`lexpos`, `flippos`, `flipneg` ...) | Pass |
| 5 | HTML tags and entities | Removed and decoded | Pass |
| 6 | Text-shape features | Shape (n, 15), values in [0, 1], flags right | Pass |
| 7 | Pieces for explanations; removing a piece; splitting into parts; style notes | Words, emoji and links once each; whole words only; sentences and "but" clauses | Pass |
| 8 | Rule-based scorer | Sign, intensifier, negation, "but" weighting, emoji, bounded | Pass |
| 9 | `pos`, `+1`, `good`, `neu`, `0`, `-1`, `bad`, `mixed`, empty | Standardised or rejected | Pass |
| 10 | Star ratings 1–5 | Negative / Neutral / Positive | Pass |
| 11 | Score and tone boundaries | −100 … +100 and the words for each range | Pass |
| 12 | Generator run twice with the same seed | Identical data; different with another seed | Pass |
| 13 | Generated data | Plausible class shares; "great" in both praise and sarcasm; "not good" mostly negative, "not bad" mostly positive; hard cases present | Pass |
| 14 | CSV: label or star-rating column; Kaggle-style names; Windows-1252 | Understood, with a note in the cleaning report | Pass |
| 15 | CSV: missing, empty, no text column, no label, 60 rows | Specific `DataError` for each | Pass |
| 16 | One data problem at a time (missing, number, blank, too short, bad label, duplicate, other capitals) | Row removed exactly as documented | Pass |
| 17 | Same text with different labels | Every copy removed | Pass |
| 18 | HTML, whitespace, 12,000-character text | Tidied / truncated, not dropped | Pass |
| 19 | Cleaning report accounting; input not modified | Rows dropped = rows removed by the listed steps | Pass |
| 20 | Columns swapped; no neutral reviews | Refused with a clear reason | Pass |
| 21 | Empty text, emoji only, unseen words, Chinese, 5,000 characters through every model | Valid probabilities, no crash | Pass |
| 22 | "was good" vs "was not good" | Opposite classes | Pass |
| 23 | Metrics, confusion and per-class scores worked by hand | Accuracy 75%, macro-F1 77.8%, 1 serious error | Pass |
| 24 | Neighbouring versus polarity-flipping mistakes | Only flips counted as serious | Pass |
| 25 | Confidence threshold on a worked example; unreachable target | 0.9 at target 0.95; "nothing trusted" when always wrong | Pass |
| 26 | Selective metrics, coverage curve, confidence bands | Counts add up; coverage only falls; empty bands handled | Pass |
| 27 | Chosen model vs baseline vs word list | Best learned beats the list by 5+ points; baseline last | Pass |
| 28 | No review on both sides of the split; equal class shares | Zero overlap; within 2 points | Pass |
| 29 | Accuracy, macro-F1, per-class F1, serious errors on unseen reviews | Above the stated floors; no overfitting | Pass |
| 30 | More often right when more sure; "but" texts harder | Top band beats low bands; error rate higher | Pass |
| 31 | One byte flipped in `model.joblib`; missing file; bad JSON; no or wrong checksum | Refused with a clear `ModelError` | Pass |
| 32 | Clear positive, negative and negation examples; sarcasm, neutral, mixed | Right sentiment; honest doubt on the hard ones | Pass |
| 33 | Explanations for a positive text | The praise words support it; taking the strongest out lowers the score | Pass |
| 34 | Highlighting | Chunks rejoin to the original text; emoji separate | Pass |
| 35 | A mixed review | At least two parts, both Positive and Negative shown | Pass |
| 36 | Batch CSV with a blank row, an over-long row, markup only and a one-word row | 4 analysed, 3 reported by line number, short one warned | Pass |
| 37 | Web: `"><script>alert(1)</script>` and `<img onerror>` in the text | Escaped; never appears as markup | Pass |
| 38 | Web: wrong content type / bad JSON / list / 60 KB body / foreign Host header | 415 / 400 / 400 / 413 / 403 | Pass |
| 39 | Web and menu: positive, negative, mixed, empty text, reports, batch, dataset tools, no model, star-rating training | Friendly output at each step | Pass |

## 8. Challenges and solutions

| Challenge | Solution |
|---|---|
| Real review data is licensed and personal | A generator with known clause polarities, negation, sarcasm and label noise, so results are reproducible and the labels are explainable |
| "not good" looked positive to the first model | Negation carried forward plus shared polarity markers for negated praise and criticism; a test keeps it fixed |
| A word appearing in only one class by accident (`great` only in sarcasm) | Widened vocabulary and sentence patterns; a test checks that words are not tied to one class |
| Accuracy hides a neglected small class | Rank by macro-F1; report per-class scores and the confusion matrix |
| Some mistakes are worse than others | Count positive ↔ negative flips separately from neighbouring-class errors |
| A confident wrong answer is worse than doubt | A confidence threshold chosen on training data; "Not sure" with the likely sentiment shown |
| Mixed reviews have two feelings | Part-by-part analysis of sentences and "but" clauses |
| Is learning worth it? | A rule-based word list is compared in the same cross-validation |
| Untrusted text shown in a web page | Everything, including the text and the highlighted words, is escaped; no scripts; strict CSP |
| HTML left over from scraping | Stripped before analysis, in training data and in every interface |
| Pickled models can be altered or damaged | SHA-256 verified before loading, plus a clear warning to load only your own models |
| An oversized web request could be lost on Windows | The server reads and discards a bounded amount of the body before sending the 413 reply |

## 9. Future enhancements

- Train and compare on a real, consented review collection and report both results side by side.
- Add aspect-level sentiment: not only "negative" but "negative about the delivery, positive about the product".
- Detect sarcasm and mixed feelings explicitly rather than leaving them to "Not sure".
- Add language detection and a multilingual model.
- Try a transformer model as a stronger but heavier option and compare it with the cheaper models here.

## 10. Conclusion

The Sentiment Analysis App meets all its objectives. It cleans messy reviews and labels and records every change, understands negation, emoji and shouting, and compares models fairly against a baseline and
a rule-based word list. On reviews it has never seen it is right 88.5% of the time (macro-F1 86.2%), makes a serious positive/negative mistake in under 2% of cases, and says "Not sure" for the doubtful
ones. Every answer comes with a score, the words behind it, a part-by-part view of mixed reviews and similar known reviews, through a menu, a command line, a web form, a JSON API and batch files. Inputs
are validated strictly, errors are handled clearly, saved models are integrity-checked, and 105 automated tests verify the behaviour.

## 11. How to run

```bash
cd SentimentAnalysisApp
pip install -r requirements.txt
python main.py --train
python main.py
python main.py --serve
python -m unittest discover -s tests -t . -v
```
