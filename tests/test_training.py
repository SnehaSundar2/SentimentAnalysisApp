"""Model training, comparison, the confidence threshold and evaluation."""

import copy

import numpy as np
import pandas as pd

from sentimentapp import datagen, evaluation as E, models as M, preprocessing as P, report, schema as S, training
from sentimentapp.exceptions import DataError
from tests.helpers import TempDirTest, shared


def one_hot(predicted, strength=0.6):
    """Probabilities that put `strength` on the predicted class and share the rest equally."""
    p = np.full((len(predicted), 3), (1 - strength) / 2)
    p[np.arange(len(predicted)), predicted] = strength
    return p


class TestEvaluationMaths(TempDirTest):
    def test_metrics_on_a_tiny_example(self):
        actual, predicted = [0, 1, 2, 2], [0, 1, 2, 0]
        m = E.overall_metrics(actual, one_hot(predicted))
        self.assertEqual(m["accuracy"], 75.0)
        self.assertAlmostEqual(m["f1_macro"], 100 * (2 / 3 + 1 + 2 / 3) / 3)      # per-class F1: 0.667, 1.0, 0.667
        self.assertEqual((m["serious_errors"], m["serious_error_rate"]), (1, 25.0))   # a positive review called negative
        self.assertAlmostEqual(m["ordinal_error"], 0.5)
        self.assertGreater(m["log_loss"], 0)
        self.assertAlmostEqual(m["balanced_accuracy"], 100 * (1 + 1 + 0.5) / 3)

    def test_neighbouring_mistakes_are_not_serious(self):
        flips = E.overall_metrics([2, 2, 0, 0], one_hot([1, 1, 1, 1]))                # Neutral instead of the polar class
        self.assertEqual(flips["serious_errors"], 0)
        self.assertAlmostEqual(flips["ordinal_error"], 1.0)
        worst = E.overall_metrics([2, 0], one_hot([0, 2]))
        self.assertEqual(worst["serious_errors"], 2)
        self.assertEqual(worst["ordinal_error"], 2.0)

    def test_per_class_and_confusion(self):
        actual, predicted = [0, 1, 2, 2], [0, 1, 2, 0]
        rows = E.per_class(actual, one_hot(predicted))
        self.assertEqual([r["class"] for r in rows], S.CLASSES)
        self.assertEqual((rows[0]["precision"], rows[0]["recall"]), (50.0, 100.0))
        self.assertEqual((rows[2]["precision"], rows[2]["recall"], rows[2]["support"]), (100.0, 50.0, 2))
        self.assertEqual(E.confusion(actual, one_hot(predicted)), [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(E.per_class([1, 1], one_hot([1, 1]))[0]["precision"], 0.0)       # a class that never occurs: defined

    def test_confidence_threshold_by_hand(self):
        # 80 predictions that are right with 90% confidence, 20 that are wrong with 50% confidence
        right, wrong = np.tile([0.9, 0.05, 0.05], (80, 1)), np.tile([0.5, 0.3, 0.2], (20, 1))
        probabilities = np.vstack([right, wrong])
        actual = np.concatenate([np.zeros(80, int), np.ones(20, int)])
        t = E.choose_confidence_threshold(actual, probabilities, target=0.95)
        self.assertTrue(t["target_met"])
        self.assertAlmostEqual(t["confidence"], 0.9)
        self.assertAlmostEqual(t["training_coverage"], 0.8)
        self.assertAlmostEqual(t["training_accuracy"], 1.0)
        loose = E.choose_confidence_threshold(actual, probabilities, target=0.75)
        self.assertAlmostEqual(loose["confidence"], 0.5)                                # everything is trusted at a looser target
        nothing = E.choose_confidence_threshold(np.ones(100, int), np.tile([0.9, 0.05, 0.05], (100, 1)))
        self.assertFalse(nothing["target_met"])                                         # always wrong: nothing is trusted
        self.assertGreater(nothing["confidence"], 0.9)

    def test_selective_metrics_and_curves(self):
        right, wrong = np.tile([0.9, 0.05, 0.05], (80, 1)), np.tile([0.5, 0.3, 0.2], (20, 1))
        probabilities = np.vstack([right, wrong])
        actual = np.concatenate([np.zeros(80, int), np.ones(20, int)])
        s = E.selective_metrics(actual, probabilities, 0.9)
        self.assertEqual((s["automatic"], s["not_sure"], s["coverage"]), (80, 20, 80.0))
        self.assertEqual((s["accuracy_when_sure"], s["accuracy_when_not_sure"]), (100.0, 0.0))
        self.assertEqual(E.selective_metrics(actual, probabilities, 2.0)["automatic"], 0)       # nothing sure
        curve = E.coverage_curve(actual, probabilities)
        self.assertEqual(curve[0], (1.0, 0.8))
        self.assertTrue(all(b[0] <= a[0] for a, b in zip(curve, curve[1:])))                     # coverage only falls
        bands = E.accuracy_by_confidence(actual, probabilities)
        self.assertEqual(sum(b["n"] for b in bands), 100)
        self.assertEqual(next(b for b in bands if b["band"] == "90-100%")["accuracy"], 100.0)
        self.assertIsNone(next(b for b in bands if b["band"] == "60-70%")["accuracy"])           # nothing in that band

    def test_segments(self):
        frame = pd.DataFrame({"text": ["not good but fine", "good", "never again", "bad", "ok then"],
                              "actual": [0, 2, 0, 0, 1], "predicted": [1, 2, 0, 0, 1]})
        summary = E.summarise_frame(frame)
        self.assertEqual(set(summary), {"length", "negation", "contrast"})
        contrast = {s["group"]: s for s in summary["contrast"]}
        self.assertEqual((contrast["Has but / however"]["errors"], contrast["Has but / however"]["n"]), (1, 1))
        self.assertEqual(contrast["No contrast"]["errors"], 0)
        negation = {s["group"]: s for s in summary["negation"]}
        self.assertEqual(negation["Has negation (not, never, n't ...)"]["n"], 2)
        self.assertEqual(summary["contrast"][0]["group"], "Has but / however")                  # most error-prone first

    def test_top_words_including_negation(self):
        texts = (["a delightful stay", "delightful staff and view", "delightful room"] * 4
                 + ["an awful stay", "awful staff and view", "awful room", "the room was not good", "not good at all"] * 3
                 + ["arrived on monday", "bought it monday", "arrived after a week"] * 4)
        labels = [2] * 12 + [0] * 15 + [1] * 12
        words = E.top_words(texts, labels, k=4, min_df=3)
        self.assertEqual(set(words), set(S.KEYS))
        self.assertEqual(words["positive"][0]["word"], "delightful")
        self.assertIn("awful", [w["word"] for w in words["negative"]])
        self.assertIn("not good", [w["word"] for w in words["negative"]])                         # negation shows up as its own feature
        self.assertIn("monday", [w["word"] for w in words["neutral"]])

    def test_error_examples_and_ceiling(self):
        frame = pd.DataFrame({"text": ["a" * 300, "short", "right", "other"], "actual": [0, 1, 2, 2], "predicted": [2, 0, 2, 1],
                              "confidence": [0.9, 0.7, 0.99, 0.5]})
        errors = E.error_examples(frame, k=2)
        self.assertEqual(len(errors), 2)
        self.assertEqual((errors[0]["actual"], errors[0]["predicted"], errors[0]["confidence"]), ("Negative", "Positive", 0.9))
        self.assertLessEqual(len(errors[0]["text"]), 170)
        self.assertEqual(errors[1]["text"], "short")                                          # most confident mistake first
        self.assertEqual(E.noise_ceiling(0.03), 97.0)


class TestTrainedModel(TempDirTest):
    def test_comparison_table(self):
        result, _, _ = shared()
        names = [c["model"] for c in result.comparison]
        self.assertEqual(set(names), {M.BASELINE, M.LEXICON, "Naive Bayes", "Logistic Regression", "Linear SVM"})
        scores = [c["f1_macro"] for c in result.comparison]
        self.assertEqual(scores, sorted(scores, reverse=True))                        # best first
        self.assertEqual(result.comparison[-1]["model"], M.BASELINE)                  # the baseline is the worst
        self.assertLess(result.comparison[-1]["f1_macro"], 0.3)                       # one class only: a third of the classes at best
        lexicon = next(c for c in result.comparison if c["model"] == M.LEXICON)
        self.assertGreater(lexicon["f1_macro"], 0.55)                                 # the word list is a respectable baseline
        self.assertGreater(result.comparison[0]["f1_macro"], lexicon["f1_macro"] + 0.05)     # learning adds a clear amount
        self.assertNotIn(result.metadata["model"], (M.BASELINE, M.LEXICON))           # a learned model is always chosen

    def test_split_sizes_and_data_flow(self):
        result, _, raw = shared()
        rows = result.metadata["rows"]
        self.assertEqual(rows["raw"], len(raw))
        self.assertEqual(rows["train"] + rows["test"], rows["clean"])
        self.assertAlmostEqual(rows["test"] / rows["clean"], 0.2, delta=0.02)
        self.assertEqual(len(result.test_frame), rows["test"])
        self.assertEqual(len(result.reference), rows["train"])
        train_texts, test_texts = set(result.reference["text"].str.lower()), set(result.test_frame["text"].str.lower())
        self.assertEqual(train_texts & test_texts, set())                             # no review appears on both sides

    def test_class_shares_are_equal_in_train_and_test(self):
        result, _, _ = shared()
        train = result.reference["label"].value_counts(normalize=True)
        test = result.test_frame["actual"].value_counts(normalize=True)
        for c in (0, 1, 2):
            self.assertAlmostEqual(train[c], test[c], delta=0.02)

    def test_accuracy_on_unseen_reviews(self):
        result, _, _ = shared()
        test, train = result.metadata["metrics"]["test"], result.metadata["metrics"]["train"]
        self.assertGreater(test["accuracy"], 80)
        self.assertGreater(test["f1_macro"], 78)
        self.assertLess(test["serious_error_rate"], 6)                                # positive ↔ negative flips are rare
        self.assertLess(test["ordinal_error"], 0.25)
        self.assertGreater(test["accuracy"], test["majority_accuracy"] + 30)          # far above "always the most common class"
        self.assertLess(train["accuracy"] - test["accuracy"], 15)                     # no serious overfitting
        for row in test["per_class"]:
            self.assertGreater(row["f1"], 65, row["class"])
        self.assertEqual(sum(r["support"] for r in test["per_class"]), test["n"])
        matrix = np.array(result.metadata["confusion"])
        self.assertEqual(matrix.sum(), test["n"])
        self.assertEqual(int(matrix.trace()), round(test["accuracy"] * test["n"] / 100))

    def test_confidence_threshold_sends_the_doubtful_ones_to_a_person(self):
        result, _, _ = shared()
        conf, test = result.metadata["confidence"], result.metadata["metrics"]["test"]
        sel = test["selective"]
        self.assertTrue(conf["target_met"])
        self.assertGreater(conf["confidence"], 1 / 3)                                 # better than a three-way tie
        self.assertLessEqual(conf["confidence"], 1.0)
        self.assertGreater(sel["coverage"], 50)                                       # most reviews are labelled automatically
        self.assertGreater(sel["accuracy_when_sure"], test["accuracy"])               # and those are more accurate than average
        self.assertGreater(sel["accuracy_when_sure"], sel["accuracy_when_not_sure"])
        self.assertEqual(sel["automatic"] + sel["not_sure"], test["n"])
        frame = result.test_frame
        self.assertEqual(int((frame["confidence"] >= conf["confidence"]).sum()), sel["automatic"])

    def test_the_model_is_more_often_right_when_it_is_more_sure(self):
        result, _, _ = shared()
        bands = {b["band"]: b for b in result.metadata["confidence_bands"]}
        self.assertEqual(sum(b["n"] for b in bands.values()), result.metadata["rows"]["test"])
        top = bands["90-100%"]
        low = [b for k, b in bands.items() if k in ("0-50%", "50-60%", "60-70%") and b["n"] >= 10]
        self.assertGreater(top["n"], 50)
        for b in low:
            self.assertGreater(top["accuracy"], b["accuracy"])

    def test_mistakes_are_listed_and_tables_are_complete(self):
        result, _, _ = shared()
        meta = result.metadata
        errors = meta["errors"]
        self.assertLessEqual(len(errors), 8)
        self.assertTrue(all(e["actual"] != e["predicted"] for e in errors))
        confidences = [e["confidence"] for e in errors]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertEqual(set(meta["top_words"]), set(S.KEYS))
        self.assertTrue(all(len(v) == 10 for v in meta["top_words"].values()))
        self.assertEqual(set(meta["segments"]), {"length", "negation", "contrast"})
        contrast = {s["group"]: s for s in meta["segments"]["contrast"]}
        self.assertGreater(contrast["Has but / however"]["error_rate"], contrast["No contrast"]["error_rate"])   # mixed reviews are harder

    def test_metadata_is_complete(self):
        result, _, _ = shared()
        for key in ("trained_at", "model", "metrics", "confidence", "environment", "comparison", "cleaning", "confusion",
                    "top_words", "errors", "segments", "coverage_curve", "confidence_bands", "seed", "class_counts", "ceiling"):
            self.assertIn(key, result.metadata, key)
        self.assertEqual(result.metadata["ceiling"], 97.0)
        self.assertLess(result.metadata["metrics"]["test"]["accuracy"], result.metadata["ceiling"] + 3)


class TestReproducibilityAndGuards(TempDirTest):
    def test_same_seed_same_result(self):
        raw = datagen.generate(800, seed=11)
        a = training.train(raw, fast=True, tune=False, cv=3)
        b = training.train(raw, fast=True, tune=False, cv=3)
        self.assertEqual(a.metadata["confidence"], b.metadata["confidence"])
        np.testing.assert_allclose(a.test_frame["confidence"], b.test_frame["confidence"])
        self.assertEqual(list(a.test_frame.index), list(b.test_frame.index))

    def test_tiny_or_unusable_data_is_refused(self):
        with self.assertRaises(DataError):
            training.train(datagen.generate(80, dirty=False), fast=True, tune=False)
        only_two = datagen.generate(900, dirty=False)
        only_two = only_two[only_two["label"] != "Neutral"]
        with self.assertRaises(DataError):
            training.train(only_two, fast=True, tune=False)

    def test_tuning_search(self):
        X, y = P.split_xy(P.clean(datagen.generate(700, dirty=False))[0])
        search, params = M.tune("Naive Bayes", X, y, cv=3, iterations=3)
        self.assertIn(params["alpha"], M.TUNING_GRIDS["Naive Bayes"]["model__alpha"])
        self.assertEqual(M.tune(M.BASELINE, X, y), (None, {}))                         # nothing to tune
        self.assertEqual(M.tune(M.LEXICON, X, y), (None, {}))
        pipe = M.pipeline_with_params("Naive Bayes", params)
        self.assertEqual(pipe.fit(X, y).predict_proba(X.head(3)).shape, (3, 3))
        svm = M.pipeline_with_params("Linear SVM", {"estimator__C": 0.3})              # the nested parameter name works too
        self.assertEqual(svm.fit(X, y).predict_proba(X.head(2)).shape, (2, 3))

    def test_best_candidate_skips_the_non_learners(self):
        results = [{"model": M.LEXICON}, {"model": M.BASELINE}, {"model": "Naive Bayes"}]
        self.assertEqual(M.best_candidate(results)["model"], "Naive Bayes")


class TestReports(TempDirTest):
    def test_text_and_html_reports(self):
        meta = shared()[0].metadata
        text = report.model_text(meta)
        for part in ("ACCURACY ON THE UNSEEN TEST SET", "CONFUSION MATRIX", "MOST TELLING WORDS", "ERRORS BY KIND", "Macro-F1",
                     "Serious errors"):
            self.assertIn(part, text)
        self.assertIn("chosen", report.comparison_text(meta))
        page = report.render_html(meta)
        self.assertEqual(page.count('<details class="tv">'), 7)                        # every chart has a table view
        self.assertIn("prefers-color-scheme:dark", page)
        self.assertIn("Data cleaning log", page)
        self.assertIn("Confusion matrix", page)
        self.assertIn("Words that signal negative", page)
        path = report.save_html(meta, self.dir / "out" / "report.html")
        self.assertTrue(path.exists())

    def test_text_in_the_report_is_escaped(self):
        meta = copy.deepcopy(shared()[0].metadata)
        meta["errors"] = [{"text": "<script>alert(1)</script> & <b>bold</b>", "actual": "Positive", "predicted": "Negative",
                           "confidence": 0.9}]
        page = report.render_html(meta)
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; &amp; &lt;b&gt;bold&lt;/b&gt;", page)

    def test_chart_helpers(self):
        for value in (7, 45, 130, 999):
            self.assertGreaterEqual(report._tick_ceiling(value, 5), value)
        columns = report.column_chart([("a", 80.0), ("b", None), ("c", 0), ("d", 55.5)], "t", vmax=100)
        self.assertEqual(columns.count("<path"), 2)                                    # nothing drawn for None or 0
        self.assertIn(">100<", columns)
        line = report.line_chart([(0.5, 0.8), (0.75, 0.9), (1.0, 0.95)], "t", "x", "y", x_range=(0.5, 1), y_range=(0.5, 1),
                                 marker=(0.9, 0.93), marker_label="chosen")
        self.assertIn('class="line"', line)
        self.assertIn("chosen", line)
        bars = report.hbar_chart([("a", 1.0), ("b", 2.0)], "t", highlight="b")
        self.assertEqual(bars.count('class="mark other"'), 1)
