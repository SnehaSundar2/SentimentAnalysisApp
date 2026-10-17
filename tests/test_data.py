"""Text processing, the rule-based scorer, schema helpers, synthetic data, loading, cleaning and features."""

import re

import numpy as np
import pandas as pd

from sentimentapp import datagen, lexicon, models as M, preprocessing as P, schema as S, textproc as T
from sentimentapp.exceptions import DataError
from tests.helpers import NEGATION, POSITIVE, SARCASM, TempDirTest


class TestNormalisation(TempDirTest):
    def tokens(self, text):
        return T.normalise_text(text).split()

    def test_negation_is_carried_forward(self):
        tokens = self.tokens("The food was not good at all.")
        self.assertIn("not_good", tokens)
        self.assertNotIn("good", tokens)                                      # "not good" and "good" are different features
        self.assertEqual(self.tokens("The food was good"), ["the", "food", "was", "good", "lexpos"])

    def test_contractions_become_not(self):
        self.assertEqual(self.tokens("It wasn't good"), ["it", "was", "not", "not_good", "flippos"])
        self.assertEqual(self.tokens("I can't complain"), ["i", "can", "not", "not_complain"])
        self.assertEqual(self.tokens("They won't refund"), ["they", "will", "not", "not_refund", "flipneg"])

    def test_negation_stops_at_punctuation_and_contrast_words(self):
        self.assertEqual(self.tokens("not good, but fine"), ["not", "not_good", "flippos", "but", "fine", "lexpos"])
        self.assertEqual(self.tokens("not good. fine"), ["not", "not_good", "flippos", "fine", "lexpos"])
        tokens = self.tokens("never really liked the old cheap design")      # at most three words are affected
        self.assertEqual([t for t in tokens if t.startswith("not_")], ["not_liked", "not_the", "not_old"])
        self.assertIn("really", tokens)                                       # intensifiers are skipped, not negated

    def test_emoji_links_mentions_and_hashtags(self):
        self.assertIn("emopos", self.tokens("love it 😍"))
        self.assertIn("emoneg", self.tokens("what a waste 😡"))
        self.assertEqual(self.tokens("@bob see https://x.com/a #love"), ["usertoken", "see", "urltoken", "love", "lexpos"])

    def test_stretched_and_shouted_words_are_flagged(self):
        tokens = self.tokens("sooo gooood")
        self.assertEqual(tokens[:2], ["soo", "good"])
        self.assertIn("elongated", tokens)
        shouted = self.tokens("This is TERRIBLE and AWFUL stuff")
        self.assertEqual(shouted.count("capsword"), 2)
        self.assertIn("terrible", shouted)                                    # and the word itself is kept, lower-cased
        self.assertEqual(self.tokens("I rated it 5 stars"), ["i", "rated", "it", "num", "stars"])

    def test_known_praise_and_criticism_words_share_polarity_features(self):
        self.assertEqual(self.tokens("It was great").count("lexpos"), 1)
        self.assertEqual(self.tokens("It was terrible").count("lexneg"), 1)
        self.assertIn("flippos", self.tokens("not nice"))                      # negated praise
        self.assertIn("flipneg", self.tokens("not bad"))                       # negated criticism
        self.assertNotIn("flippos", self.tokens("nice, not bad"))              # the comma ended the negation
        self.assertEqual(self.tokens("a zorp was fine").count("lexpos"), 1)    # unknown words add nothing, known ones do

    def test_markup_is_stripped(self):
        self.assertEqual(" ".join(T.strip_markup("<b>Loved</b> it &amp; more<br />").split()), "Loved it & more")
        self.assertEqual(T.strip_markup("I'm &#39;happy&#39;"), "I'm 'happy'")


class TestTextShape(TempDirTest):
    def test_meta_features_are_bounded_and_meaningful(self):
        frame = pd.DataFrame({"text": ["I LOVE this!!! 😍", "not good, but fine", "ok", "", "A" * 5000, "soooo bad 😡?"]})
        out = T.meta_features(frame)
        self.assertEqual(out.shape, (6, len(T.META_NAMES)))
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        col = {name: i for i, name in enumerate(T.META_NAMES)}
        self.assertGreater(out[0, col["happy_emoji"]], 0)
        self.assertGreater(out[0, col["exclamations"]], 0)
        self.assertGreater(out[1, col["negations"]], 0)
        self.assertEqual(out[1, col["contrast"]], 1.0)
        self.assertEqual(out[5, col["stretched"]], 1.0)
        self.assertGreater(out[5, col["unhappy_emoji"]], 0)
        self.assertEqual(out[2, col["contrast"]], 0.0)
        polarity = T.meta_features(pd.DataFrame({"text": ["great and lovely", "terrible", "not good", "not bad"]}))
        self.assertGreater(polarity[0, col["praise_words"]], 0)
        self.assertGreater(polarity[1, col["criticism_words"]], 0)
        self.assertGreater(polarity[2, col["negated_praise"]], 0)
        self.assertGreater(polarity[3, col["negated_criticism"]], 0)
        self.assertEqual(polarity[0, col["negated_praise"]], 0.0)

    def test_units_include_words_emoji_and_links_once(self):
        pieces = T.units("Loved it 😍 and LOVED the update, see http://x.xyz/a!")
        kinds = {kind for kind, _ in pieces}
        self.assertEqual(kinds, {"link", "word"})
        self.assertIn(("link", "http://x.xyz/a"), pieces)
        self.assertIn(("word", "😍"), pieces)
        words = [p.lower() for k, p in pieces]
        self.assertEqual(len(words), len(set(words)))                         # "Loved" and "LOVED" are one piece
        self.assertLessEqual(len(T.units(" ".join(f"word{i}x" for i in range(200)))), 40)

    def test_remove_unit(self):
        self.assertEqual(T.remove_unit("Not good, not great", "word", "not"), "good, great")
        self.assertEqual(T.remove_unit("a cat sat", "word", "at"), "a cat sat")           # whole words only
        self.assertEqual(T.remove_unit("great 😍 app", "word", "😍"), "great app")
        self.assertEqual(T.remove_unit("see http://x.xyz now", "link", "http://x.xyz"), "see now")

    def test_split_parts(self):
        parts = T.split_parts("The food was great but the service was slow. Never again!")
        self.assertEqual(parts, ["The food was great", "but the service was slow.", "Never again!"])
        self.assertEqual(T.split_parts("Great."), [])                                     # fragments too short to judge
        self.assertEqual(T.split_parts("Just one sentence here"), ["Just one sentence here"])

    def test_signals(self):
        found = T.signals("I am NOT happy. NEVER again!!! This is TERRIBLE 😡 sooo bad, but fine")
        joined = " | ".join(found)
        for expected in ("Negation found", "Contrast", "Unhappy emoji", "SHOUTED", "exclamation", "Stretched"):
            self.assertIn(expected, joined)
        self.assertEqual(T.signals("the box arrived on monday"), [])


class TestLexicon(TempDirTest):
    def test_scores_have_the_right_sign_and_size(self):
        score = lexicon.lexicon_score
        self.assertGreater(score("great"), 0.3)
        self.assertLess(score("terrible"), -0.3)
        self.assertAlmostEqual(score("the box arrived"), 0.0)
        self.assertGreater(score("very good"), score("good"))                              # intensifier
        self.assertLess(score("not good"), 0)                                              # negation flips
        self.assertGreater(score("not bad"), 0)
        self.assertLess(score("good but terrible"), 0)                                     # the second half weighs more
        self.assertGreater(score("great 😍"), score("great"))
        self.assertGreater(score("GREAT!!!"), score("great"))
        self.assertTrue(all(-1 < score(t) < 1 for t in ("amazing " * 50, "awful " * 50, "")))

    def test_classifier_interface(self):
        frame = pd.DataFrame({"text": [POSITIVE, "Terrible, awful and useless", "I bought it on Monday"]})
        model = lexicon.LexiconClassifier().fit(frame)
        self.assertEqual(model.predict(frame).tolist(), [S.POSITIVE, S.NEGATIVE, S.NEUTRAL])
        p = model.predict_proba(frame)
        self.assertEqual(p.shape, (3, 3))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertEqual(p.argmax(axis=1).tolist(), [S.POSITIVE, S.NEGATIVE, S.NEUTRAL])
        from sklearn.base import clone
        self.assertEqual(clone(model).get_params(), {"threshold": 0.25})                   # cross-validation can clone it


class TestSchemaHelpers(TempDirTest):
    def test_labels(self):
        positive = ["Positive", " pos ", "POS", "+", "+1", "1", 1, 1.0, "good", "happy"]
        neutral = ["neutral", "Neu", "0", 0, 0.0, "ok"]
        negative = ["Negative", "neg", "-", "-1", -1, -1.0, "bad", "unhappy"]
        self.assertEqual({S.canonical_label(x) for x in positive}, {S.POSITIVE})
        self.assertEqual({S.canonical_label(x) for x in neutral}, {S.NEUTRAL})
        self.assertEqual({S.canonical_label(x) for x in negative}, {S.NEGATIVE})
        for bad in ("mixed", "sarcastic", "", None, float("nan"), True, "positively"):
            self.assertIsNone(S.canonical_label(bad), repr(bad))

    def test_star_ratings(self):
        self.assertEqual([S.stars_to_label(x) for x in (1, "2", 2.0, 3, "3.0", 4, 5)],
                         [S.NEGATIVE, S.NEGATIVE, S.NEGATIVE, S.NEUTRAL, S.NEUTRAL, S.POSITIVE, S.POSITIVE])
        for bad in (0, 6, "five", "", None, float("nan")):
            self.assertIsNone(S.stars_to_label(bad), repr(bad))

    def test_score_and_tone(self):
        self.assertEqual(S.score_from_probabilities([0.0, 0.0, 1.0]), 100.0)
        self.assertEqual(S.score_from_probabilities([1.0, 0.0, 0.0]), -100.0)
        self.assertAlmostEqual(S.score_from_probabilities([0.2, 0.5, 0.3]), 10.0)
        self.assertEqual([S.tone(x) for x in (-100, -60, -59, -20, 0, 19, 20, 59, 60, 100)],
                         ["Strongly negative", "Strongly negative", "Negative", "Negative", "Neutral or mixed", "Neutral or mixed",
                          "Positive", "Positive", "Strongly positive", "Strongly positive"])


class TestDatagen(TempDirTest):
    def test_reproducible_and_dirty(self):
        a, b = datagen.generate(500, seed=1), datagen.generate(500, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(datagen.generate(500, seed=2)))
        self.assertGreater(a.isna().sum().sum(), 0)
        self.assertGreater(a.duplicated().sum(), 0)

    def test_clean_data_is_tidy_and_balanced_enough(self):
        clean = datagen.generate(3000, dirty=False)
        self.assertEqual(clean.isna().sum().sum(), 0)
        self.assertEqual(set(clean["label"]), set(S.CLASSES))
        self.assertTrue((clean["text"].str.strip() != "").all())
        shares = clean["label"].value_counts(normalize=True)
        self.assertTrue(((shares > 0.12) & (shares < 0.55)).all(), shares.to_dict())      # Neutral is the smallest class

    def test_words_do_not_belong_to_one_class_by_accident(self):
        clean = datagen.generate(6000, dirty=False)
        great = clean[clean["text"].str.contains(r"\bgreat\b", case=False)]
        self.assertGreater((great["label"] == "Positive").sum(), 20)                        # genuine praise
        self.assertGreater((great["label"] == "Negative").sum(), 5)                         # and sarcasm
        self.assertGreater((great["label"] == "Positive").mean(), 0.5)

    def test_negation_changes_the_label(self):
        clean = datagen.generate(8000, dirty=False)
        low = clean["text"].str.lower()
        not_good = clean[low.str.contains(r"\bnot (?:good|nice|decent|pleasant|solid|reliable|smooth)\b") & ~low.str.contains("but|however|although")]
        not_bad = clean[low.str.contains(r"\bnot (?:bad|poor|slow|bland|sloppy)\b") & ~low.str.contains("but|however|although")]
        self.assertGreater(len(not_good), 20)
        self.assertGreater(len(not_bad), 20)
        self.assertGreater((not_good["label"] == "Negative").mean(), 0.6)
        self.assertGreater((not_bad["label"] == "Positive").mean(), 0.5)

    def test_the_hard_cases_exist(self):
        clean = datagen.generate(4000, dirty=False)
        text = clean["text"]
        self.assertTrue(text.str.contains(r"\bbut\b|However", regex=True).any())            # mixed feelings
        self.assertTrue(text.map(lambda t: bool(re.search(r"(\w)\1{2,}", t))).any())         # stretched words
        self.assertTrue(text.str.contains(r"[^\x00-\x7f]", regex=True).any())                # emoji
        self.assertTrue(text.str.contains(r"\b[A-Z]{4,}\b", regex=True).any())               # shouting
        self.assertTrue(text.str.contains("Oh great|Wow, thanks|Just what I needed|Fantastic, the", regex=True).any())   # sarcasm

    def test_sample_data_is_cleanable(self):
        cleaned, report = P.clean(datagen.generate(3000))
        self.assertGreater(report.rows_dropped, 60)
        self.assertLess(report.rows_dropped, 500)
        self.assertEqual(cleaned["text"].str.lower().duplicated().sum(), 0)


class TestLoading(TempDirTest):
    def write(self, name, text, binary=False):
        path = self.dir / name
        path.write_bytes(text) if binary else path.write_text(text, encoding="utf-8")
        return path

    def test_common_column_names_are_understood(self):
        df = datagen.generate(200, dirty=False).rename(columns={"label": "Sentiment", "text": "Review"})
        path = self.dir / "named.csv"
        df.to_csv(path, index=False)
        loaded = P.load_dataset(path)
        self.assertEqual(list(loaded.columns), ["label", "text"])
        self.assertEqual(len(loaded), 200)
        df = datagen.generate(200, dirty=False).rename(columns={"label": "v1", "text": "v2"})
        path = self.dir / "kaggle.csv"
        df.to_csv(path, index=False)
        self.assertEqual(len(P.load_dataset(path)), 200)

    def test_star_ratings_become_labels(self):
        clean = datagen.generate(300, dirty=False)
        stars = clean["label"].map({"Negative": 1, "Neutral": 3, "Positive": 5})
        path = self.dir / "stars.csv"
        pd.DataFrame({"Review": clean["text"], "Stars": stars}).to_csv(path, index=False)
        loaded = P.load_dataset(path)
        self.assertEqual(loaded["label"].tolist(), clean["label"].tolist())
        self.assertIn("star", loaded.attrs["note"].lower())
        cleaned, report = P.clean(loaded)
        self.assertTrue(any("stars" in note.lower() for note in report.notes))
        self.assertEqual(set(cleaned["label"]), {0, 1, 2})

    def test_non_utf8_file_is_read_and_noted(self):
        df = datagen.generate(300, dirty=False)
        df.loc[0, "text"] = "Café was lovely"
        path = self.dir / "latin.csv"
        path.write_bytes(df.to_csv(index=False).encode("cp1252", errors="replace"))       # emoji become "?"
        loaded = P.load_dataset(path)
        self.assertIn("Windows-1252", loaded.attrs["note"])
        self.assertIn("Café", loaded.loc[0, "text"])

    def test_problem_files_are_explained(self):
        cases = {
            "missing.csv": (None, "not found"),
            "empty.csv": ("", "is empty"),
            "cols.csv": ("name,age\nAsha,20\n", "needs a text column"),
            "nolabel.csv": ("review,length\nhello there,5\n", "needs a label column"),
            "small.csv": (datagen.generate(60, dirty=False).to_csv(index=False), "only 60 rows"),
        }
        for name, (text, hint) in cases.items():
            path = self.dir / name if text is None else self.write(name, text)
            with self.assertRaises(DataError, msg=name) as ctx:
                P.load_dataset(path)
            self.assertIn(hint, str(ctx.exception))


class TestCleaning(TempDirTest):
    def setUp(self):
        super().setUp()
        self.base = P.clean(datagen.generate(500, dirty=False, seed=5))[0]          # already tidy and unique
        self.shown = self.base.assign(label=self.base["label"].map(lambda v: S.CLASSES[v]))

    def with_rows(self, *rows):
        return pd.concat([self.shown, pd.DataFrame(rows, columns=["label", "text"])], ignore_index=True)

    def test_clean_data_passes_through_untouched(self):
        out, report = P.clean(self.shown)
        self.assertEqual((report.rows_in, report.rows_out), (len(self.base), len(self.base)))
        self.assertEqual(report.steps, [])

    def test_rows_that_cannot_be_used_are_removed(self):
        label_of = lambda i: S.CLASSES[self.base["label"].iloc[i]]
        removed = {
            "missing text": self.with_rows(("Positive", np.nan)), "number as text": self.with_rows(("Positive", 12345)),
            "blank text": self.with_rows(("Neutral", "   \n\t")), "too short": self.with_rows(("Negative", "?")),
            "missing label": self.with_rows((np.nan, "a brand new review about nothing")),
            "unrecognised label": self.with_rows(("mixed", "another brand new review")),
            "exact duplicate": self.with_rows((label_of(0), self.base["text"].iloc[0])),
            "duplicate in other capitals": self.with_rows((label_of(1), self.base["text"].iloc[1].upper())),
        }
        for name, df in removed.items():
            out, report = P.clean(df)
            self.assertEqual(report.rows_dropped, 1, name)
            self.assertEqual(len(out), len(self.base), name)

    def test_same_text_with_different_labels_is_removed_entirely(self):
        text = self.base["text"].iloc[0]
        other = S.CLASSES[(self.base["label"].iloc[0] + 1) % 3]
        out, report = P.clean(self.with_rows((other, text)))
        self.assertEqual(len(out), len(self.base) - 1)
        self.assertNotIn(text.lower(), set(out["text"].str.lower()))
        self.assertTrue(any("different labels" in step[0] for step in report.steps))

    def test_html_whitespace_and_length_are_handled_not_dropped(self):
        out, report = P.clean(self.with_rows(("Positive", "<p>Loved   it &amp; will buy again</p>")))
        self.assertEqual(out["text"].iloc[-1], "Loved it & will buy again")
        self.assertTrue(any("HTML" in step[0] for step in report.steps))
        out, report = P.clean(self.with_rows(("Negative", "  never\t\n buying   again\x07 ")))
        self.assertEqual(out["text"].iloc[-1], "never buying again")
        self.assertTrue(any("whitespace" in step[0] for step in report.steps))
        long, report = P.clean(self.with_rows(("Neutral", "word " * 2500)))
        self.assertEqual(long["text"].str.len().max(), S.MAX_TEXT)
        self.assertTrue(any("truncated" in step[2] for step in report.steps))

    def test_label_spellings_are_standardised(self):
        rows = [(" POS ", "unique alpha review one"), ("+1", "unique beta review two"), ("good", "unique gamma review three"),
                ("neu", "unique delta review four"), ("0", "unique epsilon review five"), ("neg", "unique zeta review six"),
                ("-1", "unique eta review seven"), ("Negative ", "unique theta review eight")]
        out, report = P.clean(self.with_rows(*rows))
        self.assertEqual(out.tail(8)["label"].tolist(), [2, 2, 2, 1, 1, 0, 0, 0])
        self.assertEqual(out["label"].dtype, np.dtype("int64"))

    def test_input_is_not_modified_and_accounting_adds_up(self):
        raw = datagen.generate(1200)
        before = raw.copy()
        out, report = P.clean(raw)
        pd.testing.assert_frame_equal(raw, before)
        removed = sum(count for _, count, outcome in report.steps if outcome.startswith("removed"))
        self.assertEqual(report.rows_dropped, removed)
        self.assertEqual(report.rows_out, len(out))
        self.assertEqual(sum(report.class_counts.values()), len(out))
        self.assertEqual(set(out["label"]), {0, 1, 2})

    def test_too_little_usable_data_is_refused(self):
        swapped = datagen.generate(400, dirty=False).rename(columns={"label": "text", "text": "label"})
        with self.assertRaises(DataError) as ctx:
            P.clean(swapped)
        self.assertIn("usable rows remain", str(ctx.exception))
        no_neutral = datagen.generate(600, dirty=False)
        no_neutral = no_neutral[no_neutral["label"] != "Neutral"]
        with self.assertRaises(DataError) as ctx:
            P.clean(no_neutral)
        self.assertIn("neutral reviews remain", str(ctx.exception))


class TestFeatures(TempDirTest):
    def test_features_use_only_the_text(self):
        X, y = P.split_xy(P.clean(datagen.generate(300, dirty=False))[0])
        self.assertEqual(list(X.columns), ["text"])
        self.assertEqual(set(y), {0, 1, 2})

    def test_pipelines_cope_with_odd_input(self):
        X, y = P.split_xy(P.clean(datagen.generate(600, dirty=False))[0])
        odd = pd.DataFrame({"text": ["", "😍😍😍", "zzzzqqqq unseenword", "ok", "你好，很好用，推荐", "A" * 5000, "!!!???"]})
        for name in ("Naive Bayes", "Logistic Regression", "Linear SVM", M.LEXICON):
            pipe = M.build_pipeline(M.candidates()[name]).fit(X, y)
            p = pipe.predict_proba(odd)
            self.assertEqual(p.shape, (7, 3), name)
            self.assertTrue(np.isfinite(p).all(), name)
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_negation_features_make_a_difference(self):
        X, y = P.split_xy(P.clean(datagen.generate(1500, dirty=False, seed=3))[0])
        pipe = M.build_pipeline(M.candidates()["Logistic Regression"]).fit(X, y)
        good = pipe.predict_proba(pd.DataFrame({"text": ["The battery was good", "The battery was not good"]}))
        self.assertGreater(good[0, S.POSITIVE], 0.5)
        self.assertGreater(good[1, S.NEGATIVE], good[1, S.POSITIVE])
