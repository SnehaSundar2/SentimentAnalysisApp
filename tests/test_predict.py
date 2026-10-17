"""Input validation, model persistence, analysing texts, explanations, parts of a review and batch files."""

import json

import numpy as np
import pandas as pd

from sentimentapp import persistence, schema as S, validators as v
from sentimentapp.exceptions import DataError, ModelError, NotTrainedError, ValidationError
from sentimentapp.predictor import Predictor, highlight_pieces, require_predictor
from tests.helpers import MIXED, NEGATION, NEGATIVE, NEUTRAL, POSITIVE, SARCASM, TempDirTest, shared


class TestValidation(TempDirTest):
    def test_good_texts_are_tidied(self):
        text, warnings = v.validate_text("  <b>Loved</b>\tit &amp;\n\nwill   buy again\x07 ")
        self.assertEqual(text, "Loved it & will buy again")
        self.assertEqual(warnings, [])

    def test_unusable_texts_are_refused_with_a_reason(self):
        for bad in (None, "", "   \n\t", float("nan"), "<br /><br />"):
            with self.assertRaises(ValidationError, msg=repr(bad)) as ctx:
                v.validate_text(bad)
            self.assertIn("Enter some text", str(ctx.exception))
            self.assertIn("text", ctx.exception.errors)
        with self.assertRaises(ValidationError) as ctx:
            v.validate_text(12345)
        self.assertIn("string", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            v.validate_text("great " * 1500)
        self.assertIn("too long", str(ctx.exception))
        v.validate_text("x" * S.MAX_TEXT)                                      # exactly the limit is fine

    def test_warnings_for_texts_the_model_cannot_judge_well(self):
        _, warnings = v.validate_text("good")
        self.assertTrue(any("short" in w for w in warnings))
        _, warnings = v.validate_text("非常好用，强烈推荐给所有的朋友，质量很好")
        self.assertTrue(any("English" in w for w in warnings))
        _, warnings = v.validate_text("the delivery was quick and the box was fine")
        self.assertEqual(warnings, [])

    def test_settings_validation(self):
        self.assertEqual(v.validate_port("8040"), 8040)
        self.assertEqual(v.validate_int("5", "Rows", 1, 10), 5)
        for bad in ("abc", "80", "70000", ""):
            with self.assertRaises(ValidationError):
                v.validate_port(bad)
        for bad in ("x", "0", "11", ""):
            with self.assertRaises(ValidationError):
                v.validate_int(bad, "Rows", 1, 10)


class TestPersistence(TempDirTest):
    def test_round_trip_gives_identical_predictions(self):
        result, model_dir, _ = shared()
        a = Predictor(persistence.load(model_dir)).analyze(POSITIVE)
        b = Predictor(persistence.load(model_dir)).analyze(POSITIVE)
        self.assertEqual(a.probabilities, b.probabilities)
        original = result.pipeline.predict_proba(pd.DataFrame({"text": [POSITIVE]}))[0]
        np.testing.assert_allclose([a.probabilities[c] for c in S.CLASSES], original, atol=1e-9)
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(set(meta["files"]), {"model.joblib", "reference.csv"})
        self.assertEqual(len(meta["files"]["model.joblib"]), 64)
        self.assertEqual([p.name for p in model_dir.glob("*.tmp")], [])        # atomic saves leave no temp files

    def test_no_model_gives_instructions(self):
        for call in (lambda: persistence.load(self.dir / "empty"), lambda: require_predictor(self.dir / "empty")):
            with self.assertRaises(NotTrainedError) as ctx:
                call()
            self.assertIn("Train one first", str(ctx.exception))

    def test_tampering_and_damage_are_detected(self):
        model = self.copy_model()
        data = bytearray((model / "model.joblib").read_bytes())
        data[len(data) // 2] ^= 0x01
        (model / "model.joblib").write_bytes(bytes(data))
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("checksum", str(ctx.exception))

        model = self.copy_model()
        (model / "reference.csv").unlink()
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("missing", str(ctx.exception))

        model = self.copy_model()
        meta_path = model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["files"]["model.joblib"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ModelError):
            persistence.load(model)

        meta_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("damaged", str(ctx.exception))

        meta_path.write_text(json.dumps({"model": "x"}), encoding="utf-8")
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("no checksums", str(ctx.exception))

    def test_library_version_mismatch_warns(self):
        model = self.copy_model()
        meta_path = model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["environment"]["sklearn"] = "0.20.0"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        loaded = persistence.load(model)
        self.assertTrue(any("0.20.0" in w for w in loaded.warnings))


class TestAnalysis(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor.from_directory(shared()[1])

    def analyze(self, text, **kw):
        return self.predictor.analyze(text, **kw)

    def test_structure_and_json(self):
        a = self.analyze(POSITIVE)
        self.assertAlmostEqual(sum(a.probabilities.values()), 1.0)
        self.assertEqual(set(a.probabilities), set(S.CLASSES))
        self.assertEqual(a.label, max(a.probabilities, key=a.probabilities.get))
        self.assertEqual(a.confidence, max(a.probabilities.values()))
        self.assertEqual(a.score, 100 * (a.probabilities["Positive"] - a.probabilities["Negative"]))
        self.assertEqual(a.tone, S.tone(a.score))
        self.assertEqual(a.needs_review, a.confidence < self.predictor.threshold)
        self.assertEqual(a.verdict, "Not sure" if a.needs_review else a.label)
        data = a.to_dict()
        for key in ("sentiment", "most_likely", "confidence", "score", "tone", "needs_review", "probabilities", "warnings",
                    "signals", "words_supporting", "words_opposing", "parts", "similar_reviews", "confidence_threshold"):
            self.assertIn(key, data)
        json.dumps(data)                                                           # everything can be sent as JSON

    def test_clear_examples_get_the_right_sentiment(self):
        for text, expected, sign in ((POSITIVE, "Positive", 1), (NEGATIVE, "Negative", -1), (NEGATION, "Negative", -1)):
            a = self.analyze(text)
            self.assertEqual(a.verdict, expected, text)
            self.assertGreater(sign * a.score, 40, text)
        self.assertNotEqual(self.analyze(SARCASM).verdict, "Positive")             # sarcasm: wrong is forgivable, praise is not
        self.assertIn(self.analyze(NEUTRAL).verdict, ("Neutral", "Not sure"))      # a plain fact is not strong feeling
        mixed = self.analyze(MIXED)                                                # mixed feelings: no strong claim either way
        self.assertTrue(mixed.needs_review or abs(mixed.score) < 90)

    def test_negation_flips_the_answer(self):
        good = self.analyze("The battery was good", explain=False, parts=False, similar=False)
        not_good = self.analyze("The battery was not good", explain=False, parts=False, similar=False)
        self.assertGreater(good.score, 40)
        self.assertLess(not_good.score, -40)
        self.assertGreater(good.score - not_good.score, 80)
        awful = self.analyze("The battery was terrible", explain=False, parts=False, similar=False)
        not_awful = self.analyze("The battery wasn't terrible", explain=False, parts=False, similar=False)
        self.assertGreater(not_awful.score, awful.score + 40)

    def test_shouting_and_stretching_do_not_change_the_answer(self):
        for text in (POSITIVE, NEGATIVE):
            original = self.analyze(text, explain=False, parts=False, similar=False)
            shouted = self.analyze(text.upper(), explain=False, parts=False, similar=False)
            self.assertEqual(original.label, shouted.label)
        self.assertEqual(self.analyze("sooo gooood", explain=False, parts=False, similar=False).label, "Positive")

    def test_the_same_text_always_gets_the_same_score(self):
        a, b = self.analyze(POSITIVE), self.analyze("  " + POSITIVE + "\n")
        self.assertEqual(a.probabilities, b.probabilities)

    def test_explanation_points_at_the_words_and_is_sorted(self):
        a = self.analyze(POSITIVE)
        words = {r["word"].lower() for r in a.supporting}
        self.assertTrue(words & {"loved", "recommend", "😍", "would", "so"}, words)
        effects = [r["effect"] for r in a.supporting]
        self.assertEqual(effects, sorted(effects, reverse=True))
        self.assertTrue(all(r["effect"] >= 0.25 and r["odds"] > 1 for r in a.supporting))
        self.assertTrue(all(r["effect"] <= -0.25 and r["odds"] < 1 for r in a.opposing))
        self.assertLessEqual(len(a.supporting), 6)
        top = a.supporting[0]["word"]
        without = a.text.replace(top, "") if top in a.text else a.text
        again = self.analyze(without, explain=False, parts=False, similar=False)
        self.assertLess(again.probabilities[a.label], a.probabilities[a.label])      # taking the strongest word out lowers the score

    def test_negation_words_support_a_negative_reading(self):
        a = self.analyze("The battery was not good")
        self.assertEqual(a.label, "Negative")
        self.assertTrue({r["word"].lower() for r in a.supporting} & {"not", "good"})

    def test_highlighting_keeps_the_text_intact(self):
        a = self.analyze(POSITIVE)
        chunks = highlight_pieces(a.text, a.effects)
        self.assertEqual("".join(chunk for chunk, _ in chunks), a.text)
        self.assertTrue(all(effect is None or effect >= 0.25 for _, effect in chunks))
        marked = [chunk for chunk, effect in chunks if effect is not None]
        self.assertTrue(marked)
        link = self.analyze("Loved it, details at http://shop.example.com/page 😍")
        parts = highlight_pieces(link.text, link.effects)
        self.assertEqual("".join(c for c, _ in parts), link.text)
        self.assertIn("😍", [c for c, _ in parts])                                   # emoji are separate chunks

    def test_mixed_reviews_are_judged_part_by_part(self):
        a = self.analyze("The food was fantastic but the waiter was terrible. I wanted to love this place.")
        self.assertGreaterEqual(len(a.parts), 2)
        self.assertTrue(all(p["label"] in S.CLASSES and 0 < p["confidence"] <= 1 for p in a.parts))
        labels = [p["label"] for p in a.parts]
        self.assertIn("Positive", labels)
        self.assertIn("Negative", labels)
        self.assertEqual(self.analyze("Just one sentence here", parts=True).parts, [])

    def test_signals_and_warnings(self):
        a = self.analyze("I am NOT happy!!! NEVER again 😡")
        self.assertTrue(any("Negation" in s for s in a.signals))
        self.assertTrue(any("emoji" in s for s in a.signals))
        self.assertTrue(any("SHOUTED" in s for s in a.signals))
        self.assertEqual(self.analyze(POSITIVE).warnings, [])
        self.assertTrue(any("short" in w for w in self.analyze("good").warnings))

    def test_similar_known_reviews(self):
        a = self.analyze(NEGATIVE)
        self.assertEqual(len(a.similar), 3)
        self.assertTrue(all(s["label"] in S.CLASSES for s in a.similar))
        sims = [s["similarity"] for s in a.similar]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertGreaterEqual(sum(s["label"] == "Negative" for s in a.similar), 2)
        self.assertEqual(self.analyze("qazwsx edcrfv tgbyhn", explain=False).similar, [])        # no resemblance: nothing invented

    def test_html_in_the_input_is_removed(self):
        a = self.analyze("<b>Loved</b> it &amp; will buy again", explain=False, parts=False, similar=False)
        self.assertEqual(a.text, "Loved it & will buy again")
        self.assertEqual(a.label, "Positive")

    def test_invalid_input_never_reaches_the_model(self):
        for bad in ("", None, "   ", "x " * 3000, 42, "<br/>"):
            with self.assertRaises(ValidationError):
                self.analyze(bad)

    def test_odd_but_valid_input_does_not_crash(self):
        for text in ("😍😍😍", "A" * 5000, "http://a.xyz", "%s %d {0} <b>", "'; DROP TABLE reviews;--", "!!!???", "你好",
                     "not not not not not", "but but but"):
            a = self.analyze(text)
            self.assertAlmostEqual(sum(a.probabilities.values()), 1.0, places=6)
            self.assertTrue(-100 <= a.score <= 100)


class TestBatch(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor.from_directory(shared()[1])

    def setUp(self):
        super().setUp()
        self.source = self.dir / "reviews.csv"
        pd.DataFrame({"ID": ["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
                      "Review": [POSITIVE, NEGATIVE, "", "good", "great " * 1500, NEUTRAL, "<br/>"]}).to_csv(self.source, index=False)

    def test_good_rows_are_analysed_and_bad_rows_reported(self):
        out = self.dir / "result.csv"
        ok, errors, summary = self.predictor.predict_csv(self.source, out)
        self.assertEqual(ok, 4)
        self.assertEqual([e.split(":")[0] for e in errors], ["Line 4", "Line 6", "Line 8"])   # blank, too long, markup only
        frame = pd.read_csv(out, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        self.assertEqual(len(frame), 7)
        self.assertEqual(list(frame.columns[:2]), ["id", "review"])                           # original columns kept
        self.assertEqual(frame["id"].tolist(), [f"r{i}" for i in range(1, 8)])
        for col in ("sentiment", "confidence", "score", "p_negative", "p_neutral", "p_positive", "status"):
            self.assertIn(col, frame.columns)
        self.assertEqual(frame["sentiment"][0], "Positive")
        self.assertEqual(frame["sentiment"][1], "Negative")
        self.assertGreater(float(frame["score"][0]), 40)
        self.assertLess(float(frame["score"][1]), -40)
        self.assertEqual(frame["sentiment"][2], "")                                           # nothing invented for a bad row
        self.assertTrue(frame["status"][2].startswith("error:"))
        self.assertIn("short", frame["status"][3])                                            # judged, with a warning
        probabilities = frame.loc[0, ["p_negative", "p_neutral", "p_positive"]].astype(float)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=2)

    def test_summary_counts_add_up(self):
        out = self.dir / "result.csv"
        ok, _, s = self.predictor.predict_csv(self.source, out)
        self.assertEqual(s["analysed"], ok)
        self.assertEqual(s["negative"] + s["neutral"] + s["positive"] + s["not_sure"], ok)
        frame = pd.read_csv(out, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        self.assertEqual(int((frame["sentiment"] == "Not sure").sum()), s["not_sure"])
        self.assertEqual(int((frame["sentiment"] == "Positive").sum()), s["positive"])
        scores = [float(x) for x in frame["score"] if x != ""]
        self.assertAlmostEqual(s["average_score"], float(np.mean(scores)), delta=0.6)

    def test_not_sure_rows_say_what_they_lean_towards(self):
        path = self.dir / "doubt.csv"
        pd.DataFrame({"text": [MIXED, SARCASM, "It is what it is", "Honestly fine I suppose"]}).to_csv(path, index=False)
        out = self.dir / "doubt_out.csv"
        self.predictor.predict_csv(path, out)
        frame = pd.read_csv(out, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        unsure = frame[frame["sentiment"] == "Not sure"]
        self.assertTrue(all("leaning" in s for s in unsure["status"]))

    def test_other_column_names_work(self):
        path = self.dir / "tweets.csv"
        pd.DataFrame({"Tweet": [POSITIVE, NEGATIVE]}).to_csv(path, index=False)
        ok, errors, _ = self.predictor.predict_csv(path, self.dir / "o.csv")
        self.assertEqual((ok, errors), (2, []))

    def test_file_level_problems(self):
        out = self.dir / "result.csv"
        with self.assertRaises(DataError):
            self.predictor.predict_csv(self.dir / "missing.csv", out)
        no_cols = self.dir / "bad.csv"
        no_cols.write_text("name,age\nAsha,20\n", encoding="utf-8")
        with self.assertRaises(DataError) as ctx:
            self.predictor.predict_csv(no_cols, out)
        self.assertIn("needs a text column", str(ctx.exception))
        empty = self.dir / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaises(DataError):
            self.predictor.predict_csv(empty, out)
        self.predictor.predict_csv(self.source, out)
        with self.assertRaises(DataError) as ctx:
            self.predictor.predict_csv(self.source, out)                                      # never overwrite
        self.assertIn("already exists", str(ctx.exception))
