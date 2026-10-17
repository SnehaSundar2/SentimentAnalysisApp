"""The web interface (over real HTTP on a random local port) and the interactive menu."""

import http.client
import io
import json
import threading
import urllib.parse
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

from sentimentapp import datagen, web
from sentimentapp.cli import SentimentCLI
from sentimentapp.predictor import Predictor
from tests.helpers import MIXED, NEGATIVE, POSITIVE, TempDirTest, shared


class TestWeb(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.server = web.make_server(Predictor.from_directory(shared()[1]), port=0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        return resp.status, data, {k.lower(): v for k, v in resp.getheaders()}

    def post_form(self, text):
        return self.request("POST", "/", urllib.parse.urlencode({"text": text}),
                            {"Content-Type": "application/x-www-form-urlencoded"})

    def post_json(self, payload):
        return self.request("POST", "/api/predict", json.dumps(payload), {"Content-Type": "application/json"})

    def test_pages_and_security_headers(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Analyse sentiment", body)
        self.assertNotIn("<script", body.lower())                          # the page needs no scripts at all
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/elsewhere", b"")[0], 404)

    def test_examples_prefill_the_form(self):
        status, body, _ = self.request("GET", "/?example=mixed")
        self.assertEqual(status, 200)
        self.assertIn("The food was great but the service", body)
        for key in ("positive", "negative", "neutral", "negation", "sarcasm"):
            self.assertIn(f"/?example={key}", body)                        # all six example links are offered
        status, body, _ = self.request("GET", "/?example=<script>")
        self.assertEqual(status, 200)
        self.assertNotIn("<script>", body)

    def test_positive_result_page(self):
        status, body, _ = self.post_form(POSITIVE)
        self.assertEqual(status, 200)
        for part in ("chip pos", "sentiment score", "What moves the result", "Most similar known reviews", "<mark title"):
            self.assertIn(part, body)
        self.assertIn(">Positive<", body)
        self.assertIn("Absolutely loved", body)

    def test_negative_result_page(self):
        status, body, _ = self.post_form(NEGATIVE)
        self.assertEqual(status, 200)
        self.assertIn("chip neg", body)
        self.assertIn(">Negative<", body)
        self.assertIn("−100 negative", body)

    def test_mixed_text_shows_part_by_part(self):
        status, body, _ = self.post_form("The food was fantastic but the waiter was terrible. I wanted to love this place.")
        self.assertEqual(status, 200)
        self.assertIn("Part by part", body)
        self.assertIn("class=\"part\"", body)

    def test_a_doubtful_text_says_not_sure(self):
        status, body, _ = self.post_form("It is what it is, I guess, honestly maybe fine")
        self.assertEqual(status, 200)
        self.assertTrue(("chip unsure" in body) or ("% confident" in body))     # either honest doubt or a confident answer

    def test_empty_or_oversized_text_is_refused_and_named(self):
        status, body, _ = self.post_form("   ")
        self.assertEqual(status, 422)
        self.assertIn("Enter some text to analyse.", body)
        status, body, _ = self.post_form("great " * 1500)
        self.assertEqual(status, 422)
        self.assertIn("too long", body)
        self.assertIn("great great great", body)                           # what the user typed is still there

    def test_scripts_and_markup_in_the_text_are_escaped(self):
        evil = '"><script>alert(1)</script> loved it <img src=x onerror=alert(2)> http://x-win.xyz/<b>'
        status, body, _ = self.post_form(evil)
        self.assertEqual(status, 200)
        self.assertNotIn("<script>alert", body)
        self.assertNotIn("<img src=x", body)

    def test_json_api(self):
        status, body, headers = self.post_json({"text": POSITIVE})
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(data["sentiment"], "Positive")
        self.assertGreater(data["score"], 40)
        self.assertAlmostEqual(sum(data["probabilities"].values()), 1.0, places=2)
        status, body, _ = self.post_json({"text": ""})
        self.assertEqual(status, 422)
        self.assertIn("text", json.loads(body)["fields"])
        self.assertEqual(self.post_json({"review": POSITIVE})[0], 422)       # the wrong key is not silently accepted
        health = json.loads(self.request("GET", "/api/health")[1])
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["classes"], ["Negative", "Neutral", "Positive"])

    def test_bad_requests_are_refused(self):
        self.assertEqual(self.request("POST", "/api/predict", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/predict", "{not json", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", '["a list"]', {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", "x" * 60_000, {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/", "x" * 60_000, {"Content-Type": "application/x-www-form-urlencoded"})[0], 413)
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example.com"})[0], 403)   # DNS-rebinding guard
        self.assertEqual(self.request("GET", "/", headers={"Host": f"localhost:{self.port}"})[0], 200)


def run_cli(cli, inputs):
    out = io.StringIO()
    with patch("builtins.input", side_effect=inputs), redirect_stdout(out):
        cli.run()
    return out.getvalue()


class TestCLI(TempDirTest):
    def make(self):
        return SentimentCLI(shared()[1], self.dir / "reports")

    def test_analyse_session(self):
        out = run_cli(self.make(), ["1", POSITIVE, "0"])
        for part in ("Sentiment:", "POSITIVE", "Probabilities:", "What moves the result", "Most similar known reviews"):
            self.assertIn(part, out)
        out = run_cli(self.make(), ["1", NEGATIVE, "0"])
        self.assertIn("NEGATIVE", out)

    def test_mixed_text_prints_its_parts(self):
        out = run_cli(self.make(), ["1", "The food was fantastic but the waiter was terrible. I wanted to love this place.", "0"])
        self.assertIn("Part by part", out)

    def test_empty_text_gives_a_friendly_error(self):
        out = run_cli(self.make(), ["1", "   ", "0"])
        self.assertIn("Enter some text to analyse", out)
        self.assertNotIn("Sentiment:", out)

    def test_reports_and_menu_errors(self):
        out = run_cli(self.make(), ["5", "6", "7", "", "99", "abc", "0"])
        self.assertIn("MODEL REPORT", out)
        self.assertIn("MODEL COMPARISON", out)
        self.assertIn("Report saved to", out)
        self.assertTrue((self.dir / "reports" / "model_report.html").exists())
        self.assertEqual(out.count("Invalid option"), 2)

    def test_batch_analysis_through_the_menu(self):
        source = self.dir / "in.csv"
        pd.DataFrame({"review": [POSITIVE, NEGATIVE, "", MIXED]}).to_csv(source, index=False)
        target = self.dir / "out.csv"
        out = run_cli(self.make(), ["2", str(source), str(target), "0"])
        self.assertIn("Analysed 3 text(s)", out)
        self.assertIn("Positive", out)
        self.assertIn("Line 4:", out)
        self.assertTrue(target.exists())
        missing = run_cli(self.make(), ["2", str(self.dir / "nothing.csv"), "", "0"])
        self.assertIn("File not found", missing)

    def test_dataset_tools(self):
        path = self.dir / "mine.csv"
        out = run_cli(self.make(), ["8", "400", "", str(path), "9", str(path), "0"])
        self.assertIn("Saved 400 reviews", out)
        self.assertIn("rows read", out)
        self.assertIn("Kept:", out)
        out = run_cli(self.make(), ["8", "400", "", str(path), "0"])
        self.assertIn("already exists", out)

    def test_no_model_gives_instructions_and_training_works(self):
        empty = SentimentCLI(self.dir / "nomodel", self.dir / "reports")
        out = run_cli(empty, ["n", "1", "0"])
        self.assertIn("No trained model yet", out)
        self.assertIn("Train one first", out)
        data = self.dir / "train.csv"
        datagen.generate(700).to_csv(data, index=False)
        out = run_cli(empty, ["n", "4", str(data), "y", "0"])             # quick mode training from a CSV
        self.assertIn("MODEL REPORT", out)
        self.assertIn("Model saved to", out)
        self.assertTrue((self.dir / "nomodel" / "model.joblib").exists())

    def test_training_from_star_ratings_through_the_menu(self):
        clean = datagen.generate(700, dirty=False)
        stars = clean["label"].map({"Negative": 1, "Neutral": 3, "Positive": 5})
        data = self.dir / "stars.csv"
        pd.DataFrame({"review": clean["text"], "rating": stars}).to_csv(data, index=False)
        out = run_cli(SentimentCLI(self.dir / "stars_model", self.dir / "reports"), ["n", "4", str(data), "y", "0"])
        self.assertIn("Labels were taken from the 'rating' column", out)
        self.assertIn("Model saved to", out)
