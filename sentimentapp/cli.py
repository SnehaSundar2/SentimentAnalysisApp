"""Interactive menu-driven command-line interface.

All SentimentError subclasses are caught here and shown as friendly messages; unexpected errors are
logged and the menu keeps running.
"""

import logging
from pathlib import Path

from . import datagen, persistence, report, schema as S, training
from . import preprocessing as P
from . import validators as v
from .exceptions import SentimentError, ValidationError
from .predictor import Predictor
from .web import serve

log = logging.getLogger("sentimentapp")

MENU = """
================== SENTIMENT ANALYSIS APP ==================
 ANALYSE                            MODEL
  1. Analyse a text                  4. Train a new model
  2. Analyse texts from a CSV        5. Model report (accuracy, mistakes)
  3. Start the web interface         6. Compare models
                                     7. Save an HTML report
 DATA
  8. Create a sample dataset (CSV)   9. Check a dataset's quality
  0. Exit
============================================================"""

SYNTHETIC_ROWS = 3000


def ask(prompt):
    return input(f"  {prompt}: ").strip()


def confirm(prompt, default=False):
    answer = ask(f"{prompt} [{'Y/n' if default else 'y/N'}]").lower()
    return default if not answer else answer in {"y", "yes"}


class SentimentCLI:
    def __init__(self, model_dir="models", reports_dir="reports"):
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)
        self._predictor = None
        self.actions = {"1": self.analyze, "2": self.analyze_csv, "3": self.web, "4": self.train, "5": self.model_report,
                        "6": self.compare, "7": self.save_report, "8": self.make_dataset, "9": self.check_quality}

    # ---------------------------------------------------------- plumbing
    def predictor(self):
        if self._predictor is None:
            self._predictor = Predictor.from_directory(self.model_dir)
            for warning in self._predictor.load_warnings:
                print(f"  ! {warning}")
        return self._predictor

    def run(self):
        if not persistence.is_trained(self.model_dir):
            print("\n  No trained model yet.")
            if confirm("Train one now on the built-in sample data (about half a minute)?", default=True):
                self._run_action(self.train_default)
        while True:
            print(MENU)
            choice = ask("Choose an option")
            if choice == "0":
                print("  Goodbye!")
                return
            action = self.actions.get(choice)
            if action is None:
                print("  ! Invalid option, please enter a number from the menu.")
                continue
            self._run_action(action)

    def _run_action(self, action):
        try:
            action()
        except SentimentError as exc:
            log.warning("%s failed: %s", action.__name__, exc)
            print(f"  ! Error: {exc}")
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
        except Exception:  # safety net; the full traceback goes to the log file
            log.exception("Unexpected error in %s", action.__name__)
            print("  ! An unexpected error occurred. See sentimentapp.log for details.")

    # ---------------------------------------------------------- analysing
    def show_result(self, a):
        print(f"\n  Sentiment:        {a.verdict.upper()}   (score {a.score:+.0f} of ±100: {a.tone.lower()})")
        print("  Probabilities:    " + "   ".join(f"{name} {p * 100:.0f}%" for name, p in a.probabilities.items()))
        if a.needs_review:
            print(f"  ! Not sure: it leans {a.label.lower()} with {a.confidence * 100:.0f}% confidence "
                  f"(below {a.confidence_threshold * 100:.0f}%), so please read it yourself.")
        for warning in a.warnings:
            print(f"  ! {warning}")
        if a.parts:
            print("\n  Part by part:")
            for part in a.parts:
                shown = part["text"] if len(part["text"]) <= 70 else part["text"][:69] + "…"
                print(f"   {part['label']:<9} {shown}")
        if a.signals:
            print("\n  What stands out:")
            for signal in a.signals:
                print(f"   - {signal}")
        if a.supporting or a.opposing:
            print(f"\n  What moves the result (each piece taken out and the text re-scored; 'odds x2' doubles the odds of {a.label.lower()}):")
            top = max(abs(r["effect"]) for r in a.supporting + a.opposing) or 1
            for r in a.supporting:
                print(f"   supports  odds ×{r['odds']:<5.1f} {r['word'][:40]:<40} {report.bar(r['effect'], top, 12)}")
            for r in a.opposing:
                print(f"   against   odds ÷{1 / r['odds']:<5.1f} {r['word'][:40]:<40} {report.bar(abs(r['effect']), top, 12)}")
        if a.similar:
            print("\n  Most similar known reviews:")
            for c in a.similar:
                print(f"   [{c['label']:<8}] {c['similarity']:.2f}  {c['text']}")

    def analyze(self):
        predictor = self.predictor()
        self.show_result(predictor.analyze(ask("Text to analyse (one line)")))

    def analyze_csv(self):
        predictor = self.predictor()
        print("  The CSV needs a text column (called text, review, comment, message, tweet, content or body).")
        source = ask("CSV file with texts")
        default = Path("predictions") / f"{Path(source).stem}_sentiment.csv"
        target = ask(f"Save results to [{default}]") or default
        ok, errors, s = predictor.predict_csv(source, target)
        print(f"  Analysed {ok} text(s); saved to {target}.")
        print(f"  Positive {s['positive']} · neutral {s['neutral']} · negative {s['negative']} · not sure {s['not_sure']} "
              f"· average score {s['average_score']:+.0f}")
        if errors:
            print(f"  {len(errors)} row(s) could not be analysed:")
            for line in errors[:15]:
                print(f"    - {line}")

    def web(self):
        port = v.validate_port(ask("Port [8040]") or 8040)
        serve(self.predictor(), port)

    # ---------------------------------------------------------- training and reports
    def train_default(self):
        self._train(datagen.generate(SYNTHETIC_ROWS), "synthetic (generated)", fast=False, tune=True)

    def train(self):
        source = ask("CSV file to train on (Enter = generate the built-in sample data)")
        if source:
            df, label = P.load_dataset(source), Path(source.strip('"')).name
        else:
            rows = v.validate_int(ask(f"Reviews to generate [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 500, 30000)
            df, label = datagen.generate(rows), "synthetic (generated)"
        fast = confirm("Quick mode (a little less accurate, a little faster)?")
        tune = False if fast else confirm("Tune the best model? (adds a few seconds)", default=True)
        self._train(df, label, fast, tune)

    def _train(self, df, label, fast, tune):
        print()
        result = training.train(df, fast=fast, tune=tune, source=label, progress=lambda m: print(f"   {m}"))
        persistence.save(result, self.model_dir)
        self._predictor = None
        print("\n  Data cleaning:")
        print("\n".join("  " + line for line in result.cleaning.lines()))
        print()
        print(report.comparison_text(result.metadata))
        print()
        print(report.model_text(result.metadata))
        print(f"\n  Model saved to {self.model_dir}/ — you can now analyse text.")

    def model_report(self):
        print(report.model_text(self.predictor().meta))

    def compare(self):
        print(report.comparison_text(self.predictor().meta))

    def save_report(self):
        default = self.reports_dir / "model_report.html"
        path = Path(ask(f"File name [{default}]") or default)
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")
        report.save_html(self.predictor().meta, path)
        print(f"  Report saved to {path} — open it in a web browser.")

    # ---------------------------------------------------------- data
    def make_dataset(self):
        rows = v.validate_int(ask(f"Number of reviews [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 150, 50000)
        dirty = not ask("Include data problems for practising cleaning? [Y/n]").lower().startswith("n")
        default = Path("sample_data") / ("reviews_dirty.csv" if dirty else "reviews_clean.csv")
        path = Path(ask(f"File name [{default}]") or default)
        if path.exists():
            raise ValidationError(f"{path} already exists; choose another name.")
        path.parent.mkdir(parents=True, exist_ok=True)
        datagen.generate(rows, dirty=dirty).to_csv(path, index=False)
        print(f"  Saved {rows} reviews to {path} (synthetic data).")

    def check_quality(self):
        df = P.load_dataset(ask("CSV file to check"))
        _, cleaning = P.clean(df)
        print("\n".join("  " + line for line in cleaning.lines()))
