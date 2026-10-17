"""Entry point for the Sentiment Analysis App.

Usage:
    python main.py                                       # interactive menu
    python main.py --train                               # train on generated sample reviews
    python main.py --train --data my_reviews.csv --fast  # train on your own CSV (text + label or star rating)
    python main.py --analyze "The food was great but the service was slow"
    python main.py --serve --port 8040                   # local web interface
    python main.py --report reports/model_report.html    # HTML report of the trained model
    python main.py --generate sample_data/reviews.csv --rows 3000
"""

import argparse
import logging
import sys
from pathlib import Path

from sentimentapp import datagen, persistence, report, training
from sentimentapp import preprocessing as P
from sentimentapp import validators as v
from sentimentapp.cli import SYNTHETIC_ROWS, SentimentCLI
from sentimentapp.exceptions import SentimentError
from sentimentapp.predictor import require_predictor
from sentimentapp.web import serve


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # show emoji and block characters on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Sentiment Analysis App (machine learning project)")
    parser.add_argument("--model-dir", default="models", help="folder for the trained model (default models)")
    parser.add_argument("--train", action="store_true", help="train a model and save it")
    parser.add_argument("--data", metavar="CSV", help="with --train: your own dataset (default: generated sample reviews)")
    parser.add_argument("--rows", default=str(SYNTHETIC_ROWS), help="reviews to generate (default 3000)")
    parser.add_argument("--fast", action="store_true", help="with --train: skip hyper-parameter tuning")
    parser.add_argument("--no-tune", action="store_true", help="with --train: skip hyper-parameter tuning")
    parser.add_argument("--analyze", metavar="TEXT", help="analyse one text")
    parser.add_argument("--serve", action="store_true", help="start the local web interface")
    parser.add_argument("--port", default="8040", help="web interface port (default 8040)")
    parser.add_argument("--report", metavar="HTML", help="save an HTML report of the trained model")
    parser.add_argument("--generate", metavar="CSV", help="write a synthetic dataset to a CSV file")
    parser.add_argument("--clean-data", action="store_true", help="with --generate: a dataset without data problems")
    args = parser.parse_args(argv)

    logging.basicConfig(filename="sentimentapp.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli = SentimentCLI(args.model_dir)
    try:
        if args.generate:
            rows = v.validate_int(args.rows, "Rows", 150, 50000)
            path = Path(args.generate)
            if path.exists():
                print(f"Error: {path} already exists; choose another name.")
                return 1
            path.parent.mkdir(parents=True, exist_ok=True)
            datagen.generate(rows, dirty=not args.clean_data).to_csv(path, index=False)
            print(f"Saved {rows} synthetic reviews to {path}.")
            return 0
        if args.train:
            if args.data:
                df, label = P.load_dataset(args.data), Path(args.data).name
            else:
                df, label = datagen.generate(v.validate_int(args.rows, "Rows", 500, 50000)), "synthetic (generated)"
            result = training.train(df, fast=args.fast, tune=not (args.fast or args.no_tune), source=label,
                                    progress=lambda m: print(f"  {m}"))
            persistence.save(result, args.model_dir)
            print("\n" + "\n".join(result.cleaning.lines()) + "\n")
            print(report.comparison_text(result.metadata) + "\n")
            print(report.model_text(result.metadata))
            print(f"\nModel saved to {args.model_dir}/")
            return 0
        if args.analyze is not None:
            cli.show_result(require_predictor(args.model_dir).analyze(args.analyze))
            return 0
        if args.report:
            print(f"Report saved to {report.save_html(require_predictor(args.model_dir).meta, args.report)}")
            return 0
        if args.serve:
            serve(require_predictor(args.model_dir), v.validate_port(args.port))
            return 0
        cli.run()
        return 0
    except SentimentError as exc:
        print(f"Error: {exc}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
