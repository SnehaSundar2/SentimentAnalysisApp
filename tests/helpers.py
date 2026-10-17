"""Shared fixtures. Training takes a few seconds, so ONE small model is trained for the whole test run."""

import atexit
import shutil
import tempfile
import unittest
from pathlib import Path

from sentimentapp import datagen, persistence, schema as S, training

_SHARED = {}

POSITIVE = S.EXAMPLES["positive"][1]
NEGATIVE = S.EXAMPLES["negative"][1]
NEUTRAL = S.EXAMPLES["neutral"][1]
MIXED = S.EXAMPLES["mixed"][1]
NEGATION = S.EXAMPLES["negation"][1]
SARCASM = S.EXAMPLES["sarcasm"][1]


def shared():
    """(TrainingResult, saved model directory, raw training DataFrame) — trained once, in fast mode."""
    if not _SHARED:
        folder = Path(tempfile.mkdtemp(prefix="sentimentapp_tests_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        raw = datagen.generate(2000)
        result = training.train(raw, fast=True, tune=False, cv=3, source="synthetic test data")
        persistence.save(result, folder / "model")
        _SHARED.update(result=result, model_dir=folder / "model", raw=raw, folder=folder)
    return _SHARED["result"], _SHARED["model_dir"], _SHARED["raw"]


class TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def copy_model(self):
        """A private copy of the trained model that a test may damage (each call makes a fresh copy)."""
        _, model_dir, _ = shared()
        self._copies = getattr(self, "_copies", 0) + 1
        target = self.dir / f"model_copy_{self._copies}"
        shutil.copytree(model_dir, target)
        return target
