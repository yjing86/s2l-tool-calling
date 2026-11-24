"""Check the public CPU workflow against its shipped fixtures."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from s2l_tool_calling.cli import ROOT
from s2l_tool_calling.score import score_file


class QuickstartTests(unittest.TestCase):
    def test_example_verdicts_and_failed_rerun(self):
        source = ROOT / "examples/calls.jsonl"
        with TemporaryDirectory() as folder:
            target = Path(folder) / "verdicts.jsonl"
            counts = score_file(source, target)
            self.assertEqual(counts, dict.fromkeys(("correct", "function", "argument", "count", "parse"), 1))
            expected = target.read_bytes()
            verdicts = [json.loads(line) for line in expected.splitlines()]
            self.assertEqual(len({item["id"] for item in verdicts}), 5)
            damaged = Path(folder) / "bad.jsonl"
            damaged.write_bytes(source.read_bytes() + source.read_bytes().splitlines()[0])
            with self.assertRaisesRegex(ValueError, r"bad\.jsonl:6: duplicate id"):
                score_file(damaged, target)
            self.assertEqual(target.read_bytes(), expected)

    def test_reported_primary_effect_can_be_derived_from_correct_counts(self):
        report = json.loads((ROOT / "results/turnstile-single-call.json").read_text())
        main = report["phases"]["main"]
        locked, models = main["locked"], report["models"]
        diffs = []
        for seed in main["paired_by_seed"]:
            b, c = [models[locked[f"s{seed}-{arm}"]] for arm in ("B", "C")]
            self.assertEqual(b["requests"], c["requests"])
            diffs.append(100 * (c["correct"] - b["correct"]) / c["requests"])
        self.assertAlmostEqual(sum(diffs) / len(diffs), main["across_seed_differences_pp"]["C-B"]["mean"])
