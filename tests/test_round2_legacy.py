import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarize_round2_legacy as reporting


class LegacyPanelTests(unittest.TestCase):
    def setUp(self):
        self.ids = {"q1", "q2", "q3", "q4"}
        self.outcomes = {
            f"s{seed}-{method}": {key: key == "q2" for key in self.ids}
            for method in reporting.METHODS for seed in reporting.SEEDS
        }

    def test_pairs_identical_items_and_weights_each_seed_once(self):
        self.outcomes["s42-C"] = {key: key in ("q1", "q3") for key in self.ids}
        result = reporting.split_report(self.outcomes, self.ids)
        contrast = result["paired_contrasts"]["C minus B"]
        self.assertEqual(contrast["by_seed"]["42"], {
            "only_left_correct": 2, "only_right_correct": 1, "difference_pp": 25.0})
        self.assertEqual(contrast["mean_difference_pp"], 5.0)
        self.assertEqual(result["methods"]["C"]["mean_accuracy_pct"], 30.0)
        self.assertEqual(result["paired_contrasts"]["L-long minus L-short"]["mean_difference_pp"], 0)
        self.assertEqual(len(result["methods"]), 7)
        self.assertEqual(len(result["paired_contrasts"]), 4)

    def test_incomplete_endpoints_and_nonidentical_items_are_refused(self):
        scores = self.outcomes.pop("s46-E2b")
        with self.assertRaisesRegex(ValueError, "35-endpoint"):
            reporting.split_report(self.outcomes, self.ids)
        self.outcomes["s46-E2b"] = {**scores, "extra": False}
        with self.assertRaisesRegex(ValueError, "same requests"):
            reporting.split_report(self.outcomes, self.ids)

    def test_archive_verifier_only_runs_after_20_complete_fits_and_a_pinned_proof(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(reporting, "verify") as verify:
            root = Path(temp)
            proof = root / "proof.json"
            original = {"terminal_state": "complete", "completed_cells": 19, "completed_tasks": 60,
                        "regraded_predictions": {"dev": 8000, "test": 40000},
                        "source_inventory": "source.json", "source_inventory_sha256": "inventory-digest"}
            reporting.write(proof, original)
            with self.assertRaisesRegex(ValueError, "all 20"):
                reporting.verify_backup(root, root / "dev.jsonl", proof, reporting.digest(proof), None)
            original["completed_cells"] = 20
            reporting.write(proof, original)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                reporting.verify_backup(root, root / "dev.jsonl", proof, "other", None)
            verify.assert_not_called()
            verify.return_value = {"source_inventory_sha256": "different"}
            with self.assertRaisesRegex(ValueError, "verification differs"):
                reporting.verify_backup(root, root / "dev.jsonl", proof, reporting.digest(proof), None)
            verify.assert_called_once_with(root / "followup/round2", root / "source.json", "inventory-digest",
                                           reporting.PROTOCOL_SHA, reporting.REGISTRATION, root / "course-run",
                                           root / "dev.jsonl")


if __name__ == "__main__":
    unittest.main()
