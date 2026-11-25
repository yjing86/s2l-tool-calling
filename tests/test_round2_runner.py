"""The Round 2 queue must resume without replaying completed fits or selecting by score."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_round2 as study


class Round2Queue(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = SimpleNamespace(root=self.root, end=10**10, ledger=Mock())
        study.write(self.root / "protocol-lock.json", {"spec": {"science": "fixture"}})
        self.order = list(study.CELLS)
        self.spec = {"cell_order": self.order, "admission": {}}
        self.done = {}
        self.fits = []
        self.launch = {
            "training": {"command": []},
            "splits": {"dev": {"generation": {"versions": {}}}},
            "environment_observation": {},
        }

    def run_queue(self, max_cells=None, allow=True):
        with ExitStack() as patches:
            for name, kwargs in (
                ("completed_cells", {"side_effect": lambda *unused: self.done.copy()}),
                ("check_gpu", {"return_value": {"gpu": "simulated"}}),
                ("host_protocol", {"return_value": self.launch}),
                ("protect_workers", {}),
                ("run_cell", {"side_effect": self.fit}),
                ("budget", {"return_value": {"fits_time": allow, "free_bytes": 10**6,
                                             "required_seconds": 99, "required_bytes": 100}}),
            ):
                patches.enter_context(patch.object(study, name, **kwargs))
            return study.execute(self.runner, self.root / "primary", self.spec, max_cells)

    def fit(self, primary, runner, spec, seed, method):
        tag = study.label(seed, method)
        self.assertNotIn(tag, self.done)
        key = self.done[tag] = tag + "-step125"
        self.fits.append(tag)
        return {"endpoint": {"checkpoint": key}}

    def test_resume_reaches_all_twenty_without_duplicate_fits(self):
        partial = self.run_queue(max_cells=4)
        self.assertEqual(partial["completed"], self.order[:4])
        self.assertEqual(partial["status"], "partial_batch")
        self.assertFalse((self.root / "locked-models.json").exists())
        finished = self.run_queue()
        self.assertEqual(finished["status"], "complete")
        self.assertEqual(self.fits, self.order)
        self.assertEqual(study.read(self.root / "locked-models.json"), self.done)
        self.run_queue()
        self.assertEqual(self.fits, self.order)

    def test_budget_stop_leaves_resume_possible_and_no_locked_models(self):
        self.assertEqual(self.run_queue(allow=False)["status"], "partial_time")
        self.assertEqual(self.fits, [])
        self.assertFalse((self.root / "locked-models.json").exists())
        self.assertEqual(self.run_queue()["status"], "complete")

    def test_registration_rejects_drift_and_preexisting_output(self):
        with patch.object(study, "build_spec", return_value=self.spec):
            with self.assertRaisesRegex(ValueError, "changed"):
                study.freeze(self.runner, None, None, None, create=False)
            (self.root / "protocol-lock.json").unlink()
            (self.root / "evaluation").mkdir()
            with self.assertRaisesRegex(ValueError, "predates"):
                study.freeze(self.runner, None, None, None, create=True)

    def test_unreceipted_live_worker_is_never_restarted(self):
        partial = self.root / "training/s45-A"
        path = self.root / "attempts/train-s45-A.json"
        study.write(path, {"state": "started", "argv": ["runner", "--output", str(partial)]})
        with patch.object(study.subprocess, "check_output", return_value="python --output " + str(partial)):
            with self.assertRaisesRegex(RuntimeError, "still be active"):
                study.recover_orphans(self.runner)
        self.assertEqual(study.read(path)["state"], "started")
        with patch.object(study.subprocess, "check_output", return_value="idle process table"):
            study.recover_orphans(self.runner)
        self.assertEqual(study.read(path)["state"], "failed")
        self.runner.ledger.assert_called_once()

    def test_successful_task_is_untouched_even_with_started_intent(self):
        path = self.root / "attempts/dev-s45-B.json"
        study.write(path, {"state": "started"})
        study.write(self.root / "receipts" / path.name, {"done": True})
        with patch.object(study.subprocess, "check_output", side_effect=AssertionError):
            study.recover_orphans(self.runner)
        self.assertEqual(study.read(path)["state"], "started")

    def test_active_runtime_disambiguates_duplicate_system_torch_metadata(self):
        runner = SimpleNamespace(a=SimpleNamespace(training_python=Path("train"), eval_python=Path("eval")),
                                 big=Path("model"), scripts=Path("scripts"))
        eval_versions = {"python": "3.12.3 linux", "platform": "old-host", "torch": "2.10.0"}
        ref = {"training": {"command": ["--model", "old", "--external_plugins", "old.py"],
                            "packages": {"torch": "2.8.0+cu129"}, "runtime": {"cuda": "12.8"}},
               "splits": {split: {"generation": {"versions": eval_versions.copy()}} for split in study.SPLITS}}
        train = {"python": "3.12.3", "platform": "new-host", "torch_import": "2.8.0+cu128",
                 "active": {"torch": "2.8.0"}, "reported": {"torch": "2.8.0"}, "cuda": "12.8"}
        evaluation = {**train, "torch_import": "2.10.0+cu128", "active": {"torch": "2.10.0"}}
        with patch.object(study.subprocess, "check_output", side_effect=[json.dumps(train), json.dumps(evaluation)]):
            result = study.host_protocol(runner, ref)
        self.assertEqual(result["training"]["packages"]["torch"], "2.8.0")
        self.assertEqual(ref["training"]["packages"]["torch"], "2.8.0+cu129")
        self.assertEqual(result["environment_observation"]["training"]["torch_import"], "2.8.0+cu128")
        train["cuda"] = "12.9"
        with patch.object(study.subprocess, "check_output", return_value=json.dumps(train)):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                study.host_protocol(runner, ref)


if __name__ == "__main__":
    unittest.main()
