"""CPU-only smoke checks for the optional three-seed batch. All files are synthetic."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_difficulty_control as study


class WholeBatchTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        self.roots = {k: base / k for k in ("primary", "full", "cross", "phase1")}
        for root in self.roots.values():
            root.mkdir()
            (root / ".lock").touch()
        root, model = base / "difficulty", base / "qwen"
        root.mkdir()
        model.mkdir()
        (model / study.WEIGHTS).write_bytes(b"synthetic weights")
        (model / "config.json").write_bytes(b"synthetic config")
        self.now = study.as_time(study.CURRENT_CUTOFF) - 1800
        self.runner = SimpleNamespace(root=root, big=model, end=self.now + 1785)
        self.spec = {
            "admission": {
                "historical_max_seconds": dict(train=70, dev=10, test=20),
                "historical_max_bytes": dict(train=100, dev=10, test=20),
                "time_multiplier": 1.2,
                "disk_multiplier": 1.2,
                "hard_cutoff_utc": study.CURRENT_CUTOFF,
            },
            "phase1_cell_order": [f"p{i}" for i in range(11)],
            "cell_order": study.ORDER,
            "models_sha256": {
                "target_weights": study.digest(model / study.WEIGHTS),
                "target_config": study.digest(model / "config.json"),
            },
        }
        self.attestation = base / "private-proof.json"
        self.attestation.write_text('{"synthetic":"test fixture"}')
        self.account = base / "private-budget.json"
        self.account.write_text('{"synthetic":"test fixture"}')

    def execute_with(self, cash, good_pin=True):
        gate = SimpleNamespace(
            verify_attestation=mock.Mock(return_value={"sha": "fixture"}),
            account_seconds=mock.Mock(return_value=cash),
        )
        with mock.patch.dict(sys.modules, difficulty_admission=gate), mock.patch.object(
            study.time, "time", return_value=self.now
        ):
            return study.execute(
                self.roots,
                self.runner,
                self.spec,
                self.attestation,
                study.digest(self.attestation) if good_pin else "wrong",
                self.account,
            )

    def test_one_affordable_seed_defers_all_three(self):
        with mock.patch.object(study, "check_gpu", side_effect=AssertionError("queried the GPU")):
            status = self.execute_with(180)
        self.assertEqual(status["status"], "deferred")
        self.assertEqual(status["missing"], study.ORDER)
        self.assertFalse((self.runner.root / "admission.json").exists())
        self.assertFalse((self.runner.root / "training").exists())

    def test_bad_proof_does_not_reach_gpu(self):
        with mock.patch.object(
            study, "check_gpu", side_effect=AssertionError("queried GPU")
        ), self.assertRaisesRegex(ValueError, "attestation"):
            self.execute_with(1000, False)
        self.assertFalse((self.runner.root / "admission.json").exists())

    def test_funded_batch_uses_three_registered_seeds(self):
        fake_cell = mock.Mock(
            side_effect=lambda p, r, s, seed, method: {
                "endpoint": {"checkpoint": f"s{seed}-{method}-step123"}
            }
        )
        with mock.patch.object(study, "check_gpu", return_value={"gpu": "fixture"}), mock.patch.object(
            study, "protect_children"
        ), mock.patch.object(study, "run_cell", fake_cell):
            status = self.execute_with(1000)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(list(study.read(self.runner.root / "locked-models.json")), study.ORDER)
        self.assertEqual(
            [c.args[-2:] for c in fake_cell.call_args_list], [(42, "E2b"), (43, "E2b"), (44, "E2b")]
        )

    def test_partial_preexisting_outputs_cannot_reduce_admission(self):
        (self.runner.root / "receipts").mkdir()
        (self.runner.root / "receipts/train-s42-E2b.json").write_text("{}")
        with mock.patch.object(
            study, "check_gpu", side_effect=AssertionError("queried GPU")
        ), self.assertRaisesRegex(ValueError, "predate"):
            self.execute_with(1000)

    def test_wall_time_limit_rejects_whole_batch(self):
        self.spec["admission"]["hard_cutoff_utc"] = study.utc(self.now + 200)
        with mock.patch.object(study, "check_gpu", side_effect=AssertionError("queried GPU")):
            status = self.execute_with(1000)
        self.assertEqual(status["status"], "deferred")
        self.assertIn("clock", status["reason"])

    def test_forecast_updates_one_stage_without_using_accuracy(self):
        d = self.roots["phase1"] / "receipts"
        d.mkdir()
        (d / "train-s43-E2.json").write_text(json.dumps({"elapsed_s": 170}))
        f = study.estimate(self.spec, self.roots["phase1"], self.runner.root, self.now)
        self.assertEqual(f["observed_max_seconds"]["train"], 170)
        self.assertEqual(f["entire_fresh_batch_seconds"], 720)
        self.assertEqual(f["missing_stage_counts"], dict(train=3, dev=3, test=3))


if __name__ == "__main__":
    unittest.main()
