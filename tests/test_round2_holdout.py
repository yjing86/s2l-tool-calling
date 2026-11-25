"""Safety checks for the single-release runner, using only temporary artificial records."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import round2_holdout as h
from verify_tool_call_loss import DEMO


class ReleaseTests(unittest.TestCase):
    def test_earliest_calendar_instant_and_no_early_private_reads(self):
        for action in ("run", "summarize"):
            with self.subTest(action=action), patch.object(h, "utc_now", return_value=h.RELEASE - timedelta(microseconds=1)), \
                    patch.object(h, "read", side_effect=AssertionError("Opened a private file")):
                with self.assertRaisesRegex(ValueError, "sealed"):
                    h.released_run(SimpleNamespace(action=action), {})
        h.require_release(h.RELEASE)

    def test_public_anchor_must_precede_any_reserve_access(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = Path(td)
            frozen = {"panel": [], "registration_sha256": "a", "source_sha256": {}, "reserve_sha256": "b",
                      "not_before_utc": h.RELEASE.isoformat()}
            h.write_once(artifacts / h.OUT / "freeze.json", frozen)
            with patch.object(h, "utc_now", return_value=h.RELEASE), \
                    patch.object(h, "verify_anchor", side_effect=ValueError("unpublished")), \
                    patch.object(h, "inspect_released", side_effect=AssertionError("read reserve")):
                with self.assertRaisesRegex(ValueError, "unpublished"):
                    h.released_run(SimpleNamespace(artifacts=artifacts, anchor_commit="n/a", action="run"), {})

    def test_public_roster_preflight(self):
        cfg = h.registration()["holdout_v2"]
        self.assertEqual(len(cfg["panel_order"]), 47)
        self.assertEqual(len(cfg["existing_endpoints"]), 27)

    def test_artificial_membership_hashes_and_duplicate_groups(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            path = folder / "test.jsonl"
            rows = [{"id": 4, "group_id": "a"}, {"id": 7, "group_id": "b"}]
            def set_rows():
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                sha = h.fingerprint(path)
                metadata = {"selected_groups": 2, "test_sha256": sha}
                for key, name in (("id", "ordered_ids_sha256"), ("group_id", "ordered_groups_sha256")):
                    metadata[name] = hashlib.sha256("".join(f"{r[key]}\n" for r in rows).encode()).hexdigest()
                (folder / "manifest.json").write_text(json.dumps(metadata))
                return {"requests": 2, "reserve_sha256": sha}
            frozen = set_rows()
            _, groups = h.inspect_released(path, frozen)
            self.assertEqual(groups, {"4": "a", "7": "b"})
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "file differs"):
                h.inspect_released(path, frozen)
            rows[1]["group_id"] = "a"
            with self.assertRaisesRegex(ValueError, "membership"):
                h.inspect_released(path, set_rows())

    def test_failed_attempts_do_not_leak_into_complete_panel(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "attempts").mkdir()
            attempts = []
            def run(folder):
                attempts.append(folder)
                (folder / "completed").write_text("yes" if len(attempts) > 1 else "partial")
            def verify(folder):
                h.require((folder / "completed").read_text() == "yes", "incomplete")
                return "OK"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                h.ensure_task(out, "predictions/s42-A", run, verify)
            self.assertFalse((out / "predictions/s42-A").exists())
            self.assertEqual(h.ensure_task(out, "predictions/s42-A", run, verify), "OK")
            self.assertEqual(h.ensure_task(out, "predictions/s42-A", run, verify), "OK")
            self.assertEqual(len(attempts), 2)
            self.assertTrue(attempts[0].is_dir())

    def test_atomic_freeze_preserves_original_and_gpu_stages_honor_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "file"
            h.write_once(path, {"first": True})
            h.write_once(path, {"first": True})
            with self.assertRaisesRegex(ValueError, "differs"):
                h.write_once(path, {"first": False})
            self.assertTrue(h.read(path)["first"])
        args = h.task_args("fixture", "model", Path("out"), prompts=Path("cache"), adapter=Path("lora"))
        self.assertEqual(args[args.index("--max-model-len") + 1], "4096")
        self.assertEqual(args[args.index("--max-tokens") + 1], "512")
        self.assertIn("--enforce-eager", args)
        self.assertNotIn("--generate-only", args)

    def test_raw_output_is_rejudged_and_parse_failures_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            row = {**DEMO, "group_id": "g"}
            endpoint = {"id":"base", "adapter_sha256":{}}
            frozen = {"reserve_sha256":"r", "source_sha256":{"evaluate_experiment.py":"e"},
                      "target_files_sha256":{"tokenizer.json":"t","chat_template.jinja":"c"},
                      "evaluation_versions":{"vllm":"test"}}
            cfg = {"args":{"enforce_eager":True,"generate_only":False,"max_tokens":512,"max_model_len":4096,
                           "batch_size":32,"gpu_memory_utilization":0.85},
                   "source_sha256":"r","prompts_sha256":"p","adapter_sha256":{},"model_revision":h.BASE_REVISION,
                   "script_sha256":"e","tokenizer_sha256":"t","template_sha256":"c","greedy":True,
                   "seed":42,"dtype":"bfloat16","language_model_only":True,"stop_token_ids":[248046],
                   "max_lora_rank":16,"versions":{"vllm":"test"}}
            (folder/"config.json").write_text(json.dumps(cfg))
            pred = {"id":row["id"],"group_id":"g","raw_output":"<tool_call><function=oops>",
                    "correct":False,"reason":"parse"}
            target = folder / "predictions.jsonl"
            target.write_text(json.dumps(pred)+"\n")
            (folder/"metrics.json").write_text(json.dumps({"count":1,"correct":0,"reason_counts":{"parse":1}}))
            ref = {row["id"]:row}
            self.assertEqual(h.audit_predictions(folder,endpoint,frozen,"p",ref), {row["id"]:False})
            pred["correct"] = True
            target.write_text(json.dumps(pred)+"\n")
            with self.assertRaisesRegex(ValueError,"re-scoring"):
                h.audit_predictions(folder,endpoint,frozen,"p",ref)

    def test_all_47_synthetic_endpoints_require_historical_and_new_weights(self):
        with tempfile.TemporaryDirectory() as td:
            inputs = Path(td)
            reg = h.registration()
            records = reg["holdout_v2"]["existing_endpoints"]
            for r in records:
                tag = r["id"]
                if tag == "base":
                    continue
                if tag.startswith("full-"):
                    root = inputs / "full-reference/training"
                elif tag.endswith("-E2b"):
                    root = inputs / "followup/difficulty-control/training" / tag
                elif tag.split("-", 1)[1] in ("A", "B", "C") or tag == "s42-Bprime":
                    root = inputs / "course-run/training" / tag
                else:
                    root = inputs / "followup/phase1/training" / tag
                cp = root / ("checkpoint-" + r["checkpoint"].rsplit("-step", 1)[1])
                cp.mkdir(parents=True)
                for name, field in [(h.ADAPTER, "adapter_sha256"), (h.CONFIG, "adapter_config_sha256")]:
                    (cp / name).write_text(name + tag)
                    r[field] = h.fingerprint(cp / name)
            new = reg["cell_order"]
            for tag in new:
                cp = inputs / h.TRAIN / "training" / tag / "checkpoint-125"
                cp.mkdir(parents=True)
                for name in [h.CONFIG, h.ADAPTER]:
                    (cp / name).write_text(name + tag)
                h.write_once(inputs / h.TRAIN / "locks" / (tag + ".json"),
                             {"label": tag, "step":125, "checkpoint":tag+"-step125",
                              "target_tokens":3005228, "final_adapter_sha256":h.fingerprint(cp / h.ADAPTER)})
            h.write_once(inputs / h.TRAIN / "locked-models.json", {k:k+"-step125" for k in new})
            h.write_once(inputs / h.TRAIN / "execution.json",
                         {"status":"complete", "completed":new, "missing":[]})
            results = h.resolve_panel(inputs, reg)
            self.assertEqual([r["id"] for r in results], reg["holdout_v2"]["panel_order"])
            self.assertEqual(len(results), 47)
            (inputs / results[-1]["relative_adapter"] / h.ADAPTER).write_text("changed")
            with self.assertRaisesRegex(ValueError, "Wrong LoRA weights"):
                h.resolve_panel(inputs,reg)

    def test_complete_orphan_is_recovered_without_generating_again(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            orphan = out / "attempts/predictions-s42-A.one"
            orphan.mkdir(parents=True)
            (orphan / "completed").write_text("done")
            recovered = h.ensure_task(out, "predictions/s42-A",
                                      lambda _: self.fail("Regenerated a complete endpoint"),
                                      lambda folder: (folder/"completed").read_text())
            self.assertEqual(recovered, "done")
            self.assertTrue((out/"predictions/s42-A/completed").is_file())
