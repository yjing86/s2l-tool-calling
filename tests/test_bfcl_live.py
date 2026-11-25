"""Exercise the external panel's outcome-blind eligibility and scoring gates."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bfcl_live as live
import run_round2_bfcl as runner


def row(key, system=None, user="Find shipment 17"):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return {"id": key, "question": [messages]}


FN = {"name": "find", "parameters": {"type": "object", "properties": {}}}
TOOLS = [{"type": "function", "function": FN}]


class LivePanelTests(unittest.TestCase):
    def test_native_messages_and_two_overlap_keys(self):
        item, reason = live.candidate("live_simple", row("s0", "Use metric units"), TOOLS, set(), set())
        self.assertIsNone(reason)
        self.assertEqual(item["messages"], row("s0", "Use metric units")["question"][0])
        _, reason = live.candidate("live_simple", row("s1"), TOOLS, {live.text_key("Find shipment 42")}, set())
        self.assertEqual(reason, "training_prompt_overlap")
        _, reason = live.candidate("live_simple", row("s2"), TOOLS, set(), {live.signature(FN)})
        self.assertEqual(reason, "training_signature_overlap")
        _, reason = live.candidate("live_simple", row("s3"), TOOLS * 2, set(), set())
        self.assertEqual(reason, "colliding_names")

    def test_context_before_dedup_and_cap_independent_of_source_order(self):
        items = []
        for key, text, length in [("long", "same", 7681), ("ok", "same", 7680),
                                  ("u", "other", 100), ("v", "other", 100), ("w", "last", 80)]:
            item, _ = live.candidate("live_simple", row(key, user=text), TOOLS, set(), set())
            items.append({**item, "n": length})
        with patch.object(live, "MAX_PER_CATEGORY", 3):
            picked, audit = live.choose(items, lambda x: x["n"])
            again, _ = live.choose(list(reversed(items)), lambda x: x["n"])
        self.assertEqual(picked, again)
        self.assertEqual(len(picked), 3)
        self.assertIn("ok", [x["id"] for x, _ in picked])
        self.assertEqual(audit["live_simple"]["excluded_after_overlap_filter"], {"context": 1, "duplicate": 1})

    def test_scoring_rejects_partial_predictions_before_loading_benchmark_answers(self):
        with TemporaryDirectory() as folder:
            parent = Path(folder)
            panel, gen = parent / "panel", parent / "generation"
            panel.mkdir()
            gen.mkdir()
            live.write(panel / "prompts.jsonl", live.lines([{"id": "s0", "messages": [], "tools": TOOLS}]))
            sha = live.digest(panel / "prompts.jsonl")
            live.save_json(panel / "manifest.json", {
                "bfcl_revision": live.REVISION, "source_sha256": live.SOURCES, "prompts_sha256": sha,
                "items": [{"id": "s0", "category": "live_simple", "input_tokens": 10}],
            })
            live.save_json(gen / "config.json", {
                "args": {"generate_only": True, "max_tokens": 512, "max_model_len": 8192},
                "source_sha256": sha, "model_revision": live.BASE_REVISION, "tokenizer_sha256": live.TOKENIZER_SHA,
                "template_sha256": live.TEMPLATE_SHA, "greedy": True, "seed": 42,
            })
            (gen / "predictions.jsonl").write_text("")
            args = SimpleNamespace(panel=panel, panel_sha256=live.digest(panel / "manifest.json"), generation=gen)
            with self.assertRaisesRegex(ValueError, "exactly one prediction"):
                live.score(args, parent / "no-benchmark-present")

    def test_external_summary_requires_complete_matched_model_panel(self):
        table = {tag: {cat: {"n": 10, "correct": 4 + int(tag.endswith("-C")),
                             "accuracy": (4 + int(tag.endswith("-C"))) / 10}
                       for cat in live.CATEGORIES} for tag in runner.MODEL_IDS}
        summary = runner.aggregate(table)
        self.assertAlmostEqual(summary["descriptive_differences_pp"]["live_simple"]["C-B"], 10)
        with self.assertRaisesRegex(ValueError, "complete"):
            runner.aggregate(dict(list(table.items())[:-1]))
        table["s46-C"]["live_simple"]["n"] = 9
        with self.assertRaisesRegex(ValueError, "nonempty category"):
            runner.aggregate(table)

    def test_resume_rejects_switched_lora_and_changed_predictions(self):
        with TemporaryDirectory() as folder:
            parent = Path(folder)
            cat = "live_simple"
            ep = {"id": "s42-A", "adapter_sha256": {"adapter_model.safetensors": "adapter"}}
            lock = {"manifest_sha256": "manifest", "cache_sha256": "cache", "prompts_sha256": "source"}
            items = {"s0": {"category": cat, "input_tokens": 7}}
            cfg = {"prompts_sha256": "cache", "source_sha256": "source", "adapter_sha256": ep["adapter_sha256"],
                   "model_revision": live.BASE_REVISION, "tokenizer_sha256": live.TOKENIZER_SHA,
                   "template_sha256": live.TEMPLATE_SHA, "script_sha256": runner.EVAL_SHA, "seed": 42, "greedy": True,
                   "dtype": "bfloat16", "language_model_only": True, "stop_token_ids": [248046], "max_lora_rank": 16,
                   "args": {"generate_only": True, "enforce_eager": True, "max_tokens": 512, "max_model_len": 8192,
                            "batch_size": 32, "gpu_memory_utilization": 0.85},
                   "versions": {"vllm": "0.19.1", "torch": "2.10.0+cu128", "transformers": "5.24.0"}}
            live.save_json(parent / "config.json", cfg)
            live.save_json(parent / "metrics.json", {"count": 1})
            pred = {"id": "s0", "input_tokens": 7, "raw_output": "", "reason": "generated"}
            live.write(parent / "predictions.jsonl", live.lines([pred]))
            original = runner.check_generation(parent, ep, lock, items)
            self.assertEqual(original, runner.check_generation(parent, ep, lock, items))
            with self.assertRaisesRegex(ValueError, "settings changed"):
                runner.check_generation(parent, {**ep, "adapter_sha256": {}}, lock, items)
            (parent / "predictions.jsonl").write_text(live.lines([{**pred, "raw_output": "altered"}]))
            with self.assertRaisesRegex(ValueError, "receipt or freeze differs"):
                runner.check_generation(parent, ep, lock, items)

    def test_serial_runner_promotes_one_attempt_and_revalidates_on_resume(self):
        with TemporaryDirectory() as folder:
            parent = Path(folder)
            root = parent / runner.DATA
            (root / "panel").mkdir(parents=True)
            live.save_json(root / "panel/manifest.json", {"items": [{"id": "x", "input_tokens": 2,
                                                                      "category": "live_simple"}]})
            calls = []
            lock = {"manifest_sha256": "manifest", "panel": [
                {"id": "base", "relative_adapter": None}, {"id": "s42-A", "relative_adapter": "weights/A"},
            ]}
            args = SimpleNamespace(artifacts=parent, action="freeze", eval_python=Path("/fake/python"),
                                   hours=0.002, limit=1)

            def fake_child(python, cmd, tmp, lock_fd, deadline):
                calls.append(list(map(str, cmd)))
                (tmp / "done").touch()

            def validate(tmp, ep, lock, items):
                if not (tmp / "done").is_file():
                    raise ValueError("Missing completed synthetic generation")
                self.assertEqual(set(items), {"x"})
                return {"endpoint": ep["id"]}

            with patch.object(runner, "stage_lock", return_value=lock), patch.object(runner, "child_run", fake_child), \
                    patch.object(runner, "check_generation", validate):
                runner.queue(args)
                args.action = "generate"
                runner.queue(args)
                self.assertTrue((root / "generation/base/done").is_file())
                self.assertFalse((root / "generation/s42-A").exists())
                runner.queue(args)
                runner.queue(args)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("--adapter", calls[0])
            self.assertEqual(Path(calls[1][calls[1].index("--adapter") + 1]), parent.resolve() / "weights/A")


if __name__ == "__main__":
    unittest.main()
