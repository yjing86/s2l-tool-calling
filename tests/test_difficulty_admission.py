"""CPU-only fixtures for E2b funding, external backups and raw-output replay."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import difficulty_admission as gate


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n")


class DifficultyAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.obs = self.root / "funding.json"
        self.now = gate.timestamp("2026-11-24T18:00:00Z")
        self.billing = {"utc": "2026-11-24T17:58:00Z", "balance_usd": "2.00",
                        "account_spend_usd_per_hour": "2.00", "displayed_project_gpu_usd_per_hour": "1.00",
                        "source": "account billing page", "hard_cutoff_utc": gate.DEFAULT_CUTOFF}

    def test_cash_accounts_for_age_and_teardown_and_uses_higher_rate(self):
        put(self.obs, self.billing)
        self.assertEqual(gate.account_seconds(self.obs, self.now), 2880)
        put(self.obs, dict(self.billing, displayed_project_gpu_usd_per_hour="4.00"))
        self.assertEqual(gate.account_seconds(self.obs, self.now), 1080)
        put(self.obs, dict(self.billing, balance_usd="0.001"))
        self.assertEqual(gate.account_seconds(self.obs, self.now), 0)

    def test_stale_future_mismatched_or_invalid_billing_observations(self):
        for change in ({"utc": "2026-11-24T17:49:59Z"}, {"utc":"2026-11-24T18:00:01Z"},
                       {"hard_cutoff_utc": "2026-11-25T20:29:00Z"}, {"source": ""},
                       {"account_spend_usd_per_hour": "NaN"}, {"displayed_project_gpu_usd_per_hour": "0"}):
            with self.subTest(change=change):
                put(self.obs, dict(self.billing, **change))
                with self.assertRaises(ValueError):
                    gate.account_seconds(self.obs, self.now)
        put(self.obs, dict(self.billing, hard_cutoff_utc="2026-11-25T20:29:00Z"))
        self.assertEqual(gate.account_seconds(self.obs, self.now, "2026-11-25T20:29:00Z"), 2880)

    def test_mirror_requires_terminal_files_and_rehashes_even_optional_bytes(self):
        root = self.root / "backup"
        put(root / "terminal.json", {"done": 1})
        data = root / "adapter"; data.write_bytes(b"aaa")
        receipt = self.root / "mirror.json"
        files = {p.name: {"bytes": p.stat().st_size, "sha256": gate.digest(p)} for p in root.iterdir()}
        meta = {"course": "full-reference", "status": "complete", "full_tree": True,
                "files": files, "inventory_count": len(files), "symbolic_links": {}}
        put(receipt, meta)
        required = {"terminal.json": gate.digest(root / "terminal.json")}
        summary = gate.validate_receipt(receipt, root, "full-reference", required)
        self.assertEqual(gate.live_inventory(root), (summary["file_sizes_sha256"], summary["symbolic_links_sha256"]))
        with self.assertRaisesRegex(ValueError, "predates"):
            gate.validate_receipt(receipt, root, "full-reference", {"missing.json": "unused"})
        put(receipt, dict(meta, full_tree=False))
        with self.assertRaisesRegex(ValueError, "mirror mode"):
            gate.validate_receipt(receipt, root, "full-reference", required)
        put(receipt, meta)
        data.write_bytes(b"bbb")  # Equal length, different checkpoint bytes.
        with self.assertRaisesRegex(ValueError, "checksum"):
            gate.validate_receipt(receipt, root, "full-reference", required)

    def test_live_verification_rejects_added_files_and_wrong_replay_counts(self):
        roots = {k:self.root/k for k in gate.COURSES}
        for root in roots.values():
            put(root / "terminal", {})
        witness = {k: {"terminal": gate.digest(root/"terminal")} for k,root in roots.items()}
        jobs = [("phase1", "test", "model")]
        hashes = {k:gate.object_sha(files) for k,files in witness.items()}
        metas = {}
        for k, root in roots.items():
            shape, links = gate.live_inventory(root)
            metas[k] = {"files": 1, "verified_bytes":3, "receipt_sha256": "a"*64, "inventory_sha256":"b"*64,
                        "file_sizes_sha256":shape, "symbolic_links_sha256":links}
        proof = {"version": 1, "verifier_sha256":gate.digest(Path(gate.__file__)),
                 "terminal_tree_sha256":hashes, "replayed_requests":gate.counts_for(jobs),
                 "mirror_receipt_sha256":{k:"a"*64 for k in roots}, "primary_official_bfcl_audits_sha256":"c"*64, "mirrors":metas}
        out = self.root/"proof.json"; put(out,proof)
        with patch.object(gate, "collect", return_value=(witness,jobs)):
            self.assertEqual(gate.verify_attestation(out, roots),proof)
            put(roots["cross"]/"extra", {"x":1})
            with self.assertRaisesRegex(ValueError, "live tree differs"):
                gate.verify_attestation(out,roots)
            put(out,dict(proof, replayed_requests={}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                gate.verify_attestation(out,roots)

    def test_terminal_receipts_and_eleven_distinct_phase1_cells_are_required(self):
        roots = {k:self.root/k for k in gate.COURSES}
        evaluator = gate.digest(Path(gate.__file__).with_name("evaluate_experiment.py"))

        def task(study, tag, files):
            outputs = {}
            for name in files:
                p = roots[study] / name
                if not p.exists():
                    put(p, {})
                outputs["/original-run/" + name] = gate.digest(p)
            stamp = roots[study] / "receipts" / (tag + ".json")
            put(stamp, {"task":tag, "outputs":outputs})
            return gate.digest(stamp)

        def eval_job(study, split, checkpoint, tag):
            prefix = f"evaluation/{split}/{checkpoint}/"
            put(roots[study] / (prefix+"config.json"), {"script_sha256":evaluator})
            return task(study, tag, [prefix+f for f in ("config.json", "metrics.json", "predictions.jsonl")])

        task("primary", "summary", ["summary/internal.json", "summary/aggregate-audit.zip"])
        put(roots["primary"] / "frozen/test.jsonl", {})
        main = {t:"base" if t=="base" else t+"-step1" for t in sorted(gate.PRIMARY)}
        put(roots["primary"] / "locked-models.json", {"main":main})
        for model in main.values():
            eval_job("primary","test", model, "eval-test-"+model)
            bf = f"evaluation/bfcl/{model}/"
            task("primary", "score-bfcl-"+model,[bf+"bfcl-scores.jsonl", bf+"bfcl-summary.json"])
            put(roots["primary"] / (bf+"predictions.jsonl"), {})
        task("full", "full-summary", ["summary/full-reference.json"])
        put(roots["full"] / "full-models.json", dict(final="full-20", matched_tokens="full-10", best_dev="full-10"))
        for model in ("full-10", "full-20"):
            eval_job("full","test", model, "eval-test-"+model)
        task("cross","summary", ["summary.json","summary.md"])
        cross = {t:t+"-step1" for t in sorted(gate.CROSS)}
        put(roots["cross"] / "crossed-models.json",cross)
        for t, model in cross.items():
            eval_job("cross","dev",model,"dev-"+t)
        order = sorted(gate.PHASE1)
        phase = roots["phase1"]
        put(phase / "preregistration.json", {"spec":{"cell_order":order}})
        execution = {"status":"complete","missing":[], "planned":order,"completed":order}
        put(phase/"execution.json",execution)
        put(phase/"locked-models.json",{t:t+"-step1" for t in order})
        for t in order:
            train = f"training/{t}/"
            ck = phase / (train+"checkpoint-1/adapter_model.safetensors")
            put(ck,{"model":t})
            model = t+"-step1"
            ep = {"label":t,"checkpoint":model,"step":1,"final_adapter_sha256":gate.digest(ck)}
            put(phase / f"locks/{t}.json",ep)
            rec = {"train":task("phase1","train-"+t,[train+n for n in ("completed.json","args.json","exit.json")])}
            for split in gate.SPLIT_SIZES:
                rec[split] = eval_job("phase1",split,model,split+"-"+t)
            put(phase/f"cells/{t}.json",{"endpoint":ep,"receipt_sha256":rec})
        _, jobs = gate.collect(roots)
        self.assertEqual(gate.counts_for(jobs), {"primary":{"test":22000}, "full":{"test":4000},
                                                 "cross":{"dev":1600}, "phase1":{"dev":4400,"test":22000}})
        put(phase/"execution.json",dict(execution, completed=order[:-1], missing=order[-1:], status="partial_budget"))
        with self.assertRaisesRegex(ValueError,"All eleven"):
            gate.collect(roots)
        put(phase/"execution.json",execution)
        (phase / f"evaluation/test/{order[-1]}-step1/predictions.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError,"artifact differs"):
            gate.collect(roots)

    def test_raw_prediction_tamper_is_detected_despite_unchanged_saved_boolean(self):
        from verify_tool_call_loss import DEMO
        row = dict(DEMO, id="one", group_id="group")
        call = row["messages"][-1]["tool_calls"][0]["function"]
        key, value = next(iter(call["arguments"].items()))
        text = f"<tool_call><function={call['name']}><parameter={key}>{value}</parameter></function></tool_call>"
        verdict = gate.judge(row,text)
        self.assertTrue(verdict["correct"])
        folder = self.root/"eval"
        put(folder/"config.json", {"source_sha256":"gold", "args":{"generate_only":False}})
        put(folder/"metrics.json", {"count":1,"correct":1,"accuracy":1.,"reason_counts":{"correct":1}})
        pred = {"id":"one","group_id":"group","raw_output":text,**verdict}
        put(folder/"predictions.jsonl",pred)
        self.assertEqual(gate.replay(folder, {"one":row},"gold"),1)
        put(folder/"predictions.jsonl",dict(pred,raw_output="invalid output"))
        with self.assertRaisesRegex(ValueError,"does not reproduce"):
            gate.replay(folder,{"one":row},"gold")
