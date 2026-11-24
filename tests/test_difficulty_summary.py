"""Synthetic CPU-only regression tests for the difficulty-control evidence checks."""
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import summarize_difficulty_control as study
from evaluate_experiment import judge
from run_crossed_seeds import CONFIG_KEYS, GENERATION_ARGS, gen_protocol, token_boundary
from run_followup import ORDER, train_protocol


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def stamp(root, task, paths, code):
    put(root / "receipts" / (task + ".json"), {"task": task, "elapsed_s": 1, "script_sha256": code,
        "outputs": {str(path): study.digest(path) for path in paths}})


def row(i):
    return {"id": f"test-{i}", "group_id": i % 7, "tools": [{"function": {"name": "f", "parameters":
            {"type": "object", "properties": {}}}}], "messages": [
            {"role": "user", "content": "Call f"}, {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}]}


class StudyFixture:
    """Match the on-disk protocol, with small dummy adapters and invented calls."""
    def __init__(self, tmp):
        self.primary, self.phase, self.diff, self.eligible = [tmp / name for name in ("primary", "phase", "diff", "eligible")]
        self.roots = self.primary, self.phase, self.diff, self.eligible
        self.pool_rows = [{"id": f"train-{i}"} for i in range(4000)]
        self.gold = [row(i) for i in range(2000)]
        self.pool = self.primary / "frozen/train.jsonl"
        test = self.primary / "frozen/test.jsonl"
        jsonl(self.pool, self.pool_rows)
        jsonl(test, self.gold)
        self.config = {k: "synthetic" for k in CONFIG_KEYS}
        self.config.update({"model_revision": study.BASE_REVISION,
                            "source_sha256": study.digest(test), "script_sha256": study.digest(SCRIPTS / "evaluate_experiment.py"),
                            "args": {**dict.fromkeys(GENERATION_ARGS, 1), "generate_only": False},
                            "versions": {"vllm": "fixture"}, "adapter_sha256": {}})
        put(self.primary / "evaluation/test/base/config.json", self.config)
        self.selections = {}
        self.ids = [r["id"] for r in self.pool_rows[:1000]]
        self.masters = {"phase": {}, "eligible": {}}
        self.model_paths = {}
        self.locks = {r: {} for r in self.roots[:3]}
        self.source = {f: study.digest(SCRIPTS / f) for f in study.CORE}
        self.goal = 3005228
        for seed in study.SEEDS:
            self.selections_for(seed)
            for arm in ("A", "B", "E2", "E2b", "E3", "E4", "Bprime"):
                if arm == "Bprime" and seed == 42:
                    continue
                root = self.primary if arm in ("A", "B") else self.diff if arm == "E2b" else self.phase
                self.endpoint(root, seed, arm)
        put(self.primary / "model-paths.json", self.model_paths)
        put(self.primary / "locked-models.json", {"main": self.locks[self.primary]})
        put(self.primary / "design.json", {"pool": 4000, "subset": 1000, "test": 2000})
        master = self.phase / "selection/manifest.json"
        put(master, {"selection_manifest_sha256": self.masters["phase"]})
        eligible = self.eligible / "manifest.json"
        put(eligible, {"seeds": list(study.SEEDS), "common_target_input_token_budget": self.goal,
                       "sources": {"train_sha256": study.digest(self.pool)}, "seed_manifest_sha256": self.masters["eligible"]})
        common = {"pool_sha256": study.digest(self.pool), "primary_lock_sha256": study.digest(self.primary / "locked-models.json"),
                  "training": self.protocol, "splits": {"test": {"generation": gen_protocol(self.config)}},
                  "source_sha256": self.source}
        order = [f"s{s}-{m}" for s,m in ORDER]
        phase_spec = dict(common, cell_order=order, selections=self.selections,
                          selection_manifest_sha256=study.digest(master))
        put(self.phase / "preregistration.json", {"spec": phase_spec})
        diff_spec = dict(common, cell_order=study.LABELS, selections=self.selections, analysis={"contrasts": list(study.CONTRASTS)},
                         phase1_preregistration_sha256=study.digest(self.phase / "preregistration.json"),
                         phase1_spec_sha256=study.science_hash(phase_spec), primary_design_sha256=study.digest(self.primary / "design.json"),
                         sources={"train_sha256": study.digest(self.pool)}, eligibility_manifest_sha256=study.digest(eligible))
        put(self.diff / "preregistration.json", {"spec": diff_spec})
        put(self.diff / "dependencies.json", {"attestation_sha256": "fixture",
            "evidence": {"verifier_sha256": study.digest(SCRIPTS / "difficulty_admission.py")}})
        put(self.diff / "admission.json", {"attestation_sha256": "fixture", "decision": {
            "attestation_sha256": "fixture", "fits": {"clock": True, "cash": True, "storage": True}}})
        for root, labels in ((self.phase, order), (self.diff, study.LABELS)):
            put(root / "locked-models.json", self.locks[root])
            put(root / "execution.json", {"status": "complete", "completed": labels, "planned": labels, "missing": []})

    def selections_for(self, seed):
        for root, arms, key in ((self.primary, ("A","B"), None), (self.phase, ("Bprime","E2","E3","E4"),"phase"),
                                (self.eligible, ("E2b",),"eligible")):
            folder = (root if root == self.eligible else root / "selection") / str(seed)
            data = self.pool_rows[:1000]
            paths = []
            entries = {}
            for arm in arms:
                p = folder / (arm + ".jsonl")
                jsonl(p,data)
                paths.append(p)
                entries[arm] = {"ids": self.ids, "sha256": study.digest(p)}
            manifest = folder / "selection.json"
            put(manifest,entries["E2b"] if root == self.eligible else {"subsets": entries})
            if key is None:
                stamp(root, f"select-{seed}", [*paths, manifest], "fixture")
            else:
                mh = study.digest(manifest)
                self.masters[key][str(seed)] = mh
                for arm, rec in entries.items():
                    self.selections[f"s{seed}-{arm}"] = {"sha256":rec["sha256"], "size":1000, "seed_manifest_sha256":mh}
            if root == self.eligible:
                jsonl(self.diff / "selection" / str(seed) / "E2b.jsonl",data)

    def endpoint(self,root,seed,arm):
        tag = f"s{seed}-{arm}"
        key = tag + "-step1"
        train = root / "training" / tag
        adapter = train / "checkpoint-1/adapter_model.safetensors"
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(tag.encode())
        ah = study.digest(adapter)
        done = dict(final_step=1,input_tokens=self.goal,answer_tokens=100,adapter_sha256={"1":ah})
        rt = {"token_goal":self.goal,"starting_model_step":0,"gpu":"fixture","cuda":"fixture","trainable_parameter_count":10}
        cmd = ["swift","--seed",str(seed),"--data_seed",str(seed),"--output_dir","fixture","--dataset","fixture"]
        run = {"seed":seed,"stage":"target","pool_sha256":study.digest(self.pool),"ids":self.ids,"command":cmd,
               "data_sha256": self.selections.get(tag,{}).get("sha256",study.digest(self.primary/"selection"/str(seed)/"A.jsonl")),
               "packages":{},"sources_sha256":{},"model_config_sha256":"fixture"}
        self.protocol = train_protocol(run,rt)
        for name,data in (("completed",done),("run",run),("runtime",rt),("args",{}),("exit",{"code":0})):
            put(train/(name+".json"),data)
        jsonl(train/"steps.jsonl",[dict(step=1,input_tokens=self.goal)])
        stamp(root,"train-"+tag,[train/(n+".json") for n in ("completed","args","exit")],study.digest(SCRIPTS/"train_experiment.py"))
        ep = dict(label=tag,seed=seed,method=arm,checkpoint=key,**token_boundary(train,self.goal))
        self.locks[root][tag] = key
        if root == self.primary:
            self.model_paths[key] = [str(adapter), ah]
        else:
            put(root/"locks"/(tag+".json"),ep)
        scores={}
        for stage, gold in (("dev",self.gold[:400]),("test",self.gold)):
            folder = root / "evaluation" / stage / key
            correct = 0
            reasons = Counter()
            def results():
                nonlocal correct
                for i, example in enumerate(gold):
                    hit = (i+seed)%5 < {"A":2,"B":1,"E2":2,"E2b":3}.get(arm,1)
                    text = "<tool_call><function=f></function></tool_call>" if hit else "invalid"
                    scored = judge(example,text)
                    correct += scored["correct"]
                    reasons[scored["reason"]] += 1
                    yield dict(id=example["id"],group_id=example["group_id"],raw_output=text,**scored)
            jsonl(folder/"predictions.jsonl",results())
            put(folder/"metrics.json",dict(count=len(gold),correct=correct,reason_counts=reasons,
                                            accuracy=correct/len(gold),wall_seconds=1))
            put(folder/"config.json",dict(self.config,adapter_sha256={"adapter_model.safetensors":ah}))
            files=[folder/"metrics.json",folder/"predictions.jsonl"]
            if root != self.primary:
                files.append(folder/"config.json")
            task = f"eval-{stage}-{key}" if root == self.primary else f"{stage}-{tag}"
            stamp(root,task,files,self.config["script_sha256"])
            scores[stage]=dict(count=len(gold),correct=correct,checkpoint=key)
        if root != self.primary:
            put(root/"cells"/(tag+".json"),dict(label=tag,seed=seed,method=arm,endpoint=ep,scores=scores,
                receipt_sha256={k:study.digest(root/"receipts"/(k+"-"+tag+".json")) for k in ("train","dev","test")}))


class DifficultyTests(unittest.TestCase):
    def test_missing_phase_or_seed_suppresses_all_statistics(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); phase=root/"one"; diff=root/"two"
            order=[f"s{s}-{m}" for s,m in ORDER]
            put(phase/"execution.json",dict(planned=order,completed=order[:-1],missing=order[-1:],status="partial_budget"))
            put(diff/"execution.json",dict(planned=study.LABELS,completed=study.LABELS[:2],missing=study.LABELS[2:],status="running"))
            r=study.summarize(root/"p",phase,diff,root/"eligibility")
            self.assertEqual((r["status"],r["models"],r["contrasts"]),("incomplete",{},{}))

    def test_paired_interval_counts_each_request_and_preserves_clusters(self):
        group={"a":"X","b":"X","c":"Y"}
        pairs={s:({"a":True,"b":True,"c":False},{"a":False,"b":False,"c":True}) for s in study.SEEDS}
        result=study.paired_summary(pairs,group)
        self.assertAlmostEqual(result["mean_difference_pp"],100/3)
        self.assertEqual(result["fixed_seed_group_ci95_pp"],[-100.,100.])
        with self.assertRaises(ValueError):
            study.paired_summary({42:pairs[42]},group)

    def test_full_tree_mirror_without_selection_alias(self):
        with TemporaryDirectory() as tmp:
            fx = StudyFixture(Path(tmp))
            alias = fx.phase / "selection"
            pin = study.digest(alias / "manifest.json")
            physical = fx.phase / f".selection-snapshot-{pin}-{'2' * 32}" / "tree"
            physical.parent.mkdir()
            alias.rename(physical)
            absent = study.summarize(*fx.roots)
            self.assertEqual(absent["status"], "complete")
            self.assertFalse(alias.exists() or alias.is_symlink())
            alias.symlink_to(physical.relative_to(fx.phase), target_is_directory=True)
            self.assertEqual(study.summarize(*fx.roots), absent)
            alias.unlink()
            (physical / "43/E4.jsonl").write_text("changed")
            with self.assertRaises(ValueError):
                study.summarize(*fx.roots)
            self.assertFalse(alias.exists() or alias.is_symlink())

    def test_entire_frozen_fixture_and_tampering(self):
        with TemporaryDirectory() as tmp:
            fx=StudyFixture(Path(tmp))
            r=study.summarize(*fx.roots)
            self.assertEqual(r["status"],"complete")
            self.assertEqual(r["requests"],2000)
            self.assertEqual(set(r["contrasts"]),set(study.CONTRASTS))
            self.assertAlmostEqual(r["contrasts"]["E2b-A"]["mean_difference_pp"],20)
            self.assertEqual(r["contrasts"]["E2-A"]["mean_difference_pp"],0)
            self.assertIn("2,000",study.markdown(r))
            f=fx.diff/"training/s42-E2b/checkpoint-1/adapter_model.safetensors"
            genuine=f.read_bytes();f.write_bytes(b"swapped")
            with self.assertRaises(ValueError):
                study.summarize(*fx.roots)
            f.write_bytes(genuine)
            # A maliciously rehashed outcome still cannot pass frozen strict re-scoring.
            folder=fx.primary/"evaluation/test/s42-A-step1"; p=folder/"predictions.jsonl"
            rows=list(study.read_rows(p)); rows[0]["raw_output"]="dishonest" if rows[0]["correct"] else "<tool_call><function=f></function></tool_call>"
            jsonl(p,rows)
            stamp(fx.primary,"eval-test-s42-A-step1",[folder/"metrics.json",p],fx.config["script_sha256"])
            with self.assertRaisesRegex(ValueError,"saved verdict"):
                study.summarize(*fx.roots)
