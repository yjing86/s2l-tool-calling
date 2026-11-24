"""Audit and compare the three registered difficulty-matched single-call endpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from difficulty_admission import replay
from evaluate_experiment import BASE_REVISION
from run_crossed_seeds import gen_protocol, token_boundary
from run_difficulty_control import phase1_selection_path, science_hash
from run_followup import completed_cells, train_protocol
from summarize_course import digest, indexed, quantile, read_json, read_model, read_rows
from summarize_followup import DRAW_COUNT, DRAW_SEED, NEW, SEEDS, receipt, require, seed_sensitivity


CONTRASTS = {
    "E2b-A": "Jointly matched selection versus unrestricted random selection",
    "E2-A": "Length-matched selection versus unrestricted random selection (Phase 1 reference)",
    "E2b-B": "Jointly matched selection versus whole-answer trajectory selection (descriptive)",
    "E2b-E2": "Jointly matched selection versus length-matched selection (exploratory)",
}
LABELS = [f"s{s}-E2b" for s in SEEDS]
SCRIPTS = Path(__file__).resolve().parent
CORE = {"summarize_difficulty_control.py", "run_difficulty_control.py", "select_difficulty_control.py", "difficulty_admission.py",
        "run_followup.py", "summarize_followup.py", "summarize_course.py", "run_crossed_seeds.py",
        "evaluate_experiment.py", "verify_tool_call_loss.py"}


def completion(root, required):
    """A partial execution never exposes subset-seed estimates."""
    file = root / "execution.json"
    if not file.is_file():
        return "awaiting execution record", False
    run = read_json(file)
    planned, done, missing = (run[k] for k in ("planned", "completed", "missing"))
    require(len(set(planned)) == len(required) == len(planned) and set(planned) == set(required)
            and len(set(done)) == len(done) and done + missing == planned, "Invalid endpoint partition")
    if run["status"] != "complete":
        return run["status"], False
    require(not missing and len(done) == len(required), "Complete execution is missing endpoints")
    locks = read_json(root / "locked-models.json")
    require(set(locks) == set(required), "Final endpoint locks differ from the completed run")
    return "complete", True


def paired_summary(all_pairs, groups):
    """Resample held-out request groups once per draw, sharing each draw across all three fits."""
    require(set(all_pairs) == set(SEEDS) and bool(groups), "Need all three paired seeds")
    sizes, totals, by_seed = Counter(groups.values()), Counter(), {}
    for seed in SEEDS:
        left, right = all_pairs[seed]
        require(left.keys() == right.keys() == groups.keys(), "Unpaired outcome IDs")
        counts = Counter((left[k], right[k]) for k in groups)
        for k, g in groups.items():
            totals[g] += int(left[k]) - int(right[k])
        by_seed[str(seed)] = {"left_only": counts[True, False], "right_only": counts[False, True],
                              "both_correct": counts[True, True], "both_wrong": counts[False, False],
                              "difference_pp": 100 * (counts[True, False] - counts[False, True]) / len(groups)}
    population = [(totals[g], len(SEEDS) * sizes[g]) for g in sorted(sizes)]
    rng = random.Random(DRAW_SEED)
    draws = []
    for _ in range(DRAW_COUNT):
        numerator = denominator = 0
        for delta, exposure in rng.choices(population, k=len(population)):
            numerator += delta
            denominator += exposure
        draws.append(100 * numerator / denominator)
    draws.sort()
    differences = [item["difference_pp"] for item in by_seed.values()]
    return {"paired_by_seed": by_seed, "mean_difference_pp": statistics.mean(differences),
            "fixed_seed_group_ci95_pp": [quantile(draws, p) for p in (.025, .975)],
            "fixed_seed_group_ci90_pp": [quantile(draws, p) for p in (.05, .95)],
            "seed_t_sensitivity": seed_sensitivity(differences)}


def original_receipt(root, seed, arm, subset):
    manifest = subset.parent / "selection.json"
    receipt(root, f"select-{seed}", [manifest, subset])
    record = read_json(manifest)["subsets"][arm]
    require(record["sha256"] == digest(subset), "Original selection differs from its receipt")
    return record["ids"]


def endpoint(root, origin, seed, arm, spec, gold, groups, pool, selected, subset_ids, name, original_paths):
    label = f"s{seed}-{arm}"
    train = root / "training" / label
    done, run, runtime = [read_json(train / n) for n in ("completed.json", "run.json", "runtime.json")]
    goal = spec["training"]["runtime"]["token_goal"]
    bound = token_boundary(train, goal)
    step, adapter = bound["step"], bound["final_adapter_sha256"]
    require(name == f"{label}-step{step}" and adapter == digest(train / f"checkpoint-{step}/adapter_model.safetensors"),
            f"{label}: first-crossing adapter lock mismatch")
    require(run["stage"] == "target" and run["seed"] == seed and run["pool_sha256"] == spec["pool_sha256"]
            and train_protocol(run, runtime) == spec["training"], f"{label}: target training differs")
    rows = list(read_rows(selected))
    keys = [str(r["id"]) for r in rows]
    require(keys == list(map(str, subset_ids)) == list(map(str, run["ids"])) and len(keys) == len(set(keys)) == 1000
            and all(row == pool.get(k) for row, k in zip(rows, keys)), f"{label}: subset identity differs")
    require(digest(selected) == run["data_sha256"], f"{label}: training input digest differs")
    train_rec = receipt(root, "train-" + label, [train / n for n in ("completed.json", "args.json", "exit.json")])
    require(read_json(train / "exit.json")["code"] == 0
            and train_rec["script_sha256"] == digest(SCRIPTS / "train_experiment.py"), f"{label}: wrong training receipt")
    folder = root / "evaluation/test" / name
    pred_file, metric_file = folder / "predictions.jsonl", folder / "metrics.json"
    config = read_json(folder / "config.json")
    test_spec = spec["splits"]["test"]["generation"]
    require(gen_protocol(config) == test_spec and config["model_revision"] == BASE_REVISION and config["args"]["generate_only"] is False
            and config["adapter_sha256"].get("adapter_model.safetensors") == adapter,
            f"{label}: held-out split, generation protocol or adapter differs")
    if origin == "primary":
        require(original_paths[name][1] == adapter, f"{label}: archived primary lock mismatch")
        task = "eval-test-" + name
        outputs = [metric_file, pred_file]
    else:
        require(run["data_sha256"] == spec["selections"][label]["sha256"], f"{label}: registration mismatch")
        lock = read_json(root / "locks" / (label + ".json"))
        require(lock == dict(label=label, seed=seed, method=arm, checkpoint=name, **bound),
                f"{label}: pre-generation lock differs")
        cell = read_json(root / "cells" / (label + ".json"))
        require(cell["endpoint"] == lock and (cell["label"], cell["method"], cell["seed"]) == (label, arm, seed),
                f"{label}: cell mismatch")
        for stage in ("train", "dev", "test"):
            require(cell["receipt_sha256"][stage] == digest(root / "receipts" / f"{stage}-{label}.json"),
                    f"{label}: recorded {stage} receipt differs")
        task = "test-" + label
        outputs = [metric_file, pred_file, folder / "config.json"]
    test_rec = receipt(root, task, outputs)
    require(test_rec["script_sha256"] == test_spec["script_sha256"] == digest(SCRIPTS / "evaluate_experiment.py"),
            f"{label}: evaluation source mismatch")
    correct, metrics, _ = read_model(folder, groups, test_spec["source_sha256"], {})
    replay(folder, gold, test_spec["source_sha256"])
    if origin != "primary":
        require(cell["scores"]["test"] == {"count": len(groups), "correct": metrics["correct"], "checkpoint": name},
                f"{label}: cell score differs from its raw predictions")
    return correct, {"origin": origin, "checkpoint": name, "correct": metrics["correct"], "count": len(groups),
                     "accuracy_pct": metrics["accuracy_pct"], "adapter_sha256": adapter,
                     "predictions_sha256": digest(pred_file), "generation_config_sha256": digest(folder / "config.json"),
                     "training_subset_sha256": run["data_sha256"], "actual_input_tokens": done["input_tokens"],
                     "train_seconds": train_rec["elapsed_s"], "test_seconds": test_rec["elapsed_s"]}


def summarize(primary, phase1, difficulty, eligibility):
    dependencies = {"phase1": completion(phase1, NEW), "difficulty": completion(difficulty, LABELS)}
    state = {key: v[0] for key, v in dependencies.items()}
    if not all(v[1] for v in dependencies.values()):
        return {"status": "incomplete", "execution_status": state, "models": {}, "contrasts": {}}
    reg = phase1 / "preregistration.json"
    new_reg = difficulty / "preregistration.json"
    original, current = read_json(reg)["spec"], read_json(new_reg)["spec"]
    registry = eligibility / "manifest.json"
    primary_lock = primary / "locked-models.json"
    pins = {"phase1_preregistration_sha256": digest(reg), "eligibility_manifest_sha256": digest(registry),
            "phase1_spec_sha256": science_hash(original), "primary_design_sha256": digest(primary / "design.json"),
            "primary_lock_sha256": digest(primary_lock), "pool_sha256": digest(primary / "frozen/train.jsonl")}
    require(all(current[k] == v for k, v in pins.items()), "Difficulty registration or upstream identity differs")
    require(CORE <= set(current["source_sha256"]), "Missing registered analysis or scoring source")
    for spec in (original, current):
        for filename, expected in spec["source_sha256"].items():
            require(Path(filename).name == filename and digest(SCRIPTS / filename) == expected,
                    f"Registered source differs: {filename}")
    require(current["analysis"]["contrasts"] == list(CONTRASTS) and current["cell_order"] == LABELS,
            "Registered design differs from this analysis")
    require(original["cell_order"] == read_json(phase1 / "execution.json")["planned"]
            and current["cell_order"] == read_json(difficulty / "execution.json")["planned"],
            "Execution order differs from registration")
    require(all(original[k] == current[k] for k in ("training", "splits", "primary_lock_sha256", "pool_sha256"))
            and current["training"]["runtime"]["token_goal"] == 3005228, "Primary and follow-up protocols differ")
    dependencies_file, admission_file = difficulty / "dependencies.json", difficulty / "admission.json"
    proof, admission = read_json(dependencies_file), read_json(admission_file)
    require(proof["attestation_sha256"] == admission["attestation_sha256"] == admission["decision"]["attestation_sha256"]
            and proof["evidence"]["verifier_sha256"] == digest(SCRIPTS / "difficulty_admission.py")
            and admission["decision"]["fits"] and all(v is True for v in admission["decision"]["fits"].values()),
            "Admission and dependency evidence disagree")
    locks_by_root = {}
    for root, spec in ((phase1, original), (difficulty, current)):
        locks = read_json(root / "locked-models.json")
        require(completed_cells(SimpleNamespace(root=root), spec) == locks, "Completed receipts and locks differ")
        locks_by_root[root] = locks
    pool = indexed(read_rows(primary / "frozen/train.jsonl"), "id", "pool")
    frozen = primary / "frozen/test.jsonl"
    gold = indexed(read_rows(frozen), "id", frozen)
    test_spec = current["splits"]["test"]["generation"]
    groups = {k: str(v["group_id"]) for k, v in gold.items()}
    require(len(pool) == 4000 and len(groups) == 2000 and not (pool.keys() & gold.keys())
            and all(v["group_id"] is not None and groups[k] for k, v in gold.items())
            and digest(frozen) == test_spec["source_sha256"], "Wrong original training/test population")
    base = read_json(primary / "evaluation/test/base/config.json")
    require(gen_protocol(base) == test_spec and base["args"]["generate_only"] is False and not base["adapter_sha256"],
            "Original unadapted test generation protocol differs")
    locks_by_root[primary] = read_json(primary_lock)["main"]
    original_paths = read_json(primary / "model-paths.json")
    eligible = read_json(registry)
    require(eligible["seeds"] == list(SEEDS) and eligible["common_target_input_token_budget"] == 3005228
            and eligible["sources"]["train_sha256"] == pins["pool_sha256"] and eligible["sources"] == current["sources"],
            "Eligibility protocol differs")
    phase_selection = phase1_selection_path(phase1, original["selection_manifest_sha256"]).resolve()
    phase_manifests = read_json(phase_selection / "manifest.json")["selection_manifest_sha256"]
    for seed in SEEDS:
        directory = phase_selection / str(seed)
        file = directory / "selection.json"
        require(file.resolve() == file and digest(file) == phase_manifests[str(seed)],
                f"Phase 1 seed {seed}: selection manifest mismatch")
        records = read_json(file)["subsets"]
        for arm in ("Bprime", "E2", "E3", "E4"):
            subset = directory / f"{arm}.jsonl"
            require(subset.resolve() == subset and digest(subset) == records[arm]["sha256"],
                    f"Phase 1 seed {seed}: training list changed for {arm}")
    predictions, models = {}, {}
    for seed in SEEDS:
        for arm in ("A", "B", "E2", "E2b"):
            label = f"s{seed}-{arm}"
            if arm in ("A", "B"):
                root, origin, spec = primary, "primary", original
                selected = root / "selection" / str(seed) / f"{arm}.jsonl"
                ids = original_receipt(primary, seed, arm, selected)
            else:
                root, origin, spec = (phase1, "phase1", original) if arm == "E2" else (difficulty, "difficulty", current)
                upstream = phase_selection if arm == "E2" else eligibility
                selection_dir = upstream / str(seed)
                src = selection_dir / (arm + ".jsonl")
                selected = src if arm == "E2" else difficulty / "selection" / str(seed) / "E2b.jsonl"
                master = read_json(upstream / "manifest.json")
                info = read_json(selection_dir / "selection.json")
                provenance = spec["selections"][label]
                hash_key = "selection_manifest_sha256" if arm == "E2" else "seed_manifest_sha256"
                record = info["subsets"][arm] if arm == "E2" else info
                require(provenance["seed_manifest_sha256"] == master[hash_key][str(seed)] == digest(selection_dir / "selection.json")
                        and record["sha256"] == digest(src) == provenance["sha256"] and provenance["size"] == 1000,
                        f"{label}: subset snapshot differs from its registration")
                ids = record["ids"]
            name = locks_by_root[root][label]
            predictions[seed, arm], models[label] = endpoint(
                root, origin, seed, arm, spec, gold, groups, pool, selected, ids, name, original_paths)
    results = {}
    for name, question in CONTRASTS.items():
        left, right = name.split("-")
        pairs = {seed: (predictions[seed, left], predictions[seed, right]) for seed in SEEDS}
        results[name] = {"question": question, **paired_summary(pairs, groups)}
    return {"status": "complete", "execution_status": state, "seeds": list(SEEDS), "requests": len(gold),
            "groups": len(set(groups.values())), "target_input_token_budget": 3005228, "models": models,
            "contrasts": results, "bootstrap": {"draws": DRAW_COUNT, "seed": DRAW_SEED, "unit": "request group"},
            "inputs_sha256": {**pins, "difficulty_preregistration": digest(new_reg), "frozen_test": digest(frozen),
                              "admission": digest(admission_file), "dependencies": digest(dependencies_file),
                              "phase1_locks": digest(phase1 / "locked-models.json"),
                              "difficulty_locks": digest(difficulty / "locked-models.json")}}


def interval(ci):
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def markdown(result):
    lines = ["# Difficulty-matched single-call control", ""]
    if result["status"] != "complete":
        return "\n".join(lines + ["Cross-seed statistics await all eleven Phase 1 endpoints and all three E2b endpoints.", ""])
    lines += [f"All 12 target endpoints were scored on the same {result['requests']:,} original held-out requests "
              f"in {result['groups']:,} near-duplicate groups. The first update reaching the common "
              f"{result['target_input_token_budget']:,}-token budget determined each checkpoint.", "",
              "| Method | Seed 42 | Seed 43 | Seed 44 |", "| --- | ---: | ---: | ---: |"]
    for arm in ("A", "B", "E2", "E2b"):
        scores = [result["models"][f"s{s}-{arm}"] for s in SEEDS]
        lines.append("| " + " | ".join([arm, *(f"{m['correct']:,} ({m['accuracy_pct']:.2f}%)" for m in scores)]) + " |")
    lines += ["", "A samples uniformly. B selects using whole-answer loss trajectories. E2 draws randomly "
              "within length bins matching B. E2b matches A’s value-length bins and B’s initial argument-value "
              "difficulty bins jointly, with uniform draws within cells.", "",
              "| Contrast | Differences at seeds 42 / 43 / 44 (pp) | Mean (pp) | Paired group 95% interval | Three-seed t 95% interval |",
              "| --- | --- | ---: | --- | --- |"]
    for name, c in result["contrasts"].items():
        ds = " / ".join(f"{c['paired_by_seed'][str(s)]['difference_pp']:+.2f}" for s in SEEDS)
        lines.append(f"| {name} | {ds} | {c['mean_difference_pp']:+.2f} | {interval(c['fixed_seed_group_ci95_pp'])} "
                     f"| {interval(c['seed_t_sensitivity']['ci95_pp'])} |")
    lines += ["", "Differences are left minus right in percentage points. Each of 2,000 bootstrap draws samples "
              "entire request groups with replacement and shares that draw across all three trained seeds. "
              "The seed t interval keeps the requests fixed and has two degrees of freedom. "
              "Selection and training seeds varied together. The JSON includes the corresponding 90% intervals and discordant counts.",
              "", "These four pointwise contrasts reuse the original internal test after its primary results were observed. "
              "E2b−B is descriptive and E2b−E2 is exploratory. Accuracy differences can reflect residual variation within bins, "
              "tool composition and other differences between selected requests. Generalization to independent tool-use tasks "
              "requires a separate evaluation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=Path("local/course-run"))
    parser.add_argument("--phase1", type=Path, default=Path("local/followup/phase1"))
    parser.add_argument("--difficulty", type=Path, default=Path("local/followup/difficulty-control"))
    parser.add_argument("--eligibility", type=Path, default=Path("local/followup/e2b-eligibility"))
    parser.add_argument("--out", type=Path, help="Output directory (default: DIFFICULTY/summary)")
    args = parser.parse_args()
    result = summarize(args.primary, args.phase1, args.difficulty, args.eligibility)
    output = args.out or args.difficulty / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        temp = Path(temp)
        (temp / "difficulty-control.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (temp / "difficulty-control.md").write_text(markdown(result))
        for name in ("difficulty-control.json", "difficulty-control.md"):
            (temp / name).replace(output / name)
    print(f"Difficulty control: {result['status']}. Wrote {output}.")


if __name__ == "__main__":
    main()
