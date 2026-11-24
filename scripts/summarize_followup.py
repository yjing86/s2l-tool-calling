"""Aggregate the frozen single-call Phase 1 comparisons, uncertainty and measured GPU costs."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
from tempfile import TemporaryDirectory

from run_crossed_seeds import gen_protocol, token_boundary
from summarize_course import digest, indexed, quantile, read_json, read_model, read_rows


SEEDS = (42, 43, 44)
NEW = {f"s{s}-{m}" for s in SEEDS for m in ("Bprime", "E2", "E3", "E4")} - {"s42-Bprime"}
METHODS = ("A", "B", "Bprime", "C", "E2", "E3", "E4")
COMPARISONS = {
    "Bprime-B": "Change the scored positions to the two fields and pool their tokens into one trajectory.",
    "C-Bprime": "Keep both fields and separate their trajectories in the clustering features.",
    "E2-B": "Compare random draws matched to B's value-length-bin counts with B's trajectory selection.",
    "E2-A": "Compare length-matched random draws with unrestricted random selection.",
    "E3-B": "Compare residualized and original whole-answer trajectories.",
    "E3-A": "Compare residualized whole-answer trajectories with unrestricted random selection.",
    "E4-A": "Compare proportional tool-name stratification with unrestricted random selection.",
}
EQUIVALENCE_CONTRASTS = {"E2-B", "E2-A", "E3-A"}
DRAW_COUNT = 2000
DRAW_SEED = 31415
MARGIN_PP = 1.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(folder, task, outputs=()):
    rec = read_json(folder / "receipts" / f"{task}.json")
    require(rec["task"] == task and math.isfinite(rec["elapsed_s"]) and rec["elapsed_s"] > 0,
            f"Invalid receipt for {task}")
    for path in outputs:
        suffix = "/" + path.relative_to(folder).as_posix()
        hashes = [value for name, value in rec["outputs"].items() if name.endswith(suffix)]
        require(hashes == [digest(path)], f"Receipt mismatch for {task} / {path.name}")
    return rec


def seed_sensitivity(differences):
    if len(differences) != 3:
        return None
    mean, sd = statistics.mean(differences), statistics.stdev(differences)
    se = sd / math.sqrt(3)
    intervals = {
        name: [mean - t * se, mean + t * se]
        for name, t in (("ci95_pp", 4.30265272974946), ("ci90_pp", 2.91998558035372))
    }
    return {"df": 2, "sample_sd_pp": sd, "zero_observed_seed_variance": sd == 0, **intervals}


def equivalence_decision(ci90, planned, complete):
    if not planned:
        return "no prespecified equivalence decision"
    if not complete:
        return "incomplete seed set"
    if ci90[0] > -MARGIN_PP and ci90[1] < MARGIN_PP:
        return "within margin for the fixed trained models"
    return "equivalence not established"


def interval_direction(bounds):
    if bounds[0] > 0:
        return "positive"
    if bounds[1] < 0:
        return "negative"
    return "compatible with zero"


def summarize_contrast(name, seed_pairs, groups):
    seeds = sorted(seed_pairs)
    out = {"question": COMPARISONS[name], "seeds": seeds, "missing_seeds": sorted(set(SEEDS) - set(seeds))}
    if not seeds:
        return {**out, "paired_by_seed": {}, "mean_difference_pp": None,
                "equivalence_decision": equivalence_decision(None, name in EQUIVALENCE_CONTRASTS, False)}
    sizes = Counter(groups.values())
    by_group = Counter()
    by_seed = {}
    for seed in seeds:
        left, right = seed_pairs[seed]
        require(left.keys() == right.keys() == groups.keys(), f"{name} seed {seed}: unpaired requests")
        counts = Counter((left[i], right[i]) for i in groups)
        for i, group in groups.items():
            by_group[group] += int(left[i]) - int(right[i])
        by_seed[str(seed)] = {
            "left_only": counts[True, False], "right_only": counts[False, True],
            "both_correct": counts[True, True], "both_wrong": counts[False, False],
            "difference_pp": 100 * (counts[True, False] - counts[False, True]) / len(groups),
        }
    # The original course summary sorts group IDs and draws whole groups with replacement.
    # Each draw here is shared across seeds, and reusing this seed across contrasts aligns the draws.
    totals = [(by_group[g], len(seeds) * sizes[g]) for g in sorted(sizes)]
    rng = random.Random(DRAW_SEED)
    draws = []
    for _ in range(DRAW_COUNT):
        num = den = 0
        for d, n in rng.choices(totals, k=len(totals)):
            num += d
            den += n
        draws.append(100 * num / den)
    draws.sort()
    ci90 = [quantile(draws, p) for p in (.05, .95)]
    ci95 = [quantile(draws, p) for p in (.025, .975)]
    deltas = [by_seed[str(s)]["difference_pp"] for s in seeds]
    t_summary = seed_sensitivity(deltas)
    return {
        **out, "paired_by_seed": by_seed, "mean_difference_pp": 100 * sum(by_group.values()) / (len(groups) * len(seeds)),
        "fixed_seed_group_ci95_pp": ci95, "fixed_seed_group_ci90_pp": ci90, "seed_t_sensitivity": t_summary,
        "conditional_direction_95": interval_direction(ci95),
        "seed_sensitivity_direction_95": interval_direction(t_summary["ci95_pp"]) if t_summary else None,
        "direction_counts": {"positive": sum(x > 0 for x in deltas), "zero": deltas.count(0),
                             "negative": sum(x < 0 for x in deltas)},
        "equivalence_decision": equivalence_decision(ci90, name in EQUIVALENCE_CONTRASTS, len(seeds) == 3),
    }


def costs(seconds, proxy_dependency):
    require(all(math.isfinite(s) and s >= 0 for s in [*seconds.values(), proxy_dependency]), "Invalid timing")
    return {
        "observed_task_seconds": seconds,
        "target_train_gpu_hours": seconds["train"] / 3600,
        "observed_train_dev_test_gpu_hours": sum(seconds.values()) / 3600,
        "standalone_proxy_train_and_scoring_gpu_hours": proxy_dependency / 3600,
        "standalone_train_dev_plus_proxy_gpu_hours": (seconds["train"] + seconds["dev"] + proxy_dependency) / 3600,
        "standalone_train_dev_test_plus_proxy_gpu_hours": (sum(seconds.values()) + proxy_dependency) / 3600,
    }


def summarize(primary, followup):
    prereg_file, execution_file = followup / "preregistration.json", followup / "execution.json"
    prereg, execution = read_json(prereg_file), read_json(execution_file)
    spec = prereg["spec"]
    for filename in ("summarize_followup.py", "summarize_course.py", "run_crossed_seeds.py"):
        source = Path(__file__).resolve().with_name(filename)
        require(digest(source) == spec["source_sha256"].get(filename),
                f"Analysis code changed since Phase 1 registration: {filename}")
    completed, missing = set(execution["completed"]), set(execution["missing"])
    require(execution["status"] in ("complete", "partial_budget"), "Expected a finalized follow-up execution")
    require(len(completed) == len(execution["completed"]) and len(missing) == len(execution["missing"])
            and not completed & missing and completed | missing == NEW, "Invalid completion partition")
    require((execution["status"] == "complete") == (not missing), "Status differs from missing endpoints")
    order = spec["cell_order"]
    require(set(order) == NEW and len(order) == len(NEW) and order == execution["planned"]
            and order[:len(completed)] == execution["completed"] and order[len(completed):] == execution["missing"],
            "Completion differs from registered admission order")
    require(spec["analysis"]["contrasts"] == list(COMPARISONS)
            and spec["primary_lock_sha256"] == digest(primary / "locked-models.json"), "Preregistration differs")
    supplemental = read_json(followup / "locked-models.json")
    require(set(supplemental) == completed, "Follow-up locks differ from completed cells")
    main = read_json(primary / "locked-models.json")["main"]
    primary_models = read_json(primary / "model-paths.json")
    main_ledger = read_json(primary / "training-ledger.json")
    pool = primary / "frozen/train.jsonl"
    test = primary / "frozen/test.jsonl"
    gold = indexed(({"id": row["id"], "group_id": row["group_id"]} for row in read_rows(test)), "id", test)
    require(len(gold) == 2000 and all(row["group_id"] is not None and str(row["group_id"]) for row in gold.values()),
            "Expected the frozen 2,000 test requests and their group IDs")
    groups = {key: str(row["group_id"]) for key, row in gold.items()}
    source_hash, pool_hash = digest(test), digest(pool)
    require(source_hash == spec["splits"]["test"]["generation"]["source_sha256"]
            and pool_hash == spec["pool_sha256"], "Frozen Phase 1 source identities differ")
    common_seconds = receipt(primary, "proxy")["elapsed_s"] + receipt(primary, "score")["elapsed_s"]
    require(math.isclose(common_seconds, main_ledger["common_proxy_and_scoring_seconds"], abs_tol=1e-6),
            "Shared proxy cost differs from its receipts")
    expected_goal = read_json(primary / "training/s42-A/runtime.json")["token_goal"]
    reference = primary / "evaluation/test/base"
    base_config = read_json(reference / "config.json")
    common_generation = gen_protocol(base_config)
    require(common_generation == spec["splits"]["test"]["generation"] and not base_config["args"]["generate_only"],
            "Frozen test generation settings differ")
    checksums = {}
    _, _, reference_protocol = read_model(reference, groups, source_hash, checksums)
    receipt(primary, "eval-test-base", [reference / "predictions.jsonl", reference / "metrics.json"])
    models, predictions = {}, {}
    for method in METHODS:
        for seed in SEEDS:
            label = f"s{seed}-{method}"
            if label in missing:
                continue
            new = label in NEW
            root = followup if new else primary
            name = supplemental[label] if new else main[label]
            training = root / "training" / label
            done = read_json(training / "completed.json")
            step = done["final_step"]
            require(name == f"{label}-step{step}", f"{label}: lock is not the final equal-token checkpoint")
            run = read_json(training / "run.json")
            selected = root / "selection" / str(seed) / f"{method}.jsonl"
            require(run["seed"] == seed and run["pool_sha256"] == pool_hash and run["data_sha256"] == digest(selected),
                    f"{label}: training inputs differ")
            require(len(run["ids"]) == len(set(map(str, run["ids"]))) == 1000, f"{label}: wrong subset size")
            require(read_json(training / "runtime.json")["token_goal"] == expected_goal, f"{label}: token goal differs")
            bound = token_boundary(training, expected_goal)
            folder = root / "evaluation/test" / name
            values, stats, protocol = read_model(folder, groups, source_hash, checksums)
            conf = read_json(folder / "config.json")
            require(protocol == reference_protocol and gen_protocol(conf) == common_generation and not conf["args"]["generate_only"],
                    f"{label}: test protocol differs")
            adapter = done["adapter_sha256"][str(step)]
            require(stats["adapter_sha256"].get("adapter_model.safetensors") == adapter, f"{label}: wrong test adapter")
            if new:
                lock = read_json(followup / "locks" / f"{label}.json")
                cell = read_json(followup / "cells" / f"{label}.json")
                require(cell["label"] == label and cell["seed"] == seed and cell["method"] == method
                        and cell["endpoint"] == lock == dict(label=label, seed=seed, method=method, checkpoint=name, **bound)
                        and spec["selections"][label]["sha256"] == run["data_sha256"],
                        f"{label}: cell differs from its lock")
                dev_folder = root / "evaluation/dev" / name
                seconds = {"train": receipt(root, f"train-{label}", [
                    training / f for f in ("completed.json", "exit.json", "args.json")
                ])["elapsed_s"]}
                for task, eval_folder in (("dev", dev_folder), ("test", folder)):
                    seconds[task] = receipt(root, f"{task}-{label}", [
                        eval_folder / f for f in ("metrics.json", "predictions.jsonl", "config.json")
                    ])["elapsed_s"]
                    m = read_json(eval_folder / "metrics.json")
                    c = cell["scores"][task]
                    eval_config = read_json(eval_folder / "config.json")
                    require(c["count"] == m["count"] == (400 if task == "dev" else 2000)
                            and c["correct"] == m["correct"] and c["checkpoint"] == name
                            and gen_protocol(eval_config) == spec["splits"][task]["generation"]
                            and eval_config["adapter_sha256"].get("adapter_model.safetensors") == adapter,
                            f"{label}: {task} scores differ")
                require(all(cell["receipt_sha256"][task] == digest(root / "receipts" / f"{task}-{label}.json")
                            for task in seconds), f"{label}: recorded receipt digest differs")
                require(all(math.isclose(sec, cell["task_seconds"][task], abs_tol=1e-6)
                            for task, sec in seconds.items()), f"{label}: task timing differs")
            else:
                require(primary_models[name][1] == adapter, f"{label}: original lock digest differs")
                train_sec = receipt(root, f"train-{label}")["elapsed_s"]
                history = main_ledger["routes"][label]["history"]
                dev_sec = sum(receipt(root, "eval-dev-" + cp["name"])["elapsed_s"] for cp in history if cp["name"] != "base")
                seconds = {"train": train_sec, "dev": dev_sec, "test": receipt(
                    root, f"eval-test-{name}", [folder / "metrics.json", folder / "predictions.jsonl"]
                )["elapsed_s"]}
            predictions[method, seed] = values
            models[label] = {"checkpoint": name, "origin": "followup" if new else "primary",
                             "correct": stats["correct"], "requests": len(groups), "accuracy_pct": stats["accuracy_pct"],
                             "actual_input_tokens": done["input_tokens"], "answer_tokens": done["answer_tokens"],
                             "predictions_sha256": digest(folder / "predictions.jsonl"), "adapter_sha256": adapter,
                             **costs(seconds, common_seconds if method not in ("A", "E4") else 0)}
    contrasts = {}
    for name in COMPARISONS:
        left, right = name.split("-")
        pairs = {s: (predictions[left, s], predictions[right, s]) for s in SEEDS
                 if (left, s) in predictions and (right, s) in predictions}
        contrasts[name] = summarize_contrast(name, pairs, groups)
    new_seconds = {task: sum(models[k]["observed_task_seconds"][task] for k in completed)
                   for task in ("train", "dev", "test")}
    return {
        "status": execution["status"], "completed_new_cells": sorted(completed), "missing_new_cells": sorted(missing),
        "requests": len(groups), "near_duplicate_groups": len(set(groups.values())), "seeds": list(SEEDS),
        "input_sha256": {"frozen_test": source_hash, "pool": pool_hash, "preregistration": digest(prereg_file),
                         "execution": digest(execution_file), "primary_locks": digest(primary / "locked-models.json"),
                         "followup_locks": digest(followup / "locked-models.json")},
        "token_target": expected_goal, "models": models, "contrasts": contrasts, "patterns": patterns(contrasts),
        "interval": {"draws": DRAW_COUNT, "seed": DRAW_SEED, "unit": "near-duplicate request group",
                     "equivalence_margin_pp": MARGIN_PP, "equivalence_contrasts": sorted(EQUIVALENCE_CONTRASTS),
                     "multiplicity": "Pointwise exploratory intervals for seven planned contrasts on a previously examined test set."},
        "cost": {"completed_new_cells_seconds": new_seconds, "completed_new_cells_gpu_hours": sum(new_seconds.values()) / 3600,
                 "shared_historical_proxy_and_scoring_gpu_hours": common_seconds / 3600,
                 "scope": "Completed train, dev and test subprocess wall time on one GPU. The proxy was reused once. "
                          "Setup, selection CPU time, failed attempts, idle time and provider billing are outside these totals."},
    }


def patterns(contrasts):
    definitions = {
        "length_matching": ("E2-B", "E2-A"),
        "residualizing_length_features": ("E3-B", "E3-A"),
        "tool_stratification": ("E4-A",),
    }
    results = {}
    for key, names in definitions.items():
        if any(contrasts[n]["missing_seeds"] for n in names):
            results[key] = {"status": "incomplete paired seeds"}
            continue
        if key == "length_matching":
            e2a = contrasts["E2-A"]
            criteria = {"E2 and B conditionally equivalent": contrasts["E2-B"]["equivalence_decision"].startswith("within"),
                        "E2 minus A mean at most -1 pp": e2a["mean_difference_pp"] <= -MARGIN_PP,
                        "all three E2 minus A differences negative": e2a["direction_counts"]["negative"] == 3}
        elif key == "residualizing_length_features":
            criteria = {"E3 minus B mean at least +1 pp": contrasts["E3-B"]["mean_difference_pp"] >= MARGIN_PP,
                        "E3 and A conditionally equivalent": contrasts["E3-A"]["equivalence_decision"].startswith("within")}
        else:
            criteria = {"E4 minus A mean at least +1 pp": contrasts["E4-A"]["mean_difference_pp"] >= MARGIN_PP}
        results[key] = {"status": "prespecified pattern observed" if all(criteria.values()) else "criteria not met",
                        "criteria": criteria}
    return results


def fmt_interval(values):
    return f"[{values[0]:+.3f}, {values[1]:+.3f}]"


def markdown(result):
    completed, missing = result["completed_new_cells"], result["missing_new_cells"]
    lines = [
        "# Phase 1 single-call data-selection results", "",
        f"{len(completed)} of 11 new target-model runs completed. All primary checkpoints, including the original "
        f"seed-42 B′, are reused. Each locked endpoint was scored on {result['requests']:,} Turnstile requests. "
        f"The target training budget was {result['token_target']:,} non-padding input tokens per run.", "",
        "## Locked test accuracy", "", "| Method | Seed 42 | Seed 43 | Seed 44 |",
        "| --- | ---: | ---: | ---: |",
    ]
    if missing:
        lines[3:3] = ["The budget left these planned runs unfinished: " + ", ".join(missing) + ".", ""]
    for method in METHODS:
        cells = []
        for s in SEEDS:
            m = result["models"].get(f"s{s}-{method}")
            cells.append(f"{m['correct']} ({m['accuracy_pct']:.2f}%)" if m else "—")
        lines.append("| " + " | ".join(["B′ (E1)" if method == "Bprime" else method, *cells]) + " |")
    lines.extend([
        "", "A samples uniformly. B clusters whole-answer trajectories. B′ (E1) pools function-name and "
        "argument-value tokens into one trajectory. C keeps the two fields separate. E2 samples randomly "
        "within value-length bins using B's counts at that seed. E3 clusters whole-answer trajectories "
        "after fitting out length and field-composition covariates at each checkpoint. E4 allocates the "
        "sample proportionally across reference tool names. E2 and E3 share the original no-value sample "
        "with B/C; E4 stratifies the entire pool.",
        "", "## Paired contrasts", "",
        "The sign is left minus right, in percentage points. Each paired seed uses the same requests. "
        "The group intervals resample whole near-duplicate request groups 2,000 times and keep the observed "
        "models together. The seed t interval holds the test set fixed and treats three paired seed differences "
        "as approximately normal draws. It has two degrees of freedom. These coupled selection/training seeds "
        "share a proxy, so extrapolation to future training runs is uncertain.", "",
        "| Contrast | Paired seeds | Differences at 42 / 43 / 44 | Mean | Group 95% CI | Group 90% CI | Seed t 95% CI | Conditional equivalence |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- |",
    ])
    for name, c in result["contrasts"].items():
        if not c["seeds"]:
            lines.append(f"| {name} | 0 | — | — | — | — | — | {c['equivalence_decision']} |")
            continue
        ds = " / ".join(f"{c['paired_by_seed'][str(s)]['difference_pp']:+.2f}"
                        if s in c["seeds"] else "—" for s in SEEDS)
        t = c["seed_t_sensitivity"]
        cells = [name, str(len(c["seeds"])), ds, f"{c['mean_difference_pp']:+.3f}",
                 fmt_interval(c["fixed_seed_group_ci95_pp"]), fmt_interval(c["fixed_seed_group_ci90_pp"]),
                 fmt_interval(t["ci95_pp"]) if t else "—", c["equivalence_decision"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "", "For E2−B, E2−A and E3−A, conditional equivalence requires three complete seed pairs and an "
        "unrounded 90% paired group interval strictly inside (−1, +1) pp. This approximate interval-based TOST "
        "criterion uses a 5% level per one-sided test. A 90% interval that touches or extends beyond either "
        "margin leaves equivalence unestablished. An interval entirely inside the margins may include zero. "
        "The JSON also gives the 90% seed t "
        "sensitivity interval. With three identical measured seed differences its calculated width is zero, "
        "which supplies little information about future seed variance.", "",
        "The seven contrasts are exploratory, pointwise and unadjusted for multiplicity. The original outcome "
        "on this test was known before the follow-up was designed. B′ at seed 42 had also been evaluated. "
        "The follow-up tests whether patterns recur within this design. An independent test population would "
        "be needed to confirm transfer.", "", "## Mechanism diagnostics", "",
    ])
    for name, record in result["patterns"].items():
        crit = record.get("criteria", {})
        detail = ", ".join(f"{k} ({'yes' if v else 'no'})" for k, v in crit.items())
        lines.extend([f"- **{name.replace('_', ' ')}** — {record['status']}" + (f". {detail}." if detail else "."), ""])
    lines.extend([
        "Value length can covary with tool identity and argument difficulty. Residualization removes the fitted "
        "linear relationship with its specified covariates. Tool stratification changes allocation by tool "
        "frequency. A +1 or −1 pp point-estimate trigger describes magnitude; evidence that a population effect "
        "exceeds the margin requires an appropriate interval.", "", "## Measured single-GPU time", "",
        "| Endpoint | Train GPU h | Dev GPU h | Test GPU h | Shared proxy for a standalone run, GPU h | Standalone total, GPU h |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, m in result["models"].items():
        t = m["observed_task_seconds"]
        lines.append(f"| {label} | {t['train']/3600:.3f} | {t['dev']/3600:.3f} | {t['test']/3600:.3f} | "
                     f"{m['standalone_proxy_train_and_scoring_gpu_hours']:.3f} | {m['standalone_train_dev_test_plus_proxy_gpu_hours']:.3f} |")
    cost = result["cost"]
    lines.extend([
        "", f"Completed new train/dev/test tasks used {cost['completed_new_cells_gpu_hours']:.3f} GPU h in total. "
        f"The existing shared proxy training and scoring took {cost['shared_historical_proxy_and_scoring_gpu_hours']:.3f} "
        "GPU h in the original experiment. E2 requires B's per-bin quotas, so its standalone total includes that "
        "proxy dependency. A and E4 can select directly from the pool. The original arms evaluated multiple "
        "development checkpoints; the new arms evaluated one locked endpoint. The JSON reports target training "
        "time separately. CPU selection, failed attempts, idle time and the provider invoice are outside these figures.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=Path("local/course-run"))
    parser.add_argument("--followup", type=Path, default=Path("local/followup/phase1"))
    parser.add_argument("--out", type=Path, help="Aggregate output directory (default: FOLLOWUP/summary).")
    args = parser.parse_args()
    output = args.out or args.followup / "summary"
    result = summarize(args.primary, args.followup)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        temp = Path(temp)
        (temp / "followup.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (temp / "followup.md").write_text(markdown(result))
        for name in ("followup.json", "followup.md"):
            (temp / name).replace(output / name)
    print(f"Verified {len(result['completed_new_cells'])}/11 new endpoints across seven contrasts. Wrote {output}.")


if __name__ == "__main__":
    main()
