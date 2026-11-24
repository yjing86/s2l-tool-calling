"""Summarize the fixed 2x2 selection-seed × target-seed diagnostic on development only."""
import argparse
from collections import Counter
import json
from pathlib import Path

from summarize_course import cluster_interval, digest, indexed, paired, read_json, read_model, read_rows


# Cell order is (selection, target): (42,42), (42,43), (43,42), (43,43).
CELLS = ((42, 42), (42, 43), (43, 42), (43, 43))
CONTRASTS = [
    ("grid_mean", "Mean C−B across all four cells", (.25, .25, .25, .25)),
    ("selection_at_target42", "Selection seed 43 minus 42 at training seed 42", (-1, 0, 1, 0)),
    ("selection_at_target43", "Selection seed 43 minus 42 at training seed 43", (0, -1, 0, 1)),
    ("target_at_selection42", "Training seed 43 minus 42 at selection seed 42", (-1, 1, 0, 0)),
    ("target_at_selection43", "Training seed 43 minus 42 at selection seed 43", (0, 0, -1, 1)),
    ("selection_marginal", "Selection seed 43 minus 42, averaged over training seeds", (-.5, -.5, .5, .5)),
    ("target_marginal", "Training seed 43 minus 42, averaged over selection seeds", (-.5, .5, -.5, .5)),
    ("interaction", "Change with selection seed at training seed 43 minus the change at training seed 42", (1, -1, -1, 1)),
]


def seed_contrasts(deltas, groups, boot_seed):
    """Compute the contrast per request first, then jointly resample entire request groups."""
    sizes = Counter(groups.values())
    effects = {}
    for key, description, coeff in CONTRASTS:
        totals = Counter()
        for i, g in groups.items():
            totals[g] += sum(w * deltas[cell][i] for w, cell in zip(coeff, CELLS))
        effects[key] = {"description": description, "estimate_pp": 100 * sum(totals.values()) / len(groups),
                        "group_paired_ci95_pp": cluster_interval({g: (totals[g], n) for g, n in sizes.items()}, boot_seed)}
    return effects


def summarize(primary, crossed, dev, boot_seed):
    if dev.name != "dev.jsonl":
        raise ValueError("--dev must point to the prepared dev.jsonl")
    dev_hash = digest(dev)
    if read_json(dev.parent / "manifest.json")["sha256"]["dev.jsonl"] != dev_hash:
        raise ValueError("Development file differs from the prepared split manifest")
    rows = indexed(({"id": r["id"], "group_id": r["group_id"]} for r in read_rows(dev)), "id", dev)
    if len(rows) != 400 or any(r["group_id"] is None or r["group_id"] == "" for r in rows.values()):
        raise ValueError("Expected 400 unique frozen development IDs with nonempty group IDs")
    groups = {i: str(r["group_id"]) for i, r in rows.items()}
    primary_lock = read_json(primary / "locked-models.json")["main"]
    cross_map = read_json(crossed / "crossed-models.json")
    checksums = {"dev_split": dev_hash, "primary_lock": digest(primary / "locked-models.json"),
                 "crossed_lock": digest(crossed / "crossed-models.json")}
    stats, deltas, protocol, training_identity = {}, {}, None, None
    for sel, target in CELLS:
        subset = primary / "selection" / str(sel)
        by_arm, accuracies = {}, {}
        for arm in ("B", "C"):
            diagonal = sel == target
            run_id = f"s{sel}-{arm}" if diagonal else f"{arm}-selection{sel}-target{target}"
            root = primary if diagonal else crossed
            train = root / "training" / run_id
            done = read_json(train / "completed.json")
            meta = read_json(train / "run.json")
            step = done["final_step"]
            name = f"{run_id}-step{step}"
            key = run_id if diagonal else f"select{sel}-train{target}-{arm}"
            actual = primary_lock[key] if diagonal else cross_map[key]
            if name != actual:
                raise ValueError(f"{key}: checkpoint differs from predeclared equal-token endpoint")
            if meta["stage"] != "target" or meta["seed"] != target or meta["data_sha256"] != digest(subset / f"{arm}.jsonl"):
                raise ValueError(f"{key}: training data or target seed mismatch")
            runtime = read_json(train / "runtime.json")
            goal = runtime["token_goal"]
            end_steps = list(read_rows(train / "steps.jsonl"))[-2:]
            previous = end_steps[-2]["input_tokens"] if len(end_steps) > 1 else 0
            if (not end_steps or end_steps[-1]["step"] != step
                    or end_steps[-1]["input_tokens"] != done["input_tokens"]
                    or not previous < goal <= done["input_tokens"]):
                raise ValueError(f"{key}: expected the first optimizer boundary crossing the token goal")
            identity = (meta["model_config_sha256"], meta["pool_sha256"], meta["batch"], goal)
            if training_identity is not None and training_identity != identity:
                raise ValueError(f"{key}: comparison changed training pool, model config or effective batch")
            training_identity = identity
            eval_dir = root / "evaluation" / "dev" / name
            local = {}
            pred, perf, cfg = read_model(eval_dir, groups, dev_hash, local)
            if done["adapter_sha256"][str(step)] != perf["adapter_sha256"]["adapter_model.safetensors"]:
                raise ValueError(f"{key}: evaluated adapter differs from training endpoint")
            cfg_raw = read_json(eval_dir / "config.json")
            cfg.update({k: cfg_raw[k] for k in ("seed", "language_model_only", "max_lora_rank")})
            cfg.update({k: cfg_raw["args"][k] for k in ("max_model_len", "batch_size", "enforce_eager")})
            if protocol is not None and cfg != protocol:
                raise ValueError(f"{key}: generation or scoring protocol changed")
            protocol = cfg
            checksums[key] = {"training_record": digest(train / "run.json"), "completion": digest(train / "completed.json"),
                              "selected_training_rows": meta["data_sha256"],
                              "evaluation": {Path(k).name: v for k, v in local.items()}}
            by_arm[arm] = pred
            accuracies[arm] = {"correct": perf["correct"], "accuracy_pct": perf["accuracy_pct"], "step": step,
                               "input_tokens": done["input_tokens"], "answer_tokens": done["answer_tokens"], "token_goal": goal}
        deltas[sel, target] = {i: int(by_arm["C"][i]) - int(by_arm["B"][i]) for i in groups}
        stats[f"selection{sel}-target{target}"] = {"arms": accuracies,
            "C_minus_B": paired(by_arm["C"], by_arm["B"], groups, boot_seed)}
    return {"split": "dev", "requests": len(groups), "request_groups": len(set(groups.values())), "cells": stats,
            "contrasts": seed_contrasts(deltas, groups, boot_seed),
            "bootstrap": {"seed": boot_seed, "draws": 2000, "unit": "near-duplicate request group",
                          "intervals": "pointwise paired percentile 95%; common group draws for all cells"},
            "scope": "Exploratory, conditional on the fixed two seeds per axis and this development split; no population variance decomposition.",
            "provenance": checksums, "evaluation_protocol": protocol}


def markdown(report):
    out = ["# Crossed selection and training seeds", "",
           f"All four cells use the same development split of {report['requests']} requests in "
           f"{report['request_groups']} near-duplicate groups. B selects by whole-answer loss trajectories. "
           "C selects by separate field trajectories. Differences are in percentage points (pp).",
           "", "| Selection seed | Training seed | B correct | C correct | Only C correct | Only B correct | C−B (pp) | 95% interval (pp) |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, t in CELLS:
        cell = report["cells"][f"selection{s}-target{t}"]
        p = cell["C_minus_B"]
        lo, hi = p["paired_cluster_ci95_pp"]
        out.append(f"| {s} | {t} | {cell['arms']['B']['correct']} | {cell['arms']['C']['correct']} | "
                   f"{p['only_left_correct']} | {p['only_right_correct']} | {p['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    out += ["", "| Contrast in C−B | Estimate (pp) | 95% interval (pp) |", "|---|---:|---:|"]
    # Read the stable contrast key so older reports render in the current language.
    labels = {key: label for key, label, _ in CONTRASTS}
    for key, v in report["contrasts"].items():
        lo, hi = v["group_paired_ci95_pp"]
        out.append(f"| {labels.get(key, v['description'])} | {v['estimate_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    out += ["", "Each of the 2,000 paired bootstrap draws keeps each sampled request group together across all four "
            "cells and both methods. Each interval applies to one row. The interaction measures how much the change "
            "in C−B from switching the selection seed depends on the training seed. This exploratory diagnostic covers "
            "seeds 42 and 43 on one fixed development split. It cannot estimate population variance shares. "
            "The held-out test provides the separate assessment of generalization."]
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--crossed", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--bootstrap-seed", type=int, default=31415)
    args = p.parse_args()
    report = summarize(args.primary, args.crossed, args.dev, args.bootstrap_seed)
    for name, text in (("summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                       ("summary.md", markdown(report))):
        dest = args.crossed / name
        temp = dest.with_suffix(".tmp")
        temp.write_text(text)
        temp.replace(dest)
    print(f"Wrote four matched seed cells and eight contrasts to {args.crossed}")


if __name__ == "__main__":
    main()
