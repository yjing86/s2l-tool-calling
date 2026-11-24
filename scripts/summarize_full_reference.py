"""Compare the locked full-pool reference with the locked seed-42 course arms."""
import argparse
import json
import math
from pathlib import Path
import re

from summarize_course import digest, indexed, paired, read_json, read_model, read_rows


PRIMARY_LABELS = ("base", "s42-A", "s42-B", "s42-C")
FULL_ROLES = ("final", "best_dev", "matched_tokens")


def model_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w.-]+", value) or value in (".", ".."):
        raise ValueError(f"Invalid locked checkpoint name {value!r}")
    return value


def costs(input_tokens, answer_tokens, training_seconds, dev_seconds, selection_seconds):
    for n in (input_tokens, answer_tokens):
        if type(n) is not int or n < 0:
            raise ValueError("Training token counts must be nonnegative integers")
    if answer_tokens > input_tokens:
        raise ValueError("Supervised token count exceeds the model's non-padding input token count")
    timers = {"training": training_seconds, "dev": dev_seconds, "selection": selection_seconds}
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < -1e-6 for v in timers.values()):
        raise ValueError("Invalid GPU timing")
    hours = {k: max(0, v) / 3600 for k, v in timers.items()}
    hours["total"] = sum(hours.values())
    return {"input_tokens": input_tokens, "answer_tokens": answer_tokens, "single_gpu_hours": hours}


def primary_costs(primary, label, name, ledger, hashes):
    if label == "base":
        return costs(0, 0, 0, 0, 0)
    completed_path = primary / "training" / label / "completed.json"
    train_receipt = primary / "receipts" / f"train-{label}.json"
    for path in (completed_path, train_receipt):
        hashes["primary/" + str(path.relative_to(primary))] = digest(path)
    completed = read_json(completed_path)
    if name != f"{label}-step{completed['final_step']}":
        raise ValueError(f"{label}: locked primary checkpoint differs from completed equal-token endpoint")
    route = ledger["routes"][label]
    training = read_json(train_receipt)["elapsed_s"]
    own = route["own_training_and_dev_seconds"]
    selection = route["initial_cost"] - own
    expected = ledger["common_proxy_and_scoring_seconds"] if label in ("s42-B", "s42-C") else 0
    if not math.isclose(selection, expected, abs_tol=1e-5):
        raise ValueError(f"{label}: inconsistent selection cost in route ledger")
    return costs(completed["input_tokens"], completed["answer_tokens"], training, own-training, selection)


def summarize(primary, secondary, test, boot_seed):
    # Both locks must exist before we read a single test example or evaluation result.
    primary_lock = primary / "locked-models.json"
    full_lock = secondary / "full-models.json"
    main = read_json(primary_lock)["main"]
    full = read_json(full_lock)
    locks = {
        "primary": {label: model_name(main[label]) for label in PRIMARY_LABELS},
        "full": {role: model_name(full[role]) for role in FULL_ROLES}
    }
    hashes = {"primary/locked-models.json": digest(primary_lock), "secondary/full-models.json": digest(full_lock)}
    source_hash = digest(test)
    hashes["frozen_test_split"] = source_hash
    rows = indexed(({"id": r["id"], "group_id": r["group_id"]} for r in read_rows(test)), "id", test)
    if not rows or any(r["group_id"] is None or r["group_id"] == "" for r in rows.values()):
        raise ValueError("Every frozen request needs its near-duplicate group")
    groups = {i: str(r["group_id"]) for i, r in rows.items()}

    primary_ledger = primary / "training-ledger.json"
    full_ledger = secondary / "full-costs.json"
    primary_cost = read_json(primary_ledger)
    full_cost = read_json(full_ledger)
    hashes["primary/training-ledger.json"] = digest(primary_ledger)
    hashes["secondary/full-costs.json"] = digest(full_ledger)
    observed = {}
    reports, predictions = {}, {}
    protocol = None
    for phase, labels in locks.items():
        for label, name in labels.items():
            owner = "primary" if phase == "primary" or name == "base" else "secondary"
            key = (owner, name)
            if key not in observed:
                folder = (primary if owner == "primary" else secondary) / "evaluation/test" / name
                local_hashes = {}
                values, stats, actual = read_model(folder, groups, source_hash, local_hashes)
                if protocol is not None and actual != protocol:
                    raise ValueError(f"{key}: prompt, model revision or evaluation protocol differs")
                protocol = actual
                observed[key] = (values, stats)
                hashes.update({owner + "/" + k: v for k, v in local_hashes.items()})
            values, stats = observed[key]
            predictions[phase, label] = values
            if phase == "primary":
                use = primary_costs(primary, label, name, primary_cost, hashes)
            elif name == "base":
                use = costs(0, 0, 0, 0, 0)
            else:
                raw = full_cost["checkpoints"][name]
                use = costs(raw["input_tokens"], raw["answer_tokens"], raw["training_seconds"],
                            raw["dev_seconds"], raw["selection_seconds"])
                if use["single_gpu_hours"]["selection"] != 0:
                    raise ValueError("Full-pool training incurs no proxy selection cost")
            reports[f"{phase}/{label}"] = {"checkpoint": name, **stats, **use}

    differences = {}
    cached_pairs = {}
    for role, left_name in locks["full"].items():
        differences[role] = {}
        for arm, right_name in locks["primary"].items():
            pair = (left_name, right_name)
            if pair not in cached_pairs:
                cached_pairs[pair] = paired(predictions["full", role], predictions["primary", arm], groups, boot_seed)
            differences[role][arm] = cached_pairs[pair]

    # Reaching an early checkpoint costs less than selecting it after the entire
    # scheduled sweep. Keep both amounts even if the chosen checkpoint is base.
    selection_process = costs(0, 0, full_cost["total_training_seconds"], full_cost["total_dev_seconds"], 0)
    return {"seed": 42, "near_duplicate_groups": len(set(groups.values())), "test_requests": len(groups),
            "locks": locks, "models": reports, "full_minus_primary": differences,
            "interval": {"unit": "near-duplicate group", "draws": 2000, "method": "percentile", "seed": boot_seed},
            "full_dev_selection_sweep_gpu_hours": selection_process["single_gpu_hours"],
            "generation_protocol": protocol, "input_sha256": hashes}


def markdown(result):
    n = result["test_requests"]
    lines = ["# Full-pool reference on the frozen internal test",
             "", f"Selection and training seed 42. All estimates use the same {n} held-out requests.",
             "The full run contributes three prespecified readouts of one training trajectory.",
             "", "| Locked endpoint | Checkpoint | Correct / total | Accuracy | Input tokens | Supervised tokens | Train GPU h | Dev GPU h | Select GPU h | Total GPU h |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in result["models"].items():
        t = row["single_gpu_hours"]
        lines.append(f"| {label} | {row['checkpoint']} | {row['correct']} / {n} | {row['accuracy_pct']:.2f}% | "
                     f"{row['input_tokens']:,} | {row['answer_tokens']:,} | {t['training']:.3f} | "
                     f"{t['dev']:.3f} | {t['selection']:.3f} | {t['total']:.3f} |")
    sweep = result["full_dev_selection_sweep_gpu_hours"]
    lines += ["", f"Selecting the full model by development score required the entire scheduled sweep "
              f"({sweep['training']:.3f} training GPU h and {sweep['dev']:.3f} development GPU h). "
              "The endpoint rows show the training and development costs to reach each checkpoint. "
              "Proxy training and scoring count in each B/C standalone route. "
              "Shared base evaluation, final exams, failed attempts and idle time are outside this table.",
              "", "| Full endpoint | Primary reference | Full only correct | Reference only correct | Full minus reference (pp) | Paired 95% CI (pp) |",
              "|---|---|---:|---:|---:|---:|"]
    for role, refs in result["full_minus_primary"].items():
        for name, row in refs.items():
            lo, hi = row["paired_cluster_ci95_pp"]
            lines.append(f"| {role} | {name} | {row['only_left_correct']} | {row['only_right_correct']} | "
                         f"{row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "Intervals use 2,000 paired resamples of whole near-duplicate groups. They reflect held-out "
              "request variation conditional on this seed. Identical locked checkpoints reuse the same predictions."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--secondary", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--bootstrap-seed", type=int, default=31415)
    args = p.parse_args()
    result = summarize(args.primary, args.secondary, args.test, args.bootstrap_seed)
    folder = args.secondary / "summary"
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("full-reference.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                       ("full-reference.md", markdown(result))):
        path = folder / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)
    print(f"Verified all seven locked readouts on {result['test_requests']} requests; wrote {folder}")


if __name__ == "__main__":
    main()
