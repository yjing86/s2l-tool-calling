"""Summarize locked internal test runs. The aggregate archive contains no prompts or predictions."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import zipfile


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def indexed(rows, key, path):
    result = {}
    for row in rows:
        ident = str(row[key])
        if ident in result:
            raise ValueError(f"{path}: duplicate {key} {ident}")
        result[ident] = row
    return result


def quantile(xs, p):
    """Linearly interpolated empirical quantile (type 7) of sorted data."""
    index = (len(xs) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return xs[lo] + (index - lo) * (xs[hi] - xs[lo])


def cluster_interval(group_totals, seed):
    """Bootstrap the request-weighted ratio of summed deltas to exposure."""
    rng = random.Random(seed)
    by_group = [group_totals[k] for k in sorted(group_totals)]
    # Each draw selects whole groups; divide by the number of requests in that
    # draw to preserve the pre-registered request-level estimand.
    draws = []
    for _ in range(2000):
        num = den = 0
        for d, n in rng.choices(by_group, k=len(by_group)):
            num += d
            den += n
        draws.append(100 * num / den)
    draws.sort()
    return [quantile(draws, .025), quantile(draws, .975)]


def paired(left, right, groups, boot_seed):
    deltas = Counter()
    discordance = Counter()
    for key, group in groups.items():
        a, b = left[key], right[key]
        deltas[group] += int(a) - int(b)
        discordance[a, b] += 1
    sizes = Counter(groups.values())
    group_totals = {g: (deltas[g], size) for g, size in sizes.items()}
    return {"requests": len(groups), "groups": len(sizes),
            "only_left_correct": discordance[True, False], "only_right_correct": discordance[False, True],
            "both_correct": discordance[True, True], "both_wrong": discordance[False, False],
            "difference_pp": 100 * sum(deltas.values()) / len(groups),
            "paired_cluster_ci95_pp": cluster_interval(group_totals, boot_seed)}


def fixed_seed_summary(seed_values, differences, groups, boot_seed):
    """Use one joint draw of test groups across all paired, fixed seed results."""
    seeds = sorted(seed_values)
    sizes = Counter(groups.values())
    sums = Counter()
    for left, right in differences:
        for i, group in groups.items():
            sums[group] += int(left[i]) - int(right[i])
    group_totals = {g: (sums[g], size * len(seeds)) for g, size in sizes.items()}
    values = [seed_values[s] for s in seeds]
    return {"seeds": len(seeds), "seed_differences_pp": seed_values,
            "mean": statistics.mean(values), "min": min(values), "max": max(values),
            "sample_sd_pp": statistics.stdev(values) if len(seeds) > 1 else None,
            "test_requests": len(groups), "near_duplicate_groups": len(sizes),
            "fixed_seed_mean_paired_cluster_ci95_pp": cluster_interval(group_totals, boot_seed)}


def read_model(folder, groups, source_hash, checksums):
    config = read_json(folder / "config.json")
    metrics = read_json(folder / "metrics.json")
    if config["source_sha256"] != source_hash:
        raise ValueError(f"{folder}: test split hash mismatch")
    correct = {}
    errors = Counter()
    for row in read_rows(folder / "predictions.jsonl"):
        key = str(row["id"])
        if key in correct or key not in groups or str(row["group_id"]) != groups[key]:
            raise ValueError(f"{folder}: duplicate ID or incorrect test membership")
        if type(row["correct"]) is not bool or ((row["reason"] == "correct") != row["correct"]):
            raise ValueError(f"{folder}: invalid correctness value")
        correct[key] = row["correct"]
        errors[row["reason"]] += 1
    count = sum(correct.values())
    if set(correct) != set(groups) or metrics["count"] != len(groups) or metrics["correct"] != count:
        raise ValueError(f"{folder}: incomplete or inconsistent predictions/metrics")
    for filename in ("config.json", "predictions.jsonl", "metrics.json"):
        checksums[f"evaluation/test/{folder.name}/{filename}"] = digest(folder / filename)
    protocol = {k: config.get(k) for k in ["model_revision", "prompts_sha256", "tokenizer_sha256",
                                             "template_sha256", "stop_token_ids", "greedy", "dtype", "script_sha256"]}
    protocol["max_tokens"] = config["args"]["max_tokens"]
    return correct, {"correct": count, "requests": len(groups), "accuracy_pct": 100 * count / len(groups),
                     "first_failure_counts": dict(errors), "adapter_sha256": config["adapter_sha256"],
                     "eval_seconds": metrics["wall_seconds"]}, protocol


def summarize_ledger(path):
    if not path.exists():
        return {"recorded": False}
    events = list(read_rows(path))
    counts = Counter(row["event"] for row in events)
    details = []
    for row in events:
        # Commands and paths may contain personal account details. Include task
        # metadata only; preserve the digest of the full local ledger for audit.
        details.append({k: row[k] for k in ("task", "event", "unix", "elapsed_s") if k in row})
    return {"recorded": True, "sha256": digest(path), "counts": dict(counts), "events": details,
            "completed_task_wall_seconds": sum(s.get("elapsed_s", 0) for s in details if s["event"] == "done"),
            "failed_task_wall_seconds_recorded": sum(s.get("elapsed_s", 0) for s in details if s["event"] == "failed")}


def summary(course, test, boot_seed):
    lock_path = course / "locked-models.json"
    lock = read_json(lock_path)
    phases = {k: lock[k] for k in ("main", "equal_time") if k in lock} if "main" in lock else {"main": lock}
    rows = indexed(({"id": r["id"], "group_id": r["group_id"]} for r in read_rows(test)), "id", test)
    if not rows or any(row["group_id"] is None or row["group_id"] == "" for row in rows.values()):
        raise ValueError("Frozen test split must contain nonempty near-duplicate groups")
    groups = {i: str(r["group_id"]) for i, r in rows.items()}
    source_hash = digest(test)
    checksums = {"frozen_test_split": source_hash, "locked-models.json": digest(lock_path),
                 "summarize_course.py": digest(Path(__file__))}
    stats, predictions, phase_results = {}, {}, {}
    common_protocol = None
    for phase, labels in phases.items():
        by_seed = defaultdict(dict)
        if not isinstance(labels, dict):
            raise ValueError(f"{phase}: expected {{model label: evaluation folder name}}")
        for label, name in labels.items():
            if not isinstance(name, str) or not re.fullmatch(r"[\w.-]+", name) or name in (".", ".."):
                raise ValueError(f"{phase}: invalid evaluation folder name")
            if name not in predictions:
                correct, stats[name], protocol = read_model(course / "evaluation/test" / name, groups, source_hash, checksums)
                predictions[name] = correct
                if common_protocol is not None and common_protocol != protocol:
                    raise ValueError(f"{name}: generation protocol changed between internal exams")
                common_protocol = protocol
            if label == "base":
                continue
            match = re.fullmatch(r"s(\d+)-(A|B|C|Bprime)", label)
            if not match:
                raise ValueError(f"{phase}: unsupported model label {label}")
            by_seed[int(match[1])][match[2]] = name
        comparisons, aggregated, paired_predictions = {}, defaultdict(dict), defaultdict(list)
        for seed, methods in sorted(by_seed.items()):
            comparisons[str(seed)] = {}
            for l, r in (("C", "B"), ("B", "A"), ("C", "A"), ("Bprime", "B"), ("C", "Bprime")):
                if l not in methods or r not in methods:
                    continue
                result = paired(predictions[methods[l]], predictions[methods[r]], groups, boot_seed)
                comparisons[str(seed)][f"{l}-{r}"] = result
                aggregated[f"{l}-{r}"][seed] = result["difference_pp"]
                paired_predictions[f"{l}-{r}"].append((predictions[methods[l]], predictions[methods[r]]))
        phase_results[phase] = {"locked": labels, "paired_by_seed": comparisons,
                                "across_seed_differences_pp": {
                                    pair: fixed_seed_summary(vals, paired_predictions[pair], groups, boot_seed)
                                    for pair, vals in aggregated.items()}}
    ledger = summarize_ledger(course / "ledger.jsonl")
    if ledger["recorded"]:
        checksums["ledger.jsonl"] = ledger["sha256"]
    for receipt in sorted((course / "receipts").glob("*.json")):
        checksums[f"receipts/{receipt.name}"] = digest(receipt)
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(), "test_requests": len(groups),
              "near_duplicate_groups": len(set(groups.values())),
              "interval": {"unit": "near-duplicate group", "draws": 2000, "seed": boot_seed, "method": "percentile"},
              "models": stats, "phases": phase_results, "internal_generation_protocol": common_protocol}
    return result, checksums, ledger


def markdown(summary, ledger):
    lines = ["# Course experiment results", "",
             f"Internal test: {summary['test_requests']} requests across {summary['near_duplicate_groups']} groups.", "",
             "| Evaluated checkpoint | Correct | Requests | Accuracy |", "|---|---:|---:|---:|"]
    for model, m in summary["models"].items():
        lines.append(f"| {model} | {m['correct']} | {m['requests']} | {m['accuracy_pct']:.2f}% |")
    for phase, result in summary["phases"].items():
        lines += ["", f"## {phase}", "", "| Seed | Difference | Left only | Right only | Difference (pp) | 95% CI (pp) |",
                  "|---:|---|---:|---:|---:|---:|"]
        for seed, pairs in result["paired_by_seed"].items():
            for pair, v in pairs.items():
                ci = v["paired_cluster_ci95_pp"]
                lines.append(f"| {seed} | {pair} | {v['only_left_correct']} | {v['only_right_correct']} | "
                             f"{v['difference_pp']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
        lines += ["", "| Contrast | Matched seeds | Mean (pp) | 95% CI of fixed-seed mean (pp) | Seed range (pp) | Seed SD (pp) |", "|---|---:|---:|---:|---:|---:|"]
        for pair, v in result["across_seed_differences_pp"].items():
            lo, hi = v["fixed_seed_mean_paired_cluster_ci95_pp"]
            sd = "—" if v["sample_sd_pp"] is None else f"{v['sample_sd_pp']:.2f}"
            lines.append(f"| {pair} | {v['seeds']} | {v['mean']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                         f"[{v['min']:+.2f}, {v['max']:+.2f}] | {sd} |")
    lines += ["", "The interval for each mean resamples whole held-out request groups together for every "
              "method and every matched seed. It measures question-set uncertainty conditional on these trained runs. "
              "Seed-specific differences, their range and sample standard deviation describe variation across observed "
              "seeds. Additional seed runs evaluate the same requests, so the number of distinct test requests stays fixed."]
    if ledger["recorded"]:
        lines += ["", f"Completed task wall time recorded in the local ledger: "
                  f"{ledger['completed_task_wall_seconds']/3600:.3f} hours. Billing may include idle time and failed attempts."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--course", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--bootstrap-seed", type=int, default=31415)
    args = p.parse_args()
    result, checksums, ledger = summary(args.course, args.test, args.bootstrap_seed)
    out = args.course / "summary"
    out.mkdir(parents=True, exist_ok=True)
    files = {"internal.json": result, "checksums.json": checksums, "ledger-summary.json": ledger}
    texts = {k: json.dumps(v, indent=2, allow_nan=False) + "\n" for k, v in files.items()}
    texts["summary.md"] = markdown(result, ledger)
    for name, content in texts.items():
        temp = (out / name).with_suffix(".tmp")
        temp.write_text(content)
        temp.replace(out / name)
    archive = out / "aggregate-audit.zip"
    temp = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in texts.items():
            z.writestr(name, content)
    temp.replace(archive)
    print(f"Verified {len(result['models'])} model exams and wrote {archive}")


if __name__ == "__main__":
    main()
