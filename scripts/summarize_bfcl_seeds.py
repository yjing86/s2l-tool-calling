"""Estimate external BFCL differences conditional on the matched, locked training seeds."""
import argparse
import json
from pathlib import Path
import random
import re
from statistics import mean

from bfcl_subset import CATEGORIES, REVISION, SOURCE_HASHES
from summarize_course import digest, indexed, quantile, read_json, read_rows


def verify_receipt(course, task, files):
    receipt = read_json(course / "receipts" / f"{task}.json")
    if receipt["task"] != task:
        raise ValueError(f"Wrong completion receipt for {task}")
    for path in files:
        suffix = "/" + path.relative_to(course).as_posix()
        hashes = [value for k, value in receipt["outputs"].items() if k.endswith(suffix)]
        if hashes != [digest(path)]:
            raise ValueError(f"Output hash mismatch for {task}: {path.name}")
    return receipt["script_sha256"]


def verified_scores(course, name, adapter, manifest, items, manifest_hash, scorer_hash):
    if not re.fullmatch(r"s\d+-[ABC]-step\d+", name):
        raise ValueError(f"Unrecognized checkpoint name {name}")
    folder = course / "evaluation" / "bfcl" / name
    scores = folder / "bfcl-scores.jsonl"
    report = folder / "bfcl-summary.json"
    predictions = folder / "predictions.jsonl"
    scorer = verify_receipt(course, f"score-bfcl-{name}", [scores, report])
    generator = verify_receipt(course, f"eval-bfcl-{name}", [folder / "metrics.json", predictions])
    config = read_json(folder / "config.json")
    if (scorer != scorer_hash or generator != config["script_sha256"]
            or config["source_sha256"] != manifest["prompts_sha256"]
            or [config["args"]["adapter"], config["adapter_sha256"]["adapter_model.safetensors"]] != adapter):
        raise ValueError(f"Generator, scorer, source or frozen adapter mismatch for {name}")
    summary = read_json(report)
    if summary["manifest_sha256"] != manifest_hash or summary["predictions_sha256"] != digest(predictions):
        raise ValueError(f"Scoring input hash mismatch for {name}")
    by_id = indexed(read_rows(scores), "id", scores)
    if by_id.keys() != items.keys():
        raise ValueError(f"Scores and frozen benchmark IDs differ for {name}")
    by_category = {category: {} for category in CATEGORIES}
    for ident, item in items.items():
        row = by_id[ident]
        if type(row["valid"]) is not bool or any(row[k] != v for k, v in item.items()):
            raise ValueError(f"Invalid official verdict or changed question metadata for {name}")
        by_category[item["category"]][ident] = int(row["valid"])
    for category, values in by_category.items():
        official = summary[f"{category}/all"]
        if len(values) != official["n"] or sum(values.values()) != official["correct"]:
            raise ValueError(f"{category} scores disagree with official category counts for {name}")
    return by_category


def contrast(models, seeds, categories, comparator):
    # Each value summarizes one question over every matched run before resampling.
    pooled = {}
    by_seed = {c: {} for c in (*CATEGORIES, "balanced")}
    for c in CATEGORIES:
        ids = categories[c]
        delta = {s: [models[s, "C"][c][i] - models[s, comparator][c][i] for i in ids] for s in seeds}
        by_seed[c] = {s: 100 * sum(ds) / len(ids) for s, ds in delta.items()}
        pooled[c] = [mean(ds) for ds in zip(*delta.values())]
    by_seed["balanced"] = {s: mean(by_seed[c][s] for c in CATEGORIES) for s in seeds}
    rng = random.Random(31415)
    draws = {c: [] for c in by_seed}
    for _ in range(10_000):
        for c, values in pooled.items():
            draws[c].append(100 * sum(rng.choices(values, k=len(values))) / len(values))
        draws["balanced"].append(mean(draws[c][-1] for c in CATEGORIES))
    result = {}
    for c, values in draws.items():
        values.sort()
        result[c] = {"unique_items": sum(map(len, categories.values())) if c == "balanced" else len(categories[c]),
                     "delta_pp_by_seed": by_seed[c], "mean_delta_pp": mean(by_seed[c].values()),
                     "item_bootstrap_ci95_pp": [quantile(values, .025), quantile(values, .975)]}
    return result


def summarize(course):
    locks = course / "locked-models.json"
    manifest_path = course / "frozen/bfcl/manifest.json"
    lock = read_json(locks)["main"]
    paths = read_json(course / "model-paths.json")
    manifest = read_json(manifest_path)
    if manifest["revision"] != REVISION or manifest["source_sha256"] != SOURCE_HASHES:
        raise ValueError("The frozen manifest differs from the pinned BFCL revision or source hashes")
    items = indexed(manifest["items"], "id", manifest_path)
    categories = {c: sorted(i for i, row in items.items() if row["category"] == c) for c in CATEGORIES}
    if not all(categories.values()) or sum(map(len, categories.values())) != len(items):
        raise ValueError("Expected both frozen single-call categories")
    seeds = sorted({int(m[1]) for label in lock if (m := re.fullmatch(r"s(\d+)-[ABC]", label))})
    if not seeds or any(f"s{s}-{g}" not in lock for s in seeds for g in "ABC"):
        raise ValueError("Expected a nonempty set of complete A/B/C seed triples in the lock")
    manifest_hash = digest(manifest_path)
    scorer_hash = digest(Path(__file__).with_name("bfcl_subset.py"))
    models = {}
    for seed in seeds:
        for arm in "ABC":
            name = lock[f"s{seed}-{arm}"]
            models[seed, arm] = verified_scores(course, name, paths[name], manifest, items, manifest_hash, scorer_hash)
    return {"scope": "Exploratory frozen BFCL subset, conditional on the matched training seeds.",
            "frozen_seeds": seeds, "bfcl_revision": REVISION, "manifest_sha256": manifest_hash,
            "lock_sha256": digest(locks), "draws": 10000, "rng_seed": 31415,
            "interval": "Pointwise percentile 95%. Resample matched items within each category, holding all seeds together. Balanced assigns equal weight to each category.",
            "comparisons": {f"C-{arm}": contrast(models, seeds, categories, arm) for arm in "BA"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.course)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temp.replace(args.out)
