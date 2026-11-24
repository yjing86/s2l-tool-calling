"""Construct seeded A/B/C/Bprime subsets from paired, complete proxy trajectories."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from score_trajectories import file_hash


def describe(values):
    x = np.asarray(values, dtype=float)
    valid = x[np.isfinite(x)]
    return {"count": len(valid), "missing": len(x) - len(valid),
            "mean": float(valid.mean()) if len(valid) else None,
            "variance": float(valid.var()) if len(valid) else None,
            "p10_p50_p90": np.quantile(valid, [.1, .5, .9]).tolist() if len(valid) else [],
            "fraction_below_0.01": float((valid < .01).mean()) if len(valid) else None}


def cluster_draw(features, ids, num, seed, cluster_count, cluster_seed):
    if not len(ids):
        return [], {"columns": [], "clusters": []}
    values = features[ids]
    avg, sd = values.mean(axis=0), values.std(axis=0)
    x = (values - avg) / np.where(sd == 0, 1., sd)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite K-means feature")
    k = min(cluster_count, len(np.unique(x, axis=0)))
    km = KMeans(n_clusters=k, n_init=10, random_state=cluster_seed, algorithm="lloyd").fit(x)
    # Canonical cluster order gives a deterministic tie-break for the remainder.
    groups = [np.asarray(sorted(ids[km.labels_ == idx]), dtype=int) for idx in range(k)]
    groups = sorted((g for g in groups if len(g)), key=lambda g: int(g[0]))
    rng = np.random.default_rng(seed)
    clusters = [rng.permutation(g) for g in groups]
    quota = np.zeros(len(groups), dtype=int)
    remaining = num
    while remaining:
        for c, group in enumerate(clusters):
            if remaining and quota[c] < len(group):
                remaining -= 1
                quota[c] += 1
    sampled = sorted(int(i) for c, group in enumerate(clusters) for i in group[:quota[c]])
    return sampled, {"normalization_mean": avg.tolist(), "normalization_sd": sd.tolist(),
                     "inertia": km.inertia_, "iterations": int(km.n_iter_),
                     "clusters": [{"size": len(g), "selected": int(n), "indices": g.tolist()}
                                  for g, n in zip(groups, quota)]}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def select(args, output):
    data = [json.loads(line) for line in args.train.read_text().splitlines() if line.strip()]
    ids = [str(r["id"]) for r in data]
    if len(set(ids)) != len(ids) or not 0 < args.size <= len(ids):
        raise ValueError("Pool must contain unique IDs and at least the requested sample size")
    train_hash = file_hash(args.train)
    meta = json.loads((args.scores / "score-config.json").read_text())
    if train_hash != meta["train_sha256"]:
        raise ValueError("Scored and selected training sets differ")
    steps = args.steps
    if len(steps) != 5 or steps[0] != 0 or steps != sorted(set(steps)):
        raise ValueError("Provide exactly five unique, ascending steps, starting at zero")
    counts, feature_columns, receipts = None, {k: [] for k in ("all", "function", "value", "field_mean")}, {}
    for step in steps:
        p = args.scores / f"step-{step}.jsonl"
        receipt = json.loads(p.with_suffix(".complete.json").read_text())
        if receipt["step"] != step or receipt["rows"] != len(ids) or file_hash(p) != receipt["sha256"]:
            raise ValueError(f"Invalid completion receipt, step {step}")
        receipts[str(step)] = receipt
        records = [json.loads(line) for line in p.read_text().splitlines()]
        by_id = {str(row["id"]): row for row in records}
        if len(records) != len(ids) or set(by_id) != set(ids):
            raise ValueError(f"Different candidate pool at step {step}")
        ordered = [by_id[key] for key in ids]
        step_counts = np.array([[r[k] for k in ("all_tokens", "function_tokens", "value_tokens", "other_tokens", "input_tokens")] for r in ordered])
        if counts is None:
            counts = step_counts
        if (not np.array_equal(step_counts, counts) or (counts < 0).any() or (counts[:,1] < 1).any()
                or not np.array_equal(counts[:, 1:4].sum(axis=1), counts[:, 0])):
            raise ValueError(f"Invalid or changed token masks at step {step}")
        totals = np.asarray([[r[k + "_sum"] for k in ("all", "function", "value", "other")] for r in ordered])
        if not np.isfinite(totals).all() or (totals < 0).any() or not np.allclose(totals[:, 0], totals[:, 1:].sum(axis=1)):
            raise ValueError(f"Inconsistent loss sums at step {step}")
        for i, field in enumerate(("all", "function", "value")):
            feature_columns[field].append(totals[:, i] / np.where(counts[:, i] > 0, counts[:, i], np.nan))
        feature_columns["field_mean"].append(totals[:, 1:3].sum(axis=1) / counts[:, 1:3].sum(axis=1))
    features = {name: np.column_stack(cols) for name, cols in feature_columns.items()}
    report = {"seed": args.seed, "cluster_seed": args.cluster_seed, "size": args.size, "steps": steps,
              "numpy": np.__version__, "scikit_learn": sklearn.__version__, "train_sha256": train_hash,
              "receipts": receipts, "score_config_sha256": file_hash(args.scores / "score-config.json"),
              "script_sha256": file_hash(__file__),
              "feature_statistics": {name: [describe(col) for col in cols] for name, cols in feature_columns.items()},
              "subsets": {}, "clustering": {}}
    nonempty = np.flatnonzero(counts[:, 2] != 0)
    empty = np.flatnonzero(counts[:, 2] == 0)
    empty_quota = math.floor(args.size * len(empty) / len(data) + .5)
    # Keep k-means initialization fixed across seed runs; vary only drawing within the clusters.
    def draw(x, indices, count):
        return cluster_draw(x, indices, count, args.seed, args.clusters, args.cluster_seed)

    shared, report["clustering"]["shared_empty"] = draw(features["function"], empty, empty_quota)
    selected = {"A": sorted(np.random.default_rng(args.seed).choice(len(data), args.size, replace=False).tolist())}
    for group, x in (("B", features["all"]), ("C", np.column_stack((features["function"], features["value"]))),
                     ("Bprime", features["field_mean"])):
        subset, report["clustering"][group] = draw(x, nonempty, args.size - empty_quota)
        selected[group] = sorted(shared + subset)
    for group, subset in selected.items():
        if len(subset) != args.size or len(set(subset)) != args.size:
            raise AssertionError("Invalid draw")
        path = output / f"{group}.jsonl"
        with path.open("w") as stream:
            for i in subset:
                print(json.dumps(data[i], ensure_ascii=False), file=stream)
        tools = [data[i]["messages"][-1]["tool_calls"][0]["function"]["name"] for i in subset]
        report["subsets"][group] = {"ids": [ids[i] for i in subset], "sha256": file_hash(path),
                                    "tool_counts": dict(Counter(tools)),
                                    "answer_token_quantiles": np.quantile(counts[subset, 0], [0, .1, .5, .9, 1]).tolist(),
                                    "answer_tokens_total": int(counts[subset, 0].sum()),
                                    "input_plus_answer_tokens_total": int(counts[subset, 4].sum())}
    # Store row IDs rather than private array positions in the audit trail.
    for entry in report["clustering"].values():
        for cluster in entry["clusters"]:
            cluster["ids"] = [ids[i] for i in cluster.pop("indices")]
    report["overlap"] = {}
    for left, right in combinations(selected, 2):
        a, b = set(selected[left]), set(selected[right])
        report["overlap"][left + "-" + right] = {"intersection": len(a & b), "jaccard": len(a & b) / len(a | b)}
    write_json(output / "selection.json", report)
    print(json.dumps({"size": args.size, "empty_value_quota": empty_quota, "overlap": report["overlap"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("train", "scores", "out"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cluster-seed", type=int, default=42)
    p.add_argument("--steps", nargs="+", type=int, default=[0, 10, 30, 80, 250])
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--clusters", type=int, default=10)
    args = p.parse_args()
    if args.clusters < 1 or args.out.exists():
        p.error("--clusters must be positive and --out must be a new directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        output = Path(tmp) / "complete"
        output.mkdir()
        select(args, output)
        output.rename(args.out)


if __name__ == "__main__":
    main()
