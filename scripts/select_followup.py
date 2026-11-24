"""Select train-only controls for the exploratory Turnstile mechanism experiment.

Input is the primary run's frozen training pool, five proxy score files, and its
original selection receipts. The output is installed atomically in a new folder.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from select_experiment import cluster_draw, file_hash, write_json


STEPS = [0, 10, 30, 80, 250]
SIZE = 1000
CLUSTERS = 10
CLUSTER_SEED = 42
PREDICTORS = ["intercept", "log1p_value_tokens", "field_token_fraction", "log_answer_tokens"]
TOKEN_KEYS = ["all_tokens", "function_tokens", "value_tokens", "other_tokens", "input_tokens"]


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def training_inputs(primary):
    train = primary / "frozen/train.jsonl"
    data = rows(train)
    ids = [str(r["id"]) for r in data]
    if len(ids) != 4000 or len(set(ids)) != len(ids):
        raise ValueError("Expected 4,000 distinct training requests")
    tools = []
    for row in data:
        assistant = row["messages"][-1]
        if assistant["role"] != "assistant" or len(assistant["tool_calls"]) != 1:
            raise ValueError("Training requires single-call targets")
        tools.append(assistant["tool_calls"][0]["function"]["name"])
    config = primary / "scores/score-config.json"
    train_hash = file_hash(train)
    if read(config)["train_sha256"] != train_hash:
        raise ValueError("Proxy scores refer to a different training pool")
    fingerprints = {"train_sha256": train_hash, "score_config_sha256": file_hash(config)}
    counts, curves, receipts = None, [], {}
    for step in STEPS:
        score = primary / "scores" / f"step-{step}.jsonl"
        receipt = read(score.with_suffix(".complete.json"))
        if receipt != {"rows": len(ids), "step": step, "sha256": file_hash(score)}:
            raise ValueError(f"Invalid scoring receipt at step {step}")
        receipts[str(step)] = receipt
        points = rows(score)
        by_id = {str(r["id"]): r for r in points}
        if len(points) != len(ids) or set(by_id) != set(ids):
            raise ValueError(f"Scoring IDs differ at step {step}")
        ordered = [by_id[key] for key in ids]
        c = np.array([[r[key] for key in TOKEN_KEYS] for r in ordered])
        if (not np.issubdtype(c.dtype, np.integer) or (c < 0).any() or (c[:, 1] < 1).any()
                or not np.array_equal(c[:, 0], c[:, 1:4].sum(axis=1)) or (c[:, 4] < c[:, 0]).any()
                or (counts is not None and not np.array_equal(c, counts))):
            raise ValueError(f"Scored token masks changed or are invalid at step {step}")
        counts = c
        sums = np.array([[r[key + "_sum"] for key in ("all", "function", "value", "other")] for r in ordered])
        if not np.isfinite(sums).all() or (sums < 0).any() or not np.allclose(sums[:, 0], sums[:, 1:].sum(1)):
            raise ValueError(f"Invalid supervised loss sums at step {step}")
        curves.append(sums[:, 0] / counts[:, 0])
    return data, ids, tools, counts, np.column_stack(curves), receipts, fingerprints


def original_subsets(primary, seed, data, ids, receipts, fingerprints):
    directory = primary / "selection" / str(seed)
    receipt = read(primary / "receipts" / f"select-{seed}.json")
    original = read(directory / "selection.json")
    expected = {"seed": seed, "size": SIZE, "steps": STEPS, "cluster_seed": CLUSTER_SEED,
                "receipts": receipts, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                "script_sha256": file_hash(Path(__file__).with_name("select_experiment.py")), **fingerprints}
    if (receipt["task"] != f"select-{seed}" or receipt["script_sha256"] != expected["script_sha256"]
            or any(original.get(k) != value for k, value in expected.items())):
        raise ValueError(f"Primary selection provenance, including pinned library versions, differs for seed {seed}")
    paths = [directory / "selection.json", *(directory / f"{name}.jsonl" for name in ("A", "B", "C", "Bprime"))]
    original_hashes = {}
    for path in paths:
        values = [v for k, v in receipt["outputs"].items() if k.endswith(f"/selection/{seed}/{path.name}")]
        original_hashes[path.name] = file_hash(path)
        if values != [original_hashes[path.name]]:
            raise ValueError(f"Primary receipt mismatch for {path.name}")
    position = {key: i for i, key in enumerate(ids)}
    selections = {}
    for path in paths[1:]:
        group = path.stem
        records = rows(path)
        keys = [str(row["id"]) for row in records]
        info = original["subsets"][group]
        if (len(set(keys)) != SIZE or keys != info["ids"] or info["sha256"] != original_hashes[path.name]
                or any(key not in position or row != data[position[key]] for key, row in zip(keys, records))):
            raise ValueError(f"Invalid source subset {seed}/{group}")
        selections[group] = [position[key] for key in keys]
    return selections, original_hashes


def length_strata(values):
    """Integer inverse-empirical-CDF deciles on positive values; equal counts stay together."""
    positive = np.sort(values[values > 0])
    if not len(positive):
        return [], np.zeros(len(values), dtype=int)
    thresholds = {int(positive[(decile * len(positive) + 9) // 10 - 1]) for decile in range(1, 10)}
    cutoffs = sorted(t for t in thresholds if t < positive[-1])
    # Bin 0 contains only zeros. Positive bins use closed upper endpoints.
    return cutoffs, np.where(values == 0, 0, 1 + np.searchsorted(cutoffs, values, side="left"))


def match_length(bins, original_b, empty_ids, seed):
    rng = np.random.default_rng(seed)
    chosen = list(empty_ids)
    distribution = Counter(bins[original_b].tolist())
    for label in sorted(set(bins.tolist()) - {0}):
        pool = np.flatnonzero(bins == label)
        chosen.extend(rng.choice(pool, size=distribution[label], replace=False).tolist())
    if Counter(bins[chosen].tolist()) != distribution:
        raise AssertionError("E2 did not exactly match B within length bins")
    return sorted(chosen)


def fit_residuals(counts, losses):
    """Fit independent OLS at five checkpoints using all 4,000 training rows."""
    total, function, value = counts[:, 0], counts[:, 1], counts[:, 2]
    x = np.column_stack((np.ones(len(total)), np.log1p(value), (function + value) / total, np.log(total)))
    coef, _, rank, singular = np.linalg.lstsq(x, losses, rcond=None)
    residuals = losses - x @ coef
    sst = np.sum((losses - losses.mean(axis=0)) ** 2, axis=0)
    rss = np.sum(residuals ** 2, axis=0)
    design = {"population": "all training requests", "rows": len(x), "predictors": PREDICTORS,
              "rank": int(rank), "columns": x.shape[1], "singular_values": singular.tolist(),
              "condition_number": float(singular[0] / singular[-1]) if rank == x.shape[1] else None,
              "variance_inflation_factors": {}, "predictor_correlations": {}, "by_step": {}}
    for i in range(1, x.shape[1]):
        y = x[:, i]
        others = np.delete(x, i, axis=1)
        unexplained = np.linalg.norm(y - others @ np.linalg.lstsq(others, y, rcond=None)[0]) ** 2
        variation = np.linalg.norm(y - y.mean()) ** 2
        design["variance_inflation_factors"][PREDICTORS[i]] = (
            float(variation / unexplained) if variation > 0 and unexplained > np.finfo(float).eps * variation else None)
        for j in range(i + 1, x.shape[1]):
            correlation = np.corrcoef(y, x[:, j])[0, 1] if y.std() > 0 and x[:, j].std() > 0 else None
            design["predictor_correlations"][f"{PREDICTORS[i]} vs {PREDICTORS[j]}"] = correlation
    for j, step in enumerate(STEPS):
        design["by_step"][str(step)] = {"coefficients": dict(zip(PREDICTORS, coef[:, j].tolist())),
                                       "r_squared": float(1 - rss[j] / sst[j]) if sst[j] > 0 else None,
                                       "residual_mean": float(residuals[:, j].mean()),
                                       "residual_sd": float(residuals[:, j].std())}
    return residuals, design


def proportional_tools(tools, seed):
    """Hamilton quotas from exact integer remainders with seeded random tie-breaking."""
    groups = {}
    for i, name in enumerate(tools):
        groups.setdefault(name, []).append(i)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(names).tolist()
    quota, remainder = {}, {}
    for name in names:
        quota[name], remainder[name] = divmod(SIZE * len(groups[name]), len(tools))
    remaining = SIZE - sum(quota.values())
    extra = sorted(tie_order, key=lambda name: -remainder[name])[:remaining]
    for name in extra:
        quota[name] += 1
    chosen = []
    for name in names:
        chosen.extend(rng.choice(groups[name], size=quota[name], replace=False).tolist())
    if Counter(tools[i] for i in chosen) != Counter(quota):
        raise AssertionError("Incorrect tool quotas")
    return sorted(chosen), {"pool_counts": {n: len(groups[n]) for n in names}, "quotas": quota,
                            "remainder_denominator": len(tools), "remainder_numerators": remainder,
                            "extra_seats": extra, "tie_order": tie_order}


def stats(indices, ids, tools, counts, bins):
    counts_by_tool = Counter(tools[i] for i in indices)
    return {"ids": [ids[i] for i in indices], "distinct_tools": len(counts_by_tool),
            "tool_counts": dict(sorted(counts_by_tool.items())), "requests": len(indices),
            "empty_value_requests": int((counts[indices, 2] == 0).sum()),
            "value_length_bin_counts": {str(k): n for k, n in sorted(Counter(bins[indices].tolist()).items())},
            "token_totals": dict(zip(TOKEN_KEYS, counts[indices].sum(axis=0).tolist())),
            "answer_tokens_total": int(counts[indices, 0].sum()),
            "input_plus_answer_tokens_total": int(counts[indices, 4].sum())}


def build(primary, out, seeds):
    data, ids, tools, counts, losses, receipts, fingerprints = training_inputs(primary)
    cutoffs, bins = length_strata(counts[:, 2])
    residuals, ols = fit_residuals(counts, losses)
    nonempty = np.flatnonzero(counts[:, 2] > 0)
    empty = set(np.flatnonzero(counts[:, 2] == 0).tolist())
    provenance = {"numpy": np.__version__, "scikit_learn": sklearn.__version__, "python": platform.python_version(),
                  "script_sha256": file_hash(__file__), "cluster_seed": CLUSTER_SEED, "clusters": CLUSTERS,
                  "cluster_draw_source_sha256": file_hash(Path(__file__).with_name("select_experiment.py")),
                  "steps": STEPS, "size": SIZE, "score_receipts": receipts, **fingerprints}
    # The target callback uses twice the pool mean non-padding input length times the subset size.
    common = (2 * SIZE * int(counts[:, 4].sum()) + len(ids) - 1) // len(ids)
    manifest = {**provenance, "selection_manifest_sha256": {}, "seeds": seeds, "pool_size": len(ids),
                "pool_distinct_tools": len(set(tools)), "pool_input_tokens": int(counts[:, 4].sum()),
                "target_input_token_budget": common, "value_length_upper_thresholds": cutoffs,
                "value_length_definition": "0 is reserved for zero; positive bins use integer inverted-CDF deciles, right-closed; duplicate and terminal cutoffs removed.",
                "residual_ols": ols, "groups": {"Bprime": "E1: original pooled-field trajectory subset, identical bytes",
                "E2": "Uniform within B-matched value-token deciles; same empty IDs as B",
                "E3": "Equal cluster quotas from five residual whole-answer losses; same empty IDs as B",
                "E4": "Uniform within tool, proportional Hamilton quotas with random tie breaks"}}
    for seed in seeds:
        directory = out / str(seed)
        directory.mkdir()
        orig, orig_hashes = original_subsets(primary, seed, data, ids, receipts, fingerprints)
        shared = sorted(empty.intersection(orig["B"]))
        if (len(shared) != math.floor(SIZE * len(empty) / len(ids) + .5)
                or any(sorted(empty.intersection(orig[arm])) != shared for arm in ("C", "Bprime"))):
            raise ValueError("Original shared-empty IDs disagree")
        residual_draw, clusters = cluster_draw(residuals, nonempty, SIZE-len(shared), seed, CLUSTERS, CLUSTER_SEED)
        for cluster in clusters["clusters"]:
            cluster["ids"] = [ids[i] for i in cluster.pop("indices")]
        proportional, quotas = proportional_tools(tools, seed)
        selected = {"Bprime": orig["Bprime"], "E2": match_length(bins, orig["B"], shared, seed),
                    "E3": sorted(shared + residual_draw), "E4": proportional}
        report = {"seed": seed, **provenance, "source_selection_sha256": orig_hashes, "subsets": {},
                  "shared_empty_ids": [ids[i] for i in shared], "value_length_upper_thresholds": cutoffs,
                  "residual_ols": ols, "residual_clusters": clusters, "proportional_tool_allocation": quotas,
                  "target_input_token_budget": common, "primary_controls": {}}
        for arm in ("A", "B", "C"):
            report["primary_controls"][arm] = stats(orig[arm], ids, tools, counts, bins)
        for arm, chosen in selected.items():
            if sorted(set(chosen)) != chosen or len(chosen) != SIZE:
                raise ValueError(f"{seed}/{arm}: subset must contain 1,000 distinct rows in training order")
            path = directory / f"{arm}.jsonl"
            if arm == "Bprime":
                shutil.copyfile(primary / "selection" / str(seed) / path.name, path)
            else:
                with path.open("w") as stream:
                    for i in chosen:
                        print(json.dumps(data[i], ensure_ascii=False), file=stream)
            if rows(path) != [data[i] for i in chosen]:
                raise ValueError(f"{seed}/{arm}: written training rows differ")
            report["subsets"][arm] = {**stats(chosen, ids, tools, counts, bins), "sha256": file_hash(path)}
        combined = {**orig, **selected}
        report["overlap_counts"] = {f"{a}-{b}": len(set(combined[a]).intersection(combined[b]))
                                    for a, b in combinations(combined, 2)}
        write_json(directory / "selection.json", report)
        manifest["selection_manifest_sha256"][str(seed)] = file_hash(directory / "selection.json")
    write_json(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=Path("local/course-run"))
    parser.add_argument("--out", type=Path, default=Path("local/followup/phase1/selection"))
    parser.add_argument("--seeds", type=int, nargs="+", choices=[42, 43, 44], default=[42, 43, 44])
    args = parser.parse_args()
    if args.out.exists() or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide a new output directory and distinct seeds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp, threadpool_limits(limits=1):
        complete = Path(tmp) / "complete"
        complete.mkdir()
        build(args.primary, complete, args.seeds)
        complete.rename(args.out)
    print(f"Wrote {len(args.seeds)*4} train-only subsets to {args.out}")


if __name__ == "__main__":
    main()
