"""Reproduce the historical selectors and freeze the 20 new train-only Round 2 subsets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

import select_difficulty_control as difficulty
from select_followup import SIZE, length_strata, match_length, read, rows, training_inputs, write_json
from score_trajectories import file_hash


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
PRIMARY_ARMS = ("A", "B", "C", "E2", "E2b")
DOSES = {"L-short": 12., "L-long": 24.}
CELL_KEYS = {f"s{s}-{m}" for s in SEEDS for m in (*DOSES, *(PRIMARY_ARMS if s >= 45 else ()))}


def tilted_quota(capacity, means, seats, target, seed):
    """Exponentially tilt the pool histogram, then round and redistribute within capacity."""
    capacity, means = np.asarray(capacity), np.asarray(means, dtype=float)
    if (capacity.ndim != 1 or means.shape != capacity.shape or not np.issubdtype(capacity.dtype, np.integer)
            or not np.all(np.isfinite(means)) or np.any(means <= 0) or np.any(capacity <= 0)
            or not np.isfinite(target) or not 0 < seats <= capacity.sum()):
        raise ValueError("Invalid bin capacities, means, sample size or target")

    order = np.argsort(means)
    def extreme(indices):
        counts = capacity[indices]
        take = np.minimum(counts, np.maximum(0, seats - (np.cumsum(counts) - counts)))
        return float(take @ means[indices] / seats)

    lower, upper = extreme(order), extreme(order[::-1])
    if not lower <= target <= upper:
        raise ValueError(f"Target {target} falls outside attainable interval [{lower}, {upper}]")
    log_mean = np.log(means)
    def weights(theta):
        log_weight = np.log(capacity) + theta * log_mean
        p = np.exp(log_weight - log_weight.max())
        return p / p.sum()

    # Relative masses use the original bin frequencies, as in the research plan.
    lo, hi = -1., 1.
    while weights(lo) @ means > target + 1e-12:
        lo *= 2
    while weights(hi) @ means < target - 1e-12:
        hi *= 2
    for _ in range(80):
        theta = (lo + hi) / 2
        if weights(theta) @ means < target:
            lo = theta
        else:
            hi = theta
    theta = (lo + hi) / 2
    mass = weights(theta)
    active = np.arange(len(capacity))
    quota = np.zeros_like(capacity)
    capped = []
    rng = np.random.default_rng(seed)
    while len(active):
        remaining = seats - int(quota.sum())
        if not remaining:
            break
        expected = remaining * mass[active] / mass[active].sum()
        saturated = expected >= capacity[active]
        if saturated.any():
            blocked = active[saturated]
            capped.extend(blocked.tolist())
            quota[blocked] = capacity[blocked]
            active = active[~saturated]
            continue
        base = np.floor(expected).astype(int)
        extra = remaining - int(base.sum())
        random_order = rng.permutation(len(active))
        ranked = random_order[np.argsort(-(expected - base)[random_order], kind="stable")]
        base[ranked[:extra]] += 1
        quota[active] = base
        break
    if quota.sum() != seats or np.any(quota > capacity):
        raise AssertionError("Length-dose apportionment violated capacities")
    return quota, {"target_mean": target, "tilt": theta, "apportioned_expected_mean": float(quota @ means / seats),
                   "capacity_limited_bins": capped, "quota": quota.tolist()}


def draw_bins(bins, quota, seed):
    """Seeded uniform draws without replacement within each ascending integer bin."""
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(np.flatnonzero(bins == i), int(n), replace=False) for i, n in enumerate(quota)]
    return np.sort(np.concatenate(chosen))


def from_clusters(clusters, seed, positions):
    # Phase 1 fitted K-means only once for each feature matrix. The archived groups
    # retain its canonical pool order and its fixed round-robin choice of quotas.
    rng = np.random.default_rng(seed)
    samples = []
    for group in clusters:
        indices = np.array([positions[identifier] for identifier in group["ids"]])
        if len(indices) != group["size"] or np.any(np.diff(indices) <= 0):
            raise ValueError("Frozen trajectory cluster lost its canonical membership")
        samples.extend(rng.permutation(indices)[:group["selected"]])
    return np.sort(np.asarray(samples, dtype=int))


def difficulty_draw(seed, sets, length, loss, previous):
    a, b, capacity, lc, dc, targets = difficulty.scheme(10, length, loss, {seed: sets})
    if (lc.tolist() != previous["length_upper_bounds"] or dc.tolist() != previous["difficulty_upper_bounds"]
            or capacity.tolist() != previous["pool_joint_counts"]):
        raise ValueError("Round 2 length or difficulty bins differ from E2b")
    target = targets[seed]
    if target["flow"] != SIZE:
        raise ValueError(f"Fixed 10-by-10 margins infeasible at seed {seed}; no coarsening or retry")
    quota, _ = difficulty.choose_joint(capacity, target["a_row"], target["b_col"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, difficulty.RANDOM_STREAM]))
    selected = []
    for i, j in np.ndindex(quota.shape):
        if quota[i, j]:
            candidates = np.flatnonzero((a == i) & (b == j))
            selected.extend(rng.choice(candidates, int(quota[i, j]), replace=False))
    chosen = np.sort(np.array(selected, dtype=int))
    if not np.array_equal(difficulty.joint(a, b, quota.shape, chosen), quota):
        raise ValueError("Difficulty control did not reproduce its exact joint quotas")
    return chosen


def build(primary, phase1, previous_e2b, out):
    data, ids, tools, counts, _, receipts, fingerprints = training_inputs(primary)
    _, old_ids, _, loss, original, _ = difficulty.load_inputs(primary, phase1)
    frozen_e2b = read(previous_e2b / "manifest.json")
    registration = read(ROOT / "results/followup/e2b-registration.json")
    source_hashes = {s: file_hash(Path(__file__).with_name(s)) for s in (
        "select_experiment.py", "select_followup.py", "select_difficulty_control.py", "select_round2.py")}
    if (ids != old_ids or registration["pool_sha256"] != fingerprints["train_sha256"]
            or frozen_e2b["chosen_requested_bins_per_axis"] != 10
            or any(source_hashes[k] != registration["source_sha256"][k] for k in source_hashes if k != "select_round2.py")
            or any(frozen_e2b[k] != registration["selection_design"][k] for k in ("length_upper_bounds","difficulty_upper_bounds"))
            or np.__version__ != frozen_e2b["versions"]["numpy"] or scipy.__version__ != frozen_e2b["versions"]["scipy"]):
        raise ValueError("Frozen selection provenance or pinned numerical libraries differ")
    length = counts[:, 2]
    if (length <= 0).any() or not np.isfinite(loss).all():
        raise ValueError("Round 2 length tilt is defined for this frozen all-positive pool")
    bounds, labels = length_strata(length)
    bins = labels - 1
    if bounds != frozen_e2b["length_upper_bounds"]:
        raise ValueError("Length deciles differ from the frozen controls")
    capacities = np.bincount(bins)
    bin_means = np.bincount(bins, weights=length) / capacities
    loss_by_bin = np.bincount(bins, weights=loss) / capacities
    difficulty_bins = np.searchsorted(frozen_e2b["difficulty_upper_bounds"], loss)
    positions = {identifier: i for i, identifier in enumerate(ids)}
    historical = read(primary / "selection/42/selection.json")
    clusters = historical["clustering"]
    all_indices = np.arange(len(ids))
    for method in ("B", "C"):
        members = [positions[key] for cluster in clusters[method]["clusters"] for key in cluster["ids"]]
        if sorted(members) != all_indices.tolist() or sum(c["selected"] for c in clusters[method]["clusters"]) != SIZE:
            raise ValueError(f"Frozen {method} groups do not partition the training pool")
    manifest = {**fingerprints, "score_receipts": receipts, "source_sha256": source_hashes,
                "phase1_selection_sha256": file_hash(phase1 / "manifest.json"),
                "e2b_selection_sha256": file_hash(previous_e2b / "manifest.json"),
                "frozen_cluster_selection_sha256": file_hash(primary / "selection/42/selection.json"),
                "pool_size": len(ids), "subset_size": SIZE, "seeds": list(SEEDS), "cluster_seed": 42,
                "length_upper_bounds": bounds, "difficulty_upper_bounds": frozen_e2b["difficulty_upper_bounds"],
                "pool_bin_capacities": capacities.tolist(), "pool_bin_mean_lengths": bin_means.tolist(),
                "target_input_token_budget": (2 * SIZE * int(counts[:, 4].sum()) + len(ids)-1) // len(ids),
                "ids_sha256_format": "UTF-8 training IDs in pool order, one per line with trailing newline",
                "historical_replay": {}, "cells": {}}
    for seed in SEEDS:
        selected = {"A": np.sort(np.random.default_rng(seed).choice(len(ids), SIZE, replace=False))}
        for method in ("B", "C"):
            selected[method] = from_clusters(clusters[method]["clusters"], seed, positions)
        selected["E2"] = np.array(match_length(labels, selected["B"], [], seed))
        selected["E2b"] = difficulty_draw(seed, selected, length, loss, frozen_e2b)
        dose_meta = {}
        for method, target in DOSES.items():
            quota, dose_meta[method] = tilted_quota(capacities, bin_means, SIZE, target, seed)
            selected[method] = draw_bins(bins, quota, seed)
            if not np.array_equal(np.bincount(bins[selected[method]], minlength=len(capacities)), quota):
                raise ValueError(f"Wrong dose quotas at seed {seed}")
        (out / str(seed)).mkdir()
        for method, indices in selected.items():
            label = f"s{seed}-{method}"
            if len(indices) != SIZE or len(np.unique(indices)) != SIZE or np.any(np.diff(indices) <= 0):
                raise ValueError(f"{label} must contain 1,000 distinct items in original training order")
            ids_hash = hashlib.sha256("".join(ids[i] + "\n" for i in indices).encode()).hexdigest()
            payload = "".join(json.dumps(data[i], ensure_ascii=False) + "\n" for i in indices).encode()
            subset_hash = hashlib.sha256(payload).hexdigest()
            if seed < 45 and method in PRIMARY_ARMS:
                if method == "E2b":
                    check = registration["selections"][label]["sha256"]
                    previous_path = previous_e2b / str(seed) / "E2b.jsonl"
                else:
                    subset_dir = (phase1 if method == "E2" else primary / "selection") / str(seed)
                    previous_path = subset_dir / f"{method}.jsonl"
                    check = read(subset_dir / "selection.json")["subsets"][method]["sha256"]
                if check != subset_hash or file_hash(previous_path) != check:
                    raise ValueError(f"Historical replay failed for {label}")
                if method in original[seed] and not np.array_equal(indices, original[seed][method]):
                    raise ValueError(f"Historical request IDs differ for {label}")
                manifest["historical_replay"][label] = subset_hash
                continue
            path = out / str(seed) / f"{method}.jsonl"
            path.write_bytes(payload)
            if file_hash(path) != subset_hash or rows(path) != [data[i] for i in indices]:
                raise ValueError(f"Round-trip check failed: {label}")
            manifest["cells"][label] = {
                "seed": seed, "method": method, "path": path.relative_to(out).as_posix(),
                "size": SIZE, "sha256": subset_hash, "ids_sha256": ids_hash,
                "mean_value_tokens": float(length[indices].mean()), "mean_initial_value_nll": float(loss[indices].mean()),
                "within_length_bin_mean_nll_residual": float((loss[indices] - loss_by_bin[bins[indices]]).mean()),
                "length_counts": np.bincount(bins[indices], minlength=len(capacities)).tolist(),
                "difficulty_counts": np.bincount(difficulty_bins[indices], minlength=len(frozen_e2b["difficulty_upper_bounds"])+1).tolist(),
                "unique_tools": len(Counter(tools[i] for i in indices)), "intersection_with_same_seed_A": len(np.intersect1d(indices,selected["A"])),
                "single_pass_input_tokens": int(counts[indices, 4].sum()), "single_pass_answer_tokens": int(counts[indices, 0].sum())}
            if method in dose_meta:
                manifest["cells"][label]["length_policy"] = dose_meta[method]
    if set(manifest["cells"]) != CELL_KEYS or len(manifest["historical_replay"]) != 15:
        raise AssertionError("Missing required new selections or historical replays")
    write_json(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=Path("local/course-run"))
    parser.add_argument("--phase1-selection", type=Path, default=Path("local/followup/phase1/selection"))
    parser.add_argument("--e2b-selection", type=Path, default=Path("local/followup/e2b-eligibility"))
    parser.add_argument("--out", type=Path, default=Path("local/followup/round2/selection"))
    parser.add_argument("--verify", action="store_true", help="Replay from frozen inputs and verify every existing output byte.")
    args = parser.parse_args()
    if args.out.exists() != args.verify:
        parser.error("Use a new output directory for generation or --verify for an existing one.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp, threadpool_limits(limits=1):
        dest = Path(tmp) / "complete"
        dest.mkdir()
        build(args.primary, args.phase1_selection, args.e2b_selection, dest)
        if args.verify:
            expected = {p.relative_to(dest): file_hash(p) for p in dest.rglob("*") if p.is_file()}
            actual = {p.relative_to(args.out): file_hash(p) for p in args.out.rglob("*") if p.is_file()}
            if expected != actual:
                raise ValueError("Frozen selections differ from deterministic replay")
        else:
            if args.out.exists():
                raise FileExistsError(args.out)
            dest.rename(args.out)
    print(f"Verified 20 new subsets and 15 historical replays: {args.out}; manifest {file_hash(args.out / 'manifest.json')}")


if __name__ == "__main__":
    main()
