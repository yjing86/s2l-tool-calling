"""Choose training IDs from per-candidate proxy losses using the study's A, B or C method."""

import argparse
import json
import math
from pathlib import Path
import sys


def loss_series(values, label, length):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{label} needs one loss per checkpoint ({length} values)")
    if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError(f"{label} losses must be finite, nonnegative numbers")
    return values


def read_trajectories(source):
    ids, whole, function, value = [], [], [], []
    seen_ids = set()
    steps = None
    with source.open("rb") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))
                if not isinstance(row, dict):
                    raise ValueError("expected one JSON object")
                key = row["id"]
                if not isinstance(key, str) or not key.strip() or key in seen_ids:
                    raise ValueError("id must be a unique nonempty string")
                s = row["steps"]
                if (not isinstance(s, list) or len(s) != 5 or any(type(t) is not int for t in s)
                        or s[0] != 0 or s != sorted(set(s))):
                    raise ValueError("steps must be five strictly increasing integers starting at zero")
                if steps is not None and s != steps:
                    raise ValueError("checkpoint steps must be identical for every candidate")
                steps = s
                whole.append(loss_series(row["all"], "all", len(s)))
                function.append(loss_series(row["function"], "function", len(s)))
                v = row["value"]
                value.append(None if v is None else loss_series(v, "value", len(s)))
                seen_ids.add(key)
                ids.append(key)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not ids:
        raise ValueError(f"{source}: no candidates")
    return ids, steps, whole, function, value


def select_ids(source, count, method, clusters, seed, cluster_seed):
    import numpy as np

    # The study module imports other scripts by name when run directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from select_experiment import cluster_draw

    ids, steps, whole, function, value = read_trajectories(source)
    if not 1 <= count <= len(ids):
        raise ValueError(f"--count must be between 1 and the pool size ({len(ids)})")
    if method == "A":
        chosen = sorted(np.random.default_rng(seed).choice(len(ids), count, replace=False).tolist())
    else:
        empty = np.array([i for i, v in enumerate(value) if v is None], dtype=int)
        filled = np.array([i for i, v in enumerate(value) if v is not None], dtype=int)
        empty_quota = math.floor(count * len(empty) / len(ids) + 0.5)
        fn = np.asarray(function, dtype=float)
        features = np.asarray(whole, dtype=float)
        if method == "C":
            # Missing value slots never enter clustering; their rows use only fn.
            val = [v if v is not None else [math.nan] * len(steps) for v in value]
            features = np.column_stack((fn, val))
        shared, _ = cluster_draw(fn, empty, empty_quota, seed, clusters, cluster_seed)
        selected, _ = cluster_draw(features, filled, count - empty_quota, seed, clusters, cluster_seed)
        chosen = sorted(shared + selected)
    return {"method": method, "seed": seed, "cluster_seed": cluster_seed if method != "A" else None,
            "requested_clusters_per_stratum": clusters if method != "A" else None, "steps": steps,
            "pool_size": len(ids), "count": count, "selected_ids": [ids[i] for i in chosen]}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python3 -m s2l_tool_calling select", description=__doc__,
                                epilog="See examples/trajectories.jsonl for a complete synthetic input.")
    p.add_argument("--input", type=Path, required=True, help="UTF-8 JSONL, one precomputed trajectory per training candidate")
    p.add_argument("--count", type=int, required=True, help="number of distinct IDs to select")
    p.add_argument("--method", choices=("A", "B", "C"), default="C",
                   help="A: uniform random, B: whole-answer trajectory, C: separate fields (default)")
    p.add_argument("--clusters", type=int, default=10, help="maximum number of clusters per stratum (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="within-cluster or uniform draw seed (default: 42)")
    p.add_argument("--cluster-seed", type=int, default=42, help="K-means initialization seed (default: 42)")
    args = p.parse_args(argv)
    if args.clusters < 1 or args.seed < 0 or not 0 <= args.cluster_seed < 2**32:
        p.error("clusters must be positive, seed nonnegative, and cluster-seed in [0, 2^32)")
    try:
        result = select_ids(args.input, args.count, args.method, args.clusters, args.seed, args.cluster_seed)
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "sklearn"}:
            raise
        p.exit(2, "Install the CPU dependencies with: python3 -m pip install -r requirements/selection.txt\n")
    except (OSError, ValueError) as exc:
        p.exit(2, f"{p.prog}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
