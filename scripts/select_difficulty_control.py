"""Check, and materialize, a train-only, dual-marginal control for the frozen Phase 1 selections.

E2b matches the original random arm A's argument lengths and whole-answer arm B's
step-zero argument-value losses. Run this independently of the Phase 1 controller.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import maximum_flow

from score_trajectories import file_hash
from select_followup import read, rows, write_json


SEEDS = (42, 43, 44)
SIZE = 1000
TIERS = (10, 5, 2)
RANDOM_STREAM = 0xE2B
FOLLOWUP_PIN = "89ed9d5c2c96b4207e0f93dd4c003c3fa68e721335c81a916299e141115d29e6"


def load_inputs(primary, selections):
    train = primary / "frozen/train.jsonl"
    train_hash = file_hash(train)
    pool = rows(train)
    ids = [str(r["id"]) for r in pool]
    if len(set(ids)) != len(ids) or len(ids) != 4000:
        raise ValueError("Expected 4,000 different frozen training rows")
    by_id = {k: i for i, k in enumerate(ids)}
    score = primary / "scores/step-0.jsonl"
    score_hash = file_hash(score)
    config = primary / "scores/score-config.json"
    if (read(config)["train_sha256"] != train_hash
            or read(score.with_suffix(".complete.json")) != {"rows": len(ids), "step": 0, "sha256": score_hash}):
        raise ValueError("Proxy score receipt or training-pool hash mismatch")
    raw_scores = rows(score)
    by_score = {str(r["id"]): r for r in raw_scores}
    if len(raw_scores) != len(ids) or set(by_score) != set(ids):
        raise ValueError("Scored request IDs differ")
    ordered = [by_score[k] for k in ids]
    counts = np.array([[r["value_tokens"], r["input_tokens"], r["all_tokens"]] for r in ordered])
    sums = np.array([r["value_sum"] for r in ordered])
    length = counts[:, 0]
    if (not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any() or (length > counts[:, 2]).any()
            or (counts[:, 2] > counts[:, 1]).any() or not np.isfinite(sums).all() or (sums < 0).any()
            or (sums[length == 0] != 0).any()):
        raise ValueError("Unexpected token counts or teacher-forced value losses")
    difficulty = np.divide(sums, length, out=np.full(len(ids), np.nan), where=length > 0)
    top = selections / "manifest.json"
    master = read(top)
    if file_hash(top) != FOLLOWUP_PIN or master["train_sha256"] != train_hash:
        raise ValueError("The Phase 1 selection bundle is not the frozen input")
    controls, sources = {}, {}
    for seed in SEEDS:
        manifest = primary / "selection" / str(seed) / "selection.json"
        receipt = read(primary / "receipts" / f"select-{seed}.json")
        original = read(manifest)
        matching = [v for k, v in receipt["outputs"].items() if k.endswith(f"/selection/{seed}/selection.json")]
        meta = selections / str(seed) / "selection.json"
        followup = read(meta)
        if (matching != [file_hash(manifest)] or original["receipts"]["0"]["sha256"] != score_hash
                or original["train_sha256"] != train_hash or original["seed"] != seed
                or file_hash(meta) != master["selection_manifest_sha256"][str(seed)]):
            raise ValueError("Inconsistent primary/follow-up selection manifests")
        controls[seed], sources[seed] = {}, {}
        for arm in ("A", "B", "E2"):
            directory, definition = (meta.parent, followup) if arm == "E2" else (manifest.parent, original)
            path = directory / (arm + ".jsonl")
            records = rows(path)
            keys = [str(row["id"]) for row in records]
            spec = definition["subsets"][arm]
            positions = [by_id[k] for k in keys]
            if (file_hash(path) != spec["sha256"] or keys != spec["ids"] or len(set(keys)) != SIZE
                    or len(keys) != SIZE or any(r != pool[i] for i, r in zip(positions, records))):
                raise ValueError(f"Invalid {seed}/{arm} reference set")
            controls[seed][arm] = np.array(positions, dtype=int)
            sources[seed][arm] = spec["sha256"]
    return pool, ids, counts, difficulty, controls, {
        "train_sha256": train_hash, "step_0_sha256": score_hash, "score_config_sha256": file_hash(config),
        "primary_and_followup_subsets_sha256": sources, "phase1_selection_manifest_sha256": FOLLOWUP_PIN,
    }


def thresholds(values, k):
    """Inverse empirical CDF on nonempty rows. Equal values are never split across bins."""
    sorted_values = np.sort(values)
    if not len(sorted_values):
        raise ValueError("No positive-length training examples")
    cuts = [sorted_values[(d*len(values) + k-1)//k - 1] for d in range(1, k)]
    return np.unique([c for c in cuts if c < sorted_values[-1]]).astype(values.dtype)


def scaled_counts(source, total):
    """Largest-remainder apportionment, ties to the lower bin index."""
    denom = int(source.sum())
    if denom == 0:
        if total:
            raise ValueError("Positive target seats with zero source counts")
        return np.zeros_like(source)
    bases, remainder = np.divmod(source*total, denom)
    extra = total - int(bases.sum())
    bases[np.argsort(-remainder, kind="stable")[:extra]] += 1
    return bases


def joint(a, b, shape, idx):
    valid = idx[a[idx] >= 0]
    return np.bincount(a[valid]*shape[1] + b[valid], minlength=shape[0]*shape[1]).reshape(shape)


def flow_size(capacity, row, col):
    n, m = capacity.shape
    graph = np.zeros((n+m+2, n+m+2), dtype=np.int32)
    graph[0, 1:n+1] = row
    graph[1:n+1, n+1:-1] = capacity
    graph[n+1:-1, -1] = col
    return int(maximum_flow(csr_matrix(graph), 0, len(graph)-1).flow_value)


def scheme(k, length, loss, controls):
    active = length > 0
    lc, dc = thresholds(length[active], k), thresholds(loss[active], k)
    a, b = np.full(len(length), -1, int), np.full(len(length), -1, int)
    a[active], b[active] = np.searchsorted(lc, length[active]), np.searchsorted(dc, loss[active])
    shape = (len(lc)+1, len(dc)+1)
    capacity = joint(a, b, shape, np.arange(len(a)))
    targets = {}
    for seed, sets in controls.items():
        r = joint(a, b, shape, sets["A"]).sum(axis=1)
        c = joint(a, b, shape, sets["B"]).sum(axis=0)
        n = int(c.sum())
        rows_a, rows_pool = scaled_counts(r,n), scaled_counts(capacity.sum(axis=1),n)
        targets[seed] = {"a_row": rows_a, "b_col": c, "pool_row": rows_pool, "positive_seats": n,
                         "a_empty": SIZE-int(r.sum()), "b_empty": SIZE-n,
                         "flow": flow_size(capacity,rows_a,c), "pool_alternative_flow": flow_size(capacity,rows_pool,c)}
    return a, b, capacity, lc, dc, targets


def choose_joint(capacity, row, col):
    """Minimize weighted squared joint drift using an integral bipartite transport LP."""
    # Splitting a cell of integer capacity c into c one-seat edges represents the
    # exact marginal costs of its convex quadratic penalty. A basic transport
    # optimum is integral; assert the actual HiGHS solution satisfies that property.
    row_bins, col_bins = np.nonzero(capacity)
    cells = np.repeat(np.arange(len(row_bins)), capacity[row_bins, col_bins])
    unit = np.concatenate([np.arange(1, cap+1) for cap in capacity[row_bins, col_bins]])
    target = capacity * (row.sum()/capacity.sum())
    mean = target[row_bins[cells], col_bins[cells]]
    costs = (2*unit - 1 - 2*mean) / np.maximum(mean, 1.)
    n, m = capacity.shape
    indices = np.vstack((np.concatenate((row_bins[cells], n+col_bins[cells])), np.tile(np.arange(len(cells)),2)))
    constraints = coo_array((np.ones(indices.shape[1]), indices), shape=(n+m,len(cells))).tocsr()
    solution = linprog(costs, A_eq=constraints, b_eq=np.r_[row,col], bounds=(0,1), method="highs-ds")
    if not solution.success or not np.allclose(solution.x, np.rint(solution.x), atol=1e-7, rtol=0):
        raise RuntimeError(f"No verified integral minimum-cost allocation: {solution.message}")
    quotas = np.zeros_like(capacity)
    allocation = np.bincount(cells, weights=np.rint(solution.x), minlength=len(row_bins)).astype(int)
    quotas[row_bins, col_bins] = allocation
    if (not np.array_equal(quotas.sum(1),row) or not np.array_equal(quotas.sum(0),col)
            or (quotas > capacity).any() or (quotas < 0).any()):
        raise AssertionError("Allocation violates capacity or margins")
    objective = float(np.sum((quotas-target)**2 / np.maximum(target,1.)))
    if not np.isclose(solution.fun + np.sum(target**2/np.maximum(target,1.)), objective, atol=1e-7):
        raise AssertionError("Expanded-edge and direct joint objectives disagree")
    return quotas, objective


def summary(x):
    v = x[np.isfinite(x)]
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min_p10_p25_p50_p75_p90_p99_max": np.quantile(v,[0,.1,.25,.5,.75,.9,.99,1]).tolist()}


def continuous(left, right):
    x, y = np.sort(left[np.isfinite(left)]), np.sort(right[np.isfinite(right)])
    points = np.union1d(x,y)
    ks = np.max(np.abs(np.searchsorted(x,points,side="right")/len(x)-np.searchsorted(y,points,side="right")/len(y)))
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1))/2)
    diff = float(x.mean()-y.mean())
    return {"mean_difference": diff, "smd": diff/float(sd) if sd else None, "ks": float(ks)}


def hist_compare(left, right):
    # Arrays may represent either marginal counts or complete joint counts.
    p, q = left / left.sum(), right / right.sum()
    return {"count_difference": (left-right).tolist(), "max_abs_count_difference": int(np.abs(left-right).max()),
            "l1_count_difference": int(np.abs(left-right).sum()), "overlap": float(np.minimum(p,q).sum())}


def overlap_bounds(quota, other, capacity, shared_count):
    low = int(np.maximum(quota + other - capacity, 0).sum())+shared_count
    high = int(np.minimum(quota,other).sum())+shared_count
    expected = float(np.divide(quota*other,capacity,out=np.zeros_like(capacity,float),where=capacity>0).sum())
    return {"min_given_cells": low, "max_given_cells": high, "mean_under_cell_draw": expected+shared_count}


def run(primary, selections, out):
    started = time.monotonic()
    pool, ids, counts, loss, controls, sources = load_inputs(primary, selections)
    length = counts[:,0]
    full = np.arange(len(ids))
    tools = [r["messages"][-1]["tool_calls"][0]["function"]["name"] for r in pool]
    tiers = []
    for k in TIERS:
        a,b,capacity,lc,dc,targets = scheme(k,length,loss,controls)
        feasibility = {s: {key:v for key,v in t.items() if key not in ("a_row","b_col","pool_row")}
                       for s,t in targets.items()}
        tiers.append({"requested_bins_per_axis": k, "shape": list(capacity.shape), "seeds": feasibility})
        if all(t["flow"]==t["positive_seats"] for t in targets.values()):
            break
    else:
        write_json(out/"manifest.json", {"status":"infeasible", "tiers":tiers, **sources})
        return
    budget = (2*SIZE*int(counts[:,1].sum())+len(ids)-1)//len(ids)
    report = {"status":"feasible", "population":len(ids), "pool_empty_values":int((length==0).sum()),
              "size":SIZE, "seeds":list(SEEDS), "common_target_input_token_budget":budget,
              "train_only_scope":"Original training pool, initial proxy loss and existing A/B/E2 selections.",
              "primary_length_target":"Same-seed A length counts. B fixes the empty IDs; if necessary apportion positive A counts to B's positive total.",
              "difficulty_target":"Same-seed B histogram of step-0 per-example value_sum / value_tokens; undefined for empty values.",
              "joint_objective":"Minimize sum((selected - positive_seats * pool_cell / positive_pool_size)^2 / max(positive_seats * pool_cell / positive_pool_size, 1)) over positive cells with both margins fixed.",
              "coarsening_rule":"Try 10x10, 5x5, 2x2 in order. First maximum-flow-feasible tier across all three seeds; tie values always stay together. No reroll after continuous balance checks.",
              "rng_namespace":RANDOM_STREAM, "phase1_selections_untouched":True, "feasibility_tiers":tiers,
              "chosen_requested_bins_per_axis":k, "length_upper_bounds":lc.tolist(), "difficulty_upper_bounds":dc.tolist(),
              "pool_joint_counts":capacity.tolist(), "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
              "source_sha256":file_hash(__file__), "sources":sources, "seed_manifest_sha256":{}}
    choices = {}
    for seed in SEEDS:
        t, sets = targets[seed],controls[seed].copy()
        quota, objective = choose_joint(capacity,t["a_row"],t["b_col"])
        shared = sets["B"][length[sets["B"]]==0]
        sampled = list(shared)
        rng = np.random.default_rng(np.random.SeedSequence([seed,RANDOM_STREAM]))
        for i,j in np.ndindex(quota.shape):
            if quota[i,j]:
                candidates = np.flatnonzero((a==i)&(b==j))
                sampled.extend(rng.choice(candidates,int(quota[i,j]),replace=False))
        chosen = np.array(sorted(sampled),dtype=int)
        if len(chosen)!=SIZE or len(set(chosen))!=SIZE or not np.array_equal(joint(a,b,quota.shape,chosen),quota):
            raise AssertionError("Invalid E2b subset")
        sets.update(E2b=chosen, pool=full)
        choices[seed] = set(chosen)
        seed_dir = out/str(seed)
        seed_dir.mkdir()
        path = seed_dir/"E2b.jsonl"
        with path.open("w") as f:
            for i in chosen:
                print(json.dumps(pool[i],ensure_ascii=False),file=f)
        if rows(path) != [pool[i] for i in chosen]:
            raise AssertionError("Subset rows changed during serialization")
        stats = {}
        mats = {name:joint(a,b,quota.shape,idx) for name,idx in sets.items()}
        for name,idx in sets.items():
            stats[name] = {"requests":len(idx),"empty":int((length[idx]==0).sum()),"distinct_tools":len({tools[i] for i in idx}),
                           "length":summary(length[idx]),"difficulty":summary(loss[idx]),
                           "single_pass_input_tokens":int(counts[idx,1].sum()),"single_pass_answer_tokens":int(counts[idx,2].sum()),
                           "length_counts":mats[name].sum(1).tolist(),"difficulty_counts":mats[name].sum(0).tolist(),
                           "joint_counts":mats[name].tolist()}
        pairs = {}
        for left,right in [("E2b","A"),("E2b","B"),("E2b","E2"),("E2b","pool"),("B","A"),("E2","B"),("E2","A")]:
            x,y = sets[left],sets[right]
            overlap = len(set(x)&set(y))
            desc = {"length":continuous(length[x],length[y]),"difficulty":continuous(loss[x],loss[y]),
                    "length_bins":hist_compare(mats[left].sum(1),mats[right].sum(1)),
                    "difficulty_bins":hist_compare(mats[left].sum(0),mats[right].sum(0)),
                    "joint_bins":hist_compare(mats[left],mats[right]),"same_requests":overlap,
                    "jaccard":overlap/(len(x)+len(y)-overlap)}
            if left=="E2b":
                desc.update(overlap_bounds(quota,mats[right],capacity,len(set(shared)&set(y))))
            pairs[left+"-"+right] = desc
        meta = {"seed":seed, "ids":[ids[i] for i in chosen], "sha256":file_hash(path), "target_input_token_budget":budget,
                "selected_joint_counts":quota.tolist(), "weighted_joint_objective":objective,
                "length_target":t["a_row"].tolist(), "difficulty_target":t["b_col"].tolist(),
                "alternative_pool_length_target":t["pool_row"].tolist(), "alternative_pool_length_target_flow":t["pool_alternative_flow"],
                "pool_rounded_length_max_count_difference":int(np.max(np.abs(t["pool_row"]-t["a_row"]))),
                "occupied_pool_cells":int(np.sum(capacity>0)),"occupied_subset_cells":int(np.sum(quota>0)),
                "saturated_nonempty_cells":int(np.sum((quota==capacity)&(capacity>0))),"shared_empty_ids":[ids[i] for i in shared],
                "summaries":stats, "pairwise":pairs}
        write_json(seed_dir/"selection.json",meta)
        report["seed_manifest_sha256"][str(seed)] = file_hash(seed_dir/"selection.json")
    report["between_seed_overlap"]={f"{s}-{t}": len(choices[s]&choices[t]) for s,t in combinations(SEEDS,2)}
    report["selector_elapsed_seconds"]=time.monotonic()-started
    write_json(out/"manifest.json",report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary",type=Path,default=Path("local/course-run"))
    parser.add_argument("--phase1-selection",type=Path,default=Path("local/followup/phase1/selection"))
    parser.add_argument("--out",type=Path,default=Path("local/followup/e2b-eligibility"))
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output directory")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        output=Path(tmp)/"complete"
        output.mkdir()
        run(args.primary,args.phase1_selection,output)
        output.rename(args.out)
    result=read(args.out/"manifest.json")
    print(f"{result['status']}: {args.out}")


if __name__=="__main__":
    main()
