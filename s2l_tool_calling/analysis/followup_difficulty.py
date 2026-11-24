"""Post-hoc length/difficulty and function-exposure audit on previously locked artifacts.

python -m s2l_tool_calling.analysis.followup_difficulty \
    --primary local/course-run --phase1 local/followup/phase1 --out local/followup/phase0/difficulty

The only held-out outcomes read are already-locked primary test predictions.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from .followup_diagnostics import SEEDS, aggregate_runs, percentage, rows, save_table
from summarize_course import digest, read_json, read_model
from select_followup import STEPS, length_strata
import numpy as np

METRICS = ("mean_value_tokens", "raw_value_nll", "exact_length_residual_nll",
           "length_bin_residual_nll", "raw_top10_pct", "exact_residual_top10_pct",
           "within_exact_length_top10_pct")


def top_decile(scores):
    """Give each candidate tied at the upper-decile boundary the same fractional credit."""
    n = len(scores)
    threshold = sorted(scores.values(), reverse=True)[(n + 9) // 10 - 1]
    above = sum(x > threshold for x in scores.values())
    tied = sum(x == threshold for x in scores.values())
    credit = (n - 10 * above) / (10 * tied)
    weights = {k: 1 if v > threshold else credit if v == threshold else 0 for k, v in scores.items()}
    if not math.isclose(sum(weights.values()), n / 10):
        raise ValueError("Decile weights do not add to 10%")
    return weights, {"candidates": n, "threshold": threshold, "strictly_above": above, "at_threshold": tied,
                     "above_or_tied": above + tied, "credit_per_tie": credit}


def grouped_means(scores, labels):
    members = defaultdict(list)
    for rid, value in scores.items():
        members[labels[rid]].append(value)
    means = {g: statistics.mean(v) for g, v in members.items()}
    return {k: v - means[labels[k]] for k, v in scores.items()}


def add_e2(phase1, selections, bins):
    spec = read_json(phase1 / "preregistration.json")["spec"]
    base = phase1 / "selection"
    if spec["selection_manifest_sha256"] != digest(base / "manifest.json"):
        raise ValueError("Phase 1 selection registration differs")
    master = read_json(base / "manifest.json")
    for seed in SEEDS:
        directory = base / str(seed)
        if master["selection_manifest_sha256"][str(seed)] != digest(directory / "selection.json"):
            raise ValueError(f"Seed {seed}: selection manifest differs")
        path = directory / "E2.jsonl"
        e2 = rows(path)
        sub = read_json(directory / "selection.json")["subsets"]["E2"]
        if (sub["sha256"] != digest(path) or sub["sha256"] != spec["selections"][f"s{seed}-E2"]["sha256"]
                or list(e2) != sub["ids"] or len(e2) != 1000):
            raise ValueError(f"Seed {seed}: E2 membership differs")
        selections[seed, "E2"] = set(e2)
        if Counter(bins[k] for k in e2) != Counter(bins[k] for k in selections[seed, "B"]):
            raise ValueError(f"Seed {seed}: E2 and B length-bin counts differ")


def training_metrics(primary, selections, lengths, bins):
    groups = defaultdict(list)
    for k, n in lengths.items():
        if n > 0:
            groups[n].append(k)
    sets = {(0, "pool"): set(lengths), **selections}
    cells, thresholds = [], []
    registered = read_json(primary / "selection/42/selection.json")["receipts"]
    for step in STEPS:
        path = primary / "scores" / f"step-{step}.jsonl"
        data = rows(path)
        if (read_json(path.with_suffix(".complete.json")) != registered[str(step)] or
                registered[str(step)] != {"step": step, "sha256": digest(path), "rows": len(lengths)} or
                data.keys() != lengths.keys() or any(r["value_tokens"] != lengths[k] for k, r in data.items())):
            raise ValueError(f"Step {step}: score identity or token counts differ")
        raw = {k: r["value_sum"] / lengths[k] for k, r in data.items() if lengths[k]}
        if not all(math.isfinite(x) and x >= 0 for x in raw.values()):
            raise ValueError("Invalid mean value loss")
        exact = grouped_means(raw, lengths)
        binned = grouped_means(raw, bins)
        raw_top, raw_threshold = top_decile(raw)
        exact_top, exact_threshold = top_decile(exact)
        within = {}
        for members in groups.values():
            weights, _ = top_decile({k: raw[k] for k in members})
            within.update(weights)
        for name, meta in [("raw", raw_threshold), ("exact_length_residual", exact_threshold)]:
            thresholds.append({"step": step, "signal": name, **meta})
        for (seed, arm), members in sets.items():
            valid = sorted(members & raw.keys())
            def avg(mapping):
                return statistics.mean(mapping[k] for k in valid)
            cells.append({"step": step, "seed": seed, "method": arm, "selected": len(members),
                          "with_value_tokens": len(valid), "mean_value_tokens": statistics.mean(lengths[k] for k in members),
                          "raw_value_nll": avg(raw), "exact_length_residual_nll": avg(exact),
                          "length_bin_residual_nll": avg(binned), "raw_top10_pct": 100 * avg(raw_top),
                          "exact_residual_top10_pct": 100 * avg(exact_top), "within_exact_length_top10_pct": 100 * avg(within)})
    averages = []
    for step in STEPS:
        for arm in ("pool", "A", "B", "C", "Bprime", "E2"):
            entries = [r for r in cells if r["step"] == step and r["method"] == arm]
            metrics = {col: statistics.mean(r[col] for r in entries) for col in METRICS}
            averages.append({"step": step, "method": arm, "replicates": len(entries),
                             "selection_seeds": ",".join(str(r["seed"]) for r in entries) if arm != "pool" else "",
                             **metrics})
    return cells, averages, thresholds


def common_exposure(primary, groups, gold, freq, preds):
    folder = primary / "evaluation/test/base"
    expected = read_json(primary / "summary/checksums.json")
    actual = {}
    base, _, _ = read_model(folder, groups, digest(primary / "frozen/test.jsonl"), actual)
    if any(expected[k] != v for k, v in actual.items()):
        raise ValueError("The frozen base-model evaluation differs")
    cells = []
    for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
        for seed in SEEDS:
            counts = Counter()
            for k, name in gold.items():
                l, r = freq[seed, left][name] > 0, freq[seed, right][name] > 0
                state = "both" if l and r else "left_only" if l else "right_only" if r else "neither"
                counts[state, base[k], preds[seed, left][k], preds[seed, right][k]] += 1
            start = len(cells)
            for state in ("both", "left_only", "right_only", "neither"):
                tally = Counter()
                for (s, b, l, r), n in counts.items():
                    if s == state:
                        tally[l, r] += n
                        tally["base_correct"] += b*n
                both, l, r, neither = [tally[a, b] for a, b in ((True, True), (True, False), (False, True), (False, False))]
                size = both + l + r + neither
                cells.append({"left": left, "right": right, "seed": seed, "function_seen_by": state,
                              "requests": size, "base_correct": tally["base_correct"], "base_accuracy_pct": percentage(tally["base_correct"], size),
                              "both_correct": both, "only_left_correct": l, "only_right_correct": r, "both_wrong": neither,
                              "left_accuracy_pct": percentage(both + l, size), "right_accuracy_pct": percentage(both + r, size),
                              "net_left_minus_right": l-r, "difference_pp_on_stratum": percentage(l-r,size),
                              "contribution_to_full_difference_pp": percentage(l-r,len(gold))})
            block = cells[start:]
            if (sum(r["requests"] for r in block) != len(gold) or
                    sum(r["net_left_minus_right"] for r in block) != sum(preds[seed, left].values()) - sum(preds[seed, right].values())):
                raise ValueError("Shared strata do not reconcile to the paired full-test result")
    totals = []
    for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
        for state in ("both", "left_only", "right_only", "neither"):
            matched = [r for r in cells if (r["left"], r["right"], r["function_seen_by"]) == (left, right, state)]
            cols = ("requests","base_correct","both_correct","only_left_correct","only_right_correct","both_wrong","net_left_minus_right")
            rec = {c: sum(r[c] for r in matched) for c in cols}
            totals.append({"left":left,"right":right,"function_seen_by":state,**rec,
                           "base_accuracy_pct":percentage(rec["base_correct"],rec["requests"]),
                           "mean_full_test_contribution_pp":percentage(rec["net_left_minus_right"], len(gold)*len(SEEDS))})
    return cells, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", type=Path, default=Path("local/course-run"))
    parser.add_argument("--phase1", type=Path, default=Path("local/followup/phase1"))
    parser.add_argument("--out", type=Path, default=Path("local/followup/phase0/difficulty"))
    args = parser.parse_args()
    primary = args.primary
    _, freq, preds, selections, gold, groups, _, _, scores, _, _, _ = aggregate_runs(primary)
    lengths = {k: r["value_tokens"] for k, r in scores.items()}
    edges, indices = length_strata(np.array(list(lengths.values())))
    bins = dict(zip(lengths, map(int, indices)))
    add_e2(args.phase1, selections, bins)
    cells, averages, thresholds = training_metrics(primary, selections, lengths, bins)
    paired, totals = common_exposure(primary, groups, gold, freq, preds)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    save_table(out / "difficulty-by-seed.csv", cells)
    save_table(out / "difficulty-means.csv", averages)
    save_table(out / "difficulty-thresholds.csv", thresholds)
    save_table(out / "shared-exposure-by-seed.csv", paired)
    save_table(out / "shared-exposure-totals.csv", totals)
    exact_sizes = Counter(lengths.values())
    metadata = {"post_hoc": True, "input_splits": {"training_candidates": len(lengths), "locked_test_requests": len(gold)},
                "registered_e2_bin_upper_edges": edges, "number_of_exact_positive_lengths": len(exact_sizes) - int(0 in exact_sizes),
                "single_candidate_positive_length_strata": sum(n == 1 and value != 0 for value, n in exact_sizes.items()),
                "score_unit": "teacher-forced proxy NLL per scalar-value token averaged equally over candidates",
                "exact_residual": "candidate NLL minus full training pool mean NLL at its exact integer value-token count and step",
                "bin_residual": "candidate NLL minus full training pool mean at its registered E2 length bin and step",
                "top_decile": "exactly 10 percent of eligible mass; ties split equally across the cutoff; within-length uses this rule per exact length",
                "training_selections": "A/B/C/E2 seeds 42,43,44; historical pooled Bprime only at 42; E2 selection composition only",
                "coverage": "Each seed and pair shares requests within each exposure stratum. Totals reuse the same 2,000 requests three times."}
    (out / "diagnostic-definitions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(cells)} training-subset rows and {len(paired)} paired exposure strata to {out}.")


if __name__ == "__main__":
    main()
