"""Describe selection overlap within and across seeds, and step-zero field-loss coverage using training inputs only."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median, quantiles


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as f:
        rows = list(map(json.loads, f))
    ids = {str(row["id"]): row for row in rows}
    if len(rows) != len(ids):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def overlap_counts(left, right):
    intersection = len(left & right)
    union = len(left | right)
    return {"intersection": intersection, "union": union, "jaccard": intersection / union}


def coverage_slice(members, cohort):
    hits = len(members & cohort)
    return {"requests": hits, "cohort_requests_in_pool": len(cohort),
            "share_of_group": hits / len(members) if members else None,
            "share_of_pool_cohort_covered": hits / len(cohort) if cohort else None}


def composition(members, tools, value_tokens, pool_counts, long_values, rare_tools):
    counts = Counter(tools[i] for i in members)
    lengths = sorted(value_tokens[i] for i in members)
    deciles = quantiles(lengths, n=10, method="inclusive") if len(lengths) >= 2 else lengths * 9
    rare_pool = {i for i, name in tools.items() if name in rare_tools}
    return {"requests": len(members), "unique_tools": len(counts), "pool_unique_tools": len(pool_counts),
            "fraction_of_pool_tools": len(counts) / len(pool_counts),
            "tool_request_counts": dict(sorted(counts.items())),
            "value_token_lengths": {"unit": "tokens on all argument value spans per request",
                                    "requests": len(lengths), "sum": sum(lengths),
                                    "zero": sum(length == 0 for length in lengths),
                                    "mean": mean(lengths) if lengths else None,
                                    "p10": deciles[0] if deciles else None,
                                    "median": median(lengths) if lengths else None,
                                    "p90": deciles[-1] if deciles else None,
                                    "max": max(lengths, default=None)},
            "long_values": coverage_slice(members, long_values),
            "rare_tools": {"pool_unique_tools": len(rare_tools),
                           "unique_tools_represented": len(rare_tools.intersection(counts)),
                           **coverage_slice(members, rare_pool)}}


def describe(train, run):
    data = read_rows(train)
    pool = set(data)
    scorefile = run / "scores/step-0.jsonl"
    score = read_rows(scorefile)
    receipt = json.loads(scorefile.with_suffix(".complete.json").read_text())
    source = digest(train)
    meta = json.loads((run / "scores/score-config.json").read_text())
    if (meta["train_sha256"] != source or set(score) != pool or receipt["rows"] != len(pool)
            or digest(scorefile) != receipt["sha256"] or receipt["step"] != 0):
        raise ValueError("Training pool and scoring receipt disagree")
    value_loss = {i: r["value_sum"]/r["value_tokens"] for i, r in score.items() if r["value_tokens"]}
    q75 = quantiles(value_loss.values(), n=4, method="inclusive")[2]
    low_fn = {i for i,r in score.items() if r["function_sum"]/r["function_tokens"] < .01}
    high_value = {i for i,m in value_loss.items() if m >= q75}
    subset = low_fn & high_value
    report = {"train_sha256": source, "score_sha256": receipt["sha256"], "pool": len(pool),
              "step": 0, "function_loss_cutoff_exclusive": .01, "value_q75_inclusive": q75,
              "n_with_values": len(value_loss), "n_low_function": len(low_fn),
              "n_high_value": len(high_value), "n_meeting_both": len(subset), "seeds": {}}
    tools = {}
    for i, row in data.items():
        calls = row["messages"][-1]["tool_calls"]
        if len(calls) != 1:
            raise ValueError(f"Expected one reference call on training ID {i}")
        tools[i] = calls[0]["function"]["name"]
    pool_counts = Counter(tools.values())
    lengths = {i: r["value_tokens"] for i, r in score.items()}
    if any(type(n) is not int or n < 0 for n in lengths.values()):
        raise ValueError("Invalid count of argument-value tokens")
    cutoff = quantiles(lengths.values(), n=10, method="inclusive")[-1]
    long_values = {i for i, n in lengths.items() if n >= cutoff}
    rare = {t for t, n in pool_counts.items() if n <= 2}
    def describe_group(members):
        return composition(members, tools, lengths, pool_counts, long_values, rare)

    report["composition_rules"] = {"value_length_cutoff": cutoff, "long_values": "pool P90 or above, including ties",
                                   "rare_tools": "tools occurring in at most 2 of all candidate requests",
                                   "quantiles": "inclusive linear interpolation"}
    report["pool_composition"] = describe_group(pool)
    selected = {}
    for seed in (42,43,44):
        root = run / "selection" / str(seed)
        selection = json.loads((root/"selection.json").read_text())
        if selection["seed"] != seed or selection["train_sha256"] != source or selection["receipts"]["0"] != receipt:
            raise ValueError(f"Selection manifest mismatch at seed {seed}")
        ids = {}
        coverage = {}
        for arm in ("A","B","C","Bprime"):
            path = root/f"{arm}.jsonl"
            subset_data = read_rows(path)
            ids[arm] = set(subset_data)
            manifest_arm = selection["subsets"][arm]
            if (ids[arm] != set(manifest_arm["ids"]) or not ids[arm] <= pool
                    or len(ids[arm]) != selection["size"] or digest(path) != manifest_arm["sha256"]):
                raise ValueError(f"Selected IDs or hash mismatch: {seed}/{arm}")
            if any(data[i] != row for i, row in subset_data.items()):
                raise ValueError(f"Selected row differs from training pool: {seed}/{arm}")
            if Counter(tools[i] for i in ids[arm]) != manifest_arm["tool_counts"]:
                raise ValueError(f"Tool counts disagree with selector: {seed}/{arm}")
            hits = len(ids[arm] & subset)
            coverage[arm] = {"selected": len(ids[arm]), "hits": hits,
                             "share_of_selection": hits/len(ids[arm]), "recall_of_slice": hits/len(subset) if subset else None}
        selected[seed] = ids
        overlap = {f"{a}-{b}": overlap_counts(ids[a], ids[b]) for a, b in combinations(ids, 2)}
        groups = {**ids, "B-only": ids["B"] - ids["C"], "C-only": ids["C"] - ids["B"]}
        report["seeds"][seed] = {"coverage": coverage, "overlap": overlap,
                                 "selection_manifest_sha256": digest(root / "selection.json"),
                                 "subset_sha256": {arm: selection["subsets"][arm]["sha256"] for arm in ids},
                                 "composition": {arm: describe_group(members) for arm, members in groups.items()}}
    report["same_arm_across_seeds"] = {
        arm: {f"{a}-{b}": overlap_counts(selected[a][arm], selected[b][arm])
              for a, b in combinations(selected, 2)}
        for arm in ("A", "B", "C")
    }
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--train", type=Path, default=Path("local/prepared/train.jsonl"))
    cli.add_argument("--run", type=Path, default=Path("local/runpod/repo/local/course-run"))
    args = cli.parse_args()
    print(json.dumps(describe(args.train, args.run), indent=2, allow_nan=False))
