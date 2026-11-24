"""Describe same-seed overlap and step-zero field-loss coverage using training inputs only."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import quantiles


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as f:
        rows = list(map(json.loads, f))
    ids = {str(row["id"]): row for row in rows}
    if len(rows) != len(ids):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def describe(train, run):
    pool = set(read_rows(train))
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
    for seed in (42,43,44):
        root = run / "selection" / str(seed)
        selection = json.loads((root/"selection.json").read_text())
        if selection["seed"] != seed or selection["train_sha256"] != source or selection["receipts"]["0"] != receipt:
            raise ValueError(f"Selection manifest mismatch at seed {seed}")
        ids = {}
        coverage = {}
        for arm in ("A","B","C"):
            path = root/f"{arm}.jsonl"
            ids[arm] = set(read_rows(path))
            manifest_arm = selection["subsets"][arm]
            if (ids[arm] != set(manifest_arm["ids"]) or not ids[arm] <= pool
                    or len(ids[arm]) != selection["size"] or digest(path) != manifest_arm["sha256"]):
                raise ValueError(f"Selected IDs or hash mismatch: {seed}/{arm}")
            hits = len(ids[arm] & subset)
            coverage[arm] = {"selected": len(ids[arm]), "hits": hits,
                             "share_of_selection": hits/len(ids[arm]), "recall_of_slice": hits/len(subset) if subset else None}
        overlap = {}
        for a,b in combinations(ids, 2):
            n = len(ids[a] & ids[b])
            union = len(ids[a] | ids[b])
            overlap[a+"-"+b] = {"intersection": n, "union": union, "jaccard": n/union}
        report["seeds"][seed] = {"coverage": coverage, "overlap": overlap}
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--train", type=Path, default=Path("local/prepared/train.jsonl"))
    cli.add_argument("--run", type=Path, default=Path("local/runpod/repo/local/course-run"))
    args = cli.parse_args()
    print(json.dumps(describe(args.train, args.run), indent=2, allow_nan=False))
