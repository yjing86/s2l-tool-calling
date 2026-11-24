"""Build development accuracy versus incremental single-A40 time from completed run receipts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def metrics(root, name):
    path = root / "evaluation" / "dev" / name / "metrics.json"
    receipt = read(root / "receipts" / f"eval-dev-{name}.json")
    hashes = [value for key, value in receipt["outputs"].items()
              if key.endswith(f"/evaluation/dev/{name}/metrics.json")]
    if hashes != [hashlib.sha256(path.read_bytes()).hexdigest()]:
        raise ValueError(f"Development metrics did not match receipt: {name}")
    data = read(path)
    if not (data["count"] > 0 and 0 <= data["correct"] <= data["count"]):
        raise ValueError(f"Unexpected counts in {path}")
    return data, receipt["elapsed_s"]


def collect(root, seeds, methods, include_bprime):
    def seconds(name):
        return read(root / "receipts" / (name + ".json"))["elapsed_s"]

    base, base_cost = metrics(root, "base")
    proxy, score = seconds("proxy"), seconds("score")
    design = read(root / "design.json")
    if not set(seeds).issubset(design["seeds"]):
        raise ValueError("Requested seed outside frozen design")
    selected = {seed: seconds(f"select-{seed}") for seed in seeds}
    requested = [(s, group) for s in seeds for group in methods]
    if include_bprime:
        if not design["bprime"] or 42 not in seeds:
            raise ValueError("Bprime requires its predeclared seed 42")
        requested.append((42, "Bprime"))
    team = {"common_proxy_seconds": proxy, "common_scoring_seconds": score,
            "joint_selection_seconds_by_seed": selected, "common_base_dev_seconds": base_cost,
            "measured_target_training_seconds": 0.0, "measured_checkpoint_dev_seconds": 0.0}
    points = []
    for seed, method in requested:
        label = f"s{seed}-{method}"
        run = root / "training" / label
        final = int(read(run / "completed.json")["final_step"])
        train_cost = seconds("train-" + label)
        records = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
        if [r["step"] for r in records] != list(range(1, final + 1)):
            raise ValueError(f"Missing, repeated, or out of order step for {label}")
        if any(b["wall_seconds"] < a["wall_seconds"] for a, b in zip(records, records[1:])):
            raise ValueError(f"Non-monotonic step wall clock: {label}")
        checkpoints = [r for r in records if r["checkpoint_due"]]
        if [r["step"] for r in checkpoints] != sorted(set(range(50, final + 1, 50)) | {final}):
            raise ValueError(f"Unexpected checkpoint schedule for {label}")
        overhead = train_cost - records[-1]["wall_seconds"]
        if overhead < 0:
            raise ValueError(f"Training process finished before the final step: {label}")
        shared = proxy + score if method != "A" else 0.0
        team["measured_target_training_seconds"] += train_cost
        eval_cost = 0.0
        for record in [None, *checkpoints]:
            if record is None:
                step, low, high, selection, selection_shared, tokens = 0, 0., 0., 0., 0., 0
                data = base
            else:
                step = record["step"]
                data, spent = metrics(root, label + f"-step{step}")
                if data["count"] != base["count"]:
                    raise ValueError(f"Development denominator changed: {label} step {step}")
                eval_cost += spent
                team["measured_checkpoint_dev_seconds"] += spent
                low = train_cost if step == final else record["wall_seconds"]
                high = low if step == final else low + overhead
                selection, selection_shared = selected[seed], shared
                tokens = record["input_tokens"]
            points.append(dict(seed=seed, method=method, step=step, input_tokens=tokens,
                               final=step==final, correct=data["correct"], n=data["count"],
                               accuracy_pct=100*data["correct"]/data["count"],
                               proxy_and_score_s=selection_shared, selection_bundle_upper_s=selection,
                               train_lower_s=low, train_upper_s=high, dev_cumulative_s=eval_cost,
                               route_lower_s=selection_shared+low+eval_cost,
                               route_upper_s=selection_shared+selection+high+eval_cost))
    team["included_team_seconds"] = (proxy + score + sum(selected.values()) + base_cost
        + team["measured_target_training_seconds"] + team["measured_checkpoint_dev_seconds"])
    return points, team


NOTES = {
    "time": "Single-GPU elapsed wall seconds, including CPU preprocessing while the GPU is reserved. Reconstructed per-route cost, not the original chronological queue or an equal-cost controlled experiment.",
    "baseline": "Step 0 is the same scored base model in all routes. Time is incremental after the common base dev exam; add common_base_dev_seconds to charge this once per standalone route.",
    "selection": "A needs no model-based selection. B/C/Bprime each require proxy and score. Per-method selection time was not separately timed: the lower bound excludes this CPU job; the upper charges the entire jointly measured selection invocation for that seed.",
    "train": "At intermediate checkpoints lower is callback elapsed since on_train_begin; upper adds all unallocated launch/teardown overhead from the full-run receipt. The final checkpoint uses the complete training receipt in both bounds.",
    "dev": "Each point includes all its preceding checkpoint dev receipts plus its own, each including model load and generation. All checkpoints were generated after finishing target training; the curve reconstructs incremental sequential evaluation.",
    "team": "Included shared proxy, score, common base evaluation and each joint selector count only once across the requested routes. Training and all checkpoint exams count once per included arm. Excludes setup, smoke, benchmark, prompt preparation, retries, idle time, blind tests and optional reference jobs; not the total rental bill.",
    "points": "Points show actual completed development evaluations. Segments only connect observed checkpoints; thresholds between points are not measured."
}


def thresholds(points):
    out = []
    for seed, method in dict.fromkeys((r["seed"], r["method"]) for r in points):
        route = [r for r in points if r["seed"] == seed and r["method"] == method]
        target = min(100.0, route[0]["accuracy_pct"] + 3.0)
        first = next((r for r in route if r["accuracy_pct"] >= target), None)
        out.append({"seed": seed, "method": method, "target_pct": target,
                    "status": "observed" if first else "not reached at measured checkpoints",
                    "first_observed_step": first["step"] if first else None,
                    "observed_accuracy_pct": first["accuracy_pct"] if first else None,
                    "route_lower_s": first["route_lower_s"] if first else None,
                    "route_upper_s": first["route_upper_s"] if first else None})
    return out


def save_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(points, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    seeds = sorted({r["seed"] for r in points})
    colors = {"A":"#3368a0", "B":"#bd6421", "C":"#15825f", "Bprime":"#8655a5"}
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.4*len(seeds),4.5), sharex=True, sharey=True, squeeze=False)
    labels = {"A":"A · Random", "B":"B · Whole", "C":"C · Fields", "Bprime":"B′ · Pooled"}
    for ax, seed in zip(axes[0], seeds):
        for method, color in colors.items():
            rows = [r for r in points if r["seed"] == seed and r["method"] == method]
            if not rows:
                continue
            x = [r["route_upper_s"]/3600 for r in rows]
            y = [r["accuracy_pct"] for r in rows]
            ax.plot(x,y,marker="o",ms=4,lw=1.6,color=color)
            ax.hlines(y,[r["route_lower_s"]/3600 for r in rows],x,color=color,lw=2.2)
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Incremental A40 hours")
        ax.grid(alpha=.22)
    axes[0][0].set_ylabel("Complete-call accuracy on 400 dev items (%)")
    fig.suptitle("Development accuracy against each method’s running cost")
    handles = [Line2D([0],[0],color=color,marker="o",label=labels[m]) for m,color in colors.items()
               if any(r["method"]==m for r in points)]
    fig.legend(handles=handles,loc="lower center",ncol=len(handles),frameon=False)
    fig.tight_layout(rect=(0,.09,1,.95))
    fig.savefig(path,dpi=170,bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--course", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    p.add_argument("--include-bprime", action="store_true")
    p.add_argument("--plot", action="store_true")
    a=p.parse_args()
    if len(a.seeds) != len(set(a.seeds)):
        p.error("seeds must be unique")
    points,team=collect(a.course, a.seeds, ["A","B","C"], a.include_bprime)
    a.out.mkdir(exist_ok=True,parents=True)
    first_observed = thresholds(points)
    save_rows(a.out / "dev-cost.csv", points)
    save_rows(a.out / "dev-first-observed.csv", first_observed)
    (a.out / "dev-cost.json").write_text(json.dumps({
        "protocol": NOTES, "seeds": a.seeds, "team_scope": team,
        "points": points, "first_observed_threshold": first_observed,
    }, indent=2) + "\n")
    if a.plot:
        plot(points,a.out/"dev-cost.png")


if __name__=="__main__":
    main()
