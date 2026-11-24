"""Plot primary held-out results from summarize_course.py's aggregate internal.json."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


ARMS = (
    ("A", "A · Random", "#444b55", "o"),
    ("B", "B · Whole answer", "#16659a", "s"),
    ("C", "C · Separate fields", "#b44f02", "^"),
)


def render(summary, destination):
    main = summary["phases"]["main"]
    models = summary["models"]
    locked = main["locked"]
    seeds = sorted(main["paired_by_seed"], key=int)
    fixed = main["across_seed_differences_pp"]["C-B"]
    if set(seeds) != set(fixed["seed_differences_pp"]) or len(seeds) != fixed["seeds"]:
        raise ValueError("Fixed-seed mean and individual paired contrasts use different seeds")
    base = models[main["reference_base"]]["accuracy_pct"]
    accuracy = {arm: [models[locked[f"s{s}-{arm}"]]["accuracy_pct"] for s in seeds] for arm, *_ in ARMS}
    points, intervals = [], []
    for i, seed in enumerate(seeds):
        pair = main["paired_by_seed"][seed]["C-B"]
        difference = pair["difference_pp"]
        if not math.isclose(difference, accuracy["C"][i] - accuracy["B"][i], abs_tol=1e-8):
            raise ValueError(f"Seed {seed}: absolute accuracies and paired contrast disagree")
        points.append(difference)
        intervals.append(pair["paired_cluster_ci95_pp"])
    if not math.isclose(fixed["mean"], sum(points) / len(points), abs_tol=1e-8):
        raise ValueError("Fixed-seed mean is inconsistent with paired contrasts")
    points.append(fixed["mean"])
    intervals.append(fixed["fixed_seed_mean_paired_cluster_ci95_pp"])

    fig, (scores, effects) = plt.subplots(1, 2, figsize=(10.8, 4.6))
    positions = list(range(len(seeds)))
    for index, (arm, label, color, marker) in enumerate(ARMS):
        scores.scatter([x + (index - 1) * .15 for x in positions], accuracy[arm],
                       label=label, color=color, marker=marker, s=55, zorder=3)
    scores.axhline(base, color="#67757e", linestyle="--", linewidth=1.4, label=f"Base 2B ({base:.2f}%)")
    scores.set_xticks(positions, labels=seeds)
    scores.set_xlim(-.55, len(seeds) - .45)
    scores.yaxis.set_major_formatter(PercentFormatter(100))
    scores.set(title="Complete-call accuracy", xlabel="Selection and training seed", ylabel="Requests answered correctly")
    scores.grid(axis="y", alpha=.22)
    scores.legend(loc="upper center", bbox_to_anchor=(.5, 1.33), ncol=2, frameon=False, fontsize=9)

    labels = [f"Seed {s}" for s in seeds] + [f"Mean ({len(seeds)} fixed seeds)"]
    effects.axvline(0, color="#67757e", linestyle="--", linewidth=1.3)
    for i, (estimate, (low, high)) in enumerate(zip(points, intervals)):
        color = "#b44f02" if i == len(seeds) else "#16659a"
        effects.hlines(i, low, high, color=color, linewidth=1.7)
        effects.plot([low, high], [i, i], "|", color=color, markersize=8)
        effects.plot(estimate, i, "D" if i == len(seeds) else "o", color=color, markersize=6)
    effects.set_yticks(range(len(labels)), labels=labels)
    effects.set_ylim(len(labels) - .5, -.5)
    effects.set(title="Separate fields minus whole answer (C−B)", xlabel="Difference in percentage points")
    effects.grid(axis="x", alpha=.22)
    for ax in (scores, effects):
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"Qwen3.5-2B on {summary['test_requests']:,} held-out single-call requests",
                 y=.99, fontsize=15, weight="bold")
    fig.subplots_adjust(top=.71, bottom=.19, left=.08, right=.98, wspace=.53)
    fig.text(.5, .025, "Equal target training-token budget. Intervals are paired 95% bootstrap intervals over held-out "
             "request groups, conditional on the trained seeds.", ha="center", fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Final aggregate summary/internal.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.input.read_text()), args.output)


if __name__ == "__main__":
    main()
