"""Plot complete three-seed contrasts from summarize_followup.py's final followup.json."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


LABELS = {
    "title": "How did data selection change tool-call accuracy?",
    "subtitle": "Turnstile · {requests:,} previously examined test requests · {shown}/7 complete three-seed contrasts",
    "axis": "Complete-call accuracy difference (percentage points). Positive favors the first method.",
    "mean": "Mean",
    "question_interval": "95% paired bootstrap over request groups, conditional on the trained seeds",
    "seed_interval": "95% t interval across three paired seeds, conditional on the test requests (df = 2)",
    "contrasts": {
        "Bprime-B": "B′−B   Pooled fields − whole answer",
        "C-Bprime": "C−B′   Separate fields − pooled fields",
        "E2-B": "E2−B   Length-matched random − whole answer",
        "E2-A": "E2−A   Length-matched random − random",
        "E3-B": "E3−B   Residualized loss − whole answer",
        "E3-A": "E3−A   Residualized loss − random",
        "E4-A": "E4−A   Tool-stratified random − random",
    },
}
QUESTION_COLOR = "#165b83"
SEED_COLOR = "#a6531b"


def bounds(values, name):
    low, high = values
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError(f"Invalid interval for {name}: {values}")
    return low, high


def complete_rows(summary):
    expected = [42, 43, 44]
    if summary["seeds"] != expected or set(summary["contrasts"]) != set(LABELS["contrasts"]):
        raise ValueError("Expected the seven Phase 1 contrasts and paired seeds 42, 43, 44")
    rows = []
    for name in LABELS["contrasts"]:
        result = summary["contrasts"][name]
        if result["missing_seeds"] or result["seeds"] != expected:
            continue
        effects = [result["paired_by_seed"][str(s)]["difference_pp"] for s in expected]
        estimate = result["mean_difference_pp"]
        if not math.isclose(estimate, mean(effects), abs_tol=1e-8):
            raise ValueError(f"{name}: mean and three paired seed differences disagree")
        question_ci = bounds(result["fixed_seed_group_ci95_pp"], name)
        t = result["seed_t_sensitivity"]
        if t is not None and t["df"] != 2:
            raise ValueError(f"{name}: the seed interval must have two degrees of freedom")
        seed_ci = bounds(t["ci95_pp"], name) if t is not None else None
        rows.append((name, estimate, question_ci, seed_ci))
    if not rows:
        raise ValueError("No contrast has three complete paired seeds")
    return rows


def render(summary, destination, labels=LABELS, font_family=None):
    rows = complete_rows(summary)
    if set(labels) != set(LABELS) or set(labels["contrasts"]) != set(LABELS["contrasts"]):
        raise ValueError("The labels file must translate every figure label and all seven contrasts")
    height = 2.55 + 0.52 * len(rows)
    style = {"font.family": font_family} if font_family else {}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(11.2, height))
        try:
            fig.subplots_adjust(left=.424, right=.904, top=1 - .96 / height, bottom=1.36 / height)
            fig.text(.034, 1 - .20 / height, labels["title"], va="top", size=16, weight="bold")
            fig.text(.035, 1 - .53 / height, labels["subtitle"].format(
                requests=summary["requests"], shown=len(rows)), va="top", size=10, color="#3e4950")
            ax.axvline(0, color="#59676f", lw=1.0, ls=":", zorder=1)
            ax.grid(axis="x", lw=.55, color="#dce1e4")
            ax.set_axisbelow(True)
            for i, (_, estimate, question_ci, seed_ci) in enumerate(rows):
                ax.hlines(i - .11, *question_ci, colors=QUESTION_COLOR, lw=2.4, zorder=3)
                ax.plot(estimate, i - .11, "o", ms=5.5, color=QUESTION_COLOR, zorder=4)
                if seed_ci is not None:
                    ax.hlines(i + .11, *seed_ci, colors=SEED_COLOR, lw=1.6, linestyles="--", zorder=3)
                    ax.plot(estimate, i + .11, "s", ms=4.3, mfc="white", mec=SEED_COLOR, zorder=4)
                ax.text(1.033, i, f"{estimate:+.2f}".replace("-", "−"), transform=ax.get_yaxis_transform(),
                        size=10, va="center", ha="left", color="#26333a")
            ax.text(1.033, 1.03, labels["mean"], transform=ax.transAxes, size=9.5,
                    ha="left", va="bottom", weight="bold")
            ax.set_yticks(range(len(rows)), [labels["contrasts"][row[0]] for row in rows], fontsize=10)
            ax.set_ylim(len(rows) - .58, -.58)
            ax.tick_params(axis="y", length=0, pad=9)
            ax.tick_params(axis="x", labelsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3))
            ax.set_xlabel(labels["axis"], size=9.5, labelpad=6)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#839099")
            handles = [Line2D([], [], color=QUESTION_COLOR, lw=2.4, marker="o", ms=5,
                              label=labels["question_interval"])]
            if any(row[3] is not None for row in rows):
                handles.append(Line2D([], [], color=SEED_COLOR, lw=1.6, ls="--", marker="s", ms=4.3,
                                      mfc="white", label=labels["seed_interval"]))
            fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.03, .09 / height),
                       frameon=False, handlelength=3.1, fontsize=9, labelspacing=.45)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=210, facecolor="white")
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed Phase 1 summary/followup.json")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG, PDF or SVG path")
    parser.add_argument("--labels", type=Path, help="Optional JSON translation matching this module's LABELS")
    parser.add_argument("--font-family", help="Installed Matplotlib font family for localized text")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else LABELS
    try:
        render(json.loads(args.input.read_text(encoding="utf-8")), args.output, labels, args.font_family)
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
