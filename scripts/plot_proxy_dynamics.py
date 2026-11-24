"""Rebuild the proxy dynamics figure from raw score rows and verify the selector statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


SERIES = (
    ("all", "Whole answer", "#444b55", "o"),
    ("function", "Function name", "#16659a", "s"),
    ("value", "Argument value", "#b44f02", "^"),
)
THRESHOLD = 0.01


def summarize(run_dir):
    selection = json.loads((run_dir / "selection/42/selection.json").read_text())
    steps = selection["steps"]
    results = {key: [] for key, *_ in SERIES}
    pool_ids = None
    for i, step in enumerate(steps):
        raw = (run_dir / f"scores/step-{step}.jsonl").read_bytes()
        receipt = selection["receipts"][str(step)]
        if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
            raise ValueError(f"Score file hash differs at step {step}")
        rows = [json.loads(line) for line in raw.splitlines()]
        ids = {row["id"] for row in rows}
        if len(rows) != receipt["rows"] or len(ids) != len(rows) or (i and ids != pool_ids):
            raise ValueError(f"Score row IDs or sample size differ at step {step}")
        pool_ids = ids
        for key, *_ in SERIES:
            values = np.array([row[f"{key}_sum"] / row[f"{key}_tokens"]
                               for row in rows if row[f"{key}_tokens"] > 0])
            reference = selection["feature_statistics"][key][i]
            if len(values) != reference["count"] or len(rows) - len(values) != reference["missing"]:
                raise ValueError(f"Field coverage differs for {key} at step {step}")
            p10, median, p90 = np.quantile(values, [0.1, 0.5, 0.9])
            fraction = np.mean(values < THRESHOLD)
            np.testing.assert_allclose(
                [p10, median, p90, fraction],
                [*reference["p10_p50_p90"], reference["fraction_below_0.01"]],
                rtol=1e-10, atol=1e-12,
            )
            results[key].append((fraction, p10, median, p90))
    return steps, results, len(pool_ids)


def plot(steps, results, size, destination):
    fig, (share, spread) = plt.subplots(1, 2, figsize=(10.6, 4.6))
    x = np.arange(len(steps))
    for index, (key, name, color, marker) in enumerate(SERIES):
        fraction, low, median, high = np.asarray(results[key]).T
        share.plot(x, fraction, color=color, marker=marker, linewidth=2, label=name)
        # Horizontal offset distinguishes the three distributions at each checkpoint.
        spread.errorbar(x + (index - 1) * 0.17, median,
                        yerr=np.stack([median - low, high - median]), fmt=marker,
                        color=color, capsize=3, linewidth=1.6, markersize=5)
    share.set(title="How many requests have very low loss?", ylabel="Requests with mean loss < 0.01")
    share.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    share.set_ylim(0, 1.035)
    spread.set(title="How does loss vary across requests?", ylabel="Median loss and 10th–90th percentiles")
    spread.set_yscale("log")
    spread.axhline(THRESHOLD, color="#777777", linewidth=1, linestyle=":", zorder=0)
    for ax in (share, spread):
        ax.set_xticks(x, labels=steps)
        ax.set_xlabel("Optimizer step")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["right", "top"]].set_visible(False)
    fig.legend(*share.get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.92), frameon=False)
    fig.suptitle(f"Qwen3.5-0.8B proxy on {size:,} training candidates", fontsize=15, weight="bold")
    fig.subplots_adjust(top=.76, bottom=.19, left=.085, right=.98, wspace=.34)
    fig.text(.5, .025, "Per request: mean cross-entropy per supervised token in the indicated field. "
             "Checkpoints are spaced equally on the axis.", ha="center", fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("local/runpod/repo/local/course-run"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures/proxy-dynamics.png"))
    args = parser.parse_args()
    steps, results, size = summarize(args.run_dir)
    plot(steps, results, size, args.output)
    print("| Step | Whole <0.01 | Function <0.01 | Value <0.01 | Whole median | Function median | Value median |")
    print("|----:|----:|----:|----:|----:|----:|----:|")
    for i, step in enumerate(steps):
        table = [f"{results[k][i][0]:.2%}" for k, *_ in SERIES]
        table.extend(f"{results[k][i][2]:.6f}" for k, *_ in SERIES)
        print(f"| {step} | " + " | ".join(table) + " |")


if __name__ == "__main__":
    main()
