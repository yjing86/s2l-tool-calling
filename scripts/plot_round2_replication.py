"""Plot the same paired models on the development, original and reserved splits."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "results/followup"
    legacy = json.loads((folder / "round2-legacy.json").read_text())["splits"]
    reserved = json.loads((folder / "round2-holdout.json").read_text())
    seeds = [str(i) for i in range(42, 47)]
    panels = []
    for split, label in (("dev", "Development · 400"), ("test", "Original test · 2,000")):
        pairs = legacy[split]["paired_contrasts"]["C minus B"]["by_seed"]
        panels.append((label, [pairs[s]["difference_pp"] for s in seeds]))
    pairs = reserved["contrasts"]["C-B"]["paired_by_seed"]
    panels.append(("Reserved test · 1,500", [pairs[s]["difference_pp"] for s in seeds]))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True, layout="constrained")
    for ax, (label, differences) in zip(axes, panels):
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.axhline(0, color="0.4", linewidth=0.9)
        ax.axhline(mean(differences), color="#ad4c10", linestyle="--", linewidth=1.4,
                   label="Five-seed mean")
        ax.scatter(seeds, differences, color="#245b8c", s=42, zorder=3)
        ax.set_title(label)
        ax.set_xlabel("Paired seed")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[0].set_ylabel("C minus B accuracy (percentage points)")
    axes[-1].legend(loc="best", frameon=True, fontsize=8)
    fig.suptitle("Field trajectories versus whole-answer trajectories")
    fig.savefig(ROOT / "docs/figures/round2-replication.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
