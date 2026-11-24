"""Update the P0-1 figure from Phase 1 selection manifests and a terminal aggregate.

From the checkout root:
  python -m s2l_tool_calling.analysis.plot_followup_lengths build --summary results/followup/phase1-results.json --out results/followup/phase1-length-accuracy.csv
  python -m s2l_tool_calling.analysis.plot_followup_lengths plot --table results/followup/phase1-length-accuracy.csv --out docs/figures/phase1-length-accuracy.png

The second command needs only the public CSV, SciPy and Matplotlib.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from scripts.summarize_course import digest, read_json


SEEDS = (42, 43, 44)
METHODS = ("A", "B", "Bprime", "C", "E2", "E3", "E4")
ORIGINAL = {f"s{s}-{m}" for s in SEEDS for m in ("A", "B", "C")} | {"s42-Bprime"}
COLORS = dict(zip(METHODS, ("#30638e", "#b4442e", "#8860a1", "#137c70", "#bb7600", "#794330", "#415056")))
MARKS = {42: "o", 43: "s", 44: "^"}
ENGLISH = {"title": "Training subset length and held-out accuracy",
           "value": "Mean argument-value tokens per selected request",
           "answer": "Mean answer tokens per selected request", "accuracy": "Complete-call accuracy (%)",
           "rho": "Spearman ρ = {rho:.3f}", "fits": "{n}/21 trained endpoints  ·  2,000 shared test requests",
           "method": "Method", "seed": "Seed", "missing": "Missing fits"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build(summary_file, selection_dir, registration_file, original_csv):
    summary = read_json(summary_file)
    registration = read_json(registration_file)
    master = read_json(selection_dir / "manifest.json")
    spec = registration["spec"]
    require(summary["status"] in ("complete", "partial_budget"), "Expected a terminal Phase 1 result")
    require(summary["input_sha256"]["preregistration"] == registration["private_registration_sha256"]
            and digest(selection_dir / "manifest.json") == spec["selection_manifest_sha256"],
            "Result and selection tree must match the public Phase 1 registration")
    expected = {f"s{s}-{m}" for s in SEEDS for m in METHODS}
    done, missing = set(summary["completed_new_cells"]), set(summary["missing_new_cells"])
    require(not done & missing and done | missing == expected - ORIGINAL
            and set(summary["models"]) == done | ORIGINAL
            and (summary["status"] == "complete") == (not missing), "Inconsistent completed models")
    require(summary["requests"] == 2000 and summary["seeds"] == list(SEEDS), "Unexpected test design")
    reference = {f"s{r['seed']}-{r['method']}": r for r in read_csv(original_csv)}
    require(set(reference) == ORIGINAL, "Expected the original ten P0-1 points")
    table = []
    for seed in SEEDS:
        path = selection_dir / str(seed) / "selection.json"
        require(digest(path) == master["selection_manifest_sha256"][str(seed)], f"Seed {seed}: manifest changed")
        manifest = read_json(path)
        for method in METHODS:
            label = f"s{seed}-{method}"
            sub = (manifest["primary_controls"] if method in ("A", "B", "C") else manifest["subsets"])[method]
            n, totals = sub["requests"], sub["token_totals"]
            require(n == 1000 and totals["all_tokens"] == sum(totals[k] for k in ("value_tokens","function_tokens","other_tokens"))
                    and totals["all_tokens"] == sub["answer_tokens_total"], f"{label}: inconsistent token counts")
            x, y = totals["value_tokens"] / n, totals["all_tokens"] / n
            model = summary["models"].get(label)
            if model is not None:
                k = model["correct"]
                require(model["requests"] == 2000 and type(k) is int and 0 <= k <= 2000
                        and math.isclose(100*k/2000, model["accuracy_pct"], abs_tol=1e-12),
                        f"{label}: inconsistent test score")
                if label in reference:
                    old = reference[label]
                    require((int(old["selected"]),float(old["mean_value_tokens"]),float(old["mean_answer_tokens"]),
                             int(old["correct"]),int(old["test_requests"]),float(old["accuracy_pct"])) ==
                            (n,x,y,k,2000,model["accuracy_pct"]), f"{label}: differs from original P0-1 point")
            table.append({"seed": seed, "method": method, "origin": "primary" if label in ORIGINAL else "followup",
                          "status": "observed" if model is not None else "missing", "selected": n,
                          "mean_value_tokens": x, "mean_answer_tokens": y, "correct": model["correct"] if model else None,
                          "test_requests": 2000, "accuracy_pct": model["accuracy_pct"] if model else None})
    return table


def correlations(rows):
    from scipy.stats import spearmanr
    accuracy = [float(r["accuracy_pct"]) for r in rows]
    return {col: float(spearmanr([float(r[col]) for r in rows], accuracy).statistic)
            for col in ("mean_value_tokens", "mean_answer_tokens")}


def render(table, out, labels, font):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    expected = {(s, m) for s in SEEDS for m in METHODS}
    require(len(table) == len(expected) and {(int(r["seed"]), r["method"]) for r in table} == expected,
            "Expected all 21 planned slots, with missing fits labeled")
    observed = [r for r in table if r["status"] == "observed"]
    missing = [f"{r['method']}/{r['seed']}" for r in table if r["status"] == "missing"]
    require(len(observed) + len(missing) == len(table), "Invalid fit status")
    original = [r for r in observed if r["origin"] == "primary"]
    require(len(original) == 10, "Missing historical P0-1 points")
    rho = correlations(observed)
    with plt.rc_context({"font.family": font or "DejaVu Sans", "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10.4,4.8), sharey=True)
        try:
            fig.subplots_adjust(left=.085, right=.98, top=.78, bottom=.32, wspace=.14)
            fig.suptitle(labels["title"], x=.085, ha="left", y=.99, fontsize=15, weight="bold")
            fig.text(.085, .90, labels["fits"].format(n=len(observed)), size=10, color="#40515c")
            for ax, (col, value), axis_label in zip(axes, rho.items(), ("value","answer")):
                for r in observed:
                    ax.scatter(float(r[col]), float(r["accuracy_pct"]), s=57, marker=MARKS[int(r["seed"])],
                               facecolor=COLORS[r["method"]], edgecolor="white", linewidth=.65, alpha=.92)
                ax.set_xlabel(labels[axis_label], fontsize=10)
                ax.set_title(labels["rho"].format(rho=value), loc="left", size=11)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y",alpha=.23)
                ax.set_axisbelow(True)
            axes[0].set_ylabel(labels["accuracy"])
            methods = [Line2D([],[],lw=3,color=c,label=m.replace("Bprime","B′")) for m,c in COLORS.items()]
            seeds = [Line2D([],[],lw=0,marker=m,color="#44515a",label=str(s)) for s,m in MARKS.items()]
            for handles, title, ypos in [(methods, "method", .115), (seeds, "seed", .045)]:
                fig.text(.085,ypos,labels[title],va="center",size=9,weight="bold")
                fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.155,ypos),ncol=len(handles),
                           handlelength=1.6,frameon=False,fontsize=9)
            if missing:
                fig.text(.085,.19,labels["missing"] + " · " + ", ".join(missing),size=8)
            out.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(out,dpi=190,facecolor="white")
        finally:
            plt.close(fig)
    print(json.dumps({"historical_10": correlations(original), "observed_fits": len(observed),
                      "all_observed": rho, "missing": missing}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build", help="Join registered aggregate token counts with a verified terminal result.")
    b.add_argument("--summary", required=True, type=Path)
    b.add_argument("--selection", type=Path, default=Path("local/followup/phase1/selection"))
    b.add_argument("--registration", type=Path, default=Path("results/followup/phase1-registration.json"))
    b.add_argument("--original", type=Path, default=Path("results/followup/p01-length-accuracy.csv"))
    p = sub.add_parser("plot", help="Make the two-panel plot from the aggregate CSV.")
    p.add_argument("--table", required=True, type=Path)
    p.add_argument("--labels", type=Path, help="Optional localized label mapping.")
    p.add_argument("--font-family")
    for item in (b,p):
        item.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "build":
        table = build(args.summary, args.selection, args.registration, args.original)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    else:
        render(read_csv(args.table), args.out, read_json(args.labels) if args.labels else ENGLISH, args.font_family)


if __name__ == "__main__":
    main()
