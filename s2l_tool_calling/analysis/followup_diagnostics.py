"""Post-hoc P0-1/2/3/5 diagnostics on the locked equal-token runs (CPU).

Example, from the checkout root:
  python -m s2l_tool_calling.analysis.followup_diagnostics register --out local/followup/phase0
  python -m s2l_tool_calling.analysis.followup_diagnostics analyze --out local/followup/phase0 \
      --run local/runpod/repo/local/course-run

Requires SciPy and Matplotlib for analysis. All output tables contain aggregates.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from summarize_course import digest, fixed_seed_summary, indexed, read_json, read_model, read_rows
from verify_tool_call_loss import gold_calls


SEEDS = (42, 43, 44)
ARMS = ("A", "B", "C", "Bprime")
FREQUENCY = ("0", "1", "2-3", "4+")
SEEN = ("both", "A_only", "C_only", "neither")
CONTRASTS = (("C", "B"), ("B", "A"), ("C", "A"))
REG = "p0-1235-preregistration.json"
SPEC = {
    "status": "Post-hoc analysis plan. Published main results were already known.",
    "runs": "Equal target tokens; matched A/B/C at 42,43,44; Bprime at 42 only.",
    "P0-1": "Unweighted Spearman rank correlation of subset mean length (value and whole answer, including zeros) "
            "with request accuracy across ten dependent runs. Descriptive threshold for value length: rho <= -0.60.",
    "P0-2": "Per run, stratify by its own gold function's training frequency: 0,1,2-3,4+. "
            "For each seed cross A/C seen status with both correct, A only correct, C only correct, neither. "
            "Describe net correct differences, without subgroup significance tests or causal attribution.",
    "P0-3": "Seed 42 positive-value B/Bprime/C clusters, plus their shared no-value clusters. "
            "Weight candidates equally in each cluster; nested means a scalar gold path at depth >1. "
            "Report size, quota, value length, nested share, unique functions, mean per-row value NLL at 250. "
            "For the smallest versus largest B clusters report all ties, without formal testing.",
    "P0-5": "For C-B, B-A, C-A, reproduce fixed-seed 2000-draw joint request-group bootstrap. "
            "Compute df=2 two-sided 95% t interval of the three paired seed differences on the same fixed requests. "
            "No independent-seed or joint seed-and-request coverage claim.",
}


def rows(path):
    return indexed(read_rows(path), "id", path)


def gold(row):
    calls = gold_calls(row)
    if len(calls) != 1:
        raise ValueError("Expected a single reference call")
    return calls[0]["function"]


def save_table(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def register(out):
    path = out / REG
    if path.exists():
        if read_json(path)["protocol"] != SPEC:
            raise ValueError("Existing registration uses a different protocol")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"registered_utc": datetime.now(timezone.utc).isoformat(),
                   "script_sha256": digest(Path(__file__)), "protocol": SPEC}, stream, indent=2)
        stream.write("\n")


def percentage(k, n):
    return 100 * k / n if n else None


def aggregate_runs(run):
    from field_diagnostics import leaves

    frozen = run / "frozen"
    train = rows(frozen / "train.jsonl")
    test = rows(frozen / "test.jsonl")
    train_digest = digest(frozen / "train.jsonl")
    test_digest = digest(frozen / "test.jsonl")
    if len(train) != 4000 or len(test) != 2000 or train.keys() & test.keys():
        raise ValueError("Unexpected train/test membership for this study")
    groups = {k: str(v["group_id"]) for k, v in test.items()}
    train_info = {k: gold(row) for k, row in train.items()}
    train_func = {k: ref["name"] for k, ref in train_info.items()}
    nested = {k: any(len(path) > 1 for path, _ in leaves(ref["arguments"])) for k, ref in train_info.items()}
    test_func = {k: gold(row)["name"] for k, row in test.items()}
    score_file = run / "scores/step-250.jsonl"
    scores = rows(score_file)
    receipt = read_json(run / "scores/step-250.complete.json")
    if (scores.keys() != train.keys() or receipt != {"step": 250, "rows": len(train), "sha256": digest(score_file)}
            or read_json(run / "scores/score-config.json")["train_sha256"] != train_digest):
        raise ValueError("Training score receipt or candidate pool mismatch")
    locked = read_json(run / "locked-models.json")["main"]
    checksums = read_json(run / "summary/checksums.json")
    summary = read_json(run / "summary/internal.json")
    official = summary["phases"]["main"]
    if (official["locked"] != locked or digest(run / "locked-models.json") != checksums["locked-models.json"]
            or test_digest != checksums["frozen_test_split"]):
        raise ValueError("Locked model or test source mismatch")
    preds, selection, freq, dose, manifests = {}, {}, {}, [], {}
    common_protocol = summary["internal_generation_protocol"]
    for seed in SEEDS:
        folder = run / "selection" / str(seed)
        manifest = read_json(folder / "selection.json")
        if (manifest["seed"] != seed or manifest["train_sha256"] != train_digest
                or manifest["receipts"]["250"] != receipt):
            raise ValueError(f"Selection receipt mismatch at seed {seed}")
        manifests[seed] = manifest
        for arm in (ARMS if seed == 42 else ARMS[:3]):
            label = f"s{seed}-{arm}"
            model = locked[label]
            subset = rows(folder / f"{arm}.jsonl")
            stated = manifest["subsets"][arm]
            if (len(subset) != 1000 or list(subset) != stated["ids"] or not subset.keys() <= train.keys()
                    or digest(folder / f"{arm}.jsonl") != stated["sha256"]):
                raise ValueError(f"Selection mismatch for {label}")
            actual = {}
            correct, metrics, protocol = read_model(run / "evaluation/test" / model, groups, test_digest, actual)
            if (any(checksums[k] != v for k, v in actual.items()) or protocol != common_protocol
                    or metrics["correct"] != summary["models"][model]["correct"]):
                raise ValueError(f"Test receipt mismatch for {label}")
            key = (seed, arm)
            preds[key] = correct
            selection[key] = set(subset)
            freq[key] = Counter(train_func[k] for k in subset)
            dose.append({"seed": seed, "method": arm, "selected": len(subset),
                         "mean_value_tokens": statistics.mean(scores[k]["value_tokens"] for k in subset),
                         "mean_answer_tokens": statistics.mean(scores[k]["all_tokens"] for k in subset),
                         "correct": metrics["correct"], "test_requests": len(test),
                         "accuracy_pct": metrics["accuracy_pct"]})
    return dose, freq, preds, selection, test_func, groups, train_func, nested, scores, manifests[42], official, summary["interval"]


def coverage(freq, preds, functions):
    strata, disagreements = [], []
    for (seed, arm), counts in freq.items():
        sizes, successes = Counter(), Counter()
        for k, name in functions.items():
            f = counts[name]
            band = "0" if not f else "1" if f == 1 else "2-3" if f <= 3 else "4+"
            sizes[band] += 1
            successes[band] += preds[seed, arm][k]
        for band in FREQUENCY:
            strata.append({"seed": seed, "method": arm, "gold_function_training_frequency": band,
                           "requests": sizes[band], "correct": successes[band],
                           "accuracy_pct": percentage(successes[band], sizes[band])})
    for seed in SEEDS:
        cells = Counter()
        for k, name in functions.items():
            a, c = freq[seed, "A"][name] > 0, freq[seed, "C"][name] > 0
            seen = "both" if a and c else "A_only" if a else "C_only" if c else "neither"
            cells[seen, preds[seed, "A"][k], preds[seed, "C"][k]] += 1
        for seen in SEEN:
            both, a, c, neither = (cells[seen, x, y] for x, y in [(True, True), (True, False), (False, True), (False, False)])
            n = both + a + c + neither
            disagreements.append({"seed": seed, "gold_function_seen_by": seen, "requests": n, "both_correct": both,
                                  "only_A_correct": a, "only_C_correct": c, "both_wrong": neither, "discordant": a + c,
                                  "net_A_minus_C_correct": a - c, "within_stratum_difference_pp": percentage(a - c, n),
                                  "contribution_to_full_test_A_minus_C_pp": percentage(a - c, len(functions))})
        if sum(r["net_A_minus_C_correct"] for r in disagreements if r["seed"] == seed) != (
                sum(preds[seed, "A"].values()) - sum(preds[seed, "C"].values())):
            raise ValueError("A/C net wins do not reconcile")
    return strata, disagreements


def cluster_profile(selection, train_func, nested, scores, manifest):
    result = []
    zero = {k for k, r in scores.items() if r["value_tokens"] == 0}
    common = selection[42, "B"] & zero
    for arm in ("B", "Bprime", "C"):
        if selection[42, arm] & zero != common:
            raise ValueError("No-value memberships differ between methods")
    for arm in ("B", "Bprime", "C", "shared_empty"):
        expected = zero if arm == "shared_empty" else scores.keys() - zero
        selected = common if arm == "shared_empty" else selection[42, arm] - zero
        seen = set()
        for i, cluster in enumerate(manifest["clustering"][arm]["clusters"], 1):
            members = set(cluster["ids"])
            n = len(members)
            if (n != cluster["size"] or seen & members or not members <= expected
                    or len(members & selected) != cluster["selected"]):
                raise ValueError("Invalid cluster membership or quota")
            seen.update(members)
            has_values = [scores[k] for k in members if scores[k]["value_tokens"]]
            n_nested = sum(nested[k] for k in members)
            result.append({"seed": 42, "method_or_shared_stratum": arm, "cluster": i, "candidates": n,
                           "selected": cluster["selected"], "selected_pct": percentage(cluster["selected"], n),
                           "mean_value_tokens": statistics.mean(scores[k]["value_tokens"] for k in members),
                           "nested_candidates": n_nested, "nested_pct": percentage(n_nested, n),
                           "distinct_gold_functions": len({train_func[k] for k in members}),
                           "candidates_with_value_loss": len(has_values),
                           "mean_candidate_value_nll_step250": statistics.mean(r["value_sum"] / r["value_tokens"]
                               for r in has_values) if has_values else None})
        if seen != expected:
            raise ValueError("Cluster partition is incomplete")
    return result


def intervals(preds, groups, official, config):
    from scipy.stats import t

    result = []
    for left, right in CONTRASTS:
        diff = {str(seed): percentage(sum(preds[seed, left].values()) - sum(preds[seed, right].values()), len(groups))
                for seed in SEEDS}
        saved = official["across_seed_differences_pp"][f"{left}-{right}"]
        vectors = [(preds[s, left], preds[s, right]) for s in SEEDS]
        computed = fixed_seed_summary(diff, vectors, groups, config["seed"])
        ci = computed["fixed_seed_mean_paired_cluster_ci95_pp"]
        if ci != saved["fixed_seed_mean_paired_cluster_ci95_pp"] or diff != saved["seed_differences_pp"]:
            raise ValueError("Primary bootstrap replication differs")
        mean = statistics.mean(diff.values())
        half = float(t.ppf(.975, df=2)) * statistics.stdev(diff.values()) / math.sqrt(len(diff))
        result.append({"contrast": f"{left}-{right}", **{f"seed_{s}_pp": diff[str(s)] for s in SEEDS},
                       "mean_pp": mean, "group_bootstrap_low_pp": ci[0], "group_bootstrap_high_pp": ci[1],
                       "seed_t_low_pp": mean-half, "seed_t_high_pp": mean+half, "seed_t_df": 2,
                       "requests": len(groups), "near_duplicate_groups": len(set(groups.values()))})
    return result


def plot(out, dose, correlations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"A": "#30638e", "B": "#b4442e", "C": "#137c70", "Bprime": "#8860a1"}
    marks = {42: "o", 43: "s", 44: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True, layout="constrained")
    for ax, (key, rho) in zip(axes, correlations.items()):
        for r in dose:
            ax.scatter(r[key], r["accuracy_pct"], s=46, color=colors[r["method"]], marker=marks[r["seed"]], alpha=.9)
        ax.set(xlabel=key.replace("_", " ").capitalize(), title=f"Spearman ρ = {rho:.2f}")
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Held-out exact-match accuracy (%)")
    lines = [Line2D([], [], color=c, lw=3, label=a) for a, c in colors.items()]
    lines += [Line2D([], [], color="0.28", marker=m, lw=0, label=str(s)) for s, m in marks.items()]
    fig.legend(handles=lines, loc="outside lower center", ncol=7, frameon=False)
    fig.savefig(out / "p01-length-accuracy.png", dpi=180)
    plt.close(fig)


def summary_text(report, clusters):
    rho = report["spearman_rho"]
    net = report["descriptive_net_A_minus_C_by_exposure"]
    saturated = [r for r in clusters if r["method_or_shared_stratum"] == "B" and r["selected"] == r["candidates"]]
    largest = report["B_cluster_size_extremes"]["largest"][0]
    lines = [
        "# Post-hoc diagnostics on the locked Turnstile test",
        "",
        "The ten equal-token checkpoints share the same 2,000 test requests and the same proxy. "
        "A, B, and C each contribute seeds 42–44. The pooled-field control B′ contributes seed 42 only.",
        "",
        "## Training subset length and accuracy (P0-1)",
        "",
        f"Across ten runs, Spearman’s ρ was {rho['mean_value_tokens']:.3f} for mean argument-value length "
        f"and {rho['mean_answer_tokens']:.3f} for mean answer length. Longer training subsets co-occurred "
        "with lower accuracy. Repeated methods and seeds make these ten points dependent. Compositional "
        "differences between methods could explain the pattern. The length-matched randomized control tests this next.",
        "",
        "## Coverage of the requested function (P0-2)",
        "",
        f"Across three evaluations of the same test, A had {net['A_only']:+d} net correct calls relative to C "
        f"on requests whose gold function occurred only in A’s subset, and {net['C_only']:+d} where it occurred "
        f"only in C’s subset. The corresponding differences were {net['both']:+d} where both saw the function "
        f"and {net['neither']:+d} where neither saw it. Each seed’s source table includes all concordant and "
        "discordant requests. The membership of a frequency band depends on the selected training set, so its "
        "accuracy describes a different mix of tools and requests for each method.",
        "",
        "## Composition of seed-42 training clusters (P0-3)",
        "",
        f"B selected all {sum(r['selected'] for r in saturated)} members of {len(saturated)} clusters. "
        f"Their mean value lengths ranged from {min(r['mean_value_tokens'] for r in saturated):.1f} to "
        f"{max(r['mean_value_tokens'] for r in saturated):.1f} tokens. Its largest cluster averaged "
        f"{largest['mean_value_tokens']:.1f} tokens, with {largest['selected']} of {largest['candidates']} "
        "members selected. Thus round-robin quotas enriched some long-value clusters. This documents a selection "
        "mechanism; the subsequent effect on accuracy requires an intervention.",
        "",
        "## Uncertainty in matched contrasts (P0-5)",
        "",
        "| Contrast | Mean (pp) | Fixed-seed request-group 95% interval | Fixed-test three-seed t interval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in report["paired_intervals"]:
        lines.append(f"| {r['contrast']} | {r['mean_pp']:+.2f} | [{r['group_bootstrap_low_pp']:+.2f}, "
                     f"{r['group_bootstrap_high_pp']:+.2f}] | [{r['seed_t_low_pp']:+.2f}, {r['seed_t_high_pp']:+.2f}] |")
    lines.extend(["", "The first interval jointly resamples request groups while holding the trained runs fixed. "
                  "The t interval holds the requests fixed and assumes the three paired seed differences represent "
                  "draws from an approximately normal distribution. With only two degrees of freedom and a shared "
                  "proxy, treat it as a fragile seed-sensitivity calculation. Neither interval combines both sources of uncertainty."])
    return "\n".join(lines) + "\n"


def analyze(run, out):
    from scipy.stats import spearmanr

    if read_json(out / REG)["protocol"] != SPEC:
        raise ValueError("Diagnostics protocol differs from the saved plan")
    dose, freq, preds, selection, tf, groups, train_func, nested, scores, manifest, official, cfg = aggregate_runs(run)
    correlations = {k: float(spearmanr([r[k] for r in dose], [r["accuracy_pct"] for r in dose]).statistic)
                    for k in ("mean_value_tokens", "mean_answer_tokens")}
    strata, disagreements = coverage(freq, preds, tf)
    clusters = cluster_profile(selection, train_func, nested, scores, manifest)
    ci = intervals(preds, groups, official, cfg)
    save_table(out / "p01-length-accuracy.csv", dose)
    save_table(out / "p02-function-frequency.csv", strata)
    save_table(out / "p02-ac-disagreement.csv", disagreements)
    save_table(out / "p03-cluster-profiles.csv", clusters)
    save_table(out / "p05-intervals.csv", ci)
    plot(out, dose, correlations)
    by_seen = {s: sum(r["net_A_minus_C_correct"] for r in disagreements if r["gold_function_seen_by"] == s) for s in SEEN}
    b = [r for r in clusters if r["method_or_shared_stratum"] == "B"]
    extreme = {label: [r for r in b if r["candidates"] == f(x["candidates"] for x in b)] for label, f in
               (("smallest", min), ("largest", max))}
    report = {"spearman_rho": correlations, "runs": len(dose), "descriptive_net_A_minus_C_by_exposure": by_seen,
              "B_cluster_size_extremes": extreme, "paired_intervals": ci, "bootstrap": cfg,
              "analysis_script_sha256": digest(Path(__file__)),
              "mean_pool_value_length_among_positive": statistics.mean(r["value_tokens"] for r in scores.values() if r["value_tokens"])}
    (out / "p0-1235-aggregates.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "p0-1235-summary.md").write_text(summary_text(report, clusters), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("register", "analyze"))
    p.add_argument("--run", type=Path, default=Path("local/course-run"), help="run directory containing frozen/, scores/, selection/ and evaluation/")
    p.add_argument("--out", type=Path, default=Path("local/followup/phase0"))
    args = p.parse_args()
    if args.run.resolve() in (args.out.resolve(), *args.out.resolve().parents):
        p.error("--out must be outside the locked run")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "register":
        register(args.out)
    else:
        analyze(args.run, args.out)


if __name__ == "__main__":
    main()
