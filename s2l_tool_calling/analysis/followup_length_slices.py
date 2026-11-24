"""Slice completed Turnstile test outcomes by the token length of reference argument values.

Run with python -m s2l_tool_calling.analysis.followup_length_slices --help.
The tokenizer runs on CPU. No model weights, generation, or training are needed.
"""

import argparse
from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, quantiles
import sys
from tempfile import NamedTemporaryFile

SOURCE = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SOURCE))
from summarize_course import digest, fixed_seed_summary, indexed, read_json, read_model, read_rows
from verify_tool_call_loss import encode

SEEDS = (42, 43, 44)
METHODS = ("A", "B", "C")
COMPARISONS = (("B", "A"), ("C", "B"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_tokens(tokenizer, row):
    _, masks = encode(tokenizer, row)
    return len(masks["value"])


def reference_lengths(course, model):
    from transformers import AutoTokenizer

    train = course / "frozen/train.jsonl"
    test = course / "frozen/test.jsonl"
    scores = course / "scores/step-0.jsonl"
    metadata = read_json(course / "scores/score-config.json")
    final = read_json(course / "scores/step-0.complete.json")
    reference = read_json(course / "evaluation/test/base/config.json")
    require(
        digest(model / "tokenizer.json") == reference["tokenizer_sha256"],
        "Tokenizer differs from evaluation",
    )
    require(
        digest(model / "chat_template.jinja") == reference["template_sha256"],
        "Chat template differs from evaluation",
    )
    require(
        digest(SOURCE / "verify_tool_call_loss.py") == metadata["mask_code_sha256"],
        "Field mask implementation changed",
    )
    require(digest(train) == metadata["train_sha256"], "Training reference data changed")
    require(digest(test) == reference["source_sha256"], "Test reference data changed")
    require(final["step"] == 0 and digest(scores) == final["sha256"], "Invalid proxy scoring receipt")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    require(tokenizer.is_fast, "Character-offset masks require a fast tokenizer")
    require(
        tokenizer.chat_template == (model / "chat_template.jinja").read_text(),
        "Active chat template differs from disk",
    )
    recorded = indexed(read_rows(scores), "id", scores)
    train_rows = indexed(read_rows(train), "id", train)
    require(
        set(train_rows) == set(recorded) and len(recorded) == final["rows"] == 4000,
        "Training calibration IDs",
    )
    for key, row in train_rows.items():
        require(
            count_tokens(tokenizer, row) == recorded[key]["value_tokens"],
            f"Argument-value calibration failed at {key}",
        )
    test_rows = indexed(read_rows(test), "id", test)
    require(
        len(test_rows) == 2000 and not set(test_rows) & set(train_rows),
        "Test membership or train/test overlap",
    )
    lengths = {key: count_tokens(tokenizer, row) for key, row in test_rows.items()}
    groups = {}
    for key, row in test_rows.items():
        require(row["group_id"] is not None and str(row["group_id"]) != "", "Missing near-duplicate group")
        groups[key] = str(row["group_id"])
    provenance = {
        "tokenizer_sha256": reference["tokenizer_sha256"],
        "template_sha256": reference["template_sha256"],
        "field_mask_sha256": metadata["mask_code_sha256"],
        "train_sha256": metadata["train_sha256"],
        "test_sha256": reference["source_sha256"],
        "training_calibration_rows": len(train_rows),
        "training_calibration_value_tokens": sum(r["value_tokens"] for r in recorded.values()),
        "proxy_score_sha256": final["sha256"],
        "target_revision": reference["model_revision"],
    }
    return lengths, groups, provenance


def load_outcomes(course, groups, provenance):
    lock = read_json(course / "locked-models.json")["main"]
    identities = read_json(course / "model-paths.json")
    labels = ["base", "s42-Bprime", *[f"s{s}-{arm}" for s in SEEDS for arm in METHODS]]
    results = {}
    for label in labels:
        endpoint = lock[label]
        folder = course / "evaluation/test" / endpoint
        pred, stats, config = read_model(folder, groups, provenance["test_sha256"], {})
        for k in ("tokenizer_sha256", "template_sha256"):
            require(config[k] == provenance[k], f"{label}: {k} differs")
        require(config["model_revision"] == provenance["target_revision"], "Model revision changed")
        adapter = stats["adapter_sha256"].get("adapter_model.safetensors")
        expected = identities[endpoint]
        require(adapter == (expected[1] if expected else None), f"{label}: adapter differs from lock")
        task = f"eval-test-{endpoint}"
        receipt = read_json(course / "receipts" / (task + ".json"))
        expected_outputs = {name: digest(folder / name) for name in ("metrics.json", "predictions.jsonl")}
        outputs = {Path(p).name: h for p, h in receipt["outputs"].items()}
        require(
            receipt["task"] == task
            and receipt["script_sha256"] == config["script_sha256"]
            and len(receipt["outputs"]) == len(expected_outputs)
            and outputs == expected_outputs,
            f"{label}: invalid test score receipt",
        )
        results[label] = pred
    return results


def summarize_slice(ids, groups, outcomes, seed):
    subgroups = {i: groups[i] for i in ids}
    slices = {model: {i: values[i] for i in ids} for model, values in outcomes.items()}
    correct = {model: sum(values.values()) for model, values in slices.items()}
    n = len(ids)
    mean_acc = {arm: mean(100 * correct[f"s{s}-{arm}"] / n for s in SEEDS) for arm in METHODS}
    contrasts = {}
    for left, right in COMPARISONS:
        pairs = [(slices[f"s{s}-{left}"], slices[f"s{s}-{right}"]) for s in SEEDS]
        pp = {s: 100 * (sum(l.values()) - sum(r.values())) / n for s, (l, r) in zip(SEEDS, pairs)}
        stat = fixed_seed_summary(pp, pairs, subgroups, seed)
        contrasts[left + "-" + right] = {
            "seed_differences_pp": pp,
            "mean_pp": stat["mean"],
            "ci95_pp": stat["fixed_seed_mean_paired_cluster_ci95_pp"],
        }
    return {
        "n": n,
        "near_duplicate_groups": len(set(subgroups.values())),
        "correct": correct,
        "checkpoint_accuracy_pct": {model: 100 * count / n for model, count in correct.items()},
        "mean_accuracy_pct": mean_acc,
        "base_accuracy_pct": 100 * correct["base"] / n,
        "s42_Bprime_accuracy_pct": 100 * correct["s42-Bprime"] / n,
        "contrasts": contrasts,
    }


def short_minus_long(short, long, groups, outcomes, seed):
    """Compare the B−A gaps directly, keeping each request group and its three seeds together."""
    import numpy as np
    from summarize_followup import seed_sensitivity

    # Each group contributes paired wins and counts to both bins in the same draw.
    totals = {}
    seed_gaps = {s: 0.0 for s in SEEDS}
    for bin_index, ids in enumerate((short, long)):
        for key in ids:
            group = totals.setdefault(groups[key], [0, 0, 0, 0])
            group[2 * bin_index + 1] += 1
            for s in SEEDS:
                delta = int(outcomes[f"s{s}-B"][key]) - int(outcomes[f"s{s}-A"][key])
                group[2 * bin_index] += delta
                seed_gaps[s] += (1 if bin_index == 0 else -1) * 100 * delta / len(ids)

    # Pool groups with equal contributions; their multinomial resampling law is identical.
    counts = Counter(tuple(row) for row in totals.values())
    patterns = sorted(counts)
    probabilities = np.array([counts[p] for p in patterns], dtype=float) / len(totals)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(totals), probabilities, size=10_000) @ np.array(patterns)
    valid = draws[(draws[:, 1] > 0) & (draws[:, 3] > 0)]
    require(len(valid) > 0, "No joint resampling draw included both length bins")
    gaps = (100 / len(SEEDS)) * (valid[:, 0] / valid[:, 1] - valid[:, 2] / valid[:, 3])
    return {
        "contrast": "(B minus A, shortest bin) minus (B minus A, longest bin), percentage points",
        "scope": "Exploratory contrast chosen after inspection of the original four bins.",
        "seed_differences_pp": seed_gaps,
        "mean_pp": mean(seed_gaps.values()),
        "fixed_seed_group_bootstrap_ci95_pp": np.quantile(gaps, [0.025, 0.975]).tolist(),
        "fixed_requests_three_seed_t_ci95_pp": seed_sensitivity(list(seed_gaps.values()))["ci95_pp"],
        "groups_in_both_bins": sum(row[1] > 0 and row[3] > 0 for row in totals.values()),
        "group_union_size": len(totals),
        "bootstrap_seed": seed,
        "bootstrap_draws": len(draws),
        "draws_with_both_bins": len(valid),
    }


def compute(course, model, seed):
    lengths, groups, provenance = reference_lengths(course, model)
    outcomes = load_outcomes(course, groups, provenance)
    cuts = quantiles(lengths.values(), n=4, method="inclusive")
    buckets = [[] for _ in range(4)]
    for key, value in lengths.items():
        buckets[bisect_left(cuts, value)].append(key)
    all_ids = list(lengths)
    overall = summarize_slice(all_ids, groups, outcomes, seed)
    records = []
    for index, ids in enumerate(buckets):
        record = {
            "quartile": index + 1,
            "lower_exclusive": cuts[index - 1] if index else None,
            "upper_inclusive": cuts[index] if index < 3 else None,
            "n": len(ids),
        }
        if ids:
            record.update(summarize_slice(ids, groups, outcomes, seed))
            record["observed_min"] = min(lengths[i] for i in ids)
            record["observed_max"] = max(lengths[i] for i in ids)
            for name, effect in record["contrasts"].items():
                effect["contribution_to_all_requests_pp"] = effect["mean_pp"] * len(ids) / len(all_ids)
        records.append(record)
    occupied = [row for row in records if row["n"]]
    require(sum(row["n"] for row in occupied) == len(all_ids), "Incomplete quartiles")
    for label, total in overall["correct"].items():
        require(sum(row["correct"][label] for row in occupied) == total, f"{label}: split counts differ")
    for name, v in overall["contrasts"].items():
        total = sum(row["contrasts"][name]["contribution_to_all_requests_pp"] for row in occupied)
        require(math.isclose(total, v["mean_pp"], abs_tol=1e-10), f"{name}: weighted contrast differs")
    hist = Counter(lengths.values())
    return {
        "scope": "Exploratory fixed-seed analysis of previously opened internal Turnstile test outcomes.",
        "unit": "Number of Qwen tokens overlapping nonempty scalar argument-value spans in the reference tool call, summed per request.",
        "ties": "Inclusive empirical quartiles (type 7). Equality goes to the lower bin. Ties are never split.",
        "bootstrap": {
            "seed": seed,
            "draws": 2000,
            "unit": "near-duplicate group within each fixed slice",
            "interval": "pointwise paired percentile 95%, conditional on the cutpoints and three paired seeds",
        },
        "provenance": provenance,
        "quartile_cutpoints": cuts,
        "counts_at_cutpoints": [hist[q] for q in cuts],
        "zero_value_requests": hist[0],
        "test_value_tokens": sum(lengths.values()),
        "seeds": list(SEEDS),
        "quartiles": records,
        "all": overall,
        "short_minus_long_B_minus_A": short_minus_long(buckets[0], buckets[3], groups, outcomes, seed)
        if buckets[0] and buckets[3] else None,
    }


def markdown(summary):
    lines = [
        "# Internal test accuracy by reference argument-value length",
        "",
        "The frozen training-pool check matched all 4,000 recorded argument-value token counts. "
        "The slice variable uses the reference call. The model generations were already complete.",
        "",
        "A uses random selection, B uses whole-answer trajectories, and C uses field trajectories. "
        "The table averages exact-call accuracy over paired seeds 42, 43 and 44. Base is the original "
        "2B checkpoint before project fine-tuning. B′ was available for seed 42. Paired differences "
        "and intervals are in percentage points.",
        "",
        "| Value tokens | Requests | Base | A | B | C | B′ (42) | B−A [95%] | C−B [95%] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [*summary["quartiles"], summary["all"]]:
        if not row["n"]:
            lines.append(f"| Q{row['quartile']} (empty) | 0 | | | | | | | |")
            continue
        label = (
            f"Q{row['quartile']} ({row['observed_min']}–{row['observed_max']})"
            if "quartile" in row
            else "All"
        )
        vals = [
            row["base_accuracy_pct"],
            *[row["mean_accuracy_pct"][a] for a in METHODS],
            row["s42_Bprime_accuracy_pct"],
        ]
        pp = []
        for pair in ("B-A", "C-B"):
            v = row["contrasts"][pair]
            lo, hi = v["ci95_pp"]
            pp.append(f"{v['mean_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append("| " + " | ".join([label, str(row["n"]), *[f"{x:.2f}%" for x in vals], *pp]) + " |")
    lines.extend(
        [
            "",
            f"The quartile cutpoints were {summary['quartile_cutpoints']}. The counts exactly on them were "
            f"{summary['counts_at_cutpoints']}. Identical lengths stay together. "
            f"{summary['zero_value_requests']} requests had zero value tokens.",
            "",
            "The 95% intervals resample near-duplicate request groups within each slice while keeping all methods and "
            "three seeds aligned. These exploratory comparisons describe the fixed test set. A length-matched "
            "training control is needed to test whether training length composition caused a performance change.",
            "",
        ]
    )
    interaction = summary.get("short_minus_long_B_minus_A")
    if interaction:
        lo, hi = interaction["fixed_seed_group_bootstrap_ci95_pp"]
        t_lo, t_hi = interaction["fixed_requests_three_seed_t_ci95_pp"]
        lines.append(
            f"The exploratory shortest-minus-longest difference in B−A was {interaction['mean_pp']:+.2f} pp "
            f"(paired request-group 95% interval [{lo:+.2f}, {hi:+.2f}]; "
            f"three-seed t interval [{t_lo:+.2f}, {t_hi:+.2f}]).\n"
        )
    return "\n".join(lines)


def write_file(path, text):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(text)
        temp = Path(f.name)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--course", type=Path, required=True, help="Complete private course-run directory")
    p.add_argument(
        "--model", type=Path, required=True, help="Local Qwen3.5 target tokenizer and template directory"
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Directory for aggregate summary.json and summary.md"
    )
    p.add_argument("--bootstrap-seed", type=int, default=31415)
    args = p.parse_args()
    report = compute(args.course, args.model, args.bootstrap_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_file(
        args.out / "summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    write_file(args.out / "summary.md", markdown(report))
    print(args.out / "summary.md")


if __name__ == "__main__":
    main()
