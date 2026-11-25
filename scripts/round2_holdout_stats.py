"""Registered comparisons for a complete, already-scored Round 2 panel.

This module only operates on supplied correctness indicators. The holdout
runner owns opening authorization, frozen artifacts and independent scoring.
"""

import math
import statistics


SEEDS = (42, 43, 44, 45, 46)
PRIOR_SEEDS = SEEDS[:3]
LENGTH_METHODS = ("L-short", "A", "E2", "L-long")
FULL_MODELS = ("full-s42-step125", "full-s42-step500")
PRIMARY = "C-B"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 260205


def _pairs(left, right, seeds):
    return [(seed, f"s{seed}-{left}", f"s{seed}-{right}") for seed in seeds]


COMPARISONS = {
    f"{left}-{right}": _pairs(left, right, seeds)
    for left, right, seeds in (
        ("C", "B", SEEDS), ("C", "A", SEEDS), ("L-long", "L-short", SEEDS),
        ("B", "A", SEEDS), ("E2", "A", SEEDS), ("E2b", "A", SEEDS),
        ("E2b", "E2", SEEDS), ("L-short", "A", SEEDS),
        ("Bprime", "B", PRIOR_SEEDS), ("C", "Bprime", PRIOR_SEEDS),
        ("E3", "A", PRIOR_SEEDS), ("E4", "A", PRIOR_SEEDS),
    )
}
for _full in FULL_MODELS:
    COMPARISONS[f"{_full}-A"] = [(42, _full, "s42-A")]

EXPECTED_LABELS = {"base", *FULL_MODELS}
for _pairs_by_seed in COMPARISONS.values():
    for _, _left, _right in _pairs_by_seed:
        EXPECTED_LABELS.update((_left, _right))


def _validate(registration, outcomes, groups, length_means):
    try:
        roster = registration["holdout_v2"]["panel_order"]
    except (TypeError, KeyError) as exc:
        raise ValueError("The registration must contain its frozen panel order") from exc
    if (not isinstance(roster, list) or len(roster) != len(EXPECTED_LABELS)
            or set(roster) != EXPECTED_LABELS or set(outcomes) != EXPECTED_LABELS):
        raise ValueError("The registered and scored inventories must cover exactly the 47 endpoints")
    if len(groups) != 1500 or any(not isinstance(v, str) or not v.strip() for v in groups.values()):
        raise ValueError("Expected 1,500 registered requests with nonempty group IDs")
    if len(set(groups.values())) != 1500:
        raise ValueError("The frozen reservation contains one request per group")
    if any(not isinstance(i, str) or not i for i in groups):
        raise ValueError("Expected nonempty string request IDs")
    for label in roster:
        responses = outcomes[label]
        if responses.keys() != groups.keys() or any(type(v) is not bool for v in responses.values()):
            raise ValueError(f"{label}: incomplete or invalid correctness indicators")
    for seed in SEEDS:
        lengths = [length_means.get(f"s{seed}-{m}") for m in LENGTH_METHODS]
        if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in lengths):
            raise ValueError(f"Missing or invalid realized training length at seed {seed}")
        if max(lengths) == min(lengths):
            raise ValueError(f"Undefined length slope at seed {seed}")
    return roster


def _seed_summary(values, *, primary=False, descriptive_90=False):
    from scipy.stats import t

    n = len(values)
    if primary and n != 5:
        raise ValueError("The primary paired-seed t test requires all five seeds")
    mean = statistics.mean(values)
    out = {"mean": mean, "n_seeds": n, "df": n - 1}
    if n == 1:
        return {**out, "ci95": None}
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    out.update(sample_sd=sd, zero_observed_seed_variance=sd == 0)

    def interval(confidence):
        radius = float(t.ppf((1 + confidence) / 2, df=n - 1)) * se
        return [mean - radius, mean + radius]

    out["ci95"] = interval(0.95)
    if descriptive_90:
        out["ci90"] = interval(0.90)
    if primary:
        stat = mean / se if se else None
        p = float(t.sf(stat, df=4)) if se else 1.0
        out.update(t_statistic=stat, one_sided_p=p, superiority_supported=bool(p < 0.05))
    return out


def _bootstrap(by_question, group_ids, *, draws=BOOTSTRAP_DRAWS):
    """Share each draw of groups across all columns; keep request weighting."""
    import numpy as np

    labels = sorted(set(group_ids))
    lookup = {label: i for i, label in enumerate(labels)}
    group_index = [lookup[label] for label in group_ids]
    sizes = np.bincount(group_index, minlength=len(labels))
    totals = np.zeros((len(labels), by_question.shape[1]), dtype=np.float64)
    np.add.at(totals, group_index, by_question)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = np.empty((draws, by_question.shape[1]), dtype=np.float64)
    for start in range(0, draws, 64):
        n = min(64, draws - start)
        indices = rng.integers(len(labels), size=(n, len(labels)))
        # Keep the same random integer stream as a draw-by-draw loop.
        indices += len(labels) * np.arange(n)[:, None]
        counts = np.bincount(indices.ravel(), minlength=n * len(labels)).reshape(n, len(labels))
        sampled[start:start + n] = (counts @ totals) / (counts @ sizes)[:, None]
    return np.percentile(sampled, [2.5, 97.5, 5, 95], axis=0, method="linear")


def summarize(registration, outcomes, groups, length_means):
    """Return public aggregates only after verifying the complete paired inventory."""
    roster = _validate(registration, outcomes, groups, length_means)
    import numpy as np
    import scipy

    ids = sorted(groups)
    n = len(ids)
    scores = {model: np.array([outcomes[model][i] for i in ids], dtype=np.int8) for model in roster}
    model_counts = {model: {"correct": int(y.sum()), "requests": n, "accuracy_pct": 100 * float(y.mean())}
                    for model, y in scores.items()}
    by_question = {}
    comparisons = {}
    for name, pairs in COMPARISONS.items():
        deltas, paired_counts = [], {}
        for seed, left, right in pairs:
            a, b = scores[left], scores[right]
            d = a - b
            deltas.append(d)
            paired_counts[str(seed)] = {
                "left_only": int((d == 1).sum()), "right_only": int((d == -1).sum()),
                "both_correct": int((a & b).sum()), "both_wrong": int(((a | b) == 0).sum()),
                "difference_pp": 100 * int(d.sum()) / n,
            }
        differences = [entry["difference_pp"] for entry in paired_counts.values()]
        comparisons[name] = {
            "role": "confirmatory" if name == PRIMARY else "descriptive",
            "unit": "percentage points",
            "paired_by_seed": paired_counts,
            "paired_seed": _seed_summary(differences, primary=name == PRIMARY, descriptive_90=name == "E2b-A"),
        }
        if name == PRIMARY:
            comparisons[name]["descriptive_seed_groups_pp"] = {
                "historical_42_to_44": statistics.mean(differences[:3]),
                "new_45_and_46": statistics.mean(differences[3:]),
            }
        by_question[name] = 100 * np.mean(deltas, axis=0)

    # OLS with a fixed training-set length per endpoint is linear in the test
    # accuracies. A slope per request therefore gives the same resampled slopes
    # as fitting each seed's regression on the accuracy vector of every draw.
    slopes, seed_slopes = [], {}
    for seed in SEEDS:
        endpoints = [f"s{seed}-{m}" for m in LENGTH_METHODS]
        x = np.array([length_means[k] for k in endpoints], dtype=np.float64)
        centered = x - x.mean()
        weights = centered / (centered @ centered)
        per_row = 100 * (weights @ np.stack([scores[k] for k in endpoints]))
        slopes.append(per_row)
        seed_slopes[str(seed)] = {"lengths": dict(zip(LENGTH_METHODS, x.tolist())),
                                  "slope": float(per_row.mean())}
    comparisons["length_slope"] = {
        "role": "descriptive", "unit": "percentage points per argument-value token",
        "by_seed": seed_slopes, "paired_seed": _seed_summary([x["slope"] for x in seed_slopes.values()]),
    }
    by_question["length_slope"] = np.mean(slopes, axis=0)
    by_question["base"] = 100.0 * scores["base"]
    bounds = _bootstrap(np.column_stack(list(by_question.values())), [groups[i] for i in ids])
    for j, name in enumerate(by_question):
        target = model_counts["base"] if name == "base" else comparisons[name]
        target["fixed_seed_request_group_ci95"] = bounds[:2, j].tolist()
        if name == "E2b-A":
            target["fixed_seed_request_group_ci90"] = bounds[2:, j].tolist()
            target["descriptive_margin_pp"] = [-1.0, 1.0]
    return {
        "requests": n, "request_groups": len(set(groups.values())), "endpoints": model_counts,
        "primary": PRIMARY, "contrasts": comparisons,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "quantile": "linear",
                      "population": "new request groups, conditional on the trained endpoints"},
        "statistics_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
