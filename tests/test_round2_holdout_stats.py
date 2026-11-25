"""Synthetic-only tests of the registered Round 2 paired analysis."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import round2_holdout_stats as study

HAS_PACKAGES = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy"))


def fixture():
    groups = {f"q{i}": f"group{i}" for i in range(1500)}
    labels = sorted(study.EXPECTED_LABELS)
    outcomes = {label: dict.fromkeys(groups, False) for label in labels}
    length_means = {f"s{seed}-{method}": length for seed in study.SEEDS
                    for method, length in zip(study.LENGTH_METHODS, (12, 15.3, 18.9, 23.7))}
    return {"holdout_v2": {"panel_order": labels}}, outcomes, groups, length_means


class FrozenInputTests(unittest.TestCase):
    def test_rejects_dropped_added_or_replaced_endpoints(self):
        registration, outcomes, groups, lengths = fixture()
        self.assertEqual(len(outcomes), 47)
        for change in ("missing", "extra", "duplicate"):
            with self.subTest(change=change):
                tweaked = dict(outcomes)
                roster = list(registration["holdout_v2"]["panel_order"])
                if change == "missing":
                    tweaked.pop("s46-C")
                elif change == "extra":
                    tweaked["s42-alternative"] = outcomes["base"]
                else:
                    roster[0] = roster[1]
                with self.assertRaisesRegex(ValueError, "inventories"):
                    study.summarize({"holdout_v2": {"panel_order": roster}}, tweaked, groups, lengths)

    def test_no_missing_scores_no_invented_realized_lengths(self):
        registration, outcomes, groups, lengths = fixture()
        for bad in (None, 0, "incorrect"):
            outcomes["base"]["q75"] = bad
            with self.assertRaisesRegex(ValueError, "indicators"):
                study.summarize(registration, outcomes, groups, lengths)
        outcomes["base"].pop("q75")
        with self.assertRaisesRegex(ValueError, "indicators"):
            study.summarize(registration, outcomes, groups, lengths)
        outcomes["base"]["q75"] = False
        with self.assertRaisesRegex(ValueError, "per group"):
            study.summarize(registration, outcomes, {**groups, "q75": groups["q76"]}, lengths)
        with self.assertRaisesRegex(ValueError, "length"):
            study.summarize(registration, outcomes, groups, {**lengths, "s42-E2": math.nan})
        for m in study.LENGTH_METHODS:
            lengths[f"s42-{m}"] = 10.0
        with self.assertRaisesRegex(ValueError, "slope"):
            study.summarize(registration, outcomes, groups, lengths)


@unittest.skipUnless(HAS_PACKAGES, "Install CPU selection requirements for numerical analysis tests")
class StatisticsTests(unittest.TestCase):
    def test_group_ratio_shared_draws_match_independent_direct_sampling(self):
        import numpy as np

        labels = ["z", "z", "z", "a"]
        values = np.array([[100, 0], [100, 0], [100, 0], [0, 100]], dtype=float)
        bounds = study._bootstrap(values, labels, draws=129)
        unique = sorted(set(labels))
        rng = np.random.default_rng(260205)
        direct = []
        for _ in range(129):
            g = rng.integers(len(unique), size=len(unique))
            indices = [i for j in g for i, key in enumerate(labels) if key == unique[j]]
            direct.append(values[indices].mean(axis=0))
        expected = np.percentile(direct, [2.5, 97.5, 5, 95], axis=0, method="linear")
        np.testing.assert_allclose(bounds, expected)
        np.testing.assert_allclose(bounds[0, 0] + bounds[1, 1], 100)

    def test_zero_observed_variance_is_never_a_success(self):
        for constant in (0, 2, -1):
            result = study._seed_summary([constant] * 5, primary=True)
            self.assertIsNone(result["t_statistic"])
            self.assertEqual(result["one_sided_p"], 1.0)
            self.assertFalse(result["superiority_supported"])
            self.assertEqual(result["ci95"], [constant, constant])
        with self.assertRaisesRegex(ValueError, "all five"):
            study._seed_summary([0, 1, 2], primary=True)

    def test_seed_test_matches_known_df4_tail_and_point_intervals(self):
        values = [1, 2, 3, 4, 5]
        result = study._seed_summary(values, primary=True)
        # For df=4, P(T >= x) = (1-u)^2 (2+u)/4, u=x/sqrt(x^2+4).
        t_value = 3 * math.sqrt(2)
        u = t_value / math.hypot(t_value, 2)
        self.assertAlmostEqual(result["one_sided_p"], (1 - u) ** 2 * (2 + u) / 4)
        self.assertAlmostEqual(result["ci95"][0], 3 - 2.7764451051977987 / math.sqrt(2))
        self.assertTrue(result["superiority_supported"])
        self.assertGreater(study._seed_summary([-x for x in values], primary=True)["one_sided_p"], .99)

    def test_complete_summary_contains_only_aggregate_numerical_claims(self):
        registration, outcomes, groups, lengths = fixture()
        for seed in study.SEEDS:
            # Each seed contributes a different C-B gain. Others follow fixed
            # marginal rates which allow an independent slope calculation.
            for i in range(seed - 39):
                outcomes[f"s{seed}-C"][f"q{i}"] = True
            for method, count in zip(study.LENGTH_METHODS, (20, 15, 11, 9)):
                for i in range(count):
                    outcomes[f"s{seed}-{method}"][f"q{i}"] = True
        result = study.summarize(registration, outcomes, groups, lengths)
        self.assertEqual(len(result["endpoints"]), 47)
        self.assertNotIn("q1499", json.dumps(result, allow_nan=False))
        primary = result["contrasts"]["C-B"]
        self.assertAlmostEqual(primary["paired_seed"]["mean"], 100 * 5 / 1500)
        self.assertEqual(primary["paired_by_seed"]["42"]["left_only"], 3)
        self.assertAlmostEqual(primary["descriptive_seed_groups_pp"]["new_45_and_46"], 100 * 6.5 / 1500)
        for key, contrast in result["contrasts"].items():
            self.assertEqual("one_sided_p" in contrast["paired_seed"], key == study.PRIMARY)
        self.assertIsNone(result["contrasts"]["full-s42-step125-A"]["paired_seed"]["ci95"])
        import numpy as np

        slope = np.polyfit([12, 15.3, 18.9, 23.7], np.array([20, 15, 11, 9]) / 15, 1)[0]
        self.assertAlmostEqual(result["contrasts"]["length_slope"]["paired_seed"]["mean"], slope)
        self.assertEqual(len(result["contrasts"]["E2b-A"]["fixed_seed_request_group_ci90"]), 2)
        self.assertEqual(result["bootstrap"]["draws"], 10_000)


if __name__ == "__main__":
    unittest.main()
