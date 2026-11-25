"""Train-only length-dose selection tests using small synthetic pools."""

from importlib.util import find_spec
from pathlib import Path
import sys
import unittest


HAS_SELECTION_DEPS = all(find_spec(pkg) is not None for pkg in ("numpy", "scipy", "sklearn"))
if HAS_SELECTION_DEPS:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from select_round2 import draw_bins, tilted_quota


@unittest.skipUnless(HAS_SELECTION_DEPS, "requires requirements/selection.txt")
class Round2SelectionTests(unittest.TestCase):
    def test_tilt_attains_exactly_representable_short_and_long_means(self):
        capacities = np.array([40, 40])
        means = np.array([4.0, 12.0])
        for target, expected in [(6, [15, 5]), (10, [5, 15])]:
            with self.subTest(target=target):
                quota, _ = tilted_quota(capacities, means, 20, target, 42)
                np.testing.assert_array_equal(quota, expected)
                self.assertEqual(np.average(means, weights=quota), target)

    def test_small_bin_saturates_and_all_seats_are_allocated(self):
        capacities = np.array([100, 100, 2])
        means = np.array([1.0, 9.0, 10.0])
        quota, _ = tilted_quota(capacities, means, 100, 9.015, 45)
        # An unconstrained allocation rounds the rarest bin to three seats.
        self.assertEqual(quota[2], 2)
        self.assertEqual(sum(quota), 100)
        self.assertTrue(all((0 <= quota) & (quota <= capacities)))

    def test_unreachable_target_is_rejected(self):
        with self.assertRaises(ValueError):
            tilted_quota(np.array([40, 40]), np.array([4.0, 12.0]), 20, 20, 42)

    def test_equal_remainders_have_reproducible_integer_quotas(self):
        capacities = np.full(4, 6)
        means = np.array([1.0, 2.0, 3.0, 4.0])
        first, _ = tilted_quota(capacities, means, 6, 2.5, 42)
        replay, _ = tilted_quota(capacities, means, 6, 2.5, 42)
        np.testing.assert_array_equal(first, replay)
        np.testing.assert_array_equal(np.sort(first), [1, 1, 2, 2])

    def test_draw_is_seeded_without_replacement_and_preserves_quotas(self):
        bins = np.tile(np.arange(3), 20)
        quota = np.array([5, 0, 8])
        first = np.asarray(draw_bins(bins, quota, 45))
        replay = np.asarray(draw_bins(bins, quota, 45))
        alternate = np.asarray(draw_bins(bins, quota, 46))
        np.testing.assert_array_equal(first, replay)
        self.assertFalse(np.array_equal(first, alternate))
        for result in (first, alternate):
            np.testing.assert_array_equal(result, np.unique(result))
            np.testing.assert_array_equal(np.bincount(bins[result], minlength=3), quota)
