"""CPU checks for the frozen native60 summaries."""

from __future__ import annotations

import unittest

from week2_attack_v7c_native60_readonly_remeasurement.stats import (
    bootstrap_mean_ci,
    curve_supports_sweet_spot,
    decide,
    mean,
    sweet_outer,
)


class StatsTest(unittest.TestCase):
    def test_shard_rule_is_stable_stride(self):
        ids = [str(i) for i in range(60)]
        shards = [ids[i::4] for i in range(4)]
        self.assertEqual([len(shard) for shard in shards], [15, 15, 15, 15])
        self.assertEqual(sorted(sum(shards, []), key=int), ids)

    def test_delta_direction_and_sweet_gap(self):
        delta = {
            0.50: [0.0] * 60,
            0.35: [2.0] * 60,
            0.25: [2.0] * 60,
            0.15: [0.0] * 60,
        }
        packed = sweet_outer(delta)
        self.assertGreater(packed["sweet_mean"], packed["outer_mean"])
        self.assertGreater(packed["paired_diff_ci95"][0], 0)
        lo, hi = bootstrap_mean_ci(delta[0.25])
        self.assertGreater(lo, 0)
        self.assertLessEqual(lo, mean(delta[0.25]))
        self.assertGreaterEqual(hi, mean(delta[0.25]))

    def test_decision_priority(self):
        delta_mean = {0.25: 2.0, 0.35: 2.0}
        ci = {0.25: 1.0, 0.35: 1.0}
        sweet = {"sweet_mean": 2.0, "outer_mean": 0.0, "paired_diff_ci95": [1.0, 3.0]}
        counts = {0.50: 10, 0.35: 40, 0.25: 50, 0.15: 20}
        self.assertTrue(curve_supports_sweet_spot(counts))
        self.assertEqual(
            decide(True, True, ci, delta_mean, sweet, counts),
            "PASS_NATIVE60_SWEET_SPOT_AND_DIFFERENTIAL_SUPPORTED",
        )
        self.assertEqual(
            decide(False, True, ci, delta_mean, sweet, counts),
            "BLOCKED_NATIVE60_ALIGNMENT_MISMATCH",
        )
        weak = {"sweet_mean": 0.1, "outer_mean": 0.2, "paired_diff_ci95": [-1.0, 0.5]}
        self.assertEqual(
            decide(True, True, ci, delta_mean, weak, counts),
            "PARTIAL_NATIVE60_DIFFERENTIAL_SUPPORTED_NO_CONFIRMED_SWEET_SPOT",
        )
        none = {0.25: -1.0, 0.35: -1.0}
        self.assertEqual(
            decide(True, True, none, none, weak, counts),
            "NO_GO_NATIVE60_DIFFERENTIAL_NOT_SUPPORTED",
        )


if __name__ == "__main__":
    unittest.main()
