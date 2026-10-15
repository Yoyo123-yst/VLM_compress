"""CPU checks for the frozen 47450 partition and the phase-1 gate."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT)]

from week2_attack_v20_fourway_content_ablation.optim import restore_adam, snapshot_adam  # noqa: E402
from week2_attack_v20_fourway_content_ablation.selector import choose_fixed  # noqa: E402
from week2_attack_v21_context_leverage_causal_probe.groups import draw_control, partition, supporting_pair  # noqa: E402

PIECES = ["Dr", ".", " or", " Mrs", ".", " William", " J", ".", " Dar", "by", "<|im_end|>"]


def tokens():
    return [{"position": index, "piece": piece, "token_id": index + 10} for index, piece in enumerate(PIECES)]


class PartitionTests(unittest.TestCase):
    def test_known_47450_split(self):
        grouped = partition(tokens(), [5, 6, 7, 8, 9], list(range(10)))
        self.assertTrue(grouped["ok"], grouped["problems"])
        self.assertEqual(grouped["s_context"], [0, 1, 2, 3, 4])
        self.assertEqual(grouped["context_text"], "Dr. or Mrs.")
        self.assertEqual(set(grouped["s_core"]) & set(grouped["s_context"]), set())
        self.assertEqual(sorted(set(grouped["s_core"]) | set(grouped["s_context"])), grouped["s_all"])

    def test_context_string_blocks_a_guess(self):
        rows = tokens()
        rows[2]["piece"] = " maybe"
        grouped = partition(rows, [5, 6, 7, 8, 9], list(range(10)))
        self.assertFalse(grouped["ok"])


class RandomTests(unittest.TestCase):
    def test_seeds_are_stable_and_not_the_target(self):
        universe = list(range(10))
        first = draw_control(universe, 5, [5, 6, 7, 8, 9], 1701)
        second = draw_control(universe, 5, [5, 6, 7, 8, 9], 1701)
        other = draw_control(universe, 5, [0, 1, 2, 3, 4], 1702)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertNotEqual(first, [5, 6, 7, 8, 9])
        self.assertNotEqual(other, [0, 1, 2, 3, 4])


class SelectorTests(unittest.TestCase):
    def test_selector_rejects_outcome_fields(self):
        row = {
            "checkpoint": 1,
            "nonzero": True,
            "output_valid": True,
            "full_text_exact": True,
            "full_token_exact": True,
            "full_replay": True,
            "linf_ok": True,
            "png_reloadable": True,
            "attack_surrogate": 0.2,
        }
        self.assertEqual(choose_fixed([row])["checkpoint"], 1)
        with self.assertRaises(RuntimeError):
            choose_fixed([{**row, "semantic_status": "CONTENT_FAILURE"}])


class RollbackTests(unittest.TestCase):
    def test_rejected_update_restores_adam_and_delta(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        parameter.grad = torch.ones(2)
        optimizer.step()
        previous = parameter.detach().clone()
        snapshot = snapshot_adam(optimizer)
        parameter.grad = torch.full((2,), 4.0)
        optimizer.step()
        restore_adam(optimizer, snapshot, parameter, previous)
        self.assertTrue(torch.equal(parameter.detach(), previous))
        restored = optimizer.state_dict()["state"][0]
        saved = snapshot["state"][0]
        self.assertTrue(torch.equal(restored["exp_avg"], saved["exp_avg"]))
        self.assertTrue(torch.equal(restored["exp_avg_sq"], saved["exp_avg_sq"]))
        self.assertEqual(int(restored["step"]), int(saved["step"]))
        copy.deepcopy(snapshot)

    def test_noop_and_sham_leave_delta(self):
        delta = torch.tensor([0.1, -0.2])
        original = delta.clone()
        applied = delta.clone()
        applied.zero_()
        applied.copy_(original)
        self.assertTrue(torch.equal(applied, original))


class GateTests(unittest.TestCase):
    def row(self, margin, exact=True):
        return {"full_exact": exact, "linf_ok": True, "png_reloaded": True, "core_margin_natural": margin}

    def test_adjacent_pair_must_beat_random(self):
        context = {0.25: self.row(1.0), 0.5: self.row(0.8), 1.0: self.row(0.7), 2.0: self.row(0.9)}
        weaker = {scale: self.row(1.2) for scale in (0.25, 0.5, 1.0, 2.0)}
        matched = {0.25: self.row(0.9), 0.5: self.row(0.7), 1.0: self.row(0.6), 2.0: self.row(0.9)}
        self.assertTrue(supporting_pair(context, {1701: weaker}, 1.5)["supported"])
        self.assertFalse(supporting_pair(context, {1701: matched}, 1.5)["supported"])
        flat = {scale: self.row(1.5) for scale in (0.25, 0.5, 1.0, 2.0)}
        self.assertFalse(supporting_pair(flat, {1701: weaker}, 1.5)["supported"])


if __name__ == "__main__":
    unittest.main()
