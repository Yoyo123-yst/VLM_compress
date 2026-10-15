"""CPU checks for the frozen B+ development rules."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import (
    conditional_gap,
    decision_floor,
    decision_preservation,
    method_loss,
    soft_anchor,
)

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation")


class LossRuleTest(unittest.TestCase):
    def test_floor_never_exceeds_clean_hard_margin(self):
        clean = torch.tensor([2.0, 0.2, -0.5])
        floor = decision_floor(clean)
        self.assertTrue(torch.all(floor <= clean + 1e-6))

    def test_methods_are_not_the_same_function(self):
        compressed = torch.tensor(1.0)
        anchor = torch.tensor(0.2)
        fdp = torch.tensor(0.7)
        gap = torch.tensor(-0.4)
        native = method_loss("B-Native", compressed, anchor, fdp, gap, 2.0, 3.0)
        fdp_loss = method_loss("native-FDP", compressed, anchor, fdp, gap, 2.0, 3.0)
        plus = method_loss("B+", compressed, anchor, fdp, gap, 2.0, 3.0)
        self.assertNotEqual(float(native), float(fdp_loss))
        self.assertNotEqual(float(fdp_loss), float(plus))

    def test_gap_uses_only_the_two_train_ratios(self):
        with self.assertRaises(ValueError):
            conditional_gap(torch.tensor(1.0), [torch.tensor(0.1)])

    def test_sources_do_not_call_the_manual_logits_path(self):
        for name in ("losses.py",):
            source = (ROOT / name).read_text()
            self.assertNotIn("full_answer_logits", source)
        tree = ast.parse((ROOT / "losses.py").read_text())
        self.assertNotIn("0.15", ast.dump(tree))
        self.assertNotIn("0.50", ast.dump(tree))

    def test_anchor_and_floor_differ(self):
        soft = torch.tensor([1.0, 1.0])
        hard = torch.tensor([1.0, 1.0])
        clean = torch.tensor([3.0, 3.0])
        self.assertNotEqual(float(soft_anchor(soft, clean)), float(decision_preservation(hard, clean)))


if __name__ == "__main__":
    unittest.main()
