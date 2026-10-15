"""CPU checks for FDP. No model and no PGD."""

from __future__ import annotations

import unittest

import torch

from week2_attack_v7_budget_conditioned.fdp import (
    EPSILON_MARGIN,
    bar_soft,
    certificate,
    fdp_loss,
    greedy_tokens,
    hard_margin,
    soft_margin,
)


class FdpCpuTest(unittest.TestCase):
    def test_delta_zero_is_finite_and_includes_eos(self):
        logits = torch.tensor([[5.0, 0.0, 1.0], [4.0, 3.0, 0.0], [2.0, 0.5, 6.0]])
        tokens = torch.tensor([0, 1, 2])
        self.assertEqual(int(tokens[-1]), 2)
        clean = soft_margin(logits, tokens)
        loss = fdp_loss(clean, clean)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(loss.numel()), 1)
        self.assertEqual(int(clean.shape[0]), 3)

    def test_bar_never_exceeds_clean_soft_margin(self):
        clean = torch.tensor([3.0, 0.2, -1.0, 8.0])
        bar = bar_soft(clean)
        self.assertTrue(torch.all(bar <= clean + 1e-6))

    def test_toy_flip_increases_fdp(self):
        tokens = torch.tensor([0, 1])
        clean_logits = torch.tensor([[8.0, 0.0, 0.0], [7.0, 1.0, 0.0]])
        flipped = torch.tensor([[0.0, 8.0, 0.0], [0.0, 1.0, 7.0]])
        clean = soft_margin(clean_logits, tokens)
        before = fdp_loss(clean, soft_margin(clean_logits, tokens))
        after = fdp_loss(clean, soft_margin(flipped, tokens))
        self.assertGreater(float(after), float(before))

    def test_positive_hard_margin_replays_clean_tokens(self):
        logits = torch.tensor([[3.0, 1.0, 0.0], [0.2, 4.0, 0.0], [0.0, 0.4, 5.0]])
        tokens = torch.tensor([0, 1, 2])
        hard = hard_margin(logits, tokens)
        self.assertTrue(torch.all(hard > EPSILON_MARGIN))
        self.assertEqual(certificate(hard), "FULL_CONE_CERTIFICATE_AVAILABLE")
        self.assertTrue(torch.equal(greedy_tokens(logits), tokens))

    def test_nonpositive_hard_margin_is_unavailable_and_can_flip(self):
        logits = torch.tensor([[0.0, 2.0], [1.0, 0.0]])
        tokens = torch.tensor([0, 0])
        hard = hard_margin(logits, tokens)
        self.assertEqual(certificate(hard), "FULL_CONE_CERTIFICATE_UNAVAILABLE")
        self.assertFalse(torch.equal(greedy_tokens(logits), tokens))

    def test_teacher_forced_argmax_matches_generated_ids(self):
        generated = torch.tensor([2, 0, 1])
        logits = torch.tensor([[0.0, 0.2, 5.0], [4.0, 0.0, 0.1], [0.3, 3.0, 0.0]])
        self.assertTrue(torch.equal(greedy_tokens(logits), generated))
        self.assertTrue(torch.all(hard_margin(logits, generated) > 0))


if __name__ == "__main__":
    unittest.main()
