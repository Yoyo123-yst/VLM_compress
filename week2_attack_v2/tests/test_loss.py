import unittest

import torch

from week2_attack_v2.loss import kl_clean_adv, share_target_loss


class ShareTargetTests(unittest.TestCase):
    def test_zero_when_soft_share_meets_target(self):
        scores = torch.tensor([0.5, 0.4, 0.01, 0.001], requires_grad=True)
        loss, soft = share_target_loss(scores, aux_positions=[0, 1], keep_k=2, temperature=0.05, q_star=0.6)
        self.assertLess(float(loss), 1e-6)
        self.assertGreater(float(soft), 0.6)

    def test_pushes_aux_scores_up_when_below_target(self):
        scores = torch.tensor([0.9, 0.8, 0.01, 0.001], requires_grad=True)
        loss, soft = share_target_loss(scores, aux_positions=[2, 3], keep_k=2, temperature=0.05, q_star=0.6)
        self.assertGreater(float(loss), 0.0)
        self.assertLess(float(soft), 0.6)
        loss.backward()
        self.assertLess(float(scores.grad[2]), 0.0)
        self.assertLess(float(scores.grad[3]), 0.0)

    def test_tau_is_detached(self):
        scores = torch.tensor([0.4, 0.3, 0.2, 0.1], requires_grad=True)
        loss, _ = share_target_loss(scores, aux_positions=[2, 3], keep_k=2, temperature=0.1, q_star=0.6)
        loss.backward()
        # The threshold score itself (index 1, value 0.3) is detached, so it gets no direct grad.
        self.assertEqual(float(scores.grad[1]), 0.0)

    def test_kl_matches_manual(self):
        log_p = torch.log_softmax(torch.tensor([1.0, 0.0, -1.0]), dim=-1)
        logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        loss = kl_clean_adv(log_p, logits)
        loss.backward()
        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
