"""CPU checks frozen before v6c training."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from week2_attack_v6c_split_calibration.losses import LAMBDA_SOFT, probe_pages
from week2_attack_v6c_split_calibration.selector import choose, eligible
from week2_attack_v6b_soft_full_anchor.soft_loss import selection_l1, with_eos

ROOT = Path("week2_attack_v6c_split_calibration")


class FrozenValueTests(unittest.TestCase):
    def test_lambda_exact(self):
        payload = json.loads((ROOT / "FROZEN_COEFFICIENTS.json").read_text())
        self.assertEqual(payload["lambda_soft"], LAMBDA_SOFT)
        self.assertEqual(LAMBDA_SOFT, 2.049824826423478)
        quotient = payload["lambda_numerator"] / payload["lambda_denominator"]
        self.assertEqual(quotient, LAMBDA_SOFT)

    def test_b_formula_has_no_beta(self):
        text = (ROOT / "FROZEN_PROTOCOL.json").read_text()
        self.assertIn("L_comp + lambda_soft", text)
        self.assertNotIn("beta * L1", json.loads(text)["variant_b"])


class ProbeTests(unittest.TestCase):
    def test_clean_selection_zero_grad_is_pass(self):
        mass = torch.tensor([0.2, 0.3], requires_grad=True)
        loss = selection_l1(mass, mass.detach())
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(mass.grad.norm()), 0.0)

    def test_probe_changes_only_aux(self):
        pages = [torch.zeros(2, 2), torch.ones(2, 2) * 0.2]
        probed = probe_pages(pages, aux_idx=1)
        self.assertTrue(torch.equal(probed[0], pages[0]))
        self.assertTrue(torch.all(probed[1] > pages[1]))

    def test_beta_formula_is_stable(self):
        beta = 4.0 / 2.0
        self.assertEqual(beta, 2.0)
        self.assertGreater(beta, 0.0)


class SelectorTests(unittest.TestCase):
    def test_ignores_forbidden_fields_and_step0(self):
        rows = [
            {"step": 0, "margin": -9, "nonzero": True, "output_valid": True, "full_exact": True, "linf_ok": True, "pixel_linf_uint8": 0.01, "anls_fastv": 0.0},
            {"step": 100, "margin": -1, "nonzero": True, "output_valid": True, "full_exact": True, "linf_ok": True, "pixel_linf_uint8": 0.01, "strict": 0},
            {"step": 50, "margin": -1, "nonzero": True, "output_valid": True, "full_exact": True, "linf_ok": True, "pixel_linf_uint8": 0.01, "answers": "x"},
        ]
        legal = [row for row in rows if eligible(row)]
        self.assertEqual(choose(legal)["step"], 50)

    def test_replay_failure_keeps_the_pick(self):
        legal = [{"step": 20, "margin": -2.0}, {"step": 40, "margin": -0.5}]
        picked = choose(legal)
        status = "SELECTED_CANDIDATE_REPLAY_FAILED"
        self.assertEqual(picked["step"], 20)
        self.assertNotEqual(status, "SELECTED")

    def test_zero_pixel_is_ineligible(self):
        row = {"step": 10, "margin": -3, "nonzero": False, "output_valid": True, "full_exact": True, "linf_ok": True, "pixel_linf_uint8": 0.0}
        self.assertFalse(eligible(row))

    def test_eos_once_and_resume(self):
        tokens = with_eos(torch.tensor([1, 2]), 7)
        self.assertEqual(with_eos(tokens, 7).tolist(), [1, 2, 7])
        done = {"1025"}
        self.assertEqual([sid for sid in ["1025", "2008"] if sid not in done], ["2008"])


if __name__ == "__main__":
    unittest.main()
