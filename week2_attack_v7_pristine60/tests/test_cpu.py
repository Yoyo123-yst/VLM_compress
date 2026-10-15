"""CPU checks for the frozen selector and seeds. No GPU."""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

from week2_attack_v7_pristine60.selector import choose, eligible
from week2_attack_v7_pristine60.run_v7 import LAMBDA_SOFT, sample_seed

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7_pristine60")


class SelectorTests(unittest.TestCase):
    def test_lambda_literal(self):
        self.assertEqual(LAMBDA_SOFT, 2.049824826423478)

    def test_rejects_step_zero_and_inexact_full(self):
        self.assertFalse(eligible({
            "step": 0, "nonzero": True, "output_valid": True, "full_exact": True,
            "linf_ok": True, "pixel_linf_uint8": 8 / 255, "margin": -1,
        }))
        self.assertFalse(eligible({
            "step": 10, "nonzero": False, "output_valid": True, "full_exact": True,
            "linf_ok": True, "pixel_linf_uint8": 0, "margin": -9,
        }))
        self.assertFalse(eligible({
            "step": 10, "nonzero": True, "output_valid": True, "full_exact": False,
            "linf_ok": True, "pixel_linf_uint8": 8 / 255, "margin": -9,
        }))

    def test_rank_margin_then_earlier_step(self):
        pick = choose([
            {"margin": -1.0, "step": 50},
            {"margin": -2.0, "step": 200},
            {"margin": -2.0, "step": 100},
        ])
        self.assertEqual(pick["step"], 100)
        self.assertEqual(pick["margin"], -2.0)

    def test_choose_ignores_extra_fields(self):
        pick = choose([
            {"margin": -1.0, "step": 10, "anls_fastv": 0.0, "answers": "x", "success": 1},
            {"margin": -3.0, "step": 200, "anls_fastv": 1.0, "answers": "y", "success": 0},
        ])
        self.assertEqual(pick["step"], 200)

    def test_selector_source_has_no_label_reads(self):
        tree = ast.parse((ROOT / "selector.py").read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for banned in ("anls_score", "answers", "pred_fastv", "success"):
            self.assertNotIn(banned, names)

    def test_seed_stable(self):
        self.assertEqual(sample_seed(2026100904, "123", ""), sample_seed(2026100904, "123", ""))
        self.assertNotEqual(sample_seed(2026100904, "123", ""), sample_seed(2026100904, "124", ""))
        self.assertEqual(sample_seed(2026100904, "123", ""), int(hashlib.sha256(b"2026100904:123:").hexdigest()[:8], 16))


if __name__ == "__main__":
    unittest.main()
