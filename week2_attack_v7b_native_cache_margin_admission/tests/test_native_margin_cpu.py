"""CPU checks for native-cache margin rules. No model and no GPU."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (
    certificate_status,
    cone_holds,
    greedy_ids_if_cone,
    hard_margin_row,
    margins_for_tokens,
    require_keep_match,
    soft_margin_row,
    termination_of,
)

MODULE = Path(__file__).resolve().parents[1] / "native_cache_margin.py"


class NativeMarginCpuTest(unittest.TestCase):
    def test_truncated_sequence_has_no_appended_eos(self):
        tokens = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        self.assertEqual(termination_of(tokens, eos_token_id=99), "MAX_NEW_TOKENS_TRUNCATED")
        rows = [torch.tensor([0.0, 5.0, 1.0]) for _ in tokens]
        packed = margins_for_tokens(rows, [1] * len(tokens))
        self.assertEqual(len(packed["hard"]), len(tokens))
        self.assertNotIn(99, tokens)

    def test_eos_is_scored_only_when_generated(self):
        eos = 7
        tokens = [3, 4, eos]
        self.assertEqual(termination_of(tokens, eos), "EOS_GENERATED")
        rows = [torch.zeros(8) for _ in tokens]
        for index, token in enumerate(tokens):
            rows[index][token] = 4.0
        packed = margins_for_tokens(rows, tokens)
        self.assertEqual(len(packed["hard"]), 3)
        self.assertGreater(float(packed["hard"][-1]), 0)

    def test_cone_is_one_directional(self):
        logits = torch.tensor([[4.0, 0.0, 1.0], [0.2, 3.0, 0.0]])
        tokens = torch.tensor([0, 1])
        self.assertTrue(torch.equal(greedy_ids_if_cone(logits, tokens), tokens))
        self.assertEqual(certificate_status(True, True), "CERTIFICATE_AVAILABLE")
        self.assertEqual(certificate_status(False, True), "FULL_EXACT_CERTIFICATE_UNAVAILABLE")
        source = MODULE.read_text()
        self.assertNotIn("full_exact == cone_holds", source)
        self.assertNotIn("cone_holds == full_exact", source)

    def test_competitor_excludes_the_target_token(self):
        logits = torch.tensor([1.0, 5.0, 4.0])
        self.assertAlmostEqual(float(hard_margin_row(logits, 1)), 1.0)
        expected = 5.0 - torch.logsumexp(torch.tensor([1.0, 4.0]), dim=0)
        self.assertAlmostEqual(float(soft_margin_row(logits, 1)), float(expected), places=5)

    def test_native_module_does_not_call_manual_logits(self):
        source = MODULE.read_text()
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.extend(alias.name for alias in node.names)
        self.assertNotIn("full_answer_logits", source)
        self.assertNotIn("full_answer_logits", imported)

    def test_fastv_metadata_must_match(self):
        require_keep_match(4, "abc", 4, "abc")
        with self.assertRaises(RuntimeError):
            require_keep_match(None, "abc", 4, "abc")
        with self.assertRaises(RuntimeError):
            require_keep_match(4, "abc", 5, "abc")


if __name__ == "__main__":
    unittest.main()
