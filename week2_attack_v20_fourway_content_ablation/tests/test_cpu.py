"""CPU checks for selection, losses, the selector, semantics, and Adam rollback."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT)]

from week2_attack_v20_fourway_content_ablation.losses import (  # noqa: E402
    core_content_loss,
    dynamic_sequence_loss,
    fixed_all_loss,
    fixed_noeos_loss,
    full_fdp_loss,
)
from week2_attack_v20_fourway_content_ablation.optim import restore_adam, snapshot_adam  # noqa: E402
from week2_attack_v20_fourway_content_ablation.selection import (  # noqa: E402
    assert_core_not_silent_noeos,
    control_id_set,
    core_content_selection,
    fixed_all_selection,
    fixed_noeos_selection,
)
from week2_attack_v20_fourway_content_ablation.selector import choose_fixed  # noqa: E402
from week2_attack_v20_fourway_content_ablation.semantics import classify  # noqa: E402


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 8
    bos_token_id = None
    unk_token_id = None
    all_special_ids = [8, 9]

    def decode(self, ids, skip_special_tokens=False):
        table = {1: "PP", 2: "BS", 3: " March", 4: " 1968", 5: " Tobacco", 8: "<|pad|>", 9: "<|im_end|>"}
        return table[int(ids[0])]

    def __call__(self, text, add_special_tokens=False):
        mapping = {"PPBS": [1, 2], "March 1968": [3, 4], "Tobacco": [5]}
        if text not in mapping:
            return {"input_ids": [99]}
        return {"input_ids": mapping[text]}


class SelectionTests(unittest.TestCase):
    def test_eos_split_and_no_fallback(self):
        tokenizer = Tokenizer()
        reference = [1, 2, 9]
        controls = control_id_set(tokenizer)
        whole = fixed_all_selection(tokenizer, reference)
        trimmed = fixed_noeos_selection(tokenizer, reference, controls)
        core = core_content_selection(tokenizer, reference, ["PPBS"], trimmed.positions)
        missing = core_content_selection(tokenizer, reference, ["absent answer"], trimmed.positions)
        self.assertEqual(whole.positions, (0, 1, 2))
        self.assertIn(9, whole.token_ids)
        self.assertEqual(trimmed.positions, (0, 1))
        self.assertNotIn(9, trimmed.token_ids)
        self.assertEqual(core.positions, (0, 1))
        self.assertTrue(core.same_positions_as_noeos)
        self.assertFalse(missing.available)
        self.assertEqual(missing.positions, tuple())
        self.assertEqual(missing.reason, "CORE_SPAN_UNAVAILABLE")
        assert_core_not_silent_noeos(missing, trimmed)
        with self.assertRaises(RuntimeError):
            bad = type("S", (), {"available": False, "positions": trimmed.positions})()
            assert_core_not_silent_noeos(bad, trimmed)

    def test_core_can_be_a_proper_subset(self):
        tokenizer = Tokenizer()
        reference = [3, 4, 5, 9]
        trimmed = fixed_noeos_selection(tokenizer, reference, control_id_set(tokenizer))
        core = core_content_selection(tokenizer, reference, ["March 1968"], trimmed.positions)
        self.assertEqual(core.positions, (0, 1))
        self.assertFalse(core.same_positions_as_noeos)
        self.assertLess(len(core.positions), len(trimmed.positions))


class LossTests(unittest.TestCase):
    def test_four_losses_are_distinct_and_fdp_is_per_token(self):
        rows = [torch.tensor([0.0, 4.0, 1.0]), torch.tensor([3.0, 0.0, 1.0]), torch.tensor([0.2, 0.2, 5.0])]
        dynamic = dynamic_sequence_loss(rows[:2], [1, 0])
        fixed = fixed_all_loss(rows, [1, 0, 2])
        trimmed = fixed_noeos_loss(rows, [1, 0, 2], [0, 1])
        content = core_content_loss(rows, [1, 0, 2], [0])
        self.assertFalse(torch.allclose(dynamic, fixed))
        self.assertFalse(torch.allclose(fixed, trimmed))
        self.assertFalse(torch.allclose(trimmed, content))
        with self.assertRaises(RuntimeError):
            core_content_loss(rows, [1, 0, 2], [])
        hard = torch.tensor([1.0, 0.5, -0.25])
        clean = torch.tensor([1.2, 0.7, 0.1])
        fdp = full_fdp_loss(hard, clean)
        manual = torch.nn.functional.softplus(clean - hard).mean()
        self.assertTrue(torch.allclose(fdp, manual))
        with self.assertRaises(RuntimeError):
            full_fdp_loss(hard, clean[:-1])


class SelectorTests(unittest.TestCase):
    def test_lowest_surrogate_then_earlier_checkpoint(self):
        rows = [
            self.row(0, 1.0, False),
            self.row(5, 0.4, True),
            self.row(10, 0.4, True),
            self.row(15, 0.2, False),
        ]
        chosen = choose_fixed(rows)
        self.assertEqual(chosen["checkpoint"], 5)
        with self.assertRaises(RuntimeError):
            choose_fixed([{**rows[1], "anls": 0.0}])

    def row(self, checkpoint, surrogate, legal):
        return {
            "checkpoint": checkpoint,
            "nonzero": legal,
            "output_valid": True,
            "full_text_exact": True,
            "full_token_exact": True,
            "full_replay": True,
            "linf_ok": True,
            "png_reloadable": True,
            "attack_surrogate": surrogate,
        }


class SemanticTests(unittest.TestCase):
    def test_registered_cases(self):
        aliases = {
            "60700": ["ppbs"],
            "7006": ["rj reynolds tobacco company"],
            "2008": ["march 1968"],
        }
        self.assertEqual(classify("60700", "PPBS", aliases["60700"])["semantic_status"], "ANSWER_PRESERVED")
        self.assertEqual(classify("60700", "PPBS 14, 1967", aliases["60700"])["semantic_status"], "ANSWER_PRESERVED_AUGMENTED")
        self.assertEqual(classify("60700", "PPSB", aliases["60700"])["semantic_status"], "CONTENT_FAILURE")
        self.assertEqual(classify("60700", "Ppjs", aliases["60700"])["semantic_status"], "AMBIGUOUS_REQUIRES_REVIEW")
        self.assertEqual(classify("7006", "Tobacco Company", aliases["7006"])["semantic_status"], "AMBIGUOUS_REQUIRES_REVIEW")
        self.assertEqual(classify("7006", "RJ Reynolds Tobacco Company", aliases["7006"])["semantic_status"], "ANSWER_PRESERVED")
        self.assertEqual(classify("2008", "March 1956", aliases["2008"])["semantic_status"], "CONTENT_FAILURE")
        self.assertEqual(classify("2008", "March 1968, as advertised", aliases["2008"])["semantic_status"], "ANSWER_PRESERVED_AUGMENTED")
        self.assertEqual(classify("60700", "", aliases["60700"])["semantic_status"], "INVALID_OUTPUT")


class AdamTests(unittest.TestCase):
    def test_rollback_restores_moments_and_step(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        parameter.grad = torch.ones(2)
        optimizer.step()
        before = snapshot_adam(optimizer)
        previous = parameter.detach().clone()
        parameter.grad = torch.full((2,), 3.0)
        optimizer.step()
        self.assertFalse(torch.equal(parameter.detach(), previous))
        restore_adam(optimizer, before, parameter, previous)
        self.assertTrue(torch.equal(parameter.detach(), previous))
        restored = optimizer.state_dict()
        self.assertEqual(int(restored["state"][0]["step"]), int(before["state"][0]["step"]))
        self.assertTrue(torch.equal(restored["state"][0]["exp_avg"], before["state"][0]["exp_avg"]))
        self.assertTrue(torch.equal(restored["state"][0]["exp_avg_sq"], before["state"][0]["exp_avg_sq"]))
        copy.deepcopy(before)


if __name__ == "__main__":
    unittest.main()
