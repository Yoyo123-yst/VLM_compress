"""CPU checks for the sign step, exact uint8 radii, and slack labels."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from week2_attack_v20_fourway_content_ablation.optim import sign_step
from week2_attack_v4_answer_aligned.run_smoke import page_uint8
from week2_attack_v21b_full_safety_slack_scan.decisions import direction_summary, final_status, prefix_safe

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v21b_full_safety_slack_scan")


def _point(radius: int, legal: bool, progress: bool = True, mismatch: bool = False) -> dict:
    return {
        "requested_uint8_linf": radius,
        "full_legal": legal,
        "core_margin_decreased": progress,
        "fastv_text_changed": False,
        "quantization_radius_mismatch": mismatch,
    }


class SlackTests(unittest.TestCase):
    def test_sign_and_exact_radius(self) -> None:
        clean = torch.full((3, 8, 8), 0.5)
        grad = torch.ones_like(clean)
        delta = torch.zeros_like(clean)
        sign_step(delta, clean, grad, 4 / 255, 4 / 255)
        self.assertLess(float(delta.max()), 0.0)
        original = page_uint8(clean, torch.zeros_like(clean))
        attacked = page_uint8(clean, delta)
        import numpy as np
        gap = np.abs(np.asarray(attacked, dtype=np.int16) - np.asarray(original, dtype=np.int16)).max()
        self.assertEqual(int(gap), 4)

    def test_prefix_stops_at_first_failure(self) -> None:
        self.assertEqual(prefix_safe({1: True, 2: True, 4: False, 6: True, 8: True}), 2)

    def test_confirmed_requires_prefix_and_progress(self) -> None:
        core = direction_summary([_point(1, True), _point(2, True), _point(4, True), _point(6, False), _point(8, False)])
        other = direction_summary([_point(1, False, False)])
        self.assertEqual(final_status(core, other)["status"], "FULL_SAFETY_SLACK_CONFIRMED")

    def test_no_progress_is_not_usable(self) -> None:
        core = direction_summary([_point(1, True, False), _point(2, True, False), _point(4, True, False), _point(6, True, False), _point(8, True, False)])
        self.assertEqual(final_status(core, core)["status"], "NO_USABLE_FULL_SAFETY_SLACK")
        self.assertEqual(final_status(core, core)["next_step"], "NO_MULTISTEP_COMPARISON_AUTHORIZED")

    def test_radius_one_failure_is_not_usable(self) -> None:
        core = direction_summary([_point(1, False), _point(2, False), _point(4, False), _point(6, False), _point(8, False)])
        self.assertEqual(final_status(core, core)["status"], "NO_USABLE_FULL_SAFETY_SLACK")

    def test_only_small_prefix_is_limited(self) -> None:
        core = direction_summary([_point(1, True), _point(2, False), _point(4, False), _point(6, False), _point(8, False)])
        self.assertEqual(final_status(core, core)["status"], "LIMITED_FULL_SAFETY_SLACK")

    def test_sources_have_no_training_loop(self) -> None:
        for path in HERE.glob("*.py"):
            source = path.read_text()
            self.assertNotIn("unseen_manifest", source)
            self.assertNotIn("MAX_ACCEPTED", source)
            tree = ast.parse(source)
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    names.append(node.func.id)
            self.assertNotIn("run_train", names)


if __name__ == "__main__":
    unittest.main()
