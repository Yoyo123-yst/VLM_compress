"""CPU checks for the hinge, selector, and decision gate."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from week2_attack_v20_fourway_content_ablation.optim import restore_adam, sign_step, snapshot_adam
from week2_attack_v22_safety_slack_constraint_pilot.decisions import choose_selector, decide
from week2_attack_v22_safety_slack_constraint_pilot.objectives import safety_hinge, safety_tau
from week2_attack_v22_safety_slack_constraint_pilot.pixels import straight_through_pixels

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v22_safety_slack_constraint_pilot")


class PilotTests(unittest.TestCase):
    def test_hinge_is_inactive_above_the_floor(self) -> None:
        adv = torch.tensor([1.0, 0.05], requires_grad=True)
        tau = safety_tau(torch.tensor([1.0, 0.2]))
        loss = safety_hinge(adv, tau)
        loss.backward()
        self.assertEqual(float(adv.grad[0]), 0.0)
        self.assertLess(float(adv.grad[1]), 0.0)

    def test_straight_through_matches_forward(self) -> None:
        prepared = torch.randn(2, 5, 3, 384, 384)
        page = torch.rand(3, 768, 768, requires_grad=True)
        pixels = straight_through_pixels(prepared, 0, page)
        self.assertTrue(torch.equal(pixels.detach(), prepared.detach()))
        pixels.sum().backward()
        self.assertIsNotNone(page.grad)

    def test_selector_uses_only_loss_and_step(self) -> None:
        chosen = choose_selector([
            {"step": 0, "legal": True, "l_core": 2.0},
            {"step": 5, "legal": True, "l_core": 1.0},
            {"step": 10, "legal": True, "l_core": 1.0},
        ])
        self.assertEqual(chosen["step"], 5)
        with self.assertRaises(RuntimeError):
            choose_selector([{"step": 0, "legal": True, "l_core": 1.0, "semantic_status": "CONTENT_FAILURE"}])

    def test_sign_descends_and_adam_rollback_restores_moments(self) -> None:
        clean = torch.full((1, 3, 2, 2), 0.5)
        delta = torch.zeros_like(clean)
        sign_step(delta, clean, torch.ones_like(clean), 1 / 255, 8 / 255)
        self.assertLess(float(delta.max()), 0.0)
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        snapshot = snapshot_adam(optimizer)
        previous = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        restore_adam(optimizer, snapshot, parameter.data, previous)
        self.assertTrue(torch.equal(parameter.data, previous))

    def test_witness_requires_failure_and_full_exact(self) -> None:
        base = {"clean_core_hard_mean": 6.0, "checkpoints": [{"step": 0, "core_hard_mean": 6.0}, {"step": 1, "core_hard_mean": 5.0}]}
        t2 = {**base, "selector_core_hard_mean": 5.0, "semantic_status": "CONTENT_FAILURE", "full_exact": True}
        t1 = {**base, "checkpoints": [{"step": 0, "core_hard_mean": 6.0}, {"step": 5, "core_hard_mean": 5.5}], "selector_core_hard_mean": 5.5}
        self.assertEqual(decide(t1, t2, True)["status"], "SAFETY_SLACK_HINGE_INITIAL_WITNESS")
        t2["semantic_status"] = "ANSWER_PRESERVED"
        self.assertEqual(decide(t1, t2, True)["status"], "SURROGATE_IMPROVED_ENDPOINT_NOT_ESTABLISHED")
        t2["selector_core_hard_mean"] = 5.8
        t2["checkpoints"] = [{"step": 0, "core_hard_mean": 6.0}, {"step": 10, "core_hard_mean": 5.9}]
        self.assertEqual(decide(t1, t2, True)["status"], "NO_GO_SAFETY_SLACK_HINGE_ON_47450")

    def test_sources_do_not_call_the_forbidden_forward(self) -> None:
        for path in HERE.glob("*.py"):
            source = path.read_text()
            self.assertNotIn("pixels_from_page", source)
            self.assertNotIn("unseen_manifest", source)
            tree = ast.parse(source)
            names = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            self.assertNotIn("run_train", names)


if __name__ == "__main__":
    unittest.main()
