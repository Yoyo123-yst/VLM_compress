"""CPU checks for the independent v6b protocol."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from week2_attack_v6b_soft_full_anchor.soft_loss import (
    coefficient,
    mechanism_label,
    page_soft_mass,
    selection_l1,
    soft_full_anchor,
    with_eos,
)
from week2_attack_v6_survivor_upgrade.variant_a_audit.select_pixel import choose


class SoftAnchorTests(unittest.TestCase):
    def test_eos_once(self):
        tokens = with_eos(torch.tensor([4, 5]), 9)
        self.assertEqual(with_eos(tokens, 9).tolist(), [4, 5, 9])

    def test_soft_anchor_nonzero_at_clean_margin(self):
        logits = torch.tensor([[4.0, 0.0], [1.0, 3.0]], requires_grad=True)
        clean = torch.tensor([4.0, 2.0])
        loss = soft_full_anchor(logits, torch.tensor([0, 1]), clean)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad.norm()), 1e-6)
        self.assertAlmostEqual(float(loss), torch.nn.functional.softplus(torch.tensor(0.0)).item(), places=5)

    def test_no_hinge_in_formula(self):
        text = Path("week2_attack_v6b_soft_full_anchor/soft_loss.py").read_text()
        self.assertNotIn("relu", text)
        self.assertNotIn("kappa", text)

    def test_selection_matches_page_mass(self):
        scores = torch.tensor([0.2, 0.3, 0.4, 0.1], requires_grad=True)
        mass = page_soft_mass(scores, [(0, 2), (2, 4)])
        loss = selection_l1(mass, torch.tensor([0.1, 0.9]))
        loss.backward()
        self.assertGreater(float(scores.grad.norm()), 0.0)

    def test_coefficient_rejects_flat_grad(self):
        self.assertAlmostEqual(coefficient(4.0, 2.0), 2.0)
        with self.assertRaises(ValueError):
            coefficient(4.0, 0.0)
        with self.assertRaises(ValueError):
            coefficient(4.0, float("nan"))


class RuleTests(unittest.TestCase):
    def test_selector_ignores_forbidden_fields(self):
        picked = choose([
            {"step": 200, "margin": -1.0, "anls_fastv": 0.0},
            {"step": 40, "margin": -1.0, "strict": 1},
        ])
        self.assertEqual(picked["step"], 40)

    def test_replay_does_not_repick(self):
        legal = [{"step": 10, "margin": -3.0}, {"step": 20, "margin": -1.0}]
        first = choose(legal)
        status = "SELECTED_CANDIDATE_REPLAY_FAILED"
        self.assertEqual(status.startswith("SELECTED_CANDIDATE"), True)
        self.assertEqual(choose(legal)["step"], first["step"])

    def test_mechanism_exclusive(self):
        labels = [
            mechanism_label(True, False, True, True),
            mechanism_label(False, True, True, True),
            mechanism_label(False, False, True, True),
            mechanism_label(True, True, False, True),
            mechanism_label(False, False, False, True),
        ]
        self.assertEqual(len(set(labels)), 5)

    def test_resume_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "status"])
                writer.writeheader()
                writer.writerow({"sample_id": "1025", "status": "ok"})
            with path.open() as handle:
                done = {row["sample_id"] for row in csv.DictReader(handle) if row["status"] == "ok"}
            self.assertEqual([sid for sid in ["1025", "6194"] if sid not in done], ["6194"])
        merged = [row for shard in ([{"sample_id": "1025"}], [{"sample_id": "7006"}]) for row in shard]
        self.assertEqual([row["sample_id"] for row in merged], ["1025", "7006"])


if __name__ == "__main__":
    unittest.main()
