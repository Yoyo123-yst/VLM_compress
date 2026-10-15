"""CPU checks for the frozen v6 rules. No model and no GPU."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from week2_attack_v6_survivor_upgrade.bc_math import (
    kappa_from_clean_margin,
    mechanism_label,
    page_soft_mass,
    scale_from_grad,
    selection_l1,
    sequence_hinge,
    with_eos,
)
from week2_attack_v6_survivor_upgrade.variant_a_audit.select_pixel import choose, pixel_linf
from PIL import Image
import numpy as np


class SelectorTests(unittest.TestCase):
    def test_choose_reads_only_margin_and_step(self):
        picked = choose([
            {"step": 200, "margin": -1.0, "anls_fastv": 0.0, "answers": "hidden"},
            {"step": 50, "margin": -1.0, "anls_fastv": 1.0},
            {"step": 100, "margin": -0.2, "strict": 1},
        ])
        self.assertEqual(picked["step"], 50)

    def test_nonzero_and_linf_are_pixels(self):
        clean = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        same = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        moved = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        moved.putpixel((1, 1), (8, 0, 0))
        self.assertEqual(pixel_linf(clean, same), 0.0)
        self.assertAlmostEqual(pixel_linf(clean, moved), 8 / 255)

    def test_replay_failure_does_not_repick(self):
        legal = [{"step": 100, "margin": -2.0}, {"step": 200, "margin": -1.0}]
        picked = choose(legal)
        status = "SELECTED_CANDIDATE_REPLAY_FAILED"
        self.assertEqual(picked["step"], 100)
        self.assertNotEqual(status, "SELECTED")
        self.assertEqual(choose(legal)["step"], picked["step"])

    def test_full_exact_filter(self):
        rows = [
            {"step": 10, "margin": -5.0, "full_exact": False},
            {"step": 20, "margin": -1.0, "full_exact": True},
        ]
        legal = [row for row in rows if row["full_exact"]]
        self.assertEqual(choose(legal)["step"], 20)


class LossTests(unittest.TestCase):
    def test_eos_is_included_once(self):
        tokens = with_eos(torch.tensor([11, 12]), 99)
        self.assertEqual(tokens.tolist(), [11, 12, 99])
        self.assertEqual(with_eos(tokens, 99).tolist(), [11, 12, 99])

    def test_hinge_covers_eos_position(self):
        logits = torch.tensor([[5.0, 0.0], [0.0, 4.0]], requires_grad=True)
        kappa = kappa_from_clean_margin(torch.tensor([2.0, 0.5]))
        loss = sequence_hinge(logits, torch.tensor([0, 1]), kappa)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(kappa.tolist(), [1.0, 0.5])

    def test_page_mass_is_differentiable(self):
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        mass = page_soft_mass(scores, [(0, 2), (2, 4)])
        loss = selection_l1(mass, torch.tensor([0.5, 0.5]))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(float(scores.grad.norm()), 0.0)

    def test_detached_page_has_no_grad(self):
        raw = torch.tensor([1.0, 2.0], requires_grad=True)
        used = raw.detach()
        with self.assertRaises(RuntimeError):
            used.sum().backward()
        self.assertIsNone(raw.grad)

    def test_mechanism_labels_are_exclusive(self):
        labels = [
            mechanism_label(True, False, True, True),
            mechanism_label(False, True, True, True),
            mechanism_label(False, False, True, True),
            mechanism_label(True, True, True, True),
            mechanism_label(False, False, False, True),
        ]
        self.assertEqual(len(set(labels)), 5)
        self.assertEqual(mechanism_label(True, True, True, False), "UNEXPLAINED_OR_INVALID")


class ShardTests(unittest.TestCase):
    def test_resume_skips_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "status"])
                writer.writeheader()
                writer.writerow({"sample_id": "1025", "status": "ok"})
            done = {row["sample_id"] for row in csv.DictReader(path.open()) if row["status"] == "ok"}
            pending = [sid for sid in ["1025", "2008"] if sid not in done]
            self.assertEqual(pending, ["2008"])

    def test_merge_keeps_every_sample(self):
        shards = [[{"sample_id": "1025"}], [{"sample_id": "2008"}, {"sample_id": "7006"}]]
        merged = [row for shard in shards for row in shard]
        self.assertEqual([row["sample_id"] for row in merged], ["1025", "2008", "7006"])

    def test_scale_formula(self):
        self.assertEqual(scale_from_grad(4.0, 2.0), 2.0)
        with self.assertRaises(ValueError):
            scale_from_grad(4.0, 0.0)


if __name__ == "__main__":
    unittest.main()
