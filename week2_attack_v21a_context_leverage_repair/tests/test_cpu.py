"""CPU checks for sign descent, replay labels, and direction labels."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from week2_attack_v20_fourway_content_ablation.optim import sign_step
from week2_attack_v21a_context_leverage_repair.decisions import classify_replays, decide_directions

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v21a_context_leverage_repair")


def _path(tag: str, file_sha: str = "file") -> dict:
    return {
        "rgb_sha256": "rgb",
        "uint8_linf": 0,
        "processor_exact_hash": "proc",
        "keep_hash": "keep",
        "k": 3699,
        "generated_tokens": [1, 2],
        "natural_logits_hash": "nat",
        "frozen_logits_hash": "frz",
        "margin_hash": "margin",
        "file_sha256": file_sha,
        "memory_png_sha256": "file",
        "tag": tag,
    }


def _runs(file_sha: str = "file") -> list[dict]:
    return [{"paths": {name: _path(name, file_sha) for name in ("ORIGINAL", "ZERO_PROJECTED", "ZERO_DIRECT_SAVE")}} for _ in range(5)]


def _step(mean: float, loss: float, down: int = 0, intact: bool = True, exact: bool = True) -> dict:
    return {
        "full_exact": exact,
        "png_reloaded": True,
        "clean_core_hard_mean": 6.0,
        "core_hard_mean_natural": mean,
        "core_hard_mean_frozen": mean,
        "clean_attack_soft_loss": 1.0,
        "attack_soft_loss_natural": loss,
        "core_tokens_down_natural": down,
        "core_name_intact": intact,
    }


def _summary(context_mean: float, combined_mean: float, fdp_mean: float, core_mean: float = 5.0, core_loss: float = 0.5, down: int = 3) -> dict:
    scales = ("0.50", "1.00", "2.00")
    def pack(mean, loss=0.5, token_down=down):
        return {scale: _step(mean, loss, token_down) for scale in scales}
    return {
        "toy_sign_ok": True,
        "steps": {
            "CONTEXT_ATTACK_ONLY": pack(context_mean),
            "CONTEXT_COMBINED": pack(combined_mean),
            "FDP_ONLY": pack(fdp_mean, token_down=0),
            "CORE_ATTACK_ONLY": pack(core_mean, core_loss),
            "ALL_ATTACK_ONLY": pack(context_mean),
            "CORE_COMBINED": pack(combined_mean),
            "ALL_COMBINED": pack(combined_mean),
        },
    }


class RepairTests(unittest.TestCase):
    def test_sign_step_descends(self) -> None:
        clean = torch.full((1, 3, 2, 2), 0.5)
        delta = torch.zeros_like(clean)
        sign_step(delta, clean, torch.ones_like(clean), 1.0 / 255.0, 8.0 / 255.0)
        self.assertLess(float(delta.max()), 0.0)
        delta2 = torch.zeros_like(clean)
        sign_step(delta2, clean, -torch.ones_like(clean), 1.0 / 255.0, 8.0 / 255.0)
        self.assertGreater(float(delta2.min()), 0.0)

    def test_exact_replay(self) -> None:
        self.assertEqual(classify_replays(_runs())["status"], "ZERO_REPLAY_EXACT")

    def test_encoding_only(self) -> None:
        runs = _runs("other")
        for run in runs:
            run["paths"]["ORIGINAL"]["file_sha256"] = "file"
            run["paths"]["ORIGINAL"]["memory_png_sha256"] = "file"
        self.assertEqual(classify_replays(runs)["status"], "ENCODING_ONLY_DIFFERENCE")
        self.assertTrue(classify_replays(runs)["enter_phase_b"])

    def test_pixel_block(self) -> None:
        runs = _runs()
        runs[0]["paths"]["ZERO_PROJECTED"]["uint8_linf"] = 1
        self.assertEqual(classify_replays(runs)["status"], "BLOCKED_IMAGE_RECONSTRUCTION_MISMATCH")

    def test_processor_block(self) -> None:
        runs = _runs()
        runs[0]["paths"]["ZERO_PROJECTED"]["processor_exact_hash"] = "other"
        self.assertEqual(classify_replays(runs)["status"], "BLOCKED_PREPROCESSING_PATH_MISMATCH")

    def test_runtime_block(self) -> None:
        runs = _runs()
        runs[3]["paths"]["ORIGINAL"]["keep_hash"] = "other"
        for name in ("ZERO_PROJECTED", "ZERO_DIRECT_SAVE"):
            runs[3]["paths"][name]["keep_hash"] = "other"
        self.assertEqual(classify_replays(runs)["status"], "BLOCKED_FASTV_RUNTIME_NONREPEATABLE")

    def test_blocked_by_full(self) -> None:
        decision = decide_directions(_summary(5.0, 6.2, 6.4))
        self.assertEqual(decision["status"], "CONTEXT_EFFECT_BLOCKED_BY_FULL_CONSTRAINT")

    def test_survives_full(self) -> None:
        decision = decide_directions(_summary(5.0, 5.2, 6.4))
        self.assertEqual(decision["status"], "CONTEXT_EFFECT_SURVIVES_FULL_CONSTRAINT")

    def test_weak_token(self) -> None:
        summary = _summary(6.1, 6.1, 6.1, down=1)
        summary["steps"]["CONTEXT_ATTACK_ONLY"] = {scale: _step(6.1, 0.5, 1) for scale in ("0.50", "1.00", "2.00")}
        self.assertEqual(decide_directions(summary)["status"], "WEAK_SINGLE_TOKEN_COUPLING_ONLY")

    def test_strong_nogo(self) -> None:
        summary = _summary(6.2, 6.3, 6.4, down=0)
        self.assertEqual(decide_directions(summary)["status"], "NO_GO_CONTEXT_CAUSAL_EFFECT_STRONG")

    def test_proxy_mismatch(self) -> None:
        summary = _summary(6.2, 6.3, 6.4, core_mean=6.4, core_loss=0.4, down=0)
        self.assertEqual(decide_directions(summary)["status"], "CORE_OBJECTIVE_PROXY_MISMATCH")

    def test_sources_do_not_train_or_open_unseen(self) -> None:
        for path in HERE.glob("*.py"):
            tree = ast.parse(path.read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            names = []
            for node in calls:
                if isinstance(node.func, ast.Name):
                    names.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    names.append(node.func.attr)
            self.assertNotIn("run_train", names)
            source = path.read_text()
            self.assertNotIn("unseen_manifest", source)
            self.assertNotIn("MAX_ACCEPTED", source)


if __name__ == "__main__":
    unittest.main()
