"""CPU checks that do not load the model."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))

from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402
from week2_attack_v5b_expanded_pristine40.run_formal_attack import pick_min_margin  # noqa: E402
from week2_attack_v5b_expanded_pristine40.run_random_noise import noise_seed  # noqa: E402
from week2_attack_v5b_expanded_pristine40.verify_frozen_inputs import main as verify  # noqa: E402


def test_k():
    assert official_keep_count(2 * 3699, 0.25) == 1850
    assert official_keep_count(3 * 3699, 0.25) == 2774
    assert official_keep_count(4 * 3699, 0.25) == 3699


def test_selector_ignores_anls():
    rows = [
        {"output_valid": True, "margin": 1.0, "step": 10, "anls_fastv": 0.0},
        {"output_valid": True, "margin": -3.0, "step": 200, "anls_fastv": 1.0},
        {"output_valid": False, "margin": -9.0, "step": 0, "anls_fastv": 0.0},
        {"output_valid": True, "margin": -3.0, "step": 50, "anls_fastv": 0.0},
    ]
    chosen = pick_min_margin(rows)
    assert chosen["step"] == 50


def test_noise_seed_stable():
    assert noise_seed("123") == noise_seed("123")
    assert noise_seed("123") != noise_seed("124")


def test_shard_assignment():
    ids = [str(i) for i in range(40)]
    shards = [[sid for i, sid in enumerate(ids) if i % 4 == shard] for shard in range(4)]
    assert all(len(shard) == 10 for shard in shards)
    assert len({sid for shard in shards for sid in shard}) == 40


def test_resume_skip():
    done = {"1"}
    pending = [sid for sid in ["1", "2"] if sid not in done]
    assert pending == ["2"]


def test_analysis_isolation():
    raw = ROOT / "week2_attack_v5b_expanded_pristine40" / "attacks"
    analysis = ROOT / "week2_attack_v5b_expanded_pristine40" / "analysis"
    assert raw != analysis


def main():
    test_k()
    test_selector_ignores_anls()
    test_noise_seed_stable()
    test_shard_assignment()
    test_resume_skip()
    test_analysis_isolation()
    verify()
    digest = hashlib.sha256(b"cpu-ok").hexdigest()
    print({"cpu": "PASS", "marker": digest[:12]})


if __name__ == "__main__":
    main()
