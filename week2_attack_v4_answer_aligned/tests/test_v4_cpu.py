"""CPU checks for the v4 objective, labels, and discrete budget."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))

from adapters.fastv_select import official_keep_count as runtime_keep_count  # noqa: E402
from week2_attack_v4_answer_aligned.common import assert_protected  # noqa: E402
from week2_attack_v4_answer_aligned.loss import (  # noqa: E402
    classify_sample,
    compression_margin_loss,
    kl_clean_adv,
    normalized_total,
    official_keep_count,
    project_capped_simplex,
    quantize_in_ball,
    select_best,
)


def test_margin_direction():
    logits = torch.zeros(1, 4, requires_grad=True)
    loss = compression_margin_loss(logits, torch.tensor([1]))
    loss.backward()
    assert float(logits.grad[0, 1]) > 0
    assert float(logits.grad[0, 0]) < 0


def test_kl_zero_when_clean_matches_adv():
    logits = torch.randn(4, 9)
    log_p = logits.log_softmax(-1)
    value = float(kl_clean_adv(log_p, logits))
    assert abs(value) < 1e-5


def test_keep_count_matches_runtime_round():
    for n, ratio in ((14796, 0.25), (11097, 0.25), (7398, 0.25), (3699, 1.0)):
        assert official_keep_count(n, ratio) == runtime_keep_count(n, 1.0 - ratio)
    assert official_keep_count(11097, 0.25) != int(__import__("math").ceil(11097 * 0.25))


def test_budget_soft_topk_sums_to_k_and_share_le_one():
    scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    for k in (2, 4):
        weights = project_capped_simplex(scores, k)
        assert abs(float(weights.sum()) - k) < 1e-3
        assert float(weights.min()) >= -1e-5
        assert float(weights.max()) <= 1.0 + 1e-5
        aux = weights[:6]
        assert float(aux.sum() / k) <= 1.0 + 1e-5


def test_best_checkpoint_rule():
    rows = [
        {"step": 0, "output_valid": True, "anls_full": 1.0, "anls_fastv": 1.0, "margin": 3.0, "linf": 0.0},
        {"step": 10, "output_valid": True, "anls_full": 1.0, "anls_fastv": 0.2, "margin": 1.0, "linf": 0.03},
        {"step": 25, "output_valid": True, "anls_full": 1.0, "anls_fastv": 0.2, "margin": 0.5, "linf": 0.02},
        {"step": 50, "output_valid": True, "anls_full": 0.1, "anls_fastv": 0.0, "margin": -2.0, "linf": 0.01},
        {"step": 75, "output_valid": False, "anls_full": 1.0, "anls_fastv": 0.0, "margin": -3.0, "linf": 0.0},
    ]
    assert select_best(rows)["step"] == 25


def test_labels_are_exclusive():
    cases = {
        "SUCCESS_COMP_ONLY": classify_sample(1, 1, 1, 0.2, -1, 2, True),
        "FULL_LEAK": classify_sample(1, 1, 0.1, 0.0, -1, 2, True),
        "SURROGATE_NONFLIP": classify_sample(1, 1, 1, 0.8, 0.0, 2.0, True),
        "INVALID_OUTPUT": classify_sample(1, 1, 1, 0.0, -1, 2, False),
        "NO_EFFECT": classify_sample(1, 1, 1, 0.9, 1.9, 2.0, True),
    }
    assert len(set(cases.values())) == 5
    for name, label in cases.items():
        assert label == name


def test_protected_hash():
    assert_protected(["a", "b", "c"], ["z", "b", "c"], 0)
    try:
        assert_protected(["a", "b", "c"], ["a", "z", "c"], 0)
    except RuntimeError:
        return
    raise AssertionError("protected page change was accepted")


def test_uint8_ball():
    clean = torch.randint(0, 256, (3, 8, 8))
    adv = (clean.float() / 255.0) + (torch.rand(3, 8, 8) - 0.5) * (20 / 255)
    quantized = quantize_in_ball(clean, adv, 8)
    gap = (quantized.float() - clean.float()).abs().max()
    assert int(gap) <= 8
    reloaded = quantized.float() / 255.0
    original = clean.float() / 255.0
    assert float((reloaded - original).abs().max()) <= 8 / 255 + 1e-6


def test_evidence_idx_not_in_loss_api():
    from week2_attack_v4_answer_aligned import loss as loss_mod

    for name in ("compression_margin_loss", "kl_clean_adv", "normalized_total", "answer_margin"):
        fn = getattr(loss_mod, name)
        assert "evidence" not in inspect.signature(fn).parameters
        assert "evidence_idx" not in inspect.getsource(fn)


def test_normalized_total_uses_frozen_lambda():
    comp = torch.tensor(2.0)
    full = torch.tensor(4.0)
    total = normalized_total(comp, full, 2.0, 4.0)
    assert abs(float(total) - 2.0) < 1e-6


if __name__ == "__main__":
    test_margin_direction()
    test_kl_zero_when_clean_matches_adv()
    test_keep_count_matches_runtime_round()
    test_budget_soft_topk_sums_to_k_and_share_le_one()
    test_best_checkpoint_rule()
    test_labels_are_exclusive()
    test_protected_hash()
    test_uint8_ball()
    test_evidence_idx_not_in_loss_api()
    test_normalized_total_uses_frozen_lambda()
    print("cpu tests passed")
