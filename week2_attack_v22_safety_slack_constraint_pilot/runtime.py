"""Shared loading, prepare-path scoring, and sign-PGD rollback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt
from week2_attack_v20_fourway_content_ablation.optim import sign_step
from week2_attack_v20_fourway_content_ablation.protocol import EPS, LAMBDA_FULL, STEP_ALPHA
from week2_attack_v21b_full_safety_slack_scan.measure import score_image
from week2_attack_v22_safety_slack_constraint_pilot.objectives import core_loss, hard_margins, native_fdp, safety_hinge
from week2_attack_v22_safety_slack_constraint_pilot.pixels import straight_through_pixels

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V21 = ROOT / "week2_attack_v21_context_leverage_causal_probe"


def load_groups() -> dict:
    return json.loads((V21 / "frozen_token_groups.json").read_text())


def reference_of(groups: dict) -> tuple[list[int], list[str], list[int]]:
    return (
        [int(row["token_id"]) for row in groups["tokens"]],
        [row["decoded_piece"] for row in groups["tokens"]],
        groups["s_core"]["positions"],
    )


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delta_hash(delta: torch.Tensor) -> str:
    return rt.hash_tensor(delta)


def prepared_inputs(processor, item, image):
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    return rt.v7d.prepare(processor, attacked, item["meta"]["question"])


def attack_grads(model, processor, item, state, image, page, reference, core_pos, clean_hard, tau, kind: str, cost: dict) -> dict:
    inputs = prepared_inputs(processor, item, image)
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(item["images"]))
    prepared = inputs["pixel_values"]
    leaf = page.detach().clone().requires_grad_(True)
    pixels = straight_through_pixels(prepared, item["aux_idx"], leaf)
    _tokens, kept = rt.v7d.generate_tokens(model, inputs, prepared, spans, 0.25)
    cost["generate"] += 1
    kept = [int(index) for index in kept]
    keep_index = rt.v7d.keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), prepared.device)
    rows = rt.v7d.fastv_rows(model, inputs, pixels, reference, keep_index)
    cost["forward"] += 1
    loss = core_loss(rows, reference, core_pos)
    loss.backward()
    cost["backward"] += 1
    attack = leaf.grad.detach().clone()
    leaf.grad = None
    side_value = 0.0
    side = torch.zeros_like(attack)
    if kind in {"T1", "T2"}:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        try:
            full = rt.v7d.full_rows(model, inputs, straight_through_pixels(prepared, item["aux_idx"], leaf), state["clean_tokens"])
        finally:
            model.eval()
            model.gradient_checkpointing_disable()
        cost["forward"] += 1
        adv_hard = hard_margins(full, state["clean_tokens"])
        penalty = native_fdp(adv_hard, clean_hard) if kind == "T1" else safety_hinge(adv_hard, tau)
        (LAMBDA_FULL * penalty).backward()
        cost["backward"] += 1
        side = leaf.grad.detach().clone()
        side_value = float(penalty.detach())
        del full, adv_hard, penalty
    if not torch.isfinite(attack).all() or not torch.isfinite(side).all():
        raise RuntimeError("non-finite gradient")
    value = float(loss.detach())
    del rows, loss, pixels, leaf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "l_core": value,
        "l_side": side_value,
        "attack_grad": attack,
        "side_grad": side,
        "update_grad": attack + side,
        "kept": kept,
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denom = float(left.float().norm() * right.float().norm())
    if denom == 0.0:
        return None
    return float(torch.dot(left.float().flatten(), right.float().flatten()) / denom)


def apply_sign(delta: torch.Tensor, clean: torch.Tensor, grad: torch.Tensor) -> None:
    sign_step(delta, clean, grad, STEP_ALPHA, EPS)


def score(model, processor, item, state, image, reference, pieces, core_pos, clean_kept, cost: dict) -> dict:
    return score_image(rt, model, processor, item, state, image, reference, pieces, core_pos, clean_kept, cost)


def pack_margins(scored: dict, pieces_full: list[str]) -> dict:
    full = scored["full_margins"]
    weakest = min(full, key=lambda row: row["hard_margin"])
    return {
        "full_hard_mean": sum(row["hard_margin"] for row in full) / len(full),
        "full_hard_min": min(row["hard_margin"] for row in full),
        "weakest_full_token": weakest,
        "core_hard_mean": sum(row["hard_margin"] for row in scored["core_margins"]) / len(scored["core_margins"]),
        "core_margins": scored["core_margins"],
        "full_margin_hash": scored["full_margin_hash"],
        "core_margin_hash": scored["core_margin_hash"],
        "pieces_note": pieces_full[weakest["position"]] if weakest["position"] < len(pieces_full) else weakest["piece"],
    }


def dump(path: Path, payload: dict) -> None:
    rt.write_json(path, payload)


def read_expected(name: str) -> dict:
    path = ROOT / "week2_attack_v21b_full_safety_slack_scan" / "scan" / "CORE" / name
    return json.loads(path.read_text())
