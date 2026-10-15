"""Gradient smoke and one-sample training. This process does not import FastV."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from PIL import Image

from common import (
    CHECKPOINTS,
    EPS,
    GLOBAL_SEED,
    HERE,
    TOLERANCE,
    TRAIN_STEPS,
    fastv_modules,
    generate_tokens,
    load_images,
    load_model,
    manifest_samples,
    normalized_tiles,
    output_valid,
    page_to_float,
    page_uint8,
    prepare,
    project,
    sha_file,
    splice_aux,
    uint8_linf,
)
from guard import assert_clean, install_guard
from native_full import decision_preservation, hard_margin_row, native_rows_from_full_ids
from spatial_pool import RATIOS
from surrogate import pixel_reg, plans_for, surrogate_ce, surrogate_generate


def tensors(inputs: dict) -> dict:
    return {key: value for key, value in inputs.items() if torch.is_tensor(value)}


def enable(model) -> None:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()


def disable(model) -> None:
    model.eval()
    model.gradient_checkpointing_disable()


def aux_pixels(base_inputs, aux_idx: int, page: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return splice_aux(base_inputs["pixel_values"], aux_idx, normalized_tiles(page + delta))


def clean_hard(model, inputs, pixels, token_ids) -> torch.Tensor:
    with torch.no_grad():
        rows = native_rows_from_full_ids(model, tensors(inputs), token_ids, pixels)
        hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, token_ids)])
    return hard.detach()


def fdp_loss(model, inputs, pixels, token_ids, floor) -> torch.Tensor:
    rows = native_rows_from_full_ids(model, tensors(inputs), token_ids, pixels)
    hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, token_ids)])
    return decision_preservation(hard, floor)


def backward_norm(loss: torch.Tensor, leaf: torch.Tensor) -> tuple[float, list[int]]:
    if leaf.grad is not None:
        leaf.grad = None
    loss.backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all() or leaf.grad.shape != leaf.shape:
        raise RuntimeError("missing, non-finite, or misshapen gradient")
    norm = float(leaf.grad.float().norm())
    leaf.grad = None
    return norm, list(leaf.shape)


def calibrate(model, processor) -> None:
    gate = json.loads((HERE / "surrogate_clean_gate.json").read_text())
    families = gate["admitted_families"]
    if not families:
        raise RuntimeError("no admitted family")
    spec = next(row for row in manifest_samples() if row["sample_id"] == "1025")
    images = load_images(spec)
    inputs = prepare(processor, images, spec["question"])
    tokens, text = generate_tokens(model, processor, inputs)
    if text != spec["clean_full_pred"] or tokens != [8847, 13, 356, 13, 425, 4110, 13979, 151645]:
        raise RuntimeError("calibration baseline drifted")
    page = page_to_float(images[spec["aux_idx"]]).cuda()
    delta = torch.full_like(page, 1.0 / 255.0)
    project(delta, page, EPS)
    leaf = (page + delta).detach().requires_grad_(True)
    plans = plans_for(families[0], 0.35, "1025", 0, 0, len(images), GLOBAL_SEED)
    floor = clean_hard(model, inputs, inputs["pixel_values"], tokens)
    enable(model)
    pixels = splice_aux(inputs["pixel_values"], spec["aux_idx"], normalized_tiles(leaf))
    ce = surrogate_ce(model, inputs, pixels, plans, tokens)
    ce_norm, shape = backward_norm(ce, leaf)
    pixels = splice_aux(inputs["pixel_values"], spec["aux_idx"], normalized_tiles(leaf))
    fdp = fdp_loss(model, inputs, pixels, tokens, floor)
    fdp_norm, _ = backward_norm(fdp, leaf)
    reg = pixel_reg(leaf - page)
    reg_norm, _ = backward_norm(reg, leaf)
    evidence = inputs["pixel_values"][spec["evidence_idx"]].detach().clone().requires_grad_(True)
    held = [inputs["pixel_values"][index].detach() for index in range(inputs["pixel_values"].shape[0])]
    held[spec["aux_idx"]] = normalized_tiles(leaf).to(dtype=inputs["pixel_values"].dtype)
    held[spec["evidence_idx"]] = evidence.detach()
    surrogate_ce(model, inputs, torch.stack(held), plans, tokens).backward()
    evidence_norm = 0.0 if evidence.grad is None else float(evidence.grad.float().norm())
    disable(model)
    finite = all(math.isfinite(value) and value > 0 for value in (ce_norm, fdp_norm, reg_norm)) and evidence_norm == 0.0
    report = {
        "status": "PASS" if finite else "FAIL_GRADIENT_SMOKE",
        "losses": {"ce": float(ce.detach()), "fdp": float(fdp.detach()), "reg": float(reg.detach())},
        "grad_l2": {"ce": ce_norm, "fdp": fdp_norm, "reg": reg_norm, "evidence": evidence_norm},
        "grad_shape": shape,
        "probe": {"family": families[0], "ratio": 0.35, "offset_id": 0, "source": "+1/255"},
    }
    (HERE / "gradient_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    if not finite:
        print(json.dumps({"status": "FAIL_GRADIENT_SMOKE"}), flush=True)
        return
    config = {
        "lambda_fdp": ce_norm / fdp_norm,
        "lambda_reg": ce_norm / reg_norm,
        "admitted_families": families,
        "ratios": list(RATIOS),
        "offset_ids": [0, 1, 2],
        "steps": TRAIN_STEPS,
        "seed": GLOBAL_SEED,
        "optimizer": "Adam",
        "lr": 2 / 255,
        "eps": EPS,
        "grad_l2": report["grad_l2"],
        "frozen_before_fastv": True,
    }
    (HERE / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "lambda_fdp": config["lambda_fdp"], "lambda_reg": config["lambda_reg"]}), flush=True)


def attack(sample_id: str) -> None:
    config = json.loads((HERE / "training_config.json").read_text())
    families = list(config["admitted_families"])
    ratios = list(config["ratios"])
    spec = next(row for row in manifest_samples() if row["sample_id"] == sample_id)
    model, processor = load_model()
    clean_images = load_images(spec)
    base_inputs = prepare(processor, clean_images, spec["question"])
    clean_tokens, clean_text = generate_tokens(model, processor, base_inputs)
    if clean_text != spec["clean_full_pred"]:
        raise RuntimeError(f"{sample_id} clean text drifted")
    floor = clean_hard(model, base_inputs, base_inputs["pixel_values"], clean_tokens)
    page = page_to_float(clean_images[spec["aux_idx"]]).cuda()
    delta = torch.zeros_like(page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED)
    out = HERE / "runs" / sample_id
    out.mkdir(parents=True, exist_ok=True)
    curve = []
    for step in range(TRAIN_STEPS + 1):
        family = families[int(torch.randint(0, len(families), (1,), generator=generator))]
        ratio = ratios[int(torch.randint(0, len(ratios), (1,), generator=generator))]
        offset_id = int(torch.randint(0, 3, (1,), generator=generator)) if family == "BasePreservingJitteredPool" else 0
        plans = plans_for(family, ratio, sample_id, step, offset_id, len(clean_images), GLOBAL_SEED)
        enable(model)
        if step < TRAIN_STEPS:
            opt.zero_grad(set_to_none=True)
        ce = surrogate_ce(model, base_inputs, aux_pixels(base_inputs, spec["aux_idx"], page, delta), plans, clean_tokens)
        ce_value = float(ce.detach())
        if step < TRAIN_STEPS:
            (-ce).backward()
        del ce
        fdp = fdp_loss(model, base_inputs, aux_pixels(base_inputs, spec["aux_idx"], page, delta), clean_tokens, floor)
        fdp_value = float(fdp.detach())
        if step < TRAIN_STEPS:
            (config["lambda_fdp"] * fdp).backward()
        del fdp
        reg = pixel_reg(delta)
        reg_value = float(reg.detach())
        if step < TRAIN_STEPS:
            (config["lambda_reg"] * reg).backward()
            if delta.grad is None or not torch.isfinite(delta.grad).all():
                raise RuntimeError(f"non-finite gradient at step {step}")
        del reg
        disable(model)
        objective = -ce_value + config["lambda_fdp"] * fdp_value + config["lambda_reg"] * reg_value
        record = {
            "step": step,
            "family": family,
            "ratio": ratio,
            "offset_id": offset_id,
            "objective": objective,
            "ce": ce_value,
            "fdp": fdp_value,
            "reg": reg_value,
        }
        if step in CHECKPOINTS:
            image = page_uint8(page, delta.detach())
            dest = out / f"step_{step:03d}.png"
            image.save(dest)
            reloaded = Image.open(dest).convert("RGB")
            scored_inputs = prepare(processor, [reloaded if index == spec["aux_idx"] else image_page for index, image_page in enumerate(clean_images)], spec["question"])
            tokens, text = generate_tokens(model, processor, scored_inputs)
            linf = uint8_linf(clean_images[spec["aux_idx"]], reloaded)
            nonzero = not np.array_equal(np.asarray(clean_images[spec["aux_idx"]]), np.asarray(reloaded))
            record.update({
                "text": text,
                "tokens": tokens,
                "text_exact": text == clean_text,
                "token_exact": tokens == clean_tokens,
                "valid": output_valid(text),
                "nonzero": bool(nonzero),
                "linf": linf,
                "linf_ok": bool(nonzero and linf <= EPS + TOLERANCE),
                "disk_replay": tokens == clean_tokens and text == clean_text,
                "png_sha256": sha_file(dest),
                "png": str(dest),
            })
            print(json.dumps({"sample": sample_id, "step": step, "exact": record["token_exact"], "objective": objective}), flush=True)
        curve.append(record)
        if step == TRAIN_STEPS:
            break
        opt.step()
        project(delta, page, EPS)
        torch.cuda.empty_cache()
    eligible = [
        row for row in curve
        if row["step"] > 0 and row.get("token_exact") and row.get("text_exact") and row.get("valid")
        and row.get("nonzero") and row.get("linf_ok") and row.get("disk_replay")
    ]
    chosen = sorted(eligible, key=lambda row: (row["objective"], -row["step"]))[0] if eligible else None
    payload = {
        "sample_id": sample_id,
        "clean_text": clean_text,
        "clean_tokens": clean_tokens,
        "aux_idx": spec["aux_idx"],
        "evidence_idx": spec["evidence_idx"],
        "chosen": None if chosen is None else {key: chosen[key] for key in chosen if key != "tokens"},
        "chosen_tokens": None if chosen is None else chosen["tokens"],
        "curve_checkpoints": [row for row in curve if row["step"] in CHECKPOINTS],
        "fastv_access_count": 0,
        "fastv_modules": fastv_modules(),
        "status": "SEALED_CANDIDATE" if chosen is not None else "NO_VALID_SEALED_CANDIDATE",
    }
    if chosen is not None:
        final = out / "selected.png"
        Image.open(chosen["png"]).convert("RGB").save(final)
        payload["selected_sha256"] = sha_file(final)
        payload["selected_png"] = str(final)
    dest = HERE / "per_sample" / f"{sample_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"sample": sample_id, "status": payload["status"]}), flush=True)


def main() -> None:
    import sys
    install_guard()
    mode = sys.argv[1]
    if mode == "calibrate":
        model, processor = load_model()
        calibrate(model, processor)
    elif mode == "attack":
        attack(sys.argv[2])
    else:
        raise SystemExit("usage: train_one.py calibrate|attack ID")
    assert_clean(mode)


if __name__ == "__main__":
    main()
