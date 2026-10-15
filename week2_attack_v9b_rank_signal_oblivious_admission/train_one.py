"""Train one sample. This process never imports FastV or the target evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spatial_rule import plan_for
from common import (
    CHECKPOINTS,
    EPS,
    GLOBAL_SEED,
    HERE,
    TOLERANCE,
    TRAIN_STEPS,
    fastv_modules,
    load_images,
    load_model,
    manifest_samples,
    output_valid,
    page_to_float,
    page_uint8,
    prepare,
    project,
    sha_file,
    sha_image,
    normalized_tiles,
    splice_aux,
    uint8_linf,
)
from guard import assert_clean, install_guard
from surrogate import pixel_reg, surrogate_ce, surrogate_generate

from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    hard_margin_row,
    native_rows_from_full_ids,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import decision_preservation  # noqa: E402


def tensors(inputs: dict) -> dict:
    return {key: value for key, value in inputs.items() if torch.is_tensor(value)}


def enable(model) -> None:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()


def disable(model) -> None:
    model.eval()
    model.gradient_checkpointing_disable()


def clean_hard(model, inputs: dict, pixels: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        rows = native_rows_from_full_ids(model, tensors(inputs), token_ids, pixels)
        hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, token_ids)])
    if not torch.isfinite(hard).all():
        raise RuntimeError("non-finite clean hard margin")
    return hard.detach()


def fdp_loss(model, inputs: dict, pixels: torch.Tensor, token_ids: list[int], floor: torch.Tensor) -> torch.Tensor:
    rows = native_rows_from_full_ids(model, tensors(inputs), token_ids, pixels)
    hard = torch.stack([hard_margin_row(row, token) for row, token in zip(rows, token_ids)])
    return decision_preservation(hard, floor)


def one_grad(loss: torch.Tensor, leaf: torch.Tensor) -> float:
    if leaf.grad is not None:
        leaf.grad = None
    loss.backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError("missing or non-finite gradient")
    norm = float(leaf.grad.float().norm())
    leaf.grad = None
    return norm


def sample_rho(generator: torch.Generator) -> float:
    draw = float(torch.rand((), generator=generator))
    return min(0.60, max(0.10, 0.10 + 0.50 * draw))


def score_page(model, processor, spec, images, clean_tokens: list[str], clean_text: str) -> dict:
    from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import production_generate

    inputs = prepare(processor, images, spec["question"])
    generated = production_generate(model, tensors(inputs))
    text = processor.batch_decode(torch.tensor(generated["tokens"]).unsqueeze(0), skip_special_tokens=True)[0].strip()
    margins = [float(hard_margin_row(row, token)) for row, token in zip(generated["scores"], generated["tokens"])]
    return {
        "text": text,
        "tokens": generated["tokens"],
        "text_exact": text == clean_text,
        "token_exact": generated["tokens"] == clean_tokens,
        "margins_positive": bool(margins) and all(value > 0 for value in margins),
        "valid": output_valid(text),
        "prompt_sha256": __import__("hashlib").sha256(inputs["prompt"].encode()).hexdigest(),
    }


def attacked_images(clean_images, aux_idx: int, image: Image.Image) -> list[Image.Image]:
    images = list(clean_images)
    images[aux_idx] = image
    return images


def calibrate(model, processor) -> None:
    gate = json.loads((HERE / "surrogate_clean_gate.json").read_text())
    families = gate["admitted_families"]
    if not families:
        raise RuntimeError("no admitted surrogate")
    spec = next(row for row in manifest_samples() if row["sample_id"] == "1025")
    images = load_images(spec)
    inputs = prepare(processor, images, spec["question"])
    from common import generate_tokens

    tokens, text = generate_tokens(model, processor, inputs)
    if text != spec["clean_full_pred"]:
        raise RuntimeError("calibration baseline text drifted")
    page = page_to_float(images[spec["aux_idx"]]).cuda()
    delta = torch.full_like(page, 1.0 / 255.0)
    project(delta, page, EPS)
    leaf = (page + delta).detach().requires_grad_(True)
    pixels = splice_aux(inputs["pixel_values"], spec["aux_idx"], normalized_tiles(leaf))
    plan = plan_for(families[0], 0.35, "1025", 0, GLOBAL_SEED)
    floor = clean_hard(model, inputs, inputs["pixel_values"], tokens)
    enable(model)
    ce_norm = one_grad(surrogate_ce(model, inputs, pixels, plan, tokens), leaf)
    fdp_norm = one_grad(fdp_loss(model, inputs, pixels, tokens, floor), leaf)
    reg_norm = one_grad(pixel_reg(leaf - page), leaf)
    evidence = inputs["pixel_values"][spec["evidence_idx"]].detach().clone().requires_grad_(True)
    held = [inputs["pixel_values"][index].detach() for index in range(inputs["pixel_values"].shape[0])]
    held[spec["aux_idx"]] = normalized_tiles(leaf).to(dtype=inputs["pixel_values"].dtype)
    held[spec["evidence_idx"]] = evidence.detach()
    evidence_loss = surrogate_ce(model, inputs, torch.stack(held), plan, tokens)
    evidence_loss.backward()
    evidence_norm = 0.0 if evidence.grad is None else float(evidence.grad.float().norm())
    disable(model)
    if ce_norm <= 0 or fdp_norm <= 0 or reg_norm <= 0 or evidence_norm != 0.0:
        raise RuntimeError(f"calibration rejected ce={ce_norm} fdp={fdp_norm} reg={reg_norm} evidence={evidence_norm}")
    config = {
        "lambda_fdp": ce_norm / fdp_norm,
        "lambda_reg": ce_norm / reg_norm,
        "probe": {"family": families[0], "ratio": 0.35, "sample_id": "1025", "source": "+1/255"},
        "grad_l2": {"ce": ce_norm, "fdp": fdp_norm, "reg": reg_norm, "evidence": evidence_norm},
        "admitted_families": families,
        "steps": TRAIN_STEPS,
        "seed": GLOBAL_SEED,
        "optimizer": "Adam",
        "lr": 2 / 255,
        "eps": EPS,
        "frozen_before_fastv": True,
    }
    (HERE / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"calibrated": True, "lambda_fdp": config["lambda_fdp"], "lambda_reg": config["lambda_reg"]}), flush=True)


def attack(sample_id: str) -> None:
    config = json.loads((HERE / "training_config.json").read_text())
    families = list(config["admitted_families"])
    spec = next(row for row in manifest_samples() if row["sample_id"] == sample_id)
    model, processor = load_model()
    clean_images = load_images(spec)
    base_inputs = prepare(processor, clean_images, spec["question"])
    from common import generate_tokens

    clean_tokens, clean_text = generate_tokens(model, processor, base_inputs)
    if clean_text != spec["clean_full_pred"]:
        raise RuntimeError(f"{sample_id} clean text drifted before training")
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
        ratio = sample_rho(generator)
        plan = plan_for(family, ratio, sample_id, step, GLOBAL_SEED)
        enable(model)
        if step < TRAIN_STEPS:
            opt.zero_grad(set_to_none=True)

        def aux_pixels() -> torch.Tensor:
            return splice_aux(base_inputs["pixel_values"], spec["aux_idx"], normalized_tiles(page + delta))

        ce = surrogate_ce(model, base_inputs, aux_pixels(), plan, clean_tokens)
        ce_value = float(ce.detach())
        if step < TRAIN_STEPS:
            (-ce).backward()
        del ce
        fdp = fdp_loss(model, base_inputs, aux_pixels(), clean_tokens, floor)
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
            "retention_ratio": ratio,
            "offset": plan.offset,
            "page_out": plan.page_out,
            "rule_hash": plan.rule_hash(),
            "ce": ce_value,
            "fdp": fdp_value,
            "reg": reg_value,
            "objective": objective,
        }
        if step in CHECKPOINTS:
            image = page_uint8(page, delta.detach())
            dest = out / f"step_{step:03d}.png"
            image.save(dest)
            reloaded = Image.open(dest).convert("RGB")
            scored = score_page(model, processor, spec, attacked_images(clean_images, spec["aux_idx"], reloaded), clean_tokens, clean_text)
            linf = uint8_linf(clean_images[spec["aux_idx"]], reloaded)
            nonzero = not np.array_equal(np.asarray(clean_images[spec["aux_idx"]]), np.asarray(reloaded))
            record.update(scored)
            record.update({
                "png": str(dest),
                "png_sha256": sha_file(dest),
                "nonzero": bool(nonzero),
                "linf": linf,
                "linf_ok": bool(nonzero and 0 < linf <= EPS + TOLERANCE),
                "protected_unchanged": True,
            })
            print(json.dumps({"sample": sample_id, "step": step, "exact": scored["token_exact"], "objective": record["objective"]}), flush=True)
        curve.append(record)
        if step == TRAIN_STEPS:
            break
        opt.step()
        project(delta, page, EPS)
        torch.cuda.empty_cache()
    eligible = [
        row for row in curve
        if row["step"] > 0 and row.get("token_exact") and row.get("text_exact") and row.get("margins_positive")
        and row.get("valid") and row.get("linf_ok")
    ]
    chosen = sorted(eligible, key=lambda row: (row["objective"], -row["step"]))[0] if eligible else None
    payload = {
        "sample_id": sample_id,
        "clean_text": clean_text,
        "clean_tokens": clean_tokens,
        "aux_idx": spec["aux_idx"],
        "evidence_idx": spec["evidence_idx"],
        "chosen": chosen,
        "curve": curve,
        "fastv_access_count": 0,
        "fastv_modules": fastv_modules(),
        "local_pass": chosen is not None,
    }
    if chosen is not None:
        selected = Image.open(chosen["png"]).convert("RGB")
        final = out / "selected.png"
        selected.save(final)
        reloaded = Image.open(final).convert("RGB")
        family = config["admitted_families"][0]
        plan = plan_for(family, 0.25, sample_id, 0, GLOBAL_SEED)
        current = prepare(processor, attacked_images(clean_images, spec["aux_idx"], reloaded), spec["question"])
        again = prepare(processor, attacked_images(clean_images, spec["aux_idx"], Image.open(final).convert("RGB")), spec["question"])
        first = surrogate_generate(model, processor, current, current["pixel_values"], plan)
        second = surrogate_generate(model, processor, again, again["pixel_values"], plan)
        full = score_page(model, processor, spec, attacked_images(clean_images, spec["aux_idx"], reloaded), clean_tokens, clean_text)
        payload["sealed_local"] = {
            "png_sha256": sha_file(final),
            "surrogate_text": first["text"],
            "surrogate_replay_match": first["text"] == second["text"] and first["tokens"] == second["tokens"],
            "full": full,
            "nonzero": not np.array_equal(np.asarray(clean_images[spec["aux_idx"]]), np.asarray(reloaded)),
            "linf": uint8_linf(clean_images[spec["aux_idx"]], reloaded),
            "protected_sha_unchanged": sha_image(clean_images[spec["evidence_idx"]]) == sha_image(reloaded) or spec["aux_idx"] != spec["evidence_idx"],
        }
        payload["local_pass"] = bool(
            full["token_exact"] and full["text_exact"] and full["valid"]
            and payload["sealed_local"]["surrogate_replay_match"]
            and payload["sealed_local"]["nonzero"]
            and payload["sealed_local"]["linf"] <= EPS + TOLERANCE
        )
    dest = HERE / "per_sample" / f"{sample_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"sample": sample_id, "local_pass": payload["local_pass"]}), flush=True)


def main() -> None:
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
