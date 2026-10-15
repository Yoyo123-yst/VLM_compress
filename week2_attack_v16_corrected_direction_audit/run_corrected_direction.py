"""Minimal corrected audit for the previously mis-specified error-state run.

This is a diagnostic, not a method result.  It uses one frozen successful
60700 PNG and checks whether an explicit *margin-increasing* update keeps the
same differentiable point as the saved/evaluated PNG.
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v16_corrected_direction_audit"
V13 = ROOT / "week2_attack_v13_branch_reachability_probe"
sys.path[:0] = [
    str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"),
    str(ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"),
    str(V13),
]
import run_branch_probe as probe  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash, image_mask_of  # noqa: E402
from week2_attack_v4_answer_aligned.forward import build_embeds, score_prompt  # noqa: E402
from week2_pilot.attack import normalized_tiles  # noqa: E402
from metrics_task import anls_score  # noqa: E402

SAMPLE = "60700"
RATIO = 0.25
EPS = 8 / 255
STEPS = 3
LR = 2 / 255
LAMBDA = float(os.environ.get("LAMBDA_FULL", str(probe.LAMBDA_FULL)))
SEL_LAMBDA = float(os.environ.get("SEL_LAMBDA", "0"))
START = V13 / "runs/state_retention_v1/60700/AWARE_FREEZE/step_011.png"
REFERENCE = [4406, 16680, 151645]  # natural FastV error PPSB at the start PNG
OUT = HERE / os.environ.get("OUT_TAG", "60700_margin_increase")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(full: dict) -> bool:
    return bool(full["full_text_exact"] and full["full_token_exact"] and full["full_replay"])


def selection_stability_loss(model, inputs, spans, aux_idx, page, kept):
    """Differentiable margin between the observed Top-K kept/dropped boundary."""
    _, mask = image_mask_of(model, inputs["input_ids"])
    tiles = normalized_tiles(page).to(dtype=inputs["pixel_values"].dtype).unsqueeze(0)
    embeds = build_embeds(
        model, inputs["input_ids"], mask, inputs["pixel_values"],
        inputs["image_sizes"], aux_idx, tiles,
    )
    scores = score_prompt(model.model.language_model, embeds)
    visual = [p for start, end in spans for p in range(start, end)]
    visual_index = torch.tensor(visual, device=page.device, dtype=torch.long)
    values = scores.index_select(0, visual_index)
    kept_set = set(int(x) for x in kept)
    keep_mask = torch.tensor([int(x) in kept_set for x in visual], device=page.device, dtype=torch.bool)
    if bool(keep_mask.all()) or bool((~keep_mask).all()):
        return scores.sum() * 0.0, None
    gap = values[keep_mask].min() - values[~keep_mask].max()
    # Minimize this loss to increase the boundary gap around the observed set.
    return torch.nn.functional.softplus(-gap), float(gap.detach())


def main() -> None:
    probe.configure_determinism()
    model, processor = base.load_model_frozen()
    model.eval()
    item = v7d.load_dev(SAMPLE)
    start = Image.open(START).convert("RGB")
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = start
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    base_page = v7d.page_to_float(item["images"][item["aux_idx"]]).to("cuda")
    start_page = v7d.page_to_float(start).to("cuda")
    state = {"inputs": inputs, "spans": spans, "page": start_page, "base_page": base_page}
    clean_inputs = v7d.prepare(processor, item["images"], item["meta"]["question"])
    clean_spans = v7d.spans_of(clean_inputs, int(model.config.image_token_index), len(item["images"]))
    clean_tokens, _ = v7d.generate_tokens(model, clean_inputs, clean_inputs["pixel_values"], clean_spans, None)
    state["clean_tokens"] = clean_tokens
    state["clean_text"] = v7d.decode(processor, torch.tensor(clean_tokens)).strip()
    with torch.no_grad():
        clean_rows = v7d.full_token_rows(
            model, clean_inputs, base_page, item["aux_idx"], clean_tokens, False
        )
        state["clean_hard"] = torch.stack([
            v7d.hard_margin_row(row, token) for row, token in zip(clean_rows, clean_tokens)
        ]).detach()
    delta = (start_page - base_page).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=LR)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []

    for step in range(STEPS + 1):
        # The exact same point is used for gradients, PNG construction, and guards.
        page = (base_page + delta.detach()).clamp(0, 1)
        png = page_uint8(base_page, page - base_page)
        path = OUT / f"step_{step:03d}.png"
        png.save(path)
        full = base.full_exact_check(model, processor, item, state, png)
        attacked_now = list(item["images"])
        attacked_now[item["aux_idx"]] = png
        ii = v7d.prepare(processor, attacked_now, item["meta"]["question"])
        ss = v7d.spans_of(ii, int(model.config.image_token_index), len(attacked_now))
        ft, kept = v7d.generate_tokens(model, ii, ii["pixel_values"], ss, RATIO)
        text = v7d.decode(processor, torch.tensor(ft)).strip()
        anls = anls_score(text, list(item["meta"]["answers"]))
        records.append({
            "step": step, "fast_text": text, "fast_tokens": ft,
            "fast_anls": float(anls), "full_exact": exact(full),
            "keep_hash": keep_hash(kept), "png_sha256": sha(path),
            "linf_uint8": int(np.max(np.abs(np.asarray(png, dtype=np.int16) -
                                             np.asarray(item["images"][item["aux_idx"]], dtype=np.int16)))),
        })
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
        if step == STEPS:
            break

        opt.zero_grad(set_to_none=True)
        leaf = (base_page + delta).clamp(0, 1)
        rows, _, kept_now = probe.fast_rows_for_reference(model, state, item, leaf, REFERENCE)
        # mean_soft is a target-vs-rivals margin.  Negating it makes Adam
        # increase the observed error-sequence margin instead of suppressing it.
        attack = -v7d.mean_soft(rows, REFERENCE)
        fdp, _ = base.fdp_value(model, state, item, leaf)
        sel, sel_gap = selection_stability_loss(
            model, state["inputs"], state["spans"], item["aux_idx"], leaf, kept_now
        ) if SEL_LAMBDA else (leaf.sum() * 0.0, None)
        total = attack + LAMBDA * fdp + SEL_LAMBDA * sel
        total.backward()
        old_delta = delta.detach().clone()
        old_opt = copy.deepcopy(opt.state_dict())
        opt.step()
        project(delta, base_page, EPS)
        candidate = page_uint8(base_page, (base_page + delta.detach()).clamp(0, 1) - base_page)
        guard = base.full_exact_check(model, processor, item, state, candidate)
        if not exact(guard):
            with torch.no_grad():
                delta.copy_(old_delta)
            opt.load_state_dict(old_opt)
        del rows, leaf, fdp, attack, sel, total, ii, ss
        torch.cuda.empty_cache()

    (OUT / "trajectory.json").write_text(json.dumps({
        "sample": SAMPLE, "ratio": RATIO, "reference": REFERENCE,
        "start_png": str(START.relative_to(ROOT)), "steps": records,
        "objective": "-mean_soft(reference)+lambda*full_fdp+sel_lambda*topk_boundary",
        "lambda_full": LAMBDA, "lambda_selection": SEL_LAMBDA,
        "same_gradient_and_png_point": True,
    }, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
