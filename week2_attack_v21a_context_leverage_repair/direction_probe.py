"""One-step attack-only, Full-only, and combined directions. No multi-step training."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
HERE = ROOT / "week2_attack_v21a_context_leverage_repair"
V21 = ROOT / "week2_attack_v21_context_leverage_causal_probe"

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v20_fourway_content_ablation.losses import fixed_noeos_loss, full_fdp_loss  # noqa: E402
from week2_attack_v20_fourway_content_ablation.optim import sign_step  # noqa: E402
from week2_attack_v20_fourway_content_ablation.protocol import EPS, LAMBDA_FULL, STEP_ALPHA, TRAIN_RATIO  # noqa: E402
from week2_attack_v21a_context_leverage_repair.decisions import decide_directions  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import hard_margin_row, soft_margin_row  # noqa: E402

SCALES = (0.25, 0.5, 1.0, 2.0)
CORE_TEXT = "William J. Darby"


def pack_grad(grad: torch.Tensor) -> dict:
    flat = grad.detach().float()
    return {
        "hash": rt.hash_tensor(grad),
        "l1": float(flat.abs().sum()),
        "l2": float(flat.norm()),
        "linf": float(flat.abs().max()),
        "finite": bool(torch.isfinite(flat).all()),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denom = float(left.float().norm() * right.float().norm())
    if denom == 0.0:
        return None
    return float(torch.dot(left.float().flatten(), right.float().flatten()) / denom)


def sign_rates(left: torch.Tensor, right: torch.Tensor) -> dict:
    left_sign = left.detach().float().sign()
    right_sign = right.detach().float().sign()
    active = (left_sign != 0) & (right_sign != 0)
    if int(active.sum()) == 0:
        return {"element_agreement": None, "element_conflict": None, "pixel_conflict": None}
    agreement = float((left_sign[active] == right_sign[active]).float().mean())
    pixel = (left_sign * right_sign).sum(dim=0)
    active_pixels = pixel != 0
    conflict = float((pixel[active_pixels] < 0).float().mean()) if int(active_pixels.sum()) else None
    return {
        "element_agreement": agreement,
        "element_conflict": 1.0 - agreement,
        "pixel_conflict": conflict,
    }


def token_table(rows, reference, positions, pieces, clean_rows) -> list[dict]:
    table = []
    for index in positions:
        hard = float(hard_margin_row(rows[index], int(reference[index])).detach())
        soft = float(soft_margin_row(rows[index], int(reference[index])).detach())
        table.append({
            "position": index,
            "piece": pieces[index],
            "hard": hard,
            "soft": soft,
            "hard_delta": hard - clean_rows[index]["hard"],
            "soft_delta": soft - clean_rows[index]["soft"],
        })
    return table


def summarize_tokens(table: list[dict], clean_mean: float) -> dict:
    deltas = [row["hard_delta"] for row in table]
    return {
        "core_hard_mean": sum(row["hard"] for row in table) / len(table),
        "core_hard_min": min(row["hard"] for row in table),
        "core_hard_max": max(row["hard"] for row in table),
        "core_tokens_down": sum(delta < 0.0 for delta in deltas),
        "core_tokens_up": sum(delta > 0.0 for delta in deltas),
        "max_token_drop": min(deltas),
        "max_token_rise": max(deltas),
        "clean_core_hard_mean": clean_mean,
    }


def teacher(model, state, item, page, reference, kept):
    pixels = rt.v7d.pixels_from_page(state["inputs"], item["aux_idx"], page)
    keep_index = rt.v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
    return rt.v7d.fastv_rows(model, state["inputs"], pixels, reference, keep_index)


def official_fast(model, processor, item, image, reference, frozen_kept):
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    tokens, kept = rt.v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, TRAIN_RATIO)
    kept = [int(index) for index in kept]
    text = rt.v7d.decode(processor, torch.tensor(tokens)).strip()
    frozen_source = kept if frozen_kept is None else frozen_kept
    natural_index = rt.v7d.keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), inputs["pixel_values"].device)
    frozen_index = rt.v7d.keep_index_from_kept(frozen_source, spans, int(inputs["input_ids"].shape[1]), inputs["pixel_values"].device)
    with torch.no_grad():
        natural_rows = rt.v7d.fastv_rows(model, inputs, inputs["pixel_values"], reference, natural_index)
        frozen_rows = rt.v7d.fastv_rows(model, inputs, inputs["pixel_values"], reference, frozen_index)
    return {
        "text": text,
        "tokens": [int(token) for token in tokens],
        "kept": kept,
        "k": len(kept),
        "keep_hash": rt.keep_hash(kept),
        "natural_rows": natural_rows,
        "frozen_rows": frozen_rows,
    }


def baseline_tables(fast, reference, pieces, core_pos, context_pos) -> dict:
    def pack(rows, positions):
        return {
            index: {
                "hard": float(hard_margin_row(rows[index], int(reference[index])).detach()),
                "soft": float(soft_margin_row(rows[index], int(reference[index])).detach()),
            }
            for index in positions
        }
    return {
        "natural_core": pack(fast["natural_rows"], core_pos),
        "natural_context": pack(fast["natural_rows"], context_pos),
        "frozen_core": pack(fast["frozen_rows"], core_pos),
        "frozen_context": pack(fast["frozen_rows"], context_pos),
    }


def main() -> None:
    phase_a = json.loads((HERE / "phase_a_decision.json").read_text())
    if not phase_a["enter_phase_b"]:
        raise SystemExit("phase A did not pass")
    groups = json.loads((V21 / "frozen_token_groups.json").read_text())
    reference = [int(row["token_id"]) for row in groups["tokens"]]
    pieces = [row["decoded_piece"] for row in groups["tokens"]]
    core_pos = groups["s_core"]["positions"]
    context_pos = groups["s_context"]["positions"]
    all_pos = groups["s_all"]["positions"]
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    state = rt.prepare_clean_state(model, processor, item)
    original = item["images"][item["aux_idx"]].convert("RGB")
    clean_fast = official_fast(model, processor, item, original, reference, None)
    frozen_kept = clean_fast["kept"]
    clean_tables = baseline_tables(clean_fast, reference, pieces, core_pos, context_pos)
    clean_core_soft = sum(row["soft"] for row in clean_tables["natural_core"].values()) / len(core_pos)
    clean_page = state["page"].detach()
    with torch.no_grad():
        _tokens, attack_kept = rt.v7d.generate_tokens(
            model, state["inputs"], rt.v7d.pixels_from_page(state["inputs"], item["aux_idx"], clean_page), state["spans"], TRAIN_RATIO
        )
    attack_kept = [int(index) for index in attack_kept]
    delta = torch.zeros_like(clean_page, requires_grad=True)
    rows = teacher(model, state, item, clean_page + delta, reference, attack_kept)
    losses = {
        "g_core": fixed_noeos_loss(rows, reference, core_pos),
        "g_context": fixed_noeos_loss(rows, reference, context_pos),
        "g_all": fixed_noeos_loss(rows, reference, all_pos),
    }
    loss_values = {name: float(value.detach()) for name, value in losses.items()}
    grads = {}
    for name, loss in losses.items():
        if delta.grad is not None:
            delta.grad = None
        loss.backward(retain_graph=True)
        grads[name] = delta.grad.detach().clone()
        if not torch.isfinite(grads[name]).all():
            raise RuntimeError(f"non-finite {name}")
    del rows, losses
    delta.grad = None
    hard = rt.hard_rows(model, state, item, clean_page + delta, state["clean_tokens"])
    fdp = full_fdp_loss(hard, state["clean_hard"])
    fdp_value = float(fdp.detach())
    fdp.backward()
    grads["g_fdp"] = delta.grad.detach().clone()
    if not torch.isfinite(grads["g_fdp"]).all():
        raise RuntimeError("non-finite g_fdp")
    del fdp, hard, delta
    directions = {
        "CORE_ATTACK_ONLY": grads["g_core"],
        "CONTEXT_ATTACK_ONLY": grads["g_context"],
        "ALL_ATTACK_ONLY": grads["g_all"],
        "FDP_ONLY": LAMBDA_FULL * grads["g_fdp"],
        "CORE_COMBINED": grads["g_core"] + LAMBDA_FULL * grads["g_fdp"],
        "CONTEXT_COMBINED": grads["g_context"] + LAMBDA_FULL * grads["g_fdp"],
        "ALL_COMBINED": grads["g_all"] + LAMBDA_FULL * grads["g_fdp"],
    }
    geometry = {"losses": loss_values, "fdp_loss": fdp_value, "lambda_full": LAMBDA_FULL, "grads": {}, "pairs": {}}
    for name, grad in grads.items():
        geometry["grads"][name] = pack_grad(grad)
    names = list(grads)
    for left_i, left_name in enumerate(names):
        for right_name in names[left_i + 1:]:
            geometry["pairs"][f"{left_name}__{right_name}"] = {
                "cosine": cosine(grads[left_name], grads[right_name]),
                **sign_rates(grads[left_name], grads[right_name]),
            }
    rt.write_json(HERE / "gradient_geometry.json", geometry)
    steps = {}
    csv_rows = []
    out_dir = HERE / "direction_results"
    for direction, grad in directions.items():
        steps[direction] = {}
        for scale in SCALES:
            step = torch.zeros_like(clean_page)
            sign_step(step, clean_page, grad, STEP_ALPHA * scale, EPS)
            image = rt.project_image(clean_page, step)
            path = out_dir / direction / f"scale_{scale:.2f}.png"
            rt.save_png(image, path)
            reloaded = rt.reload_png(path)
            guard = rt.full_guard(model, processor, item, state, reloaded)
            fast = official_fast(model, processor, item, reloaded, reference, frozen_kept)
            natural_core = token_table(fast["natural_rows"], reference, core_pos, pieces, clean_tables["natural_core"])
            frozen_core = token_table(fast["frozen_rows"], reference, core_pos, pieces, clean_tables["frozen_core"])
            natural_context = token_table(fast["natural_rows"], reference, context_pos, pieces, clean_tables["natural_context"])
            frozen_context = token_table(fast["frozen_rows"], reference, context_pos, pieces, clean_tables["frozen_context"])
            natural_stats = summarize_tokens(natural_core, sum(row["hard"] for row in clean_tables["natural_core"].values()) / len(core_pos))
            frozen_stats = summarize_tokens(frozen_core, sum(row["hard"] for row in clean_tables["frozen_core"].values()) / len(core_pos))
            soft_now = sum(row["soft"] for row in natural_core) / len(natural_core)
            record = {
                "direction": direction,
                "scale": scale,
                "png_sha256": rt.sha_file(path),
                "float_image_hash": rt.hash_tensor(clean_page + step),
                "rgb_sha256": hashlib.sha256(np.asarray(reloaded, dtype=np.uint8).tobytes()).hexdigest(),
                "uint8_linf": rt.uint8_linf(original, reloaded),
                "png_reloaded": True,
                "full_text_exact": guard["full_text_exact"],
                "full_token_exact": guard["full_token_exact"],
                "full_replay": guard["full_replay"],
                "full_exact": bool(guard["full_text_exact"] and guard["full_token_exact"] and guard["full_replay"]),
                "output_valid": guard["output_valid"],
                "full_text": guard["full_text"],
                "natural_text": fast["text"],
                "natural_tokens": fast["tokens"],
                "k": fast["k"],
                "keep_hash": fast["keep_hash"],
                "keep_indices": fast["kept"],
                "keep_jaccard_vs_clean": rt.jaccard_kept(fast["kept"], frozen_kept),
                "core_name_intact": CORE_TEXT in fast["text"],
                "clean_core_hard_mean": natural_stats["clean_core_hard_mean"],
                "core_hard_mean_natural": natural_stats["core_hard_mean"],
                "core_hard_min_natural": natural_stats["core_hard_min"],
                "core_hard_max_natural": natural_stats["core_hard_max"],
                "core_tokens_down_natural": natural_stats["core_tokens_down"],
                "core_tokens_up_natural": natural_stats["core_tokens_up"],
                "max_token_drop_natural": natural_stats["max_token_drop"],
                "max_token_rise_natural": natural_stats["max_token_rise"],
                "core_hard_mean_frozen": frozen_stats["core_hard_mean"],
                "core_hard_min_frozen": frozen_stats["core_hard_min"],
                "core_hard_max_frozen": frozen_stats["core_hard_max"],
                "core_tokens_down_frozen": frozen_stats["core_tokens_down"],
                "core_tokens_up_frozen": frozen_stats["core_tokens_up"],
                "max_token_drop_frozen": frozen_stats["max_token_drop"],
                "max_token_rise_frozen": frozen_stats["max_token_rise"],
                "clean_attack_soft_loss": clean_core_soft,
                "attack_soft_loss_natural": soft_now,
                "core_natural": natural_core,
                "core_frozen": frozen_core,
                "context_natural": natural_context,
                "context_frozen": frozen_context,
            }
            steps[direction][f"{scale:.2f}"] = record
            for keep_name, table in (("natural", natural_core), ("frozen", frozen_core), ("natural_context", natural_context), ("frozen_context", frozen_context)):
                for token in table:
                    csv_rows.append({
                        "direction": direction,
                        "scale": scale,
                        "keep": keep_name,
                        "position": token["position"],
                        "piece": token["piece"],
                        "clean_hard": token["hard"] - token["hard_delta"],
                        "updated_hard": token["hard"],
                        "hard_delta": token["hard_delta"],
                        "clean_soft": token["soft"] - token["soft_delta"],
                        "updated_soft": token["soft"],
                        "soft_delta": token["soft_delta"],
                    })
            print(json.dumps({
                "direction": direction,
                "scale": scale,
                "mean_natural": record["core_hard_mean_natural"],
                "mean_frozen": record["core_hard_mean_frozen"],
                "down": record["core_tokens_down_natural"],
                "text": fast["text"],
                "full_exact": record["full_exact"],
                "linf": record["uint8_linf"],
            }), flush=True)
            del fast
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rt.write_json(out_dir / f"{direction}.json", steps[direction])
    decision_steps = {
        direction: {key: value for key, value in scale_rows.items() if key != "0.25"}
        for direction, scale_rows in steps.items()
    }
    decision = decide_directions({"toy_sign_ok": True, "steps": decision_steps})
    rt.write_json(HERE / "direction_summary.json", {"decision": decision, "clean_text": clean_fast["text"], "clean_keep_hash": rt.keep_hash(frozen_kept)})
    with (HERE / "per_token_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    rt.write_json(HERE / "final_decision.json", decision)
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
