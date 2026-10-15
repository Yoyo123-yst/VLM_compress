"""Compare autograd directional predictions with continuous surrogate forward."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(ROOT / "week2_attack_v23_differential_subspace_pilot")]
from week2_attack_v23_differential_subspace_pilot import pilot as v23  # noqa: E402
from week2_attack_v23_differential_subspace_pilot.pilot import CORE, margin_tensor, token_pieces  # noqa: E402
from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_pilot.attack import normalized_tiles  # noqa: E402


def differentiable_tiles(page: torch.Tensor) -> torch.Tensor:
    """Pure torch counterpart of the backward branch used by STE pixels."""
    high = page.view(3, 2, 384, 2, 384).permute(1, 3, 0, 2, 4).reshape(4, 3, 384, 384)
    resize_mode = __import__("os").environ.get("VATTACK_RESIZE_MODE", "bicubic")
    if resize_mode == "bilinear":
        base = F.interpolate(page.unsqueeze(0), size=(384, 384), mode="bilinear", align_corners=False).squeeze(0)
    else:
        antialias = __import__("os").environ.get("VATTACK_DETERMINISTIC_RESIZE", "0") != "1"
        base = F.interpolate(page.unsqueeze(0), size=(384, 384), mode="bicubic", align_corners=False, antialias=antialias).squeeze(0)
    return (torch.cat([base.unsqueeze(0), high], dim=0) - 0.5) / 0.5


def core_rows(model, processor, item, state, page, reference, positions):
    inputs = state["inputs"]
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(item["images"]))
    base_page = rt.v7d.page_to_float(item["images"][item["aux_idx"]].convert("RGB")).to(page.device)
    base_torch = differentiable_tiles(base_page)
    candidate_torch = differentiable_tiles(page)
    prepared_aux = inputs["pixel_values"][item["aux_idx"]].detach().to(dtype=candidate_torch.dtype)
    surrogate_aux = prepared_aux + (candidate_torch - base_torch)
    tiles = rt.v7d.splice_aux(inputs["pixel_values"], item["aux_idx"], surrogate_aux.to(inputs["pixel_values"].dtype))
    keep = rt.v7d.keep_index_from_kept(state["clean_fast_kept"], spans, int(inputs["input_ids"].shape[1]), tiles.device)
    with torch.no_grad(): rows = rt.v7d.fastv_rows(model, inputs, tiles, reference, keep)
    return [{"hard": float(rt.v7d.hard_margin_row(rows[p], int(reference[p]))),
             "soft": float(rt.v7d.soft_margin_row(rows[p], int(reference[p])))} for p in positions]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rt.configure(); model, processor = rt.v7d.load_model_frozen(); item = rt.v7d.load_dev(args.sample)
    state = rt.prepare_clean_state(model, processor, item); image = item["images"][item["aux_idx"]].convert("RGB")
    page = rt.v7d.page_to_float(image).to("cuda"); reference = list(state["clean_fast_tokens"]); positions = [int(x) for x in CORE[args.sample]]
    with torch.no_grad():
        clean_full = rt.v7d.full_rows(model, state["inputs"], state["inputs"]["pixel_values"], state["clean_tokens"])
        state["clean_soft"] = [float(x) for x in margin_tensor(clean_full, state["clean_tokens"]).cpu()]
    gs = v23.differentiable_state(model, processor, item, state, reference, positions, image)
    raw_sign = -gs["raw"].sign()
    clean = [float(x) for x in gs["core_soft"].tolist()]
    rows = []
    # Zero-step repeat is part of the diagnostic: it must be exactly stable.
    zero_a = core_rows(model, processor, item, state, page, reference, positions)
    zero_b = core_rows(model, processor, item, state, page, reference, positions)
    zero_a_mean = sum(x["soft"] for x in zero_a) / len(zero_a)
    zero_b_mean = sum(x["soft"] for x in zero_b) / len(zero_b)
    # Use tiny steps for the derivative check, and both signs for a central FD.
    for frac in [0.001, 0.005, 0.01, 0.025, 0.05, 0.125]:
        delta = raw_sign * (frac / 255.0)
        predicted = float(torch.stack(gs["core_grads"]).mean(dim=0).flatten() @ delta.flatten())
        actual = core_rows(model, processor, item, state, (page + delta).clamp(0, 1), reference, positions)
        opposite = core_rows(model, processor, item, state, (page - delta).clamp(0, 1), reference, positions)
        soft_mean = sum(x["soft"] for x in actual) / len(actual)
        hard_mean = sum(x["hard"] for x in actual) / len(actual)
        opposite_mean = sum(x["soft"] for x in opposite) / len(opposite)
        rows.append({"sample_id":args.sample,"pixels":frac,"predicted_change":predicted,
                     "clean_soft_mean":sum(clean)/len(clean),"zero_repeat_delta":zero_b_mean-zero_a_mean,
                     "surrogate_soft_mean":soft_mean,"opposite_soft_mean":opposite_mean,
                     "surrogate_hard_mean":hard_mean,"surrogate_margins":actual,
                     "surrogate_soft_change":soft_mean-zero_a_mean,
                     "central_fd_change":(soft_mean-opposite_mean)/2.0})
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    rt.write_json(out / "results.json", {"sample_id":args.sample,"rows":rows})


if __name__ == "__main__": main()
