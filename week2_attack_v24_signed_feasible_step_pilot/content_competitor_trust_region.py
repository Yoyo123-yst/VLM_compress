"""Small content-competitor trust-region pilot.

The only change from the repaired trust-region pilot is the surrogate target:
at each registered core position it lowers a fixed clean-prefix margin
z(reference)-z(non-equivalent competitor). Real PNG replay and Full exact are
still the acceptance criteria.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v24_signed_feasible_step_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(ROOT / "week2_attack_v23_differential_subspace_pilot")]

from week2_attack_v23_differential_subspace_pilot.pilot import CORE, margin_tensor, score_image, sha_tensor, token_pieces  # noqa: E402
from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v22_safety_slack_constraint_pilot.pixels import straight_through_pixels  # noqa: E402


def clean_candidates(model, processor, item, state, reference, positions):
    inputs = state["inputs"]
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(item["images"]))
    keep = rt.v7d.keep_index_from_kept(state["clean_fast_kept"], spans, int(inputs["input_ids"].shape[1]), inputs["pixel_values"].device)
    with torch.no_grad():
        rows = rt.v7d.fastv_rows(model, inputs, inputs["pixel_values"], reference, keep)
    result = {}
    eligible = []
    for p in positions:
        ref_piece = processor.tokenizer.decode([int(reference[p])], skip_special_tokens=False)
        ref_norm = re.sub(r"[^a-z0-9]+", "", ref_piece.lower())
        # Keep complete content tokens only. A leading-space token is a word
        # boundary; position zero is allowed to start a word. Punctuation,
        # one-letter initials and continuation fragments are excluded.
        if ref_norm and len(ref_norm) >= 2 and (ref_piece[:1].isspace() or p == 0):
            eligible.append(int(p))
    if not eligible:
        raise RuntimeError("no eligible complete content positions")
    for p in eligible:
        ref_piece = processor.tokenizer.decode([int(reference[p])], skip_special_tokens=False)
        chosen = None
        for tok in torch.topk(rows[p].float(), k=min(64, rows[p].numel())).indices.tolist():
            piece = processor.tokenizer.decode([int(tok)], skip_special_tokens=False)
            norm = re.sub(r"[^a-z0-9]+", "", piece.lower())
            ref_norm = re.sub(r"[^a-z0-9]+", "", ref_piece.lower())
            boundary = piece[:1].isspace() or p == 0
            if int(tok) != int(reference[p]) and boundary and norm and len(norm) >= 2 and norm != ref_norm and ref_norm not in norm and norm not in ref_norm:
                chosen = (int(tok), piece)
                break
        if chosen is None:
            raise RuntimeError(f"no non-equivalent competitor for core position {p}")
        result[int(p)] = {"reference": int(reference[p]), "reference_piece": ref_piece, "competitor": chosen[0], "competitor_piece": chosen[1]}
    return result


def competitor_state(model, processor, item, state, image, reference, positions, candidates):
    attacked = list(item["images"]); attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    prepared = inputs["pixel_values"]
    page = rt.v7d.page_to_float(image).to(prepared.device).detach().requires_grad_(True)
    pixels = straight_through_pixels(prepared, item["aux_idx"], page)
    generated, kept = rt.v7d.generate_tokens(model, inputs, prepared, spans, 0.25)
    keep_index = rt.v7d.keep_index_from_kept([int(x) for x in kept], spans, int(inputs["input_ids"].shape[1]), prepared.device)
    rows = rt.v7d.fastv_rows(model, inputs, pixels, reference, keep_index)
    losses = []
    grads = []
    margins = []
    for p in positions:
        ref = candidates[int(p)]["reference"]
        comp = candidates[int(p)]["competitor"]
        loss = rows[p][ref] - rows[p][comp]
        losses.append(loss)
        grads.append(torch.autograd.grad(loss, page, retain_graph=True)[0].detach().clone())
        margins.append(float(loss.detach()))
    return {"raw": torch.stack(grads).mean(dim=0), "grads": grads, "margins": margins, "generated": [int(x) for x in generated], "kept": [int(x) for x in kept]}


def fixed_competitor(model, processor, item, state, image, reference, positions, candidates):
    attacked = list(item["images"]); attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    keep = rt.v7d.keep_index_from_kept(state["clean_fast_kept"], spans, int(inputs["input_ids"].shape[1]), inputs["pixel_values"].device)
    with torch.no_grad(): rows = rt.v7d.fastv_rows(model, inputs, inputs["pixel_values"], reference, keep)
    return [float(rows[p][candidates[int(p)]["reference"]] - rows[p][candidates[int(p)]["competitor"]]) for p in positions]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rt.configure(); model, processor = rt.v7d.load_model_frozen(); item = rt.v7d.load_dev(args.sample); state = rt.prepare_clean_state(model, processor, item)
    reference = list(state["clean_fast_tokens"]); raw_positions = [int(x) for x in CORE[args.sample]]; pieces = token_pieces(processor, reference)
    positions = raw_positions
    candidates = clean_candidates(model, processor, item, state, reference, positions)
    positions = sorted(candidates)
    current = item["images"][item["aux_idx"]].convert("RGB")
    cost = {"forward": 0, "backward": 0, "generate": 0}
    current_score = score_image(rt, model, processor, item, state, current, reference, pieces, positions, list(state["clean_fast_kept"]), cost)
    current_margin = sum(fixed_competitor(model, processor, item, state, current, reference, positions, candidates)) / len(positions)
    rows = []; accepted = 0
    for step in range(1, int(args.steps) + 1):
        cs = competitor_state(model, processor, item, state, current, reference, positions, candidates)
        accepted_this = False
        for radius in [1, 2]:
            page = rt.v7d.page_to_float(current).to("cuda")
            delta = -(float(radius) / 255.0) * cs["raw"].sign()
            path = out / f"step_{step:02d}_r{radius}.png"; rt.save_png(rt.project_image(page, delta), path); reload = rt.reload_png(path)
            proposal_score = score_image(rt, model, processor, item, state, reload, reference, pieces, positions, list(state["clean_fast_kept"]), cost)
            proposal_margin = sum(fixed_competitor(model, processor, item, state, reload, reference, positions, candidates)) / len(positions)
            full_exact = bool(proposal_score["full_text_exact"] and proposal_score["full_token_exact"] and proposal_score["full_replay"])
            improves = proposal_margin < current_margin - 1e-4
            rec = {"step": step, "radius": radius, "base_margin": current_margin, "candidate_margin": proposal_margin, "predicted_change": float(torch.stack(cs["grads"]).mean(dim=0).flatten() @ delta.flatten()), "accepted": bool(full_exact and improves), "full_exact": full_exact, "fast_text": proposal_score["fast_text"], "keep_hash": proposal_score["keep_hash"], "png_linf": rt.uint8_linf(current, reload), "candidate_margins": fixed_competitor(model, processor, item, state, reload, reference, positions, candidates), "semantic_text_change": proposal_score["fast_text"] != current_score["fast_text"]}
            if full_exact and improves:
                current, current_score, current_margin = reload, proposal_score, proposal_margin; accepted += 1; accepted_this = True
                rows.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True); break
            rows.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
        if any(r["accepted"] and r["semantic_text_change"] for r in rows[-2:]): break
    status = "CONTENT_COMPETITOR_TEXT_CHANGED_REVIEW" if any(r["accepted"] and r["semantic_text_change"] for r in rows) else "CONTENT_COMPETITOR_NO_TEXT_CHANGE"
    rt.write_json(out / "results.json", {"sample_id": args.sample, "candidates": candidates, "rows": rows, "accepted": accepted, "cost": cost, "final_fast_text": current_score["fast_text"]})
    rt.write_json(out / "final_decision.json", {"status": status, "sample_id": args.sample, "accepted": accepted, "steps": len(set(r["step"] for r in rows)), "unseen_started": False})
    print(json.dumps({"status": status, "accepted": accepted}, ensure_ascii=False))


if __name__ == "__main__": main()
