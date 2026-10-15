"""Small real-PNG trust-region pilot after the directional gate.

This is a diagnostic, not a training run: at most ten accepted updates per
sample, every proposal is PNG-reloaded and must pass the Full exact guard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v24_signed_feasible_step_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(ROOT / "week2_attack_v23_differential_subspace_pilot")]

from week2_attack_v23_differential_subspace_pilot import pilot as v23  # noqa: E402
from week2_attack_v23_differential_subspace_pilot.pilot import CORE, margin_tensor, score_image, sha_tensor, token_pieces  # noqa: E402
from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402


def set_clean_soft(model, state):
    with torch.no_grad():
        rows = rt.v7d.full_rows(model, state["inputs"], state["inputs"]["pixel_values"], state["clean_tokens"])
        state["clean_soft"] = [float(x) for x in margin_tensor(rows, state["clean_tokens"]).cpu()]


def core_mean(scored):
    return float(sum(float(x["soft_margin"]) for x in scored["core_margins"]) / len(scored["core_margins"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev(args.sample)
    state = rt.prepare_clean_state(model, processor, item)
    set_clean_soft(model, state)
    current = item["images"][item["aux_idx"]].convert("RGB")
    reference = list(state["clean_fast_tokens"])
    pieces = token_pieces(processor, reference)
    positions = [int(x) for x in CORE[args.sample]]
    clean_kept = list(state["clean_fast_kept"])
    cost = {"forward": 0, "backward": 0, "generate": 0}
    current_score = score_image(rt, model, processor, item, state, current, reference, pieces, positions, clean_kept, cost)
    rows = []
    accepted = 0
    for step in range(1, int(args.steps) + 1):
        gs = v23.differentiable_state(model, processor, item, state, reference, positions, current)
        descent = -gs["raw"].sign()
        base_mean = core_mean(current_score)
        accepted_this = False
        proposals = []
        for radius in [1, 2]:
            page = rt.v7d.page_to_float(current).to("cuda")
            delta = (float(radius) / 255.0) * descent
            candidate = rt.project_image(page, delta)
            path = out / f"step_{step:02d}_r{radius}.png"
            rt.save_png(candidate, path)
            reload = rt.reload_png(path)
            proposal = score_image(rt, model, processor, item, state, reload, reference, pieces, positions, clean_kept, cost)
            proposal_mean = core_mean(proposal)
            legal = bool(proposal["full_text_exact"] and proposal["full_token_exact"] and proposal["full_replay"])
            improves = bool(proposal_mean < base_mean - 1e-4)
            rec = {"step": step, "radius": radius, "base_core_mean": base_mean, "candidate_core_mean": proposal_mean,
                   "predicted_change": float(torch.stack(gs["core_grads"]).mean(dim=0).flatten() @ delta.flatten()),
                   "accepted": bool(legal and improves), "full_exact": legal, "fast_text": proposal["fast_text"],
                   "keep_hash": proposal["keep_hash"], "png_linf": rt.uint8_linf(current, reload),
                   "core_margins": proposal["core_margins"], "direction_hash": sha_tensor(delta),
                   "semantic_candidate_changed": bool(proposal["fast_text"] != current_score["fast_text"])}
            proposals.append(rec)
            if legal and improves:
                current = reload
                current_score = proposal
                accepted += 1
                accepted_this = True
                break
        rows.append({"step": step, "accepted": accepted_this, "proposals": proposals, "accepted_count": accepted})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        if any(p.get("semantic_candidate_changed") for p in proposals if p.get("accepted")):
            break
    status = "TRUST_REGION_TEXT_CHANGED_REQUIRES_SEMANTIC_REVIEW" if any(
        p.get("accepted") and p.get("semantic_candidate_changed") for row in rows for p in row["proposals"]
    ) else "TRUST_REGION_NO_CONTENT_FAILURE"
    rt.write_json(out / "results.json", {"sample_id": args.sample, "rows": rows, "accepted": accepted, "cost": cost, "final_fast_text": current_score["fast_text"]})
    rt.write_json(out / "final_decision.json", {"status": status, "sample_id": args.sample, "accepted": accepted, "steps": len(rows), "unseen_started": False})
    print(json.dumps({"status": status, "accepted": accepted, "steps": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
