"""Repeated real-PNG finite-difference gate for the repaired gradient diagnostic."""
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


def fixed_core(model, processor, item, state, image, reference, positions):
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    keep = rt.v7d.keep_index_from_kept(state["clean_fast_kept"], spans, int(inputs["input_ids"].shape[1]), inputs["pixel_values"].device)
    with torch.no_grad():
        rows = rt.v7d.fastv_rows(model, inputs, inputs["pixel_values"], reference, keep)
    return [float(rt.v7d.soft_margin_row(rows[p], int(reference[p]))) for p in positions]


def one_png(model, processor, item, state, original, reference, positions, delta, path):
    page = rt.v7d.page_to_float(original).to("cuda")
    image = rt.project_image(page, delta)
    sha = rt.save_png(image, path)
    reloaded = rt.reload_png(path)
    pieces = token_pieces(processor, reference)
    cost = {"forward": 0, "backward": 0, "generate": 0}
    natural = score_image(rt, model, processor, item, state, reloaded, reference, pieces, positions, list(state["clean_fast_kept"]), cost)
    fixed = fixed_core(model, processor, item, state, reloaded, reference, positions)
    return {
        "png_sha256": sha,
        "png_linf": rt.uint8_linf(original, reloaded),
        "natural_core_soft": [float(x["soft_margin"]) for x in natural["core_margins"]],
        "natural_core_mean": float(sum(float(x["soft_margin"]) for x in natural["core_margins"]) / len(positions)),
        "fixed_core_soft": fixed,
        "fixed_core_mean": float(sum(fixed) / len(fixed)),
        "fast_text": natural["fast_text"],
        "keep_hash": natural["keep_hash"],
        "full_exact": bool(natural["full_text_exact"] and natural["full_token_exact"] and natural["full_replay"]),
        "full_text_exact": bool(natural["full_text_exact"]),
        "full_token_exact": bool(natural["full_token_exact"]),
        "full_replay": bool(natural["full_replay"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev(args.sample)
    state = rt.prepare_clean_state(model, processor, item)
    original = item["images"][item["aux_idx"]].convert("RGB")
    reference = list(state["clean_fast_tokens"])
    positions = [int(x) for x in CORE[args.sample]]
    with torch.no_grad():
        clean_full = rt.v7d.full_rows(model, state["inputs"], state["inputs"]["pixel_values"], state["clean_tokens"])
        state["clean_soft"] = [float(x) for x in margin_tensor(clean_full, state["clean_tokens"]).cpu()]
    gs = v23.differentiable_state(model, processor, item, state, reference, positions, original)
    descent = -gs["raw"].sign()
    clean_soft = [float(x) for x in gs["core_soft"].tolist()]
    clean_mean = float(sum(clean_soft) / len(clean_soft))
    rows = []
    for radius in [1, 2]:
        for sign_name, sign in [("plus_descent", 1.0), ("minus_descent", -1.0)]:
            delta = sign * (float(radius) / 255.0) * descent
            predicted = float(torch.stack(gs["core_grads"]).mean(dim=0).flatten() @ delta.flatten())
            repeats = []
            for rep in [1, 2]:
                path = out / f"{args.sample}_{sign_name}_r{radius}_rep{rep}.png"
                result = one_png(model, processor, item, state, original, reference, positions, delta, path)
                repeats.append(result)
            row = {
                "sample_id": args.sample,
                "radius": radius,
                "direction": sign_name,
                "direction_hash": sha_tensor(delta),
                "predicted_change": predicted,
                "clean_core_mean": clean_mean,
                "repeats": repeats,
                "repeat_png_equal": repeats[0]["png_sha256"] == repeats[1]["png_sha256"],
                "repeat_keep_equal": repeats[0]["keep_hash"] == repeats[1]["keep_hash"],
                "repeat_text_equal": repeats[0]["fast_text"] == repeats[1]["fast_text"],
                "repeat_core_equal": repeats[0]["natural_core_soft"] == repeats[1]["natural_core_soft"],
            }
            row["natural_change"] = repeats[0]["natural_core_mean"] - clean_mean
            row["fixed_change"] = repeats[0]["fixed_core_mean"] - clean_mean
            row["full_exact"] = bool(all(x["full_exact"] for x in repeats))
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    status = "PNG_FINITE_DIFFERENCE_PASS" if all(
        row["full_exact"] and row["repeat_png_equal"] and row["repeat_keep_equal"] and row["repeat_text_equal"] and row["repeat_core_equal"]
        for row in rows
    ) else "PNG_FINITE_DIFFERENCE_NONREPEATABLE"
    rt.write_json(out / "results.json", {"sample_id": args.sample, "clean_core_soft": clean_soft, "rows": rows})
    rt.write_json(out / "final_decision.json", {"status": status, "sample_id": args.sample, "rows": len(rows), "multistep_started": False, "unseen_started": False})
    print(json.dumps({"status": status, "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
