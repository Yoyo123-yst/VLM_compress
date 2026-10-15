"""Two-sample response calibration after the no-model math gate."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v24_signed_feasible_step_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(ROOT / "week2_attack_v23_differential_subspace_pilot")]
from week2_attack_v23_differential_subspace_pilot import pilot as v23  # noqa: E402
from week2_attack_v23_differential_subspace_pilot.pilot import CORE, margin_tensor, score_image, sha_tensor, token_pieces  # noqa: E402
from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v24_signed_feasible_step_pilot.pilot import solve_signed, CONFIG  # noqa: E402


def fixed_core(model, processor, item, state, image, reference, pieces, core_positions):
    attacked = list(item["images"]); attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    pixels = inputs["pixel_values"]
    keep_index = rt.v7d.keep_index_from_kept(state["clean_fast_kept"], spans, int(inputs["input_ids"].shape[1]), pixels.device)
    with torch.no_grad():
        rows = rt.v7d.fastv_rows(model, inputs, pixels, reference, keep_index)
    return [{"position": int(p), "hard_margin": float(rt.v7d.hard_margin_row(rows[p], int(reference[p]))),
             "soft_margin": float(rt.v7d.soft_margin_row(rows[p], int(reference[p]))), "piece": pieces[p]}
            for p in core_positions]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rt.configure(); model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev(args.sample); state = rt.prepare_clean_state(model, processor, item)
    original = item["images"][item["aux_idx"]].convert("RGB"); reference = list(state["clean_fast_tokens"])
    positions = [int(x) for x in CORE[args.sample]]; pieces = token_pieces(processor, reference)
    with torch.no_grad():
        clean_full = rt.v7d.full_rows(model, state["inputs"], state["inputs"]["pixel_values"], state["clean_tokens"])
        state["clean_soft"] = [float(x) for x in margin_tensor(clean_full, state["clean_tokens"]).cpu()]
    gs = v23.differentiable_state(model, processor, item, state, reference, positions, original)
    rows = []
    clean_core = [{"hard_margin": float(x), "soft_margin": float(y)} for x, y in zip(gs["core_soft"].tolist(), gs["core_soft"].tolist())]
    for radius in [1, 2]:
        qp, qp_meta = solve_signed(gs["core_grads"], gs["full_grads"], gs["full_soft"].tolist(), radius, CONFIG["full_target_margin"], CONFIG["ridge"], CONFIG["coefficient_norm"])
        directions = {"RAW": (-float(radius) / 255.0) * gs["raw"].sign(), "QP": qp}
        for name, delta in directions.items():
            if delta is None or not qp_meta.get("candidate_valid", False) and name == "QP":
                continue
            predicted = float(torch.stack(gs["core_grads"]).mean(dim=0).flatten() @ delta.flatten())
            page = rt.v7d.page_to_float(original).to("cuda")
            image = rt.project_image(page, delta); path = out / f"{args.sample}_{name}_{radius}.png"; rt.save_png(image, path)
            reload = rt.reload_png(path)
            natural = score_image(rt, model, processor, item, state, reload, reference, pieces, positions, list(state["clean_fast_kept"]), {"forward":0,"backward":0,"generate":0})
            fixed = fixed_core(model, processor, item, state, reload, reference, pieces, positions)
            rows.append({"sample_id":args.sample,"direction":name,"radius":radius,"predicted_core_change":predicted,
                         "natural_core_margins":natural["core_margins"],"fixed_keep_core_margins":fixed,
                         "fast_text":natural["fast_text"],"keep_hash":natural["keep_hash"],"full_exact":bool(natural["full_text_exact"] and natural["full_token_exact"] and natural["full_replay"]),
                         "png_linf":rt.uint8_linf(original,reload),"qp_solver":qp_meta if name=="QP" else None})
            print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    rt.write_json(out / "results.json", {"sample_id":args.sample,"clean_fast_text":state["clean_fast_text"],"rows":rows})

if __name__ == "__main__": main()
