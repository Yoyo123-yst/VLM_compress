"""One-step differential-sensitivity pilot.

This is deliberately separate from all previous attack directories.  It
compares four equal-norm pixel directions at one compression ratio and never
selects a checkpoint.  The DIFF direction is estimated in the span of the
per-token FastV and Full soft-margin gradients; legality is decided only after
PNG reload and Full exact replay.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v23_differential_subspace_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v20_fourway_content_ablation.protocol import LAMBDA_FULL, SEED  # noqa: E402
from week2_attack_v20_fourway_content_ablation.selection import control_id_set  # noqa: E402
from week2_attack_v21b_full_safety_slack_scan.measure import score_image  # noqa: E402
from week2_attack_v22_safety_slack_constraint_pilot.pixels import straight_through_pixels  # noqa: E402
from week2_attack_v22_safety_slack_constraint_pilot.objectives import hard_margins  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import hard_margin_row, soft_margin_row  # noqa: E402

CONFIG = json.loads((HERE / "config.json").read_text())
MANIFEST = json.loads((ROOT / "week2_attack_v20_fourway_content_ablation" / "semantic_manifest.json").read_text())
CORE = {row["sample_id"]: row["core_token_positions"] for row in MANIFEST["samples"]}


def sha_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare(processor, item, image):
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(item.get("image_token_index", 0) or 0), len(attacked))
    return inputs, spans


def token_pieces(processor, ids):
    return [processor.tokenizer.decode([int(x)], skip_special_tokens=False) for x in ids]


def margin_tensor(rows, ids, soft=True):
    fn = soft_margin_row if soft else hard_margin_row
    return torch.stack([fn(row, int(tok)) for row, tok in zip(rows, ids)])


def differentiable_state(model, processor, item, state, reference, core_positions, image):
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = rt.v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    prepared = inputs["pixel_values"]
    page = rt.v7d.page_to_float(image).to(prepared.device).detach().requires_grad_(True)
    pixels = straight_through_pixels(prepared, item["aux_idx"], page)

    generated, kept = rt.v7d.generate_tokens(model, inputs, prepared, spans, 0.25)
    kept = [int(x) for x in kept]
    keep_index = rt.v7d.keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), prepared.device)
    fast_rows = rt.v7d.fastv_rows(model, inputs, pixels, reference, keep_index)
    core_soft = margin_tensor([fast_rows[i] for i in core_positions], [reference[i] for i in core_positions])
    core_grads = []
    for value in core_soft:
        grad = torch.autograd.grad(value, page, retain_graph=True)[0].detach().clone()
        core_grads.append(grad)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    try:
        full_rows = rt.v7d.full_rows(model, inputs, pixels, state["clean_tokens"])
    finally:
        model.eval()
        model.gradient_checkpointing_disable()
    full_soft = margin_tensor(full_rows, state["clean_tokens"])
    full_hard = margin_tensor(full_rows, state["clean_tokens"], soft=False)
    full_grads = []
    for value in full_soft:
        grad = torch.autograd.grad(value, page, retain_graph=True)[0].detach().clone()
        full_grads.append(grad)

    core_loss = core_soft.mean()
    clean_soft = torch.tensor(state["clean_soft"], device=page.device, dtype=full_soft.dtype)
    native_loss = core_loss + float(LAMBDA_FULL) * torch.nn.functional.softplus(clean_soft - full_hard).mean()
    native_grad = torch.autograd.grad(native_loss, page)[0].detach().clone()
    return {
        "page": page.detach(),
        "prepared_hash": sha_tensor(prepared),
        "reference": reference,
        "kept": kept,
        "core_soft": core_soft.detach(),
        "full_soft": full_soft.detach(),
        "full_hard": full_hard.detach(),
        "core_grads": core_grads,
        "full_grads": full_grads,
        "raw": torch.stack(core_grads).mean(dim=0),
        "native": native_grad,
        "generated": [int(x) for x in generated],
    }


def diff_direction(core_grads, full_grads, mu: float):
    # Work in the small span of observed per-token gradients, never construct
    # a dense pixel-space Jacobian or eigensystem.
    cg = torch.stack([x.flatten() for x in core_grads])
    fg = torch.stack([x.flatten() for x in full_grads])
    basis = torch.cat([cg, fg], dim=0).T
    basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-12)
    cproj = cg @ basis
    fproj = fg @ basis
    a = cproj.T @ cproj
    b = fproj.T @ fproj + float(mu) * torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
    solved = torch.linalg.solve(b, a)
    eigenvalues, eigenvectors = torch.linalg.eig(solved)
    index = int(torch.argmax(eigenvalues.real).item())
    coeff = eigenvectors[:, index].real
    vector = (basis @ coeff).reshape_as(core_grads[0])
    core_mean = torch.stack(core_grads).mean(dim=0)
    # The actual pixel update uses sign(vector), so orient against the
    # quantized update rather than against the continuous vector itself.
    if torch.sum(vector.sign() * core_mean) < 0:
        vector = -vector
    return vector.detach(), {
        "eigenvalue": float(eigenvalues[index].real.detach()),
        "basis_rank": int(basis.shape[1]),
        "core_response": float((core_mean.flatten() @ vector.sign().flatten()).detach()),
    }


def save_and_score(model, processor, item, state, reference, pieces, core_positions, clean_kept, original, direction, radius, out):
    page = rt.v7d.page_to_float(original).to("cuda")
    delta = -(float(radius) / 255.0) * direction.sign()
    candidate = rt.project_image(page, delta)
    path = out / f"{radius:02d}.png"
    png_sha = rt.save_png(candidate, path)
    reloaded = rt.reload_png(path)
    cost = {"forward": 0, "backward": 0, "generate": 0}
    scored = score_image(rt, model, processor, item, state, reloaded, reference, pieces, core_positions, clean_kept, cost)
    scored.update({
        "radius": int(radius),
        "png_sha256": png_sha,
        "uint8_linf": rt.uint8_linf(original, reloaded),
        "direction_hash": sha_tensor(direction),
        "full_exact": bool(scored["full_text_exact"] and scored["full_token_exact"] and scored["full_replay"]),
        "cost": cost,
    })
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=",".join(CONFIG["samples"]))
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    sample_list = [x.strip() for x in args.samples.split(",") if x.strip()]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rt.configure()
    torch.manual_seed(int(CONFIG["seed"]))
    model, processor = rt.v7d.load_model_frozen()
    all_rows = []
    for sample_id in sample_list:
        item = rt.v7d.load_dev(sample_id)
        state = rt.prepare_clean_state(model, processor, item)
        original = item["images"][item["aux_idx"]].convert("RGB")
        reference = list(state["clean_fast_tokens"])
        positions = [int(x) for x in CORE.get(sample_id, [])]
        if not positions:
            raise RuntimeError(f"no registered core span for {sample_id}")
        pieces = token_pieces(processor, reference)
        with torch.no_grad():
            clean_full = rt.v7d.full_rows(model, state["inputs"], state["inputs"]["pixel_values"], state["clean_tokens"])
            state["clean_soft"] = [float(x) for x in margin_tensor(clean_full, state["clean_tokens"]).detach().cpu()]
        clean_kept = list(state["clean_fast_kept"])
        base = score_image(rt, model, processor, item, state, original, reference, pieces, positions, clean_kept, {"forward": 0, "backward": 0, "generate": 0})
        grad_state = differentiable_state(model, processor, item, state, reference, positions, original)
        diff, diff_meta = diff_direction(grad_state["core_grads"], grad_state["full_grads"], float(CONFIG["mu"]))
        rng = torch.Generator(device="cuda").manual_seed(int(CONFIG["seed"]) + int(sample_id))
        random = torch.randn(grad_state["raw"].shape, generator=rng, device=grad_state["raw"].device, dtype=grad_state["raw"].dtype)
        directions = {
            "RAW": grad_state["raw"],
            "NATIVE": grad_state["native"],
            "DIFF": diff,
            "RANDOM": random,
        }
        sample_out = out_root / "results" / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        rt.write_json(sample_out / "preflight.json", {
            "sample_id": sample_id,
            "reference_tokens": reference,
            "reference_pieces": pieces,
            "core_positions": positions,
            "base_fast_text": base["fast_text"],
            "base_full_text": base["full_text"],
            "base_keep_hash": base["keep_hash"],
            "base_processor_hash": base["processor_exact_hash"],
            "base_full_exact": bool(base["full_text_exact"] and base["full_token_exact"] and base["full_replay"]),
            "gradient_prepared_hash": grad_state["prepared_hash"],
            "diff_meta": diff_meta,
            "direction_hashes": {name: sha_tensor(value) for name, value in directions.items()},
        })
        for name, direction in directions.items():
            for radius in CONFIG["radii_uint8"]:
                row = save_and_score(model, processor, item, state, reference, pieces, positions, clean_kept, original, direction, radius, sample_out / name)
                row.update({"sample_id": sample_id, "direction": name})
                all_rows.append(row)
                print(json.dumps({"sample": sample_id, "direction": name, "radius": radius, "fast_text": row["fast_text"], "full_exact": row["full_exact"], "linf": row["uint8_linf"]}, ensure_ascii=False), flush=True)
        del state
        torch.cuda.empty_cache()
    rt.write_json(out_root / "results.json", {"config": CONFIG, "rows": all_rows})
    print(json.dumps({"status": "COMPLETED", "rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
