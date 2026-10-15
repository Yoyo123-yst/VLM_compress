"""Check the differentiable tile transform against real processor output.

No model forward/backward is used. The model is loaded only because the
existing loader returns the project-pinned processor together with it.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_pilot.attack import normalized_tiles, page_to_float  # noqa: E402


def cos(a, b):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--samples", default="47450,60700"); ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rt.configure(); model, processor = rt.v7d.load_model_frozen(); del model
    rows = []
    for sample_id in [x.strip() for x in args.samples.split(",") if x.strip()]:
        item = rt.v7d.load_dev(sample_id); original = item["images"][item["aux_idx"]].convert("RGB")
        attacked = list(item["images"]); inputs = rt.v7d.prepare(processor, attacked, item["meta"]["question"])
        prepared = inputs["pixel_values"]
        page = page_to_float(original)
        surrogate = normalized_tiles(page)
        aux = prepared[item["aux_idx"]].detach().cpu()
        base = {"sample_id": sample_id, "prepared_shape": list(aux.shape), "surrogate_shape": list(surrogate.shape),
                "base_max_abs": float((aux - surrogate).abs().max()), "base_mean_abs": float((aux - surrogate).abs().mean()),
                "base_cos": cos(aux, surrogate)}
        for seed in range(3):
            gen = torch.Generator().manual_seed(20261014 + seed)
            direction = torch.where(torch.randn(page.shape, generator=gen) >= 0, 1.0, -1.0)
            base_u8 = page.mul(255).round().to(torch.int16)
            plus_u8 = (base_u8 + direction.to(torch.int16)).clamp(0, 255).to(torch.uint8)
            plus_page = plus_u8.float() / 255.0
            plus_sur = normalized_tiles(plus_page)
            plus_img = Image.fromarray(plus_u8.permute(1,2,0).cpu().numpy(), mode="RGB")
            plus_attacked = list(item["images"]); plus_attacked[item["aux_idx"]] = plus_img
            plus_inputs = rt.v7d.prepare(processor, plus_attacked, item["meta"]["question"])
            plus_real = plus_inputs["pixel_values"][item["aux_idx"]].detach().cpu()
            real_delta = plus_real - aux; sur_delta = plus_sur - surrogate
            base.update({f"seed{seed}_delta_cos": cos(real_delta, sur_delta),
                         f"seed{seed}_delta_norm_ratio": float(real_delta.norm() / sur_delta.norm().clamp_min(1e-30)),
                         f"seed{seed}_delta_max_abs": float(real_delta.abs().max()),
                         f"seed{seed}_sur_delta_max_abs": float(sur_delta.abs().max())})
        rows.append(base); print(json.dumps(base, ensure_ascii=False), flush=True)
    rt.write_json(out / "results.json", {"rows": rows, "used_model_forward": False, "used_model_backward": False})


if __name__ == "__main__": main()
