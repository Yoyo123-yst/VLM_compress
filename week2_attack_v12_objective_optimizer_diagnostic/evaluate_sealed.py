"""Independent evaluator for a sealed PNG at all registered FastV ratios."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("VATTACK_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("VATTACK_DETERMINISTIC_RESIZE", "1")
os.environ.setdefault("VATTACK_RESIZE_MODE", "bilinear")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v12_objective_optimizer_diagnostic"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(HERE)]

import numpy as np
import torch
from PIL import Image

import week2_attack_v11_objective_optimizer_diagnostic.run_v11 as base
from metrics_task import anls_score
from run_eval import decode
from week2_attack_v4_answer_aligned.common import keep_hash
from week2_attack_v4_answer_aligned.loss import output_valid
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d

base.HERE = HERE
EVAL_RATIOS = (0.15, 0.25, 0.35, 0.50)


def configure() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True)
    p.add_argument("--group", required=True, choices=sorted(base.GROUPS))
    p.add_argument("--tag", required=True)
    p.add_argument("--step", type=int, default=None)
    args = p.parse_args()
    configure()
    traj_dir = HERE / "runs" / args.tag / args.group / args.sample
    traj = json.loads((traj_dir / "trajectory.json").read_text())
    selected = traj.get("sealed_by_fixed_margin")
    step = int(args.step if args.step is not None else selected["step"])
    png = traj_dir / f"step_{step:03d}.png"
    if not png.exists():
        raise FileNotFoundError(png)
    item = v7d.load_dev(args.sample)
    model, processor = base.load_model_frozen()
    attacked = list(item["images"])
    clean = item["images"][item["aux_idx"]]
    attacked[item["aux_idx"]] = Image.open(png).convert("RGB")
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))

    full_a, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    full_b, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
    full_text_a = decode(processor, torch.tensor(full_a)).strip()
    full_text_b = decode(processor, torch.tensor(full_b)).strip()
    answers = list(item["meta"]["answers"])
    ratios = {}
    for ratio in EVAL_RATIOS:
        tokens, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
        text = decode(processor, torch.tensor(tokens)).strip()
        ratios[f"{ratio:.2f}"] = {
            "text": text,
            "tokens": tokens,
            "anls": anls_score(text, answers),
            "valid": bool(output_valid(text)),
            "k": len(kept),
            "keep_hash": keep_hash(kept),
            "endpoint_failure": bool(output_valid(text) and anls_score(text, answers) < base.ANLS_CLEAN),
        }
    arr_a = np.asarray(clean, dtype=np.int16)
    arr_b = np.asarray(attacked[item["aux_idx"]], dtype=np.int16)
    linf_u8 = int(np.max(np.abs(arr_a - arr_b)))
    result = {
        "status": "DONE",
        "sample_id": args.sample,
        "group": args.group,
        "tag": args.tag,
        "step": step,
        "png": str(png),
        "png_sha256": sha_file(png),
        "full_text": full_text_a,
        "full_text_replay": full_text_b,
        "full_text_exact": int(full_text_a == traj["clean_text"] and full_text_b == traj["clean_text"]),
        "full_token_exact": int(full_a == traj["clean_tokens"] and full_b == traj["clean_tokens"]),
        "full_replay": int(full_a == full_b and full_text_a == full_text_b),
        "linf_uint8": linf_u8,
        "ratios": ratios,
    }
    (traj_dir / "evaluator.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
