"""Independent PNG evaluator. A fresh process reads sealed files from disk."""

from __future__ import annotations

import sys

sys.path[:0] = [
    "/home/yuesiting/autodl-tmp/multimodal_attack_project",
    "/home/yuesiting/autodl-tmp/multimodal_attack_project/week1_mechanism",
    "/home/yuesiting/autodl-tmp/multimodal_attack_project/gate_experiment",
]

import argparse
import json
from pathlib import Path

import torch

from metrics_task import anls_score

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt
from week2_attack_v20_fourway_content_ablation.protocol import EPSILON_UINT8, EVAL_RATIOS, TRAIN_RATIO
from week2_attack_v20_fourway_content_ablation.semantics import aliases_for, classify

HERE = rt.HERE


def score_image(model, processor, item, state, path: Path, aliases: list[str], sample_id: str) -> dict:
    image = rt.reload_png(path)
    guard = rt.full_guard(model, processor, item, state, image)
    clean = item["images"][item["aux_idx"]]
    linf = rt.uint8_linf(clean, image)
    ratios = {}
    for ratio in EVAL_RATIOS:
        fast = rt.fastv_generate(model, processor, item, image, ratio)
        label = classify(sample_id, fast["text"], aliases)
        ratios[f"{ratio:.2f}"] = {
            "text": fast["text"],
            "tokens": fast["tokens"],
            "valid": fast["valid"],
            "anls": float(anls_score(fast["text"], list(item["meta"]["answers"]))),
            "semantic_status": label["semantic_status"],
            "semantic_reason": label["reason"],
            "k": fast["k"],
            "keep_hash": fast["keep_hash"],
            "keep_jaccard_vs_clean": rt.jaccard_kept(fast["kept"], state["clean_fast_kept"]),
            "role": "train" if ratio == TRAIN_RATIO else "transfer_diagnostic",
        }
    return {
        "png": str(path),
        "png_sha256": rt.sha_file(path),
        "uint8_linf": linf,
        "linf_ok": linf <= EPSILON_UINT8,
        "full_text": guard["full_text"],
        "full_tokens": guard["full_tokens"],
        "full_text_exact": guard["full_text_exact"],
        "full_token_exact": guard["full_token_exact"],
        "full_replay": guard["full_replay"],
        "output_valid": guard["output_valid"],
        "png_reloadable": True,
        "ratios": ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / "semantic_manifest.json").read_text())
    by_id = {row["sample_id"]: row for row in manifest["samples"]}
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    run = Path(args.run)
    out = Path(args.out)
    for sample_id in [item.strip() for item in args.samples.split(",") if item.strip()]:
        item = rt.v7d.load_dev(sample_id)
        state = rt.prepare_clean_state(model, processor, item)
        aliases = by_id[sample_id]["normalized_aliases"]
        if aliases != aliases_for(sample_id, by_id[sample_id]["gold_answers"]):
            raise RuntimeError("semantic manifest aliases drifted from the frozen rule table")
        folder = run / "per_run" / sample_id / args.method / "checkpoints"
        rows = []
        if folder.exists():
            for path in sorted(folder.glob("accepted_*.png")):
                scored = score_image(model, processor, item, state, path, aliases, sample_id)
                scored["checkpoint"] = int(path.stem.split("_")[1])
                rows.append(scored)
                print(json.dumps({"sample": sample_id, "method": args.method, "checkpoint": scored["checkpoint"], "r25": scored["ratios"]["0.25"]["semantic_status"]}), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        rt.write_json(out / sample_id / f"{args.method}.json", {"sample_id": sample_id, "method": args.method, "checkpoints": rows})


if __name__ == "__main__":
    main()
