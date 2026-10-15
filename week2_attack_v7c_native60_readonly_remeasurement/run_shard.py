"""Readonly native-cache remeasure for one frozen shard. Does not write pixels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7c_native60_readonly_remeasurement"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks, install  # noqa: E402
from run_causal_v2 import decode, prepare, spans_of  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v7_pristine60.load_samples import load_items, manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    cache_replay,
    float_list,
    margins_for_tokens,
    production_generate,
    require_keep_match,
    termination_of,
)

BUDGETS = (0.50, 0.35, 0.25, 0.15)
STOP = HERE / "STOP_ALIGNMENT.json"


def load_protocol() -> dict:
    return json.loads((HERE / "FROZEN_PROTOCOL.json").read_text())


def shard_ids(protocol: dict, shard: int) -> list[str]:
    ids = [row["sample_id"] for row in protocol["samples"]]
    if ids != sorted(ids, key=int):
        raise RuntimeError("protocol sample order is not the frozen integer sort")
    if shard not in range(4):
        raise RuntimeError(f"shard must be 0..3, got {shard}")
    return ids[shard::4]


def summarize_path(tokens: list[int], scores: list[torch.Tensor], rows: list[torch.Tensor], eos: int) -> dict:
    if len(scores) != len(tokens) or len(rows) != len(tokens):
        return {"match": False, "mismatch": {"index": 0, "reason": "length", "tokens": len(tokens), "scores": len(scores), "rows": len(rows)}}
    mismatch = None
    gaps = []
    for index, token in enumerate(tokens):
        g_argmax = int(scores[index].argmax())
        t_argmax = int(rows[index].argmax())
        gap = float((scores[index].float().cpu() - rows[index].float().cpu()).abs().max())
        gaps.append(gap)
        if mismatch is None and not (g_argmax == token == t_argmax):
            mismatch = {
                "index": index,
                "generated_token": token,
                "g_argmax": g_argmax,
                "t_argmax": t_argmax,
                "max_abs_logit_diff": gap,
            }
    soft = float_list(margins_for_tokens(rows, tokens)["soft"])
    ordered = sorted(soft)
    mid = ordered[len(ordered) // 2] if len(ordered) % 2 else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
    return {
        "match": mismatch is None,
        "mismatch": mismatch,
        "token_ids": tokens,
        "termination": termination_of(tokens, eos),
        "token_count": len(tokens),
        "soft_margin": soft,
        "mean_soft": sum(soft) / len(soft),
        "median_soft": mid,
        "min_soft": min(soft),
        "max_abs_logit_diff": max(gaps) if gaps else None,
        "step_argmax_match": mismatch is None,
    }


def fail(shard_dir: Path, diagnosis: dict) -> None:
    (shard_dir / "MISMATCH.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    STOP.write_text(json.dumps(diagnosis, indent=2) + "\n")
    raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    mine = shard_ids(protocol, args.shard)
    by_id = {row["sample_id"]: row for row in protocol["samples"]}
    shard_dir = HERE / "shards" / f"shard_{args.shard}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_dir / "conditions.jsonl"
    done = set()
    if out_path.exists():
        existing = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        counts: dict[str, int] = {}
        for row in existing:
            counts[row["sample_id"]] = counts.get(row["sample_id"], 0) + 1
        done = {sid for sid, count in counts.items() if count == 10}
        kept = [row for row in existing if row["sample_id"] in done]
        if len(kept) != len(existing):
            out_path.write_text("".join(json.dumps(row) + "\n" for row in kept))
    specs = {row["sample_id"]: row for row in manifest()["samples"]}
    needed = [sid for sid in mine if sid not in done]
    if not needed:
        print(json.dumps({"shard": args.shard, "status": "already_complete", "n": len(mine)}), flush=True)
        return
    items = {item["meta"]["sample_id"]: item for item in load_items([specs[sid] for sid in needed])}
    model, processor = load_model_frozen()
    eos = int(model.generation_config.eos_token_id)
    image_token_id = int(model.config.image_token_index)
    with out_path.open("a") as handle:
        for sid in needed:
            if STOP.exists():
                print(json.dumps({"shard": args.shard, "stopped": True}), flush=True)
                raise SystemExit(3)
            spec = by_id[sid]
            item = items[sid]
            for kind, path, digest in (
                ("B", ROOT / spec["b_png"], spec["b_sha256"]),
                ("random_noise", ROOT / spec["noise_png"], spec["noise_sha256"]),
            ):
                images = list(item["images"])
                images[item["aux_idx"]] = Image.open(path).convert("RGB")
                clear_hooks(model)
                inputs = prepare(processor, images, item["meta"]["question"])
                generated = production_generate(model, inputs)
                replay, _meta = cache_replay(model, inputs, generated["tokens"])
                full = summarize_path(generated["tokens"], generated["scores"], replay, eos)
                full_text = decode(processor, torch.tensor(generated["tokens"]))
                t_text = decode(processor, torch.tensor([int(row.argmax()) for row in replay]))
                if full_text != t_text:
                    full["match"] = False
                    full["mismatch"] = full["mismatch"] or {"reason": "decoded text differs", "g_text": full_text, "t_text": t_text}
                base = {
                    "sample_id": sid,
                    "M": spec["M"],
                    "image_kind": kind,
                    "png": str(path.relative_to(ROOT)),
                    "sha256": digest,
                    "full_text": full_text,
                    "full_token_ids": full.get("token_ids"),
                    "full_termination": full.get("termination"),
                    "full_token_count": full.get("token_count"),
                    "full_soft_margin": full.get("soft_margin"),
                    "full_mean_soft": full.get("mean_soft"),
                    "full_median_soft": full.get("median_soft"),
                    "full_min_soft": full.get("min_soft"),
                    "full_gt_match": full["match"],
                    "appended_eos": False,
                }
                if not full["match"]:
                    fail(shard_dir, {**base, "path": "full", "r": None, "mismatch": full["mismatch"]})
                full_row = {
                    **base,
                    "r": None,
                    "fastv_text": None,
                    "fastv_token_ids": None,
                    "fastv_termination": None,
                    "fastv_token_count": None,
                    "fastv_soft_margin": None,
                    "fastv_mean_soft": None,
                    "fastv_median_soft": None,
                    "fastv_min_soft": None,
                    "fastv_gt_match": None,
                    "fastv_k": None,
                    "fastv_keep_hash": None,
                    "mismatch": None,
                }
                handle.write(json.dumps(full_row) + "\n")
                handle.flush()
                for ratio in BUDGETS:
                    if STOP.exists():
                        raise SystemExit(3)
                    clear_hooks(model)
                    inputs = prepare(processor, images, item["meta"]["question"])
                    spans = spans_of(inputs, image_token_id, len(images))
                    state = install(model, "fastv", ratio, image_token_id)
                    state.spans = spans
                    generated = production_generate(model, inputs)
                    ref_kept = list(state.kept_visual)
                    clear_hooks(model)
                    state = install(model, "fastv", ratio, image_token_id)
                    state.spans = spans
                    replay, _meta = cache_replay(model, inputs, generated["tokens"])
                    got_kept = list(state.kept_visual)
                    clear_hooks(model)
                    fast = summarize_path(generated["tokens"], generated["scores"], replay, eos)
                    fast_text = decode(processor, torch.tensor(generated["tokens"]))
                    fast_t = decode(processor, torch.tensor([int(row.argmax()) for row in replay]))
                    keep_error = ""
                    try:
                        require_keep_match(len(got_kept), keep_hash(got_kept), len(ref_kept), keep_hash(ref_kept))
                    except RuntimeError as exc:
                        keep_error = str(exc)
                    if fast_text != fast_t or keep_error or not fast["match"]:
                        fail(shard_dir, {
                            **base,
                            "path": "fastv",
                            "r": ratio,
                            "fastv_text": fast_text,
                            "mismatch": fast["mismatch"],
                            "text_mismatch": fast_text != fast_t,
                            "keep_error": keep_error,
                            "k_generate": len(ref_kept),
                            "k_replay": len(got_kept),
                            "keep_hash_generate": keep_hash(ref_kept) if ref_kept else None,
                        })
                    row = {
                        **base,
                        "r": ratio,
                        "fastv_text": fast_text,
                        "fastv_token_ids": fast["token_ids"],
                        "fastv_termination": fast["termination"],
                        "fastv_token_count": fast["token_count"],
                        "fastv_soft_margin": fast["soft_margin"],
                        "fastv_mean_soft": fast["mean_soft"],
                        "fastv_median_soft": fast["median_soft"],
                        "fastv_min_soft": fast["min_soft"],
                        "fastv_gt_match": True,
                        "fastv_k": len(ref_kept),
                        "fastv_keep_hash": keep_hash(ref_kept),
                        "mismatch": None,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
            print(json.dumps({"shard": args.shard, "sample_id": sid, "done": True}), flush=True)
    print(json.dumps({"shard": args.shard, "status": "complete", "n": len(mine)}), flush=True)


if __name__ == "__main__":
    main()
