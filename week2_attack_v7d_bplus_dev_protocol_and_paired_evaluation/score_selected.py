"""Disk-reload replay and canonical strict for one selected development sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from adapters.runtime import clear_hooks, install  # noqa: E402
from metrics_task import anls_score, normalize  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import uint8_linf  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    cache_replay,
    soft_margin_row,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.run_v7d import (  # noqa: E402
    EPS,
    generate_tokens,
    load_dev,
    prepare,
    spans_of,
)

EVAL_RATIOS = (0.50, 0.35, 0.25, 0.15)
TRAIN_RATIOS = (0.25, 0.35)


def pollution(attack: str, clean: str) -> int:
    left, right = normalize(attack), normalize(clean)
    return int(left != right and (left in right or right in left))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-dir", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    item = load_dev(args.sample)
    run = Path(args.method_dir) / args.sample
    selected = json.loads((run / "selected.json").read_text())
    curve = json.loads((run / "curve.json").read_text())
    chosen = selected["chosen"]
    clean_text = selected["clean_text"]
    answers = list(item["meta"]["answers"])
    report = {
        "sample_id": args.sample,
        "chosen_step": None if chosen is None else int(chosen["step"]),
        "selector_legal": True,
    }
    if chosen is None:
        report.update({
            "full_exact": 0,
            "replay": 0,
            "nonzero": 0,
            "linf_ok": 0,
            "invalid": 1,
            "strict_train": 0,
            "pollution": 0,
            "by_ratio": {},
            "reason": "no_eligible_checkpoint",
        })
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"sample": args.sample, "strict_train": 0, "reason": "no_eligible_checkpoint"}), flush=True)
        return
    step = int(chosen["step"])
    png = run / "checkpoints" / f"step_{step:03d}.png"
    stored = next(row for row in curve if int(row["step"]) == step)
    digest = hashlib.sha256(png.read_bytes()).hexdigest()
    report["hash_match"] = int(digest == stored["png_sha256"])
    image = Image.open(png).convert("RGB")
    clean = item["images"][item["aux_idx"]]
    linf = uint8_linf(clean, image)
    nonzero = int(not np.array_equal(np.asarray(image), np.asarray(clean)))
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    model, processor = load_model_frozen()
    model.eval()
    inputs = prepare(processor, attacked, item["meta"]["question"])
    spans = spans_of(inputs, int(model.config.image_token_index), len(attacked))
    pixels = inputs["pixel_values"]

    def once(ratio):
        tokens, kept = generate_tokens(model, inputs, pixels, spans, ratio)
        from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.run_v7d import decode
        text = decode(processor, torch.tensor(tokens)).strip()
        local = dict(inputs)
        clear_hooks(model)
        keep = None
        if ratio is not None:
            state = install(model, "fastv", ratio, int(model.config.image_token_index))
            state.spans = spans
        rows, _ = cache_replay(model, local, tokens)
        if ratio is not None:
            hook_kept = list(state.kept_visual)
            clear_hooks(model)
            keep = {"k": len(kept), "keep_hash": keep_hash(kept), "hook_match": int(list(kept) == hook_kept)}
        margin = float(torch.stack([soft_margin_row(row, token) for row, token in zip(rows, tokens)]).mean())
        return {"tokens": tokens, "text": text, "margin": margin, "keep": keep}

    first_full = once(None)
    second_full = once(None)
    by_ratio = {}
    replay_ok = first_full["tokens"] == second_full["tokens"] and first_full["text"] == stored["texts"]["full"]
    for ratio in EVAL_RATIOS:
        a = once(ratio)
        b = once(ratio)
        key = f"{ratio:.2f}"
        stored_text = stored["texts"].get(f"fastv_{ratio}")
        text_match = True if stored_text is None else a["text"] == stored_text
        same = a["tokens"] == b["tokens"] and text_match
        replay_ok = replay_ok and same
        anls = anls_score(a["text"], answers)
        by_ratio[key] = {
            "text": a["text"],
            "anls": anls,
            "margin": a["margin"],
            "keep": a["keep"],
            "strict_at_ratio": 0,
        }
    full_exact = int(first_full["text"] == clean_text)
    valid = int(output_valid(first_full["text"]) and all(output_valid(row["text"]) for row in by_ratio.values()))
    linf_ok = int(nonzero and 0 < linf <= EPS + 1e-6)
    m_full = first_full["margin"]
    for key, row in by_ratio.items():
        row["delta"] = m_full - row["margin"]
        row["strict_at_ratio"] = int(
            valid and nonzero and linf_ok and full_exact and replay_ok and report["hash_match"] and row["anls"] < 0.5
        )
    train_anls_fail = all(by_ratio[f"{ratio:.2f}"]["anls"] < 0.5 for ratio in TRAIN_RATIOS)
    strict_train = int(valid and nonzero and linf_ok and full_exact and replay_ok and report["hash_match"] and train_anls_fail)
    report.update({
        "full_text": first_full["text"],
        "clean_text": clean_text,
        "full_exact": full_exact,
        "replay": int(replay_ok and report["hash_match"]),
        "nonzero": nonzero,
        "linf": linf,
        "linf_ok": linf_ok,
        "invalid": int(not valid),
        "m_full": m_full,
        "pollution": pollution(first_full["text"], clean_text),
        "strict_train": strict_train,
        "by_ratio": by_ratio,
        "selector_margin": chosen["margin"],
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"sample": args.sample, "strict_train": strict_train, "full_exact": full_exact, "replay": report["replay"]}), flush=True)


if __name__ == "__main__":
    main()
