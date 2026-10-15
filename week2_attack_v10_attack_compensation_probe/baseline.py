"""Replay clean pages and the historical Native-FDP PNGs. Stop on any mismatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V7D = ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"
HERE = ROOT / "week2_attack_v10_attack_compensation_probe"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(HERE)]

from adapters.runtime import clear_hooks, install  # noqa: E402
from run_eval import build_prompt, decode  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import cache_replay  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from run_probe import (  # noqa: E402
    FROZEN_1025,
    IDS,
    Access,
    historical_curve,
    install_counters,
    sha_file,
    sha_tensor,
)

V9C = json.loads((ROOT / "week2_attack_v9c_multiscale_rank_oblivious" / "baseline_recovery.json").read_text())
V9C_ROWS = {row["sample_id"]: row for row in V9C["generate_rows"]}


def fastv_once(model, processor, inputs, spans, ratio):
    tokens, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
    text = decode(processor, torch.tensor(tokens)).strip()
    clear_hooks(model)
    state = install(model, "fastv", ratio, int(model.config.image_token_index))
    state.spans = spans
    _ = cache_replay(model, dict(inputs), tokens)
    hook = list(state.kept_visual)
    clear_hooks(model)
    return {
        "text": text,
        "tokens": tokens,
        "k": len(kept),
        "keep_hash": keep_hash(kept),
        "hook_match": int(list(kept) == hook),
    }


def main() -> None:
    model, processor = load_model_frozen()
    install_counters(model)
    mismatches = []
    rows = []
    for sample_id in IDS:
        item = v7d.load_dev(sample_id)
        hist = historical_curve(sample_id)
        inputs = v7d.prepare(processor, item["images"], item["meta"]["question"])
        spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(item["images"]))
        prompt = build_prompt(item["meta"]["question"], len(item["images"]), "A")
        full_a, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
        full_b, _ = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, None)
        text_a = decode(processor, torch.tensor(full_a)).strip()
        text_b = decode(processor, torch.tensor(full_b)).strip()
        row = {
            "sample_id": sample_id,
            "full_text": text_a,
            "full_tokens": full_a,
            "full_replay": full_a == full_b and text_a == text_b,
            "historical_full_text": hist["texts"]["full"],
            "prompt_sha256": None if prompt is None else __import__("hashlib").sha256(prompt.encode()).hexdigest(),
            "pixel_values_sha256": sha_tensor(inputs["pixel_values"]),
            "input_ids_sha256": sha_tensor(inputs["input_ids"]),
        }
        if text_a != hist["texts"]["full"] or not row["full_replay"]:
            mismatches.append(f"FULL_{sample_id}")
        if sample_id == "1025" and full_a != FROZEN_1025:
            mismatches.append("TOKEN_1025")
        prior = V9C_ROWS.get(sample_id)
        if prior is not None:
            row["v9c_token_match"] = full_a == prior["tokens"]
            row["v9c_pixel_match"] = row["pixel_values_sha256"] == prior["pixel_values_sha256"]
            row["v9c_prompt_match"] = prompt == prior["prompt"]
            if not row["v9c_token_match"] or not row["v9c_pixel_match"] or not row["v9c_prompt_match"]:
                mismatches.append(f"V9C_PATH_{sample_id}")
        fast = {}
        for ratio in (0.25, 0.35):
            key = f"{ratio:.2f}"
            got = fastv_once(model, processor, inputs, spans, ratio)
            hist_text = hist["texts"][f"fastv_{ratio}"]
            hist_keep = hist["keeps"][str(ratio)]
            got["historical_text"] = hist_text
            got["historical_k"] = hist_keep["k"]
            got["historical_keep_hash"] = hist_keep["keep_hash"]
            got["match"] = got["text"] == hist_text and got["k"] == hist_keep["k"] and got["keep_hash"] == hist_keep["keep_hash"] and got["hook_match"] == 1
            if not got["match"]:
                mismatches.append(f"FASTV_{sample_id}_{key}")
            fast[key] = got
        scored = json.loads((V7D / "scored" / "NATIVE_FDP" / f"{sample_id}.json").read_text())
        selected = json.loads((V7D / "runs" / "NATIVE_FDP" / sample_id / "selected.json").read_text())
        step = int(selected["chosen"]["step"])
        png = V7D / "runs" / "NATIVE_FDP" / sample_id / "checkpoints" / f"step_{step:03d}.png"
        curve = json.loads((V7D / "runs" / "NATIVE_FDP" / sample_id / "curve.json").read_text())
        stored = next(item_row for item_row in curve if int(item_row["step"]) == step)
        if sha_file(png) != stored["png_sha256"]:
            mismatches.append(f"PNG_HASH_{sample_id}")
        from PIL import Image
        image = Image.open(png).convert("RGB")
        attacked = list(item["images"])
        attacked[item["aux_idx"]] = image
        attack_inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
        attack_spans = v7d.spans_of(attack_inputs, int(model.config.image_token_index), len(attacked))
        attack_tokens, _ = v7d.generate_tokens(model, attack_inputs, attack_inputs["pixel_values"], attack_spans, None)
        attack_text = decode(processor, torch.tensor(attack_tokens)).strip()
        native = {"full_text": attack_text, "full_match": attack_text == scored["full_text"], "ratios": {}}
        if attack_text != scored["full_text"]:
            mismatches.append(f"NATIVE_FULL_{sample_id}")
        for ratio in (0.25, 0.35):
            got = fastv_once(model, processor, attack_inputs, attack_spans, ratio)
            ref = scored["by_ratio"][str(ratio)]
            ok = got["text"] == ref["text"] and got["k"] == ref["keep"]["k"] and got["keep_hash"] == ref["keep"]["keep_hash"] and got["hook_match"] == 1
            native["ratios"][f"{ratio:.2f}"] = {"text": got["text"], "k": got["k"], "keep_hash": got["keep_hash"], "match": ok}
            if not ok:
                mismatches.append(f"NATIVE_FASTV_{sample_id}_{ratio}")
        row["clean_fastv"] = fast
        row["native_fdp_replay"] = native
        rows.append(row)
        print(json.dumps({"sample": sample_id, "full": text_a, "mismatches": mismatches}), flush=True)
    payload = {
        "status": "PASS" if not mismatches else "BASELINE_OR_REPLAY_DIVERGED",
        "mismatches": mismatches,
        "rows": rows,
        "counters": {"forward": Access.forward, "backward": Access.backward, "generate": Access.generate, "fastv": Access.fastv},
    }
    (HERE / "baseline_replay.json").write_text(json.dumps(payload, indent=2) + "\n")
    if mismatches:
        (HERE / "final_decision.json").write_text(json.dumps({"decision": "BASELINE_OR_REPLAY_DIVERGED", "mismatches": mismatches}, indent=2) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
