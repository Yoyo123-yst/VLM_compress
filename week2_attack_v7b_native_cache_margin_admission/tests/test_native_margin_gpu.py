"""GPU admission for the 10 frozen conditions. No PGD."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7b_native_cache_margin_admission"
V7 = ROOT / "week2_attack_v7_pristine60"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks, install  # noqa: E402
from run_causal_v2 import decode, prepare, spans_of  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v7_pristine60.load_samples import load_items, manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    FASTV_RATIO,
    aux_soft_gradient,
    cache_replay,
    certificate_status,
    cone_holds,
    float_list,
    margins_for_tokens,
    native_rows_from_full_ids,
    production_generate,
    require_keep_match,
    termination_of,
)

CONDITIONS = [
    ("60744", "random_noise"),
    ("60744", "clean"),
    ("1023", "clean"),
    ("51140", "random_noise"),
    ("22317", "clean"),
    ("22317", "B"),
    ("22317", "random_noise"),
    ("790", "B"),
    ("7029", "B"),
    ("1170", "random_noise"),
]
GRADIENT_CONDITIONS = CONDITIONS[:3]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_path(sample_id: str) -> Path:
    hits = list((V7 / "controls" / "single").glob(f"shard_s*/{sample_id}/noise.png"))
    if len(hits) != 1:
        raise RuntimeError(f"noise png count for {sample_id} is {len(hits)}")
    return hits[0]


def attack_path(selected: dict, sample_id: str) -> Path:
    row = selected[sample_id]
    return ROOT / row["artifact"] / row["png"]


def condition_image(item, kind: str, selected: dict) -> tuple[Image.Image, str, str]:
    sid = item["meta"]["sample_id"]
    if kind == "clean":
        image = item["images"][item["aux_idx"]]
        return image, "dataset_aux_page", item["page_sha"][item["aux_idx"]]
    if kind == "B":
        path = attack_path(selected, sid)
        return Image.open(path).convert("RGB"), str(path.relative_to(ROOT)), sha_file(path)
    path = noise_path(sid)
    return Image.open(path).convert("RGB"), str(path.relative_to(ROOT)), sha_file(path)


def compare_rows(tokens: list[int], generated_scores: list[torch.Tensor], replay_rows: list[torch.Tensor]) -> dict:
    steps = []
    mismatch = None
    for index, token in enumerate(tokens):
        g_argmax = int(generated_scores[index].argmax())
        t_argmax = int(replay_rows[index].argmax())
        gap = float((generated_scores[index].float().cpu() - replay_rows[index].float().cpu()).abs().max())
        step = {
            "index": index,
            "generated_token": token,
            "g_argmax": g_argmax,
            "t_argmax": t_argmax,
            "is_eos": None,
            "max_abs_logit_diff": gap,
            "cache_step": index,
        }
        steps.append(step)
        if mismatch is None and (g_argmax != token or t_argmax != token or g_argmax != t_argmax):
            mismatch = {"index": index, "generated_token": token, "g_argmax": g_argmax, "t_argmax": t_argmax, "max_abs_logit_diff": gap}
    return {
        "g_t_token_match": all(step["g_argmax"] == step["generated_token"] for step in steps),
        "g_t_argmax_match": all(step["g_argmax"] == step["t_argmax"] == step["generated_token"] for step in steps),
        "steps": steps,
        "first_mismatch": mismatch,
    }


def write_decision(decision: str, reason: str) -> None:
    payload = {
        "decision": decision,
        "reason": reason,
        "existing_b_strict": "52/60 unchanged; this stage did not recompute it",
        "retrospective_certificate_for_b": False,
        "admits_future_bplus_basis": decision == "PASS_NATIVE_CACHE_MARGIN_ADMITTED",
    }
    (HERE / "final_decision.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    specs = {row["sample_id"]: row for row in manifest()["samples"]}
    selected = {row["sample_id"]: row for row in __import__("csv").DictReader((V7 / "attacks" / "selector" / "selected.csv").open())}
    needed = sorted({sid for sid, _ in CONDITIONS})
    items = {item["meta"]["sample_id"]: item for item in load_items([specs[sid] for sid in needed])}
    model, processor = load_model_frozen()
    eos = int(model.generation_config.eos_token_id)
    align = HERE / "alignment"
    health = HERE / "health"
    align.mkdir(parents=True, exist_ok=True)
    health.mkdir(parents=True, exist_ok=True)
    clean_tokens = {}
    clean_text = {}
    full_rows = []
    for sid in needed:
        inputs = prepare(processor, items[sid]["images"], items[sid]["meta"]["question"])
        generated = production_generate(model, inputs)
        clean_tokens[sid] = generated["tokens"]
        clean_text[sid] = decode(processor, torch.tensor(generated["tokens"]))
    for sid, kind in CONDITIONS:
        item = items[sid]
        image, source, digest = condition_image(item, kind, selected)
        images = list(item["images"])
        images[item["aux_idx"]] = image
        inputs = prepare(processor, images, item["meta"]["question"])
        generated = production_generate(model, inputs)
        replay, cache_meta = cache_replay(model, inputs, generated["tokens"])
        compared = compare_rows(generated["tokens"], generated["scores"], replay)
        text = decode(processor, torch.tensor(generated["tokens"]))
        hard = float_list(margins_for_tokens(replay, generated["tokens"])["hard"])
        soft = float_list(margins_for_tokens(replay, generated["tokens"])["soft"])
        holds = cone_holds(hard)
        exact = text == clean_text[sid]
        record = {
            "sample_id": sid,
            "image_kind": kind,
            "source": source,
            "sha256": digest,
            "clean_text": clean_text[sid],
            "full_text": text,
            "fastv_text": None,
            "token_ids": generated["tokens"],
            "clean_token_ids": clean_tokens[sid],
            "termination": termination_of(generated["tokens"], eos),
            "appended_eos": False,
            "g_t_token_match": compared["g_t_token_match"],
            "g_t_argmax_match": compared["g_t_argmax_match"],
            "hard_margin": hard,
            "soft_margin": soft,
            "cone_holds": holds,
            "full_exact": exact,
            "certificate_status": certificate_status(holds, exact),
            "fastv_k": None,
            "fastv_keep_hash": None,
            "first_mismatch": compared["first_mismatch"],
            "steps": compared["steps"],
            "cache_positions": cache_meta,
            "inputs": {key: list(value.shape) if torch.is_tensor(value) else type(value).__name__ for key, value in inputs.items()},
        }
        for step, token in zip(record["steps"], generated["tokens"]):
            step["is_eos"] = token == eos
            step["is_final"] = step["index"] == len(generated["tokens"]) - 1
        (align / f"{sid}__{kind}.json").write_text(json.dumps(record) + "\n")
        full_rows.append(record)
        print(json.dumps({"full": sid, "kind": kind, "match": record["g_t_argmax_match"], "termination": record["termination"]}), flush=True)
        if not record["g_t_argmax_match"] or not record["g_t_token_match"]:
            summary = {"pass": False, "n": len(full_rows), "rows": [{"sample_id": row["sample_id"], "image_kind": row["image_kind"], "match": row["g_t_argmax_match"]} for row in full_rows]}
            (health / "full_alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            write_decision("BLOCKED_NATIVE_FULL_GENERATE_MISMATCH", f"{sid} {kind} Full G/T argmax mismatch")
            raise SystemExit(2)
    (health / "full_alignment_summary.json").write_text(json.dumps({"pass": True, "n": len(full_rows)}, indent=2) + "\n")
    fast_rows = []
    for sid, kind in CONDITIONS:
        item = items[sid]
        image, source, digest = condition_image(item, kind, selected)
        images = list(item["images"])
        images[item["aux_idx"]] = image
        clear_hooks(model)
        inputs = prepare(processor, images, item["meta"]["question"])
        token_id = int(model.config.image_token_index)
        spans = spans_of(inputs, token_id, len(images))
        state = install(model, "fastv", FASTV_RATIO, token_id)
        state.spans = spans
        generated = production_generate(model, inputs)
        ref_kept = list(state.kept_visual)
        clear_hooks(model)
        state = install(model, "fastv", FASTV_RATIO, token_id)
        state.spans = spans
        replay, _cache_meta = cache_replay(model, inputs, generated["tokens"])
        got_kept = list(state.kept_visual)
        clear_hooks(model)
        try:
            require_keep_match(len(got_kept), keep_hash(got_kept), len(ref_kept), keep_hash(ref_kept))
            keep_ok = True
        except RuntimeError as exc:
            keep_ok = False
            keep_error = str(exc)
        else:
            keep_error = ""
        compared = compare_rows(generated["tokens"], generated["scores"], replay)
        text = decode(processor, torch.tensor(generated["tokens"]))
        hard = float_list(margins_for_tokens(replay, generated["tokens"])["hard"])
        soft = float_list(margins_for_tokens(replay, generated["tokens"])["soft"])
        path = align / f"{sid}__{kind}.json"
        record = json.loads(path.read_text())
        record["fastv_text"] = text
        record["fastv_token_ids"] = generated["tokens"]
        record["fastv_termination"] = termination_of(generated["tokens"], eos)
        record["fastv_g_t_token_match"] = compared["g_t_token_match"]
        record["fastv_g_t_argmax_match"] = compared["g_t_argmax_match"]
        record["fastv_hard_margin"] = hard
        record["fastv_soft_margin"] = soft
        record["fastv_k"] = len(ref_kept)
        record["fastv_keep_hash"] = keep_hash(ref_kept)
        record["fastv_keep_match"] = keep_ok
        record["fastv_first_mismatch"] = compared["first_mismatch"]
        record["fastv_keep_error"] = keep_error
        record["fastv_max_abs_logit_diff"] = max(step["max_abs_logit_diff"] for step in compared["steps"])
        path.write_text(json.dumps(record) + "\n")
        fast_rows.append(record)
        print(json.dumps({"fastv": sid, "kind": kind, "match": compared["g_t_argmax_match"], "keep": keep_ok, "k": len(ref_kept)}), flush=True)
        if not compared["g_t_argmax_match"] or not keep_ok:
            (health / "fastv_alignment_summary.json").write_text(json.dumps({"pass": False, "n": len(fast_rows)}, indent=2) + "\n")
            write_decision("BLOCKED_NATIVE_FASTV_GENERATE_MISMATCH", f"{sid} {kind} FastV alignment failed")
            raise SystemExit(3)
    (health / "fastv_alignment_summary.json").write_text(json.dumps({"pass": True, "n": len(fast_rows)}, indent=2) + "\n")
    gradients = []
    for sid, kind in GRADIENT_CONDITIONS:
        item = items[sid]
        image, _, _ = condition_image(item, kind, selected)
        images = list(item["images"])
        images[item["aux_idx"]] = image
        inputs = prepare(processor, images, item["meta"]["question"])
        generated = production_generate(model, inputs)
        replay, _cache_meta = cache_replay(model, inputs, generated["tokens"])
        pixels = inputs["pixel_values"].detach()
        with torch.inference_mode():
            native = native_rows_from_full_ids(model, inputs, generated["tokens"], pixels)
        tokens = list(generated["tokens"])
        native_match = all(int(row.argmax()) == int(ref.argmax()) for row, ref in zip(native, replay))
        del native, replay, generated
        torch.cuda.empty_cache()
        evidence = item["evidence_idx"] if isinstance(item["evidence_idx"], int) else int(item["evidence_idx"][0])
        try:
            grad = aux_soft_gradient(model, inputs, tokens, item["aux_idx"], evidence)
        except RuntimeError as exc:
            gradients.append({"sample_id": sid, "image_kind": kind, "error": str(exc)})
            (health / "gradient_health.json").write_text(json.dumps({"pass": False, "rows": gradients}, indent=2) + "\n")
            write_decision("BLOCKED_NATIVE_MARGIN_NOT_DIFFERENTIABLE", f"{sid} {kind} native margin gradient failed: {exc}")
            raise SystemExit(4)
        grad.update({"sample_id": sid, "image_kind": kind, "native_forward_argmax_matches_cache": native_match})
        gradients.append(grad)
        print(json.dumps({"grad": sid, "kind": kind, **{k: grad[k] for k in ("aux_grad_l2", "evidence_grad_l2", "native_forward_argmax_matches_cache")}}), flush=True)
        ok = (
            native_match
            and grad["aux_grad_l2"] is not None
            and grad["aux_grad_l2"] > 0
            and grad["evidence_grad_l2"] == 0.0
        )
        if not ok:
            (health / "gradient_health.json").write_text(json.dumps({"pass": False, "rows": gradients}, indent=2) + "\n")
            write_decision("BLOCKED_NATIVE_MARGIN_NOT_DIFFERENTIABLE", f"{sid} {kind} native margin gradient failed")
            raise SystemExit(4)
    (health / "gradient_health.json").write_text(json.dumps({"pass": True, "rows": gradients}, indent=2) + "\n")
    write_decision("PASS_NATIVE_CACHE_MARGIN_ADMITTED", "CPU tests, Full 10/10, FastV 10/10, and gradient health passed")
    print(json.dumps({"decision": "PASS_NATIVE_CACHE_MARGIN_ADMITTED"}))


if __name__ == "__main__":
    main()
