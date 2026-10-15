"""Seal 32 uint8 noise controls per development sample, then Full-gate them.

This process does not import or call FastV.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V9C = ROOT / "week2_attack_v9c_multiscale_rank_oblivious"
OUT = V9C / "noise_multiseed_diagnostic"
sys.path.insert(0, str(V9C))

from common import (  # noqa: E402
    answers_of,
    load_build_prompt,
    load_model,
    manifest_samples,
    output_valid,
    prepare,
    generate_tokens,
    sha_file,
    uint8_linf,
)

IDS = ("1025", "2008", "7006", "47450", "39204")
RATIOS = (0.15, 0.25, 0.35, 0.50)
S = 32
BASE_SEED = 2026101122
ATTEMPT_STRIDE = 10_000_000
MAX_ATTEMPTS = 20


def seed_value(sample_id: str, seed_index: int, attempt: int) -> int:
    return BASE_SEED + int(sample_id) * 1000 + seed_index + attempt * ATTEMPT_STRIDE


def draw_noise(clean: np.ndarray, sample_id: str, seed_index: int) -> tuple[np.ndarray, int, int]:
    for attempt in range(MAX_ATTEMPTS):
        seed = seed_value(sample_id, seed_index, attempt)
        delta = np.random.default_rng(seed).integers(-8, 9, size=clean.shape, dtype=np.int16)
        noisy = np.clip(clean.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        diff = np.abs(noisy.astype(np.int16) - clean.astype(np.int16))
        if int(diff.max()) == 8 and bool(diff.any()):
            return noisy, seed, attempt
    raise RuntimeError(f"resample cap reached for {sample_id} seed_index={seed_index}")


def load_pages(sample_id: str, count: int) -> list[Image.Image]:
    folder = V9C / "baseline_pages" / sample_id
    return [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(count)]


def full_text(model, processor, images, question: str) -> tuple[str, list[int]]:
    tokens, text = generate_tokens(model, processor, prepare(processor, images, question))
    return text, tokens


def main() -> None:
    build_prompt, prompt_hash, prompt_path = load_build_prompt()
    specs = {row["sample_id"]: row for row in manifest_samples()}
    attack = json.loads((V9C / "target_eval_controls" / "attack_replay.json").read_text())
    attack_rows = {row["sample_id"]: row for row in attack["rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": "5-sample development multi-seed noise diagnostic",
        "model": "/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf",
        "prompt_path": prompt_path,
        "prompt_sha256": prompt_hash,
        "prompt_wrapper": "historical build_prompt template A, executed from source",
        "generation": {"do_sample": False, "max_new_tokens": 16, "use_cache": True, "pinpoints": [[768, 768]], "dtype": "bfloat16", "attn_implementation": "sdpa"},
        "samples": list(IDS),
        "ratios": list(RATIOS),
        "seeds_per_sample": S,
        "seed_rule": "BASE_SEED + int(sample_id) * 1000 + seed_index + attempt * 10000000",
        "base_seed": BASE_SEED,
        "noise": "independent uint8 uniform integers in [-8, 8] on the auxiliary page only",
        "resample_rule": "redraw with the next attempt until max abs uint8 delta is exactly 8 and the delta is nonzero; record attempt count; cap 20",
        "decision_rule": {
            "endpoint_failure": "valid and ANLS < 0.5",
            "induced_random_failure": "clean is not an endpoint failure, Full gate passed, and the random output is an endpoint failure",
            "wilson": "95 percent Wilson score interval",
            "supported_only_if": "consistency checks pass and sample 1025 at r=0.25 is an attack induced failure and 0 of 32 Full-gated seeds are induced random failures",
            "single_attack_is_not_a_significance_test": True,
            "otherwise_if_consistent": "GRAYBOX_SIGNAL_NOT_ESTABLISHED_ABOVE_RANDOM",
            "if_inconsistent": "DIAGNOSTIC_INVALID_OR_REPLAY_DIVERGED",
        },
        "inputs": {},
    }
    gate_path = OUT / "full_gate.jsonl"
    if gate_path.exists():
        gate_path.unlink()
    model, processor = load_model()
    passed = 0
    for sample_id in IDS:
        spec = specs[sample_id]
        baseline = V9C / "baseline_pages" / sample_id
        page_hash = [sha_file(baseline / f"page_{index}.png") for index in range(spec["M"])]
        if page_hash != spec["page_sha256"]:
            raise SystemExit(f"baseline hash mismatch {sample_id}")
        attack_row = attack_rows[sample_id]
        attack_hash = attack_row.get("png_sha256")
        if attack_row.get("evaluated"):
            sealed = V9C / "sealed_png" / sample_id / f"page_{spec['aux_idx']}.png"
            if sha_file(sealed) != attack_hash:
                raise SystemExit(f"attack png hash mismatch {sample_id}")
        manifest["inputs"][sample_id] = {
            "aux_idx": spec["aux_idx"],
            "evidence_idx": spec["evidence_idx"],
            "M": spec["M"],
            "clean_full_text": spec["clean_full_pred"],
            "baseline_page_sha256": page_hash,
            "attack_png_sha256": attack_hash,
            "attack_status": attack_row.get("status"),
        }
        clean_pages = load_pages(sample_id, spec["M"])
        clean_aux = np.asarray(clean_pages[spec["aux_idx"]].convert("RGB"), dtype=np.uint8)
        reference = full_text(model, processor, clean_pages, spec["question"])[0]
        if reference != spec["clean_full_pred"] or not output_valid(reference):
            raise SystemExit(f"clean Full text mismatch before noise {sample_id}: {reference!r}")
        for seed_index in range(S):
            noisy, seed, resamples = draw_noise(clean_aux, sample_id, seed_index)
            dest = OUT / "noise_png" / sample_id / f"seed_{seed_index:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            for index in range(spec["M"]):
                source = baseline / f"page_{index}.png"
                target = dest / f"page_{index}.png"
                if index == spec["aux_idx"]:
                    Image.fromarray(noisy, mode="RGB").save(target)
                else:
                    shutil.copyfile(source, target)
            reloaded = [Image.open(dest / f"page_{index}.png").convert("RGB") for index in range(spec["M"])]
            changed = [index for index in range(spec["M"]) if sha_file(dest / f"page_{index}.png") != page_hash[index]]
            linf_uint8 = int(np.abs(np.asarray(reloaded[spec["aux_idx"]], dtype=np.int16) - clean_aux.astype(np.int16)).max())
            linf = uint8_linf(clean_pages[spec["aux_idx"]], reloaded[spec["aux_idx"]])
            first_text, first_tokens = full_text(model, processor, reloaded, spec["question"])
            second = [Image.open(dest / f"page_{index}.png").convert("RGB") for index in range(spec["M"])]
            second_text, second_tokens = full_text(model, processor, second, spec["question"])
            png_hash = sha_file(dest / f"page_{spec['aux_idx']}.png")
            reasons = []
            if changed != [spec["aux_idx"]]:
                reasons.append("CHANGED_PAGE_INDEX")
            if linf_uint8 != 8:
                reasons.append("LINF")
            if png_hash == page_hash[spec["aux_idx"]]:
                reasons.append("ZERO_OR_IDENTICAL")
            if not output_valid(first_text) or not output_valid(second_text):
                reasons.append("INVALID_OUTPUT")
            if first_text != spec["clean_full_pred"] or second_text != spec["clean_full_pred"] or first_text != second_text:
                reasons.append("FULL_TEXT")
            if first_tokens != second_tokens:
                reasons.append("FULL_TOKEN")
            status = "PASS" if not reasons else "INVALID_CONTROL_FULL_GATE"
            if status == "PASS":
                passed += 1
            record = {
                "sample_id": sample_id,
                "seed_index": seed_index,
                "seed": seed,
                "resample_count": resamples,
                "aux_idx": spec["aux_idx"],
                "status": status,
                "reasons": reasons,
                "full_text": second_text,
                "full_token_ids": second_tokens,
                "full_replay_match": first_text == second_text and first_tokens == second_tokens,
                "clean_full_text": spec["clean_full_pred"],
                "linf_uint8": linf_uint8,
                "linf": linf,
                "changed_pages": changed,
                "png_path": str(dest / f"page_{spec['aux_idx']}.png"),
                "png_sha256": png_hash,
                "page_sha256": [sha_file(dest / f"page_{index}.png") for index in range(spec["M"])],
                "answers": answers_of(spec),
            }
            with gate_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({"sample": sample_id, "seed_index": seed_index, "status": status, "resamples": resamples, "passed": passed}), flush=True)
    manifest["full_gate_pass"] = passed
    manifest["full_gate_total"] = len(IDS) * S
    manifest["prepare_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"full_gate_pass": passed, "total": len(IDS) * S}), flush=True)


if __name__ == "__main__":
    main()
