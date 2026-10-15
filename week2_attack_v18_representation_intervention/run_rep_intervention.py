"""Minimal causal intervention at FastV's layer-3 pruning boundary.

For existing strict attack PNGs, capture the clean pre-pruning hidden states,
then replace one page's attack hidden states with clean hidden states before the
real FastV prune hook runs. Endpoint generation is unforced. This is a
mechanism probe, not an attack method or a new result table.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v18_representation_intervention"
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
sys.path[:0] = [
    str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"),
    str(ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation"),
]

from adapters.runtime import clear_hooks, install  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import production_generate  # noqa: E402

IDS = ("1025", "2008", "7006", "39196")
RATIO = 0.25


def configure():
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("VATTACK_ATTN_IMPLEMENTATION", "sdpa")
    os.environ.setdefault("VATTACK_DETERMINISTIC_RESIZE", "1")
    os.environ.setdefault("VATTACK_RESIZE_MODE", "bilinear")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_with_capture(model, processor, inputs, spans, ratio, replacement=None, capture=False):
    """Run real FastV; replacement is a page-index -> clean hidden slice map."""
    clear_hooks(model)
    state = install(model, "fastv", ratio, int(model.config.image_token_index))
    state.spans = spans
    holder = {}
    layer = model.model.language_model.layers[3]

    def intervene(module, args, kwargs):
        hidden = args[0]
        if hidden.shape[1] == 1 or not spans:
            return None
        if capture:
            holder["hidden"] = hidden.detach().clone()
        if replacement:
            updated = hidden.clone()
            for page, clean_hidden in replacement.items():
                start, end = spans[page]
                updated[:, start:end, :] = clean_hidden[:, start:end, :].to(updated.dtype)
            return (updated,), kwargs
        return None

    handle = layer.register_forward_pre_hook(intervene, with_kwargs=True, prepend=True)
    try:
        out = production_generate(model, inputs)
        return out, list(state.kept_visual), holder.get("hidden")
    finally:
        handle.remove()
        clear_hooks(model)


def prepare(processor, item, image):
    images = list(item["images"])
    images[item["aux_idx"]] = image
    inputs = v7d.prepare(processor, images, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(item["model_image_token_index"]), len(images))
    return images, inputs, spans


def decode(processor, tokens):
    return v7d.decode(processor, torch.tensor(tokens)).strip()


def endpoint(model, processor, inputs, spans, ratio):
    out, kept, _ = run_with_capture(model, processor, inputs, spans, ratio)
    return {
        "tokens": out["tokens"],
        "text": decode(processor, out["tokens"]),
        "kept": kept,
    }


def main():
    configure()
    model, processor = base.load_model_frozen()
    model.eval()
    rows = []
    for sid in IDS:
        item = v7d.load_dev(sid)
        item["model_image_token_index"] = int(model.config.image_token_index)
        clean_image = item["images"][item["aux_idx"]]
        attack_path = V5B / "attacks" / "artifacts" / sid / "final.png"
        attack_image = Image.open(attack_path).convert("RGB")

        clean_images, clean_inputs, clean_spans = prepare(processor, item, clean_image)
        attack_images, attack_inputs, attack_spans = prepare(processor, item, attack_image)
        if clean_spans != attack_spans:
            raise RuntimeError(f"{sid}: clean/attack spans differ")

        full_clean = v7d.generate_tokens(model, clean_inputs, clean_inputs["pixel_values"], clean_spans, None)[0]
        full_attack = v7d.generate_tokens(model, attack_inputs, attack_inputs["pixel_values"], attack_spans, None)[0]
        clean_out, clean_kept, clean_hidden = run_with_capture(
            model, processor, clean_inputs, clean_spans, RATIO, capture=True
        )
        attack_out, attack_kept, attack_hidden = run_with_capture(
            model, processor, attack_inputs, attack_spans, RATIO, capture=True
        )
        if clean_hidden is None or attack_hidden is None:
            raise RuntimeError(f"{sid}: no pre-prune hidden capture")
        if clean_hidden.shape != attack_hidden.shape:
            raise RuntimeError(f"{sid}: hidden shapes differ {clean_hidden.shape} {attack_hidden.shape}")

        groups = {}
        aux = int(item["aux_idx"])
        pages = list(range(len(clean_spans)))
        for name, page_set in {
            "aux": [aux],
            "evidence": [int(item["evidence_idx"]) if isinstance(item["evidence_idx"], int) else int(item["evidence_idx"][0])],
            "other_pages": [p for p in pages if p not in {aux, int(item["evidence_idx"]) if isinstance(item["evidence_idx"], int) else int(item["evidence_idx"][0])}],
            "all_visual": pages,
        }.items():
            repl = {p: clean_hidden for p in page_set}
            out, kept, _ = run_with_capture(
                model, processor, attack_inputs, attack_spans, RATIO, replacement=repl
            )
            text = decode(processor, out["tokens"])
            groups[name] = {
                "pages_replaced": page_set,
                "text": text,
                "tokens": out["tokens"],
                "anls": float(anls_score(text, list(item["meta"]["answers"]))),
                "keep_hash": hashlib.sha256(json.dumps(kept).encode()).hexdigest(),
                "k": len(kept),
            }
        # Directional controls start from the attack hidden state. The exact
        # clean displacement is the repair intervention; random and orthogonal
        # directions have the same norm but different orientation.
        start, end = attack_spans[aux]
        clean_slice = clean_hidden[:, start:end, :]
        attack_slice = attack_hidden[:, start:end, :]
        diff_norm = (attack_slice - clean_slice).float().norm()
        directional = {"exact_clean": groups["aux"]}
        for kind in ("random", "orthogonal"):
            directional[kind] = []
            for seed_offset in range(3):
                gen = torch.Generator(device=clean_slice.device).manual_seed(
                    20261013 + int(sid) + 100 * seed_offset + (0 if kind == "random" else 50)
                )
                noise = torch.randn(clean_slice.shape, device=clean_slice.device, dtype=clean_slice.dtype, generator=gen)
                if kind == "orthogonal":
                    direction = attack_slice - clean_slice
                    denom = direction.float().pow(2).sum().clamp_min(1e-8)
                    noise = noise - direction * ((noise.float() * direction.float()).sum() / denom).to(noise.dtype)
                noise_norm = noise.float().norm().clamp_min(1e-8)
                controlled = attack_hidden.clone()
                controlled[:, start:end, :] = attack_slice + noise * (diff_norm / noise_norm).to(noise.dtype)
                out, kept, _ = run_with_capture(
                    model, processor, attack_inputs, attack_spans, RATIO,
                    replacement={aux: controlled},
                )
                text = decode(processor, out["tokens"])
                directional[kind].append({
                    "seed": seed_offset,
                    "text": text,
                    "tokens": out["tokens"],
                    "anls": float(anls_score(text, list(item["meta"]["answers"]))),
                    "k": len(kept),
                    "norm": float((controlled[:, start:end, :] - attack_slice).float().norm().detach().cpu()),
                })
        groups["directional_controls"] = directional
        # Reverse intervention: inject the attacked auxiliary representation
        # into the clean image while leaving pixels and the natural keep rule
        # unchanged. This is the bidirectional counterpart of repair.
        for name, page_set in {
            "reverse_aux": [aux],
            "reverse_all_visual": pages,
        }.items():
            repl = {p: attack_hidden for p in page_set}
            out, kept, _ = run_with_capture(
                model, processor, clean_inputs, clean_spans, RATIO, replacement=repl
            )
            text = decode(processor, out["tokens"])
            groups[name] = {
                "pages_replaced": page_set,
                "text": text,
                "tokens": out["tokens"],
                "anls": float(anls_score(text, list(item["meta"]["answers"]))),
                "keep_hash": hashlib.sha256(json.dumps(kept).encode()).hexdigest(),
                "k": len(kept),
            }
        record = {
            "sample_id": sid,
            "ratio": RATIO,
            "attack_png": str(attack_path.relative_to(ROOT)),
            "attack_png_sha256": sha(attack_path),
            "full_clean": {"tokens": full_clean, "text": decode(processor, full_clean)},
            "full_attack": {"tokens": full_attack, "text": decode(processor, full_attack)},
            "fast_clean": {"tokens": clean_out["tokens"], "text": decode(processor, clean_out["tokens"]), "k": len(clean_kept)},
            "fast_attack": {"tokens": attack_out["tokens"], "text": decode(processor, attack_out["tokens"]), "k": len(attack_kept)},
            "interventions": groups,
        }
        rows.append(record)
        summary_interventions = {}
        for k, v in groups.items():
            if k == "directional_controls":
                summary_interventions[k] = {}
                for kind, seeds in v.items():
                    if isinstance(seeds, list):
                        summary_interventions[k][kind] = {
                            str(item["seed"]): item["text"] for item in seeds
                        }
                    else:
                        summary_interventions[k][kind] = seeds["text"]
            else:
                summary_interventions[k] = v["text"]
        print(json.dumps({"id": sid, "clean": record["fast_clean"]["text"], "attack": record["fast_attack"]["text"], "interventions": summary_interventions}, ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "representation_intervention.json").write_text(json.dumps({"scope": "diagnostic_only", "rows": rows}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
