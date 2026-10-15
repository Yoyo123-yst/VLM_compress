"""Freeze holdout checkpoints into complete clean/attack/noise PNG sets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v11c_deterministic_native_fdp"
OUT = HERE / "holdout_transfer"
NOISE_PREFIX = "v11c-holdout-noise-20261012|"
EPS_U8 = 8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(sample: str) -> int:
    return int(hashlib.sha256(f"{NOISE_PREFIX}{sample}".encode()).hexdigest()[:8], 16)


def save_pages(sample: str, branch: str, pages: list[Image.Image]) -> list[str]:
    folder = OUT / "prepared" / branch / sample
    folder.mkdir(parents=True, exist_ok=True)
    hashes = []
    for index, page in enumerate(pages):
        path = folder / f"page_{index}.png"
        page.convert("RGB").save(path, format="PNG")
        hashes.append(sha(path))
    return hashes


def main() -> None:
    from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d

    frozen = json.loads((HERE / "holdout_ids.json").read_text())
    manifest = {
        "status": "PREPARED",
        "protocol": "prospective v11c holdout; no target FastV calls during preparation",
        "selection": frozen,
        "noise_rule": NOISE_PREFIX,
        "samples": {},
    }
    for sample in frozen["ids"]:
        item = run_v7d.load_pristine(sample)
        selected_path = HERE / "holdout_train" / sample / "selected.json"
        selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
        chosen = selected.get("chosen")
        if not chosen:
            manifest["samples"][sample] = {"status": "NO_VALID_SELECTED"}
            continue
        checkpoint = HERE / "holdout_train" / sample / "checkpoints" / f"step_{int(chosen['step']):03d}.png"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        clean_pages = [page.convert("RGB") for page in item["images"]]
        attack_pages = list(clean_pages)
        attack_pages[item["aux_idx"]] = Image.open(checkpoint).convert("RGB")
        base = np.asarray(clean_pages[item["aux_idx"]], dtype=np.int16)
        rng = np.random.default_rng(seed(sample))
        noise = rng.integers(-EPS_U8, EPS_U8 + 1, size=base.shape, dtype=np.int16)
        noisy = np.clip(base + noise, 0, 255).astype(np.uint8)
        noise_pages = list(clean_pages)
        noise_pages[item["aux_idx"]] = Image.fromarray(noisy, mode="RGB")
        branch_hashes = {
            "clean": save_pages(sample, "clean", clean_pages),
            "attack": save_pages(sample, "attack", attack_pages),
            "noise": save_pages(sample, "noise", noise_pages),
        }
        manifest["samples"][sample] = {
            "status": "OK",
            "selected": chosen,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "M": len(clean_pages),
            "aux_idx": int(item["aux_idx"]),
            "evidence_idx": int(item["evidence_idx"] if isinstance(item["evidence_idx"], int) else item["evidence_idx"][0]),
            "question": item["meta"]["question"],
            "answers": item["meta"]["answers"],
            "clean_text": selected["clean_text"],
            "page_hashes": branch_hashes,
            "noise_seed": seed(sample),
        }
    out = OUT / "prepared_manifest.json"
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "samples": len(manifest["samples"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
