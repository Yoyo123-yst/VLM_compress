"""Freeze selected v11c checkpoints as complete multi-page PNG sets.

This script does not load the model or call FastV. It only reconstructs the
full page list, replaces the selected auxiliary page, and creates a matched
uint8-noise control.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v11c_deterministic_native_fdp"
SAMPLES = ("1025", "2008", "47450", "60700", "7006", "39204")
NOISE_SEED_PREFIX = "v11c-transfer-noise-20261012|"
EPS_U8 = 8


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_seed(sample: str) -> int:
    return int(hashlib.sha256(f"{NOISE_SEED_PREFIX}{sample}".encode()).hexdigest()[:8], 16)


def save_pages(sample: str, branch: str, pages: list[Image.Image]) -> list[str]:
    out = HERE / "transfer" / "prepared" / branch / sample
    out.mkdir(parents=True, exist_ok=True)
    hashes = []
    for index, page in enumerate(pages):
        path = out / f"page_{index}.png"
        page.convert("RGB").save(path, format="PNG")
        hashes.append(sha_file(path))
    return hashes


def main() -> None:
    # Importing v7d is allowed here: this is data reconstruction only, not a
    # model/target evaluation process.
    from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d

    manifest = {"status": "PREPARED", "samples": {}, "noise_rule": NOISE_SEED_PREFIX}
    for sample in SAMPLES:
        item = run_v7d.load_dev(sample)
        selected = json.loads((HERE / "training_h0" / sample / "selected.json").read_text())
        chosen = selected.get("chosen")
        if not chosen:
            manifest["samples"][sample] = {"status": "NO_VALID_SELECTED"}
            continue
        step = int(chosen["step"])
        checkpoint = HERE / "training_h0" / sample / "checkpoints" / f"step_{step:03d}.png"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        clean_pages = [page.convert("RGB") for page in item["images"]]
        attack_pages = list(clean_pages)
        attack_pages[item["aux_idx"]] = Image.open(checkpoint).convert("RGB")
        base = np.asarray(clean_pages[item["aux_idx"]], dtype=np.int16)
        rng = np.random.default_rng(noise_seed(sample))
        noise = rng.integers(-EPS_U8, EPS_U8 + 1, size=base.shape, dtype=np.int16)
        noisy = np.clip(base + noise, 0, 255).astype(np.uint8)
        noise_pages = list(clean_pages)
        noise_pages[item["aux_idx"]] = Image.fromarray(noisy, mode="RGB")
        branches = {
            "clean": clean_pages,
            "attack": attack_pages,
            "noise": noise_pages,
        }
        branch_hashes = {name: save_pages(sample, name, pages) for name, pages in branches.items()}
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
            "noise_seed": noise_seed(sample),
        }
    out = HERE / "transfer" / "prepared_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "samples": manifest["samples"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
