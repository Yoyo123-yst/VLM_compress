"""Seal locally accepted PNGs. This process does not load FastV."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from common import ADMISSION_IDS, EPS, HERE, TOLERANCE, fastv_modules, load_images, manifest_samples, sha_file, sha_image, uint8_linf
from guard import assert_clean, install_guard


def main() -> None:
    install_guard()
    sealed = HERE / "sealed_candidates"
    sealed.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id in ADMISSION_IDS:
        path = HERE / "per_sample" / f"{sample_id}.json"
        if not path.exists():
            rows.append({"sample_id": sample_id, "sealed": False, "reason": "missing training record"})
            continue
        record = json.loads(path.read_text())
        if not record.get("local_pass") or record.get("chosen") is None:
            rows.append({"sample_id": sample_id, "sealed": False, "reason": "local gate failed"})
            continue
        spec = next(row for row in manifest_samples() if row["sample_id"] == sample_id)
        clean = load_images(spec)
        aux = Image.open(record["chosen"]["png"]).convert("RGB")
        if uint8_linf(clean[spec["aux_idx"]], aux) > EPS + TOLERANCE:
            rows.append({"sample_id": sample_id, "sealed": False, "reason": "linf"})
            continue
        dest = sealed / sample_id
        dest.mkdir(parents=True, exist_ok=True)
        pages = []
        for index, image in enumerate(clean):
            page = aux if index == spec["aux_idx"] else image
            target = dest / f"page_{index}.png"
            page.save(target)
            pages.append({"index": index, "sha256": sha_file(target), "aux": index == spec["aux_idx"]})
        rows.append({
            "sample_id": sample_id,
            "sealed": True,
            "pages": pages,
            "clean_text": record["clean_text"],
            "clean_tokens": record["clean_tokens"],
            "aux_idx": spec["aux_idx"],
            "evidence_idx": spec["evidence_idx"],
            "evidence_sha256": sha_image(clean[spec["evidence_idx"]]),
            "training_log_sha256": sha_file(path),
            "source_png_sha256": record["chosen"]["png_sha256"],
        })
    manifest = {"samples": rows, "fastv_access_count": 0, "fastv_modules": fastv_modules()}
    (HERE / "seal_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sealed": sum(row["sealed"] for row in rows)}), flush=True)
    assert_clean("seal")


if __name__ == "__main__":
    main()
