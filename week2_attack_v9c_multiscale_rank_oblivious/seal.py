"""Copy accepted auxiliary pages into sealed_png. No FastV import."""

from __future__ import annotations

import json

from PIL import Image

from common import ADMISSION_IDS, EPS, GLOBAL_SEED, HERE, TOLERANCE, fastv_modules, load_images, manifest_samples, sha_file, uint8_linf
from guard import assert_clean, install_guard


def main() -> None:
    install_guard()
    rows = []
    for sample_id in ADMISSION_IDS:
        record = json.loads((HERE / "per_sample" / f"{sample_id}.json").read_text())
        if record.get("status") != "SEALED_CANDIDATE":
            rows.append({"sample_id": sample_id, "sealed": False, "status": record.get("status", "NOT_RUN")})
            continue
        spec = next(row for row in manifest_samples() if row["sample_id"] == sample_id)
        clean = load_images(spec)
        aux = Image.open(record["selected_png"]).convert("RGB")
        linf = uint8_linf(clean[spec["aux_idx"]], aux)
        if linf > EPS + TOLERANCE:
            rows.append({"sample_id": sample_id, "sealed": False, "status": "LINF_REJECTED", "linf": linf})
            continue
        dest = HERE / "sealed_png" / sample_id
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
            "checkpoint": record["chosen"]["step"],
            "family": "BasePreservingAdaptivePool",
            "seed": GLOBAL_SEED,
            "nonzero": True,
            "linf": linf,
            "png_sha256": sha_file(dest / f"page_{spec['aux_idx']}.png"),
            "clean_text": record["clean_text"],
            "clean_tokens": record["clean_tokens"],
            "aux_idx": spec["aux_idx"],
            "evidence_idx": spec["evidence_idx"],
        })
    manifest = {"samples": rows, "sealed_count": sum(bool(row["sealed"]) for row in rows), "fastv_access_count": 0, "fastv_modules": fastv_modules()}
    (HERE / "seal_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sealed": manifest["sealed_count"]}), flush=True)
    assert_clean("seal")


if __name__ == "__main__":
    main()
