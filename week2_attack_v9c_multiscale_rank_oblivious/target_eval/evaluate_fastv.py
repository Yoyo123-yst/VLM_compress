"""Isolated FastV evaluation. Reads only sealed PNGs and the seal manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v9c_multiscale_rank_oblivious"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(HERE)]

from common import EPS, TOLERANCE, answers_of, anls_score, manifest_samples, output_valid, sha_file, uint8_linf  # noqa: E402
from run_causal_v2 import load_model, run_fastv, run_plain  # noqa: E402

RATIOS = (0.15, 0.25, 0.35, 0.50)


def run_once(model, processor, images, spec) -> dict:
    token_id = int(model.config.image_token_index)
    full, _, _, _ = run_plain(model, processor, images, spec["question"], token_id, False, spec["evidence_idx"])
    fast = {}
    for ratio in RATIOS:
        pred, _, kept, _ = run_fastv(
            model, processor, images, spec["question"], token_id, ratio, spec["evidence_idx"], spec["aux_idx"]
        )
        fast[f"{ratio:.2f}"] = {"text": pred, "anls": anls_score(pred, answers_of(spec)), "valid": output_valid(pred), "k": len(kept)}
    return {"full": full, "fast": fast}


def main() -> None:
    manifest = json.loads((HERE / "seal_manifest.json").read_text())
    model, processor = load_model()
    rows = []
    for item in manifest["samples"]:
        if not item.get("sealed"):
            rows.append({
                "sample_id": item["sample_id"],
                "evaluated": False,
                "status": item.get("status", "NOT_RUN"),
                "strict": {f"{ratio:.2f}": False for ratio in RATIOS},
            })
            continue
        spec = next(row for row in manifest_samples() if row["sample_id"] == item["sample_id"])
        folder = HERE / "sealed_png" / item["sample_id"]
        images = [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(spec["M"])]
        first = run_once(model, processor, images, spec)
        reloaded = [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(spec["M"])]
        second = run_once(model, processor, reloaded, spec)
        clean_aux = Image.open(HERE / "baseline_pages" / item["sample_id"] / f"page_{spec['aux_idx']}.png").convert("RGB")
        linf = uint8_linf(clean_aux, images[spec["aux_idx"]])
        nonzero = sha_file(HERE / "baseline_pages" / item["sample_id"] / f"page_{spec['aux_idx']}.png") != item["png_sha256"]
        full_exact = second["full"] == item["clean_text"] and first["full"] == second["full"]
        full_valid = output_valid(second["full"])
        replay = first["full"] == second["full"] and all(first["fast"][key]["text"] == second["fast"][key]["text"] for key in first["fast"])
        strict = {}
        for ratio in RATIOS:
            key = f"{ratio:.2f}"
            fast_ok = second["fast"][key]["valid"] and second["fast"][key]["anls"] < 0.5 and first["fast"][key]["text"] == second["fast"][key]["text"]
            strict[key] = bool(full_valid and nonzero and linf <= EPS + TOLERANCE and full_exact and replay and fast_ok)
        rows.append({
            "sample_id": item["sample_id"],
            "evaluated": True,
            "clean_full_text": item["clean_text"],
            "clean_full_tokens": item["clean_tokens"],
            "attacked_full_text": second["full"],
            "full_exact": full_exact,
            "valid": full_valid and all(second["fast"][f"{ratio:.2f}"]["valid"] for ratio in RATIOS),
            "replay": replay,
            "nonzero": nonzero,
            "linf": linf,
            "png_sha256": item["png_sha256"],
            "fastv": second["fast"],
            "strict": strict,
            "target_access_count": 8,
        })
        print(json.dumps({"sample": item["sample_id"], "full": second["full"], "strict": strict}), flush=True)
    summary = {}
    for ratio in RATIOS:
        key = f"{ratio:.2f}"
        summary[key] = f"{sum(bool(row.get('strict', {}).get(key)) for row in rows)}/5"
    payload = {
        "status": "GRAYBOX_TRANSFER_EVALUATED",
        "rows": rows,
        "strict": summary,
        "full_exact": f"{sum(bool(row.get('full_exact')) for row in rows)}/5",
        "valid": f"{sum(bool(row.get('valid')) for row in rows)}/5",
        "replay": f"{sum(bool(row.get('replay')) for row in rows)}/5",
        "linf_ok": f"{sum(bool(row.get('evaluated')) and row.get('linf', 1) <= EPS + TOLERANCE for row in rows)}/5",
        "evaluated": f"{sum(bool(row.get('evaluated')) for row in rows)}/5",
    }
    (HERE / "target_eval").mkdir(exist_ok=True)
    (HERE / "target_eval" / "fastv_transfer.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"strict": summary, "full_exact": payload["full_exact"]}), flush=True)


if __name__ == "__main__":
    main()
