"""Independent FastV evaluation. Start this only after the trainer has exited."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v9b_rank_signal_oblivious_admission"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from common import EPS, TOLERANCE, answers_of, anls_score, manifest_samples, normalize, sha_file, uint8_linf  # noqa: E402
from run_causal_v2 import load_model, run_fastv, run_plain  # noqa: E402


def evaluate_once(model, processor, images, spec) -> dict:
    token_id = int(model.config.image_token_index)
    full, _, _, _ = run_plain(model, processor, images, spec["question"], token_id, False, spec["evidence_idx"])
    fast = {}
    for ratio in (0.25, 0.35):
        pred, _, kept, _ = run_fastv(
            model, processor, images, spec["question"], token_id, ratio, spec["evidence_idx"], spec["aux_idx"]
        )
        fast[str(ratio)] = {"text": pred, "anls": anls_score(pred, answers_of(spec)), "k": len(kept)}
    return {"full": full, "fast": fast}


def pollution(clean: str, attacked: str) -> bool:
    left, right = normalize(clean), normalize(attacked)
    if left == right:
        return False
    return left in right or right in left


def main() -> None:
    manifest = json.loads((HERE / "seal_manifest.json").read_text())
    model, processor = load_model()
    rows = []
    for item in manifest["samples"]:
        if not item.get("sealed"):
            rows.append({"sample_id": item["sample_id"], "evaluated": False, "strict": False, "reason": item.get("reason")})
            continue
        spec = next(row for row in manifest_samples() if row["sample_id"] == item["sample_id"])
        folder = HERE / "sealed_candidates" / item["sample_id"]
        images = [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(spec["M"])]
        first = evaluate_once(model, processor, images, spec)
        reloaded = [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(spec["M"])]
        second = evaluate_once(model, processor, reloaded, spec)
        aux = images[spec["aux_idx"]]
        clean_aux = Image.open(HERE / "baseline_pages" / item["sample_id"] / f"page_{spec['aux_idx']}.png").convert("RGB")
        linf = uint8_linf(clean_aux, aux)
        nonzero = sha_file(HERE / "baseline_pages" / item["sample_id"] / f"page_{spec['aux_idx']}.png") != sha_file(folder / f"page_{spec['aux_idx']}.png")
        full_exact = first["full"] == item["clean_text"]
        replay = first == second
        valid = bool(first["full"].strip()) and all(first["fast"][key]["text"].strip() for key in ("0.25", "0.35"))
        anls_025 = first["fast"]["0.25"]["anls"]
        anls_035 = first["fast"]["0.35"]["anls"]
        base = valid and nonzero and linf <= EPS + TOLERANCE and full_exact and replay
        row = {
            "sample_id": item["sample_id"],
            "evaluated": True,
            "full_text": first["full"],
            "full_exact": full_exact,
            "replay": replay,
            "valid": valid,
            "nonzero": nonzero,
            "linf": linf,
            "anls_r025": anls_025,
            "anls_r035": anls_035,
            "fastv_text": {key: first["fast"][key]["text"] for key in ("0.25", "0.35")},
            "pollution": pollution(item["clean_text"], first["fast"]["0.25"]["text"]) or pollution(item["clean_text"], first["fast"]["0.35"]["text"]),
            "strict_r025": bool(base and anls_025 < 0.5),
            "strict_r035": bool(base and anls_035 < 0.5),
            "strict": bool(base and anls_025 < 0.5 and anls_035 < 0.5),
        }
        rows.append(row)
        print(json.dumps({"sample": item["sample_id"], "strict": row["strict"], "full": first["full"]}), flush=True)
    strict = sum(bool(row.get("strict")) for row in rows)
    full_exact = sum(bool(row.get("full_exact")) for row in rows)
    replay = sum(bool(row.get("replay")) for row in rows)
    if strict >= 2 and full_exact == 5 and replay == 5:
        decision = "GO_RANK_OBLIVIOUS_GRAYBOX"
    elif strict == 1:
        decision = "WEAK_SIGNAL_ONE_OF_FIVE"
    else:
        decision = "NO_GO_ZERO_TRANSFER"
    payload = {
        "rows": rows,
        "strict": f"{strict}/5",
        "strict_r025": f"{sum(bool(row.get('strict_r025')) for row in rows)}/5",
        "strict_r035": f"{sum(bool(row.get('strict_r035')) for row in rows)}/5",
        "full_exact": f"{full_exact}/5",
        "replay": f"{replay}/5",
        "invalid": sum(not bool(row.get("valid", False)) for row in rows),
        "pollution": sum(bool(row.get("pollution")) for row in rows),
        "decision": decision,
    }
    (HERE / "target_eval").mkdir(exist_ok=True)
    (HERE / "target_eval" / "fastv_transfer.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"decision": decision, "strict": payload["strict"]}), flush=True)


if __name__ == "__main__":
    main()
