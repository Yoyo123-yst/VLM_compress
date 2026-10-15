"""Independent clean/attack/noise FastV replay for frozen v11c PNGs."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v11c_deterministic_native_fdp"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from gate_experiment.run_causal_v2 import run_fastv, run_plain  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d  # noqa: E402
from week2_attack_v9c_multiscale_rank_oblivious.common import output_valid  # noqa: E402

RATIOS = (0.15, 0.25, 0.35, 0.50)
SAMPLES = ("1025", "2008", "47450", "60700", "7006", "39204")


def keep_hash(kept) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in kept).encode()).hexdigest()


def load_branch(sample: str, branch: str, count: int) -> list[Image.Image]:
    folder = HERE / "transfer" / "prepared" / branch / sample
    return [Image.open(folder / f"page_{i}.png").convert("RGB") for i in range(count)]


def measure(model, processor, images, row: dict, image_token_id: int) -> dict:
    full, _, _, _ = run_plain(
        model, processor, images, row["question"], image_token_id, False, row["evidence_idx"]
    )
    fast = {}
    for ratio in RATIOS:
        pred, _, kept, _ = run_fastv(
            model, processor, images, row["question"], image_token_id,
            ratio, row["evidence_idx"], row["aux_idx"]
        )
        fast[f"{ratio:.2f}"] = {
            "text": pred,
            "valid": output_valid(pred),
            "anls": float(anls_score(pred, row["answers"])),
            "k": len(kept),
            "keep_hash": keep_hash(kept),
        }
    return {"full_text": full, "full_valid": output_valid(full), "fastv": fast}


def same_endpoint(a: dict, b: dict) -> bool:
    if a["full_text"] != b["full_text"]:
        return False
    return all(
        a["fastv"][key]["text"] == b["fastv"][key]["text"]
        and a["fastv"][key]["keep_hash"] == b["fastv"][key]["keep_hash"]
        for key in a["fastv"]
    )


def main() -> None:
    manifest = json.loads((HERE / "transfer" / "prepared_manifest.json").read_text())
    model, processor = __import__("gate_experiment.run_causal_v2", fromlist=["load_model"]).load_model()
    image_token_id = int(model.config.image_token_index)
    report = {
        "status": "RUNNING",
        "protocol": "independent disk-reloaded clean/attack/matched-noise FastV replay",
        "attn_implementation": os.environ.get("VATTACK_ATTN_IMPLEMENTATION", "sdpa"),
        "samples": {},
    }
    for sample in SAMPLES:
        row = manifest["samples"][sample]
        if row.get("status") != "OK":
            report["samples"][sample] = {"status": row.get("status", "NOT_RUN")}
            continue
        measured = {}
        replay = {}
        for branch in ("clean", "attack", "noise"):
            images = load_branch(sample, branch, row["M"])
            measured[branch] = measure(model, processor, images, row, image_token_id)
            # Read the same PNGs again and verify endpoint replay.
            replay[branch] = same_endpoint(
                measured[branch], measure(model, processor, load_branch(sample, branch, row["M"]), row, image_token_id)
            )
        clean = measured["clean"]
        attack = measured["attack"]
        noise = measured["noise"]
        attack_nonzero = row["page_hashes"]["attack"] != row["page_hashes"]["clean"]
        noise_nonzero = row["page_hashes"]["noise"] != row["page_hashes"]["clean"]
        ratios = {}
        for key in clean["fastv"]:
            c = clean["fastv"][key]
            a = attack["fastv"][key]
            n = noise["fastv"][key]
            ratios[key] = {
                "clean": c,
                "attack": a,
                "noise": n,
                "attack_strict": bool(attack["full_valid"] and attack["full_text"] == row["clean_text"] and attack_nonzero and a["valid"] and a["anls"] < 0.5),
                "noise_strict": bool(noise["full_valid"] and noise["full_text"] == row["clean_text"] and noise_nonzero and n["valid"] and n["anls"] < 0.5),
                "induced": bool(c["anls"] >= 0.5 and a["anls"] < 0.5),
                "beats_noise": bool(a["anls"] < n["anls"]),
            }
        report["samples"][sample] = {
            "status": "OK",
            "clean_text": row["clean_text"],
            "full": {branch: measured[branch]["full_text"] for branch in measured},
            "replay": replay,
            "ratios": ratios,
            "attack_nonzero": attack_nonzero,
            "noise_nonzero": noise_nonzero,
        }
    rows = [value for value in report["samples"].values() if value.get("status") == "OK"]
    report["status"] = "TRANSFER_EVALUATED"
    report["summary"] = {
        "samples_measured": len(rows),
        "all_replay_pass": all(all(item["replay"].values()) for item in rows),
        "induced_cells": sum(int(cell["induced"]) for item in rows for cell in item["ratios"].values()),
        "attack_strict_cells": sum(int(cell["attack_strict"]) for item in rows for cell in item["ratios"].values()),
        "noise_strict_cells": sum(int(cell["noise_strict"]) for item in rows for cell in item["ratios"].values()),
        "attack_beats_noise_cells": sum(int(cell["beats_noise"]) for item in rows for cell in item["ratios"].values()),
    }
    out = HERE / "transfer" / "transfer_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
