"""Independent FastV evaluation for one frozen holdout shard."""

from __future__ import annotations

import argparse
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
from week2_attack_v9c_multiscale_rank_oblivious.common import output_valid  # noqa: E402

RATIOS = (0.15, 0.25, 0.35, 0.50)


def keep_hash(kept) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in kept).encode()).hexdigest()


def load_branch(sample: str, branch: str, count: int) -> list[Image.Image]:
    folder = HERE / "holdout_transfer" / "prepared" / branch / sample
    return [Image.open(folder / f"page_{i}.png").convert("RGB") for i in range(count)]


def measure(model, processor, images, row: dict, token_id: int) -> dict:
    full, _, _, _ = run_plain(model, processor, images, row["question"], token_id, False, row["evidence_idx"])
    fastv = {}
    for ratio in RATIOS:
        pred, _, kept, _ = run_fastv(model, processor, images, row["question"], token_id, ratio, row["evidence_idx"], row["aux_idx"])
        fastv[f"{ratio:.2f}"] = {
            "text": pred,
            "valid": output_valid(pred),
            "anls": float(anls_score(pred, row["answers"])),
            "k": len(kept),
            "keep_hash": keep_hash(kept),
        }
    return {"full_text": full, "full_valid": output_valid(full), "fastv": fastv}


def same(a: dict, b: dict) -> bool:
    return a["full_text"] == b["full_text"] and all(
        a["fastv"][key]["text"] == b["fastv"][key]["text"]
        and a["fastv"][key]["keep_hash"] == b["fastv"][key]["keep_hash"]
        for key in a["fastv"]
    )


def evaluate(samples: list[str], output: Path) -> None:
    manifest = json.loads((HERE / "holdout_transfer" / "prepared_manifest.json").read_text())
    model, processor = __import__("gate_experiment.run_causal_v2", fromlist=["load_model"]).load_model()
    token_id = int(model.config.image_token_index)
    report = {
        "status": "RUNNING",
        "protocol": "independent disk-reloaded clean/attack/matched-noise holdout FastV replay",
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "samples": {},
    }
    for sample in samples:
        row = manifest["samples"].get(sample, {})
        if row.get("status") != "OK":
            report["samples"][sample] = {"status": row.get("status", "NOT_RUN")}
            continue
        measured, replay = {}, {}
        for branch in ("clean", "attack", "noise"):
            images = load_branch(sample, branch, row["M"])
            measured[branch] = measure(model, processor, images, row, token_id)
            replay[branch] = same(measured[branch], measure(model, processor, load_branch(sample, branch, row["M"]), row, token_id))
        clean, attack, noise = measured["clean"], measured["attack"], measured["noise"]
        attack_nonzero = row["page_hashes"]["attack"] != row["page_hashes"]["clean"]
        noise_nonzero = row["page_hashes"]["noise"] != row["page_hashes"]["clean"]
        ratios = {}
        for key in clean["fastv"]:
            c, a, n = clean["fastv"][key], attack["fastv"][key], noise["fastv"][key]
            ratios[key] = {
                "clean": c, "attack": a, "noise": n,
                "attack_strict": bool(attack["full_valid"] and attack["full_text"] == row["clean_text"] and attack_nonzero and a["valid"] and a["anls"] < 0.5),
                "noise_strict": bool(noise["full_valid"] and noise["full_text"] == row["clean_text"] and noise_nonzero and n["valid"] and n["anls"] < 0.5),
                "induced": bool(c["anls"] >= 0.5 and a["anls"] < 0.5),
                "beats_noise": bool(a["anls"] < n["anls"]),
            }
        report["samples"][sample] = {
            "status": "OK", "clean_text": row["clean_text"],
            "full": {branch: measured[branch]["full_text"] for branch in measured},
            "replay": replay, "ratios": ratios,
            "attack_nonzero": attack_nonzero, "noise_nonzero": noise_nonzero,
        }
    report["status"] = "TRANSFER_EVALUATED"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "samples": len(report["samples"])}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="comma-separated frozen holdout ids")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frozen = json.loads((HERE / "holdout_ids.json").read_text())["ids"]
    samples = [value for value in args.samples.split(",") if value]
    if any(value not in frozen for value in samples):
        raise RuntimeError("evaluation samples must be in the frozen holdout list")
    evaluate(samples, Path(args.out))


if __name__ == "__main__":
    main()
