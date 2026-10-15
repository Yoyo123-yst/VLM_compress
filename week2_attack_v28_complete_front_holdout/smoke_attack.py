"""One-attempt branch smoke used only to validate the frozen v28 adapter."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v25_full_answer_competition_pilot import run_iterative_compare as runner  # noqa: E402
from week2_attack_v28_complete_front_holdout.load_samples import load_item  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="4742")
    parser.add_argument("--method", choices=("native_fdp", "complete_target", "complete_front"), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--replicate", default="a")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    runner.v7d.load_dev = load_item
    target = OUT / "frozen_targets.json"
    result = runner.run_one(args.sample, args.method, args.device,
                            OUT / "smoke" / args.replicate,
                            max_accepted=1, max_attempts=1,
                            targets_path=str(target))
    key = {"native_fdp": "native_attack_backwards",
           "complete_target": "complete_sequence_backwards",
           "complete_front": "front_backwards"}[args.method]
    expected = {"native_fdp": 1, "complete_target": 2, "complete_front": 1}[args.method]
    delta = result["updates"][0]["counter_delta"]
    actual = sum(delta[k] for k in ("native_attack_backwards", "complete_sequence_backwards", "front_backwards", "full_fdp_backwards"))
    expected_total = {"native_fdp": 2, "complete_target": 3, "complete_front": 4}[args.method]
    if actual != expected_total or delta[key] != expected:
        raise RuntimeError(f"SMOKE_BACKWARD_MISMATCH method={args.method} actual={actual} expected={expected_total}")
    print(json.dumps({"sample": args.sample, "method": args.method, "replicate": args.replicate,
                      "accepted": result["accepted"], "attempts": result["attempts"],
                      "backwards": actual, "grad_hash": result["updates"][0]["grad_hash"],
                      "delta_hash": result["updates"][0]["delta_candidate_hash"],
                      "png_sha256": result["updates"][0]["png_sha256"],
                      "guard": result["updates"][0]["guard"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
