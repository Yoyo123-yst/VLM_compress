"""Run the frozen clean-only complete-target generator for v30."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v25_full_answer_competition_pilot import generate_clean_candidates as candidates  # noqa: E402
from week2_attack_v29_batch_compare.load_samples import load_item  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    admission = json.loads((OUT / "admission_decision.json").read_text())
    if admission["samples"].get(args.sample, {}).get("clean_status") != "VALID_CORRECT":
        raise SystemExit(f"CLEAN_SEMANTIC_GATE_FAILED: {args.sample}")
    torch.cuda.set_device(args.device)
    candidates.v7d.load_dev = load_item
    started = time.perf_counter()
    row = candidates.build(args.sample, str(args.device))
    row["candidate_generation_wall_seconds"] = round(time.perf_counter() - started, 3)
    row["candidate_generation_protocol"] = "v29 clean-only cap: at most 24 raw branches and 6 complete continuations"
    dest = OUT / "candidate_manifest" / f"{args.sample}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": args.sample, "status": row["status"], "candidate_count": len(row.get("candidates", []))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
