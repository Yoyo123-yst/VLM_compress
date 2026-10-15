"""Run the frozen clean-only candidate generator on admitted v28 IDs."""
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
from week2_attack_v28_complete_front_holdout.load_samples import load_item  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    clean = json.loads((OUT / "clean_screen" / f"{args.sample}.json").read_text())
    if not clean["clean_gate_eligible"]:
        raise SystemExit(f"CLEAN_GATE_FAILED: {args.sample}; no target search will run")
    torch.cuda.set_device(args.device)
    # Narrow loader adapter: no model methods, generation, or hooks are wrapped.
    candidates.v7d.load_dev = load_item
    started = time.perf_counter()
    row = candidates.build(args.sample, str(args.device))
    row["candidate_generation_wall_seconds"] = round(time.perf_counter() - started, 3)
    row["candidate_generation_compute_accounting"] = (
        "wall time recorded; attack-phase forward/backward/generate counters do not include this clean-only stage"
    )
    dest = OUT / "candidate_manifest" / f"{args.sample}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": args.sample, "status": row["status"],
                      "candidate_count": len(row.get("candidates", [])), "out": str(dest)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
