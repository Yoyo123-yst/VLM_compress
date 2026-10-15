"""Five-attempt real-PNG probe for a semantically valid target with unknown local capacity.

This is an isolated diagnostic. It must not be counted as a v30 method result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v25_full_answer_competition_pilot import run_iterative_compare as runner  # noqa: E402
from week2_attack_v29_batch_compare.load_samples import load_item  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--max-accepted", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--out-tag", default="unknown_capacity_probe")
    args = parser.parse_args()

    admission = json.loads((OUT / "clean_admission" / "5786.json").read_text())
    if not admission.get("clean_gate_eligible", False):
        raise RuntimeError("CLEAN_ADMISSION_NOT_ELIGIBLE")
    target_path = OUT / "unknown_capacity_probe_targets.json"
    target = json.loads(target_path.read_text())["samples"]["5786"]
    if target["estimated_feasible_at_eps"] is not None:
        raise RuntimeError("PROBE_TARGET_MUST_HAVE_UNKNOWN_ESTIMATE")
    # Use the v30 dataset loader, not the historical development loader.
    runner.v7d.load_dev = load_item
    result = runner.run_one(
        "5786",
        "complete_target",
        str(args.device),
        OUT / args.out_tag,
        max_accepted=args.max_accepted,
        max_attempts=args.max_attempts,
        targets_path=target_path,
    )
    result["probe_status"] = "UNKNOWN_CAPACITY_REAL_PROBE_COMPLETED"
    result["semantic_target_status"] = target["semantic_status"]
    result["estimated_feasible_at_eps"] = None
    result["probe_contract"] = {
        "sample": "5786",
        "wrong_target": "670",
        "max_accepted": args.max_accepted,
        "max_attempts": args.max_attempts,
        "not_a_method_result": True,
        "backward_budget_nominal": args.max_attempts * 2,
    }
    out = OUT / args.out_tag / "complete_target" / "5786" / "probe_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["probe_status"], "accepted": result["accepted"], "attempts": result["attempts"], "out": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
