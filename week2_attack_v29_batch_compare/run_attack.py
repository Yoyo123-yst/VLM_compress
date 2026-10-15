"""One frozen v29 attack trajectory; every attempt spends its planned budget."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v25_full_answer_competition_pilot import run_iterative_compare as runner  # noqa: E402
from week2_attack_v29_batch_compare.load_samples import load_item  # noqa: E402

BUDGETS = {"native_fdp": (20, 2), "complete_target": (13, 3), "complete_front": (10, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--method", choices=tuple(BUDGETS), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--targets", required=False)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    budget, backward_per_attempt = BUDGETS[args.method]
    clean = json.loads((OUT / "clean_screen" / f"{args.sample}.json").read_text())
    if not clean["clean_gate_eligible"]:
        raise SystemExit(f"CLEAN_GATE_FAILED: {args.sample}; attack prohibited")
    targets_path = args.targets or str(OUT / "frozen_targets.json")
    # Attempt cap equals accepted cap: loop terminates only after all planned
    # attempts, unless it has completed that exact number of attempts. A reject
    # therefore spends compute and is never silently replaced by another step.
    runner.v7d.load_dev = load_item
    payload = runner.run_one(
        args.sample, args.method, args.device, OUT / "attacks",
        max_accepted=budget, max_attempts=budget, targets_path=targets_path,
    )
    expected_bwd = budget * backward_per_attempt
    actual_bwd = sum(int(row["counter_delta"][key]) for row in payload["updates"]
                     for key in ("native_attack_backwards", "complete_sequence_backwards", "front_backwards", "full_fdp_backwards"))
    if payload["attempts"] != budget or actual_bwd != expected_bwd:
        raise RuntimeError(f"BUDGET_ACCOUNTING_MISMATCH attempts={payload['attempts']}/{budget} backward={actual_bwd}/{expected_bwd}")
    payload["fixed_attempt_budget"] = budget
    payload["backward_budget_planned"] = expected_bwd
    payload["backward_budget_actual"] = actual_bwd
    payload["rejections_consume_budget"] = True
    payload["selector_rule"] = (
        f"last accepted PNG after exactly {budget} attempts; rejected updates consume budget; "
        "no intermediate-best or oracle selection"
    )
    path = OUT / "attacks" / args.method / args.sample / "result.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": args.sample, "method": args.method, "attempts": payload["attempts"],
                      "accepted": payload["accepted"], "backwards": actual_bwd,
                      "seconds": payload["wall_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
