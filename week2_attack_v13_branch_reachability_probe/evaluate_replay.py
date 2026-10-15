"""Fresh-process replay for a sealed construction-probe PNG.

The trainer's checkpoint score is insufficient by itself: this evaluator loads
the model again, reads the PNG from disk, and evaluates the Full/FastV endpoint
twice.  It does not choose a checkpoint or alter pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v13_branch_reachability_probe"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

import torch
from PIL import Image

import run_branch_probe as probe  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True)
    # Replay does not implement a selector and therefore must not hard-code
    # the construction variants.  In particular, the state-retention ablation
    # seals the already selected AWARE branch under a separate variant name.
    p.add_argument("--variant", required=True)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--tag", default="construct_v1r1")
    args = p.parse_args()
    probe.configure_determinism()
    model, processor = base.load_model_frozen()
    item = v7d.load_dev(args.sample)
    state = v7d.prepare_item(model, processor, item)
    clean_fast = probe.fast_clean_reference(model, processor, item, state)
    trajectory = json.loads((HERE / "runs" / args.tag / args.sample / args.variant / "trajectory.json").read_text())
    choice = trajectory["choice"]
    path = HERE / "runs" / args.tag / args.sample / args.variant / f"step_{args.step:03d}.png"
    if not path.is_file():
        raise SystemExit(f"missing sealed PNG: {path}")
    image = Image.open(path).convert("RGB")
    first = probe.score_checkpoint(model, processor, item, state, image, clean_fast, choice)
    second = probe.score_checkpoint(model, processor, item, state, image, clean_fast, choice)
    fast_replay = (
        first["fast"]["tokens"] == second["fast"]["tokens"]
        and first["fast"]["text"] == second["fast"]["text"]
        and first["fast"]["keep_hash"] == second["fast"]["keep_hash"]
        and first["fast"]["k"] == second["fast"]["k"]
    )
    strict = bool(probe.legal_endpoint(first, clean_fast) and fast_replay)
    result = {
        "status": "DONE",
        "scope": "fresh-process replay; endpoint verification only, no selector",
        "sample_id": args.sample,
        "variant": args.variant,
        "step": args.step,
        "png": str(path.relative_to(ROOT)),
        "png_sha256": probe.sha_file(path),
        "first": first,
        "second": second,
        "fast_replay": fast_replay,
        "strict_endpoint_replayed": strict,
    }
    out = path.with_name(f"step_{args.step:03d}.fresh_replay.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "DONE", "sample": args.sample, "variant": args.variant, "step": args.step, "strict_endpoint_replayed": strict}), flush=True)


if __name__ == "__main__":
    main()
