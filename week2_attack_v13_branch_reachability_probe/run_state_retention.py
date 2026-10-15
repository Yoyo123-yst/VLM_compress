"""One-sample causal ablation for the AWARE branch-recovery failure.

This file intentionally does *not* discover candidates or choose a branch.
It loads the frozen AWARE choice from construct_v1r1/60700/screen.json and
keeps all attack settings fixed.  The sole intervention is temporal:

1. perform the usual AWARE update with the usual two-call exact Full guard;
2. at the first accepted state whose *fixed selected-pair* margin is <= tau,
   snapshot delta and Adam state and stop future parameter updates;
3. continue the step counter through step 25 and evaluate the sealed PNG.

The natural FastV answer/ANLS is never consulted to decide whether to freeze.
Thus a retained endpoint would specifically show that the prior step-25
recovery was caused by updates after the selected branch had crossed, not by
candidate re-selection or an oracle checkpoint choice.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v13_branch_reachability_probe"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

import run_branch_probe as probe  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project  # noqa: E402


SOURCE_TAG = "construct_v1r1"
SAMPLE = "60700"
VARIANT = "AWARE_FREEZE"
DEFAULT_TAG = "state_retention_v1"


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def optimizer_hashes(opt: torch.optim.Optimizer, delta: torch.Tensor) -> dict:
    state = opt.state.get(delta, {})
    result = {"step": int(state.get("step", 0)) if not torch.is_tensor(state.get("step", 0)) else int(state["step"].item())}
    for key in ("exp_avg", "exp_avg_sq"):
        result[key] = tensor_hash(state[key]) if key in state else None
    return result


def exact_full(guard: dict) -> bool:
    return bool(guard["full_text_exact"] and guard["full_token_exact"] and guard["full_replay"])


def fixed_pair_state(model, state: dict, item: dict, delta: torch.Tensor, reference: list[int], choice: dict) -> dict:
    """Current selected-pair value under the current FastV keep set.

    The leaf is only to keep the checkpointed FastV forward path identical to
    training; no backward pass is performed here.  It is deliberately not a
    natural-answer test and cannot influence the freeze decision beyond the
    pre-registered fixed pair margin.
    """
    leaf = (state["page"] + delta.detach()).clone().requires_grad_(True)
    rows, generated, kept = probe.fast_rows_for_reference(model, state, item, leaf, reference)
    t, y, c = choice["position"], choice["target_id"], choice["candidate_id"]
    gap = rows[t][y] - rows[t][c]
    result = {
        "fixed_pair_gap": float(gap.detach()),
        "fixed_pair_threshold_met": bool(float(gap.detach()) <= probe.TAU),
        "current_fast_tokens_for_clean_reference": generated,
        "current_keep_hash": keep_hash(kept),
        "current_k": len(kept),
    }
    del rows, generated, kept, leaf
    torch.cuda.empty_cache()
    return result


def validate_frozen_inputs(model, processor, item: dict, state: dict, source_screen: dict) -> dict:
    """Fail closed if this fresh runtime no longer reproduces the frozen clean state."""
    full = probe.clean_full(model, processor, item, state)
    fast = probe.fast_clean_reference(model, processor, item, state)
    expected_full = source_screen["full_clean"]
    expected_fast = source_screen["fast_clean"]
    full_match = (
        full["tokens_a"] == expected_full["tokens_a"]
        and full["tokens_b"] == expected_full["tokens_b"]
        and full["text_a"] == expected_full["text_a"]
        and full["text_b"] == expected_full["text_b"]
    )
    fast_match = (
        fast["tokens"] == expected_fast["tokens"]
        and fast["text"] == expected_fast["text"]
        and fast["keep_hash"] == expected_fast["keep_hash"]
        and fast["k"] == expected_fast["k"]
        and fast["second_keep_hash"] == expected_fast["second_keep_hash"]
        and fast["second_k"] == expected_fast["second_k"]
    )
    return {"current_full": full, "current_fast": fast, "full_matches_source": full_match, "fast_matches_source": fast_match, "ok": bool(full_match and fast_match)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--steps", type=int, default=probe.STEPS)
    args = parser.parse_args()
    if args.steps != probe.STEPS:
        raise SystemExit(f"This registered causal check is fixed at {probe.STEPS} steps, not {args.steps}.")

    probe.configure_determinism()
    source_root = HERE / "runs" / SOURCE_TAG / SAMPLE
    source_screen_path = source_root / "screen.json"
    if not source_screen_path.is_file():
        raise SystemExit(f"missing frozen source screen: {source_screen_path}")
    source_screen = json.loads(source_screen_path.read_text())
    choice = source_screen.get("aware_choice")
    if not source_screen.get("branch_admitted") or not choice:
        raise SystemExit("frozen source has no admitted AWARE branch")

    out = HERE / "runs" / args.tag / SAMPLE / VARIANT
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "CAUSAL_ABLATION_REGISTERED",
        "source_tag": SOURCE_TAG,
        "source_screen": str(source_screen_path.relative_to(ROOT)),
        "source_screen_sha256": probe.sha_file(source_screen_path),
        "sample_id": SAMPLE,
        "ratio": probe.RATIO,
        "steps": probe.STEPS,
        "epsilon_uint8": 8,
        "choice": choice,
        "freeze_rule": "first accepted post-update state with fixed selected-pair gap <= 0.0 and two-call exact Full guard",
        "freeze_rule_uses_natural_fastv_output": False,
        "post_freeze_rule": "do not update pixels or Adam; continue recording state steps through 25",
        "guard": "all pre-freeze updates use the existing two-call real Full exact guard; rejected updates restore delta and Adam state",
        "endpoint": "unforced saved/reloaded PNG; Full text/token/replay exact plus FastV natural output recorded separately",
        "stop_rule": "if no legal threshold crossing occurs, record NO_FREEZE_EVENT; if step-25 endpoint is not retained after fresh replay, stop this innovation branch",
    }
    (out / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")

    model, processor = base.load_model_frozen()
    base.install_counters(model)
    item = v7d.load_dev(SAMPLE)
    state = v7d.prepare_item(model, processor, item)
    baseline = validate_frozen_inputs(model, processor, item, state, source_screen)
    if not baseline["ok"]:
        payload = {"status": "BLOCKED_CLEAN_REPLAY_DIVERGED", "protocol": protocol, "baseline": baseline}
        (out / "trajectory.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"status": payload["status"]}), flush=True)
        return

    torch.manual_seed(probe.SEED)
    torch.cuda.manual_seed_all(probe.SEED)
    np.random.seed(probe.SEED % (2**32))
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=probe.ADAM_LR)
    reference = baseline["current_fast"]["tokens"]
    base.reset_counters()
    started = time.perf_counter()
    curve, updates = [], []
    frozen = False
    freeze_event = None
    checkpoints = set(probe.CHECKPOINTS)

    for state_step in range(probe.STEPS + 1):
        if state_step in checkpoints:
            image = page_uint8(state["page"], delta)
            path = out / f"step_{state_step:03d}.png"
            image.save(path)
            row = probe.score_checkpoint(model, processor, item, state, Image.open(path).convert("RGB"), baseline["current_fast"], choice)
            row.update({
                "step": state_step,
                "png_sha256": probe.sha_file(path),
                "delta_sha256": tensor_hash(delta),
                "frozen": frozen,
            })
            row["strict_success"] = probe.legal_endpoint(row, baseline["current_fast"])
            curve.append(row)
        if state_step == probe.STEPS:
            break

        if frozen:
            updates.append({
                "step": state_step,
                "update_skipped_after_freeze": True,
                "delta_sha256": tensor_hash(delta),
                "optimizer": optimizer_hashes(opt, delta),
            })
            continue

        direction, meta = probe.branch_loss_and_grad(model, state, item, delta, reference, choice)
        before_delta = delta.detach().clone()
        before_opt = copy.deepcopy(opt.state_dict())
        before_hash = tensor_hash(before_delta)
        delta.grad = direction
        opt.step()
        project(delta, state["page"], probe.EPS)
        delta.grad = None
        guarded = base.full_exact_check(model, processor, item, state, page_uint8(state["page"], delta))
        rolled_back = not exact_full(guarded)
        if rolled_back:
            delta.data.copy_(before_delta)
            opt.load_state_dict(before_opt)
            post_pair = None
        else:
            post_pair = fixed_pair_state(model, state, item, delta, reference, choice)
            if post_pair["fixed_pair_threshold_met"]:
                frozen = True
                # Seal the exact causal intervention state even when it falls
                # between the pre-registered 5-step checkpoints.  It is not a
                # selected best checkpoint: its index is mechanically fixed by
                # the first legal pair-margin crossing.
                freeze_path = out / f"step_{state_step + 1:03d}.png"
                page_uint8(state["page"], delta).save(freeze_path)
                freeze_event = {
                    "frozen_at_state_step": state_step + 1,
                    "threshold": probe.TAU,
                    "post_update_full": guarded,
                    "post_update_pair": post_pair,
                    "frozen_delta_sha256": tensor_hash(delta),
                    "frozen_optimizer": optimizer_hashes(opt, delta),
                    "frozen_png": str(freeze_path.relative_to(ROOT)),
                    "frozen_png_sha256": probe.sha_file(freeze_path),
                }
        updates.append({
            "step": state_step,
            **meta,
            "pre_update_delta_sha256": before_hash,
            "post_update_full": guarded,
            "rolled_back": rolled_back,
            "post_update_pair": post_pair,
            "frozen_after_update": bool(freeze_event and freeze_event["frozen_at_state_step"] == state_step + 1),
            "post_update_delta_sha256": tensor_hash(delta),
            "optimizer": optimizer_hashes(opt, delta),
        })
        torch.cuda.empty_cache()

    curve_by_step = {row["step"]: row for row in curve}
    final = curve_by_step[probe.STEPS]
    frozen_png_unchanged_to_final = bool(freeze_event and freeze_event["frozen_png_sha256"] == final["png_sha256"])
    payload = {
        "status": "DONE" if freeze_event else "NO_FREEZE_EVENT",
        "scope": "single-sample causal state-retention ablation; no candidate reselection, no new budget, no sample expansion",
        "protocol": protocol,
        "baseline": baseline,
        "choice": choice,
        "curve": curve,
        "updates": updates,
        "freeze_event": freeze_event,
        "final_step_25": final,
        "step_25_strict_endpoint_trainer": final["strict_success"],
        "frozen_png_unchanged_to_final_checkpoint": frozen_png_unchanged_to_final,
        "counters": base.counters(),
        "elapsed_sec": time.perf_counter() - started,
        "next_gate": "Run fresh-process replay at freeze state (if sealed) and step 25. Only a retained step-25 error after fresh replay supports considering state-gated multi-budget allocation.",
    }
    (out / "trajectory.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "freeze_step": freeze_event["frozen_at_state_step"] if freeze_event else None,
        "step25_strict": final["strict_success"],
        "elapsed_sec": payload["elapsed_sec"],
    }), flush=True)


if __name__ == "__main__":
    main()
