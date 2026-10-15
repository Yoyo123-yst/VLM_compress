"""One trajectory. Sign-PGD, prepare-path forward, hard Full guard, finite attempts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
HERE = ROOT / "week2_attack_v22_safety_slack_constraint_pilot"

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v20_fourway_content_ablation.semantics import classify  # noqa: E402
from week2_attack_v22_safety_slack_constraint_pilot.decisions import choose_selector  # noqa: E402
from week2_attack_v22_safety_slack_constraint_pilot.runtime import (  # noqa: E402
    apply_sign,
    attack_grads,
    cosine,
    delta_hash,
    load_groups,
    reference_of,
    score,
)

CHECKPOINTS = {0, 1, 2, 5, 10, 15, 20, 25}
MAX_ACCEPTED = 25
MAX_ATTEMPTS = 40


def heartbeat(name: str, row: dict) -> None:
    path = HERE / "runtime" / f"{name}.heartbeat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", choices=("T0", "T1", "T2"), required=True)
    args = parser.parse_args()
    name = args.trajectory
    preflight = json.loads((HERE / "preflight.json").read_text())
    if preflight["status"] != "PASS":
        raise SystemExit("preflight did not pass")
    groups = load_groups()
    reference, pieces, core_pos = reference_of(groups)
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    state = rt.prepare_clean_state(model, processor, item)
    original = item["images"][item["aux_idx"]].convert("RGB")
    clean_page = state["page"].detach()
    tau = torch.tensor(preflight["tau"], dtype=torch.float32, device=clean_page.device)
    clean_hard = torch.tensor(preflight["clean_full_hard_margins"], dtype=torch.float32, device=clean_page.device)
    out = HERE / "trajectories" / name
    out.mkdir(parents=True, exist_ok=True)
    cost = {"forward": 0, "backward": 0, "generate": 0}
    delta = torch.zeros_like(clean_page)
    image = original
    clean_kept = None
    records = []
    accepted = 0
    attempts = 0
    stopped = None
    while True:
        point_cost = {"forward": 0, "backward": 0, "generate": 0}
        scored = score(model, processor, item, state, image, reference, pieces, core_pos, clean_kept or [], point_cost)
        if clean_kept is None:
            clean_kept = scored["keep_indices"]
            scored = score(model, processor, item, state, image, reference, pieces, core_pos, clean_kept, point_cost)
        core_mean = sum(row["hard_margin"] for row in scored["core_margins"]) / len(scored["core_margins"])
        full_hard = [row["hard_margin"] for row in scored["full_margins"]]
        weakest = min(scored["full_margins"], key=lambda row: row["hard_margin"])
        l_core_now = sum(row["soft_margin"] for row in scored["core_margins"]) / len(scored["core_margins"])
        row = {
            "trajectory": name,
            "accepted_after": accepted,
            "attempts": attempts,
            "l_core": l_core_now,
            "full_hard_min": min(full_hard),
            "weakest_full_token": weakest,
            "full_margins": scored["full_margins"],
            "core_margins": scored["core_margins"],
            "core_hard_mean": core_mean,
            "full_text_exact": scored["full_text_exact"],
            "full_token_exact": scored["full_token_exact"],
            "full_replay": scored["full_replay"],
            "full_text": scored["full_text"],
            "full_tokens": scored["full_tokens"],
            "fast_text": scored["fast_text"],
            "fast_tokens": scored["fast_tokens"],
            "fast_valid": scored["fast_output_valid"],
            "k": scored["k"],
            "keep_hash": scored["keep_hash"],
            "uint8_linf": rt.uint8_linf(original, image),
            "png_sha256": None,
            "semantic_status_diagnostic": classify("47450", scored["fast_text"], [])["semantic_status"],
        }
        if accepted in CHECKPOINTS and not any(item_row.get("checkpoint_step") == accepted for item_row in records):
            dest = HERE / "sealed_png" / f"{name}_step_{accepted:02d}.png"
            row["png_sha256"] = rt.save_png(image, dest)
            row["checkpoint_step"] = accepted
            row["checkpoint_saved"] = True
            rt.write_json(out / f"step_{accepted:02d}.json", row)
        records.append(row)
        for key in cost:
            cost[key] += point_cost[key]
        heartbeat(name, {"accepted": accepted, "attempts": attempts, "l_core": l_core_now, "core_mean": core_mean, "fast_text": scored["fast_text"]})
        print(json.dumps({"trajectory": name, "accepted": accepted, "core_mean": core_mean, "text": scored["fast_text"], "full_exact": scored["full_text_exact"]}), flush=True)
        if accepted >= MAX_ACCEPTED or attempts >= MAX_ATTEMPTS or stopped:
            break
        before = delta_hash(delta)
        snapshot = delta.detach().clone()
        try:
            grads = attack_grads(
                model, processor, item, state, image, clean_page + delta, reference, core_pos, clean_hard, tau, name, point_cost
            )
        except torch.cuda.OutOfMemoryError as exc:
            rt.write_json(out / "blocked.json", {"status": "BLOCKED_OOM"})
            raise SystemExit(2) from exc
        except RuntimeError as exc:
            rt.write_json(out / "blocked.json", {"status": "BLOCKED_NONFINITE_GRADIENT", "error": str(exc)})
            raise SystemExit(2) from exc
        apply_sign(delta, clean_page, grads["update_grad"])
        candidate = rt.project_image(clean_page, delta)
        attempt_path = out / "attempts" / f"attempt_{attempts:03d}.png"
        rt.save_png(candidate, attempt_path)
        reloaded = rt.reload_png(attempt_path)
        guard_cost = {"forward": 0, "backward": 0, "generate": 0}
        guarded = score(model, processor, item, state, reloaded, reference, pieces, core_pos, clean_kept, guard_cost)
        legal = bool(
            guarded["full_text_exact"] and guarded["full_token_exact"] and guarded["full_replay"]
            and guarded["full_output_valid"] and rt.uint8_linf(original, reloaded) <= 8
        )
        attempt_row = {
            "attempt": attempts,
            "l_core_before": row["l_core"],
            "l_core_graph": grads["l_core"],
            "l_side": grads["l_side"],
            "grad_cosine": cosine(grads["attack_grad"], grads["side_grad"]),
            "accepted": legal,
            "reason": "FULL_GUARD_PASS" if legal else "FULL_GUARD_REJECT",
            "optimizer_before": before,
            "uint8_linf": rt.uint8_linf(original, reloaded),
            "full_text_exact": guarded["full_text_exact"],
            "full_token_exact": guarded["full_token_exact"],
            "full_text": guarded["full_text"],
            "core_hard_mean": sum(item_row["hard_margin"] for item_row in guarded["core_margins"]) / len(guarded["core_margins"]),
            "core_margins": guarded["core_margins"],
            "full_hard_min": min(item_row["hard_margin"] for item_row in guarded["full_margins"]),
            "weakest_full_token": min(guarded["full_margins"], key=lambda item_row: item_row["hard_margin"]),
            "fast_text": guarded["fast_text"],
            "fast_tokens": guarded["fast_tokens"],
            "k": guarded["k"],
            "keep_hash": guarded["keep_hash"],
        }
        if not legal:
            delta.copy_(snapshot)
            attempt_row["optimizer_after_rollback"] = delta_hash(delta)
            attempt_row["rollback_matches"] = attempt_row["optimizer_after_rollback"] == before
            if attempt_row["rollback_matches"] is not True:
                rt.write_json(out / "blocked.json", {"status": "BLOCKED_ROLLBACK_MISMATCH", "attempt": attempt_row})
                raise SystemExit(2)
            if name == "T0":
                stopped = "FIRST_FULL_GUARD_FAILURE"
        else:
            image = reloaded
            accepted += 1
            attempt_row["optimizer_after_rollback"] = None
        attempts += 1
        for key in cost:
            cost[key] += point_cost[key] + guard_cost[key]
        rt.write_json(out / "attempts" / f"attempt_{attempts - 1:03d}.json", attempt_row)
        if stopped:
            break
    selector_rows = []
    for saved in records:
        if "checkpoint_step" not in saved:
            continue
        selector_rows.append({
            "step": saved["checkpoint_step"],
            "legal": bool(saved["full_text_exact"] and saved["full_token_exact"] and saved["full_replay"] and saved["uint8_linf"] <= 8),
            "l_core": saved["l_core"],
        })
    chosen = choose_selector(selector_rows)
    rt.write_json(out / "selector.json", {"rule": "lowest L_core, then smaller step", "chosen": chosen, "rows": selector_rows, "stopped": stopped, "cost": cost})
    print(json.dumps({"trajectory": name, "selector": chosen, "stopped": stopped, "cost": cost}), flush=True)


if __name__ == "__main__":
    main()
