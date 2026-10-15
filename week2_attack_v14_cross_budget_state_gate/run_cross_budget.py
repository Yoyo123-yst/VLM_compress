"""State-gated allocation from the sealed 60700 AWARE freeze PNG.

The r=0.25 branch is already satisfied and must stay satisfied.  The only
new question is whether further updates aimed at the same token pair under
r=0.35 add a natural FastV failure that the sealed PNG does not already have.
No candidate is re-screened and no other sample is used.
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
HERE = ROOT / "week2_attack_v14_cross_budget_state_gate"
V13 = ROOT / "week2_attack_v13_branch_reachability_probe"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(V13)]

import run_branch_probe as probe  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402

SAMPLE = "60700"
SCHEDULE = (0.25, 0.35)
OBSERVE = (0.15, 0.25, 0.35, 0.50)
EXTRA_STEPS = 15
FREEZE_PNG = V13 / "runs" / "state_retention_v1" / SAMPLE / "AWARE_FREEZE" / "step_011.png"
SCREEN = V13 / "runs" / "construct_v1r1" / SAMPLE / "screen.json"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def rows_at_ratio(model, state, item, page, reference, ratio: float):
    pixels = v7d.pixels_from_page(state["inputs"], item["aux_idx"], page)
    generated, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], ratio)
    keep = v7d.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
    rows = v7d.fastv_rows(model, state["inputs"], pixels, reference, keep)
    return rows, generated, kept


def natural_fast(model, processor, item, image: Image.Image, ratio: float) -> dict:
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    tokens, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, ratio)
    text = probe.decode(processor, torch.tensor(tokens)).strip()
    anls = probe.anls_score(text, list(item["meta"]["answers"]))
    return {
        "tokens": tokens,
        "text": text,
        "anls": anls,
        "valid": bool(probe.output_valid(text)),
        "endpoint_failure": bool(probe.output_valid(text) and anls < 0.5),
        "k": len(kept) if kept is not None else None,
        "keep_hash": None if kept is None else keep_hash(kept),
    }


def aligned(tokens: list[int], choice: dict) -> bool:
    position = int(choice["position"])
    return len(tokens) > position and int(tokens[position]) == int(choice["target_id"])


def pair_gap(model, state, item, page, reference, ratio: float, choice: dict) -> dict:
    rows, generated, kept = rows_at_ratio(model, state, item, page, reference, ratio)
    position = int(choice["position"])
    gap = rows[position][int(choice["target_id"])] - rows[position][int(choice["candidate_id"])]
    result = {
        "ratio": ratio,
        "pair_gap": float(gap.detach()),
        "threshold_met": bool(float(gap.detach()) <= probe.TAU),
        "keep_hash": keep_hash(kept),
        "k": len(kept),
        "generated_tokens": generated,
    }
    del rows
    return result


def branch_direction(model, state, item, delta, reference, ratio: float, choice: dict):
    delta.grad = None
    page = state["page"] + delta
    rows, generated, kept = rows_at_ratio(model, state, item, page, reference, ratio)
    position = int(choice["position"])
    gap = rows[position][int(choice["target_id"])] - rows[position][int(choice["candidate_id"])]
    attack_loss = torch.relu(gap + probe.TAU)
    attack_loss.backward()
    attack = delta.grad.detach().clone()
    delta.grad = None
    fdp, violated = base.fdp_value(model, state, item, page)
    (probe.LAMBDA_FULL * fdp).backward()
    full = delta.grad.detach().clone()
    delta.grad = None
    return attack + full, {
        "pair_gap": float(gap.detach()),
        "branch_loss": float(attack_loss.detach()),
        "attack_l2": float(attack.float().norm()),
        "full_l2": float(full.float().norm()),
        "full_proxy_violated": bool(violated),
        "keep_hash": keep_hash(kept),
        "generated_tokens": generated,
    }


def score_image(model, processor, item, state, image: Image.Image, references: dict, choice: dict) -> dict:
    full = base.full_exact_check(model, processor, item, state, image)
    page = v7d.page_to_float(image).to(state["page"].device, state["page"].dtype)
    ratios = {}
    for ratio in OBSERVE:
        row = natural_fast(model, processor, item, image, ratio)
        reference = references.get(ratio)
        if reference is not None and aligned(reference, choice):
            row["pair"] = pair_gap(model, state, item, page, reference, ratio, choice)
            row["branch_aligned"] = True
        else:
            row["pair"] = None
            row["branch_aligned"] = False
        ratios[f"{ratio:.2f}"] = row
    clean = item["images"][item["aux_idx"]]
    linf = int(np.max(np.abs(np.asarray(image, dtype=np.int16) - np.asarray(clean, dtype=np.int16))))
    return {"full": full, "ratios": ratios, "linf_uint8": linf, "linf_ok": bool(0 < linf <= 8)}


def exact_full(guard: dict) -> bool:
    return bool(guard["full_text_exact"] and guard["full_token_exact"] and guard["full_replay"])


def write_protocol(out: Path) -> dict:
    protocol = {
        "status": "CROSS_BUDGET_STATE_GATE_REGISTERED",
        "sample_id": SAMPLE,
        "source_png": str(FREEZE_PNG.relative_to(ROOT)),
        "source_png_sha256": sha_file(FREEZE_PNG),
        "branch": "frozen AWARE choice from construct_v1r1; not re-screened",
        "held_budget": 0.25,
        "scheduled_budget": 0.35,
        "observed_ratios": list(OBSERVE),
        "extra_steps_cap": EXTRA_STEPS,
        "decision_tree": [
            "If r=0.35 clean FastV is not the same target token at the frozen branch position, do not schedule.",
            "If the sealed PNG already has natural FastV ANLS<0.5 at r=0.35 with Full exact and r=0.25 still failing, record CHECKPOINT_ALREADY_COVERS and do not update.",
            "If the r=0.35 pair margin is already <= 0, record HINGE_ALREADY_MET and do not update.",
            "Otherwise run at most 15 Adam steps of the same branch hinge at r=0.35 plus the unchanged Full-FDP term.",
            "Reject an update unless Full text, token, and replay stay exact and the r=0.25 pair gap stays <= 0.",
            "Freeze at the first accepted state whose r=0.35 pair gap is <= 0.",
            "STATE_GATE_ADDS_ENDPOINT only if that saved PNG newly creates a natural r=0.35 ANLS<0.5 while r=0.25 failure and Full exact remain.",
        ],
        "anls_used_for_update_acceptance": False,
        "sample_expansion": "NOT_RUN",
    }
    (out / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def run() -> None:
    probe.configure_determinism()
    out = HERE / "runs" / SAMPLE
    out.mkdir(parents=True, exist_ok=True)
    protocol = write_protocol(out)
    choice = json.loads(SCREEN.read_text())["aware_choice"]
    model, processor = base.load_model_frozen()
    item = v7d.load_dev(SAMPLE)
    state = v7d.prepare_item(model, processor, item)
    references = {}
    clean_rows = {}
    for ratio in OBSERVE:
        tokens_a, kept_a = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], ratio)
        tokens_b, _kept_b = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], ratio)
        text = probe.decode(processor, torch.tensor(tokens_a)).strip()
        references[ratio] = tokens_a
        clean_rows[f"{ratio:.2f}"] = {
            "tokens": tokens_a,
            "replay": tokens_a == tokens_b,
            "text": text,
            "anls": probe.anls_score(text, list(item["meta"]["answers"])),
            "aligned_to_aware_branch": aligned(tokens_a, choice),
            "keep_hash": keep_hash(kept_a),
        }
    image = Image.open(FREEZE_PNG).convert("RGB")
    checkpoint = score_image(model, processor, item, state, image, references, choice)
    r025 = checkpoint["ratios"]["0.25"]
    r035 = checkpoint["ratios"]["0.35"]
    covers = bool(
        exact_full(checkpoint["full"])
        and r025["endpoint_failure"]
        and r035["endpoint_failure"]
        and r035["branch_aligned"]
    )
    hinge_met = bool(r035["branch_aligned"] and r035["pair"] and r035["pair"]["threshold_met"])
    held_ok = bool(r025["branch_aligned"] and r025["pair"] and r025["pair"]["threshold_met"] and r025["endpoint_failure"])
    schedule = bool(r035["branch_aligned"] and held_ok and exact_full(checkpoint["full"]) and not covers and not hinge_met)
    payload = {
        "protocol": protocol,
        "choice": {key: choice[key] for key in ("position", "target_id", "target_piece", "candidate_id", "candidate_piece")},
        "clean": clean_rows,
        "checkpoint_png": str(FREEZE_PNG.relative_to(ROOT)),
        "checkpoint": checkpoint,
        "schedule_updates": schedule,
    }
    if not schedule:
        if not r035["branch_aligned"] or not held_ok:
            status = "BLOCKED_BRANCH_NOT_ALIGNED"
        elif covers:
            status = "CHECKPOINT_ALREADY_COVERS"
        elif hinge_met:
            status = "HINGE_ALREADY_MET_NO_UPDATE"
        else:
            status = "NO_SCHEDULE"
        payload["status"] = status
        (out / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"status": status}), flush=True)
        return

    attacked_page = v7d.page_to_float(image).to(state["page"].device, state["page"].dtype)
    delta = (attacked_page - state["page"]).detach().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=probe.ADAM_LR)
    updates = []
    sealed = None
    started = time.perf_counter()
    for step in range(EXTRA_STEPS):
        if sealed is not None:
            break
        direction, meta = branch_direction(model, state, item, delta, references[0.35], 0.35, choice)
        before_delta = delta.detach().clone()
        before_opt = copy.deepcopy(opt.state_dict())
        delta.grad = direction
        opt.step()
        project(delta, state["page"], probe.EPS)
        delta.grad = None
        current = page_uint8(state["page"], delta)
        guarded = base.full_exact_check(model, processor, item, state, current)
        page = state["page"] + delta.detach()
        held = pair_gap(model, state, item, page, references[0.25], 0.25, choice)
        accept = exact_full(guarded) and held["threshold_met"]
        if not accept:
            delta.data.copy_(before_delta)
            opt.load_state_dict(before_opt)
            active = None
        else:
            active = pair_gap(model, state, item, state["page"] + delta.detach(), references[0.35], 0.35, choice)
            if active["threshold_met"]:
                path = out / "state_gate.png"
                page_uint8(state["page"], delta).save(path)
                sealed = {"step": step + 1, "png": str(path.relative_to(ROOT)), "png_sha256": sha_file(path), "pair": active}
        updates.append({"step": step, "meta": meta, "accepted": accept, "held_025": held, "active_035": active, "full_text": guarded.get("full_text")})
        print(json.dumps({"step": step, "accepted": accept, "gap35": None if active is None else active["pair_gap"]}), flush=True)
        torch.cuda.empty_cache()
    if sealed is None:
        path = out / "state_gate.png"
        page_uint8(state["page"], delta).save(path)
        sealed = {"step": EXTRA_STEPS, "png": str(path.relative_to(ROOT)), "png_sha256": sha_file(path), "pair": None, "cap_reached": True}
    final_image = Image.open(out / "state_gate.png").convert("RGB")
    final = score_image(model, processor, item, state, final_image, references, choice)
    gained = bool(
        (not r035["endpoint_failure"])
        and final["ratios"]["0.35"]["endpoint_failure"]
        and final["ratios"]["0.25"]["endpoint_failure"]
        and exact_full(final["full"])
    )
    lost = bool((not final["ratios"]["0.25"]["endpoint_failure"]) or (not exact_full(final["full"])))
    if lost:
        status = "STATE_GATE_LOST_025_OR_FULL"
    elif gained:
        status = "STATE_GATE_ADDS_ENDPOINT"
    else:
        status = "STATE_GATE_NO_ADDED_ENDPOINT"
    payload.update({
        "status": status,
        "updates": updates,
        "sealed": sealed,
        "final": final,
        "elapsed_sec": time.perf_counter() - started,
    })
    (out / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": status, "sealed_step": sealed["step"]}), flush=True)


def replay() -> None:
    probe.configure_determinism()
    result_path = HERE / "runs" / SAMPLE / "result.json"
    result = json.loads(result_path.read_text())
    png = FREEZE_PNG if result["status"] in {"CHECKPOINT_ALREADY_COVERS", "HINGE_ALREADY_MET_NO_UPDATE", "BLOCKED_BRANCH_NOT_ALIGNED", "NO_SCHEDULE"} else HERE / "runs" / SAMPLE / "state_gate.png"
    model, processor = base.load_model_frozen()
    item = v7d.load_dev(SAMPLE)
    state = v7d.prepare_item(model, processor, item)
    choice = result["choice"]
    references = {float(key): value["tokens"] for key, value in result["clean"].items()}
    image = Image.open(png).convert("RGB")
    first = score_image(model, processor, item, state, image, references, choice)
    second = score_image(model, processor, item, state, image, references, choice)
    same = json.dumps(first["ratios"], sort_keys=True) == json.dumps(second["ratios"], sort_keys=True) and first["full"] == second["full"]
    payload = {"png": str(png.relative_to(ROOT)), "png_sha256": sha_file(png), "first": first, "second": second, "replay_match": same}
    (HERE / "runs" / SAMPLE / "fresh_replay.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"replay_match": same, "r025": first["ratios"]["0.25"]["text"], "r035": first["ratios"]["0.35"]["text"], "full": first["full"]["full_text"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "replay"))
    args = parser.parse_args()
    if args.mode == "run":
        run()
    else:
        replay()
