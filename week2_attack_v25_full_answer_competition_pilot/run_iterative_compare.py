"""Small, isolated three-way complete-answer pilot.

No monkey patches, hooks, or wrapped model methods are used here.  Every
candidate is quantized to PNG and reloaded before the Full guard is evaluated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "week2_attack_v25_full_answer_competition_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from run_causal_v2 import prepare, spans_of  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402
from week2_attack_v13_branch_reachability_probe import run_branch_probe as probe  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation.losses import decision_preservation  # noqa: E402
from week2_attack_v25_full_answer_competition_pilot import generate_clean_candidates as candidates  # noqa: E402
from week2_attack_v25_full_answer_competition_pilot.score_smoke import dedupe, encode_answer  # noqa: E402

EPS = 8.0 / 255.0
ALPHA = 1.0 / 255.0
LAMBDA_FULL = 1.0
RATIO = 0.25
FRONT_BETA = 1.0
FRONT_TAU = 0.25
MAX_ACCEPTED = 10
MAX_ATTEMPTS = 20
METHODS = ("native_fdp", "complete_target", "complete_front", "alias_group")


def h_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().float().cpu().numpy().tobytes()).hexdigest()


def h_rgb(image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def h_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def sequence_scores(model, inputs, pixels, sequences, keep_index):
    values = []
    for seq in sequences:
        rows = v7d.fastv_rows(model, inputs, pixels, seq, keep_index)
        values.append(torch.stack([torch.log_softmax(row, -1)[int(tok)] for row, tok in zip(rows, seq)]).mean())
    return values


def generate_counted(model, inputs, pixels, spans, ratio, counters):
    counters["generate_calls"] += 1
    return v7d.generate_tokens(model, inputs, pixels, spans, ratio)


def fastv_rows_counted(model, inputs, pixels, tokens, keep_index, counters):
    counters["fastv_score_forwards"] += 1
    return v7d.fastv_rows(model, inputs, pixels, tokens, keep_index)


def full_rows_counted(model, inputs, pixels, tokens, counters):
    counters["full_score_forwards"] += 1
    return v7d.full_rows(model, inputs, pixels, tokens)


def objective_and_grad(model, state, item, method, correct, wrong, delta, counters):
    page = (state["page"] + delta).detach().clone().requires_grad_(True)
    pixels = candidates.anchored_pixels_from_page(state, item, page)
    # The natural keep set is measured from the same anchored pixels used by
    # the differentiable loss; generation itself is not wrapped or patched.
    fast_tokens, kept = generate_counted(model, state["inputs"], pixels, state["spans"], RATIO, counters)
    keep_index = probe.keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
    if method == "native_fdp":
        rows = fastv_rows_counted(model, state["inputs"], pixels, fast_tokens, keep_index, counters)
        attack = torch.stack([v7d.soft_margin_row(row, tok) for row, tok in zip(rows, fast_tokens)]).mean()
        attack_info = {"fast_tokens": [int(x) for x in fast_tokens], "correct_scores": [], "wrong_score": None}
        attack_value = None
    else:
        # Score one sequence at a time.  Retaining all complete-answer graphs
        # is needlessly expensive and caused the earlier pilot OOM.
        seq_values = []
        seq_grads = []
        # complete-target/front use only the primary clean expression.  Alias
        # scoring is a separate method (alias_group); otherwise the comparison
        # pays for graphs that never enter the attack direction.
        sequences = (correct + [wrong]) if method == "alias_group" else ([correct[0], wrong])
        for seq in sequences:
            seq_pixels = candidates.anchored_pixels_from_page(state, item, page)
            seq_rows = fastv_rows_counted(model, state["inputs"], seq_pixels, seq, keep_index, counters)
            seq_score = torch.stack([torch.log_softmax(row, -1)[int(tok)] for row, tok in zip(seq_rows, seq)]).mean()
            seq_value = float(seq_score.detach())
            seq_score.backward()
            counters["complete_sequence_backwards"] += 1
            if page.grad is None or not torch.isfinite(page.grad).all():
                raise RuntimeError("non-finite complete-answer gradient")
            seq_values.append(seq_value)
            seq_grads.append(page.grad.detach().clone())
            page.grad = None
            del seq_pixels, seq_rows, seq_score
            gc.collect()
            torch.cuda.empty_cache()
        correct_scores = seq_values[:-1]
        wrong_score = seq_values[-1]
        if method in ("complete_target", "complete_front"):
            attack_value = correct_scores[0] - wrong_score
            grad_attack = seq_grads[0] - seq_grads[-1]
        else:
            weights = torch.softmax(torch.tensor(correct_scores, dtype=torch.float64), dim=0).tolist()
            attack_value = float(torch.logsumexp(torch.tensor(correct_scores, dtype=torch.float64), dim=0)) - wrong_score
            grad_attack = sum((float(weight) * grad for weight, grad in zip(weights, seq_grads)), torch.zeros_like(seq_grads[0])) - seq_grads[-1]
        attack_info = {
            "fast_tokens": [int(x) for x in fast_tokens],
            "correct_scores": correct_scores,
            "wrong_score": wrong_score,
        }
        if method == "complete_front":
            first_pos = next((i for i, (a, b) in enumerate(zip(correct[0], wrong)) if a != b), min(len(correct[0]), len(wrong)))
            if first_pos >= len(wrong):
                raise RuntimeError("wrong target has no first divergence position")
            front_pixels = candidates.anchored_pixels_from_page(state, item, page)
            front_rows = fastv_rows_counted(model, state["inputs"], front_pixels, wrong, keep_index, counters)
            front_logits = torch.log_softmax(front_rows[first_pos], dim=-1)
            wrong_logit = front_logits[int(wrong[first_pos])]
            competitor_mask = torch.ones_like(front_logits, dtype=torch.bool)
            competitor_mask[int(wrong[first_pos])] = False
            competitor_logit = front_logits.masked_fill(~competitor_mask, -torch.inf).max()
            front_loss = torch.relu(FRONT_TAU + competitor_logit - wrong_logit)
            front_value = float(front_loss.detach())
            front_loss.backward()
            counters["front_backwards"] += 1
            if page.grad is None or not torch.isfinite(page.grad).all():
                raise RuntimeError("non-finite first-divergence gradient")
            grad_front = page.grad.detach().clone()
            page.grad = None
            attack_value = attack_value + FRONT_BETA * front_value
            grad_attack = grad_attack + FRONT_BETA * grad_front
            attack_info["first_divergence_position"] = first_pos
            attack_info["front_loss"] = front_value
            del front_pixels, front_rows, front_logits, front_loss, grad_front
        del seq_grads
    # Do not retain FastV and Full graphs together.  The sum of the two
    # gradients is identical, while sequential backward keeps the 40-GiB
    # model below the card limit.
    if method == "native_fdp":
        attack_value = float(attack.detach())
        attack.backward()
        counters["native_attack_backwards"] += 1
        if page.grad is None or not torch.isfinite(page.grad).all():
            raise RuntimeError("non-finite attack gradient")
        grad_attack = page.grad.detach().clone()
        page.grad = None
        del attack, rows
    else:
        pass
    fast_tokens_out = [int(x) for x in fast_tokens]
    kept_out = [int(x) for x in kept]
    keep_hash_out = probe.keep_hash(kept)
    attack_info_out = dict(attack_info)
    del pixels, fast_tokens, kept, keep_index, attack_info
    gc.collect()
    torch.cuda.empty_cache()
    pixels_full = candidates.anchored_pixels_from_page(state, item, page)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    try:
        full_rows = full_rows_counted(model, state["inputs"], pixels_full, state["clean_tokens"], counters)
    finally:
        model.eval()
        model.gradient_checkpointing_disable()
    hard = torch.stack([v7d.hard_margin_row(row, tok) for row, tok in zip(full_rows, state["clean_tokens"])])
    hard_margins_out = [float(x) for x in hard.detach().cpu()]
    fdp = decision_preservation(hard, state["clean_hard"])
    fdp_value = float(fdp.detach())
    fdp.backward()
    counters["full_fdp_backwards"] += 1
    if page.grad is None or not torch.isfinite(page.grad).all():
        raise RuntimeError("non-finite Full-FDP gradient")
    grad = grad_attack + LAMBDA_FULL * page.grad.detach()
    del fdp, full_rows, hard, pixels_full, grad_attack
    torch.cuda.empty_cache()
    return {
        "attack_loss": attack_value,
        "fdp_loss": fdp_value,
        "total_loss": attack_value + LAMBDA_FULL * fdp_value,
        "fast_tokens": fast_tokens_out,
        "kept": kept_out,
        "keep_hash": keep_hash_out,
        "hard_margins": hard_margins_out,
        **attack_info_out,
    }, grad


def replay(model, processor, item, state, image, counters) -> dict:
    images = list(item["images"])
    images[item["aux_idx"]] = image
    inputs = prepare(processor, images, item["meta"]["question"])
    spans = spans_of(inputs, int(model.config.image_token_index), len(images))
    full, _ = generate_counted(model, inputs, inputs["pixel_values"], spans, None, counters)
    fast, kept = generate_counted(model, inputs, inputs["pixel_values"], spans, RATIO, counters)
    full_text = decode(processor, torch.tensor(full)).strip()
    fast_text = decode(processor, torch.tensor(fast)).strip()
    return {
        "full_tokens": [int(x) for x in full],
        "full_text": full_text,
        "full_text_exact": bool(full_text == state["clean_text"]),
        "full_token_exact": bool(full == state["clean_tokens"]),
        "full_replay": True,
        "fast_tokens": [int(x) for x in fast],
        "fast_text": fast_text,
        "keep_hash": probe.keep_hash(kept),
        "k": len(kept),
    }


def guard_replay(model, processor, item, state, image, counters) -> dict:
    first = replay(model, processor, item, state, image, counters)
    second = replay(model, processor, item, state, image, counters)
    first["full_replay"] = bool(first["full_tokens"] == second["full_tokens"] and first["full_text"] == second["full_text"])
    first["fast_replay"] = bool(first["fast_tokens"] == second["fast_tokens"] and first["fast_text"] == second["fast_text"] and first["keep_hash"] == second["keep_hash"])
    return first


def load_state(model, processor, item, counters):
    state = v7d.prepare_item(model, processor, item)
    fast, kept = generate_counted(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], RATIO, counters)
    state["clean_fast_tokens"] = fast
    state["clean_fast_kept"] = kept
    state["clean_fast_keep_hash"] = probe.keep_hash(kept)
    return state


def run_one(sample, method, device, out_root, max_accepted=MAX_ACCEPTED, max_attempts=MAX_ATTEMPTS, targets_path=None):
    probe.configure_determinism()
    torch.cuda.set_device(int(device))
    model, processor = base.load_model_frozen()
    model.eval()
    # This is an input-gradient attack, not weight training.  Keeping model
    # parameter gradients would waste tens of GiB and can change the replay
    # memory footprint between methods.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_file = Path(targets_path) if targets_path else (HERE / "frozen_targets.json")
    frozen = json.loads(target_file.read_text())["samples"][sample]
    item = v7d.load_dev(sample)
    counters = {"generate_calls": 0, "fastv_score_forwards": 0, "full_score_forwards": 0,
                "native_attack_backwards": 0, "complete_sequence_backwards": 0,
                "front_backwards": 0, "full_fdp_backwards": 0}
    state = load_state(model, processor, item, counters)
    zero = candidates.zero_injection_audit(model, state, item, {
        "tokens": state["clean_fast_tokens"], "kept": state["clean_fast_kept"],
        "keep_hash": state["clean_fast_keep_hash"], "k": len(state["clean_fast_kept"]),
        "text": decode(processor, torch.tensor(state["clean_fast_tokens"])).strip(),
    })
    if not all((zero["pixel_tensor_exact"], zero["fast_tokens_exact"], zero["fast_keep_exact"])):
        raise RuntimeError(f"ZERO_REPLAY_NOT_EXACT: {zero}")
    correct = dedupe([[int(x) for x in state["clean_fast_tokens"]] + []] + [encode_answer(processor, x) for x in frozen["correct_aliases"]])
    wrong = [int(x) for x in frozen["wrong_target_tokens"]]
    if decode(processor, torch.tensor(wrong)).strip() != frozen["wrong_target_text"]:
        raise RuntimeError("frozen wrong target decode mismatch")
    original = item["images"][item["aux_idx"]].convert("RGB")
    delta = torch.zeros_like(state["page"])
    records = []
    accepted = 0
    attempts = 0
    started = time.perf_counter()
    dest = out_root / method / sample
    dest.mkdir(parents=True, exist_ok=True)
    while accepted < max_accepted and attempts < max_attempts:
        attempts += 1
        previous = delta.detach().clone()
        before_counters = dict(counters)
        info, grad = objective_and_grad(model, state, item, method, correct, wrong, delta, counters)
        candidate_delta = (delta - ALPHA * grad.sign()).clamp(-EPS, EPS)
        candidate_image = probe.page_uint8(state["page"], candidate_delta)
        png_path = dest / f"attempt_{attempts:03d}.png"
        candidate_image.save(png_path, format="PNG")
        reloaded = Image.open(png_path).convert("RGB")
        guard = guard_replay(model, processor, item, state, reloaded, counters)
        actual = np.asarray(reloaded, dtype=np.int16)
        orig = np.asarray(original, dtype=np.int16)
        linf = int(np.abs(actual - orig).max())
        changed = h_rgb(reloaded) != h_rgb(original)
        accept = bool(guard["full_text_exact"] and guard["full_token_exact"] and guard["full_replay"] and guard["fast_replay"] and linf <= 8 and changed)
        if accept:
            delta = candidate_delta.detach()
            accepted += 1
            final_path = dest / f"accepted_{accepted:03d}.png"
            final_path.write_bytes(png_path.read_bytes())
        else:
            delta = previous
            png_path.unlink(missing_ok=True)
        records.append({
            "attempt": attempts, "accepted": accept, "accepted_after": accepted,
            "attack_loss": info["attack_loss"], "fdp_loss": info["fdp_loss"], "total_loss": info["total_loss"],
            "grad_hash": h_tensor(grad), "grad_l2": float(grad.norm()), "grad_linf": float(grad.abs().max()),
            "delta_before_hash": h_tensor(previous), "delta_candidate_hash": h_tensor(candidate_delta), "delta_after_hash": h_tensor(delta),
            "png_sha256": h_file(final_path if accept else (dest / f"attempt_{attempts:03d}.png")) if accept else None,
            "uint8_linf": linf, "guard": guard, "surrogate_info": info,
            "wall_seconds": time.perf_counter() - started,
            "counter_delta": {k: counters[k] - before_counters[k] for k in counters},
        })
        torch.cuda.empty_cache()
    payload = {
        "status": "DONE", "sample_id": sample, "method": method, "optimizer": "sign_pgd",
        "ratio": RATIO, "alpha": ALPHA, "eps": EPS, "lambda_full": LAMBDA_FULL,
        "max_accepted": max_accepted, "max_attempts": max_attempts, "accepted": accepted, "attempts": attempts,
        "zero_injection_audit": zero, "correct_sequences": [decode(processor, torch.tensor(x)).strip() for x in correct],
        "wrong_sequence": frozen["wrong_target_text"], "updates": records,
        "clean_full_text": state["clean_text"], "clean_fast_text": decode(processor, torch.tensor(state["clean_fast_tokens"])).strip(),
        "clean_fast_keep_hash": state["clean_fast_keep_hash"], "wall_seconds": time.perf_counter() - started,
        "script_sha256": h_file(Path(__file__)),
        "selector_rule": "last accepted checkpoint after max_accepted accepted updates; no oracle checkpoint selection",
        "counters": counters,
    }
    json_write(dest / "result.json", payload)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, choices=("47450", "7006", "60700", "47567", "50779", "51133", "6194", "62941", "64506"))
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-accepted", type=int, default=MAX_ACCEPTED)
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    ap.add_argument("--targets", default=None, help="frozen target manifest; defaults to frozen_targets.json")
    args = ap.parse_args()
    result = run_one(args.sample, args.method, args.device, Path(args.out), args.max_accepted, args.max_attempts, args.targets)
    print(json.dumps({"sample": args.sample, "method": args.method, "status": result["status"], "accepted": result["accepted"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
