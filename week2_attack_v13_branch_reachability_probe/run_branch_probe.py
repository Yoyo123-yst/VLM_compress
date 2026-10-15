"""Small, falsifiable probe for Full-aware compression branch selection.

This is deliberately a construction experiment, not a new attack result.  It
keeps the target, source-bounded auxiliary-page variable, epsilon, Full guard,
and Native-FDP preservation term fixed.  The sole manipulated factor is which
FastV answer branch is attacked at r=0.25:

  SEQ   fixed clean FastV sequence (reference baseline)
  NAIVE feasible branch with smallest estimated FastV flip radius
  AWARE feasible branch maximizing Full local capacity / FastV flip radius

Candidate continuation and ANLS are diagnostics only.  The selector never
uses reference answers or continuation ANLS.  A final endpoint must still be
an unforced PNG replay with Full text exact and FastV ANLS < 0.5.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v13_branch_reachability_probe"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from adapters.runtime import clear_hooks, install  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from run_eval import decode  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash  # noqa: E402
from week2_attack_v4_answer_aligned.forward import keep_index_from_kept  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project  # noqa: E402
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (  # noqa: E402
    MAX_NEW_TOKENS,
    cache_replay,
    hard_margin_row,
)
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d  # noqa: E402
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base  # noqa: E402


RATIO = 0.25
EPS = 8.0 / 255.0
PROBE_ALPHA = 1.0 / 255.0
TAU = 0.0
STEPS = 25
CHECKPOINTS = (0, 5, 10, 15, 20, 25)
SEED = 2026101301
MAX_CANDIDATES = 3
TOPK_PER_POSITION = 32
LAMBDA_FULL = base.LAMBDA_FULL
ADAM_LR = base.ADAM_LR
VARIANTS = ("SEQ", "NAIVE", "AWARE")


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VATTACK_ATTN_IMPLEMENTATION", "sdpa")
    os.environ.setdefault("VATTACK_DETERMINISTIC_RESIZE", "1")
    os.environ.setdefault("VATTACK_RESIZE_MODE", "bilinear")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_piece(processor, token_id: int) -> str:
    return processor.tokenizer.decode([int(token_id)], skip_special_tokens=False)


def is_content_piece(piece: str, token_id: int, processor) -> bool:
    if int(token_id) in set(getattr(processor.tokenizer, "all_special_ids", [])):
        return False
    stripped = piece.strip()
    # Deliberately lexical, not an external NER model: candidate selection must
    # remain target-internal and cannot use labels or a language-model oracle.
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", stripped))


def piece_priority(piece: str) -> int:
    return 2 if re.search(r"[0-9]", piece) else 1


def lexical_key(text: str) -> str:
    """Label-free equivalence used to reject case/punctuation-only branches."""
    return "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]", text.lower()))


def is_meaningful_competitor(target_piece: str, candidate_piece: str, candidate_id: int, processor) -> bool:
    if not is_content_piece(candidate_piece, candidate_id, processor):
        return False
    key = lexical_key(candidate_piece)
    target_key = lexical_key(target_piece)
    if not key or key == target_key:
        return False
    # A bare one-letter alphabetic continuation is the common source of
    # harmless case/subword alternatives (e.g. PP -> P -> pbs).  Digits stay
    # eligible because a single digit can change a document answer materially.
    return bool(re.search(r"[0-9\u4e00-\u9fff]", key) or len(key) >= 2)


def replace_aux(inputs: dict, item: dict, page: torch.Tensor) -> dict:
    local = dict(inputs)
    local["pixel_values"] = v7d.pixels_from_page(inputs, item["aux_idx"], page).detach()
    return local


def clean_full(model, processor, item: dict, state: dict) -> dict:
    tokens_a, _ = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], None)
    tokens_b, _ = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], None)
    text_a = decode(processor, torch.tensor(tokens_a)).strip()
    text_b = decode(processor, torch.tensor(tokens_b)).strip()
    return {
        "text_a": text_a,
        "text_b": text_b,
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "text_exact": bool(text_a == state["clean_text"] and text_b == state["clean_text"]),
        "token_exact": bool(tokens_a == state["clean_tokens"] and tokens_b == state["clean_tokens"]),
        "replay": bool(tokens_a == tokens_b and text_a == text_b),
    }


def fast_clean_reference(model, processor, item: dict, state: dict) -> dict:
    tokens_a, kept_a = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], RATIO)
    tokens_b, kept_b = v7d.generate_tokens(model, state["inputs"], state["inputs"]["pixel_values"], state["spans"], RATIO)
    text_a = decode(processor, torch.tensor(tokens_a)).strip()
    text_b = decode(processor, torch.tensor(tokens_b)).strip()
    anls = anls_score(text_a, list(item["meta"]["answers"]))
    return {
        "tokens": tokens_a,
        "text": text_a,
        "anls": anls,
        "valid": bool(output_valid(text_a)),
        "replay": bool(tokens_a == tokens_b and text_a == text_b),
        "keep_hash": keep_hash(kept_a),
        "k": len(kept_a),
        "second_keep_hash": keep_hash(kept_b),
        "second_k": len(kept_b),
    }


def fast_rows_for_reference(model, state: dict, item: dict, page: torch.Tensor, reference: list[int]):
    pixels = v7d.pixels_from_page(state["inputs"], item["aux_idx"], page)
    generated, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], RATIO)
    keep = keep_index_from_kept(kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device)
    rows = v7d.fastv_rows(model, state["inputs"], pixels, reference, keep)
    return rows, generated, kept


def full_cache_hard(model, state: dict, item: dict, page: torch.Tensor) -> list[float]:
    """Actual Full cache path evaluated on the clean Full token sequence.

    This is a local safety diagnostic, not a certificate: exact Full generate
    remains the acceptance guard below.
    """
    local = replace_aux(state["inputs"], item, page)
    with torch.inference_mode():
        rows, _ = cache_replay(model, local, state["clean_tokens"])
        return [float(hard_margin_row(row, token).detach().cpu()) for row, token in zip(rows, state["clean_tokens"])]


def forced_fastv_continuation(model, processor, state: dict, prefix: list[int]) -> dict:
    """Conditionally complete a branch for screening only; never an endpoint."""
    local = dict(state["inputs"])
    device = local["input_ids"].device
    added = torch.tensor(prefix, dtype=local["input_ids"].dtype, device=device).view(1, -1)
    prompt_len = int(local["input_ids"].shape[1])
    local["input_ids"] = torch.cat([local["input_ids"], added], dim=1)
    if local.get("attention_mask") is not None:
        extension = torch.ones((local["attention_mask"].shape[0], len(prefix)), dtype=local["attention_mask"].dtype, device=device)
        local["attention_mask"] = torch.cat([local["attention_mask"], extension], dim=1)
    clear_hooks(model)
    hook_state = install(model, "fastv", RATIO, int(model.config.image_token_index))
    hook_state.spans = state["spans"]
    try:
        with torch.inference_mode():
            output = model.generate(
                **local,
                do_sample=False,
                max_new_tokens=max(1, MAX_NEW_TOKENS - len(prefix)),
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
            )
        all_answer = [int(x) for x in output.sequences[0, prompt_len:].detach().cpu().tolist()]
        return {
            "tokens": all_answer,
            "text": decode(processor, torch.tensor(all_answer)).strip(),
            "keep_hash": keep_hash(list(hook_state.kept_visual)),
            "k": len(hook_state.kept_visual),
        }
    finally:
        clear_hooks(model)


def candidate_pool(model, processor, item: dict, state: dict, clean_fast: dict) -> list[dict]:
    """Build at most three lexical content competitors without answer labels."""
    reference = clean_fast["tokens"]
    leaf = state["page"].detach().clone().requires_grad_(True)
    rows, generated, kept = fast_rows_for_reference(model, state, item, leaf, reference)
    if generated != reference:
        raise RuntimeError("clean FastV reference diverged before candidate discovery")
    candidates = []
    special = set(getattr(processor.tokenizer, "all_special_ids", []))
    for t, (row, target) in enumerate(zip(rows, reference)):
        target_piece = token_piece(processor, target)
        if not is_content_piece(target_piece, target, processor):
            continue
        values, ids = torch.topk(row.detach(), TOPK_PER_POSITION)
        for rank, (value, candidate) in enumerate(zip(values.tolist(), ids.tolist())):
            if int(candidate) == int(target) or int(candidate) in special:
                continue
            candidate_piece = token_piece(processor, candidate)
            if not is_meaningful_competitor(target_piece, candidate_piece, candidate, processor):
                continue
            gap = float((row[int(target)] - row[int(candidate)]).detach().cpu())
            if not math.isfinite(gap) or gap <= 0:
                continue
            candidates.append({
                "position": t,
                "target_id": int(target),
                "target_piece": target_piece,
                "candidate_id": int(candidate),
                "candidate_piece": candidate_piece,
                "candidate_rank_at_position": rank,
                "clean_fast_gap": gap,
                "content_priority": piece_priority(target_piece),
            })
            break
    # At most one high-probability competitor per content position.  No ANLS,
    # ground-truth, or forced-continuation result appears in this ranking.
    candidates.sort(key=lambda row: (-row["content_priority"], row["clean_fast_gap"], row["position"]))
    return candidates[:MAX_CANDIDATES]


def estimate_candidate(model, processor, item: dict, state: dict, clean_fast: dict, raw: dict) -> dict:
    """Estimate a branch's L-infinity FastV flip radius and Full capacity."""
    reference = clean_fast["tokens"]
    leaf = state["page"].detach().clone().requires_grad_(True)
    rows, generated, kept = fast_rows_for_reference(model, state, item, leaf, reference)
    if generated != reference:
        raise RuntimeError("clean FastV reference diverged during reachability estimate")
    t, y, c = raw["position"], raw["target_id"], raw["candidate_id"]
    gap = rows[t][y] - rows[t][c]
    gap.backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError("candidate pair margin has no finite auxiliary gradient")
    grad = leaf.grad.detach().clone()
    grad_l1 = float(grad.float().abs().sum())
    need_fast = float(gap.detach()) / max(grad_l1, 1e-12)
    direction = -grad.sign()
    clean_hard = full_cache_hard(model, state, item, state["page"])
    probe_page = state["page"] + PROBE_ALPHA * direction
    with torch.no_grad():
        probe_page = (state["page"] + (probe_page - state["page"]).clamp(-EPS, EPS)).clamp(0, 1)
    probe_hard = full_cache_hard(model, state, item, probe_page)
    slopes = [(before - after) / PROBE_ALPHA for before, after in zip(clean_hard, probe_hard)]
    capacities = [margin / slope for margin, slope in zip(clean_hard, slopes) if margin > 0 and slope > 0]
    # A nonpositive clean Full cache margin means this local boundary proxy is
    # unavailable.  We do not manufacture a safety score from its absolute
    # value; exact Full generation remains meaningful but cannot justify this
    # selection estimate.
    local_full_available = bool(clean_hard) and all(value > 0 for value in clean_hard) and bool(capacities)
    full_capacity = min(capacities) if local_full_available else 0.0
    ratio = full_capacity / need_fast if need_fast > 0 else 0.0
    prefix = reference[:t] + [c]
    continuation = forced_fastv_continuation(model, processor, state, prefix)
    result = dict(raw)
    result.update({
        "fast_gradient_l1": grad_l1,
        "fast_gradient_l2": float(grad.float().norm()),
        "estimated_fast_flip_linf": need_fast,
        "full_clean_hard_min": min(clean_hard) if clean_hard else None,
        "full_probe_hard_min": min(probe_hard) if probe_hard else None,
        "full_local_capacity_linf": full_capacity,
        "full_local_margin_available": local_full_available,
        "reachability_ratio": ratio,
        "estimated_feasible_at_eps": bool(local_full_available and need_fast <= EPS and need_fast < full_capacity),
        "clean_keep_hash": keep_hash(kept),
        "clean_k": len(kept),
        "forced_continuation": continuation,
        "forced_continuation_anls_diagnostic": anls_score(continuation["text"], list(item["meta"]["answers"])),
        "continuation_lexically_equivalent_to_clean": lexical_key(continuation["text"]) == lexical_key(clean_fast["text"]),
        "selection_uses_anls": False,
    })
    del rows, generated, kept, grad, leaf
    torch.cuda.empty_cache()
    return result


def select_candidate(candidates: list[dict], variant: str) -> dict | None:
    feasible = [row for row in candidates if row["estimated_feasible_at_eps"]]
    if not feasible:
        return None
    if variant == "NAIVE":
        return sorted(feasible, key=lambda row: (row["estimated_fast_flip_linf"], row["position"], row["candidate_id"]))[0]
    if variant == "AWARE":
        return sorted(feasible, key=lambda row: (-row["reachability_ratio"], row["estimated_fast_flip_linf"], row["position"], row["candidate_id"]))[0]
    raise ValueError(variant)


def branch_loss_and_grad(model, state: dict, item: dict, delta: torch.Tensor, reference: list[int], choice: dict) -> tuple[torch.Tensor, dict]:
    delta.grad = None
    page = state["page"] + delta
    rows, generated, kept = fast_rows_for_reference(model, state, item, page, reference)
    t, y, c = choice["position"], choice["target_id"], choice["candidate_id"]
    pair_gap = rows[t][y] - rows[t][c]
    attack_loss = torch.relu(pair_gap + TAU)
    attack_loss.backward()
    if delta.grad is None or not torch.isfinite(delta.grad).all():
        raise RuntimeError("non-finite branch attack gradient")
    attack = delta.grad.detach().clone()
    delta.grad = None
    fdp, full_proxy_violated = base.fdp_value(model, state, item, page)
    (LAMBDA_FULL * fdp).backward()
    if delta.grad is None or not torch.isfinite(delta.grad).all():
        raise RuntimeError("non-finite Full gradient")
    full = delta.grad.detach().clone()
    delta.grad = None
    return attack + full, {
        "pair_gap": float(pair_gap.detach()),
        "branch_loss": float(attack_loss.detach()),
        "fdp": float(fdp.detach()),
        "attack_l2": float(attack.float().norm()),
        "full_l2": float(full.float().norm()),
        "full_proxy_violated": bool(full_proxy_violated),
        "current_keep_hash": keep_hash(kept),
        "current_k": len(kept),
        "current_fast_tokens": generated,
    }


def sequence_loss_and_grad(model, state: dict, item: dict, delta: torch.Tensor, reference: list[int]) -> tuple[torch.Tensor, dict]:
    delta.grad = None
    page = state["page"] + delta
    rows, generated, kept = fast_rows_for_reference(model, state, item, page, reference)
    attack_loss = v7d.mean_soft(rows, reference)
    attack_loss.backward()
    if delta.grad is None or not torch.isfinite(delta.grad).all():
        raise RuntimeError("non-finite sequence attack gradient")
    attack = delta.grad.detach().clone()
    delta.grad = None
    fdp, full_proxy_violated = base.fdp_value(model, state, item, page)
    (LAMBDA_FULL * fdp).backward()
    if delta.grad is None or not torch.isfinite(delta.grad).all():
        raise RuntimeError("non-finite Full gradient")
    full = delta.grad.detach().clone()
    delta.grad = None
    return attack + full, {
        "fixed_sequence_margin": float(attack_loss.detach()),
        "fdp": float(fdp.detach()),
        "attack_l2": float(attack.float().norm()),
        "full_l2": float(full.float().norm()),
        "full_proxy_violated": bool(full_proxy_violated),
        "current_keep_hash": keep_hash(kept),
        "current_k": len(kept),
        "current_fast_tokens": generated,
    }


def score_checkpoint(model, processor, item: dict, state: dict, image: Image.Image, clean_fast: dict, choice: dict | None) -> dict:
    clean = item["images"][item["aux_idx"]]
    linf = int(np.max(np.abs(np.asarray(image, dtype=np.int16) - np.asarray(clean, dtype=np.int16))))
    full = base.full_exact_check(model, processor, item, state, image)
    attacked = list(item["images"])
    attacked[item["aux_idx"]] = image
    inputs = v7d.prepare(processor, attacked, item["meta"]["question"])
    spans = v7d.spans_of(inputs, int(model.config.image_token_index), len(attacked))
    fast_tokens, kept = v7d.generate_tokens(model, inputs, inputs["pixel_values"], spans, RATIO)
    fast_text = decode(processor, torch.tensor(fast_tokens)).strip()
    endpoint_failure = bool(output_valid(fast_text) and anls_score(fast_text, list(item["meta"]["answers"])) < 0.5)
    result = {
        "full": full,
        "fast": {
            "tokens": fast_tokens,
            "text": fast_text,
            "anls": anls_score(fast_text, list(item["meta"]["answers"])),
            "valid": bool(output_valid(fast_text)),
            "k": len(kept),
            "keep_hash": keep_hash(kept),
            "endpoint_failure": endpoint_failure,
        },
        "nonzero": bool(linf > 0),
        "linf_uint8": linf,
        "linf_ok": bool(0 < linf <= 8),
    }
    if choice is not None:
        page = v7d.page_to_float(image).to("cuda")
        rows, _, _ = fast_rows_for_reference(model, {"inputs": inputs, "spans": spans}, item, page, clean_fast["tokens"])
        result["selected_pair_gap"] = float((rows[choice["position"]][choice["target_id"]] - rows[choice["position"]][choice["candidate_id"]]).detach())
        del rows
    return result


def legal_endpoint(row: dict, clean_fast: dict) -> bool:
    return bool(
        clean_fast["anls"] >= 0.5
        and clean_fast["valid"]
        and row["full"]["full_text_exact"]
        and row["full"]["full_token_exact"]
        and row["full"]["full_replay"]
        and row["fast"]["valid"]
        and row["fast"]["endpoint_failure"]
        and row["nonzero"]
        and row["linf_ok"]
    )


def load_or_screen(model, processor, sample: str, out: Path) -> dict:
    item = v7d.load_dev(sample)
    state = v7d.prepare_item(model, processor, item)
    full = clean_full(model, processor, item, state)
    fast = fast_clean_reference(model, processor, item, state)
    report = {
        "sample_id": sample,
        "ratio": RATIO,
        "full_clean": full,
        "fast_clean": fast,
        "clean_admitted": bool(full["text_exact"] and full["token_exact"] and full["replay"] and fast["replay"] and fast["valid"] and fast["anls"] >= 0.5),
        "candidate_limit": MAX_CANDIDATES,
        "candidate_selector_uses_labels": False,
    }
    if report["clean_admitted"]:
        raw = candidate_pool(model, processor, item, state, fast)
        estimated = [estimate_candidate(model, processor, item, state, fast, candidate) for candidate in raw]
        # This still does not use answers: it drops continuations that return
        # the same normalized answer via different capitalization/punctuation.
        report["candidates"] = [row for row in estimated if not row["continuation_lexically_equivalent_to_clean"]]
        report["naive_choice"] = select_candidate(report["candidates"], "NAIVE")
        report["aware_choice"] = select_candidate(report["candidates"], "AWARE")
        report["branch_admitted"] = bool(report["naive_choice"] and report["aware_choice"])
    else:
        report["candidates"] = []
        report["naive_choice"] = None
        report["aware_choice"] = None
        report["branch_admitted"] = False
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_variant(model, processor, sample: str, variant: str, steps: int, tag: str) -> dict:
    out = HERE / "runs" / tag / sample / variant
    screen = json.loads((HERE / "runs" / tag / sample / "screen.json").read_text())
    if not screen.get("clean_admitted"):
        payload = {"status": "BLOCKED_CLEAN_GATE", "sample_id": sample, "variant": variant, "screen": screen}
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    if variant != "SEQ" and not screen.get("branch_admitted"):
        payload = {"status": "BLOCKED_NO_FEASIBLE_BRANCH", "sample_id": sample, "variant": variant, "screen": screen}
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    item = v7d.load_dev(sample)
    state = v7d.prepare_item(model, processor, item)
    clean_fast = fast_clean_reference(model, processor, item, state)
    reference = clean_fast["tokens"]
    choice = None if variant == "SEQ" else screen[f"{variant.lower()}_choice"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED % (2**32))
    delta = torch.zeros_like(state["page"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=ADAM_LR)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = {step for step in CHECKPOINTS if step <= steps}
    checkpoints.add(steps)
    curve, updates = [], []
    started = time.perf_counter()
    base.reset_counters()
    for step in range(steps + 1):
        if step in checkpoints:
            image = page_uint8(state["page"], delta)
            path = out / f"step_{step:03d}.png"
            image.save(path)
            row = score_checkpoint(model, processor, item, state, Image.open(path).convert("RGB"), clean_fast, choice)
            row.update({"step": step, "png_sha256": sha_file(path)})
            row["strict_success"] = legal_endpoint(row, clean_fast)
            curve.append(row)
        if step == steps:
            break
        if variant == "SEQ":
            direction, meta = sequence_loss_and_grad(model, state, item, delta, reference)
        else:
            direction, meta = branch_loss_and_grad(model, state, item, delta, reference, choice)
        before_delta = delta.detach().clone()
        before_opt = copy.deepcopy(opt.state_dict())
        delta.grad = direction
        opt.step()
        project(delta, state["page"], EPS)
        delta.grad = None
        guarded = base.full_exact_check(model, processor, item, state, page_uint8(state["page"], delta))
        rolled_back = not bool(guarded["full_text_exact"] and guarded["full_token_exact"] and guarded["full_replay"])
        if rolled_back:
            delta.data.copy_(before_delta)
            opt.load_state_dict(before_opt)
        updates.append({"step": step, **meta, "post_update_full": guarded, "rolled_back": rolled_back})
        torch.cuda.empty_cache()
    strict_steps = [row["step"] for row in curve if row["strict_success"]]
    payload = {
        "status": "DONE",
        "sample_id": sample,
        "variant": variant,
        "ratio": RATIO,
        "steps": steps,
        "choice": choice,
        "selector": "fixed_clean_sequence" if variant == "SEQ" else ("lowest estimated FastV flip radius" if variant == "NAIVE" else "maximum Full capacity / FastV flip radius"),
        "strict_success_any": bool(strict_steps),
        "first_strict_success_step": strict_steps[0] if strict_steps else None,
        "curve": curve,
        "updates": updates,
        "counters": base.counters(),
        "elapsed_sec": time.perf_counter() - started,
        "full_guard": "two real Full generate calls after every update; rejected update restores delta and Adam state",
    }
    (out / "trajectory.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_protocol(tag: str, samples: list[str], steps: int) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "CONSTRUCTION_PROBE_NOT_METHOD_RESULT",
        "tag": tag,
        "samples": samples,
        "ratio": RATIO,
        "steps": steps,
        "epsilon_uint8": 8,
        "max_candidates_per_sample": MAX_CANDIDATES,
        "variants": {
            "SEQ": "fixed whole clean FastV sequence plus Native-FDP",
            "NAIVE": "same candidate pool; choose minimum estimated FastV flip radius",
            "AWARE": "same candidate pool; choose maximum Full local capacity divided by FastV flip radius",
        },
        "candidate_filter": "lexical content tokens only; excludes special tokens, punctuation-only pieces, and EOS; no labels",
        "continuation": "forced continuation is screening diagnostic only and cannot be an endpoint",
        "full_capacity": "one-sided local cache-margin estimate along negative FastV pair-gradient sign; unavailable if any clean Full cache margin is nonpositive",
        "endpoint": "unforced saved/reloaded PNG must yield Full text/token exact and replay plus valid FastV ANLS<0.5",
        "selector_uses_answer_labels": False,
        "equality": "same r=0.25, epsilon, Adam, lambda_full, Full guard, update count, and checkpoint schedule across three variants; screening compute is logged separately",
        "stop_condition": "If either clean gate or branch-admission fails, record BLOCKED and do not invent a substitute candidate.",
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("screen", "run"))
    parser.add_argument("--sample", required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--tag", default="construct_v1")
    args = parser.parse_args()
    configure_determinism()
    write_protocol(args.tag, [args.sample], args.steps)
    model, processor = base.load_model_frozen()
    base.install_counters(model)
    sample_root = HERE / "runs" / args.tag / args.sample
    if args.mode == "screen":
        started = time.perf_counter()
        base.reset_counters()
        report = load_or_screen(model, processor, args.sample, sample_root)
        report["screen_counters"] = base.counters()
        report["screen_elapsed_sec"] = time.perf_counter() - started
        (sample_root / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "SCREEN_DONE", "sample": args.sample, "clean_admitted": report["clean_admitted"], "branch_admitted": report["branch_admitted"]}), flush=True)
        return
    if args.variant is None:
        raise SystemExit("--variant is required for mode=run")
    result = run_variant(model, processor, args.sample, args.variant, args.steps, args.tag)
    print(json.dumps({"status": result["status"], "sample": args.sample, "variant": args.variant, "strict_success_any": result.get("strict_success_any")}), flush=True)


if __name__ == "__main__":
    main()
