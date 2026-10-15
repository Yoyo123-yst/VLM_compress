"""Generate and freeze clean-only complete answer candidates.

This is a candidate-admission stage only. It never updates pixels and never
reads an attacked PNG. Candidate ranking uses clean FastV logits only; answer
labels are retained for later semantic review, not for candidate selection.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "week2_attack_v25_full_answer_competition_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]

from week2_attack_v13_branch_reachability_probe import run_branch_probe as probe
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d


TOPK_PER_POSITION = 32
COMPETITORS_PER_POSITION = 6
MAX_RAW_BRANCHES = 24
MAX_COMPLETE_CANDIDATES = 6


def anchored_pixels_from_page(state: dict, item: dict, page: torch.Tensor) -> torch.Tensor:
    """Differentiable page splice whose zero point is the real processor tensor.

    `normalized_tiles(clean_page)` is close to, but not bit-identical with,
    the processor output.  Replacing the processor tiles outright therefore
    changes the FastV keep-set at zero perturbation.  Anchoring only the
    differentiable tile delta to the processor tensor keeps the exact clean
    forward while preserving gradients for pixel updates.
    """
    clean_page = state["page"].detach()
    tile_delta = v7d.normalized_tiles(page) - v7d.normalized_tiles(clean_page)
    base_tiles = state["inputs"]["pixel_values"][item["aux_idx"]].detach()
    anchored_tiles = base_tiles + tile_delta.to(dtype=base_tiles.dtype)
    return v7d.splice_aux(state["inputs"]["pixel_values"], item["aux_idx"], anchored_tiles)


def fast_rows_for_reference_anchored(model, state: dict, item: dict, page: torch.Tensor, reference: list[int]):
    pixels = anchored_pixels_from_page(state, item, page)
    generated, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], probe.RATIO)
    keep = probe.keep_index_from_kept(
        kept, state["spans"], int(state["inputs"]["input_ids"].shape[1]), page.device
    )
    rows = v7d.fastv_rows(model, state["inputs"], pixels, reference, keep)
    return rows, generated, kept


def full_cache_hard_anchored(model, state: dict, item: dict, page: torch.Tensor) -> list[float]:
    local = dict(state["inputs"])
    local["pixel_values"] = anchored_pixels_from_page(state, item, page).detach()
    with torch.inference_mode():
        rows, _ = probe.cache_replay(model, local, state["clean_tokens"])
        return [
            float(probe.hard_margin_row(row, token).detach().cpu())
            for row, token in zip(rows, state["clean_tokens"])
        ]


def zero_injection_audit(model, state: dict, item: dict, fast: dict) -> dict:
    pixels = anchored_pixels_from_page(state, item, state["page"])
    pixel_exact = bool(torch.equal(pixels, state["inputs"]["pixel_values"]))
    tokens, kept = v7d.generate_tokens(model, state["inputs"], pixels, state["spans"], probe.RATIO)
    return {
        "pixel_tensor_exact": pixel_exact,
        "fast_tokens_exact": bool(tokens == fast["tokens"]),
        "fast_keep_hash": probe.keep_hash(kept),
        "reference_keep_hash": fast["keep_hash"],
        "fast_keep_exact": bool(probe.keep_hash(kept) == fast["keep_hash"]),
        "k": len(kept),
        "reference_k": fast["k"],
    }


def estimate_candidate_anchored(model, processor, item: dict, state: dict, fast: dict, raw: dict) -> dict:
    reference = fast["tokens"]
    leaf = state["page"].detach().clone().requires_grad_(True)
    rows, generated, kept = fast_rows_for_reference_anchored(model, state, item, leaf, reference)
    if generated != reference:
        raise RuntimeError("anchored clean FastV reference diverged during reachability estimate")
    position, target, candidate = raw["position"], raw["target_id"], raw["candidate_id"]
    gap = rows[position][target] - rows[position][candidate]
    gap.backward()
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError("candidate pair margin has no finite anchored auxiliary gradient")
    grad = leaf.grad.detach().clone()
    grad_l1 = float(grad.float().abs().sum())
    need_fast = float(gap.detach()) / max(grad_l1, 1e-12)
    direction = -grad.sign()
    clean_hard = full_cache_hard_anchored(model, state, item, state["page"])
    with torch.no_grad():
        probe_page = (state["page"] + probe.PROBE_ALPHA * direction).clamp(0, 1)
    probe_hard = full_cache_hard_anchored(model, state, item, probe_page)
    slopes = [(before - after) / probe.PROBE_ALPHA for before, after in zip(clean_hard, probe_hard)]
    capacities = [margin / slope for margin, slope in zip(clean_hard, slopes) if margin > 0 and slope > 0]
    local_full_available = bool(clean_hard) and all(value > 0 for value in clean_hard) and bool(capacities)
    full_capacity = min(capacities) if local_full_available else 0.0
    ratio = full_capacity / need_fast if need_fast > 0 else 0.0
    prefix = reference[:position] + [candidate]
    continuation = probe.forced_fastv_continuation(model, processor, state, prefix)
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
        "estimated_feasible_at_eps": bool(local_full_available and need_fast <= probe.EPS and need_fast < full_capacity),
        "clean_keep_hash": probe.keep_hash(kept),
        "clean_k": len(kept),
        "forced_continuation": continuation,
        "forced_continuation_anls_diagnostic": probe.anls_score(continuation["text"], list(item["meta"]["answers"])),
        "continuation_lexically_equivalent_to_clean": probe.lexical_key(continuation["text"]) == probe.lexical_key(fast["text"]),
        "selection_uses_anls": False,
        "pixel_mapping": "processor_anchored_differentiable_tile_delta_v1",
    })
    del rows, generated, kept, grad, leaf
    torch.cuda.empty_cache()
    return result


def expanded_branch_pool(model, processor, item: dict, state: dict, fast: dict) -> list[dict]:
    """Enumerate several lexical branches before looking at continuations.

    v13 retained one competitor per position and truncated the global list to
    three before completing any branch.  That procedure is suitable for its
    local reachability probe, but it is not a complete-answer candidate
    generator.  This pool deliberately stays label-free while widening the
    clean FastV branch search.
    """
    reference = fast["tokens"]
    leaf = state["page"].detach().clone().requires_grad_(True)
    rows, generated, _ = fast_rows_for_reference_anchored(model, state, item, leaf, reference)
    if generated != reference:
        raise RuntimeError("clean FastV reference diverged before expanded candidate discovery")
    special = set(getattr(processor.tokenizer, "all_special_ids", []))
    branches: list[dict] = []
    for position, (row, target) in enumerate(zip(rows, reference)):
        target_piece = probe.token_piece(processor, target)
        if not probe.is_content_piece(target_piece, target, processor):
            continue
        values, ids = torch.topk(row.detach(), TOPK_PER_POSITION)
        kept_here = 0
        for rank, candidate in enumerate(ids.tolist()):
            candidate = int(candidate)
            if candidate == int(target) or candidate in special:
                continue
            candidate_piece = probe.token_piece(processor, candidate)
            if not probe.is_meaningful_competitor(target_piece, candidate_piece, candidate, processor):
                continue
            gap = float((row[int(target)] - row[candidate]).detach().cpu())
            if not math.isfinite(gap) or gap <= 0:
                continue
            branches.append({
                "position": position,
                "target_id": int(target),
                "target_piece": target_piece,
                "candidate_id": candidate,
                "candidate_piece": candidate_piece,
                "candidate_rank_at_position": rank,
                "clean_fast_gap": gap,
                "content_priority": probe.piece_priority(target_piece),
            })
            kept_here += 1
            if kept_here >= COMPETITORS_PER_POSITION:
                break
    del rows, leaf
    # Rank only by clean model quantities.  Continuation text and labels do not
    # influence which branches are executed.
    branches.sort(key=lambda row: (-row["content_priority"], row["clean_fast_gap"], row["position"], row["candidate_id"]))
    return branches[:MAX_RAW_BRANCHES]


def complete_branch_screen(model, processor, state: dict, fast: dict, branches: list[dict]) -> tuple[list[dict], list[dict]]:
    """Complete branches, reject obvious non-answers, and deduplicate text."""
    clean_key = probe.lexical_key(fast["text"])
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for raw in branches:
        prefix = fast["tokens"][: raw["position"]] + [raw["candidate_id"]]
        continuation = probe.forced_fastv_continuation(model, processor, state, prefix)
        text = continuation["text"]
        key = probe.lexical_key(text)
        lexical_similarity = difflib.SequenceMatcher(None, clean_key, key).ratio() if clean_key or key else 1.0
        rejection = None
        if not probe.output_valid(text):
            rejection = "INVALID_OUTPUT"
        elif not key:
            rejection = "EMPTY_NORMALIZED_OUTPUT"
        elif key == clean_key:
            rejection = "LEXICALLY_EQUIVALENT"
        elif key in seen:
            rejection = "DUPLICATE_COMPLETE_TEXT"
        elif len(key) < max(3, int(math.ceil(0.45 * max(1, len(clean_key))))):
            rejection = "LIKELY_TRUNCATED_OUTPUT"
        record = dict(raw)
        record.update({
            "forced_continuation": continuation,
            "complete_lexical_key": key,
            "complete_lexical_similarity_to_clean": lexical_similarity,
            "screen_rejection": rejection,
        })
        if rejection is None:
            seen.add(key)
            accepted.append(record)
        else:
            rejected.append(record)
    # Prefer plausible high-probability branches, using sequence divergence
    # only as a tie-breaker.  Ground-truth answers and ANLS are absent here.
    accepted.sort(key=lambda row: (
        row["clean_fast_gap"],
        row["complete_lexical_similarity_to_clean"],
        row["position"],
        row["candidate_id"],
    ))
    return accepted[:MAX_COMPLETE_CANDIDATES], rejected


def image_sha(image) -> str:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build(sample: str, device: str) -> dict:
    probe.configure_determinism()
    device_index = int(device)
    torch.cuda.set_device(device_index)
    model, processor = base.load_model_frozen()
    model.eval()
    item = v7d.load_dev(sample)
    state = v7d.prepare_item(model, processor, item)
    full = probe.clean_full(model, processor, item, state)
    fast = probe.fast_clean_reference(model, processor, item, state)
    zero_audit = zero_injection_audit(model, state, item, fast)
    row = {
        "sample_id": sample,
        "ratio": probe.RATIO,
        "device": device_index,
        "source": "clean_dataset_page_only",
        "clean_aux_png_sha256": image_sha(item["images"][item["aux_idx"]]),
        "full_clean": full,
        "fast_clean": fast,
        "zero_injection_audit": zero_audit,
        "candidate_selector_uses_labels": False,
        "status": "CLEAN_GATE_FAILED",
        "candidates": [],
    }
    admitted = bool(
        full["text_exact"] and full["token_exact"] and full["replay"]
        and fast["replay"] and fast["valid"] and fast["anls"] >= 0.5
        and zero_audit["pixel_tensor_exact"] and zero_audit["fast_tokens_exact"]
        and zero_audit["fast_keep_exact"] and zero_audit["k"] == zero_audit["reference_k"]
    )
    if admitted:
        raw = expanded_branch_pool(model, processor, item, state, fast)
        screened, rejected = complete_branch_screen(model, processor, state, fast, raw)
        estimated = []
        for candidate in screened:
            screening = {
                "complete_lexical_key": candidate["complete_lexical_key"],
                "complete_lexical_similarity_to_clean": candidate["complete_lexical_similarity_to_clean"],
                "screen_rejection": candidate["screen_rejection"],
                "screen_forced_continuation": candidate["forced_continuation"],
            }
            raw_candidate = {key: candidate[key] for key in (
                "position", "target_id", "target_piece", "candidate_id",
                "candidate_piece", "candidate_rank_at_position",
                "clean_fast_gap", "content_priority",
            )}
            result = estimate_candidate_anchored(model, processor, item, state, fast, raw_candidate)
            result.update(screening)
            estimated.append(result)
        for c in estimated:
            c["candidate_source"] = "clean_fastv_expanded_branch_complete_continuation_v2"
            c["attack_png_used"] = False
            c["semantic_review"] = "PENDING_MANUAL_REVIEW"
        row["candidate_protocol"] = {
            "version": 2,
            "topk_per_position": TOPK_PER_POSITION,
            "competitors_per_position": COMPETITORS_PER_POSITION,
            "max_raw_branches": MAX_RAW_BRANCHES,
            "max_complete_candidates": MAX_COMPLETE_CANDIDATES,
            "ground_truth_used_for_selection": False,
            "anls_used_for_selection": False,
        }
        row["proposal_counts"] = {
            "raw_branches": len(raw),
            "screen_admitted_after_limit": len(screened),
            "screen_rejected": len(rejected),
            "estimated": len(estimated),
        }
        row["rejected_proposals"] = rejected
        row["status"] = "CLEAN_ADMITTED_CANDIDATES_PENDING_REVIEW" if estimated else "NO_COMPLETE_CANDIDATE_AFTER_SCREEN"
        row["candidates"] = estimated
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else HERE / "candidate_manifest" / f"{args.sample}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = build(args.sample, args.device)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": args.sample, "status": result["status"], "candidate_count": len(result["candidates"]), "out": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
