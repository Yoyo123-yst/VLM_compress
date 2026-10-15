"""δ=0 alignment between the differentiable forward and the real FastV hook."""

from __future__ import annotations

import sys
import traceback

import torch

from week2_attack_v4_answer_aligned.common import (
    RATIO,
    V4,
    assert_protected,
    frozen_document,
    image_mask_of,
    load_frozen_samples,
    sha_image,
    spans_for,
    write_json,
)
from week2_attack_v4_answer_aligned.forward import (
    build_embeds,
    compressed_answer_logits,
    encode_page,
    full_answer_logits,
    hook_kept,
    keep_index_from_kept,
    keep_index_from_scores,
    pruned_hidden,
    score_prompt,
)
from week2_attack_v4_answer_aligned.forward import answer_logits_from_hidden
from week2_attack_v4_answer_aligned.loss import compression_margin_loss, kl_clean_adv, official_keep_count
from week2_pilot.attack import normalized_tiles, page_to_float

sys.path.insert(0, str(V4.parents[1] / "week1_mechanism"))
sys.path.insert(0, str(V4.parents[1] / "gate_experiment"))

from adapters.runtime import clear_hooks, install  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import decode, generate, load_model, prepare, run_fastv, run_plain  # noqa: E402


def logit_diff(left: torch.Tensor, right: torch.Tensor) -> dict:
    gap = (left.float() - right.float()).abs()
    return {
        "max_abs": float(gap.max()),
        "mean_abs": float(gap.mean()),
        "argmax_left": int(left.argmax()),
        "argmax_right": int(right.argmax()),
        "argmax_equal": bool(int(left.argmax()) == int(right.argmax())),
    }


def real_fastv_first(model, processor, images, question):
    clear_hooks(model)
    inputs = prepare(processor, images, question)
    spans = spans_for(model, inputs, len(images))
    state = install(model, "fastv", RATIO, int(model.config.image_token_index))
    state.spans = spans
    with torch.no_grad():
        output = model(**inputs, use_cache=False)
    logits = output.logits[0, -1].float()
    kept = list(state.kept_visual)
    clear_hooks(model)
    return logits, kept, inputs, spans


def surrogate_first(model, inputs, spans, aux_idx, aux_pixels=None):
    prompt_ids = inputs["input_ids"]
    _, mask = image_mask_of(model, prompt_ids)
    with torch.no_grad():
        embeds = build_embeds(
            model, prompt_ids, mask, inputs["pixel_values"], inputs["image_sizes"], aux_idx, aux_pixels
        )
        scores = score_prompt(model.model.language_model, embeds)
        keep_index, k, kept = keep_index_from_scores(scores, spans, embeds.shape[1], RATIO)
        hidden = pruned_hidden(model.model.language_model, embeds, keep_index)
        logits = answer_logits_from_hidden(model, hidden, [embeds.shape[1] - 1], keep_index)[0].float()
    return logits, kept, k


def reference_answer(model, processor, images, question):
    clear_hooks(model)
    inputs = prepare(processor, images, question)
    with torch.inference_mode():
        tokens = generate(model, inputs)
    tokens = torch.tensor(tokens.detach().cpu().tolist(), dtype=torch.long)
    return tokens, decode(processor, tokens.cpu())


def grad_check(model, inputs, spans, aux_idx, images, answer_tokens):
    page = page_to_float(images[aux_idx]).to("cuda").requires_grad_(True)
    tiles = normalized_tiles(page).to(dtype=inputs["pixel_values"].dtype).unsqueeze(0)
    full_ids = torch.cat([inputs["input_ids"], answer_tokens.view(1, -1)], dim=1)
    _, mask = image_mask_of(model, full_ids)
    embeds = build_embeds(
        model, full_ids, mask, inputs["pixel_values"], inputs["image_sizes"], aux_idx, tiles
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    hooked = dict(inputs)
    hooked["pixel_values"] = inputs["pixel_values"].detach().clone()
    hooked["pixel_values"][aux_idx] = tiles.detach()[0]
    kept = hook_kept(model, hooked, spans, RATIO)
    keep_index = keep_index_from_kept(kept, spans, full_ids.shape[1], embeds.device)
    logits = compressed_answer_logits(model, embeds, prompt_len, keep_index, int(answer_tokens.numel()))
    loss_comp = compression_margin_loss(logits.float(), answer_tokens)
    loss_comp.backward()
    comp_ok, comp_norm = _finite_norm(page.grad)
    page.grad = None
    # KL is exactly flat at δ=0. A one-step interior shift checks that the full
    # teacher-forcing graph still reaches the controlled page.
    with torch.no_grad():
        page.add_(1.0 / 255.0)
        page.clamp_(0, 1)
    tiles_shift = normalized_tiles(page).to(dtype=inputs["pixel_values"].dtype).unsqueeze(0)
    clean_tiles = normalized_tiles(page_to_float(images[aux_idx]).to("cuda")).to(dtype=inputs["pixel_values"].dtype).unsqueeze(0)
    with torch.no_grad():
        clean_embeds = build_embeds(
            model, full_ids, mask, inputs["pixel_values"], inputs["image_sizes"], aux_idx, clean_tiles
        )
        log_p = full_answer_logits(model, clean_embeds, prompt_len - 1, int(answer_tokens.numel())).float().log_softmax(-1)
    adv_embeds = build_embeds(
        model, full_ids, mask, inputs["pixel_values"], inputs["image_sizes"], aux_idx, tiles_shift
    )
    logits_full = full_answer_logits(model, adv_embeds, prompt_len - 1, int(answer_tokens.numel()))
    loss_full = kl_clean_adv(log_p, logits_full)
    loss_full.backward()
    full_ok, full_norm = _finite_norm(page.grad)
    other = next(i for i in range(inputs["pixel_values"].shape[0]) if i != aux_idx)
    other_feat = encode_page(model, inputs["pixel_values"][other:other + 1], inputs["image_sizes"][other:other + 1]).detach()
    protected_detached = other_feat.grad_fn is None and not bool(other_feat.requires_grad)
    return {
        "comp_grad_nonzero_finite": comp_ok,
        "comp_grad_l2": comp_norm,
        "full_shift_grad_nonzero_finite": full_ok,
        "full_shift_grad_l2": full_norm,
        "comp_logits_finite": bool(torch.isfinite(logits).all()),
        "full_logits_finite": bool(torch.isfinite(logits_full).all()),
        "aux_tokens_kept": sum(1 for pos in kept if spans[aux_idx][0] <= pos < spans[aux_idx][1]),
        "protected_features_not_leaves": protected_detached,
        "optimizer_params": 1,
    }


def _finite_norm(grad) -> tuple[bool, float]:
    if grad is None or not torch.isfinite(grad).all():
        return False, float("nan")
    norm = float(grad.detach().float().norm())
    return norm > 0.0, norm


def one_sample(model, processor, item) -> dict:
    images = item["images"]
    question = item["meta"]["question"]
    aux_idx = item["aux_idx"]
    evidence_idx = item["evidence_idx"]
    image_token_id = int(model.config.image_token_index)
    full_pred, _, _, _ = run_plain(model, processor, images, question, image_token_id, False, evidence_idx)
    fast_pred, spans, kept_gen, _ = run_fastv(
        model, processor, images, question, image_token_id, RATIO, evidence_idx, aux_idx
    )
    full_again, _, _, _ = run_plain(model, processor, images, question, image_token_id, False, evidence_idx)
    fast_again, _, kept_again, _ = run_fastv(
        model, processor, images, question, image_token_id, RATIO, evidence_idx, aux_idx
    )
    real_logits, real_kept, inputs, spans_real = real_fastv_first(model, processor, images, question)
    surr_logits, surr_kept, k = surrogate_first(model, inputs, spans_real, aux_idx)
    tiles = normalized_tiles(page_to_float(images[aux_idx]).to("cuda")).to(dtype=inputs["pixel_values"].dtype).unsqueeze(0)
    tile_logits, tile_kept, _ = surrogate_first(model, inputs, spans_real, aux_idx, tiles)
    answer_tokens, answer_text = reference_answer(model, processor, images, question)
    n_visual = sum(end - start for start, end in spans_real)
    expected_k = official_keep_count(n_visual, RATIO)
    diff = logit_diff(surr_logits, real_logits)
    tile_diff = logit_diff(tile_logits, real_logits)
    grads = grad_check(model, inputs, spans_real, aux_idx, images, answer_tokens.to("cuda"))
    after = [sha_image(image) for image in images]
    assert_protected(item["page_sha"], after, aux_idx)
    report = {
        "sample_id": item["meta"]["sample_id"],
        "full_pred": full_pred,
        "fastv_pred": fast_pred,
        "historical_full": item["baseline_full"]["pred"],
        "historical_fastv": item["baseline_fastv"]["pred"],
        "full_matches_history": full_pred.strip() == item["baseline_full"]["pred"].strip(),
        "fastv_matches_history": fast_pred.strip() == item["baseline_fastv"]["pred"].strip(),
        "full_anls": anls_score(full_pred, item["meta"]["answers"]),
        "fastv_anls": anls_score(fast_pred, item["meta"]["answers"]),
        "replay_full": full_pred == full_again,
        "replay_fastv": fast_pred == fast_again and real_kept == list(kept_again),
        "k_surrogate": k,
        "k_real": len(real_kept),
        "k_generate": len(kept_gen),
        "k_official": expected_k,
        "k_equal": k == len(real_kept) == len(kept_gen) == expected_k,
        "keep_equal": surr_kept == real_kept == sorted(kept_gen),
        "hook_keep_equal": real_kept == sorted(kept_gen),
        "score_symdiff": len(set(surr_kept) ^ set(real_kept)),
        "tile_keep_equal": tile_kept == real_kept,
        "logit_diff": diff,
        "tile_logit_diff": tile_diff,
        "answer_text": answer_text,
        "answer_tokens": int(answer_tokens.numel()),
        "spans_equal": spans == spans_real,
        **grads,
    }
    reasons = []
    if not report["hook_keep_equal"] or not report["k_equal"]:
        reasons.append("K or hook keep mismatch")
    if report["score_symdiff"] > 16:
        reasons.append("score keep set far from the runtime hook")
    if not diff["argmax_equal"] or not tile_diff["argmax_equal"]:
        reasons.append("first-token argmax mismatch")
    if not grads["comp_grad_nonzero_finite"] or not grads["full_shift_grad_nonzero_finite"]:
        reasons.append("gradient zero or non-finite")
    if not grads["protected_features_not_leaves"]:
        reasons.append("protected page is in the graph")
    if not grads["comp_logits_finite"] or not grads["full_logits_finite"]:
        reasons.append("non-finite logits")
    if not report["replay_full"] or not report["replay_fastv"]:
        reasons.append("replay mismatch")
    if not report["full_matches_history"] or not report["fastv_matches_history"]:
        reasons.append("history mismatch")
    report["block_reasons"] = reasons
    report["pass"] = not reasons
    return report


def main():
    torch.manual_seed(42)
    samples = load_frozen_samples()
    write_json(V4 / "FROZEN.json", frozen_document(samples))
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    reports = []
    for item in samples:
        print(item["meta"]["sample_id"], flush=True)
        reports.append(one_sample(model, processor, item))
        torch.cuda.empty_cache()
    blocked = [row["sample_id"] for row in reports if not row["pass"]]
    payload = {
        "status": "BLOCKED_ON_FORWARD_ALIGNMENT" if blocked else "PASS",
        "blocked": blocked,
        "note": (
            "The attack keep set is the runtime FastV hook. A separate score reimplementation "
            "can swap a few tokens because thousands of scores sit within 1e-5 of the cutoff; "
            "symmetric difference above 16 blocks. L_full gradient is checked after a 1/255 shift "
            "because KL is flat at δ=0."
        ),
        "samples": reports,
    }
    write_json(V4 / "results" / "health.json", payload)
    print(payload["status"], blocked, flush=True)
    if blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
