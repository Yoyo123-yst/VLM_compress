"""GPU health for FDP. Does not run PGD and does not write historical trees."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7_budget_conditioned"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks  # noqa: E402
from run_causal_v2 import decode, generate, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.common import keep_hash, spans_for  # noqa: E402
from week2_attack_v4_answer_aligned.forward import (  # noqa: E402
    build_embeds,
    full_answer_logits,
    hook_kept,
)
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402
from week2_attack_v7_budget_conditioned.constants import CANDIDATE_R, HEALTH_ID  # noqa: E402
from week2_attack_v7_budget_conditioned.fdp import certificate, fdp_loss, greedy_tokens, hard_margin, soft_margin  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import historical_1025, load_model_frozen, make_session  # noqa: E402
from week2_pilot.attack import normalized_tiles  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    model, processor = load_model_frozen()
    item = historical_1025()
    if item["meta"]["sample_id"] != HEALTH_ID:
        raise SystemExit("health sample is not 1025")
    session = make_session(model, processor, item)
    clear_hooks(model)
    inputs = prepare(processor, item["images"], item["meta"]["question"])
    with torch.inference_mode():
        raw = generate(model, inputs)
    generated = [int(x) for x in raw.detach().cpu().tolist()]
    tokens = [int(x) for x in session.answer_tokens.detach().cpu().tolist()]
    eos = int(model.generation_config.eos_token_id)
    with torch.no_grad():
        logits = full_answer_logits(
            model, session.embeds_from(session.clean_page), session.prompt_len - 1, int(session.answer_tokens.numel())
        ).float()
        hard = hard_margin(logits, session.answer_tokens)
        soft = soft_margin(logits, session.answer_tokens)
        loss = fdp_loss(soft, soft)
        argmax = [int(x) for x in greedy_tokens(logits).detach().cpu().tolist()]
    clean_soft = soft.detach()
    aux = session.clean_page.detach().clone().requires_grad_(True)
    pixels = session.pixels.detach().clone().requires_grad_(True)
    tiles = normalized_tiles(aux).to(dtype=pixels.dtype).unsqueeze(0)
    embeds = build_embeds(
        model, session.full_ids, session.mask, pixels, session.sizes, session.aux_idx, tiles
    )
    adv_logits = full_answer_logits(model, embeds, session.prompt_len - 1, int(session.answer_tokens.numel())).float()
    adv_soft = soft_margin(adv_logits, session.answer_tokens)
    fdp_loss(clean_soft, adv_soft).backward()
    evidence = item["evidence_idx"]
    evidence_ids = [evidence] if isinstance(evidence, int) else list(evidence)
    evidence_ids = [i for i in evidence_ids if i != item["aux_idx"]]
    evidence_norm = 0.0 if pixels.grad is None else float(pixels.grad[evidence_ids].float().norm())
    aux_norm = 0.0 if aux.grad is None else float(aux.grad.float().norm())
    spans = spans_for(model, inputs, len(item["images"]))
    n_visual = sum(end - start for start, end in spans)
    ratio_rows = []
    removed = []
    for ratio in CANDIDATE_R:
        row = {"r": ratio}
        try:
            kept = hook_kept(model, inputs, spans, ratio)
            expected = official_keep_count(n_visual, ratio)
            row.update({"k": len(kept), "expected_k": expected, "ok": len(kept) == expected and 1 <= len(kept) <= n_visual})
        except Exception as exc:
            row.update({"ok": False, "error": str(exc)})
        if not row["ok"]:
            removed.append(ratio)
        ratio_rows.append(row)
    text_generated = decode(processor, torch.tensor(generated))
    text_session = decode(processor, session.answer_tokens)
    report = {
        "sample_id": HEALTH_ID,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "includes_eos": tokens[-1] == eos,
        "generate_ends_with_eos": generated[-1] == eos if generated else False,
        "teacher_forced_argmax_matches_tokens": argmax == tokens,
        "generate_text_matches_session_text": text_generated == text_session,
        "certificate": certificate(hard),
        "evidence_grad_l2": evidence_norm,
        "aux_grad_l2": aux_norm,
        "aux_grad_finite": aux.grad is not None and bool(torch.isfinite(aux.grad).all().item()),
        "ratio_rows": ratio_rows,
        "removed_r": removed,
        "keep_hash_r025": None,
    }
    kept_025 = hook_kept(model, inputs, spans, 0.25)
    report["keep_hash_r025"] = keep_hash(kept_025)
    report["k_r025"] = len(kept_025)
    report["pass"] = bool(
        report["loss_finite"]
        and report["includes_eos"]
        and report["teacher_forced_argmax_matches_tokens"]
        and report["generate_text_matches_session_text"]
        and evidence_norm == 0.0
        and report["aux_grad_finite"]
        and aux_norm > 0.0
        and 0.25 not in removed
    )
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "health_1025.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "ratio_legality.json").write_text(json.dumps({"rows": ratio_rows, "removed_r": removed}, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "aux_grad": aux_norm, "evidence_grad": evidence_norm, "removed": removed, "certificate": report["certificate"]}))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
