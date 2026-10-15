"""Native-cache margins. Logits come from the production model forward."""

from __future__ import annotations

import torch

MAX_NEW_TOKENS = 16
FASTV_RATIO = 0.25


def hard_margin_row(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("logits must be one vocabulary row")
    chosen = logits[token_id]
    rival = logits.clone()
    rival[token_id] = float("-inf")
    return chosen - rival.max()


def soft_margin_row(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("logits must be one vocabulary row")
    chosen = logits[token_id]
    rival = logits.clone()
    rival[token_id] = float("-inf")
    return chosen - torch.logsumexp(rival, dim=0)


def margins_for_tokens(rows: list[torch.Tensor], token_ids: list[int]) -> dict:
    if len(rows) != len(token_ids):
        raise ValueError(f"margin rows {len(rows)} != real tokens {len(token_ids)}")
    hard = [hard_margin_row(row, token) for row, token in zip(rows, token_ids)]
    soft = [soft_margin_row(row, token) for row, token in zip(rows, token_ids)]
    return {"hard": hard, "soft": soft}


def termination_of(token_ids: list[int], eos_token_id: int, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if not token_ids:
        return "OTHER_STOPPING_CRITERION"
    if token_ids[-1] == eos_token_id:
        return "EOS_GENERATED"
    if len(token_ids) == max_new_tokens:
        return "MAX_NEW_TOKENS_TRUNCATED"
    return "OTHER_STOPPING_CRITERION"


def cone_holds(hard_margins: list[float]) -> bool:
    if not hard_margins:
        return False
    return all(value > 0 for value in hard_margins)


def certificate_status(holds: bool, full_exact: bool) -> str:
    if full_exact and not holds:
        return "FULL_EXACT_CERTIFICATE_UNAVAILABLE"
    if holds:
        return "CERTIFICATE_AVAILABLE"
    return "CERTIFICATE_UNAVAILABLE"


def greedy_ids_if_cone(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """If every hard margin is positive, greedy ids equal the supplied ids."""
    if logits.ndim != 2 or token_ids.ndim != 1 or logits.shape[0] != token_ids.shape[0]:
        raise ValueError("logits [T,V] must align with token ids")
    hard = torch.stack([hard_margin_row(logits[i], int(token_ids[i])) for i in range(token_ids.shape[0])])
    if bool(torch.all(hard > 0)):
        return logits.argmax(dim=-1)
    return logits.argmax(dim=-1)


def require_keep_match(got_k: int | None, got_hash: str | None, ref_k: int | None, ref_hash: str | None) -> None:
    if got_k is None or got_hash is None or ref_k is None or ref_hash is None:
        raise RuntimeError("FastV K or keep hash is missing")
    if got_k != ref_k or got_hash != ref_hash:
        raise RuntimeError(f"FastV keep mismatch: {got_k}/{got_hash} != {ref_k}/{ref_hash}")


def production_generate(model, inputs: dict) -> dict:
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    tokens = [int(x) for x in output.sequences[0, prompt_len:].detach().cpu().tolist()]
    scores = [row[0].float() for row in output.scores]
    if len(scores) != len(tokens):
        raise RuntimeError(f"generate scores {len(scores)} != tokens {len(tokens)}")
    return {
        "tokens": tokens,
        "scores": scores,
        "prompt_len": prompt_len,
        "chosen": [int(row.argmax()) for row in scores],
    }


def cache_replay(model, inputs: dict, token_ids: list[int]) -> tuple[list[torch.Tensor], list[dict]]:
    """Prefill plus one cached step per generated token except the last feed."""
    if not token_ids:
        raise ValueError("no real tokens to replay")
    meta = []
    with torch.inference_mode():
        first = model(**inputs, use_cache=True)
        rows = [first.logits[0, -1].float()]
        past = first.past_key_values
        cache_len = int(past.get_seq_length()) if past is not None else 0
        meta.append({"step": 0, "cache_length": cache_len, "position_id": cache_len - 1})
        attention = inputs.get("attention_mask")
        device = rows[0].device
        for token in token_ids[:-1]:
            if attention is not None:
                attention = torch.cat([attention, torch.ones_like(attention[:, :1])], dim=1)
            step_ids = torch.tensor([[int(token)]], device=device)
            position = int(past.get_seq_length())
            step = model(input_ids=step_ids, attention_mask=attention, past_key_values=past, use_cache=True)
            rows.append(step.logits[0, -1].float())
            past = step.past_key_values
            meta.append({"step": len(meta), "cache_length": int(past.get_seq_length()), "position_id": position})
    if len(rows) != len(token_ids):
        raise RuntimeError("replay length differs from real generate length")
    return rows, meta


def float_list(values: list[torch.Tensor]) -> list[float]:
    return [float(value.detach().float().cpu()) for value in values]


def stacked_pixels(pixel_values: torch.Tensor, aux_idx: int, evidence_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aux page is the only leaf. Evidence is a detached copy of its own leaf."""
    if pixel_values.ndim < 1:
        raise ValueError("pixel_values has no page axis")
    aux = pixel_values[aux_idx].detach().clone().requires_grad_(True)
    evidence = pixel_values[evidence_idx].detach().clone().requires_grad_(True)
    pages = []
    for index in range(pixel_values.shape[0]):
        if index == aux_idx:
            pages.append(aux)
        elif index == evidence_idx:
            pages.append(evidence.detach())
        else:
            pages.append(pixel_values[index].detach())
    return torch.stack(pages), aux, evidence


def native_rows_from_full_ids(model, inputs: dict, token_ids: list[int], pixel_values: torch.Tensor) -> list[torch.Tensor]:
    prompt = inputs["input_ids"]
    device = prompt.device
    extra = torch.tensor(token_ids, dtype=prompt.dtype, device=device).view(1, -1)
    full_ids = torch.cat([prompt, extra], dim=1)
    attention = inputs.get("attention_mask")
    if attention is not None:
        ones = torch.ones(attention.shape[0], len(token_ids), dtype=attention.dtype, device=attention.device)
        attention = torch.cat([attention, ones], dim=1)
    kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "pixel_values"}
    }
    start = int(prompt.shape[1]) - 1
    keep = torch.arange(start, start + len(token_ids), device=device)
    output = model(
        input_ids=full_ids,
        attention_mask=attention,
        pixel_values=pixel_values,
        use_cache=False,
        logits_to_keep=keep,
        **kwargs,
    )
    return [output.logits[0, index].float() for index in range(len(token_ids))]


def aux_soft_gradient(model, inputs: dict, token_ids: list[int], aux_idx: int, evidence_idx: int) -> dict:
    pixels, aux, evidence = stacked_pixels(inputs["pixel_values"], aux_idx, evidence_idx)
    was_training = model.training
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    def mean_soft(pixel_values: torch.Tensor) -> torch.Tensor:
        rows = native_rows_from_full_ids(model, inputs, token_ids, pixel_values)
        soft = torch.stack([soft_margin_row(row, token) for row, token in zip(rows, token_ids)])
        if not torch.isfinite(soft).all():
            raise RuntimeError("non-finite native soft margin")
        return soft.mean()

    try:
        loss = mean_soft(pixels)
        loss.backward()
    finally:
        if not was_training:
            model.eval()
        model.gradient_checkpointing_disable()
    aux_l2 = None if aux.grad is None or not torch.isfinite(aux.grad).all() else float(aux.grad.float().norm())
    evidence_l2 = 0.0 if evidence.grad is None else float(evidence.grad.float().norm())
    return {"aux_grad_l2": aux_l2, "evidence_grad_l2": evidence_l2, "rows": len(token_ids)}

