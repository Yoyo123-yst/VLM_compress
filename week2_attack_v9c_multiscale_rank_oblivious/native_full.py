"""Full-only margin helpers copied from modules that do not import FastV.

Source checks below confirm the copied expressions. This file does not import those modules.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
_MARGIN = (ROOT / "week2_attack_v7b_native_cache_margin_admission" / "native_cache_margin.py").read_text()
_LOSS = (ROOT / "week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation" / "losses.py").read_text()
if "return chosen - rival.max()" not in _MARGIN or "return F.softplus(clean_hard - hard_margin).mean()" not in _LOSS:
    raise RuntimeError("Full-only source expressions no longer match the copied helpers")


def hard_margin_row(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("logits must be one vocabulary row")
    chosen = logits[token_id]
    rival = logits.clone()
    rival[token_id] = float("-inf")
    return chosen - rival.max()


def decision_preservation(hard_margin: torch.Tensor, clean_hard: torch.Tensor) -> torch.Tensor:
    return F.softplus(clean_hard - hard_margin).mean()


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
        if key not in {"input_ids", "attention_mask", "pixel_values", "prompt"} and torch.is_tensor(value)
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
