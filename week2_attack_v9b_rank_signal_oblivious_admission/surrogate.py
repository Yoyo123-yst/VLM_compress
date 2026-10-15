"""Surrogate path through the real LLaVA-OneVision forward. No FastV."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spatial_rule import apply_plan, pack_sequence
from common import image_spans


def page_features(model, inputs: dict, pixels: torch.Tensor) -> list[torch.Tensor]:
    return model.model.get_image_features(
        pixel_values=pixels,
        image_sizes=inputs["image_sizes"],
        batch_num_images=inputs.get("batch_num_images"),
    )


def compressed_embeds(model, inputs: dict, pixels: torch.Tensor, plan) -> tuple[torch.Tensor, list[int]]:
    features = page_features(model, inputs, pixels)
    pages = [apply_plan(feature, plan) for feature in features]
    spans = image_spans(inputs["input_ids"], int(model.config.image_token_index), len(pages))
    packed = pack_sequence(model, inputs["input_ids"], spans, pages)
    return packed, [int(page.shape[0]) for page in pages]


def surrogate_generate(model, processor, inputs: dict, pixels: torch.Tensor, plan) -> dict:
    with torch.inference_mode():
        embeds, counts = compressed_embeds(model, inputs, pixels, plan)
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        hidden = model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits
        if not torch.isfinite(hidden).all():
            raise RuntimeError("non-finite surrogate logits")
        output = model.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            do_sample=False,
            max_new_tokens=16,
            use_cache=True,
        )
    if output.shape[1] > embeds.shape[1]:
        tokens = [int(token) for token in output[0, embeds.shape[1]:].tolist()]
    else:
        tokens = [int(token) for token in output[0].tolist()]
    text = processor.batch_decode(torch.tensor(tokens).unsqueeze(0), skip_special_tokens=True)[0].strip()
    return {
        "text": text,
        "tokens": tokens,
        "page_out": counts,
        "packed_length": int(embeds.shape[1]),
        "hidden_shape": list(hidden.shape),
        "finite": True,
    }


def surrogate_ce(model, inputs: dict, pixels: torch.Tensor, plan, token_ids: list[int]) -> torch.Tensor:
    embeds, _ = compressed_embeds(model, inputs, pixels, plan)
    extra = model.get_input_embeddings()(torch.tensor(token_ids, device=embeds.device).view(1, -1))
    full = torch.cat([embeds, extra.to(dtype=embeds.dtype)], dim=1)
    mask = torch.ones(full.shape[:2], dtype=torch.long, device=full.device)
    start = int(embeds.shape[1]) - 1
    keep = torch.arange(start, start + len(token_ids), device=full.device)
    logits = model(inputs_embeds=full, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits[0]
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite surrogate teacher-forced logits")
    target = torch.tensor(token_ids, device=logits.device)
    return F.cross_entropy(logits.float(), target)


def pixel_reg(delta: torch.Tensor) -> torch.Tensor:
    horizontal = (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean()
    vertical = (delta[:, 1:, :] - delta[:, :-1, :]).abs().mean()
    return delta.pow(2).mean() + horizontal + vertical
