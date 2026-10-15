"""Differentiable native margins.

FastV keep indices come from the installed runtime hook. The backward path
replays those detached indices with an out-of-place gather because the hook
resizes shared tensors in place and that resize is not differentiable.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

from week2_attack_v4_answer_aligned.forward import hook_kept, keep_index_from_kept
from week2_attack_v7b_native_cache_margin_admission.native_cache_margin import (
    hard_margin_row,
    native_rows_from_full_ids,
    soft_margin_row,
)

TRAIN_RATIOS = (0.25, 0.35)


def splice_aux(pixel_values: torch.Tensor, aux_idx: int, tiles: torch.Tensor) -> torch.Tensor:
    pages = [pixel_values[index].detach() for index in range(pixel_values.shape[0])]
    pages[aux_idx] = tiles.to(dtype=pixel_values.dtype)
    return torch.stack(pages)


def _layer_cached(layer, hidden, position_ids, cache_position, cos, sin, past_k, past_v):
    past = DynamicCache()
    if past_k.numel() != 0:
        past.update(past_k, past_v, layer.self_attn.layer_idx)
    hidden = layer(
        hidden,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=past,
        use_cache=True,
        cache_position=cache_position,
        position_embeddings=(cos, sin),
    )
    slot = past.layers[layer.self_attn.layer_idx]
    return hidden, slot.keys, slot.values


def _run_layers(model, hidden, position_ids, cache_position, caches, keep_index):
    lm = model.model.language_model
    cos, sin = lm.rotary_emb(hidden, position_ids)
    updated = []
    for index, layer in enumerate(lm.layers):
        if keep_index is not None and index == 3:
            hidden = hidden.index_select(1, keep_index)
            position_ids = keep_index.view(1, -1)
            cache_position = keep_index
            cos, sin = lm.rotary_emb(hidden, position_ids)
        past_k, past_v = caches[index]
        hidden, past_k, past_v = checkpoint(
            _layer_cached,
            layer,
            hidden,
            position_ids,
            cache_position,
            cos,
            sin,
            past_k,
            past_v,
            use_reentrant=False,
        )
        updated.append((past_k, past_v))
    return lm.norm(hidden), updated


def fastv_rows(model, inputs: dict, pixel_values: torch.Tensor, token_ids: list[int], keep_index: torch.Tensor, trace: list | None = None) -> list[torch.Tensor]:
    if not token_ids:
        raise ValueError("FastV margin needs the tokens generate actually produced")
    lm = model.model
    ids = inputs["input_ids"]
    prompt_len = int(ids.shape[1])
    device = ids.device
    embeds = lm.get_input_embeddings()(ids)
    features = lm.get_image_features(
        pixel_values,
        inputs["image_sizes"],
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )
    features = torch.cat(features, dim=0).to(dtype=embeds.dtype)
    image_mask, _ = lm.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
    hidden = embeds.masked_scatter(image_mask, features)
    empty = hidden.new_zeros(1, 1, 0, 1)
    caches = [(empty, empty) for _ in lm.language_model.layers]
    position = torch.arange(prompt_len, device=device).view(1, -1)
    hidden, caches = _run_layers(model, hidden, position, position.view(-1), caches, keep_index)
    if trace is not None:
        trace.append({
            "step": 0,
            "cache_len_layer0": int(caches[0][0].shape[-2]),
            "cache_len_layer3": int(caches[3][0].shape[-2]),
            "position_id": prompt_len - 1,
        })
    rows = [model.lm_head(hidden[:, -1, :])[0].float()]
    for offset, token in enumerate(token_ids[:-1]):
        step_ids = torch.tensor([[int(token)]], dtype=ids.dtype, device=device)
        step_hidden = lm.get_input_embeddings()(step_ids)
        pos_id = prompt_len + offset
        step_position = torch.tensor([[pos_id]], device=device)
        step_hidden, caches = _run_layers(model, step_hidden, step_position, step_position.view(-1), caches, None)
        if trace is not None:
            trace.append({
                "step": offset + 1,
                "cache_len_layer0": int(caches[0][0].shape[-2]),
                "cache_len_layer3": int(caches[3][0].shape[-2]),
                "position_id": pos_id,
            })
        rows.append(model.lm_head(step_hidden[:, -1, :])[0].float())
    if len(rows) != len(token_ids):
        raise RuntimeError("FastV differentiable length differs from generate")
    return rows


def mean_soft(rows: list[torch.Tensor], token_ids: list[int]) -> torch.Tensor:
    return torch.stack([soft_margin_row(row, token) for row, token in zip(rows, token_ids)]).mean()


def mean_hard(rows: list[torch.Tensor], token_ids: list[int]) -> torch.Tensor:
    return torch.stack([hard_margin_row(row, token) for row, token in zip(rows, token_ids)]).mean()


def full_rows(model, inputs: dict, pixel_values: torch.Tensor, token_ids: list[int]) -> list[torch.Tensor]:
    return native_rows_from_full_ids(model, inputs, token_ids, pixel_values)


def fastv_keep(model, inputs: dict, spans, ratio: float) -> tuple[list[int], torch.Tensor]:
    if ratio not in TRAIN_RATIOS and ratio not in (0.15, 0.50):
        raise ValueError(f"ratio {ratio} is outside the frozen budget set")
    kept = hook_kept(model, inputs, spans, ratio)
    keep_index = keep_index_from_kept(kept, spans, int(inputs["input_ids"].shape[1]), inputs["input_ids"].device)
    return kept, keep_index
