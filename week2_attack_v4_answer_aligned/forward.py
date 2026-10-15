"""Compressed teacher-forcing that uses the real FastV keep rule.

Scores come from a prompt-only forward. The query is the last prompt token.
The keep count uses official round. The discrete keep set is detached.
"""

from __future__ import annotations

import torch

from week2_attack_v4_answer_aligned.loss import official_keep_count
from week2_pilot.attack import EarlyExit, last_query_scores

PRUNE_LAYER = 3
SCORE_LAYER = 2


def encode_page(model, pixels, image_size):
    feats = model.model.get_image_features(pixels, image_size)
    if len(feats) != 1:
        raise RuntimeError(f"expected one page feature, got {len(feats)}")
    return feats[0]


def assemble(text_embeds, image_mask, parts):
    features = torch.cat(parts, dim=0).to(dtype=text_embeds.dtype)
    n_tokens = int(image_mask[0, :, 0].sum())
    if n_tokens != features.shape[0]:
        raise RuntimeError(f"image tokens {n_tokens} != features {features.shape[0]}")
    return text_embeds.masked_scatter(image_mask, features)


def build_embeds(model, input_ids, image_mask, pixels, sizes, aux_idx, aux_pixels=None):
    """Text plus per-page features. Only the controlled page can carry gradient."""
    text = model.get_input_embeddings()(input_ids)
    parts = []
    for i in range(pixels.shape[0]):
        if aux_pixels is not None and i == aux_idx:
            feat = encode_page(model, aux_pixels, sizes[i:i + 1])
        else:
            feat = encode_page(model, pixels[i:i + 1], sizes[i:i + 1]).detach()
        parts.append(feat)
    return assemble(text, image_mask, parts)


def score_prompt(language_model, prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Last-prompt-token attention probs. prompt_embeds must not contain answer tokens."""
    position_ids = torch.arange(prompt_embeds.shape[1], device=prompt_embeds.device).unsqueeze(0)
    position_embeddings = language_model.rotary_emb(prompt_embeds, position_ids)
    layer = language_model.layers[SCORE_LAYER].self_attn
    holder: dict[str, torch.Tensor] = {}

    def hook(module, args, kwargs, output):
        hidden = kwargs.get("hidden_states")
        if hidden is None or hidden.shape[1] == 1:
            return
        holder["scores"] = last_query_scores(module, hidden, kwargs)
        raise EarlyExit()

    handle = layer.register_forward_hook(hook, with_kwargs=True)
    try:
        hidden = prompt_embeds
        for decoder in language_model.layers:
            hidden = decoder(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
    except EarlyExit:
        pass
    finally:
        handle.remove()
    if "scores" not in holder:
        raise RuntimeError("score layer did not run")
    if holder["scores"].shape[0] != prompt_embeds.shape[1]:
        raise RuntimeError("score length does not match the prompt")
    return holder["scores"]


def keep_index_from_scores(scores: torch.Tensor, spans, seq_len: int, keep_ratio: float):
    """Match runtime.prune: round keep count, then pack text + kept visual tokens."""
    visual = [pos for start, end in spans for pos in range(start, end)]
    visual_index = torch.tensor(visual, device=scores.device, dtype=torch.long)
    k = official_keep_count(len(visual), keep_ratio)
    visual_scores = scores.index_select(0, visual_index)
    relative = torch.topk(visual_scores.detach(), k).indices
    kept = visual_index[relative]
    visual_set = set(visual)
    first, last = spans[0][0], spans[-1][1]
    middle = [i for i in range(first, last) if i not in visual_set]
    pieces = []
    if first > 0:
        pieces.append(torch.arange(first, device=scores.device))
    pieces.append(kept)
    if middle:
        pieces.append(torch.tensor(middle, device=scores.device, dtype=torch.long))
    if last < seq_len:
        pieces.append(torch.arange(last, seq_len, device=scores.device))
    keep_index = torch.unique(torch.cat(pieces), sorted=True)
    kept_visual = sorted(int(x) for x in kept.detach().cpu().tolist())
    return keep_index, k, kept_visual


def _run_layer(decoder, hidden, position_ids, position_embeddings):
    out = decoder(
        hidden,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        use_cache=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def pruned_hidden(language_model, embeds: torch.Tensor, keep_index: torch.Tensor) -> torch.Tensor:
    """Full layers, hard-pruned at the input of layer 3. keep_index is detached."""
    keep_index = keep_index.to(device=embeds.device, dtype=torch.long).detach()
    position_ids = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
    position_embeddings = language_model.rotary_emb(embeds, position_ids)
    hidden = embeds
    for index, decoder in enumerate(language_model.layers):
        if index == PRUNE_LAYER:
            hidden = hidden.index_select(1, keep_index)
            position_ids = keep_index.view(1, -1)
            position_embeddings = language_model.rotary_emb(hidden, position_ids)

        def run(hidden_states, decoder=decoder, position_ids=position_ids, position_embeddings=position_embeddings):
            return _run_layer(decoder, hidden_states, position_ids, position_embeddings)

        hidden = torch.utils.checkpoint.checkpoint(run, hidden, use_reentrant=False)
    return language_model.norm(hidden)


def answer_logits_from_hidden(model, hidden: torch.Tensor, predictor_positions: list[int], keep_index: torch.Tensor) -> torch.Tensor:
    location = {int(pos): i for i, pos in enumerate(keep_index.detach().cpu().tolist())}
    missing = [pos for pos in predictor_positions if pos not in location]
    if missing:
        raise RuntimeError(f"answer predictors were pruned: {missing[:4]}")
    index = torch.tensor([location[pos] for pos in predictor_positions], device=hidden.device)
    selected = hidden.index_select(1, index)
    return model.lm_head(selected)[0]


def full_answer_logits(model, embeds: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """Unpruned teacher-forcing logits. Predictors are positions start .. start+length-1."""
    language_model = model.model.language_model
    position_ids = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
    position_embeddings = language_model.rotary_emb(embeds, position_ids)
    hidden = embeds
    for decoder in language_model.layers:
        def run(hidden_states, decoder=decoder, position_ids=position_ids, position_embeddings=position_embeddings):
            return _run_layer(decoder, hidden_states, position_ids, position_embeddings)

        hidden = torch.utils.checkpoint.checkpoint(run, hidden, use_reentrant=False)
    hidden = language_model.norm(hidden)
    return model.lm_head(hidden[:, start:start + length, :])[0]


def keep_index_from_kept(kept_visual: list[int], spans, seq_len: int, device) -> torch.Tensor:
    """Pack text plus an already chosen visual keep set, matching runtime.prune."""
    visual = [pos for start, end in spans for pos in range(start, end)]
    visual_set = set(visual)
    first, last = spans[0][0], spans[-1][1]
    middle = [i for i in range(first, last) if i not in visual_set]
    pieces = []
    if first > 0:
        pieces.append(torch.arange(first, device=device))
    pieces.append(torch.tensor(kept_visual, device=device, dtype=torch.long))
    if middle:
        pieces.append(torch.tensor(middle, device=device, dtype=torch.long))
    if last < seq_len:
        pieces.append(torch.arange(last, seq_len, device=device))
    return torch.unique(torch.cat(pieces), sorted=True)


def hook_kept(model, inputs, spans, keep_ratio: float) -> list[int]:
    """Prompt-only FastV keep set from the real runtime hook."""
    from adapters.runtime import clear_hooks, install

    clear_hooks(model)
    state = install(model, "fastv", keep_ratio, int(model.config.image_token_index))
    state.spans = spans
    with torch.no_grad():
        model(**inputs, use_cache=False)
    kept = list(state.kept_visual)
    clear_hooks(model)
    return kept


def compressed_answer_logits(model, full_embeds, prompt_len: int, keep_index: torch.Tensor, answer_len: int):
    """Teacher-forced logits on a sequence pruned with a detached keep set."""
    if answer_len < 1:
        raise ValueError("answer_len must be positive")
    hidden = pruned_hidden(model.model.language_model, full_embeds, keep_index)
    predictors = [prompt_len - 1 + t for t in range(answer_len)]
    logits = answer_logits_from_hidden(model, hidden, predictors, keep_index)
    return logits
