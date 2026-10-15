"""Question-specific aux-page attack, v2.

L = L_share + λ L_full. L_share is a one-sided target-share loss.
L_full is KL(p_clean || p_adv) on the clean greedy answer, teacher-forced.
The loss does not read the evidence-page index. Reported metrics use hard FastV.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))
sys.path.insert(0, str(ROOT))

from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import (  # noqa: E402
    append_csv,
    budget,
    first_other,
    generate,
    load_model,
    prepare,
    run_fastv,
    run_plain,
    spans_of,
)
from run_step0_v2 import draw_questions, load_images, select_pages  # noqa: E402
from week2_attack_v2.loss import ceil_keep_count, kl_clean_adv, share_target_loss  # noqa: E402
from week2_pilot.attack import (  # noqa: E402
    EarlyExit,
    float_to_page,
    forward_visual_scores,
    last_query_scores,
    normalized_tiles,
    page_to_float,
    share_loss,
)

OUT_DIR = ROOT / "week2_attack_v2" / "results"
BASELINE = ROOT / "gate_experiment" / "results" / "clean_baseline.csv"
RATIO = 0.5
EPS = 8 / 255
Q_STAR = 0.6
SCORE_LAYER = 2
FIELDS = [
    "sample_id", "eps", "arm", "lambda", "T_c", "aux_share", "evidence_share",
    "evidence_keep", "anls_compressed", "anls_full", "csf_strict", "csf_loose", "M",
    "clean_aux_share", "clean_evidence_share", "evidence_tokens", "steps", "q_star", "loss",
]


def subset_questions() -> list[dict]:
    full_m, full_anls, fastv_anls = {}, {}, {}
    with BASELINE.open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "full":
                full_m[row["sample_id"]] = int(float(row["M"]))
                full_anls[row["sample_id"]] = float(row["anls"])
            if row["method"] == "fastv" and abs(float(row["r"]) - RATIO) < 1e-6:
                fastv_anls[row["sample_id"]] = float(row["anls"])
    questions = draw_questions(400, 42)
    pool = [
        q for q in questions
        if full_m[q["sample_id"]] > 1
        and full_anls[q["sample_id"]] >= 0.5
        and fastv_anls[q["sample_id"]] >= 0.5
    ]
    pool.sort(key=lambda q: q["draw_rank"])
    return pool


def tuning_questions(pool: list[dict], n: int) -> list[dict]:
    rng = random.Random(42)
    picked = rng.sample(pool, n)
    picked.sort(key=lambda q: q["draw_rank"])
    return picked


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as handle:
        for row in csv.DictReader(handle):
            keys.add((row["sample_id"], row["arm"], row["lambda"], row["T_c"], row["steps"]))
    return keys


def encode_page(model, pixels, image_size):
    feats = model.model.get_image_features(pixels, image_size)
    if len(feats) != 1:
        raise RuntimeError(f"expected one page feature, got {len(feats)}")
    return feats[0]


def assemble(text_embeds, image_mask, parts):
    features = torch.cat(parts, dim=0).to(dtype=text_embeds.dtype)
    if int(image_mask[0, :, 0].sum()) != features.shape[0]:
        raise RuntimeError(
            f"image tokens {int(image_mask[0, :, 0].sum())} != features {features.shape[0]}"
        )
    return text_embeds.masked_scatter(image_mask, features)


def _positions(language_model, embeds):
    position_ids = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
    # No additive mask: SDPA then uses the causal kernel and does not store an S×S matrix.
    return position_ids, language_model.rotary_emb(embeds, position_ids)


def prefix_scores(language_model, embeds):
    """FastV scores at layer 2. The sequence is the prompt, so the last token is the query."""
    position_ids, position_embeddings = _positions(language_model, embeds)
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
        hidden = embeds
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
    return holder["scores"]


def answer_logits(model, embeds, start: int, length: int):
    language_model = model.model.language_model
    position_ids, position_embeddings = _positions(language_model, embeds)
    hidden = embeds
    for decoder in language_model.layers:
        def run(hidden_states, decoder=decoder):
            return decoder(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )

        hidden = torch.utils.checkpoint.checkpoint(run, hidden, use_reentrant=False)
    hidden = language_model.norm(hidden)
    return model.lm_head(hidden[:, start:start + length, :])[0]


def project(delta, clean, eps):
    with torch.no_grad():
        delta.clamp_(-eps, eps)
        delta.copy_((clean + delta).clamp(0, 1) - clean)


def attack_v1(model, processor, images, question, aux_idx, steps, eps):
    torch.manual_seed(0)
    clean = page_to_float(images[aux_idx]).to("cuda")
    delta = torch.zeros_like(clean, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    base = prepare(processor, images, question)
    spans = spans_of(base, int(model.config.image_token_index), len(images))
    visual = [pos for start, end in spans for pos in range(start, end)]
    aux_ids = [i for i, pos in enumerate(visual) if spans[aux_idx][0] <= pos < spans[aux_idx][1]]
    keep_k = ceil_keep_count(len(visual), RATIO)
    pixels = base["pixel_values"]
    last = 0.0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tiles = normalized_tiles((clean + delta).clamp(0, 1)).to(dtype=pixels.dtype)
        batch = pixels.detach().clone()
        batch[aux_idx] = tiles
        inputs = dict(base)
        inputs["pixel_values"] = batch
        scores = forward_visual_scores(model, inputs)
        loss = share_loss(scores[visual], aux_ids, keep_k, 1e-4)
        loss.backward()
        opt.step()
        project(delta, clean, eps)
        last = float(loss.detach())
        if step % 50 == 0 or step + 1 == steps:
            print(json.dumps({"arm": "v1", "step": step, "loss": last}), flush=True)
    return float_to_page((clean + delta).detach()), last


def attack_v2(model, processor, images, question, aux_idx, steps, eps, lam, c, lr):
    clean_img = page_to_float(images[aux_idx]).to("cuda")
    delta = torch.zeros_like(clean_img, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    base = prepare(processor, images, question)
    prompt_len = int(base["input_ids"].shape[1])
    spans = spans_of(base, int(model.config.image_token_index), len(images))
    visual = [pos for start, end in spans for pos in range(start, end)]
    aux_ids = [i for i, pos in enumerate(visual) if spans[aux_idx][0] <= pos < spans[aux_idx][1]]
    keep_k = ceil_keep_count(len(visual), RATIO)
    pixels = base["pixel_values"]
    sizes = base["image_sizes"]
    with torch.inference_mode():
        answer = generate(model, base)
    if answer.numel() == 0:
        raise RuntimeError("clean greedy answer is empty")
    full_ids = torch.cat([base["input_ids"], answer.view(1, -1)], dim=1)
    image_token_id = int(model.config.image_token_index)
    with torch.no_grad():
        text_full = model.get_input_embeddings()(full_ids)
        mask_1d = full_ids[0].eq(image_token_id)
        image_mask = mask_1d.view(1, -1, 1).expand_as(text_full)
        cache = []
        for i in range(len(images)):
            if i == aux_idx:
                cache.append(None)
                continue
            cache.append(encode_page(model, pixels[i:i + 1], sizes[i:i + 1]).detach())
        clean_tiles = normalized_tiles(clean_img).to(dtype=pixels.dtype)
        clean_feat = encode_page(model, clean_tiles.unsqueeze(0), sizes[aux_idx:aux_idx + 1]).detach()
        clean_parts = [clean_feat if i == aux_idx else cache[i] for i in range(len(images))]
        clean_embeds = assemble(text_full, image_mask, clean_parts)
        clean_logits = answer_logits(model, clean_embeds, prompt_len - 1, int(answer.numel()))
        log_p_clean = clean_logits.float().log_softmax(dim=-1).detach()

    last = 0.0
    last_share = 0.0
    last_t = 0.0
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tiles = normalized_tiles((clean_img + delta).clamp(0, 1)).to(dtype=pixels.dtype)
        feat = encode_page(model, tiles.unsqueeze(0), sizes[aux_idx:aux_idx + 1])
        parts = [feat if i == aux_idx else cache[i] for i in range(len(images))]
        embeds = assemble(text_full, image_mask, parts)
        scores = prefix_scores(model.model.language_model, embeds[:, :prompt_len])
        visual_scores = scores[visual].float()
        temperature = torch.clamp(c * visual_scores.std(unbiased=False), min=1e-6).detach()
        loss_share, soft_share = share_target_loss(
            visual_scores, aux_ids, keep_k, temperature, Q_STAR
        )
        # Free the score graph before the full-depth KL backward.
        loss_share.backward()
        share_value = float(loss_share.detach())
        last_share = float(soft_share.detach())
        last_t = float(temperature)
        del feat, embeds, scores, visual_scores, loss_share, soft_share
        tiles = normalized_tiles((clean_img + delta).clamp(0, 1)).to(dtype=pixels.dtype)
        feat = encode_page(model, tiles.unsqueeze(0), sizes[aux_idx:aux_idx + 1])
        parts = [feat if i == aux_idx else cache[i] for i in range(len(images))]
        embeds = assemble(text_full, image_mask, parts)
        logits = answer_logits(model, embeds, prompt_len - 1, int(answer.numel()))
        loss_full = kl_clean_adv(log_p_clean, logits)
        (lam * loss_full).backward()
        opt.step()
        project(delta, clean_img, eps)
        full_value = float(loss_full.detach())
        last = share_value + lam * full_value
        if step % 50 == 0 or step + 1 == steps:
            print(json.dumps({
                "arm": "v2", "step": step, "loss": last, "share_loss": share_value,
                "full_loss": full_value, "soft_share": last_share, "T": last_t,
                "sec_per_step": (time.time() - t0) / (step + 1),
            }), flush=True)
        del feat, embeds, logits, loss_full
    page = float_to_page((clean_img + delta).detach())
    del delta, opt
    torch.cuda.empty_cache()
    return page, last, last_share, last_t


def measure(model, processor, images, meta, image_token_id, evidence_idx, aux_idx, e_ids):
    pred, spans, kept, _ = run_fastv(
        model, processor, images, meta["question"], image_token_id, RATIO, evidence_idx, aux_idx
    )
    stats = budget(kept, spans, evidence_idx, aux_idx, e_ids)
    full_pred, _, _, _ = run_plain(
        model, processor, images, meta["question"], image_token_id, False, evidence_idx
    )
    return stats, anls_score(pred, meta["answers"]), anls_score(full_pred, meta["answers"])


def flags(anls_full, anls_compressed, clean_compressed):
    strict = int(anls_full >= 0.5 and anls_compressed < 0.5)
    loose = int(anls_full >= 0.5 and clean_compressed - anls_compressed >= 0.3)
    return strict, loose


def noise_page(image, eps, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    clean = page_to_float(image)
    delta = (torch.rand(clean.shape, generator=gen) * 2 - 1) * eps
    return float_to_page((clean + delta).clamp(0, 1))


def write_row(path, seen, row):
    key = (row["sample_id"], row["arm"], row["lambda"], row["T_c"], row["steps"])
    if key in seen:
        return
    append_csv(path, row, FIELDS)
    seen.add(key)
    print(json.dumps({
        "id": row["sample_id"], "arm": row["arm"], "lambda": row["lambda"], "c": row["T_c"],
        "anls_c": row["anls_compressed"], "anls_f": row["anls_full"], "aux": row["aux_share"],
        "csf": row["csf_strict"],
    }), flush=True)


def one(model, processor, meta, pages, donors, image_token_id, seen, path, steps, grid, do_controls, do_v1, lr):
    images, evidence_idx, _, _ = select_pages(pages, meta["answer_page_idx"], meta["sample_id"], 768)
    aux_idx = first_other(len(images), evidence_idx)
    _, spans, _, e_ids = run_plain(
        model, processor, images, meta["question"], image_token_id, True, evidence_idx
    )
    del spans
    clean_stats, clean_c, clean_f = measure(
        model, processor, images, meta, image_token_id, evidence_idx, aux_idx, e_ids
    )

    def base_row(arm, lam, tc, stats, anls_c, anls_f, loss, n_steps):
        strict, loose = flags(anls_f, anls_c, clean_c)
        return {
            "sample_id": meta["sample_id"], "eps": f"{EPS:.6f}" if arm not in ("clean", "dense") else "0.000000",
            "arm": arm, "lambda": lam, "T_c": tc,
            "aux_share": f"{stats['aux_share']:.6f}",
            "evidence_share": f"{stats['evidence_share']:.6f}",
            "evidence_keep": f"{stats['evidence_keep']:.6f}",
            "anls_compressed": f"{anls_c:.4f}", "anls_full": f"{anls_f:.4f}",
            "csf_strict": strict, "csf_loose": loose, "M": len(images),
            "clean_aux_share": f"{clean_stats['aux_share']:.6f}",
            "clean_evidence_share": f"{clean_stats['evidence_share']:.6f}",
            "evidence_tokens": 3699, "steps": str(n_steps), "q_star": Q_STAR, "loss": loss,
        }

    if do_controls:
        write_row(path, seen, base_row("clean", "", "", clean_stats, clean_c, clean_f, "", 0))
        dense = images.copy()
        dense[aux_idx] = donors[meta["sample_id"]].resize((768, 768), Image.Resampling.BICUBIC)
        d_stats, d_c, d_f = measure(model, processor, dense, meta, image_token_id, evidence_idx, aux_idx, e_ids)
        write_row(path, seen, base_row("dense", "", "", d_stats, d_c, d_f, "", 0))
        noisy = images.copy()
        noisy[aux_idx] = noise_page(images[aux_idx], EPS, 42 + meta["draw_rank"])
        n_stats, n_c, n_f = measure(model, processor, noisy, meta, image_token_id, evidence_idx, aux_idx, e_ids)
        write_row(path, seen, base_row("noise", "", "", n_stats, n_c, n_f, "", 0))
    if do_v1:
        key = (meta["sample_id"], "v1", "0", "fixed-1e-4", str(steps))
        if key not in seen:
            page, loss = attack_v1(model, processor, images, meta["question"], aux_idx, steps, EPS)
            attacked = images.copy()
            attacked[aux_idx] = page
            stats, anls_c, anls_f = measure(
                model, processor, attacked, meta, image_token_id, evidence_idx, aux_idx, e_ids
            )
            write_row(path, seen, base_row("v1", "0", "fixed-1e-4", stats, anls_c, anls_f, f"{loss:.6f}", steps))
    for lam, c in grid:
        key = (meta["sample_id"], "v2", f"{lam:g}", f"{c:g}", str(steps))
        if key in seen:
            continue
        page, loss, _soft, _temp = attack_v2(
            model, processor, images, meta["question"], aux_idx, steps, EPS, lam, c, lr
        )
        attacked = images.copy()
        attacked[aux_idx] = page
        stats, anls_c, anls_f = measure(
            model, processor, attacked, meta, image_token_id, evidence_idx, aux_idx, e_ids
        )
        write_row(path, seen, base_row("v2", f"{lam:g}", f"{c:g}", stats, anls_c, anls_f, f"{loss:.6f}", steps))


def parse_grid(text: str) -> list[tuple[float, float]]:
    grid = []
    for piece in text.split(","):
        lam, c = piece.split(":")
        grid.append((float(lam), float(c)))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=40, help="0 means the full new subset")
    parser.add_argument("--screen", type=int, default=0, help="If >0, tune on this many of the n-question draw")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--grid", type=str, default="")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--v1", action="store_true")
    parser.add_argument("--lr", type=float, default=2.0 / 255.0)
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    pool = subset_questions()
    questions = pool if args.n == 0 else tuning_questions(pool, args.n)
    if args.screen:
        rng = random.Random(42)
        questions = rng.sample(questions, args.screen)
        questions.sort(key=lambda q: q["draw_rank"])
    mine = [q for i, q in enumerate(questions) if i % args.nshards == args.shard]
    print(json.dumps({"pool": len(pool), "running": len(questions), "mine": len(mine), "shard": args.shard}), flush=True)
    seen = done_keys(args.out)
    all_pages = load_images(questions)
    donors = {}
    for i, meta in enumerate(questions):
        nxt = questions[(i + 1) % len(questions)]
        pages, evidence_idx, _, _ = select_pages(
            all_pages[nxt["sample_id"]], nxt["answer_page_idx"], nxt["sample_id"], 768
        )
        donors[meta["sample_id"]] = pages[evidence_idx]
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    image_token_id = int(model.config.image_token_index)
    grid = parse_grid(args.grid) if args.grid else []
    for meta in mine:
        try:
            one(
                model, processor, meta, all_pages[meta["sample_id"]], donors, image_token_id,
                seen, args.out, args.steps, grid, args.controls, args.v1, args.lr,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(json.dumps({"id": meta["sample_id"], "error": "OOM"}), flush=True)
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
