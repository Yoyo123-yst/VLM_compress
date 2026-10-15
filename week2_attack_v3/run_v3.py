"""Attack v3: same aux-page attack at several FastV ratios.

q* = min(0.95, 1/(r*M)). Loss is the v2 one-sided share target plus λ L_full.
λ=0.5, c=0.5, ε=8/255, 400 steps. The evidence-page index is not in the loss.
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

from hijack import transfer_keep  # noqa: E402
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
from week2_attack_v2.run_v2 import (  # noqa: E402
    answer_logits,
    assemble,
    encode_page,
    flags,
    noise_page,
    prefix_scores,
    project,
    subset_questions,
    tuning_questions,
)
from week2_pilot.attack import float_to_page, normalized_tiles, page_to_float  # noqa: E402

BASELINE = ROOT / "gate_experiment" / "results" / "clean_baseline.csv"
EPS = 8 / 255
LAM = 0.5
TEMP_C = 0.5
RATIOS = (0.5, 0.35, 0.25)
FIELDS = [
    "sample_id", "r", "arm", "aux_share", "aux_ceiling", "evidence_share",
    "evidence_keep", "anls_compressed", "anls_full", "csf_strict", "csf_loose", "M", "steps",
    "q_star", "soft_share",
]
ORACLE_FIELDS = [
    "sample_id", "r", "arm", "M", "s", "anls", "aux_share", "evidence_share", "evidence_keep",
    "n_moved", "n_requested", "skipped",
]


def baseline_m() -> dict[str, int]:
    out = {}
    with BASELINE.open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "full":
                out[row["sample_id"]] = int(float(row["M"]))
    return out


def select_questions() -> list[dict]:
    """The v2 40-question draw, topped up so M=2 and M=3 each have at least 8."""
    pool = subset_questions()
    masses = baseline_m()
    chosen = tuning_questions(pool, 40)
    chosen_ids = {q["sample_id"] for q in chosen}
    rng = random.Random(42)
    for mass in (2, 3):
        have = sum(masses[q["sample_id"]] == mass for q in chosen)
        if have >= 8:
            continue
        extra = [q for q in pool if q["sample_id"] not in chosen_ids and masses[q["sample_id"]] == mass]
        rng.shuffle(extra)
        take = extra[: 8 - have]
        chosen.extend(take)
        chosen_ids.update(q["sample_id"] for q in take)
    chosen.sort(key=lambda q: q["draw_rank"])
    return chosen


def ceiling(ratio: float, mass: int) -> float:
    return 1.0 / (ratio * mass)


def target_share(ratio: float, mass: int) -> float:
    return min(0.95, ceiling(ratio, mass))


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as handle:
        for row in csv.DictReader(handle):
            keys.add((row["sample_id"], row["r"], row["arm"], row["steps"]))
    return keys


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def measure(model, processor, images, meta, image_token_id, ratio, evidence_idx, aux_idx, e_ids):
    pred, spans, kept, _ = run_fastv(
        model, processor, images, meta["question"], image_token_id, ratio, evidence_idx, aux_idx
    )
    stats = budget(kept, spans, evidence_idx, aux_idx, e_ids)
    full_pred, _, _, _ = run_plain(
        model, processor, images, meta["question"], image_token_id, False, evidence_idx
    )
    return stats, anls_score(pred, meta["answers"]), anls_score(full_pred, meta["answers"])


def attack(model, processor, images, question, aux_idx, steps, ratio, mass, curve_path, sample_id):
    q_star = target_share(ratio, mass)
    cap = ceiling(ratio, mass)
    clean_img = page_to_float(images[aux_idx]).to("cuda")
    delta = torch.zeros_like(clean_img, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    base = prepare(processor, images, question)
    prompt_len = int(base["input_ids"].shape[1])
    spans = spans_of(base, int(model.config.image_token_index), len(images))
    visual = [pos for start, end in spans for pos in range(start, end)]
    aux_ids = [i for i, pos in enumerate(visual) if spans[aux_idx][0] <= pos < spans[aux_idx][1]]
    keep_k = ceil_keep_count(len(visual), ratio)
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
        image_mask = full_ids[0].eq(image_token_id).view(1, -1, 1).expand_as(text_full)
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

    last_share = 0.0
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tiles = normalized_tiles((clean_img + delta).clamp(0, 1)).to(dtype=pixels.dtype)
        feat = encode_page(model, tiles.unsqueeze(0), sizes[aux_idx:aux_idx + 1])
        parts = [feat if i == aux_idx else cache[i] for i in range(len(images))]
        embeds = assemble(text_full, image_mask, parts)
        scores = prefix_scores(model.model.language_model, embeds[:, :prompt_len])
        visual_scores = scores[visual].float()
        temperature = torch.clamp(TEMP_C * visual_scores.std(unbiased=False), min=1e-6).detach()
        loss_share, soft_share = share_target_loss(
            visual_scores, aux_ids, keep_k, temperature, q_star
        )
        loss_share.backward()
        share_value = float(loss_share.detach())
        last_share = float(soft_share.detach())
        del feat, embeds, scores, visual_scores, loss_share, soft_share
        tiles = normalized_tiles((clean_img + delta).clamp(0, 1)).to(dtype=pixels.dtype)
        feat = encode_page(model, tiles.unsqueeze(0), sizes[aux_idx:aux_idx + 1])
        parts = [feat if i == aux_idx else cache[i] for i in range(len(images))]
        embeds = assemble(text_full, image_mask, parts)
        logits = answer_logits(model, embeds, prompt_len - 1, int(answer.numel()))
        loss_full = kl_clean_adv(log_p_clean, logits)
        (LAM * loss_full).backward()
        opt.step()
        project(delta, clean_img, EPS)
        if step % 50 == 0 or step + 1 == steps:
            row = {
                "sample_id": sample_id, "r": ratio, "step": step, "soft_share": last_share,
                "q_star": q_star, "ceiling": cap, "share_loss": share_value,
                "full_loss": float(loss_full.detach()),
                "sec_per_step": (time.time() - t0) / (step + 1),
            }
            append_jsonl(curve_path, row)
            print(json.dumps({"arm": "attack", **row}), flush=True)
        del feat, embeds, logits, loss_full
    page = float_to_page((clean_img + delta).detach())
    del delta, opt
    torch.cuda.empty_cache()
    return page, last_share, q_star, cap


def oracle_curve(model, processor, images, meta, image_token_id, ratio, evidence_idx, aux_idx, e_ids, path, seen):
    """Hijack curve at this r. Used for r=0.35, which has no saved curve."""
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        s_tag = f"{s:g}"
        key = (meta["sample_id"], f"{ratio:g}", "hijack", s_tag)
        if key in seen:
            continue

        def make_override(s=s):
            def override(spans, sink, s=s):
                def fn(visual_index, scores, kept, spans=spans, sink=sink, s=s):
                    new_kept, moved, requested = transfer_keep(
                        visual_index, scores, kept, spans[evidence_idx], spans[aux_idx], drop_fraction=s
                    )
                    sink["moved"] = moved
                    sink["requested"] = requested
                    return new_kept
                return fn
            return override

        pred, hspans, kept, sink = run_fastv(
            model, processor, images, meta["question"], image_token_id,
            ratio, evidence_idx, aux_idx, make_override(),
        )
        stats = budget(kept, hspans, evidence_idx, aux_idx, e_ids)
        append_csv(path, {
            "sample_id": meta["sample_id"], "r": f"{ratio:g}", "arm": "hijack", "M": len(images),
            "s": s_tag, "anls": f"{anls_score(pred, meta['answers']):.4f}",
            "aux_share": f"{stats['aux_share']:.6f}",
            "evidence_share": f"{stats['evidence_share']:.6f}",
            "evidence_keep": f"{stats['evidence_keep']:.6f}",
            "n_moved": sink["moved"], "n_requested": sink["requested"], "skipped": 0,
        }, ORACLE_FIELDS)
        seen.add(key)


def one(model, processor, meta, pages, donors, image_token_id, seen, path, oracle_path, oracle_seen, curve_path, steps, ratios):
    images, evidence_idx, _, _ = select_pages(pages, meta["answer_page_idx"], meta["sample_id"], 768)
    aux_idx = first_other(len(images), evidence_idx)
    mass = len(images)
    _, _, _, e_ids = run_plain(
        model, processor, images, meta["question"], image_token_id, True, evidence_idx
    )
    donor = donors[meta["sample_id"]].resize((768, 768), Image.Resampling.BICUBIC)
    for ratio in ratios:
        cap = ceiling(ratio, mass)
        clean_stats, clean_c, clean_f = measure(
            model, processor, images, meta, image_token_id, ratio, evidence_idx, aux_idx, e_ids
        )

        def write(arm, images_now, n_steps, soft, q_star):
            key = (meta["sample_id"], f"{ratio:g}", arm, str(n_steps))
            if key in seen:
                return
            if images_now is images and arm == "clean":
                stats, anls_c, anls_f = clean_stats, clean_c, clean_f
            else:
                stats, anls_c, anls_f = measure(
                    model, processor, images_now, meta, image_token_id, ratio, evidence_idx, aux_idx, e_ids
                )
            strict, loose = flags(anls_f, anls_c, clean_c)
            append_csv(path, {
                "sample_id": meta["sample_id"], "r": f"{ratio:g}", "arm": arm,
                "aux_share": f"{stats['aux_share']:.6f}", "aux_ceiling": f"{cap:.6f}",
                "evidence_share": f"{stats['evidence_share']:.6f}",
                "evidence_keep": f"{stats['evidence_keep']:.6f}",
                "anls_compressed": f"{anls_c:.4f}", "anls_full": f"{anls_f:.4f}",
                "csf_strict": strict, "csf_loose": loose, "M": mass, "steps": n_steps,
                "q_star": f"{q_star:.6f}", "soft_share": "" if soft == "" else f"{soft:.6f}",
            }, FIELDS)
            seen.add(key)
            print(json.dumps({
                "id": meta["sample_id"], "r": ratio, "arm": arm, "M": mass,
                "aux": stats["aux_share"], "keep": stats["evidence_keep"],
                "anls_c": anls_c, "anls_f": anls_f, "csf": strict,
            }), flush=True)

        write("clean", images, 0, "", target_share(ratio, mass))
        dense = images.copy()
        dense[aux_idx] = donor
        write("dense", dense, 0, "", target_share(ratio, mass))
        noisy = images.copy()
        noisy[aux_idx] = noise_page(images[aux_idx], EPS, 42 + meta["draw_rank"] + int(ratio * 1000))
        write("noise", noisy, 0, "", target_share(ratio, mass))
        if ratio == 0.35:
            oracle_curve(
                model, processor, images, meta, image_token_id, ratio,
                evidence_idx, aux_idx, e_ids, oracle_path, oracle_seen,
            )
        attack_key = (meta["sample_id"], f"{ratio:g}", "attack", str(steps))
        if attack_key in seen:
            continue
        page, soft, q_star, _cap = attack(
            model, processor, images, meta["question"], aux_idx, steps, ratio, mass, curve_path, meta["sample_id"]
        )
        attacked = images.copy()
        attacked[aux_idx] = page
        write("attack", attacked, steps, soft, q_star)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle-out", type=Path, required=True)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--ratios", type=str, default="0.5,0.35,0.25")
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    questions = select_questions()
    masses = baseline_m()
    from collections import Counter
    print(json.dumps({
        "n": len(questions),
        "M": dict(Counter(masses[q["sample_id"]] for q in questions)),
        "shard": args.shard,
    }), flush=True)
    mine = [q for i, q in enumerate(questions) if i % args.nshards == args.shard]
    ratios = tuple(float(x) for x in args.ratios.split(",") if x)
    seen = done_keys(args.out)
    oracle_seen = set()
    if args.oracle_out.exists():
        with args.oracle_out.open() as handle:
            for row in csv.DictReader(handle):
                oracle_seen.add((row["sample_id"], row["r"], row["arm"], row["s"]))
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
    for meta in mine:
        try:
            one(
                model, processor, meta, all_pages[meta["sample_id"]], donors, image_token_id,
                seen, args.out, args.oracle_out, oracle_seen, args.curve, args.steps, ratios,
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
