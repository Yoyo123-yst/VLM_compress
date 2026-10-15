"""Quick r=0.25 probe. Clean metrics are read from the frozen baseline."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))
sys.path.insert(0, str(ROOT))

from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import append_csv, budget, first_other, generate, load_model, prepare, run_fastv, run_plain, spans_of  # noqa: E402
from run_step0_v2 import load_images, select_pages  # noqa: E402
from week2_attack_v2.loss import ceil_keep_count, kl_clean_adv, share_target_loss  # noqa: E402
from week2_attack_v2.run_v2 import (  # noqa: E402
    answer_logits,
    assemble,
    encode_page,
    flags,
    prefix_scores,
    project,
    subset_questions,
    tuning_questions,
)
from week2_pilot.attack import float_to_page, normalized_tiles, page_to_float  # noqa: E402

BASELINE = ROOT / "gate_experiment" / "results" / "clean_baseline.csv"
RATIO = 0.25
LAM = 0.5
TEMP_C = 0.5
EPS = 8 / 255
FIELDS = [
    "sample_id", "M", "arm", "aux_share", "aux_ceiling", "evidence_share",
    "evidence_keep", "anls_compressed", "anls_full", "csf_strict", "csf_loose", "steps_used",
    "q_star", "soft_share", "stop_reason",
]


def cached_rows() -> tuple[dict, dict]:
    full, fast = {}, {}
    with BASELINE.open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "full":
                full[row["sample_id"]] = row
            if row["method"] == "fastv" and abs(float(row["r"]) - RATIO) < 1e-6:
                fast[row["sample_id"]] = row
    return full, fast


def select_twenty() -> list[dict]:
    full, fast = cached_rows()
    pool = subset_questions()
    forty = tuning_questions(pool, 40)

    def ok(q):
        return float(full[q["sample_id"]]["anls"]) >= 0.5 and float(fast[q["sample_id"]]["anls"]) >= 0.5

    primary = [q for q in forty if ok(q)]
    rng = random.Random(42)
    if len(primary) >= 20:
        picked = rng.sample(primary, 20)
    else:
        picked = list(primary)
        have = {q["sample_id"] for q in picked}
        extra = [q for q in pool if q["sample_id"] not in have and ok(q)]
        rng.shuffle(extra)
        picked.extend(extra[: 20 - len(picked)])
    picked.sort(key=lambda q: q["draw_rank"])
    return picked


def ceiling(mass: int) -> float:
    return 1.0 / (RATIO * mass)


def target(mass: int) -> float:
    return min(0.95, ceiling(mass))


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {(row["sample_id"], row["arm"]) for row in csv.DictReader(handle)}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def attack(model, processor, images, question, aux_idx, max_steps, mass, curve_path, sample_id):
    q_star = target(mass)
    cap = ceiling(mass)
    clean_img = page_to_float(images[aux_idx]).to("cuda")
    delta = torch.zeros_like(clean_img, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
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
    full_ids = torch.cat([base["input_ids"], answer.view(1, -1)], dim=1)
    image_token_id = int(model.config.image_token_index)
    with torch.no_grad():
        text_full = model.get_input_embeddings()(full_ids)
        image_mask = full_ids[0].eq(image_token_id).view(1, -1, 1).expand_as(text_full)
        cache = [None if i == aux_idx else encode_page(model, pixels[i:i + 1], sizes[i:i + 1]).detach() for i in range(len(images))]
        clean_feat = encode_page(
            model, normalized_tiles(clean_img).to(dtype=pixels.dtype).unsqueeze(0), sizes[aux_idx:aux_idx + 1]
        ).detach()
        clean_embeds = assemble(text_full, image_mask, [clean_feat if i == aux_idx else cache[i] for i in range(len(images))])
        log_p_clean = answer_logits(model, clean_embeds, prompt_len - 1, int(answer.numel())).float().log_softmax(-1).detach()

    history = []
    reason = "max_steps"
    t0 = time.time()
    last_share = 0.0
    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        tiles = normalized_tiles((clean_img + delta).clamp(0, 1)).to(dtype=pixels.dtype)
        feat = encode_page(model, tiles.unsqueeze(0), sizes[aux_idx:aux_idx + 1])
        embeds = assemble(text_full, image_mask, [feat if i == aux_idx else cache[i] for i in range(len(images))])
        scores = prefix_scores(model.model.language_model, embeds[:, :prompt_len])
        visual_scores = scores[visual].float()
        temperature = torch.clamp(TEMP_C * visual_scores.std(unbiased=False), min=1e-6).detach()
        loss_share, soft_share = share_target_loss(visual_scores, aux_ids, keep_k, temperature, q_star)
        loss_share.backward()
        last_share = float(soft_share.detach())
        del feat, embeds, scores, visual_scores, loss_share, soft_share
        tiles = normalized_tiles((clean_img + delta).clamp(0, 1)).to(dtype=pixels.dtype)
        feat = encode_page(model, tiles.unsqueeze(0), sizes[aux_idx:aux_idx + 1])
        embeds = assemble(text_full, image_mask, [feat if i == aux_idx else cache[i] for i in range(len(images))])
        logits = answer_logits(model, embeds, prompt_len - 1, int(answer.numel()))
        loss_full = kl_clean_adv(log_p_clean, logits)
        (LAM * loss_full).backward()
        opt.step()
        project(delta, clean_img, EPS)
        history.append(last_share)
        if step % 50 == 0 or step + 1 == max_steps:
            append_jsonl(curve_path, {
                "sample_id": sample_id, "step": step, "soft_share": last_share,
                "q_star": q_star, "ceiling": cap if False else ceiling(mass),
            })
        del feat, embeds, logits, loss_full
        if last_share >= 0.98 * q_star:
            reason = "reached_q"
            break
        if len(history) >= 31 and history[-1] - history[-31] < 0.002:
            reason = "plateau_30"
            break
    if reason != "max_steps":
        append_jsonl(curve_path, {
            "sample_id": sample_id, "step": len(history) - 1, "soft_share": last_share,
            "q_star": q_star, "ceiling": ceiling(mass), "stop": reason,
        })
    page = float_to_page((clean_img + delta).detach())
    del delta, opt
    torch.cuda.empty_cache()
    return page, last_share, len(history), reason, (time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    questions = select_twenty()[: args.limit]
    mine = [q for i, q in enumerate(questions) if i % args.nshards == args.shard]
    full, fast = cached_rows()
    print(json.dumps({"limit": args.limit, "mine": [q["sample_id"] for q in mine]}), flush=True)
    seen = done_keys(args.out)
    pages = load_images(mine)
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    image_token_id = int(model.config.image_token_index)
    for meta in mine:
        if (meta["sample_id"], "attack") in seen:
            continue
        t_sample = time.time()
        images, evidence_idx, _, _ = select_pages(pages[meta["sample_id"]], meta["answer_page_idx"], meta["sample_id"], 768)
        aux_idx = first_other(len(images), evidence_idx)
        mass = len(images)
        cap = ceiling(mass)
        q_star = target(mass)
        cached = fast[meta["sample_id"]]
        full_row = full[meta["sample_id"]]
        clean_c = float(cached["anls"])
        clean_f = float(full_row["anls"])
        if (meta["sample_id"], "clean") not in seen:
            append_csv(args.out, {
                "sample_id": meta["sample_id"], "M": mass, "arm": "clean",
                "aux_share": cached["aux_share"], "aux_ceiling": f"{cap:.6f}",
                "evidence_share": cached["evidence_share"], "evidence_keep": cached["evidence_keep"],
                "anls_compressed": f"{clean_c:.4f}", "anls_full": f"{clean_f:.4f}",
                "csf_strict": 0, "csf_loose": 0, "steps_used": 0,
                "q_star": f"{q_star:.6f}", "soft_share": "", "stop_reason": "cached",
            }, FIELDS)
            seen.add((meta["sample_id"], "clean"))
        page, soft, used, reason, opt_sec = attack(
            model, processor, images, meta["question"], aux_idx, args.steps, mass, args.curve, meta["sample_id"]
        )
        attacked = images.copy()
        attacked[aux_idx] = page
        pred, spans, kept, _ = run_fastv(
            model, processor, attacked, meta["question"], image_token_id, RATIO, evidence_idx, aux_idx
        )
        stats = budget(kept, spans, evidence_idx, aux_idx, [])
        full_pred, _, _, _ = run_plain(
            model, processor, attacked, meta["question"], image_token_id, False, evidence_idx
        )
        anls_c = anls_score(pred, meta["answers"])
        anls_f = anls_score(full_pred, meta["answers"])
        strict, loose = flags(anls_f, anls_c, clean_c)
        append_csv(args.out, {
            "sample_id": meta["sample_id"], "M": mass, "arm": "attack",
            "aux_share": f"{stats['aux_share']:.6f}", "aux_ceiling": f"{cap:.6f}",
            "evidence_share": f"{stats['evidence_share']:.6f}",
            "evidence_keep": f"{stats['evidence_keep']:.6f}",
            "anls_compressed": f"{anls_c:.4f}", "anls_full": f"{anls_f:.4f}",
            "csf_strict": strict, "csf_loose": loose, "steps_used": used,
            "q_star": f"{q_star:.6f}", "soft_share": f"{soft:.6f}", "stop_reason": reason,
        }, FIELDS)
        print(json.dumps({
            "id": meta["sample_id"], "M": mass, "steps": used, "reason": reason,
            "opt_sec": round(opt_sec, 1), "wall_sec": round(time.time() - t_sample, 1),
            "soft": soft, "aux": stats["aux_share"], "keep": stats["evidence_keep"],
            "anls_c": anls_c, "anls_f": anls_f, "csf": strict, "q": q_star,
        }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
