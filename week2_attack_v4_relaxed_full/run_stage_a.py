"""Replay v4 best images and run the frozen 2x2 / margin / gradient diagnosis.

Does not optimize a new attack and does not write into the v4 directory.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks, install  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import decode, generate, load_model, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.common import (  # noqa: E402
    IDS,
    RATIO,
    V4,
    keep_hash,
    load_frozen_samples,
    sha_image,
    spans_for,
    write_json,
)
from week2_attack_v4_answer_aligned.forward import (  # noqa: E402
    compressed_answer_logits,
    hook_kept,
    keep_index_from_kept,
)
from week2_attack_v4_answer_aligned.loss import answer_margin, official_keep_count  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import (  # noqa: E402
    Session,
    page_uint8,
    project,
    uint8_linf,
)
from week2_pilot.attack import normalized_tiles, page_to_float  # noqa: E402

OUT = ROOT / "week2_attack_v4_relaxed_full"
V4_SCALES = json.loads((V4 / "results" / "lambda_scale.json").read_text())
WRONG = 0.5


def fastv_generate(model, processor, images, question, forced=None):
    clear_hooks(model)
    inputs = prepare(processor, images, question)
    spans = spans_for(model, inputs, len(images))
    state = install(model, "fastv", RATIO, int(model.config.image_token_index))
    state.spans = spans
    if forced is not None:
        target = [int(x) for x in forced]

        def override(visual_index, visual_scores, kept, target=target):
            out = torch.tensor(target, device=kept.device, dtype=torch.long)
            if int(out.numel()) != int(kept.numel()):
                raise RuntimeError(f"forced keep {int(out.numel())} != K {int(kept.numel())}")
            return out

        state.keep_override = override
    tokens = generate(model, inputs)
    pred = decode(processor, tokens)
    kept = list(state.kept_visual)
    clear_hooks(model)
    return pred, kept


def plain_generate(model, processor, images, question):
    clear_hooks(model)
    inputs = prepare(processor, images, question)
    tokens = generate(model, inputs)
    clear_hooks(model)
    return decode(processor, tokens)


def load_smoke() -> dict[str, dict]:
    with (V4 / "results" / "smoke.csv").open() as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def load_step0() -> dict[str, dict]:
    found = {}
    for line in (V4 / "results" / "checkpoints.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["step"]) == 0 and row["sample_id"] not in found:
            found[row["sample_id"]] = row
    return found


def adv_images(item, best: Image.Image):
    images = list(item["images"])
    images[item["aux_idx"]] = best
    return images


def classify_cell(anls: float) -> bool:
    return anls < WRONG


def mechanism(ac_wrong: bool, ca_wrong: bool, aa_wrong: bool) -> str:
    if ac_wrong and not ca_wrong:
        return "content"
    if ca_wrong and not ac_wrong:
        return "selection"
    if (not ac_wrong) and (not ca_wrong) and aa_wrong:
        return "interaction"
    if ac_wrong and ca_wrong:
        return "both_sufficient"
    return "unexplained"


def token_table(processor, token_ids, logits):
    special = set(processor.tokenizer.all_special_ids)
    eos_id = processor.tokenizer.eos_token_id
    margins = answer_margin(logits, token_ids).detach().float().cpu()
    ids = token_ids.detach().cpu().tolist()
    rows = []
    first_flip = None
    for pos, tid in enumerate(ids):
        row_logits = logits[pos].detach().float()
        rival = row_logits.clone()
        rival[tid] = float("-inf")
        rival_id = int(rival.argmax())
        pred_id = int(row_logits.argmax())
        if first_flip is None and pred_id != tid:
            first_flip = pos
        piece = processor.tokenizer.convert_ids_to_tokens(int(tid))
        rows.append({
            "position": pos,
            "token_id": int(tid),
            "piece": piece,
            "special": int(tid) in special,
            "eos": int(tid) == eos_id,
            "gold_logit": float(row_logits[tid]),
            "rival_id": rival_id,
            "rival_piece": processor.tokenizer.convert_ids_to_tokens(rival_id),
            "rival_logit": float(rival[rival_id]),
            "margin": float(margins[pos]),
            "pred_id": pred_id,
            "loss_share": float(margins[pos]) / max(1, len(ids)),
        })
    visible = [row for row in rows if not row["special"] and not row["eos"]]
    values = [row["margin"] for row in rows]
    visible_values = [row["margin"] for row in visible] or [0.0]
    capped = [max(value, -1.0) for value in visible_values]
    return {
        "tokens": rows,
        "first_disagreement": first_flip,
        "mean_all": sum(values) / len(values),
        "mean_visible": sum(visible_values) / len(visible_values),
        "max_all": max(values),
        "capped_mean_visible": sum(capped) / len(capped),
        "eos_count": sum(row["eos"] for row in rows),
        "special_count": sum(row["special"] for row in rows),
        "most_negative": min(rows, key=lambda row: row["margin"]),
    }


def weighted_grad(session, page, lam: float):
    page = page.detach().to("cuda").requires_grad_(True)
    scale_c = float(V4_SCALES["scale_comp_l2"])
    scale_f = float(V4_SCALES["scale_full_l2"])
    loss_c, margin, _, _ = session.compression(page)
    (loss_c / scale_c).backward()
    g_comp = page.grad.detach().float().clone()
    page.grad = None
    loss_f = session.full_kl(page)
    (lam * loss_f / scale_f).backward()
    g_full = page.grad.detach().float().clone()
    denom = float(g_comp.norm() * g_full.norm())
    cosine = float(torch.dot(g_comp.reshape(-1), g_full.reshape(-1)) / denom) if denom > 0 else None
    combined = g_comp + g_full

    def pack(vec):
        return {"l2": float(vec.norm()), "linf": float(vec.abs().max())}

    return {
        "margin": margin,
        "loss_comp": float(loss_c.detach()),
        "loss_full": float(loss_f.detach()),
        "lambda": lam,
        "comp": pack(g_comp),
        "full": pack(g_full),
        "cosine": cosine,
        "combined": pack(combined),
        "full_over_comp": pack(g_full)["l2"] / max(pack(g_comp)["l2"], 1e-12),
    }


def early_page(session, steps: int = 10):
    delta = torch.zeros_like(session.clean_page, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=2.0 / 255.0)
    scale_c = float(V4_SCALES["scale_comp_l2"])
    scale_f = float(V4_SCALES["scale_full_l2"])
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_c, _, _, _ = session.compression(session.clean_page + delta)
        (loss_c / scale_c).backward()
        loss_f = session.full_kl(session.clean_page + delta)
        (1.0 * loss_f / scale_f).backward()
        opt.step()
        project(delta, session.clean_page, 8 / 255)
    return page_uint8(session.clean_page, delta)


def main():
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    smoke = load_smoke()
    step0 = load_step0()
    samples = load_frozen_samples()
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    replay_rows = []
    blocked = []
    for item in samples:
        sid = item["meta"]["sample_id"]
        best = Image.open(V4 / "artifacts" / sid / "best.png").convert("RGB")
        provenance = json.loads((V4 / "artifacts" / sid / "provenance.json").read_text())
        images = adv_images(item, best)
        pred_full = plain_generate(model, processor, images, item["meta"]["question"])
        pred_fast, kept = fastv_generate(model, processor, images, item["meta"]["question"])
        visual_n = sum(end - start for start, end in spans_for(
            model, prepare(processor, images, item["meta"]["question"]), len(images)
        ))
        k_expected = official_keep_count(visual_n, RATIO)
        linf = uint8_linf(item["images"][item["aux_idx"]], best)
        row = {
            "sample_id": sid,
            "pred_full": pred_full,
            "pred_fastv": pred_fast,
            "expected_full": smoke[sid]["pred_full"],
            "expected_fastv": smoke[sid]["pred_fastv"],
            "full_match": pred_full == smoke[sid]["pred_full"],
            "fastv_match": pred_fast == smoke[sid]["pred_fastv"],
            "keep_hash": keep_hash(kept),
            "keep_hash_match": keep_hash(kept) == smoke[sid]["keep_hash"],
            "k": len(kept),
            "k_match": False,
            "k_provenance": len(provenance["keep"]),
            "sha_match": sha_image(best) == provenance["best_sha256"],
            "linf": linf,
            "linf_match": abs(linf - float(smoke[sid]["linf_uint8"])) < 1e-6,
            "valid": bool(pred_full.strip() and pred_fast.strip()),
        }
        row["k_match"] = row["k"] == k_expected == row["k_provenance"]
        reasons = []
        for key in ("full_match", "fastv_match", "keep_hash_match", "k_match", "sha_match", "linf_match", "valid"):
            if not row[key]:
                reasons.append(key)
        row["reasons"] = reasons
        if reasons:
            blocked.append(sid)
        replay_rows.append(row)
        print(json.dumps({"replay": sid, "ok": not reasons, "reasons": reasons}), flush=True)
    (OUT / "diagnostics").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "diagnostics" / "replay.json", {"blocked": blocked, "samples": replay_rows})
    if blocked:
        write_json(OUT / "results" / "stage_a_decision.json", {
            "stage": "A",
            "health": "replay_failed",
            "completed_samples": 6 - len(blocked),
            "decision": "BLOCKED_ON_DIAGNOSTIC_REPLAY",
            "evidence": blocked,
            "next_action": "stop",
            "next_command": "",
        })
        raise SystemExit(2)

    cells = []
    margin_path = OUT / "diagnostics" / "token_margin.jsonl"
    grad_path = OUT / "diagnostics" / "weighted_gradients.jsonl"
    margin_path.write_text("")
    grad_path.write_text("")
    for item in samples:
        sid = item["meta"]["sample_id"]
        best = Image.open(V4 / "artifacts" / sid / "best.png").convert("RGB")
        clean_pred, clean_keep = fastv_generate(model, processor, item["images"], item["meta"]["question"])
        attacked = adv_images(item, best)
        adv_pred, adv_keep = fastv_generate(model, processor, attacked, item["meta"]["question"])
        if clean_pred != step0[sid]["pred_fastv"]:
            raise SystemExit(f"CC mismatch {sid}: {clean_pred!r} != {step0[sid]['pred_fastv']!r}")
        if adv_pred != smoke[sid]["pred_fastv"]:
            raise SystemExit(f"AA mismatch {sid}")
        ac_pred, _ = fastv_generate(model, processor, attacked, item["meta"]["question"], clean_keep)
        ca_pred, _ = fastv_generate(model, processor, item["images"], item["meta"]["question"], adv_keep)
        answers = item["meta"]["answers"]
        scores = {
            "CC": anls_score(clean_pred, answers),
            "CA": anls_score(ca_pred, answers),
            "AC": anls_score(ac_pred, answers),
            "AA": anls_score(adv_pred, answers),
        }
        preds = {"CC": clean_pred, "CA": ca_pred, "AC": ac_pred, "AA": adv_pred}
        ac_wrong = classify_cell(scores["AC"])
        ca_wrong = classify_cell(scores["CA"])
        aa_wrong = classify_cell(scores["AA"])
        label = mechanism(ac_wrong, ca_wrong, aa_wrong)
        swapped = len(set(clean_keep) ^ set(adv_keep))
        cells.append({
            "sample_id": sid,
            "pred_CC": preds["CC"], "pred_CA": preds["CA"], "pred_AC": preds["AC"], "pred_AA": preds["AA"],
            "anls_CC": scores["CC"], "anls_CA": scores["CA"], "anls_AC": scores["AC"], "anls_AA": scores["AA"],
            "ac_wrong": int(ac_wrong), "ca_wrong": int(ca_wrong), "aa_wrong": int(aa_wrong),
            "mechanism": label,
            "keep_symdiff": swapped,
            "k": len(clean_keep),
        })
        print(json.dumps({"cell": sid, "mechanism": label, "scores": scores, "symdiff": swapped}), flush=True)

        clear_hooks(model)
        inputs = prepare(processor, item["images"], item["meta"]["question"])
        with torch.inference_mode():
            raw = generate(model, inputs)
        answer = torch.tensor(raw.detach().cpu().tolist(), dtype=torch.long)
        session = Session(model, processor, item, answer)
        for name, image in (("clean", item["images"][item["aux_idx"]]), ("best", best)):
            page = page_to_float(image).to("cuda")
            clear_hooks(model)
            tiles = normalized_tiles(page).to(dtype=session.pixels.dtype)
            batch = session.pixels.detach().clone()
            batch[session.aux_idx] = tiles.detach()
            hooked = dict(session.inputs)
            hooked["pixel_values"] = batch
            kept = hook_kept(model, hooked, session.spans, RATIO)
            keep_index = keep_index_from_kept(kept, session.spans, session.full_ids.shape[1], page.device)
            embeds = session.embeds_from(page)
            with torch.no_grad():
                logits = compressed_answer_logits(
                    model, embeds, session.prompt_len, keep_index, int(answer.numel())
                ).float()
            table = token_table(processor, answer.to(logits.device), logits)
            record = {"sample_id": sid, "image": name, "summary": {k: v for k, v in table.items() if k != "tokens"}, "tokens": table["tokens"]}
            with margin_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            grad = weighted_grad(session, page, 1.0)
            grad.update({"sample_id": sid, "image": name})
            with grad_path.open("a") as handle:
                handle.write(json.dumps(grad) + "\n")
            print(json.dumps({"grad": sid, "image": name, "ratio": grad["full_over_comp"], "mean_visible": table["mean_visible"]}), flush=True)
        early = early_page(session, 10)
        grad = weighted_grad(session, page_to_float(early).to("cuda"), 1.0)
        grad.update({"sample_id": sid, "image": "step10"})
        with grad_path.open("a") as handle:
            handle.write(json.dumps(grad) + "\n")
        torch.cuda.empty_cache()

    with (OUT / "diagnostics" / "content_selection_2x2.csv").open("w", newline="") as handle:
        fields = list(cells[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cells)
    lines = ["# 阶段 A 机制诊断", "", "重放门禁通过：6/6 best checkpoint 的 Full/FastV 解码、keep hash、K、SHA-256 和 uint8 L∞ 与 v4 记录一致。", ""]
    for row in cells:
        lines.append(
            f"- {row['sample_id']}: {row['mechanism']}。CC={row['pred_CC']!r} ({row['anls_CC']:.3f}), "
            f"CA={row['pred_CA']!r} ({row['anls_CA']:.3f}), AC={row['pred_AC']!r} ({row['anls_AC']:.3f}), "
            f"AA={row['pred_AA']!r} ({row['anls_AA']:.3f}), keep 对称差 {row['keep_symdiff']} / {row['k']}。"
        )
    lines.append("")
    lines.append("错的定义是 ANLS < 0.5。content = 只换图像内容就错；selection = 只换保留集就错；interaction = 单独都不错、合在一起才错；both_sufficient = 两条都足以打错。")
    (OUT / "diagnostics" / "diagnosis.md").write_text("\n".join(lines) + "\n")
    write_json(OUT / "results" / "stage_a_decision.json", {
        "stage": "A",
        "health": "PASS",
        "completed_samples": 6,
        "invalid_outputs": 0,
        "replay_pass": 6,
        "decision": "STAGE_A_PASS",
        "evidence": [f"{row['sample_id']}:{row['mechanism']}" for row in cells],
        "next_action": "run lambda_full=0.25",
        "next_command": "run_lambda.py --lambda-full 0.25 --steps 2 --ids 50743 --tag smoke2",
    })
    print("STAGE_A_PASS", flush=True)


if __name__ == "__main__":
    main()
