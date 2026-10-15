"""Read-only dose diagnosis. Does not rerun PGD and does not write historical trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7_budget_conditioned"
V7 = ROOT / "week2_attack_v7_pristine60"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks  # noqa: E402
from metrics_task import anls_score  # noqa: E402
from run_causal_v2 import budget, run_fastv, run_plain  # noqa: E402
from week2_attack_v4_answer_aligned.common import assert_protected, jaccard, keep_hash, sha_image  # noqa: E402
from week2_attack_v4_answer_aligned.forward import compressed_answer_logits, full_answer_logits, keep_index_from_kept  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import uint8_linf  # noqa: E402
from week2_attack_v7_budget_conditioned.constants import CANDIDATE_R, EPSILON_MARGIN  # noqa: E402
from week2_attack_v7_budget_conditioned.fdp import certificate, hard_margin, sequence_margin, soft_margin  # noqa: E402
from week2_attack_v7_pristine60.load_samples import load_items, manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen, make_session  # noqa: E402
from week2_pilot.attack import page_to_float  # noqa: E402


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_ratios() -> list[float]:
    removed = set(json.loads((HERE / "ratio_legality.json").read_text()).get("removed_r", []))
    return [ratio for ratio in CANDIDATE_R if ratio not in removed]


def locate_noise(sample_id: str) -> Path:
    hits = list((V7 / "controls" / "single").glob(f"shard_s*/{sample_id}/noise.png"))
    if len(hits) != 1:
        raise RuntimeError(f"noise png count for {sample_id} is {len(hits)}")
    return hits[0]


def pack_margin(values: torch.Tensor) -> dict:
    data = [float(x) for x in values.detach().float().cpu().tolist()]
    return {"values": data, "mean": float(sum(data) / len(data)), "min": min(data), "max": max(data)}


def measure(session, image: Image.Image, tokens: torch.Tensor, ratios: list[float]) -> dict:
    attacked = list(session.images)
    attacked[session.aux_idx] = image
    after = [sha_image(page) for page in attacked]
    assert_protected(session.item["page_sha"], after, session.aux_idx)
    token_id = int(session.model.config.image_token_index)
    question = session.question
    evidence = session.item["evidence_idx"]
    aux = session.aux_idx
    pred_full, spans_full, _, _ = run_plain(
        session.model, session.processor, attacked, question, token_id, False, evidence
    )
    by_r = {}
    for ratio in ratios:
        pred, spans, kept, _ = run_fastv(
            session.model, session.processor, attacked, question, token_id, ratio, evidence, aux
        )
        if spans != session.spans:
            raise RuntimeError(f"span mismatch at r={ratio}")
        stats = budget(kept, spans, evidence, aux, [])
        by_r[f"{ratio:.2f}"] = {
            "pred_fastv": pred,
            "anls_fastv": anls_score(pred, session.item["meta"]["answers"]),
            "kept": [int(x) for x in kept],
            "keep_hash": keep_hash([int(x) for x in kept]),
            "k": len(kept),
            "evidence_share": stats["evidence_share"],
            "aux_share": stats["aux_share"],
        }
    page = page_to_float(image).to(session.clean_page.device)
    clear_hooks(session.model)
    embeds = session.embeds_from(page)
    full_logits = full_answer_logits(
        session.model, embeds, session.prompt_len - 1, int(tokens.numel())
    ).float()
    full_hard = hard_margin(full_logits, tokens)
    full_soft = soft_margin(full_logits, tokens)
    for ratio in ratios:
        key = f"{ratio:.2f}"
        keep_index = keep_index_from_kept(by_r[key]["kept"], session.spans, session.full_ids.shape[1], embeds.device)
        logits = compressed_answer_logits(
            session.model, embeds, session.prompt_len, keep_index, int(tokens.numel())
        ).float()
        soft = soft_margin(logits, tokens)
        by_r[key]["m_r"] = float(sequence_margin(soft))
        by_r[key]["hard_margin"] = pack_margin(hard_margin(logits, tokens))
        by_r[key]["soft_margin"] = pack_margin(soft)
    return {
        "pred_full": pred_full,
        "anls_full": anls_score(pred_full, session.item["meta"]["answers"]),
        "valid": bool(pred_full.strip()),
        "linf": uint8_linf(session.images[session.aux_idx], image),
        "m_infinity": float(sequence_margin(full_soft)),
        "full_hard": pack_margin(full_hard),
        "full_soft": pack_margin(full_soft),
        "certificate": certificate(full_hard),
        "cone_holds": bool(torch.all(full_hard > EPSILON_MARGIN).item()),
        "by_r": by_r,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, required=True)
    args = parser.parse_args()
    ratios = active_ratios()
    if not ratios:
        raise SystemExit("no legal FastV ratio remains")
    specs = [row for index, row in enumerate(manifest()["samples"]) if index % args.nshards == args.shard]
    selected = {
        row["sample_id"]: row
        for row in __import__("csv").DictReader((V7 / "attacks" / "selector" / "selected.csv").open())
    }
    out_dir = HERE / "stage1" / f"shard_s{args.shard}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [spec for spec in specs if not (out_dir / f"{spec['sample_id']}.json").exists()]
    if not pending:
        print(json.dumps({"shard": args.shard, "done": len(specs), "skipped": True}))
        return
    items = {item["meta"]["sample_id"]: item for item in load_items(pending)}
    model, processor = load_model_frozen()
    for spec in pending:
        sid = spec["sample_id"]
        started = time.time()
        item = items[sid]
        session = make_session(model, processor, item)
        tokens = session.answer_tokens
        b_path = ROOT / selected[sid]["artifact"] / selected[sid]["png"]
        noise_path = locate_noise(sid)
        images = {
            "clean": item["images"][item["aux_idx"]],
            "B": Image.open(b_path).convert("RGB"),
            "random_noise": Image.open(noise_path).convert("RGB"),
        }
        measured = {name: measure(session, image, tokens, ratios) for name, image in images.items()}
        clean_text = measured["clean"]["pred_full"]
        record = {
            "sample_id": sid,
            "M": spec["M"],
            "complete": True,
            "seconds": round(time.time() - started, 2),
            "clean_full_text": clean_text,
            "png": {
                "B": file_sha(b_path),
                "random_noise": file_sha(noise_path),
                "clean_page": sha_image(images["clean"]),
            },
            "images": measured,
        }
        for name, row in measured.items():
            row["full_exact"] = row["pred_full"] == clean_text
            row["margin_text_conflict"] = bool(row["full_exact"] != row["cone_holds"])
            row["replay"] = "DISK_RELOAD_GENERATE" if name != "clean" else "DATASET_PAGE"
            for key, bucket in row["by_r"].items():
                clean_bucket = measured["clean"]["by_r"][key]
                bucket["jaccard_vs_clean"] = jaccard(clean_bucket["kept"], bucket["kept"])
                bucket["strict"] = int(
                    name != "clean"
                    and row["full_exact"]
                    and row["anls_full"] >= 0.5
                    and bucket["anls_fastv"] < 0.5
                    and row["valid"]
                    and str(bucket["pred_fastv"]).strip()
                    and 0 < row["linf"] <= 8 / 255 + 1e-6
                    and measured["clean"]["anls_full"] >= 0.5
                    and clean_bucket["anls_fastv"] >= 0.5
                )
        for row in measured.values():
            for bucket in row["by_r"].values():
                bucket.pop("kept")
        (out_dir / f"{sid}.json").write_text(json.dumps(record) + "\n")
        print(json.dumps({"id": sid, "seconds": record["seconds"]}), flush=True)
        del session
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
