"""Shared loading and bookkeeping for the frozen 6-question set."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V4 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))
sys.path.insert(0, str(ROOT))

from run_causal_v2 import spans_of  # noqa: E402
from run_step0_v2 import draw_questions, load_images, select_pages  # noqa: E402
from week2_pilot.attack import page_to_float  # noqa: E402

BASELINE = ROOT / "gate_experiment" / "results" / "clean_baseline.csv"
MODEL_PATH = "/home/yuesiting/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf"
RATIO = 0.25
EPS = 8 / 255
SEED = 42
IDS = ["50743", "4912", "56776", "49242", "33923", "5169"]
GROUPS = {
    "50743": "oracle_reachable_current_fail",
    "4912": "oracle_reachable_current_fail",
    "56776": "oracle_reachable_current_fail",
    "49242": "historical_success",
    "33923": "historical_success",
    "5169": "oracle_resistant",
}
REASONS = {
    "50743": "当前攻击未成功，oracle 全转移后成功",
    "4912": "当前攻击未成功，oracle 全转移后成功",
    "56776": "当前攻击未成功，oracle 全转移后成功",
    "49242": "当前攻击已经成功，用来检查新方法是否保住已有能力",
    "33923": "当前攻击已经成功，用来检查新方法是否保住已有能力",
    "5169": "当前攻击未成功，oracle 全转移后仍未成功，用来看纯预算删除的边界",
}
CALIBRATION_ID = "50743"
CHECKPOINTS = [0, 10, 25, 50, 75, 100, 150, 200]


def code_version() -> str:
    digest = hashlib.sha256()
    for path in sorted(V4.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for path in sorted((V4 / "tests").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sha_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def keep_hash(kept: list[int]) -> str:
    raw = ",".join(str(x) for x in kept).encode()
    return hashlib.sha256(raw).hexdigest()


def jaccard(left: list[int], right: list[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def baseline_table() -> dict[str, dict[str, dict]]:
    import csv

    table: dict[str, dict[str, dict]] = {}
    with BASELINE.open() as handle:
        for row in csv.DictReader(handle):
            if row["sample_id"] not in IDS:
                continue
            if row["method"] == "full":
                table.setdefault(row["sample_id"], {})["full"] = row
            if row["method"] == "fastv" and abs(float(row["r"]) - RATIO) < 1e-6:
                table.setdefault(row["sample_id"], {})["fastv"] = row
    return table


def load_frozen_samples() -> list[dict]:
    questions = {q["sample_id"]: q for q in draw_questions(400, SEED)}
    missing = [sample_id for sample_id in IDS if sample_id not in questions]
    if missing:
        raise RuntimeError(f"frozen ids missing from the seed-42 draw: {missing}")
    metas = [questions[sample_id] for sample_id in IDS]
    pages = load_images(metas)
    base = baseline_table()
    samples = []
    for meta in metas:
        images, evidence_idx, order, dropped = select_pages(
            pages[meta["sample_id"]], meta["answer_page_idx"], meta["sample_id"], 768
        )
        aux_idx = next(i for i in range(len(images)) if i != evidence_idx)
        samples.append({
            "meta": meta,
            "images": images,
            "evidence_idx": evidence_idx,
            "aux_idx": aux_idx,
            "order": order,
            "dropped": dropped,
            "page_sha": [sha_image(image) for image in images],
            "baseline_full": base[meta["sample_id"]]["full"],
            "baseline_fastv": base[meta["sample_id"]]["fastv"],
        })
    return samples


def frozen_document(samples: list[dict]) -> dict:
    return {
        "role": "development diagnostic set, not a generalization claim",
        "r": RATIO,
        "eps": EPS,
        "seed": SEED,
        "calibration_sample": CALIBRATION_ID,
        "lambda_full_after_grad_norm": 1.0,
        "margin_definition": "mean_t (z_t[y_ref,t] - max_{v != y_ref,t} z_t[v])",
        "margin_drop_for_surrogate_label": 1.0,
        "model_path": MODEL_PATH,
        "processor": {
            "image_grid_pinpoints": [[768, 768]],
            "dtype": "bfloat16",
            "attention": "sdpa",
            "max_new_tokens": 16,
            "do_sample": False,
        },
        "code_version": code_version(),
        "git": None,
        "samples": [
            {
                "sample_id": item["meta"]["sample_id"],
                "group": GROUPS[item["meta"]["sample_id"]],
                "reason": REASONS[item["meta"]["sample_id"]],
                "question": item["meta"]["question"],
                "answers": item["meta"]["answers"],
                "order": item["order"],
                "evidence_idx_after_select": item["evidence_idx"],
                "aux_idx": item["aux_idx"],
                "page_sha256": item["page_sha"],
                "baseline_full_pred": item["baseline_full"]["pred"],
                "baseline_fastv_pred": item["baseline_fastv"]["pred"],
                "baseline_full_anls": float(item["baseline_full"]["anls"]),
                "baseline_fastv_anls": float(item["baseline_fastv"]["anls"]),
            }
            for item in samples
        ],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def image_mask_of(model, input_ids):
    image_token_id = int(model.config.image_token_index)
    text = model.get_input_embeddings()(input_ids)
    mask = input_ids[0].eq(image_token_id).view(1, -1, 1).expand_as(text)
    return text, mask


def spans_for(model, inputs, n_images):
    return spans_of(inputs, int(model.config.image_token_index), n_images)


def assert_protected(before: list[str], after: list[str], aux_idx: int) -> None:
    for index, (left, right) in enumerate(zip(before, after)):
        if index == aux_idx:
            continue
        if left != right:
            raise RuntimeError(f"protected page {index} hash changed")


def finite_nonzero(grad: torch.Tensor) -> tuple[bool, float]:
    if grad is None or not torch.isfinite(grad).all():
        return False, float("nan")
    norm = float(grad.detach().float().norm())
    return norm > 0.0, norm
