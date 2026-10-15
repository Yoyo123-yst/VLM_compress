"""Freeze an independent holdout before any holdout attack is run."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from run_step0_v2 import draw_questions, load_images, select_pages  # noqa: E402
from week2_attack_v4_answer_aligned.common import sha_image  # noqa: E402

DEV = {"50743", "4912", "56776", "49242", "33923", "5169"}
SEED = 20261008
OUT = ROOT / "week2_attack_v4_relaxed_full" / "holdout"
BASELINE = ROOT / "gate_experiment" / "results" / "clean_baseline.csv"


def main():
    questions = {row["sample_id"]: row for row in draw_questions(400, 42)}
    table: dict[str, dict] = {}
    with BASELINE.open() as handle:
        for row in csv.DictReader(handle):
            sid = row["sample_id"]
            if row["method"] == "full" and row["r"] == "1.0":
                table.setdefault(sid, {})["full"] = float(row["anls"])
                table[sid]["M"] = int(row["M"])
            if row["method"] == "fastv" and row["r"] == "0.25":
                table.setdefault(sid, {})["fastv"] = float(row["anls"])
    eligible = []
    for sid in sorted(table):
        item = table[sid]
        if sid in DEV or sid not in questions:
            continue
        if item.get("M", 0) <= 1:
            continue
        if item.get("full", 0.0) < 0.5 or item.get("fastv", 0.0) < 0.5:
            continue
        eligible.append(sid)
    rng = random.Random(SEED)
    chosen = rng.sample(eligible, min(20, len(eligible)))
    metas = [questions[sid] for sid in chosen]
    pages = load_images(metas)
    samples = []
    for meta in metas:
        sid = meta["sample_id"]
        images, evidence_idx, order, dropped = select_pages(
            pages[sid], meta["answer_page_idx"], sid, 768
        )
        if len(images) <= 1:
            raise RuntimeError(f"{sid} collapsed to one page")
        samples.append({
            "sample_id": sid,
            "M": len(images),
            "clean_full_anls": table[sid]["full"],
            "clean_fastv_anls": table[sid]["fastv"],
            "question": meta["question"],
            "answer_page_idx": meta["answer_page_idx"],
            "evidence_idx": evidence_idx,
            "order": order,
            "dropped": dropped,
            "page_sha": [sha_image(image) for image in images],
        })
    payload = {
        "seed": SEED,
        "source": "gate_experiment/results/clean_baseline.csv intersect draw_questions(400, seed=42)",
        "excluded_dev": sorted(DEV),
        "filters": ["clean Full ANLS >= 0.5", "clean FastV r=0.25 ANLS >= 0.5", "M > 1", "present in seed-42 draw"],
        "eligible_count": len(eligible),
        "selected_count": len(samples),
        "balanced_by_M": False,
        "filtered_by_attack_difficulty": False,
        "samples": samples,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    payload["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUT / "frozen_manifest.csv").open("w", newline="") as handle:
        fields = ["sample_id", "M", "clean_full_anls", "clean_fastv_anls", "evidence_idx", "order"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in samples:
            writer.writerow({key: row[key] if key != "order" else json.dumps(row[key]) for key in fields})
    lines = [
        "# Holdout provenance",
        "",
        "冻结发生在任何 holdout 攻击之前。抽样没有看攻击结果，没有按 M 补齐，也没有按历史攻击难度筛选。",
        "",
        f"- 来源：`clean_baseline.csv` 与 `draw_questions(400, seed=42)` 的交集。",
        f"- 排除开发集：{', '.join(sorted(DEV))}。",
        "- 条件：clean Full ANLS ≥ 0.5，clean FastV r=0.25 ANLS ≥ 0.5，M > 1。",
        f"- 合格样本 {len(eligible)}，种子 {SEED}，抽 {len(samples)} 题。",
        f"- manifest sha256（不含本字段）：{payload['manifest_sha256']}。",
        "",
        "抽中的 ID：",
        "",
    ]
    for row in samples:
        lines.append(f"- {row['sample_id']} M={row['M']} Full {row['clean_full_anls']:.3f} FastV {row['clean_fastv_anls']:.3f}")
    (OUT / "provenance.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"eligible": len(eligible), "selected": [row["sample_id"] for row in samples]}))


if __name__ == "__main__":
    main()
