"""Build pristine eligible pool and freeze 40-sample manifest."""

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
from week2_attack_v4_answer_aligned.loss import official_keep_count  # noqa: E402
from week2_attack_v5_pristine40.common import (  # noqa: E402
    BASELINE,
    POOL_SEED,
    RATIO,
    TOKENS_PER_PAGE,
    V5,
    WRONG,
    code_hashes,
    load_scales,
    write_json,
)


def load_excluded() -> set[str]:
    payload = json.loads((V5 / "exclusions" / "historical_attack_ids.json").read_text())
    return {row["sample_id"] for row in payload["samples"]}


def baseline_table() -> dict[str, dict]:
    table: dict[str, dict] = {}
    with BASELINE.open() as handle:
        for row in csv.DictReader(handle):
            sid = row["sample_id"]
            if row["method"] == "full" and row["r"] == "1.0":
                table.setdefault(sid, {})["full"] = row
                table[sid]["M"] = int(row["M"])
            if row["method"] == "fastv" and abs(float(row["r"]) - RATIO) < 1e-6:
                table.setdefault(sid, {})["fastv"] = row
    return table


def main():
    excluded = load_excluded()
    questions = {q["sample_id"]: q for q in draw_questions(400, 42)}
    base = baseline_table()
    all_clean = []
    pristine = []
    overlap = []
    for sid in sorted(base, key=int):
        item = base[sid]
        if "full" not in item or "fastv" not in item:
            continue
        if int(item["M"]) <= 1:
            continue
        if float(item["full"]["anls"]) < WRONG or float(item["fastv"]["anls"]) < WRONG:
            continue
        if sid not in questions:
            continue
        row = {
            "sample_id": sid,
            "M": int(item["M"]),
            "clean_full_anls": float(item["full"]["anls"]),
            "clean_fastv_anls": float(item["fastv"]["anls"]),
            "clean_full_pred": item["full"]["pred"],
            "clean_fastv_pred": item["fastv"]["pred"],
            "in_draw_400": True,
        }
        all_clean.append(row)
        if sid in excluded:
            overlap.append({**row, "reason": "historical_attack_exclusion"})
        else:
            pristine.append(row)
    pool_dir = V5 / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    fields = list(all_clean[0].keys()) if all_clean else []
    with (pool_dir / "all_clean_eligible.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(all_clean)
    with (pool_dir / "all_pristine_eligible.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(pristine)
    with (pool_dir / "excluded_overlap.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields + ["reason"])
        w.writeheader()
        w.writerows(overlap)
    by_excl = {r["sample_id"]: r["sources"] for r in json.loads((V5 / "exclusions" / "historical_attack_ids.json").read_text())["samples"]}

    def has_non_oracle_attack(sid: str) -> bool:
        for source in by_excl.get(sid, []):
            if "oracle_hijack" not in source:
                return True
        return False

    pristine_attack_only = [r for r in all_clean if not has_non_oracle_attack(r["sample_id"])]

    audit_lines = [
        "# Pool audit",
        "",
        f"- Clean eligible (M>1, baseline ANLS ok, in draw): {len(all_clean)}",
        f"- Historical overlap removed (full union incl. oracle inspection): {len(overlap)}",
        f"- Pristine eligible (full union): **{len(pristine)}**",
        f"- Diagnostic only — pristine if excluding oracle-only inspection: {len(pristine_attack_only)} (not used for sampling)",
        f"- Pool seed for selection: {POOL_SEED}",
        "",
    ]
    if len(pristine) < 40:
        audit_lines.append("**BLOCKED_INSUFFICIENT_PRISTINE_POOL** — cannot draw 40.")
        (pool_dir / "pool_audit.md").write_text("\n".join(audit_lines) + "\n")
        write_json(V5 / "pool" / "decision.json", {"decision": "BLOCKED_INSUFFICIENT_PRISTINE_POOL", "pristine_n": len(pristine)})
        print(json.dumps({"decision": "BLOCKED", "pristine": len(pristine)}))
        return
    rng = random.Random(POOL_SEED)
    chosen_ids = sorted(rng.sample([r["sample_id"] for r in pristine], 40), key=int)
    metas = [questions[sid] for sid in chosen_ids]
    pages = load_images(metas)
    scales = load_scales()
    excl_hash = json.loads((V5 / "exclusions" / "historical_attack_ids.json").read_text())["union_sha256"]
    samples = []
    for sid in chosen_ids:
        meta = questions[sid]
        images, evidence_idx, order, dropped = select_pages(
            pages[sid], meta["answer_page_idx"], sid, 768
        )
        aux_idx = next(i for i in range(len(images)) if i != evidence_idx)
        m = len(images)
        expected_k = official_keep_count(m * TOKENS_PER_PAGE, RATIO)
        b = base[sid]
        samples.append({
            "sample_id": sid,
            "question": meta["question"],
            "answers": meta.get("answers", []),
            "M": m,
            "answer_page_idx": meta["answer_page_idx"],
            "evidence_idx": evidence_idx,
            "aux_idx": aux_idx,
            "order": order,
            "dropped": dropped,
            "page_sha": [sha_image(im) for im in images],
            "clean_full_pred": b["full"]["pred"],
            "clean_fastv_pred": b["fastv"]["pred"],
            "clean_full_anls": float(b["full"]["anls"]),
            "clean_fastv_anls": float(b["fastv"]["anls"]),
            "expected_k": expected_k,
        })
    manifest = {
        "pool_seed": POOL_SEED,
        "n_selected": 40,
        "pristine_pool_size": len(pristine),
        "exclusion_union_sha256": excl_hash,
        "lambda_full": 0.25,
        "steps": 200,
        "eps": 8 / 255,
        "primary_selector": "fixed_step_200",
        "secondary_selector": "prospective_min_compressed_margin",
        "scales": scales,
        "code_hashes": code_hashes(),
        "samples": samples,
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    frozen = V5 / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    write_json(frozen / "pristine40_manifest.json", manifest)
    with (frozen / "pristine40_manifest.csv").open("w", newline="") as handle:
        fields = ["sample_id", "M", "aux_idx", "evidence_idx", "expected_k", "clean_full_anls", "clean_fastv_anls"]
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for row in samples:
            w.writerow({k: row[k] for k in fields})
    frozen_doc = {
        **manifest,
        "role": "pristine40 fixed-endpoint validation",
        "verify_only_after_write": True,
    }
    write_json(frozen / "FROZEN.json", frozen_doc)
    m_counts = {2: 0, 3: 0, 4: 0}
    for row in samples:
        m_counts[row["M"]] = m_counts.get(row["M"], 0) + 1
    audit_lines.append(f"- Selected 40 with M distribution: {m_counts}")
    audit_lines.append(f"- Manifest SHA-256: `{manifest['manifest_sha256']}`")
    (pool_dir / "pool_audit.md").write_text("\n".join(audit_lines) + "\n")
    prov = ["# Frozen pristine40", "", f"Manifest hash: {manifest['manifest_sha256']}", "", "IDs:"]
    for row in samples:
        prov.append(f"- {row['sample_id']} M={row['M']} K={row['expected_k']}")
    (frozen / "provenance.md").write_text("\n".join(prov) + "\n")
    print(json.dumps({"selected": 40, "pristine_pool": len(pristine), "manifest_sha": manifest["manifest_sha256"], "M": m_counts}))


if __name__ == "__main__":
    main()
