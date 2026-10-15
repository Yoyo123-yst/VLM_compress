"""Scan week2 attack trees and build the historical attack exclusion union."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5 = ROOT / "week2_attack_v5_pristine40"
SCAN_ROOTS = [
    ROOT / "week2_pilot",
    ROOT / "week2_attack_v2",
    ROOT / "week2_attack_v3",
    ROOT / "week2_attack_v3_quick",
    ROOT / "week2_attack_v4_answer_aligned",
    ROOT / "week2_attack_v4_relaxed_full",
]
KNOWN_DEV = {"50743", "4912", "56776", "49242", "33923", "5169"}
SID_RE = re.compile(r"^\d{1,6}$")

# Derived / analysis exports — IDs already appear in attack runs or holdout manifest.
SKIP_PATH_FRAGMENTS = (
    "selector_audit/",
    "/results/all_samples.csv",
    "/results/all_checkpoints.csv",
    "/results/holdout/paired.csv",
    "/diagnostics/",
    "content_selection_2x2",
)

# Clean-only evaluation tables inside week2 (not adversarial optimization).
SKIP_EXACT_NAMES = {
    "clean_baseline.csv",
}


def should_skip(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    if path.name in SKIP_EXACT_NAMES:
        return True
    return any(fragment in rel for fragment in SKIP_PATH_FRAGMENTS)


def add(record: dict, sid: str, source: str):
    sid = str(sid).strip()
    if not sid or not SID_RE.match(sid):
        return
    entry = record.setdefault(
        sid,
        {"sample_id": sid, "sources": [], "reason": "historically_attacked_or_inspected"},
    )
    if source not in entry["sources"]:
        entry["sources"].append(source)


def scan_csv(path: Path, record: dict):
    if should_skip(path):
        return
    try:
        with path.open() as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "sample_id" not in reader.fieldnames:
                return
            has_arm = "arm" in reader.fieldnames
            for row in reader:
                if has_arm and row.get("arm") == "clean":
                    continue
                add(record, row["sample_id"], f"csv:{path.relative_to(ROOT)}")
    except Exception:
        return


def scan_jsonl(path: Path, record: dict):
    if should_skip(path):
        return
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "sample_id" in row:
            add(record, row["sample_id"], f"jsonl:{path.relative_to(ROOT)}")


def scan_json_manifest(path: Path, record: dict):
    if should_skip(path):
        return
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return
    rel = str(path.relative_to(ROOT))
    for key in ("samples", "ids"):
        if key in payload and isinstance(payload[key], list):
            for item in payload[key]:
                if isinstance(item, dict) and "sample_id" in item:
                    add(record, item["sample_id"], f"json:{rel}")
                elif isinstance(item, str):
                    add(record, item, f"json:{rel}")
    if "excluded_dev" in payload:
        for sid in payload["excluded_dev"]:
            add(record, sid, f"json:{rel}:excluded_dev")


def scan_artifact_dirs(root: Path, record: dict):
    for sub in (root / "artifacts", root / "runs"):
        if not sub.exists():
            continue
        for path in sub.rglob("*"):
            if path.is_dir() and SID_RE.match(path.name):
                if any(path.glob("*.png")) or (path / "provenance.json").exists():
                    add(record, path.name, f"artifact_dir:{path.relative_to(ROOT)}")


def scan_tree(root: Path, record: dict):
    if not root.exists():
        return
    scan_artifact_dirs(root, record)
    for path in root.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        name = path.name.lower()
        if name.endswith(".csv"):
            scan_csv(path, record)
        elif name.endswith(".jsonl"):
            scan_jsonl(path, record)
        elif name in ("frozen_manifest.json", "pristine40_manifest.json", "frozen.json", "frozens.json"):
            scan_json_manifest(path, record)
        elif name == "provenance.json":
            try:
                row = json.loads(path.read_text())
                if "sample_id" in row:
                    add(record, row["sample_id"], f"provenance:{path.relative_to(ROOT)}")
            except Exception:
                pass


def source_category(source: str) -> str:
    if "oracle_hijack" in source:
        return "oracle_inspection"
    if "artifact_dir:" in source or "checkpoints" in source or "attack" in source or "smoke" in source:
        return "adversarial_run"
    if "shard_s" in source:
        return "adversarial_run"
    return "other"


def main():
    record: dict[str, dict] = {}
    per_root_counts = {}
    for scan_root in SCAN_ROOTS:
        before = len(record)
        scan_tree(scan_root, record)
        if scan_root.name == "week2_attack_v4_answer_aligned":
            for sid in KNOWN_DEV:
                add(record, sid, "hardcoded:v4_dev_IDS")
        per_root_counts[scan_root.name] = len(record) - before
    holdout = ROOT / "week2_attack_v4_relaxed_full" / "holdout" / "frozen_manifest.json"
    if holdout.exists():
        scan_json_manifest(holdout, record)
    out = V5 / "exclusions"
    out.mkdir(parents=True, exist_ok=True)
    items = [record[sid] for sid in sorted(record, key=int)]
    category_counts = defaultdict(int)
    for row in items:
        cats = {source_category(s) for s in row["sources"]}
        for cat in cats:
            category_counts[cat] += 1
    payload = {
        "n_unique": len(items),
        "per_root_new_ids_approx": per_root_counts,
        "covers_v4_dev_six": all(sid in record for sid in KNOWN_DEV),
        "holdout_20_covered": sum(
            1 for sid in json.loads(holdout.read_text())["samples"] if sid["sample_id"] in record
        )
        if holdout.exists()
        else None,
        "source_category_unique_ids": dict(category_counts),
        "samples": items,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    payload["union_sha256"] = hashlib.sha256(raw).hexdigest()
    (out / "historical_attack_ids.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out / "historical_attack_ids.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "reason", "sources"])
        writer.writeheader()
        for row in items:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "reason": row["reason"],
                    "sources": "|".join(row["sources"]),
                }
            )
    lines = [
        "# Historical attack exclusion union",
        "",
        f"- Unique IDs: **{len(items)}**",
        f"- Union SHA-256: `{payload['union_sha256']}`",
        f"- Covers v4 dev six: {payload['covers_v4_dev_six']}",
        f"- Holdout 20 in union: {payload.get('holdout_20_covered')}/20",
        "",
        "## Per-scan-root approximate new IDs",
        "",
    ]
    for name, count in per_root_counts.items():
        lines.append(f"- {name}: +{count}")
    lines.extend(["", "## Source categories (ID may have multiple)", ""])
    for name, count in sorted(category_counts.items()):
        lines.append(f"- {name}: {count}")
    (out / "exclusion_sources.md").write_text("\n".join(lines) + "\n")
    audit = {
        "scan_roots": [str(p) for p in SCAN_ROOTS],
        "n_unique": len(items),
        "union_sha256": payload["union_sha256"],
        "covers_v4_dev": payload["covers_v4_dev_six"],
        "holdout_covered": payload.get("holdout_20_covered"),
        "source_category_unique_ids": dict(category_counts),
    }
    (out / "exclusion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
