"""Repair only provenance sidecars for the frozen 20-sample unseen manifest.

The sample IDs and their order are treated as immutable.  A byte backup is
written before the JSON is normalized so the repair is recoverable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "true_unseen_manifest.json"
BACKUP = HERE / "true_unseen_manifest.json.pre_provenance_repair"
SIDE = HERE / "true_unseen_manifest.sha256"
SEMANTIC_SIDE = HERE / "true_unseen_manifest.semantic.sha256"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def semantic_digest(obj: dict) -> str:
    # Exclude provenance fields from the semantic identity.  This makes the
    # hash stable under sidecar repair while preserving every sample field.
    body = {k: v for k, v in obj.items() if k not in {
        "manifest_sha256", "semantic_sha256", "file_sha256",
        "provenance_repair", "provenance",
    }}
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return digest(raw)


def main() -> None:
    original = MANIFEST.read_bytes()
    obj = json.loads(original)
    before = [str(row["sample_id"]) for row in obj["samples"]]
    if len(before) != 20 or len(set(before)) != 20:
        raise RuntimeError(f"unexpected unseen manifest size/uniqueness: {len(before)}")
    if not BACKUP.exists():
        BACKUP.write_bytes(original)
    sem = semantic_digest(obj)
    obj["semantic_sha256"] = sem
    obj["provenance_repair"] = {
        "kind": "sidecar-only-repair",
        "sample_ids_preserved": True,
        "sample_order_preserved": True,
    }
    rendered = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    MANIFEST.write_bytes(rendered)
    after = [str(row["sample_id"]) for row in json.loads(rendered)["samples"]]
    if after != before:
        raise RuntimeError("manifest sample IDs/order changed during provenance repair")
    file_hash = digest(rendered)
    SIDE.write_text(file_hash + "  true_unseen_manifest.json\n")
    SEMANTIC_SIDE.write_text(sem + "\n")
    print(json.dumps({
        "status": "REPAIRED_PROVENANCE_ONLY",
        "n": len(after),
        "sample_ids": after,
        "semantic_sha256": sem,
        "file_sha256": file_hash,
        "backup": str(BACKUP),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
