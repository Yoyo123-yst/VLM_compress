"""Apply the frozen dev12 gates. Does not change thresholds."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v6c_split_calibration")
DEV = ["1025", "2008", "7006", "39204", "47450", "47567", "50779", "51133", "60700", "6194", "62941", "64506"]
A_JACCARD = 0.8355726620046946


def load_variant(name: str):
    selected = {row["sample_id"]: row for row in csv.DictReader((ROOT / name / "selector" / "selected.csv").open())}
    replay = {}
    for path in (ROOT / name / "selector").glob("replay_s*.json"):
        for row in json.loads(path.read_text()):
            replay[row["sample_id"]] = row
    cells = []
    keeps = []
    for path in (ROOT / name / "mechanism").glob("cells_s*.json"):
        payload = json.loads(path.read_text())
        cells.extend(payload["rows"])
        keeps.extend(payload["keeps"])
    cells = {row["sample_id"]: row for row in cells}
    keeps = {row["sample_id"]: row for row in keeps}
    return selected, replay, cells, keeps


def strict_of(row, replay, clean):
    rep = replay[row["sample_id"]]
    base = clean[row["sample_id"]]
    return int(
        row["status"] == "SELECTED"
        and row["full_exact"] in ("True", "true", "1")
        and row["nonzero"] in ("True", "true", "1")
        and row["linf_ok"] in ("True", "true", "1")
        and str(row["output_valid"]) in ("True", "true", "1")
        and rep["ok"]
        and float(base["clean_full_anls"]) >= 0.5
        and float(base["clean_fastv_anls"]) >= 0.5
        and float(row["anls_full"]) >= 0.5
        and float(row["anls_fastv"]) < 0.5
    )


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize(name: str):
    import json as _json
    manifest = {
        row["sample_id"]: row
        for row in _json.loads(Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v5b_expanded_pristine40/frozen_manifest.json").read_text())["samples"]
    }
    selected, replay, cells, keeps = load_variant(name)
    rows = []
    for sid in DEV:
        row = selected[sid]
        cell = cells[sid]
        keep = keeps[sid]
        strict = strict_of(row, replay, manifest)
        pollution = int(strict and int(cell["pollution"]))
        rows.append({
            "sample_id": sid,
            "strict": strict,
            "full_exact": int(row["full_exact"] in ("True", "true", "1")),
            "replay": int(bool(replay[sid]["ok"])),
            "nonzero": int(row["nonzero"] in ("True", "true", "1")),
            "linf_ok": int(row["linf_ok"] in ("True", "true", "1")),
            "invalid": int(not int(cell["output_valid"])),
            "mechanism": cell["mechanism"],
            "pollution": pollution,
            "jaccard": float(keep["jaccard"]),
            "evidence_delta": abs(float(keep["evidence_share_adv"]) - float(keep["evidence_share_clean"])),
            "aux_delta": abs(float(keep["aux_share_adv"]) - float(keep["aux_share_clean"])),
            "anls_fastv": float(row["anls_fastv"]),
            "pred_fastv": row["pred_fastv"],
            "step": row["step"],
        })
    strict_rows = [row for row in rows if row["strict"]]
    content = sum(row["mechanism"] == "CONTENT_SUFFICIENT" for row in strict_rows)
    selection = sum(row["mechanism"] == "SELECTION_SUFFICIENT" for row in strict_rows)
    interaction = sum(row["mechanism"] == "INTERACTION_REQUIRED" for row in strict_rows)
    summary = {
        "n": len(rows),
        "strict": sum(row["strict"] for row in rows),
        "full_exact": sum(row["full_exact"] for row in rows),
        "replay": sum(row["replay"] for row in rows),
        "nonzero": sum(row["nonzero"] for row in rows),
        "linf_ok": sum(row["linf_ok"] for row in rows),
        "invalid": sum(row["invalid"] for row in rows),
        "pollution": sum(row["pollution"] for row in rows),
        "content": content,
        "selection": selection,
        "interaction": interaction,
        "both": sum(row["mechanism"] == "BOTH_INDEPENDENTLY_SUFFICIENT" for row in strict_rows),
        "median_jaccard": statistics.median(row["jaccard"] for row in rows),
        "median_evidence_delta": statistics.median(row["evidence_delta"] for row in rows),
        "median_aux_delta": statistics.median(row["aux_delta"] for row in rows),
        "p90_evidence_delta": percentile([row["evidence_delta"] for row in rows], 0.9),
        "p90_aux_delta": percentile([row["aux_delta"] for row in rows], 0.9),
    }
    summary["common_pass"] = (
        summary["full_exact"] == 12
        and summary["replay"] == 12
        and summary["nonzero"] == 12
        and summary["linf_ok"] == 12
        and summary["invalid"] == 0
        and summary["strict"] >= 7
        and summary["pollution"] <= 1
    )
    summary["c_extra_pass"] = (
        summary["common_pass"]
        and summary["median_evidence_delta"] <= 0.01
        and summary["median_aux_delta"] <= 0.01
        and summary["p90_evidence_delta"] <= 0.03
        and summary["p90_aux_delta"] <= 0.03
        and summary["median_jaccard"] >= A_JACCARD
        and content > selection + interaction
    )
    return rows, summary


if __name__ == "__main__":
    raise SystemExit("call summarize from the decision script after both mechanism audits exist")
