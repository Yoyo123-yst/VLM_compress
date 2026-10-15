"""Post-process frozen pristine60 artifacts. Does not rerun the attack or edit hashed files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from week1_mechanism.metrics_task import normalize
from week2_attack_v7_pristine60.analyze_v7 import mcnemar, strict_of, wilson
from week2_attack_v7_pristine60.load_samples import manifest
from week2_attack_v7_pristine60.parents import parent_hashes

V7 = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7_pristine60")
WRONG = 0.5
FROZEN_FILE_SHA = "884aa0117847b2b24373a8d101a545664b048119f51b4148b9e870e1027c5e15"
CLASSES = (
    "CONTENT_SUFFICIENT",
    "SELECTION_SUFFICIENT",
    "BOTH_INDEPENDENTLY_SUFFICIENT",
    "INTERACTION_REQUIRED",
    "UNEXPLAINED_OR_INVALID",
    "COUNTERFACTUAL_INVALID",
)
ERROR_TYPES = (
    "CORE_ANSWER_CHANGED",
    "UNRELATED_ANSWER_REPLACEMENT",
    "SPELLING_OR_CASE_ONLY",
    "EXTRA_TEXT_POLLUTION",
    "TRUNCATION_OR_EOS",
    "OTHER",
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": quantile(values, 0.5),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
    }


# Read from clean FastV, selected FastV, and gold. A shared character substring is not enough.
REVIEWED_ERROR = {
    "2288": ("CORE_ANSWER_CHANGED", "person name replaced: Dr. G. E. Inglett -> Ferguson Egan"),
    "18736": ("TRUNCATION_OR_EOS", "currency amount collapsed to the single digit 2"),
    "41800": ("UNRELATED_ANSWER_REPLACEMENT", "Vanguard E Poly replaced by the non-name string Yeezr-pypypx, 5d00"),
    "45827": ("CORE_ANSWER_CHANGED", "disease answer replaced: osteoporotic disease -> Endometrial cancers"),
    "49432": ("TRUNCATION_OR_EOS", "division name collapsed to the single token I"),
    "50906": ("CORE_ANSWER_CHANGED", "person name replaced: Dr Darby -> Dorothy M. Bryan"),
    "51660": ("CORE_ANSWER_CHANGED", "date replaced: May 4, 1960 -> 11/75"),
    "5412": ("CORE_ANSWER_CHANGED", "number replaced: 18432 -> 2826"),
    "56403": ("CORE_ANSWER_CHANGED", "number replaced: 400 -> 25.35"),
    "57531": ("UNRELATED_ANSWER_REPLACEMENT", "brand Sunfeast replaced by the garbled product string Bataleefee Taki Chakratai Biscuits"),
    "58434": ("CORE_ANSWER_CHANGED", "identifier changed from DWRITE 066084/066085 to DWRITE_0966 plus extra codes"),
    "60744": ("CORE_ANSWER_CHANGED", "institution replaced: Simon Fraser University -> University of Kentucky"),
    "65118": ("CORE_ANSWER_CHANGED", "number replaced: 300 -> 14"),
    "790": ("CORE_ANSWER_CHANGED", "person name replaced: Michael F. Jacobson -> J. Alexander Downer"),
    "8122": ("UNRELATED_ANSWER_REPLACEMENT", "person name Susan King replaced by the code W7T"),
    "25716": ("CORE_ANSWER_CHANGED", "time replaced: 12:30 p.m. -> 2.12"),
    "44779": ("UNRELATED_ANSWER_REPLACEMENT", "product name TrueGlide replaced by DESIGN BATICALE 8"),
    "49123": ("CORE_ANSWER_CHANGED", "number replaced: 3150 -> 00M 36275,41280"),
    "50803": ("UNRELATED_ANSWER_REPLACEMENT", "time 10:30 replaced by the sentence Time to sail the great New York Time"),
    "51298": ("CORE_ANSWER_CHANGED", "year span 2015-2016 replaced by OcT 22"),
    "51702": ("CORE_ANSWER_CHANGED", "date February 13/11, 1977 replaced by 15-2180"),
    "5497": ("CORE_ANSWER_CHANGED", "person name corrupted: Norman J. Macdonald -> NORMAN J J JILJI"),
    "56084": ("CORE_ANSWER_CHANGED", "answer replaced: Alcoholics -> Smoking"),
    "56808": ("TRUNCATION_OR_EOS", "currency amount $52.00 collapsed to 2"),
    "57887": ("EXTRA_TEXT_POLLUTION", "gold 1mg remains and the drug name Trimegestone is added"),
    "58449": ("CORE_ANSWER_CHANGED", "date replaced: March 31, 1982 -> December, 2009"),
    "62321": ("CORE_ANSWER_CHANGED", "department name shortened and changed to STATE DEPT OF SOCIETY"),
    "1170": ("UNRELATED_ANSWER_REPLACEMENT", "medical phrase replaced by confidential"),
    "22317": ("UNRELATED_ANSWER_REPLACEMENT", "quoted title replaced by Qbient"),
    "43150": ("CORE_ANSWER_CHANGED", "date replaced: 24-JUN-2013/2014 -> 04 Nove 11 - 25"),
    "47572": ("CORE_ANSWER_CHANGED", "roman numeral replaced: VIII -> XV"),
    "50312": ("CORE_ANSWER_CHANGED", "number replaced: 604 -> 47"),
    "51140": ("UNRELATED_ANSWER_REPLACEMENT", "date June 7, 2016 replaced by at 47, 797, 798"),
    "53463": ("UNRELATED_ANSWER_REPLACEMENT", "case label CRIS replaced by Esophagus"),
    "55160": ("CORE_ANSWER_CHANGED", "number replaced: 30 -> 224"),
    "5586": ("CORE_ANSWER_CHANGED", "person name corrupted: Daniel Johnson -> Daniel K. Jahanan, Jahanan"),
    "58371": ("TRUNCATION_OR_EOS", "currency amount collapsed to 1.5"),
    "58827": ("CORE_ANSWER_CHANGED", "role phrase changed: Officers of the institute -> MEMBER, SHIP, GOVERNMENT CORPORATION"),
    "62650": ("UNRELATED_ANSWER_REPLACEMENT", "field Chemistry replaced by an unrelated solar-physics sentence"),
    "6942": ("UNRELATED_ANSWER_REPLACEMENT", "NA replaced by P.0.C."),
    "7937": ("CORE_ANSWER_CHANGED", "number replaced: 98 -> 1.00 1.00"),
    "1023": ("CORE_ANSWER_CHANGED", "phone number replaced"),
    "41828": ("CORE_ANSWER_CHANGED", "number 399 replaced by VC 333999; the digit sequence is not the gold answer"),
    "46737": ("EXTRA_TEXT_POLLUTION", "U.S. Pharmacist becomes U.S. Pharmacopeia followed by extra tokens"),
    "51070": ("CORE_ANSWER_CHANGED", "column value 2 replaced by 1-18"),
    "5159": ("UNRELATED_ANSWER_REPLACEMENT", "place Fort Morgan replaced by the unreadable string FORDHORSEMFGFACTORY"),
    "53993": ("UNRELATED_ANSWER_REPLACEMENT", "Ohio replaced by SUPERSUPEMERICAO, U.S.A"),
    "5615": ("CORE_ANSWER_CHANGED", "phone number replaced"),
    "58421": ("TRUNCATION_OR_EOS", "HEATING IN OPEN AIR shortened to HER"),
    "59379": ("CORE_ANSWER_CHANGED", "time replaced: 2:00 PM -> 19:10"),
    "64689": ("CORE_ANSWER_CHANGED", "dose 500 mg replaced by 122"),
    "8072": ("CORE_ANSWER_CHANGED", "number replaced: 40 -> 2"),
}


def error_type(sample_id: str, clean: str, adv: str, answers: list[str]) -> tuple[str, str]:
    reviewed = REVIEWED_ERROR.get(sample_id)
    if reviewed:
        return reviewed
    clean_n = normalize(clean)
    adv_n = normalize(adv)
    if clean_n == adv_n:
        return "SPELLING_OR_CASE_ONLY", "normalized clean FastV equals normalized adversarial FastV"
    return "OTHER", "no reviewed text relation was recorded for this sample"


def cell_strict(anls, text: str) -> bool:
    try:
        value = float(anls)
    except (TypeError, ValueError):
        return False
    return bool(text.strip()) and value < WRONG


def classify_cell(raw: dict, spec: dict, selected: dict) -> dict:
    cc_match = raw.get("pred_CC") == spec["clean_fastv_pred"]
    aa_match = raw.get("pred_AA") == selected.get("pred_fastv")
    valid = bool(raw.get("output_valid")) and all(str(raw.get(key, "")).strip() for key in ("pred_CC", "pred_CA", "pred_AC", "pred_AA"))
    ac = cell_strict(raw.get("anls_AC"), raw.get("pred_AC", ""))
    ca = cell_strict(raw.get("anls_CA"), raw.get("pred_CA", ""))
    aa = cell_strict(raw.get("anls_AA"), raw.get("pred_AA", ""))
    if not cc_match or not aa_match:
        label = "COUNTERFACTUAL_INVALID"
        basis = "CC does not reproduce clean FastV" if not cc_match else "AA does not reproduce selected FastV"
        if not cc_match and not aa_match:
            basis = "CC and AA both fail to reproduce the recorded FastV texts"
    elif not valid:
        label = "UNEXPLAINED_OR_INVALID"
        basis = "one or more cell texts are empty or the row is marked invalid"
    elif ac and not ca:
        label = "CONTENT_SUFFICIENT"
        basis = "AC strict and CA non-strict"
    elif ca and not ac:
        label = "SELECTION_SUFFICIENT"
        basis = "CA strict and AC non-strict"
    elif ac and ca:
        label = "BOTH_INDEPENDENTLY_SUFFICIENT"
        basis = "AC strict and CA strict"
    elif aa and not ac and not ca:
        label = "INTERACTION_REQUIRED"
        basis = "AA strict while AC and CA are non-strict"
    else:
        label = "UNEXPLAINED_OR_INVALID"
        basis = "cell pattern does not match the exclusive rules"
    return {
        "sample_id": raw["sample_id"],
        "M": spec["M"],
        "step": raw.get("step"),
        "pred_CC": raw.get("pred_CC", ""),
        "pred_CA": raw.get("pred_CA", ""),
        "pred_AC": raw.get("pred_AC", ""),
        "pred_AA": raw.get("pred_AA", ""),
        "anls_CC": raw.get("anls_CC"),
        "anls_CA": raw.get("anls_CA"),
        "anls_AC": raw.get("anls_AC"),
        "anls_AA": raw.get("anls_AA"),
        "output_valid": int(valid),
        "keep_hash_clean": raw.get("keep_hash_clean"),
        "keep_hash_adv": raw.get("keep_hash_adv"),
        "K": raw.get("K"),
        "cc_reproduces_clean_fastv": int(cc_match),
        "aa_reproduces_selected_fastv": int(aa_match),
        "hook_mechanism": raw.get("mechanism"),
        "mechanism": label,
        "basis": basis,
        "jaccard": raw.get("jaccard"),
        "symmetric_difference": raw.get("symmetric_difference"),
        "evidence_share_clean": raw.get("evidence_share_clean"),
        "evidence_share_adv": raw.get("evidence_share_adv"),
        "aux_share_clean": raw.get("aux_share_clean"),
        "aux_share_adv": raw.get("aux_share_adv"),
        "evidence_share_delta": None if raw.get("evidence_share_adv") is None else float(raw["evidence_share_adv"]) - float(raw["evidence_share_clean"]),
        "aux_share_delta": None if raw.get("aux_share_adv") is None else float(raw["aux_share_adv"]) - float(raw["aux_share_clean"]),
        "gold": " | ".join(spec["answers"]),
        "clean_fastv": spec["clean_fastv_pred"],
        "selected_fastv": selected.get("pred_fastv", ""),
    }


def manifest_canonical_sha(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def immutability(frozen: dict) -> dict:
    code_changed = []
    for rel, recorded in frozen["code_hashes"].items():
        path = V7 / rel
        current = sha_file(path) if path.exists() else None
        if current != recorded:
            code_changed.append(rel)
    parents = parent_hashes()
    parent_changed = sorted(rel for rel, digest in parents.items() if frozen["parent_hashes"].get(rel) != digest)
    payload = json.loads((V7 / "frozen_manifest.json").read_text())
    canonical = manifest_canonical_sha(payload)
    recorded_manifest = (V7 / "frozen_manifest.sha256").read_text().strip()
    attack_paths = list((V7 / "attacks").glob("shard_s*/*/checkpoints.jsonl"))
    return {
        "result": "PASS" if not code_changed and not parent_changed and canonical == frozen["frozen_manifest_sha256"] == recorded_manifest == payload["manifest_sha256"] and sha_file(V7 / "FROZEN.json") == FROZEN_FILE_SHA else "FAIL",
        "frozen_json_sha256": sha_file(V7 / "FROZEN.json"),
        "frozen_json_expected": FROZEN_FILE_SHA,
        "manifest_canonical_sha256": canonical,
        "manifest_recorded": frozen["frozen_manifest_sha256"],
        "code_changed": code_changed,
        "parent_changed": parent_changed,
        "parent_n": len(parents),
        "attack_checkpoint_files": len(attack_paths),
    }


def main() -> None:
    specs = {row["sample_id"]: row for row in manifest()["samples"]}
    selected = {row["sample_id"]: row for row in read_csv(V7 / "attacks" / "selector" / "selected.csv")}
    raw_cells = []
    for path in sorted((V7 / "mechanism_audit").glob("cells_s*.json")):
        raw_cells.extend(json.loads(path.read_text()))
    if len(raw_cells) != 60 or len({row["sample_id"] for row in raw_cells}) != 60:
        raise SystemExit(f"mechanism cells incomplete: {len(raw_cells)}")
    classified = [classify_cell(row, specs[row["sample_id"]], selected[row["sample_id"]]) for row in raw_cells]
    replay = {}
    for path in sorted((V7 / "attacks" / "replay").glob("replay_s*.json")):
        for item in json.loads(path.read_text()):
            replay[item["sample_id"]] = item
    primary_strict = {}
    for sid, spec in specs.items():
        primary_strict[sid] = int(strict_of(selected[sid], spec, bool(replay[sid].get("ok"))))
    for row in classified:
        row["primary_strict"] = primary_strict[row["sample_id"]]
        row["cohort"] = "strict_success" if row["primary_strict"] else "non_strict_failure_diagnosis"
        if row["primary_strict"] and row["sample_id"] not in REVIEWED_ERROR:
            raise SystemExit(f"missing reviewed error label for {row['sample_id']}")
        kind, reason = error_type(row["sample_id"], row["clean_fastv"], row["selected_fastv"], specs[row["sample_id"]]["answers"])
        row["error_type"] = kind if row["primary_strict"] else ""
        row["error_type_reason"] = reason if row["primary_strict"] else ""
    write_csv(V7 / "mechanism_audit" / "cells.csv", classified)
    mechanism_dir = V7 / "mechanism"
    mechanism_dir.mkdir(exist_ok=True)
    by_id = {row["sample_id"]: row for row in classified}
    for src in sorted((V7 / "mechanism_audit").glob("cells_s*.json")):
        raw = json.loads(src.read_text())
        for row in raw:
            row["mechanism"] = by_id[row["sample_id"]]["mechanism"]
            row["reproduction_basis"] = by_id[row["sample_id"]]["basis"]
        (mechanism_dir / src.name).write_text(json.dumps(raw, indent=2) + "\n")

    strict_rows = [row for row in classified if row["cohort"] == "strict_success"]
    fail_rows = [row for row in classified if row["cohort"] != "strict_success"]
    counts = Counter(row["mechanism"] for row in strict_rows)
    fail_counts = Counter(row["mechanism"] for row in fail_rows)
    error_counts = Counter(row["error_type"] for row in strict_rows)
    n_strict = len(strict_rows)

    def floats(key: str, rows: list[dict]) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) not in ("", None)]

    share = {
        "jaccard": summary_stats(floats("jaccard", classified)),
        "symmetric_difference": summary_stats(floats("symmetric_difference", classified)),
        "evidence_share_delta": summary_stats(floats("evidence_share_delta", classified)),
        "aux_share_delta": summary_stats(floats("aux_share_delta", classified)),
    }
    by_m = {}
    for m in (2, 3, 4):
        ids = [sid for sid, spec in specs.items() if int(spec["M"]) == m]
        k = sum(primary_strict[sid] for sid in ids)
        by_m[str(m)] = {"strict": k, "n": len(ids), "rate": k / len(ids), "wilson95": list(wilson(k, len(ids)))}

    mech_summary = []
    for name in CLASSES:
        k = counts[name]
        lo, hi = wilson(k, n_strict) if n_strict else (None, None)
        mech_summary.append({"cohort": "strict_success", "mechanism": name, "n": k, "rate": (k / n_strict) if n_strict else None, "wilson95_lo": lo, "wilson95_hi": hi})
    for name, k in sorted(fail_counts.items()):
        mech_summary.append({"cohort": "non_strict_failure_diagnosis", "mechanism": name, "n": k, "rate": k / len(fail_rows) if fail_rows else None, "wilson95_lo": None, "wilson95_hi": None})
    write_csv(V7 / "mechanism_audit" / "summary.csv", mech_summary)
    content_n = counts["CONTENT_SUFFICIENT"] + counts["BOTH_INDEPENDENTLY_SUFFICIENT"]
    selection_n = counts["SELECTION_SUFFICIENT"]
    unexplained_n = counts["UNEXPLAINED_OR_INVALID"] + counts["COUNTERFACTUAL_INVALID"]
    content_dominant = n_strict > 0 and content_n > n_strict / 2 and selection_n < content_n
    mechanism_inconclusive = n_strict == 0 or unexplained_n > n_strict / 5 or unexplained_n == max(counts.values())
    cc_ok = sum(row["cc_reproduces_clean_fastv"] for row in classified)
    aa_ok = sum(row["aa_reproduces_selected_fastv"] for row in classified)
    audit = {
        "n": 60,
        "cc_reproduces": cc_ok,
        "aa_reproduces": aa_ok,
        "strict_success_counts": dict(counts),
        "non_strict_counts": dict(fail_counts),
        "content_or_both": content_n,
        "selection": selection_n,
        "content_dominant_claim_allowed": bool(content_dominant and not mechanism_inconclusive),
        "mechanism_inconclusive": bool(mechanism_inconclusive),
        "error_types_among_strict": dict(error_counts),
        "keep_set": share,
        "cell_strict_definition": "nonempty text and FastV ANLS < 0.5; Full exact stays the already validated primary state",
    }
    (V7 / "mechanism_audit" / "mechanism_decision.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# mechanism audit",
        "",
        f"- CC reproduces clean FastV: {cc_ok}/60",
        f"- AA reproduces selected FastV: {aa_ok}/60",
        f"- Strict-success mechanism counts: {dict(counts)}",
        f"- Non-strict diagnosis counts, not mixed into the success proportion: {dict(fail_counts)}",
        f"- Content-dominant claim allowed: {audit['content_dominant_claim_allowed']}",
        "",
        "Cell strict means the compressed-path text is nonempty and its ANLS is below 0.5. A strict primary sample keeps its already validated Full-exact state.",
        "",
    ]
    for row in classified:
        lines.append(f"- {row['sample_id']} M={row['M']} {row['cohort']} {row['mechanism']}: {row['basis']}; error {row['error_type'] or '-'}")
    (V7 / "mechanism_audit" / "mechanism_report.md").write_text("\n".join(lines) + "\n")

    from week2_attack_v7_pristine60.analyze_v7 import main as analyze_main
    analyze_main()
    stats_doc = json.loads((V7 / "statistics.json").read_text())
    repeat = json.loads((V7 / "repeat10" / "repeatability_decision.json").read_text())
    endpoint_n = repeat["labels"]["EXACT_REPRODUCTION"] + repeat["labels"]["ENDPOINT_REPRODUCTION"]
    repeat_limitation = repeat["strict_agree"] < 8
    frozen = json.loads((V7 / "FROZEN.json").read_text())
    integrity = immutability(frozen)
    (V7 / "raw_immutability_check.json").write_text(json.dumps(integrity, indent=2) + "\n")
    gates = stats_doc["gates"]
    primary_pass = all(gates.values()) and integrity["result"] == "PASS"
    limitations = []
    if repeat_limitation:
        limitations.append("PASS_PRIMARY_WITH_REPEATABILITY_LIMITATION")
    if mechanism_inconclusive:
        limitations.append("PASS_PRIMARY_MECHANISM_INCONCLUSIVE")
    if integrity["result"] != "PASS":
        decision = "FAIL_PROTOCOL_INTEGRITY"
    elif not all(gates.values()):
        decision = "PRIMARY_GATES_NOT_MET"
    elif not limitations:
        decision = "PASS_PRISTINE60_SOURCE_BOUNDED_COMPRESSION_ONLY"
    elif len(limitations) == 1:
        decision = limitations[0]
    else:
        decision = "PASS_PRIMARY_WITH_REPEATABILITY_LIMITATION"
    traj = read_csv(V7 / "paired_attack_vs_random_trajectory.csv")
    single = read_csv(V7 / "paired_attack_vs_single_noise.csv")

    def pair_counts(rows: list[dict], key: str) -> dict:
        return {
            "attack_only": sum(int(row["attack_strict"]) == 1 and int(row[key]) == 0 for row in rows),
            "control_only": sum(int(row["attack_strict"]) == 0 and int(row[key]) == 1 for row in rows),
            "both_strict": sum(int(row["attack_strict"]) == 1 and int(row[key]) == 1 for row in rows),
            "both_nonstrict": sum(int(row["attack_strict"]) == 0 and int(row[key]) == 0 for row in rows),
        }

    steps = Counter(int(row["step"]) for row in selected.values())
    final = {
        "decision": decision,
        "limitations": limitations,
        "primary_strict": f"{stats_doc['primary_strict']}/60",
        "wilson95": stats_doc["wilson95"],
        "gates": gates,
        "integrity": integrity["result"],
        "repeatability": {
            "strict_agree": repeat["strict_agree"],
            "endpoint_reproduction": endpoint_n,
            "labels": repeat["labels"],
            "limitation": repeat_limitation,
        },
        "mechanism": audit,
        "trajectory_rule": "NO_FEASIBLE_RANDOM_CANDIDATE stays a control failure on the frozen 60-sample denominator and is not redrawn",
        "content_dominant_claim_allowed": audit["content_dominant_claim_allowed"],
        "selection_stable_claim": False,
        "analyze_v7_decision": stats_doc["decision"],
    }
    (V7 / "final_decision.json").write_text(json.dumps(final, indent=2) + "\n")
    flat = [
        {"item": "decision", "value": decision},
        {"item": "primary_strict", "value": stats_doc["primary_strict"]},
        {"item": "wilson95_lo", "value": stats_doc["wilson95"][0]},
        {"item": "wilson95_hi", "value": stats_doc["wilson95"][1]},
        {"item": "full_exact", "value": stats_doc["full_exact"]},
        {"item": "full_text_changed", "value": stats_doc["full_text_change"]},
        {"item": "replay", "value": stats_doc["replay"]},
        {"item": "nonzero", "value": stats_doc["nonzero"]},
        {"item": "linf", "value": stats_doc["linf"]},
        {"item": "invalid", "value": stats_doc["invalid"]},
        {"item": "single_strict", "value": sum(int(row["single_strict"]) for row in single)},
        {"item": "trajectory_strict", "value": sum(int(row["trajectory_strict"]) for row in traj)},
        {"item": "trajectory_no_feasible", "value": 1},
        {"item": "trajectory_no_feasible_id", "value": "56084"},
        {"item": "trajectory_mcnemar_p", "value": stats_doc["trajectory_mcnemar_p"]},
        {"item": "single_mcnemar_p", "value": stats_doc["single_mcnemar_p"]},
        {"item": "repeat_strict_agree", "value": repeat["strict_agree"]},
        {"item": "repeat_endpoint_reproduction", "value": endpoint_n},
        {"item": "cc_reproduces", "value": cc_ok},
        {"item": "aa_reproduces", "value": aa_ok},
    ]
    for name, k in counts.items():
        flat.append({"item": f"strict_mechanism_{name}", "value": k})
    for name, k in error_counts.items():
        flat.append({"item": f"error_{name}", "value": k})
    for m, row in by_m.items():
        flat.append({"item": f"strict_M{m}", "value": f"{row['strict']}/{row['n']}"})
    write_csv(V7 / "final_statistics.csv", flat)
    lo, hi = stats_doc["wilson95"]
    claim = "source-bounded FastV r=0.25 compression-only attack on this frozen 60-sample draw" if decision.startswith("PASS") else "none"
    if audit["content_dominant_claim_allowed"]:
        claim_extra = "content-dominant / survivor-poisoning evidence is allowed for this FastV run"
    else:
        claim_extra = "content-dominant / survivor-poisoning evidence is not supported by the strict-success class counts"
    report = f"""# pristine60 final report

## 1. Decision

{decision}

## 2. Primary endpoint

Prospective selector strict CSFR is {stats_doc['primary_strict']}/60. Wilson 95% CI is [{lo:.6f}, {hi:.6f}]. Full exact is {stats_doc['full_exact']}/60 and Full text changed is {stats_doc['full_text_change']}/60. Replay is {stats_doc['replay']}/60, nonzero is {stats_doc['nonzero']}/60, uint8 L∞ legal is {stats_doc['linf']}/60, and invalid is {stats_doc['invalid']}/60. Selected steps: {dict(sorted(steps.items()))}.

M=2 strict is {by_m['2']['strict']}/{by_m['2']['n']} (Wilson [{by_m['2']['wilson95'][0]:.4f}, {by_m['2']['wilson95'][1]:.4f}]). M=3 is {by_m['3']['strict']}/{by_m['3']['n']} (Wilson [{by_m['3']['wilson95'][0]:.4f}, {by_m['3']['wilson95'][1]:.4f}]). M=4 is {by_m['4']['strict']}/{by_m['4']['n']} (Wilson [{by_m['4']['wilson95'][0]:.4f}, {by_m['4']['wilson95'][1]:.4f}]).

## 3. Random controls

Single random noise strict is {sum(int(row['single_strict']) for row in single)}/60. Pairing: {pair_counts(single, 'single_strict')}. Exact McNemar p is {stats_doc['single_mcnemar_p']}.

Seven-candidate trajectory strict is {sum(int(row['trajectory_strict']) for row in traj)}/60. Pairing: {pair_counts(traj, 'trajectory_strict')}. Exact McNemar p is {stats_doc['trajectory_mcnemar_p']}. ID 56084 has no feasible random candidate. It stays a control failure on the frozen 60-sample denominator and was not redrawn.

## 4. Repeatability

Strict-label agreement is {repeat['strict_agree']}/10. Success-endpoint reproduction (exact image match or strict endpoint with Full exact and replay) is {endpoint_n}/10. Labels: {repeat['labels']}. PNG hash agreement is {repeat['png_agree']}/10 and keep-hash agreement is {repeat['keep_agree']}/10. A different selected step is not scored as a failure by itself.

## 5. Mechanism

CC reproduces clean FastV on {cc_ok}/60 and AA reproduces the selected FastV text on {aa_ok}/60. Among the {n_strict} primary strict successes: CONTENT_SUFFICIENT {counts['CONTENT_SUFFICIENT']}/{n_strict} (Wilson [{wilson(counts['CONTENT_SUFFICIENT'], n_strict)[0]:.4f}, {wilson(counts['CONTENT_SUFFICIENT'], n_strict)[1]:.4f}]), SELECTION_SUFFICIENT {counts['SELECTION_SUFFICIENT']}/{n_strict}, BOTH_INDEPENDENTLY_SUFFICIENT {counts['BOTH_INDEPENDENTLY_SUFFICIENT']}/{n_strict}, INTERACTION_REQUIRED {counts['INTERACTION_REQUIRED']}/{n_strict} (Wilson [{wilson(counts['INTERACTION_REQUIRED'], n_strict)[0]:.4f}, {wilson(counts['INTERACTION_REQUIRED'], n_strict)[1]:.4f}]), UNEXPLAINED_OR_INVALID {counts['UNEXPLAINED_OR_INVALID']}/{n_strict}, COUNTERFACTUAL_INVALID {counts['COUNTERFACTUAL_INVALID']}/{n_strict}. These proportions use only the strict-success denominator. Non-strict samples are diagnosed separately: {dict(fail_counts)}.

Jaccard {share['jaccard']}. Symmetric difference {share['symmetric_difference']}. Evidence-share change {share['evidence_share_delta']}. Auxiliary-share change {share['aux_share_delta']}.

## 6. Error quality

Among primary strict successes the auditable text classes are {dict(error_counts)}. The label for each sample was assigned by reading clean FastV, adversarial FastV, and gold, and the reason is stored beside the three strings.

## 7. Protocol integrity

raw_immutability_check result is {integrity['result']}. FROZEN.json sha256 is {integrity['frozen_json_sha256']}. Manifest canonical sha256 matches the frozen record: {integrity['manifest_canonical_sha256'] == integrity['manifest_recorded']}. Code files changed: {integrity['code_changed'] or 'none'}. Parent files changed: {integrity['parent_changed'] or 'none'}.

## 8. Allowed claim / prohibited claim

Allowed claim: {claim}. {claim_extra}.

Prohibited from these artifacts: selection-stable; all compressors or models; a universal visual-token compression vulnerability. The development 40-sample rate is not combined with this 60-sample rate.

## 9. Next step

Keep this 60-sample draw, the selected checkpoints, and the no-feasible trajectory case frozen. A later study can test another compressor or another model on a new unseen draw; that result is not implied by this file.
"""
    (V7 / "final_report.md").write_text(report)
    print(json.dumps({"decision": decision, "limitations": limitations, "content_dominant": audit["content_dominant_claim_allowed"], "counts": dict(counts), "errors": dict(error_counts), "integrity": integrity["result"]}))


if __name__ == "__main__":
    main()
