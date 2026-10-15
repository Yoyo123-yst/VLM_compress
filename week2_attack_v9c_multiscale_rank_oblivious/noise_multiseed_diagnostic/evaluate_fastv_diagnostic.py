"""Load sealed noise PNGs and score FastV. No noise is redrawn here."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V9C = ROOT / "week2_attack_v9c_multiscale_rank_oblivious"
OUT = V9C / "noise_multiseed_diagnostic"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(V9C)]

from common import anls_score, answers_of, load_build_prompt, manifest_samples, output_valid, sha_file  # noqa: E402
from run_causal_v2 import load_model, run_fastv, run_plain  # noqa: E402

IDS = ("1025", "2008", "7006", "47450", "39204")
RATIOS = (0.15, 0.25, 0.35, 0.50)


class Access:
    fastv = 0
    full = 0


def wilson(successes: int, total: int) -> dict | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    z2 = z * z
    den = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / den
    margin = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / den
    return {"k": successes, "n": total, "rate": p, "low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def ratio_key(ratio: float) -> str:
    return f"{ratio:.2f}"


def load_pages(folder: Path, count: int) -> list[Image.Image]:
    return [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(count)]


def fastv_once(model, processor, images, spec, ratio: float) -> dict:
    Access.fastv += 1
    text, _, kept, _ = run_fastv(
        model, processor, images, spec["question"], int(model.config.image_token_index), ratio, spec["evidence_idx"], spec["aux_idx"]
    )
    anls = anls_score(text, answers_of(spec))
    valid = output_valid(text)
    return {
        "text": text,
        "anls": anls,
        "valid": valid,
        "endpoint_failure": bool(valid and anls < 0.5),
        "k": len(kept),
        "keep_sha256": hashlib.sha256(",".join(str(int(x)) for x in kept).encode()).hexdigest(),
    }


def plain_once(model, processor, images, spec) -> str:
    Access.full += 1
    text, _, _, _ = run_plain(model, processor, images, spec["question"], int(model.config.image_token_index), False, spec["evidence_idx"])
    return text


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    _, prompt_hash, _ = load_build_prompt()
    consistency = []
    if prompt_hash != manifest["prompt_sha256"]:
        consistency.append("PROMPT_HASH")
    gates = [json.loads(line) for line in (OUT / "full_gate.jsonl").read_text().splitlines() if line.strip()]
    if len(gates) != len(IDS) * 32:
        consistency.append("GATE_COUNT")
    specs = {row["sample_id"]: row for row in manifest_samples()}
    previous_clean = {row["sample_id"]: row for row in json.loads((V9C / "target_eval_controls" / "clean_baseline.json").read_text())["rows"]}
    attack_rows = {row["sample_id"]: row for row in json.loads((V9C / "target_eval_controls" / "attack_replay.json").read_text())["rows"]}
    for sample_id in IDS:
        spec = specs[sample_id]
        attack = attack_rows[sample_id]
        if attack.get("evaluated"):
            got = sha_file(V9C / "sealed_png" / sample_id / f"page_{spec['aux_idx']}.png")
            if got != attack.get("png_sha256"):
                consistency.append(f"ATTACK_HASH_{sample_id}")
        base = [sha_file(V9C / "baseline_pages" / sample_id / f"page_{index}.png") for index in range(spec["M"])]
        if base != spec["page_sha256"]:
            consistency.append(f"BASELINE_HASH_{sample_id}")
    if consistency:
        finish(manifest, [], [], consistency, None)
        return
    model, processor = load_model()
    clean_rows = []
    for sample_id in IDS:
        spec = specs[sample_id]
        images = load_pages(V9C / "baseline_pages" / sample_id, spec["M"])
        first = plain_once(model, processor, images, spec)
        second = plain_once(model, processor, load_pages(V9C / "baseline_pages" / sample_id, spec["M"]), spec)
        ratios = {}
        status = "OK"
        if first != second or first != spec["clean_full_pred"]:
            status = "FULL_REPLAY_DIVERGED"
            consistency.append(f"CLEAN_FULL_{sample_id}")
        for ratio in RATIOS:
            key = ratio_key(ratio)
            scored = fastv_once(model, processor, images, spec, ratio)
            prior = previous_clean[sample_id]["ratios"][key]
            scored["historical_text"] = prior["text"]
            scored["historical_match"] = scored["text"] == prior["text"]
            if not scored["historical_match"]:
                status = "CLEAN_FASTV_DIVERGED"
                consistency.append(f"CLEAN_FASTV_{sample_id}_{key}")
            ratios[key] = scored
        clean_rows.append({"sample_id": sample_id, "status": status, "full_text": second, "full_replay": first == second, "ratios": ratios})
        print(json.dumps({"branch": "clean", "sample": sample_id, "status": status}), flush=True)
        if consistency:
            finish(manifest, clean_rows, [], consistency, None)
            return
    (OUT / "clean_baseline.json").write_text(json.dumps({"rows": clean_rows, "fastv_access_count": Access.fastv}, indent=2) + "\n")
    clean_by = {row["sample_id"]: row for row in clean_rows}
    records_path = OUT / "random_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    for gate in gates:
        spec = specs[gate["sample_id"]]
        clean = clean_by[gate["sample_id"]]
        if gate["status"] != "PASS":
            for ratio in RATIOS:
                write_record(records_path, gate, ratio, None, clean, False)
            continue
        if sha_file(Path(gate["png_path"])) != gate["png_sha256"]:
            consistency.append(f"PNG_HASH_{gate['sample_id']}_{gate['seed_index']}")
            gate = {**gate, "status": "PNG_HASH_MISMATCH"}
            for ratio in RATIOS:
                write_record(records_path, gate, ratio, None, clean, False)
            continue
        images = load_pages(Path(gate["png_path"]).parent, spec["M"])
        replay = plain_once(model, processor, images, spec)
        if replay != gate["full_text"] or replay != spec["clean_full_pred"]:
            consistency.append(f"NOISE_FULL_REPLAY_{gate['sample_id']}_{gate['seed_index']}")
            failed = {**gate, "status": "REPLAY_DIVERGED", "eval_full_text": replay}
            for ratio in RATIOS:
                write_record(records_path, failed, ratio, None, clean, False)
            continue
        for ratio in RATIOS:
            scored = fastv_once(model, processor, images, spec, ratio)
            write_record(records_path, gate, ratio, scored, clean, True)
        print(json.dumps({"branch": "noise", "sample": gate["sample_id"], "seed_index": gate["seed_index"], "fastv": Access.fastv}), flush=True)
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    finish(manifest, clean_rows, records, consistency, attack_rows)


def write_record(path: Path, gate: dict, ratio: float, scored: dict | None, clean: dict, sent: bool) -> None:
    key = ratio_key(ratio)
    clean_cell = clean["ratios"][key]
    induced = bool(sent and scored and (not clean_cell["endpoint_failure"]) and scored["endpoint_failure"])
    row = {
        "sample_id": gate["sample_id"],
        "seed_index": gate["seed_index"],
        "seed": gate["seed"],
        "resample_count": gate["resample_count"],
        "ratio": ratio,
        "full_gate_status": gate["status"],
        "sent_to_fastv": sent,
        "png_path": gate["png_path"],
        "png_sha256": gate["png_sha256"],
        "linf": gate["linf"],
        "changed_pages": gate["changed_pages"],
        "full_text": gate.get("eval_full_text", gate.get("full_text")),
        "clean_endpoint_failure": clean_cell["endpoint_failure"],
        "clean_text": clean_cell["text"],
        "clean_anls": clean_cell["anls"],
        "text": None if scored is None else scored["text"],
        "anls": None if scored is None else scored["anls"],
        "valid": None if scored is None else scored["valid"],
        "endpoint_failure": None if scored is None else scored["endpoint_failure"],
        "induced_random_failure": induced,
        "k": None if scored is None else scored["k"],
        "keep_sha256": None if scored is None else scored["keep_sha256"],
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def finish(manifest: dict, clean_rows: list, records: list, consistency: list, attack_rows: dict | None) -> None:
    summary = build_summary(clean_rows, records, consistency, attack_rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    decision = {
        "decision": summary["decision"],
        "scope": "5-sample development multi-seed noise diagnostic",
        "not_unseen": True,
        "not_budget_independent": True,
        "not_cross_compressor": True,
        "not_generally_effective_graybox": True,
        "single_attack_is_not_statistical_significance": True,
        "fastv_access_count": Access.fastv,
        "full_calls_in_eval_process": Access.full,
        "consistency": consistency,
    }
    (OUT / "final_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    manifest["fastv_access_count"] = Access.fastv
    manifest["consistency"] = consistency
    manifest["output_sha256"] = {
        name: sha_file(OUT / name)
        for name in ("full_gate.jsonl", "clean_baseline.json", "random_records.jsonl", "summary.json", "final_decision.json")
        if (OUT / name).exists()
    }
    manifest["evaluate_script_sha256"] = sha_file(Path(__file__))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(summary, consistency)
    print(json.dumps({"decision": summary["decision"], "fastv_access_count": Access.fastv, "consistency": consistency}), flush=True)


def build_summary(clean_rows: list, records: list, consistency: list, attack_rows: dict | None) -> dict:
    clean = {row["sample_id"]: row for row in clean_rows}
    per_ratio = {}
    for ratio in RATIOS:
        key = ratio_key(ratio)
        cells = [row for row in records if row["ratio"] == ratio]
        by_sample = {}
        for sample_id in IDS:
            group = [row for row in cells if row["sample_id"] == sample_id]
            passed = [row for row in group if row["sent_to_fastv"]]
            induced = [row for row in passed if row["induced_random_failure"]]
            by_sample[sample_id] = {
                "full_gate_pass": len(passed),
                "full_gate_invalid": len(group) - len(passed),
                "induced_random_failure": wilson(len(induced), len(passed)),
                "induced_over_32": None if not group else len(induced) / 32.0,
                "clean_endpoint_failure": None if sample_id not in clean else clean[sample_id]["ratios"][key]["endpoint_failure"],
                "clean_text": None if sample_id not in clean else clean[sample_id]["ratios"][key]["text"],
                "clean_anls": None if sample_id not in clean else clean[sample_id]["ratios"][key]["anls"],
            }
        pooled_k = sum(item["induced_random_failure"]["k"] for item in by_sample.values() if item["induced_random_failure"])
        pooled_n = sum(item["induced_random_failure"]["n"] for item in by_sample.values() if item["induced_random_failure"])
        per_ratio[key] = {"by_sample": by_sample, "pooled_passing_controls": wilson(pooled_k, pooled_n)}
    attack_cells = []
    primary = None
    if attack_rows and clean:
        for sample_id in IDS:
            attack = attack_rows[sample_id]
            for ratio in RATIOS:
                key = ratio_key(ratio)
                clean_cell = clean[sample_id]["ratios"][key]
                evaluated = bool(attack.get("evaluated"))
                attack_cell = None if not evaluated else attack["ratios"][key]
                attack_induced = bool(evaluated and (not clean_cell["endpoint_failure"]) and attack_cell["valid"] and attack_cell["anls"] < 0.5)
                random_cell = per_ratio[key]["by_sample"][sample_id]["induced_random_failure"]
                item = {
                    "sample_id": sample_id,
                    "ratio": ratio,
                    "attack_status": attack.get("status"),
                    "attack_evaluated": evaluated,
                    "attack_text": None if attack_cell is None else attack_cell["text"],
                    "attack_anls": None if attack_cell is None else attack_cell["anls"],
                    "attack_induced_failure": attack_induced,
                    "random": random_cell,
                }
                if attack_induced:
                    attack_cells.append(item)
                if sample_id == "1025" and key == "0.25":
                    detail = [row for row in records if row["sample_id"] == "1025" and row["ratio"] == 0.25]
                    primary = {**item, "clean_text": clean_cell["text"], "clean_anls": clean_cell["anls"], "seeds": detail}
    decision = "DIAGNOSTIC_INVALID_OR_REPLAY_DIVERGED"
    if not consistency and primary is not None:
        random_cell = primary["random"] or {}
        outside = bool(primary["attack_induced_failure"] and random_cell.get("k") == 0 and random_cell.get("n") == 32)
        decision = "GRAYBOX_SIGNAL_SUPPORTED_ON_DEVELOPMENT_DIAGNOSTIC" if outside else "GRAYBOX_SIGNAL_NOT_ESTABLISHED_ABOVE_RANDOM"
    return {
        "decision": decision,
        "scope": "5-sample development multi-seed noise diagnostic",
        "overall_denominator": 5,
        "sealed_attack_coverage": "4/5",
        "sample_39204_attack": "NO_VALID_SEALED_CANDIDATE",
        "full_gate_pass": sum(1 for row in records if row["ratio"] == 0.25 and row["sent_to_fastv"]) if records else None,
        "full_gate_total": 160,
        "consistency": consistency,
        "per_ratio": per_ratio,
        "attack_induced_cells": attack_cells,
        "sample_1025_r025": primary,
        "fastv_access_count": Access.fastv,
        "note": "One sealed attack observation is not a rate and is not a significance test against the 32-seed Wilson interval.",
    }


def write_report(summary: dict, consistency: list) -> None:
    lines = [
        "# Multi-seed noise diagnostic",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        "This is a 5-sample development diagnostic. It is not unseen generalization, not budget-independent transfer, not cross-compressor transfer, and not evidence that the gray-box attack is generally effective. One sealed attack failure is not a statistical test against the random ensemble.",
        "",
        f"Consistency flags: {consistency if consistency else 'none'}.",
        f"FastV calls in the evaluation process: {summary['fastv_access_count']}.",
        f"Full gate pass among 160 sealed noise PNGs: {summary['full_gate_pass']}.",
        "Overall denominator remains 5. Sealed-attack coverage is 4/5. Sample 39204 stays `NO_VALID_SEALED_CANDIDATE` and is not removed.",
        "",
        "## Random induced-failure rate",
        "",
        "Induced means the clean page is not an endpoint failure, the noise passed the Full gate, and the noise FastV output is valid with ANLS < 0.5. Sample 1025 at r=0.15 is a clean endpoint failure, so its noise failures are not induced.",
        "",
    ]
    for key, block in summary["per_ratio"].items():
        lines.append(f"### r={key}")
        pooled = block["pooled_passing_controls"]
        if pooled:
            lines.append(f"Pooled passing controls: {pooled['k']}/{pooled['n']} = {pooled['rate']:.4f}, Wilson 95% CI [{pooled['low']:.4f}, {pooled['high']:.4f}].")
        for sample_id, cell in block["by_sample"].items():
            rate = cell["induced_random_failure"]
            rendered = "NOT_RUN" if rate is None else f"{rate['k']}/{rate['n']} = {rate['rate']:.4f}, CI [{rate['low']:.4f}, {rate['high']:.4f}]"
            lines.append(f"- {sample_id}: clean_failed={cell['clean_endpoint_failure']}, gate {cell['full_gate_pass']}/32, induced {rendered}.")
        lines.append("")
    primary = summary.get("sample_1025_r025")
    lines.append("## Sample 1025, r=0.25")
    lines.append("")
    if not primary:
        lines.append("Primary cell was not measured.")
    else:
        lines.append(f"Clean: `{primary['clean_text']}` ANLS {primary['clean_anls']}.")
        lines.append(f"Sealed attack, not rerun: `{primary['attack_text']}` ANLS {primary['attack_anls']}; induced={primary['attack_induced_failure']}.")
        rate = primary["random"]
        if rate:
            lines.append(f"Random induced failures: {rate['k']}/{rate['n']} = {rate['rate']:.4f}, Wilson 95% CI [{rate['low']:.4f}, {rate['high']:.4f}].")
        lines.append("")
        lines.append("| seed_index | seed | text | ANLS | endpoint | induced | gate |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in primary.get("seeds") or []:
            lines.append(f"| {row['seed_index']} | {row['seed']} | {row['text']} | {row['anls']} | {row['endpoint_failure']} | {row['induced_random_failure']} | {row['full_gate_status']} |")
    lines.append("")
    lines.append("Attack-induced cells and their random Wilson intervals are listed in `summary.json`. The attack PNG was not rerun and was not modified.")
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
