"""Post-hoc clean, sealed-attack, and matched-noise controls.

This process is the only one allowed to call FastV. It does not train or reseal.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v9c_multiscale_rank_oblivious"
OUT = HERE / "target_eval_controls"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(HERE)]

from common import (  # noqa: E402
    EPS,
    TOLERANCE,
    answers_of,
    anls_score,
    manifest_samples,
    output_valid,
    sha_file,
    uint8_linf,
)
from run_causal_v2 import load_model, run_fastv, run_plain  # noqa: E402

RATIOS = (0.15, 0.25, 0.35, 0.50)
IDS = ("1025", "2008", "7006", "39204", "47450")
NOISE_SEED_RULE = "sha256('v9c-control-noise-2026101121|' + sample_id)[:8] as uint32"
SCOPE = "5-sample development post-hoc control"
TOKEN_SCOPE = "pre-seal token exact + target-side text exact"


class Access:
    fastv_calls = 0
    full_calls = 0


def noise_seed(sample_id: str) -> int:
    raw = hashlib.sha256(f"v9c-control-noise-2026101121|{sample_id}".encode()).hexdigest()[:8]
    return int(raw, 16)


def keep_hash(kept) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in kept).encode()).hexdigest()


def ratio_key(ratio: float) -> str:
    return f"{ratio:.2f}"


def measure(model, processor, images, spec) -> dict:
    token_id = int(model.config.image_token_index)
    Access.full_calls += 1
    full, _, _, _ = run_plain(model, processor, images, spec["question"], token_id, False, spec["evidence_idx"])
    fast = {}
    for ratio in RATIOS:
        Access.fastv_calls += 1
        pred, _, kept, _ = run_fastv(
            model, processor, images, spec["question"], token_id, ratio, spec["evidence_idx"], spec["aux_idx"]
        )
        fast[ratio_key(ratio)] = {
            "text": pred,
            "token_ids": None,
            "valid": output_valid(pred),
            "anls": anls_score(pred, answers_of(spec)),
            "k": len(kept),
            "keep_hash": keep_hash(kept),
        }
    return {"full_text": full, "full_token_ids": None, "full_valid": output_valid(full), "fast": fast}


def replay_ok(first: dict, second: dict) -> bool:
    if first["full_text"] != second["full_text"]:
        return False
    return all(first["fast"][key]["text"] == second["fast"][key]["text"] and first["fast"][key]["keep_hash"] == second["fast"][key]["keep_hash"] for key in first["fast"])


def load_folder(folder: Path, count: int) -> list[Image.Image]:
    return [Image.open(folder / f"page_{index}.png").convert("RGB") for index in range(count)]


def page_hashes(folder: Path, count: int) -> list[str]:
    return [sha_file(folder / f"page_{index}.png") for index in range(count)]


def precheck() -> dict:
    seal = json.loads((HERE / "seal_manifest.json").read_text())
    historical = json.loads((HERE / "target_eval" / "fastv_transfer.json").read_text())
    specs = {row["sample_id"]: row for row in manifest_samples()}
    rows = []
    ok = True
    for sample_id in IDS:
        spec = specs[sample_id]
        base = HERE / "baseline_pages" / sample_id
        hashes = page_hashes(base, spec["M"])
        match = hashes == spec["page_sha256"]
        item = next(row for row in seal["samples"] if row["sample_id"] == sample_id)
        sealed_ok = True
        if item.get("sealed"):
            for page in item["pages"]:
                got = sha_file(HERE / "sealed_png" / sample_id / f"page_{page['index']}.png")
                sealed_ok = sealed_ok and got == page["sha256"]
                if page["aux"]:
                    sealed_ok = sealed_ok and got == item["png_sha256"]
                else:
                    sealed_ok = sealed_ok and got == hashes[page["index"]]
        ok = ok and match and sealed_ok
        rows.append({
            "sample_id": sample_id,
            "aux_idx": spec["aux_idx"],
            "evidence_idx": spec["evidence_idx"],
            "M": spec["M"],
            "clean_page_sha256": hashes,
            "manifest_page_sha256_match": match,
            "sealed": bool(item.get("sealed")),
            "sealed_png_match": sealed_ok,
            "historical_clean_fastv_r025": {
                "source": "DEV_MANIFEST_COPY clean_fastv_pred from the r=0.25 clean screen",
                "text": spec.get("clean_fastv_pred"),
                "anls": spec.get("clean_fastv_anls"),
                "k": spec.get("clean_k"),
                "keep_hash": spec.get("clean_keep_hash"),
            },
            "noise_seed": noise_seed(sample_id) if item.get("sealed") else None,
        })
    payload = {
        "scope": SCOPE,
        "precheck_pass": ok,
        "token_exact_scope": TOKEN_SCOPE,
        "target_evaluator": str(HERE / "target_eval" / "evaluate_fastv.py"),
        "generate": {"do_sample": False, "max_new_tokens": 16, "use_cache": True, "caller": "run_causal_v2.generate"},
        "prompt": "week1_mechanism/run_eval.py build_prompt template A",
        "ratios": list(RATIOS),
        "noise": {
            "rule": "uint8 uniform integer noise in [-8, 8] on the auxiliary page only, then clip to [0, 255]",
            "linf_bound_uint8": 8,
            "seed_rule": NOISE_SEED_RULE,
            "does_not_read_fastv": True,
            "does_not_copy_attack_residual": True,
        },
        "samples": rows,
        "historical_attack_source": str(HERE / "target_eval" / "fastv_transfer.json"),
        "historical_attack_unchanged": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precheck.json").write_text(json.dumps(payload, indent=2) + "\n")
    if not ok:
        raise SystemExit("precheck failed")
    return historical


def strict_attack(full_exact: bool, valid: bool, replay: bool, nonzero: bool, linf: float, anls: float) -> bool:
    return bool(valid and nonzero and linf <= EPS + TOLERANCE and full_exact and replay and anls < 0.5)


def strict_clean(full_exact: bool, valid: bool, replay: bool, anls: float) -> bool:
    return bool(valid and full_exact and replay and anls < 0.5)


def evaluate_images(model, processor, images, spec, clean_text: str) -> dict:
    folder_note = None
    first = measure(model, processor, images, spec)
    second = measure(model, processor, images, spec)
    replay = replay_ok(first, second)
    full_exact = second["full_text"] == clean_text and first["full_text"] == second["full_text"]
    return {"first": first, "second": second, "replay": replay, "full_exact": full_exact, "note": folder_note}


def make_noise(spec, seed: int) -> Path:
    clean = load_folder(HERE / "baseline_pages" / spec["sample_id"], spec["M"])
    aux = np.asarray(clean[spec["aux_idx"]].convert("RGB"), dtype=np.int16)
    rng = np.random.default_rng(seed)
    delta = rng.integers(-8, 9, size=aux.shape, dtype=np.int16)
    out = np.clip(aux + delta, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(out, mode="RGB")
    dest = OUT / "noise_png" / spec["sample_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(clean):
        page = noisy if index == spec["aux_idx"] else image
        page.save(dest / f"page_{index}.png")
    return dest


def main() -> None:
    historical = precheck()
    hist_rows = {row["sample_id"]: row for row in historical["rows"]}
    specs = {row["sample_id"]: row for row in manifest_samples()}
    model, processor = load_model()
    clean_rows = []
    attack_rows = []
    diverged = False
    for sample_id in IDS:
        spec = specs[sample_id]
        images = load_folder(HERE / "baseline_pages" / sample_id, spec["M"])
        measured = evaluate_images(model, processor, images, spec, spec["clean_full_pred"])
        status = "OK"
        if not measured["full_exact"]:
            status = "FULL_TEXT_MISMATCH"
        elif not measured["replay"]:
            status = "REPLAY_MISMATCH"
        elif not measured["second"]["full_valid"]:
            status = "INVALID_OUTPUT"
        ratios = {}
        hist = spec.get("clean_fastv_pred")
        for key, fast in measured["second"]["fast"].items():
            historical_match = None
            if key == "0.25":
                historical_match = {
                    "text_match": fast["text"] == hist,
                    "anls_recorded": spec.get("clean_fastv_anls"),
                    "anls_now": fast["anls"],
                    "keep_hash_recorded": spec.get("clean_keep_hash"),
                    "keep_hash_now": fast["keep_hash"],
                    "k_recorded": spec.get("clean_k"),
                    "k_now": fast["k"],
                }
            ratios[key] = {
                **fast,
                "strict": strict_clean(measured["full_exact"], fast["valid"] and measured["second"]["full_valid"], measured["replay"], fast["anls"]) if status == "OK" else None,
                "historical_r025": historical_match,
            }
        clean_rows.append({
            "sample_id": sample_id,
            "status": status,
            "full_text": measured["second"]["full_text"],
            "full_token_ids": None,
            "full_exact_text": measured["full_exact"],
            "token_exact_scope": TOKEN_SCOPE,
            "preseal_full_tokens": json.loads((HERE / "baseline_recovery.json").read_text())["generate_rows"] and next(row["tokens"] for row in json.loads((HERE / "baseline_recovery.json").read_text())["generate_rows"] if row["sample_id"] == sample_id),
            "replay": measured["replay"],
            "page_sha256": page_hashes(HERE / "baseline_pages" / sample_id, spec["M"]),
            "ratios": ratios,
        })
        print(json.dumps({"branch": "clean", "sample": sample_id, "status": status, "full": measured["second"]["full_text"]}), flush=True)

    for sample_id in IDS:
        spec = specs[sample_id]
        hist = hist_rows[sample_id]
        if not hist.get("evaluated"):
            attack_rows.append({"sample_id": sample_id, "status": "NO_VALID_SEALED_CANDIDATE", "evaluated": False, "ratios": None})
            continue
        folder = HERE / "sealed_png" / sample_id
        if page_hashes(folder, spec["M"])[spec["aux_idx"]] != hist["png_sha256"]:
            attack_rows.append({"sample_id": sample_id, "status": "PNG_HASH_MISMATCH", "evaluated": False})
            continue
        images = load_folder(folder, spec["M"])
        clean_aux = Image.open(HERE / "baseline_pages" / sample_id / f"page_{spec['aux_idx']}.png").convert("RGB")
        linf = uint8_linf(clean_aux, images[spec["aux_idx"]])
        if abs(linf - hist["linf"]) > 1e-6:
            attack_rows.append({"sample_id": sample_id, "status": "LINF_MISMATCH", "linf": linf, "historical_linf": hist["linf"], "evaluated": False})
            continue
        measured = evaluate_images(model, processor, images, spec, spec["clean_full_pred"])
        status = "OK"
        if measured["second"]["full_text"] != hist["attacked_full_text"] or any(measured["second"]["fast"][key]["text"] != hist["fastv"][key]["text"] for key in measured["second"]["fast"]):
            status = "REPLAY_DIVERGED"
            diverged = True
        elif not measured["full_exact"]:
            status = "FULL_TEXT_MISMATCH"
        elif not measured["replay"]:
            status = "REPLAY_MISMATCH"
        elif not measured["second"]["full_valid"] or any(not item["valid"] for item in measured["second"]["fast"].values()):
            status = "INVALID_OUTPUT"
        ratios = {}
        for key, fast in measured["second"]["fast"].items():
            ratios[key] = {
                **fast,
                "historical_text": hist["fastv"][key]["text"],
                "historical_anls": hist["fastv"][key]["anls"],
                "text_match": fast["text"] == hist["fastv"][key]["text"],
                "strict": strict_attack(measured["full_exact"], fast["valid"] and measured["second"]["full_valid"], measured["replay"], True, linf, fast["anls"]) if status == "OK" else None,
            }
        attack_rows.append({
            "sample_id": sample_id,
            "status": status,
            "evaluated": True,
            "full_text": measured["second"]["full_text"],
            "full_token_ids": None,
            "full_exact_text": measured["full_exact"],
            "token_exact_scope": TOKEN_SCOPE,
            "replay": measured["replay"],
            "linf": linf,
            "png_sha256": hist["png_sha256"],
            "ratios": ratios,
        })
        print(json.dumps({"branch": "attack", "sample": sample_id, "status": status}), flush=True)
        if diverged:
            break

    noise_rows = []
    if diverged:
        noise_rows.append({"status": "NOT_RUN", "reason": "REPLAY_DIVERGED"})
    else:
        for sample_id in IDS:
            spec = specs[sample_id]
            if not hist_rows[sample_id].get("evaluated"):
                noise_rows.append({"sample_id": sample_id, "status": "NO_VALID_SEALED_CANDIDATE", "evaluated": False, "ratios": None})
                continue
            seed = noise_seed(sample_id)
            folder = make_noise(spec, seed)
            images = load_folder(folder, spec["M"])
            clean_aux = Image.open(HERE / "baseline_pages" / sample_id / f"page_{spec['aux_idx']}.png").convert("RGB")
            linf = uint8_linf(clean_aux, images[spec["aux_idx"]])
            measured = evaluate_images(model, processor, images, spec, spec["clean_full_pred"])
            status = "OK"
            if sha_file(folder / f"page_{spec['aux_idx']}.png") == hist_rows[sample_id]["png_sha256"]:
                status = "NOISE_COLLIDED_WITH_ATTACK"
            elif not measured["full_exact"]:
                status = "FULL_TEXT_MISMATCH"
            elif not measured["replay"]:
                status = "REPLAY_MISMATCH"
            elif not measured["second"]["full_valid"]:
                status = "INVALID_OUTPUT"
            ratios = {}
            for key, fast in measured["second"]["fast"].items():
                ratios[key] = {
                    **fast,
                    "strict": strict_attack(measured["full_exact"], fast["valid"] and measured["second"]["full_valid"], measured["replay"], linf > 0, linf, fast["anls"]) if status == "OK" else None,
                }
            noise_rows.append({
                "sample_id": sample_id,
                "status": status,
                "evaluated": status == "OK",
                "seed": seed,
                "full_text": measured["second"]["full_text"],
                "full_token_ids": None,
                "full_exact_text": measured["full_exact"],
                "replay": measured["replay"],
                "linf": linf,
                "png_sha256": sha_file(folder / f"page_{spec['aux_idx']}.png"),
                "ratios": ratios,
            })
            print(json.dumps({"branch": "noise", "sample": sample_id, "status": status, "linf": linf}), flush=True)

    (OUT / "clean_baseline.json").write_text(json.dumps({"scope": SCOPE, "rows": clean_rows, "fastv_access_count": Access.fastv_calls}, indent=2) + "\n")
    (OUT / "attack_replay.json").write_text(json.dumps({"scope": SCOPE, "replay_diverged": diverged, "rows": attack_rows, "fastv_access_count": Access.fastv_calls}, indent=2) + "\n")
    (OUT / "noise_control.json").write_text(json.dumps({"scope": SCOPE, "rows": noise_rows, "fastv_access_count": Access.fastv_calls}, indent=2) + "\n")
    summary = summarize(clean_rows, attack_rows, noise_rows, diverged)
    (OUT / "controls.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "final_decision.json").write_text(json.dumps({"decision": summary["decision"], "scope": SCOPE, "fastv_access_count": Access.fastv_calls, "full_calls": Access.full_calls}, indent=2) + "\n")
    print(json.dumps({"decision": summary["decision"], "fastv_access_count": Access.fastv_calls}), flush=True)


def by_id(rows: list[dict]) -> dict:
    return {row["sample_id"]: row for row in rows if "sample_id" in row}


def summarize(clean_rows, attack_rows, noise_rows, diverged: bool) -> dict:
    clean = by_id(clean_rows)
    attack = by_id(attack_rows)
    noise = by_id(noise_rows)
    per_ratio = {}
    for ratio in RATIOS:
        key = ratio_key(ratio)
        induced = []
        anls_attack_clean = []
        anls_attack_noise = []
        clean_strict = attack_strict = noise_strict = 0
        for sample_id in IDS:
            c = clean.get(sample_id, {})
            a = attack.get(sample_id, {})
            n = noise.get(sample_id, {})
            c_ratio = (c.get("ratios") or {}).get(key)
            a_ratio = (a.get("ratios") or {}).get(key) if a.get("ratios") else None
            n_ratio = (n.get("ratios") or {}).get(key) if n.get("ratios") else None
            if c.get("status") == "OK" and c_ratio and c_ratio.get("strict"):
                clean_strict += 1
            if a.get("status") == "OK" and a_ratio and a_ratio.get("strict"):
                attack_strict += 1
            if n.get("status") == "OK" and n_ratio and n_ratio.get("strict"):
                noise_strict += 1
            if c.get("status") == "OK" and a.get("status") == "OK" and c_ratio and a_ratio:
                anls_attack_clean.append({"sample_id": sample_id, "delta": a_ratio["anls"] - c_ratio["anls"]})
                if c_ratio["anls"] >= 0.5 and a_ratio["anls"] < 0.5:
                    induced.append(sample_id)
            if a.get("status") == "OK" and n.get("status") == "OK" and a_ratio and n_ratio:
                anls_attack_noise.append({"sample_id": sample_id, "delta": a_ratio["anls"] - n_ratio["anls"]})
        per_ratio[key] = {
            "clean_strict": f"{clean_strict}/5",
            "attack_strict": f"{attack_strict}/5",
            "random_strict": f"{noise_strict}/5" if not diverged else "NOT_RUN",
            "attack_minus_clean_anls": anls_attack_clean,
            "attack_minus_random_anls": anls_attack_noise if not diverged else "NOT_RUN",
            "induced_failure_ids": induced,
            "induced_failure_count": len(induced),
            "induced_unmeasured": [sample_id for sample_id in IDS if attack.get(sample_id, {}).get("status") != "OK" or clean.get(sample_id, {}).get("status") != "OK"],
        }
    cell = None
    if clean.get("1025", {}).get("status") == "OK" and attack.get("1025", {}).get("status") == "OK":
        c = clean["1025"]["ratios"]["0.25"]
        a = attack["1025"]["ratios"]["0.25"]
        n = noise.get("1025", {}).get("ratios", {}) or {}
        ncell = n.get("0.25") if isinstance(n, dict) else None
        cell = {
            "clean_text": c["text"],
            "clean_anls": c["anls"],
            "attack_text": a["text"],
            "attack_anls": a["anls"],
            "induced": c["anls"] >= 0.5 and a["anls"] < 0.5,
            "full_exact": clean["1025"]["full_exact_text"] and attack["1025"]["full_exact_text"],
            "replay": clean["1025"]["replay"] and attack["1025"]["replay"],
            "random_anls": None if not ncell else ncell["anls"],
            "random_text": None if not ncell else ncell["text"],
        }
    r015 = per_ratio["0.15"]
    induced_015 = "1025" in r015["induced_failure_ids"]
    if diverged:
        decision = "REPLAY_DIVERGED"
    elif cell is None or cell["random_anls"] is None:
        decision = "GRAYBOX_CONTROL_INCONCLUSIVE"
    elif not (cell["induced"] and cell["full_exact"] and cell["replay"]):
        decision = "GRAYBOX_CONTROL_NOT_SUPPORTED"
    elif not (cell["attack_anls"] < cell["random_anls"]):
        decision = "GRAYBOX_CONTROL_NOT_SUPPORTED"
    else:
        decision = "GRAYBOX_CONTROL_CONFIRMED"
    return {
        "scope": SCOPE,
        "token_exact_scope": TOKEN_SCOPE,
        "decision": decision,
        "coverage": "4/5",
        "overall_attack_strict": {key: per_ratio[key]["attack_strict"] for key in per_ratio},
        "conditional_sealed_strict": {
            key: f"{sum(1 for sample_id in IDS if attack.get(sample_id, {}).get('status') == 'OK' and (attack[sample_id].get('ratios') or {}).get(key, {}).get('strict'))}/4"
            for key in per_ratio
        },
        "per_ratio": per_ratio,
        "sample_1025": {
            "r025": cell,
            "r015_induced_failure": induced_015,
            "r015_rule": "induced only if clean r=0.15 ANLS>=0.5 and attack r=0.15 ANLS<0.5",
        },
        "fastv_access_count": Access.fastv_calls,
        "full_calls": Access.full_calls,
    }


if __name__ == "__main__":
    main()
