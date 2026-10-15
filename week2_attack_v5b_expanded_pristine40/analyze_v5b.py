"""Read-only statistics for v5b. Writes only under results/ and analysis/."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
WRONG = 0.5
EPS = 8 / 255


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def signflip_p(values: list[float]) -> float | None:
    nonzero = [v for v in values if v != 0]
    k = len(nonzero)
    if k == 0:
        return 1.0
    if k > 20:
        return None
    observed = abs(sum(nonzero))
    count = 0
    total = 1 << k
    for mask in range(total):
        total_sum = 0.0
        for i, value in enumerate(nonzero):
            total_sum += value if (mask >> i) & 1 else -value
        if abs(total_sum) >= observed - 1e-12:
            count += 1
    return count / total


def bootstrap_ci(values: list[float], seed: int = 2026100804, reps: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def quadrant(full_anls: float, fast_anls: float, valid: bool) -> str:
    if not valid:
        return "INVALID_OUTPUT"
    full_wrong = full_anls < WRONG
    fast_wrong = fast_anls < WRONG
    if fast_wrong and not full_wrong:
        return "fastv_wrong_full_right"
    if (not fast_wrong) and full_wrong:
        return "fastv_right_full_wrong"
    if fast_wrong and full_wrong:
        return "both_wrong"
    return "both_right"


def strict_ok(row: dict, clean_full: str, replay_ok: bool) -> bool:
    return (
        float(row["anls_full"]) >= WRONG
        and float(row["anls_fastv"]) < WRONG
        and row["pred_full"] == clean_full
        and bool(row["output_valid"])
        and replay_ok
        and float(row["linf"]) <= EPS + 1e-6
    )


def load_checkpoints() -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for path in sorted((V5B / "attacks").glob("shard_s*/checkpoints.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                grouped[row["sample_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return grouped


def load_noise() -> dict[str, dict]:
    found = {}
    for path in sorted((V5B / "random_control").glob("shard_s*/noise.csv")):
        for row in csv.DictReader(path.open()):
            found[row["sample_id"]] = row
    return found


def load_replay() -> dict[tuple[str, str], dict]:
    found = {}
    for path in sorted((V5B / "replay").glob("shard_s*.json")):
        for row in json.loads(path.read_text()):
            found[(row["sample_id"], row["kind"])] = row
    return found


def arm_summary(rows: list[dict]) -> dict:
    n = len(rows)
    strict = sum(row["strict"] for row in rows)
    quads = defaultdict(int)
    for row in rows:
        quads[row["quadrant"]] += 1
    lo, hi = wilson(strict, n)
    return {
        "n": n,
        "strict": strict,
        "strict_csfr": strict / n if n else None,
        "wilson95": [lo, hi],
        "quadrants": dict(quads),
        "full_errors": sum(float(row["anls_full"]) < WRONG for row in rows),
        "full_text_unchanged": sum(row["full_text_same"] for row in rows),
        "invalid": sum(row["quadrant"] == "INVALID_OUTPUT" for row in rows),
        "replay_pass": sum(row["replay_ok"] for row in rows),
        "linf_pass": sum(float(row["linf"]) <= EPS + 1e-6 for row in rows),
        "mean_G": sum(row["G"] for row in rows) / n if n else None,
    }


def main():
    manifest = json.loads((V5B / "frozen_manifest.json").read_text())
    specs = {row["sample_id"]: row for row in manifest["samples"]}
    grouped = load_checkpoints()
    noise = load_noise()
    replay = load_replay()
    if len(grouped) != 40 or len(noise) != 40:
        raise SystemExit(f"incomplete raw attack={len(grouped)} noise={len(noise)}")
    primary_rows = []
    secondary_rows = []
    oracle_rows = []
    noise_rows = []
    paired = []
    all_ck = []
    for sid, spec in specs.items():
        rows = grouped[sid]
        clean_full = spec["clean_full_pred"]
        s_f = float(spec["clean_full_anls"])
        s_c = float(spec["clean_fastv_anls"])
        for row in rows:
            row["D_F"] = s_f - float(row["anls_full"])
            row["D_C"] = s_c - float(row["anls_fastv"])
            row["G"] = row["D_C"] - row["D_F"]
            row["quadrant"] = quadrant(float(row["anls_full"]), float(row["anls_fastv"]), bool(row["output_valid"]))
            all_ck.append(row)
        final = next(row for row in rows if int(row["step"]) == 200)
        margin = next(row for row in rows if row.get("selected_by_min_margin"))
        oracle_pool = [row for row in rows if row["output_valid"]]
        oracle = sorted(oracle_pool, key=lambda row: (-(row["quadrant"] == "fastv_wrong_full_right" and row["pred_full"] == clean_full), -row["G"], row["step"]))[0]
        def pack(row, kind):
            rep = replay.get((sid, kind), {"ok": False})
            replay_ok = bool(rep.get("ok"))
            return {
                "sample_id": sid,
                "M": spec["M"],
                "step": int(row["step"]) if "step" in row and str(row["step"]) != "" else "",
                "anls_full": float(row["anls_full"]),
                "anls_fastv": float(row["anls_fastv"]),
                "pred_full": row["pred_full"],
                "pred_fastv": row["pred_fastv"],
                "output_valid": bool(row["output_valid"]),
                "margin": float(row["margin"]) if "margin" in row else "",
                "full_kl": float(row["full_kl"]) if "full_kl" in row else "",
                "keep_hash": row.get("keep_hash", ""),
                "k": row.get("k", ""),
                "aux_share": row.get("aux_share", ""),
                "evidence_share": row.get("evidence_share", ""),
                "linf": float(row["linf"]) if "linf" in row else float(row.get("uint8_linf", 1)),
                "D_F": row["D_F"] if "D_F" in row else s_f - float(row["anls_full"]),
                "D_C": row["D_C"] if "D_C" in row else s_c - float(row["anls_fastv"]),
                "G": row["G"] if "G" in row else (s_c - float(row["anls_fastv"])) - (s_f - float(row["anls_full"])),
                "quadrant": row["quadrant"] if "quadrant" in row else quadrant(float(row["anls_full"]), float(row["anls_fastv"]), bool(row["output_valid"])),
                "full_text_same": row["pred_full"] == clean_full,
                "replay_ok": replay_ok,
                "strict": strict_ok(row, clean_full, replay_ok),
            }
        noise_row = noise[sid]
        noise_row = {
            **noise_row,
            "anls_full": float(noise_row["anls_full"]),
            "anls_fastv": float(noise_row["anls_fastv"]),
            "output_valid": str(noise_row["output_valid"]) in ("True", "1", "true"),
            "linf": float(noise_row["uint8_linf"]),
        }
        primary_rows.append(pack(final, "final"))
        secondary_rows.append(pack(margin, "min_margin"))
        oracle_packed = pack(oracle, "final")
        oracle_packed["strict"] = (
            oracle["quadrant"] == "fastv_wrong_full_right" and oracle["pred_full"] == clean_full and bool(oracle["output_valid"])
        )
        oracle_packed["status"] = "GROUND_TRUTH_ORACLE_UPPER_BOUND"
        oracle_rows.append(oracle_packed)
        noise_packed = pack(noise_row, "random_noise")
        noise_rows.append(noise_packed)
        paired.append({
            "sample_id": sid,
            "attack_strict": int(primary_rows[-1]["strict"]),
            "noise_strict": int(noise_rows[-1]["strict"]),
            "G_attack": primary_rows[-1]["G"],
            "G_noise": noise_rows[-1]["G"],
            "G_diff": primary_rows[-1]["G"] - noise_rows[-1]["G"],
        })
    b = sum(row["attack_strict"] and not row["noise_strict"] for row in paired)
    c = sum(row["noise_strict"] and not row["attack_strict"] for row in paired)
    gaps = [row["G_diff"] for row in paired]
    p_mc = mcnemar(b, c)
    p_sign = signflip_p(gaps)
    boot = bootstrap_ci(gaps)
    primary = arm_summary(primary_rows)
    both_wrong = primary["quadrants"].get("both_wrong", 0)
    full_pass = (
        primary["strict"] >= 27
        and primary["wilson95"][0] > 0.5
        and primary["full_errors"] == 0
        and primary["full_text_unchanged"] == 40
        and both_wrong == 0
        and primary["invalid"] == 0
        and primary["replay_pass"] == 40
        and primary["linf_pass"] == 40
    )
    noise_sep = b > c and p_mc < 0.05
    text_gap = 40 - primary["full_text_unchanged"]
    if primary["n"] != 40:
        decision = "BLOCKED_INCOMPLETE_RESULTS"
    elif full_pass and noise_sep and p_sign is not None and p_sign >= 0.05:
        decision = "NOT_ESTABLISHED"
    elif full_pass and noise_sep:
        decision = "HOLDOUT_SUPPORTS_STRICT_COMP_ONLY_ON_EXPANDED_PRISTINE40"
    elif full_pass and not noise_sep:
        decision = "SUPPORT_WITH_LIMITATION_RANDOM_CONTROL_NOT_SEPARATED"
    elif (
        primary["strict"] >= 27
        and primary["wilson95"][0] > 0.5
        and primary["full_errors"] == 0
        and both_wrong == 0
        and primary["invalid"] == 0
        and primary["replay_pass"] == 40
        and primary["linf_pass"] == 40
        and noise_sep
        and text_gap > 0
    ):
        decision = "NOT_COMPLETE_PASS_FULL_TEXT_CHANGED"
    elif primary["strict"] > 20:
        decision = "WEAK_SIGNAL_ON_EXPANDED_PRISTINE40"
    else:
        decision = "NO_SUPPORT_ON_EXPANDED_PRISTINE40"
    raw_hashes = {}
    for path in sorted((V5B / "attacks").glob("shard_s*/*")):
        if path.is_file():
            raw_hashes[str(path.relative_to(V5B))] = hashlib.sha256(path.read_bytes()).hexdigest()
    analysis = V5B / "analysis"
    results = V5B / "results"
    analysis.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    def dump(path, rows):
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    dump(results / "fixed_step200.csv", primary_rows)
    dump(results / "min_margin.csv", secondary_rows)
    dump(results / "oracle_diagnostic.csv", oracle_rows)
    dump(results / "random_noise.csv", noise_rows)
    dump(results / "all_checkpoints.csv", all_ck)
    dump(results / "all_samples.csv", primary_rows)
    dump(analysis / "paired_attack_vs_noise.csv", paired)
    stats = {
        "decision": decision,
        "primary": primary,
        "secondary": arm_summary(secondary_rows),
        "oracle": {"strict": sum(row["strict"] for row in oracle_rows), "status": "GROUND_TRUTH_ORACLE_UPPER_BOUND"},
        "noise": arm_summary(noise_rows),
        "mcnemar_p": p_mc,
        "candidate_only": b,
        "noise_only": c,
        "mean_G_diff": sum(gaps) / len(gaps),
        "bootstrap_ci95": list(boot),
        "signflip_p_exact": p_sign,
        "by_M": {},
    }
    for m in (2, 3, 4):
        subset = [row for row in primary_rows if int(row["M"]) == m]
        stats["by_M"][str(m)] = arm_summary(subset) if subset else None
    (analysis / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (analysis / "replay_summary.json").write_text(json.dumps({
        "final": sum(row["replay_ok"] for row in primary_rows),
        "min_margin": sum(row["replay_ok"] for row in secondary_rows),
        "noise": sum(row["replay_ok"] for row in noise_rows),
    }, indent=2) + "\n")
    (analysis / "raw_immutability_check.json").write_text(json.dumps({"files": raw_hashes, "note": "hashes recorded at analysis start; analysis does not write these paths"}, indent=2) + "\n")
    (results / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    lines = [
        "# v5b statistics",
        "",
        f"- Decision: `{decision}`",
        f"- Fixed step 200 strict: {primary['strict']}/40, Wilson {primary['wilson95']}",
        f"- Full text unchanged: {primary['full_text_unchanged']}/40",
        f"- Replay: {primary['replay_pass']}/40",
        f"- Noise strict: {stats['noise']['strict']}/40",
        f"- McNemar p={p_mc}, attack-only {b}, noise-only {c}",
        f"- Sign-flip p={p_sign}",
        f"- Oracle upper bound strict count: {stats['oracle']['strict']}/40",
        f"- Min-margin strict: {stats['secondary']['strict']}/40",
    ]
    (analysis / "statistics.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "strict": primary["strict"], "noise": stats["noise"]["strict"], "mcnemar": p_mc}))


if __name__ == "__main__":
    main()
