"""Stage I gates. Reads diagnosis JSON only. Does not launch an attack."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from week2_attack_v7_budget_conditioned.constants import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CANDIDATE_R

HERE = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project/week2_attack_v7_budget_conditioned")


def load_rows() -> list[dict]:
    rows = []
    for path in sorted((HERE / "stage1").glob("shard_s*/*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_DRAWS))
    return means[int(0.025 * BOOTSTRAP_DRAWS)], means[int(0.975 * BOOTSTRAP_DRAWS) - 1]


def delta(row: dict, image: str, key: str) -> float:
    clean = row["images"]["clean"]["by_r"][key]["m_r"]
    attack = row["images"][image]["by_r"][key]["m_r"]
    d_r = clean - attack
    d_inf = row["images"]["clean"]["m_infinity"] - row["images"][image]["m_infinity"]
    return d_r - d_inf


def main() -> None:
    rows = load_rows()
    if len(rows) != 60 or len({row["sample_id"] for row in rows}) != 60:
        raise SystemExit(f"stage1 incomplete: {len(rows)}")
    legality = json.loads((HERE / "ratio_legality.json").read_text())
    ratios = [f"{ratio:.2f}" for ratio in CANDIDATE_R if ratio not in set(legality.get("removed_r", []))]
    by_r = {}
    monotone_hits = 0
    for row in rows:
        ordered = [delta(row, "B", key) for key in ratios]
        # Stronger compression is the smaller keep ratio, which is later in the frozen descending list.
        if ordered == sorted(ordered):
            monotone_hits += 1
    for key in ratios:
        paired = [delta(row, "B", key) - delta(row, "random_noise", key) for row in rows]
        lo, hi = bootstrap_ci(paired)
        strict_rows = [row for row in rows if row["images"]["B"]["by_r"][key]["strict"] == 1]
        other_rows = [row for row in rows if row["images"]["B"]["by_r"][key]["strict"] != 1]
        strict_delta = [delta(row, "B", key) for row in strict_rows]
        other_delta = [delta(row, "B", key) for row in other_rows]
        by_r[key] = {
            "n": 60,
            "paired_mean_B_minus_noise": sum(paired) / len(paired),
            "paired_bootstrap95": [lo, hi],
            "frac_delta_B_positive_all": sum(value > 0 for value in (delta(row, "B", key) for row in rows)) / 60,
            "frac_delta_B_positive_strict": (sum(value > 0 for value in strict_delta) / len(strict_delta)) if strict_delta else None,
            "median_delta_strict": statistics.median(strict_delta) if strict_delta else None,
            "median_delta_nonstrict": statistics.median(other_delta) if other_delta else None,
            "strict_n": len(strict_rows),
            "mean_aux_share_delta": statistics.fmean(
                row["images"]["B"]["by_r"][key]["aux_share"] - row["images"]["clean"]["by_r"][key]["aux_share"] for row in rows
            ),
            "mean_evidence_share_delta": statistics.fmean(
                row["images"]["B"]["by_r"][key]["evidence_share"] - row["images"]["clean"]["by_r"][key]["evidence_share"] for row in rows
            ),
        }
    focus = by_r["0.25"]
    conflicts = [
        row["sample_id"]
        for row in rows
        if any(row["images"][name]["margin_text_conflict"] for name in ("clean", "B", "random_noise"))
    ]
    gates = {
        "A_paired_delta_025": focus["paired_mean_B_minus_noise"] > 0 and focus["paired_bootstrap95"][0] > 0,
        "B_sixty_percent_positive": focus["frac_delta_B_positive_strict"] is not None and focus["frac_delta_B_positive_strict"] >= 0.60,
        "C_median_order": (
            focus["median_delta_strict"] is not None
            and focus["median_delta_nonstrict"] is not None
            and focus["median_delta_strict"] > focus["median_delta_nonstrict"]
        ),
        "D_margin_text_consistent": not conflicts,
    }
    decision = "STAGE1_GATES_PASS" if all(gates.values()) else "NO_GO_NO_COMPRESSION_CONDITIONAL_EFFECT"
    payload = {
        "decision": decision,
        "gates": gates,
        "conflicts": conflicts,
        "by_r": by_r,
        "samples_with_monotone_stronger_compression": monotone_hits,
        "monotone_claim_allowed": monotone_hits > 30,
        "cohort": "existing pristine60 Variant B diagnosis; not a B+ generalization result",
    }
    (HERE / "stage1_decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Stage I diagnosis",
        "",
        f"Decision: {decision}",
        "",
        "This is a mechanism diagnosis on the existing pristine60 Variant B images. It is not a generalization result.",
        "",
    ]
    for key, row in by_r.items():
        lines.append(
            f"- r={key}: paired mean {row['paired_mean_B_minus_noise']:.6f}, "
            f"bootstrap [{row['paired_bootstrap95'][0]:.6f}, {row['paired_bootstrap95'][1]:.6f}], "
            f"strict {row['strict_n']}/60, positive-among-strict {row['frac_delta_B_positive_strict']}, "
            f"median strict {row['median_delta_strict']}, median nonstrict {row['median_delta_nonstrict']}"
        )
    lines.append("")
    lines.append(f"Gates: {gates}")
    lines.append(f"Margin/text conflicts: {conflicts or 'none'}")
    lines.append(f"Samples whose Delta grows as the keep ratio falls: {monotone_hits}/60")
    (HERE / "stage1_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "gates": gates}))


if __name__ == "__main__":
    main()
