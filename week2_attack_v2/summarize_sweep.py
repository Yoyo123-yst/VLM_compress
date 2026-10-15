"""Summarize the λ, c sweep and choose the operating point."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "week2_attack_v2" / "results" / "lambda_sweep.csv"
OUT = ROOT / "week2_attack_v2" / "results" / "lambda_sweep.md"


def mean(rows, key):
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    rows = list(csv.DictReader(CSV.open()))
    clean = {r["sample_id"]: r for r in rows if r["arm"] == "clean"}
    groups = defaultdict(list)
    for row in rows:
        if row["arm"] != "v2":
            continue
        groups[(row["lambda"], row["T_c"], row["steps"])].append(row)
    lines = [
        "# λ 和 c 的选择",
        "",
        "筛选集是新子集（不压缩 ANLS≥0.5 且 FastV r=0.5 ANLS≥0.5 且 M>1）里种子 42 抽出的 40 题，再从中抽 12 题做网格。",
        "每格 Adam、步长 2/255、ε=8/255。L_share 用份额刻度的下侧平方 `(relu(q* - soft_share))²`，q*=0.6。",
        "原始的 token 计数平方大约是 10⁶，λ∈{0.5,1,2,5} 会失去作用，所以除以保留数，目标和用户写的计数形式只差一个常数。",
        "L_full 是答案 span 上的平均 KL(p_clean || p_adv)，teacher-forcing 用干净贪心答案。",
        "",
        "| λ | c | steps | n | 辅图份额 | 压缩 ANLS | 不压缩 ANLS | 不压缩掉幅 | CSFR 严格 | CSFR 宽松 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = []
    for (lam, c, steps), items in sorted(groups.items(), key=lambda kv: (float(kv[0][0]), float(kv[0][1]))):
        ids = [r["sample_id"] for r in items if r["sample_id"] in clean]
        paired = [r for r in items if r["sample_id"] in clean]
        if not paired:
            continue
        drop = mean([clean[r["sample_id"]] for r in paired], "anls_full") - mean(paired, "anls_full")
        # mean() on dicts via the list of clean rows
        drop = mean([clean[r["sample_id"]] for r in paired], "anls_full") - mean(paired, "anls_full")
        strict = mean(paired, "csf_strict")
        loose = mean(paired, "csf_loose")
        lines.append(
            f"| {lam} | {c} | {steps} | {len(paired)} | {mean(paired,'aux_share'):.3f} | "
            f"{mean(paired,'anls_compressed'):.3f} | {mean(paired,'anls_full'):.3f} | {drop:.3f} | "
            f"{strict:.3f} | {loose:.3f} |"
        )
        ranked.append((drop, strict, loose, lam, c, steps, len(paired), mean(paired, "aux_share")))
    eligible = [r for r in ranked if r[0] < 0.03]
    lines += ["", "## 选择", ""]
    if not ranked:
        lines.append("还没有 v2 行。")
    elif eligible:
        best = max(eligible, key=lambda r: (r[1], r[2], -r[0]))
        lines.append(
            f"不压缩掉幅 < 0.03 的格子里，严格 CSFR 最高的是 λ={best[3]}，c={best[4]}，steps={best[5]} "
            f"（n={best[6]}，CSFR 严格 {best[1]:.3f}，宽松 {best[2]:.3f}，掉幅 {best[0]:.3f}，辅图份额 {best[7]:.3f}）。"
        )
    else:
        best = min(ranked, key=lambda r: (r[0], -r[1]))
        lines.append(
            f"没有任何格子把不压缩掉幅压到 0.03 以下。掉幅最小的是 λ={best[3]}，c={best[4]}，steps={best[5]} "
            f"（掉幅 {best[0]:.3f}，严格 CSFR {best[1]:.3f}，辅图份额 {best[7]:.3f}）。"
        )
    # v1 reference at the same steps if present
    for steps in sorted({r["steps"] for r in rows if r["arm"] == "v1"}):
        v1 = [r for r in rows if r["arm"] == "v1" and r["steps"] == steps and r["sample_id"] in clean]
        if not v1:
            continue
        drop = mean([clean[r["sample_id"]] for r in v1], "anls_full") - mean(v1, "anls_full")
        lines.append(
            f"同一步数的 v1（无 L_full）：n={len(v1)}，辅图份额 {mean(v1,'aux_share'):.3f}，"
            f"压缩 ANLS {mean(v1,'anls_compressed'):.3f}，不压缩掉幅 {drop:.3f}，"
            f"严格 CSFR {mean(v1,'csf_strict'):.3f}，宽松 {mean(v1,'csf_loose'):.3f}。"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
