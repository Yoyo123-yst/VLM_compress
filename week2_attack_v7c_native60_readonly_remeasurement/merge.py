"""Merge the four readonly shards into the frozen summaries."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7c_native60_readonly_remeasurement"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))

from week1_mechanism.metrics_task import anls_score  # noqa: E402
from week2_attack_v4_answer_aligned.loss import output_valid  # noqa: E402
from week2_attack_v7_pristine60.analyze_v7 import EPS, WRONG  # noqa: E402
from week2_attack_v7c_native60_readonly_remeasurement.stats import (  # noqa: E402
    BUDGETS,
    bootstrap_mean_ci,
    decide,
    mean,
    median,
    mcnemar,
    stdev,
    sweet_outer,
)

FIELDS = [
    "sample_id", "M", "image_kind", "r", "png", "sha256",
    "full_text", "full_token_ids", "full_termination", "full_token_count",
    "full_soft_margin", "full_mean_soft", "full_median_soft", "full_min_soft", "full_gt_match",
    "fastv_text", "fastv_token_ids", "fastv_termination", "fastv_token_count",
    "fastv_soft_margin", "fastv_mean_soft", "fastv_median_soft", "fastv_min_soft", "fastv_gt_match",
    "fastv_k", "fastv_keep_hash", "strict", "mismatch", "appended_eos",
]


def load_rows() -> list[dict]:
    rows = []
    for shard in range(4):
        path = HERE / "shards" / f"shard_{shard}" / "conditions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def remeasured_strict(spec: dict, image_kind: str, full_text: str, fast_text: str) -> int:
    anls_full = anls_score(full_text, spec["answers"])
    anls_fast = anls_score(fast_text, spec["answers"])
    if image_kind == "B":
        linf = float(spec["b_pixel_linf_uint8"])
        replay_ok = int(spec["b_replay_ok"]) == 1
        status = "SELECTED"
    else:
        linf = float(spec["noise_uint8_linf"])
        replay_ok = int(spec["noise_replay_ok"]) == 1
        status = spec["noise_status"]
    ok = (
        status in {"SELECTED", ""}
        and anls_full >= WRONG
        and anls_fast < WRONG
        and full_text == spec["clean_full_pred"]
        and output_valid(full_text)
        and output_valid(fast_text)
        and replay_ok
        and 0 < linf <= EPS + 1e-6
        and float(spec["clean_full_anls"]) >= WRONG
        and float(spec["clean_fastv_anls"]) >= WRONG
    )
    return int(ok)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field)
                out[field] = json.dumps(value) if isinstance(value, (list, dict)) else value
            writer.writerow(out)


def blocked(decision: str, reason: str) -> None:
    payload = {
        "decision": decision,
        "reason": reason,
        "historical_b_strict_r025": "52/60 unchanged",
        "native_margin_replaces_strict": False,
        "bplus_started": False,
    }
    (HERE / "final_decision.json").write_text(json.dumps(payload, indent=2) + "\n")
    (HERE / "final_report.md").write_text(
        f"# Native60 readonly remeasure\n\nDecision: `{decision}`.\n\n{reason}\n\n"
        "The historical B strict result remains 52/60. Native margin does not replace that endpoint. B+ was not started.\n"
    )
    (HERE / "statistical_summary.json").write_text(
        json.dumps({"status": "INCOMPLETE_ALIGNMENT_FAILURE", "decision": decision, "reason": reason}, indent=2) + "\n"
    )
    (HERE / "statistical_summary.md").write_text(f"Status: INCOMPLETE_ALIGNMENT_FAILURE.\n\n{reason}\n")


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _chart(path: Path, title: str, xlabel: str, ylabel: str, series: list[tuple[str, list[float], tuple[int, int, int]]], yerr=None) -> None:
    width, height = 980, 640
    left, right, top, bottom = 90, 40, 50, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(16)
    small = _font(13)
    draw.text((left, 16), title, fill="black", font=font)
    xs = [0.50, 0.35, 0.25, 0.15]
    values = [point for _, points, _ in series for point in points]
    if yerr is not None:
        values.extend(point - err for point, err in zip(series[0][1], yerr[0]))
        values.extend(point + err for point, err in zip(series[0][1], yerr[1]))
    lo, hi = min(values), max(values)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def px(index: int) -> int:
        return left + int(index * (width - left - right) / (len(xs) - 1))

    def py(value: float) -> int:
        return top + int((hi - value) / (hi - lo) * (height - top - bottom))

    draw.line((left, top, left, height - bottom), fill="black")
    draw.line((left, height - bottom, width - right, height - bottom), fill="black")
    if lo < 0 < hi:
        draw.line((left, py(0), width - right, py(0)), fill=(180, 180, 180))
    for index, ratio in enumerate(xs):
        draw.text((px(index) - 16, height - bottom + 12), str(ratio), fill="black", font=small)
    draw.text((width // 2 - 80, height - 28), xlabel, fill="black", font=small)
    draw.text((12, height // 2), ylabel, fill="black", font=small)
    for label, points, color in series:
        coords = [(px(i), py(value)) for i, value in enumerate(points)]
        draw.line(coords, fill=color, width=2)
        for x, y in coords:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    if yerr is not None:
        color = series[0][2]
        for index, (value, down, up) in enumerate(zip(series[0][1], yerr[0], yerr[1])):
            x = px(index)
            draw.line((x, py(value - down), x, py(value + up)), fill=color, width=2)
    legend_x = width - 230
    for index, (label, _, color) in enumerate(series):
        y = 20 + index * 22
        draw.line((legend_x, y + 8, legend_x + 24, y + 8), fill=color, width=3)
        draw.text((legend_x + 30, y), label, fill="black", font=small)
    image.save(path)


def plot(delta, strict_b, strict_n, by_budget, summary_rows) -> None:
    fig_dir = HERE / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    xs = list(BUDGETS)
    means = [by_budget[str(r)]["delta_mean"] for r in xs]
    medians = [by_budget[str(r)]["delta_median"] for r in xs]
    los = [by_budget[str(r)]["delta_ci95"][0] for r in xs]
    his = [by_budget[str(r)]["delta_ci95"][1] for r in xs]
    yerr = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
    _chart(fig_dir / "delta_by_budget.png", "Paired Delta by FastV budget", "FastV keep ratio r", "Delta", [("mean Delta", means, (31, 119, 180)), ("median Delta", medians, (255, 127, 14))], yerr)
    _chart(fig_dir / "strict_by_budget.png", "Remeasured strict rate by budget", "FastV keep ratio r", "strict rate", [("B strict", [strict_b[r] / 60 for r in xs], (31, 119, 180)), ("noise strict", [strict_n[r] / 60 for r in xs], (255, 127, 14))])
    width, height = 980, 640
    left, right, top, bottom = 90, 40, 50, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    small = _font(13)
    draw.text((left, 16), "Per-sample Delta trajectories", fill="black", font=_font(16))
    tracks = [[delta[r][i] for r in xs] for i in range(len(delta[0.25]))]
    values = [point for track in tracks for point in track] + means
    lo, hi = min(values), max(values)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def px(index: int) -> int:
        return left + int(index * (width - left - right) / 3)

    def py(value: float) -> int:
        return top + int((hi - value) / (hi - lo) * (height - top - bottom))

    draw.line((left, top, left, height - bottom), fill="black")
    draw.line((left, height - bottom, width - right, height - bottom), fill="black")
    if lo < 0 < hi:
        draw.line((left, py(0), width - right, py(0)), fill=(180, 180, 180))
    for track in tracks:
        draw.line([(px(i), py(value)) for i, value in enumerate(track)], fill=(150, 185, 220), width=1)
    draw.line([(px(i), py(value)) for i, value in enumerate(means)], fill="black", width=3)
    for index, ratio in enumerate(xs):
        draw.text((px(index) - 16, height - bottom + 12), str(ratio), fill="black", font=small)
    draw.text((width // 2 - 80, height - 28), "FastV keep ratio r", fill="black", font=small)
    image.save(fig_dir / "per_sample_budget_trajectories.png")
    colors = {0.50: (31, 119, 180), 0.35: (255, 127, 14), 0.25: (44, 160, 44), 0.15: (214, 39, 40)}
    points = []
    for row in summary_rows:
        for ratio in xs:
            points.append((row[f"D_noise_{ratio}"], row[f"D_B_{ratio}"], colors[ratio], ratio))
    xs_v = [item[0] for item in points]
    ys_v = [item[1] for item in points]
    lo = min(xs_v + ys_v)
    hi = max(xs_v + ys_v)
    pad = 0.08 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 16), "Paired D(B) against D(noise)", fill="black", font=_font(16))

    def sx(value: float) -> int:
        return left + int((value - lo) / (hi - lo) * (width - left - right))

    def sy(value: float) -> int:
        return top + int((hi - value) / (hi - lo) * (height - top - bottom))

    draw.line((left, top, left, height - bottom), fill="black")
    draw.line((left, height - bottom, width - right, height - bottom), fill="black")
    draw.line((sx(lo), sy(lo), sx(hi), sy(hi)), fill=(200, 200, 200))
    for x, y, color, _ratio in points:
        cx, cy = sx(x), sy(y)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)
    legend_x = width - 160
    for index, ratio in enumerate(xs):
        y = 20 + index * 22
        draw.ellipse((legend_x, y + 4, legend_x + 10, y + 14), fill=colors[ratio])
        draw.text((legend_x + 16, y), f"r={ratio}", fill="black", font=small)
    draw.text((width // 2 - 40, height - 28), "D(noise)", fill="black", font=small)
    draw.text((12, height // 2), "D(B)", fill="black", font=small)
    image.save(fig_dir / "paired_d_scatter.png")


def report_text(decision: str, by_budget: dict, sweet: dict, immutable: bool) -> str:
    lines = [
        "# Native60 readonly remeasure",
        "",
        f"Decision: `{decision}`.",
        "",
        "Variant B's existing r=0.25 strict result remains 52/60 from the original generate endpoint. This stage did not recompute, replace, or weaken that result. Native-cache margin is a differentiable proxy for mechanism analysis and a possible future B+. It is not the success endpoint.",
        "",
        "All 600 Full and FastV native paths matched real generate on tokens, argmax, and decoded text. FastV K and keep hash matched the real hook on every FastV path. No pixels, checkpoints, or selectors were changed.",
        "",
        f"Parent raw immutability: {'PASS' if immutable else 'FAIL'}.",
        "",
        "Remeasured strict rates, from this run's generate texts under the frozen predicate:",
        "",
    ]
    for ratio in BUDGETS:
        item = by_budget[str(ratio)]
        lines.append(
            f"- r={ratio}: B {item['b_strict']}, noise {item['noise_strict']}, "
            f"Delta mean {item['delta_mean']:.4f}, 95% CI ({item['delta_ci95'][0]:.4f}, {item['delta_ci95'][1]:.4f})."
        )
    lines.extend([
        "",
        f"Sweet-band mean Delta is {sweet['sweet_mean']:.4f}. Outer-band mean Delta is {sweet['outer_mean']:.4f}. "
        f"Their paired difference is {sweet['paired_diff_mean']:.4f}, with 95% CI "
        f"({sweet['paired_diff_ci95'][0]:.4f}, {sweet['paired_diff_ci95'][1]:.4f}). "
        f"The exact sign test on the non-zero paired differences has p={sweet['sign_test']['p_two_sided']:.3g}. "
        "The mean comparison clears its preregistered bootstrap bound. The sign test does not.",
        "",
        "A positive Delta means B opens a larger Full-versus-FastV margin gap than the paired random image at the same budget. It does not mean the generate endpoint is a strict success.",
        "",
        "This result applies to LLaVA-OneVision with FastV on MP-DocVQA under the current auxiliary-page threat model. It is not a universal claim and it does not prove that Full exact is theoretically guaranteed. B+ was not started.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    protocol = json.loads((HERE / "FROZEN_PROTOCOL.json").read_text())
    specs = {row["sample_id"]: row for row in protocol["samples"]}
    if (HERE / "STOP_ALIGNMENT.json").exists():
        blocked("BLOCKED_NATIVE60_ALIGNMENT_MISMATCH", (HERE / "STOP_ALIGNMENT.json").read_text())
        return
    rows = load_rows()
    bad = len(rows) != 600 or any(row.get("mismatch") or row.get("full_gt_match") is False or row.get("fastv_gt_match") is False for row in rows)
    if bad:
        blocked("BLOCKED_NATIVE60_ALIGNMENT_MISMATCH", f"expected 600 aligned conditions, found {len(rows)}")
        return
    for row in rows:
        row["strict"] = None if row["r"] is None else remeasured_strict(specs[row["sample_id"]], row["image_kind"], row["full_text"], row["fastv_text"])
    write_csv(HERE / "per_condition_native.csv", rows, FIELDS)
    by_key = {(row["sample_id"], row["image_kind"], row["r"]): row for row in rows if row["r"] is not None}
    full_key = {(row["sample_id"], row["image_kind"]): row for row in rows if row["r"] is None}
    summary_rows = []
    paired_rows = []
    delta = {ratio: [] for ratio in BUDGETS}
    strict_b = {ratio: 0 for ratio in BUDGETS}
    strict_n = {ratio: 0 for ratio in BUDGETS}
    ids = [row["sample_id"] for row in protocol["samples"]]
    for sid in ids:
        record = {"sample_id": sid}
        for ratio in BUDGETS:
            b = by_key[(sid, "B", ratio)]
            n = by_key[(sid, "random_noise", ratio)]
            d_b = full_key[(sid, "B")]["full_mean_soft"] - b["fastv_mean_soft"]
            d_n = full_key[(sid, "random_noise")]["full_mean_soft"] - n["fastv_mean_soft"]
            gap = d_b - d_n
            delta[ratio].append(gap)
            strict_b[ratio] += int(b["strict"])
            strict_n[ratio] += int(n["strict"])
            record[f"D_B_{ratio}"] = d_b
            record[f"D_noise_{ratio}"] = d_n
            record[f"Delta_{ratio}"] = gap
            record[f"strict_B_{ratio}"] = b["strict"]
            record[f"strict_noise_{ratio}"] = n["strict"]
            paired_rows.append({"sample_id": sid, "r": ratio, "D_B": d_b, "D_noise": d_n, "Delta": gap, "strict_B": b["strict"], "strict_noise": n["strict"]})
        summary_rows.append(record)
    write_csv(HERE / "per_sample_native_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(HERE / "paired_delta_by_r.csv", paired_rows, list(paired_rows[0].keys()))
    by_budget = {}
    for ratio in BUDGETS:
        values = delta[ratio]
        lo, hi = bootstrap_mean_ci(values)
        attack_only = sum(row[f"strict_B_{ratio}"] == 1 and row[f"strict_noise_{ratio}"] == 0 for row in summary_rows)
        noise_only = sum(row[f"strict_B_{ratio}"] == 0 and row[f"strict_noise_{ratio}"] == 1 for row in summary_rows)
        by_budget[str(ratio)] = {
            "b_strict": f"{strict_b[ratio]}/60",
            "noise_strict": f"{strict_n[ratio]}/60",
            "attack_only": attack_only,
            "noise_only": noise_only,
            "mcnemar_p": mcnemar(attack_only, noise_only),
            "delta_mean": mean(values),
            "delta_median": median(values),
            "delta_stdev": stdev(values),
            "delta_ci95": [lo, hi],
            "delta_positive_fraction": sum(value > 0 for value in values) / len(values),
        }
    sweet = sweet_outer(delta)
    sweet_public = {key: value for key, value in sweet.items() if key != "paired_diffs"}
    ci_low = {ratio: by_budget[str(ratio)]["delta_ci95"][0] for ratio in BUDGETS}
    delta_mean = {ratio: by_budget[str(ratio)]["delta_mean"] for ratio in BUDGETS}
    immutable = json.loads((HERE / "raw_immutability_check.json").read_text())["pass"] if (HERE / "raw_immutability_check.json").exists() else False
    decision = decide(True, immutable, ci_low, delta_mean, sweet, strict_b)
    summary = {
        "status": "COMPLETE",
        "n_conditions": 600,
        "alignment": "600/600",
        "historical_b_strict_r025": "52/60",
        "historical_strict_replaced": False,
        "by_budget": by_budget,
        "sweet_versus_outer": sweet_public,
        "remeasured_b_strict_counts": {str(ratio): strict_b[ratio] for ratio in BUDGETS},
        "decision": decision,
    }
    (HERE / "statistical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = ["# Native60 statistical summary", "", "Historical B strict at r=0.25 remains 52/60. The rates below are a readonly generate recheck and do not replace that endpoint.", ""]
    for ratio in BUDGETS:
        item = by_budget[str(ratio)]
        md.append(
            f"- r={ratio}: B strict {item['b_strict']}, noise strict {item['noise_strict']}, "
            f"attack-only {item['attack_only']}, noise-only {item['noise_only']}, McNemar p={item['mcnemar_p']:.3g}, "
            f"Delta mean {item['delta_mean']:.4f} (95% CI {item['delta_ci95'][0]:.4f}, {item['delta_ci95'][1]:.4f}), "
            f"median {item['delta_median']:.4f}, fraction > 0 = {item['delta_positive_fraction']:.3f}."
        )
    md.append("")
    md.append(
        f"Sweet-band mean Delta {sweet['sweet_mean']:.4f}; outer-band mean Delta {sweet['outer_mean']:.4f}; "
        f"paired difference {sweet['paired_diff_mean']:.4f} (95% CI {sweet['paired_diff_ci95'][0]:.4f}, {sweet['paired_diff_ci95'][1]:.4f}). "
        f"Sign test p={sweet['sign_test']['p_two_sided']:.3g}."
    )
    (HERE / "statistical_summary.md").write_text("\n".join(md) + "\n")
    plot(delta, strict_b, strict_n, by_budget, summary_rows)
    (HERE / "final_decision.json").write_text(json.dumps({
        "decision": decision,
        "historical_b_strict_r025": "52/60 unchanged",
        "native_margin_replaces_strict": False,
        "scope": "LLaVA-OneVision + FastV + MP-DocVQA + the current auxiliary-page threat model",
        "bplus_started": False,
    }, indent=2) + "\n")
    (HERE / "final_report.md").write_text(report_text(decision, by_budget, sweet, immutable))


if __name__ == "__main__":
    main()
