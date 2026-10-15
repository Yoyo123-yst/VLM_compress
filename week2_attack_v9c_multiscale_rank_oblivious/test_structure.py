"""CPU structure tests. Do not load the language model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from spatial_pool import (
    BASE,
    HIGH,
    PAGE_TOKENS,
    apply_plan,
    make_plan,
    pack_sequence,
)

SEED = 2026101117
EXPECTED = {
    0.25: {"height": 14, "width": 13, "tokens": 925},
    0.35: {"height": 21, "width": 26, "tokens": 1296},
    0.50: {"height": 33, "width": 33, "tokens": 1851},
}


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def get_input_embeddings(self):
        return self.embed


def main() -> None:
    checks = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": ok, "detail": detail})
        if not ok:
            raise RuntimeError(f"{name}: {detail}")

    for ratio, expected in EXPECTED.items():
        plan = make_plan("BasePreservingAdaptivePool", ratio, "1025", 0, 0, 0, SEED)
        record(f"grid-{ratio}", (plan.height, plan.width, plan.page_out) == (expected["height"], expected["width"], expected["tokens"]))
        values = torch.arange(PAGE_TOKENS * 3, dtype=torch.float32).view(PAGE_TOKENS, 3)
        output = apply_plan(values, plan)
        record(f"length-{ratio}", output.shape == (plan.page_out, 3))
        record(f"base-{ratio}", torch.equal(output[:BASE], values[:BASE]))
        expected = values[plan.groups[BASE]].mean(dim=0)
        record(f"pool-mean-{ratio}", torch.allclose(output[BASE], expected))
        record(f"newline-count-{ratio}", plan.page_out == BASE + plan.height * (plan.width + 1))
        again = apply_plan(values, plan)
        record(f"replay-{ratio}", torch.equal(output, again))
        record(f"hash-{ratio}", plan.rule_hash() == make_plan("BasePreservingAdaptivePool", ratio, "1025", 0, 0, 0, SEED).rule_hash())

    groups = []
    for offset_id in (0, 1, 2):
        plan = make_plan("BasePreservingJitteredPool", 0.25, "1025", 0, offset_id, 0, SEED)
        groups.append(plan.groups)
        replay = make_plan("BasePreservingJitteredPool", 0.25, "1025", 0, offset_id, 0, SEED)
        record(f"jitter-replay-{offset_id}", plan.groups == replay.groups and plan.rule_hash() == replay.rule_hash())
    record("jitter-offset-differs", groups[0] != groups[1] or groups[0] != groups[2])
    page_shifts = [
        make_plan("BasePreservingJitteredPool", 0.25, "1025", 0, 0, page, SEED).groups
        for page in (0, 1, 2)
    ]
    record("page-offsets-differ", len({json.dumps(item) for item in page_shifts}) > 1)

    left = torch.arange(PAGE_TOKENS, dtype=torch.float32).view(PAGE_TOKENS, 1)
    right = left + 10000
    plan_a = make_plan("BasePreservingAdaptivePool", 0.50, "1025", 0, 0, 0, SEED)
    plan_b = make_plan("BasePreservingAdaptivePool", 0.50, "1025", 0, 0, 1, SEED)
    pooled_left = apply_plan(left, plan_a)
    pooled_right = apply_plan(right, plan_b)
    record("no-cross-page", float(pooled_left.max()) < 10000 and float(pooled_right.min()) >= 10000)

    model = Tiny()
    image_id = 1
    text = [2, 3]
    ids = text[:1] + [image_id] * PAGE_TOKENS + text[1:] + [image_id] * PAGE_TOKENS + [4]
    input_ids = torch.tensor(ids).view(1, -1)
    spans = [(1, 1 + PAGE_TOKENS), (2 + PAGE_TOKENS, 2 + 2 * PAGE_TOKENS)]
    packed = pack_sequence(model, input_ids, spans, [pooled_left.expand(-1, 4), pooled_right.expand(-1, 4)])
    record("pack-length", packed.shape[1] == 3 + plan_a.page_out + plan_b.page_out)
    record("all", all(item["pass"] for item in checks))
    payload = {"status": "PASS", "not_an_official_compressor": True, "checks": checks}
    Path(__file__).with_name("surrogate_structure_tests.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"pass": True, "checks": len(checks)}))


if __name__ == "__main__":
    main()
