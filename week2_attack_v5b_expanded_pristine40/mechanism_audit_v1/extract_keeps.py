"""Save clean and final FastV keep index lists for Jaccard. No optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
V5B = ROOT / "week2_attack_v5b_expanded_pristine40"
OUT = V5B / "mechanism_audit_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from adapters.runtime import clear_hooks  # noqa: E402
from run_causal_v2 import load_model  # noqa: E402
from week2_attack_v5b_expanded_pristine40.load_samples import load_formal_items, manifest  # noqa: E402
from week2_attack_v5b_expanded_pristine40.mechanism_audit_v1.run_counterfactuals import fastv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    specs = [row for i, row in enumerate(manifest()["samples"]) if i % args.nshards == args.shard]
    items = load_formal_items(specs)
    dest = OUT / "shards" / f"keeps_s{args.shard}.jsonl"
    done = set()
    if dest.exists():
        done = {json.loads(line)["sample_id"] for line in dest.read_text().splitlines() if line.strip()}
    model, processor = load_model()
    for param in model.parameters():
        param.requires_grad_(False)
    for item in items:
        sid = item["meta"]["sample_id"]
        if sid in done:
            continue
        final = Image.open(V5B / "attacks" / "artifacts" / sid / "final.png").convert("RGB")
        attacked = list(item["images"])
        attacked[item["aux_idx"]] = final
        _, clean_keep, _, _ = fastv(model, processor, item["images"], item["meta"]["question"])
        _, adv_keep, _, _ = fastv(model, processor, attacked, item["meta"]["question"])
        with dest.open("a") as handle:
            handle.write(json.dumps({"sample_id": sid, "clean": clean_keep, "adv": adv_keep}) + "\n")
        print(json.dumps({"keep": sid, "symdiff": len(set(clean_keep) ^ set(adv_keep))}), flush=True)
        clear_hooks(model)


if __name__ == "__main__":
    main()
