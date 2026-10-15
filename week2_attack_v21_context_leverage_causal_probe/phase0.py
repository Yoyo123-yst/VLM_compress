"""Hash v20 and re-verify the 47450 clean FastV partition. No optimization."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment")]
V20 = ROOT / "week2_attack_v20_fourway_content_ablation"
HERE = ROOT / "week2_attack_v21_context_leverage_causal_probe"

from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402
from week2_attack_v21_context_leverage_causal_probe.groups import RANDOM_SEEDS, draw_control, jaccard, partition  # noqa: E402


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v20_targets() -> list[Path]:
    targets = sorted(V20.glob("*.py"))
    targets += [
        V20 / "config.json",
        V20 / "semantic_manifest.json",
        V20 / "admission.json",
        V20 / "full_dev_fourway_v1" / "per_run" / "47450" / "C.json",
        V20 / "full_dev_fourway_v1" / "per_run" / "47450" / "D.json",
        V20 / "independent_eval" / "47450" / "C.json",
        V20 / "independent_eval" / "47450" / "D.json",
    ]
    return targets


def write_provenance(name: str) -> dict:
    rows = {str(path.relative_to(ROOT)): sha_file(path) for path in v20_targets()}
    payload = {"label": name, "files": rows}
    rt.write_json(HERE / "sealed" / f"v20_{name}.json", payload)
    return payload


def group_record(tokens: list[dict], positions: list[int]) -> dict:
    return {
        "positions": positions,
        "token_ids": [tokens[pos]["token_id"] for pos in positions],
        "pieces": [tokens[pos]["decoded_piece"] for pos in positions],
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    before = write_provenance("before")
    manifest = json.loads((V20 / "semantic_manifest.json").read_text())
    admission = json.loads((V20 / "admission.json").read_text())
    row = next(item for item in manifest["samples"] if item["sample_id"] == "47450")
    admitted = next(item for item in admission["samples"] if item["sample_id"] == "47450")
    rt.configure()
    model, processor = rt.v7d.load_model_frozen()
    item = rt.v7d.load_dev("47450")
    state = rt.prepare_clean_state(model, processor, item)
    pieces = [processor.tokenizer.decode([int(token)], skip_special_tokens=False) for token in state["clean_fast_tokens"]]
    tokens = []
    special = set(int(token) for token in (processor.tokenizer.all_special_ids or []))
    for index, (token_id, piece) in enumerate(zip(state["clean_fast_tokens"], pieces)):
        control = int(token_id) in special or piece.strip() == "" or (piece.startswith("<|") and piece.endswith("|>"))
        tokens.append({
            "position": index,
            "token_id": int(token_id),
            "decoded_piece": piece,
            "special_or_control_or_eos": control,
            "group": None,
        })
    problems = []
    if state["clean_text"] != row["clean_full_text"]:
        problems.append("full_text")
    if state["clean_tokens"] != row["clean_full_token_ids"]:
        problems.append("full_tokens")
    if state["clean_fast_text"] != row["clean_fastv_text"]:
        problems.append("fastv_text")
    if state["clean_fast_tokens"] != row["clean_fastv_token_ids"]:
        problems.append("fastv_tokens")
    if state["clean_fast_keep_hash"] != row["clean_fastv_keep_hash"] or state["clean_fast_kept"].__len__() != row["clean_fastv_k"]:
        problems.append("keep_set")
    if pieces != row["selection_b"]["pieces"]:
        problems.append("pieces")
    if item["aux_idx"] != admitted["provenance"]["aux_idx"] or item["evidence_idx"] != admitted["provenance"]["evidence_idx"]:
        problems.append("page_roles")
    if list(item["order"]) != list(admitted["provenance"]["order"]) or list(item["page_sha"]) != list(admitted["provenance"]["page_sha256"]):
        problems.append("page_order")
    grouped = partition(
        [{"position": token["position"], "piece": token["decoded_piece"]} for token in tokens],
        row["selection_d"]["positions"],
        row["selection_c"]["positions"],
    )
    if not grouped["ok"]:
        problems.extend(grouped["problems"])
    if grouped["s_core"] != row["selection_d"]["positions"] or grouped["s_all"] != row["selection_c"]["positions"]:
        problems.append("positions_differ_from_v20")
    for token in tokens:
        if token["position"] in grouped["s_core"]:
            token["group"] = "S_core"
        elif token["position"] in grouped["s_context"]:
            token["group"] = "S_context"
        elif token["special_or_control_or_eos"]:
            token["group"] = "control"
        else:
            token["group"] = "outside_frozen_sets"
    status = "PASS" if not problems else "BLOCKED_TOKEN_GROUP_REPLAY_MISMATCH"
    random_rows = []
    random_status = "PASS"
    if status == "PASS":
        for kind, forbidden in (("R_core", grouped["s_core"]), ("R_context", grouped["s_context"])):
            for seed in RANDOM_SEEDS:
                picked = draw_control(grouped["s_all"], len(forbidden), forbidden, seed)
                if picked is None:
                    random_status = "BLOCKED_RANDOM_POSITION_CONTROL_UNDERDETERMINED"
                    continue
                random_rows.append({
                    "name": f"{kind}_{seed}",
                    "kind": kind,
                    "seed": seed,
                    "positions": picked,
                    "token_ids": [tokens[pos]["token_id"] for pos in picked],
                    "pieces": [tokens[pos]["decoded_piece"] for pos in picked],
                    "jaccard_s_core": jaccard(set(picked), set(grouped["s_core"])),
                    "jaccard_s_context": jaccard(set(picked), set(grouped["s_context"])),
                })
    payload = {
        "status": status,
        "problems": problems,
        "sample_id": "47450",
        "clean_full_text": state["clean_text"],
        "clean_fastv_text": state["clean_fast_text"],
        "clean_fastv_k": len(state["clean_fast_kept"]),
        "clean_fastv_keep_hash": state["clean_fast_keep_hash"],
        "aux_idx": item["aux_idx"],
        "evidence_idx": item["evidence_idx"],
        "order": list(item["order"]),
        "tokens": tokens,
        "s_all": group_record(tokens, grouped["s_all"]),
        "s_core": group_record(tokens, grouped["s_core"]),
        "s_context": group_record(tokens, grouped["s_context"]),
        "self_patch": "SELF_PATCH_NOT_APPLICABLE_PIXEL_OBJECTIVE",
    }
    rt.write_json(HERE / "frozen_token_groups.json", payload)
    rt.write_json(HERE / "random_controls.json", {"status": random_status, "controls": random_rows})
    rt.write_json(HERE / "provenance.json", {"v20_before": before, "phase0_status": status, "random_status": random_status})
    print(json.dumps({"status": status, "problems": problems, "random": random_status, "context": payload["s_context"]["pieces"], "core": payload["s_core"]["pieces"]}, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
