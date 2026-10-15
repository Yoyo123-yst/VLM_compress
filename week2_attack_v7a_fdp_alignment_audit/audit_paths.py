"""Compare generate, cache teacher-forcing, and manual full forward.

Does not run PGD and does not write any historical directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v7a_fdp_alignment_audit"
V7 = ROOT / "week2_attack_v7_pristine60"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week1_mechanism"))
sys.path.insert(0, str(ROOT / "gate_experiment"))

from run_causal_v2 import decode, prepare  # noqa: E402
from week2_attack_v4_answer_aligned.forward import full_answer_logits  # noqa: E402
from week2_attack_v4_answer_aligned.run_smoke import Session  # noqa: E402
from week2_attack_v7_budget_conditioned.constants import EPSILON_MARGIN  # noqa: E402
from week2_attack_v7_budget_conditioned.fdp import hard_margin  # noqa: E402
from week2_attack_v7_pristine60.load_samples import load_items, manifest  # noqa: E402
from week2_attack_v7_pristine60.run_v7 import load_model_frozen  # noqa: E402
from week2_pilot.attack import page_to_float  # noqa: E402

PLAN = {
    "60744": ["clean", "B", "random_noise"],
    "22317": ["clean", "B", "random_noise"],
    "5615": ["clean"],
    "1023": ["clean"],
    "51140": ["random_noise"],
    "56084": ["clean"],
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_png(sample_id: str) -> Path:
    hits = list((V7 / "controls" / "single").glob(f"shard_s*/{sample_id}/noise.png"))
    if len(hits) != 1:
        raise RuntimeError(f"noise png count for {sample_id} is {len(hits)}")
    return hits[0]


def hard_from_row(row: torch.Tensor, token_id: int) -> float:
    rival = row.clone()
    rival[token_id] = float("-inf")
    return float(row[token_id] - rival.max())


def cache_logits(model, inputs: dict, prefix: list[int]) -> list[torch.Tensor]:
    """Logits that predict prefix[t] after the prompt plus prefix[:t]."""
    if not prefix:
        raise ValueError("prefix is empty")
    first = model(**inputs, use_cache=True)
    rows = [first.logits[0, -1].float()]
    past = first.past_key_values
    attention = inputs.get("attention_mask")
    for token in prefix[:-1]:
        token_ids = torch.tensor([[token]], device=rows[0].device)
        if attention is not None:
            attention = torch.cat([attention, torch.ones_like(attention[:, :1])], dim=1)
        step = model(
            input_ids=token_ids,
            attention_mask=attention,
            past_key_values=past,
            use_cache=True,
        )
        rows.append(step.logits[0, -1].float())
        past = step.past_key_values
    return rows


def generate_once(model, inputs: dict) -> dict:
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=16,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    tokens = [int(x) for x in output.sequences[0, prompt_len:].detach().cpu().tolist()]
    scores = [row[0].float().cpu() for row in output.scores]
    if len(scores) != len(tokens):
        raise RuntimeError(f"scores {len(scores)} != tokens {len(tokens)}")
    chosen = [int(row.argmax()) for row in scores]
    return {"tokens": tokens, "scores": scores, "chosen": chosen, "prompt_len": prompt_len}


def summarize_inputs(inputs: dict) -> dict:
    summary = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def audit_condition(model, processor, item, image: Image.Image, clean_tokens: list[int], label: str) -> dict:
    images = list(item["images"])
    images[item["aux_idx"]] = image
    inputs = prepare(processor, images, item["meta"]["question"])
    eos = int(model.generation_config.eos_token_id)
    first = generate_once(model, inputs)
    second = generate_once(model, inputs)
    tokens = first["tokens"]
    termination = "MODEL_EOS" if tokens and tokens[-1] == eos else "MAX_LENGTH_TERMINATION"
    if termination == "MAX_LENGTH_TERMINATION" and len(tokens) != 16:
        termination = f"UNEXPECTED_LENGTH_{len(tokens)}"
    teacher = cache_logits(model, inputs, clean_tokens)
    session = Session(model, processor, item, torch.tensor(clean_tokens, dtype=torch.long))
    page = page_to_float(image).to(session.clean_page.device)
    with torch.no_grad():
        manual_logits = full_answer_logits(
            model, session.embeds_from(page), session.prompt_len - 1, len(clean_tokens)
        ).float().cpu()
    manual_tokens = [int(x) for x in manual_logits.argmax(dim=-1).tolist()]
    manual_hard = [hard_from_row(manual_logits[i], clean_tokens[i]) for i in range(len(clean_tokens))]
    teacher_tokens = [int(row.argmax()) for row in teacher]
    teacher_hard = [hard_from_row(teacher[i].cpu(), clean_tokens[i]) for i in range(len(clean_tokens))]
    steps = []
    comparable = min(len(tokens), len(clean_tokens))
    prefix_holds = True
    for index in range(len(clean_tokens)):
        shared = prefix_holds and index < comparable
        g_token = tokens[index] if index < len(tokens) else None
        if shared and g_token != clean_tokens[index]:
            prefix_holds = False
            shared = True
        g_row = first["scores"][index] if index < len(first["scores"]) else None
        t_row = teacher[index].cpu()
        m_row = manual_logits[index]
        steps.append({
            "index": index,
            "clean_token": clean_tokens[index],
            "g_token": g_token,
            "g_argmax": int(g_row.argmax()) if g_row is not None else None,
            "t_argmax": teacher_tokens[index],
            "m_argmax": manual_tokens[index],
            "shared_prefix": shared,
            "g_equals_t": g_token == teacher_tokens[index] if shared else None,
            "t_equals_m": teacher_tokens[index] == manual_tokens[index],
            "g_equals_clean": g_token == clean_tokens[index] if g_token is not None else False,
            "t_hard_margin": teacher_hard[index],
            "m_hard_margin": manual_hard[index],
            "g_hard_margin_on_clean": hard_from_row(g_row, clean_tokens[index]) if g_row is not None and shared else None,
            "max_abs_g_t": float((g_row - t_row).abs().max()) if g_row is not None and shared else None,
            "max_abs_t_m": float((t_row - m_row).abs().max()),
            "max_abs_g_m": float((g_row - m_row).abs().max()) if g_row is not None and shared else None,
        })
    text = decode(processor, torch.tensor(tokens))
    clean_text = decode(processor, torch.tensor(clean_tokens))
    cone = all(value > EPSILON_MARGIN for value in teacher_hard)
    return {
        "label": label,
        "token_ids": tokens,
        "repeat_token_ids": second["tokens"],
        "repeat_stable": tokens == second["tokens"],
        "clean_token_ids": clean_tokens,
        "decoded_text": text,
        "clean_decoded_text": clean_text,
        "token_exact": tokens == clean_tokens,
        "text_exact": text == clean_text,
        "termination": termination,
        "eos_token_id": eos,
        "cone_holds_on_T": cone,
        "teacher_argmax": teacher_tokens,
        "manual_argmax": manual_tokens,
        "steps": steps,
        "inputs": summarize_inputs(inputs),
        "use_cache": True,
        "generation_config": model.generation_config.to_dict(),
        "min_t_hard_margin": min(teacher_hard),
        "min_m_hard_margin": min(manual_hard),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    specs = {row["sample_id"]: row for row in manifest()["samples"]}
    wanted = [sid for sid in PLAN if not args.only or sid == args.only]
    items = {item["meta"]["sample_id"]: item for item in load_items([specs[sid] for sid in wanted])}
    selected = {
        row["sample_id"]: row
        for row in __import__("csv").DictReader((V7 / "attacks" / "selector" / "selected.csv").open())
    }
    model, processor = load_model_frozen()
    out_dir = HERE / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for sid in wanted:
        item = items[sid]
        clean_image = item["images"][item["aux_idx"]]
        clean_inputs = prepare(processor, item["images"], item["meta"]["question"])
        clean_gen = generate_once(model, clean_inputs)
        clean_tokens = clean_gen["tokens"]
        images = {"clean": clean_image}
        if "B" in PLAN[sid]:
            images["B"] = Image.open(ROOT / selected[sid]["artifact"] / selected[sid]["png"]).convert("RGB")
        if "random_noise" in PLAN[sid]:
            images["random_noise"] = Image.open(noise_png(sid)).convert("RGB")
        for label in PLAN[sid]:
            print(json.dumps({"start": sid, "image": label}), flush=True)
            report = audit_condition(model, processor, item, images[label], clean_tokens, label)
            report["sample_id"] = sid
            if label == "clean":
                report["clean_generate_matches_reference"] = report["token_ids"] == clean_tokens
            path = out_dir / f"{sid}_{label}.json"
            path.write_text(json.dumps(report) + "\n")
            reports.append(report)
            print(json.dumps({
                "id": sid,
                "image": label,
                "termination": report["termination"],
                "token_exact": report["token_exact"],
                "text_exact": report["text_exact"],
                "repeat_stable": report["repeat_stable"],
                "g_t": all(step["g_equals_t"] is not False for step in report["steps"] if step["shared_prefix"]),
                "t_m": all(step["t_equals_m"] for step in report["steps"]),
                "min_t": report["min_t_hard_margin"],
            }), flush=True)
    (HERE / "alignment_index.json").write_text(json.dumps({"n": len(reports)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
