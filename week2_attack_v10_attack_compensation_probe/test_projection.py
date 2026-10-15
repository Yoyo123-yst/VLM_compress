"""CPU tests for the attack/compensation intersection projection."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from projection import apply_uint8, delta_norms, project_pair_int  # noqa: E402


def check(name: str, cond: bool, detail: str) -> dict:
    return {"name": name, "pass": bool(cond), "detail": detail}


def main() -> None:
    rows = []
    clean = np.array([[[128, 128, 128]]], dtype=np.uint8)
    attack, comp = project_pair_int(np.array([[[20, -20, 3]]]), np.array([[[0, 0, 0]]]), clean, False)
    rows.append(check("positive_negative_bounds", int(attack.max()) == 8 and int(attack.min()) == -8, f"{attack.tolist()}"))

    saturated = np.array([[[0, 255, 250]]], dtype=np.uint8)
    attack, comp = project_pair_int(np.array([[[-8, 8, 8]]]), np.array([[[0, 0, 0]]]), saturated, False)
    image = apply_uint8(saturated, attack)
    got = np.asarray(image)
    rows.append(check("saturated_pixels", got[0, 0, 0] == 0 and got[0, 0, 1] == 255 and got[0, 0, 2] == 255, got.tolist()))

    attack, comp = project_pair_int(np.array([[[8]]]), np.array([[[8]]]), np.array([[[128]]], dtype=np.uint8), False)
    rows.append(check(
        "components_legal_sum_illegal",
        abs(int(attack[0, 0, 0])) <= 8 and abs(int(comp[0, 0, 0])) <= 8 and abs(int(attack[0, 0, 0] + comp[0, 0, 0])) <= 8,
        f"attack={int(attack[0,0,0])} comp={int(comp[0,0,0])}",
    ))
    rows.append(check("euclidean_pair", int(attack[0, 0, 0]) == 4 and int(comp[0, 0, 0]) == 4, f"{int(attack[0,0,0])},{int(comp[0,0,0])}"))

    clean = np.array([[[10]]], dtype=np.uint8)
    attack, comp = project_pair_int(np.array([[[8.6]]]), np.array([[[-1.6]]]), clean, False)
    rows.append(check("float_to_uint8", int(attack[0, 0, 0]) == 8 and int(comp[0, 0, 0]) == -2, f"{int(attack[0,0,0])},{int(comp[0,0,0])}"))

    clean = np.full((2, 2, 3), 100, dtype=np.uint8)
    attack, comp = project_pair_int(np.full((2, 2, 3), 6), np.full((2, 2, 3), 6), clean, False)
    page = apply_uint8(clean, attack + comp)
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    reloaded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
    norms = delta_norms(clean, reloaded.astype(np.int16) - clean.astype(np.int16))
    rows.append(check("png_reload_linf", norms["linf_uint8"] <= 8, str(norms)))
    frozen, moved = project_pair_int(np.array([[[8]]]), np.array([[[8]]]), np.array([[[128]]], dtype=np.uint8), True)
    rows.append(check("frozen_attack_slice", int(frozen[0, 0, 0]) == 8 and int(moved[0, 0, 0]) == 0, f"{int(frozen[0,0,0])},{int(moved[0,0,0])}"))
    payload = {"status": "PASS" if all(row["pass"] for row in rows) else "FAIL", "checks": rows}
    (HERE / "projection_tests.json").write_text(__import__("json").dumps(payload, indent=2) + "\n")
    if payload["status"] != "PASS":
        raise SystemExit(payload)
    print(payload["status"], len(rows))


if __name__ == "__main__":
    main()
