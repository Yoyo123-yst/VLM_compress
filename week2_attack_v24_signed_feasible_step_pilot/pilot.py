"""Tiny signed, per-token Full-constrained step pilot.

This is a rescue diagnostic for v23.  It does not start multistep training or
checkpoint selection.  The step is chosen in the span of observed gradients,
with a signed objective that lowers the registered FastV core margins and
linearized constraints for every Full token.  The PNG-reloaded endpoint is
the only result that counts.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path("/home/yuesiting/autodl-tmp/multimodal_attack_project")
HERE = ROOT / "week2_attack_v24_signed_feasible_step_pilot"
sys.path[:0] = [str(ROOT), str(ROOT / "week1_mechanism"), str(ROOT / "gate_experiment"), str(ROOT / "week2_attack_v23_differential_subspace_pilot")]

from week2_attack_v23_differential_subspace_pilot import pilot as v23  # noqa: E402
from week2_attack_v23_differential_subspace_pilot.pilot import CONFIG as V23_CONFIG  # noqa: E402
from week2_attack_v23_differential_subspace_pilot.pilot import CORE, margin_tensor, score_image, sha_tensor, token_pieces  # noqa: E402
from week2_attack_v20_fourway_content_ablation import runtime_utils as rt  # noqa: E402

CONFIG = {
    "tag": "signed_feasible_step_pilot_v1",
    "samples": ["47450", "60700"],
    "ratio": 0.25,
    "radii_uint8": [1, 2],
    "full_target_margin": 0.10,
    "coefficient_norm": 1.0,
    "ridge": 1e-3,
    "seed": int(V23_CONFIG["seed"]),
}


def solve_signed(core_grads, full_grads, full_soft, radius, target, ridge, coeff_norm):
    """Solve a small signed QP in gradient span, then scale to pixel radius.

    The coefficient norm is a trust region.  Full constraints are imposed per
    token on the linearized soft margin; exact legality is checked after PNG
    reload by the caller.
    """
    cg = torch.stack([x.flatten() for x in core_grads])
    fg = torch.stack([x.flatten() for x in full_grads])
    basis = torch.cat([cg, fg], dim=0).T
    basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-12)
    B = basis.detach().float().cpu().numpy()
    c = (torch.stack(core_grads).mean(dim=0).flatten() @ basis).detach().float().cpu().numpy()
    F = (fg @ basis).detach().float().cpu().numpy()
    slack = np.asarray(full_soft, dtype=np.float64) - float(target)
    # The QP is only a local diagnostic; use the requested pixel radius when
    # converting its direction, while retaining the per-token sign constraints.
    scale = float(radius) / 255.0
    rhs = -slack / max(scale, 1e-12)
    n = B.shape[1]
    def fun(a):
        return float(c @ a + ridge * (a @ a))
    def jac(a):
        return c + 2.0 * ridge * a
    cons = [{"type": "ineq", "fun": lambda a, row=row, rr=rr: float(row @ a - rr),
             "jac": lambda a, row=row, rr=rr: row} for row, rr in zip(F, rhs)]
    cons.append({"type": "ineq", "fun": lambda a: float(coeff_norm**2 - a @ a),
                 "jac": lambda a: -2.0 * a})
    x0 = np.zeros(n, dtype=np.float64)
    result = minimize(fun, x0, jac=jac, constraints=cons, method="SLSQP",
                      options={"maxiter": 300, "ftol": 1e-10, "disp": False})
    if not result.success:
        return None, {"candidate_valid": False, "solver_success": False,
                      "solver_message": str(result.message), "basis_rank": int(n)}
    alpha = result.x
    raw = B @ alpha
    raw_max = float(np.max(np.abs(raw)))
    pixel_scale = min(scale / max(raw_max, 1e-12), 1.0)
    full_change = F @ alpha * pixel_scale
    for _ in range(40):
        if np.all(full_change >= rhs - 1e-9):
            break
        pixel_scale *= 0.5
        full_change = F @ alpha * pixel_scale
    delta_np = (raw * pixel_scale).reshape(core_grads[0].shape)
    delta = torch.from_numpy(delta_np).to(core_grads[0].device, dtype=core_grads[0].dtype)
    executed_linf = float(np.max(np.abs(delta_np)))
    candidate_valid = bool(np.all(full_change >= rhs - 1e-7) and executed_linf <= scale + 1e-8 and executed_linf > 0.0)
    return delta.detach(), {
        "candidate_valid": candidate_valid,
        "solver_success": True,
        "solver_message": str(result.message),
        "basis_rank": int(n),
        "objective": float(fun(alpha)),
        "coeff_norm": float(np.linalg.norm(alpha)),
        "pixel_scale": pixel_scale,
        "executed_linf": executed_linf,
        "linearized_core_change": float(c @ alpha * pixel_scale),
        "linearized_full_changes": [float(x * pixel_scale) for x in (F @ alpha)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=",".join(CONFIG["samples"]))
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rt.configure(); torch.manual_seed(CONFIG["seed"])
    model, processor = rt.v7d.load_model_frozen()
    rows = []
    for sample_id in [x.strip() for x in args.samples.split(",") if x.strip()]:
        item = rt.v7d.load_dev(sample_id)
        state = rt.prepare_clean_state(model, processor, item)
        original = item["images"][item["aux_idx"]].convert("RGB")
        reference = list(state["clean_fast_tokens"])
        positions = [int(x) for x in CORE[sample_id]]
        pieces = token_pieces(processor, reference)
        with torch.no_grad():
            clean_full = rt.v7d.full_rows(model, state["inputs"], state["inputs"]["pixel_values"], state["clean_tokens"])
            state["clean_soft"] = [float(x) for x in margin_tensor(clean_full, state["clean_tokens"]).cpu()]
        base = score_image(rt, model, processor, item, state, original, reference, pieces, positions, list(state["clean_fast_kept"]), {"forward":0,"backward":0,"generate":0})
        gs = v23.differentiable_state(model, processor, item, state, reference, positions, original)
        sample_out = out / "results" / sample_id; sample_out.mkdir(parents=True, exist_ok=True)
        for radius in CONFIG["radii_uint8"]:
            direction, meta = solve_signed(gs["core_grads"], gs["full_grads"], gs["full_soft"].tolist(), radius, CONFIG["full_target_margin"], CONFIG["ridge"], CONFIG["coefficient_norm"])
            if direction is None or not meta.get("candidate_valid", False):
                rows.append({"sample_id": sample_id, "direction": "QP_SIGNED_FULL_CONSTRAINED",
                             "radius": radius, "candidate_valid": False, "solver": meta})
                continue
            page = rt.v7d.page_to_float(original).to("cuda")
            # `direction` is already the signed descent step returned by the
            # QP.  Do not negate it again (the v23 sign-PGD convention used
            # the opposite representation).
            candidate = rt.project_image(page, direction)
            path = sample_out / f"QP_{radius:02d}.png"; png_sha = rt.save_png(candidate, path)
            reloaded = rt.reload_png(path)
            scored = score_image(rt, model, processor, item, state, reloaded, reference, pieces, positions, list(state["clean_fast_kept"]), {"forward":0,"backward":0,"generate":0})
            row = {"sample_id": sample_id, "direction": "QP_SIGNED_FULL_CONSTRAINED", "radius": radius,
                   "png_sha256": png_sha, "uint8_linf": rt.uint8_linf(original, reloaded),
                   "direction_hash": sha_tensor(direction), "solver": meta, **scored}
            row["full_exact"] = bool(row["full_text_exact"] and row["full_token_exact"] and row["full_replay"])
            rows.append(row)
            print(json.dumps({"sample":sample_id,"radius":radius,"full_exact":row["full_exact"],"fast_text":row["fast_text"],"core_margin":row.get("core_hard_mean"),"solver_success":meta["solver_success"]}, ensure_ascii=False), flush=True)
        rt.write_json(sample_out / "preflight.json", {"sample_id":sample_id,"base":base,"reference":reference,"core_positions":positions})
    rt.write_json(out / "results.json", {"config":CONFIG,"rows":rows})
    status = "COMPLETED_QP_PILOT" if len(rows) == len(CONFIG["samples"])*len(CONFIG["radii_uint8"]) else "INCOMPLETE"
    rt.write_json(out / "final_decision.json", {"status":status,"samples":CONFIG["samples"],"rows":len(rows),"multistep_started":False,"unseen_started":False})
    print(json.dumps({"status":status,"rows":len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
