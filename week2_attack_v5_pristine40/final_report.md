# Pristine40 validation — final report

**Decision:** `BLOCKED_INSUFFICIENT_PRISTINE_POOL`  
**Stage:** Frozen answer-aligned attack (λ_full=0.25, ε=8/255, 200 steps) — **no attack runs on unseen 40** (pool empty under protocol).

## 1. Historical exclusion

- Scanned: `week2_pilot/`, `week2_attack_v2/`, `week2_attack_v3/`, `week2_attack_v3_quick/`, `week2_attack_v4_answer_aligned/`, `week2_attack_v4_relaxed_full/`.
- Derived analysis exports (`selector_audit/`, `all_samples.csv`, `all_checkpoints.csv`, holdout `paired.csv`, diagnostics) are not used as primary ID sources (IDs already appear in run artifacts / checkpoints).
- **Union:** 280 unique `sample_id`s; SHA-256 `84095f32aafc1f76d93b2b11f7c418c4c055650afc9feb286dc2c8c6e97d1e29`.
- **v4 dev six:** covered.
- **relaxed_full holdout 20:** 20/20 covered.
- Artifacts: `exclusions/historical_attack_ids.json`, `exclusion_sources.md`, `exclusion_audit.json`.

## 2. Pristine candidate pool

| Metric | Value |
|--------|------:|
| Clean eligible (M>1, in draw 400 seed 42, Full & FastV r=0.25 ANLS≥0.5) | 106 |
| Overlap with full exclusion union | 106 |
| **Pristine eligible (protocol)** | **0** |
| Diagnostic: eligible if only non-oracle adversarial runs counted | 52 |

Sampling seed `2026100801` was **not** used (blocked before freeze).

**Why zero?** Every ID in the 106-sample clean-eligible slice of the 400-draw already appears in the exclusion union. For 52 IDs the only week2 touch is `week2_pilot/results/oracle_hijack_r050.csv` (inspection / non-baseline manipulation); protocol reason `historically_attacked_or_inspected` still excludes them. The remaining 54 IDs have PGD/smoke/holdout/v2/v3 attack provenance.

This is stricter than relaxed_full holdout freezing (which excluded only the v4 dev six, yielding 100 eligible → 20 holdout attacks). Pristine40 correctly removes **all** prior attack and inspection IDs; the draw-400∩baseline slice has no survivors.

## 3–8. Not run

- `frozen/pristine40_manifest.json` — **not created**
- Clean gate — **NOT_RUN**
- 40-sample attack (fixed final / min-margin) — **NOT_RUN**
- Random noise arm — **NOT_RUN**
- Disk replay — **NOT_RUN**

## 9. Engineering smoke (dev 50743 only)

| Run | GPU | Steps | Wall clock | Notes |
|-----|-----|------:|-----------|--------|
| `tests/smoke_2/` | **6** | 2 | ~33 s | step 2: Full/FastV ANLS 1.0, NO_EFFECT |
| `tests/smoke_25/` | **4** | 25 | ~200 s | step 10/25: SUCCESS_COMP_ONLY, Full 1.0 |

λ_full=0.25, v4 frozen scales; **not** part of pristine40 primary results.

## 10. Answers to pre-registered report questions

1. **Full historical exclusion?** Yes, automated scan + holdout manifest + v4 dev six; union hash above.
2. **Pristine pool size?** 0 (106 clean-eligible, all excluded).
3. **40 frozen?** No — blocked.
4. **fixed final ≥27/40?** N/A.
5. **Wilson LB >0.5?** N/A.
6. **Full errors 0?** N/A.
7. **Full text 40/40?** N/A.
8. **final.png replay 40/40?** N/A.
9. **min-margin frozen pre-run?** Defined in protocol; not executed.
10. **min-margin vs fixed?** N/A.
11. **Random noise strict failures?** N/A.
12. **Attack vs noise McNemar?** N/A.
13. **M=2 dominance?** N/A.
14. **Failures / anomalies?** Pool exhaustion under honest exclusion.
15. **Evidence level?** **No support / blocked** — cannot test generalization on unseen 40 within this draw∩baseline universe.

## 11. Limitations

- Expanding the candidate set beyond `draw_questions(400,42) ∩ clean_baseline` would be a **new** pre-registration (not done here).
- Using the 52 oracle-only IDs would violate the stated exclusion rule unless the protocol is amended to distinguish oracle inspection from PGD attack history.

## Paths

- Root: `week2_attack_v5_pristine40/`
- Decision: `final_decision.json`
- Pool audit: `pool/pool_audit.md`
