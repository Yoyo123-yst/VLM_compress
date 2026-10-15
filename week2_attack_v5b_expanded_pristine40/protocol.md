# v5b expanded pristine40

v5 (`week2_attack_v5_pristine40`) stopped with `BLOCKED_INSUFFICIENT_PRISTINE_POOL` because the old `draw_questions(400, seed=42)` slice had zero IDs outside the historical exclusion union.

v5b expands the candidate universe to the same MP-DocVQA validation split already used by this project. It does not change the dataset, the exclusion definition, the attack, λ, steps, or ε. Oracle-only historical IDs stay excluded.

Seeds: SCREEN_SEED=2026100802, FINAL_SAMPLE_SEED=2026100803, NOISE_SEED=2026100804.
SCREEN_N=200, FINAL_N=40.
