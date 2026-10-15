# Change log

- 2026-10-08: Froze the v6c protocol before training. Parent v5b, v6, and v6b files were hashed and not written.
- lambda_soft is the published v6b quotient. beta is still pending the single auxiliary probe.
- 2026-10-08: One probe froze beta at 19.954104837532157. Health on 1025, 6194, and 7006 passed. B and C smokes passed. Dev12 selected B (strict 11/12); C was not selected (strict 10/12, pollution 2).
- 2026-10-09: The existing 40-sample confirmation of B finished at strict 38/40, with Full exact, replay, nonzero, and L∞ all 40/40, and a McNemar p of 5.82e-11 against the read-only v5b noise file. pristine60 was not started.
