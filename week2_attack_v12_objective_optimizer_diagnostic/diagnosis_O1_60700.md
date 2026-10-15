# Diagnosis: O1 / 60700 smoke failure

The failure is localized to the OPGD branch, not to runtime replay or PNG
serialization.

At steps 0--2, `O1/60700` has `full_text_exact=1`, `full_replay=1`, and
`linf_ok=1`. At update 2, the Full proxy is marked violated:

```
attack_l2 = 21.4452
full_l2   = 80.5538
direction_l2 = 80.4316
branch = full_primary
```

The next checkpoint (step 3) has `full_text_exact=0` while replay and the
pixel bound still pass. The direction is therefore dominated by the Full
gradient, and the Full exact endpoint is lost.

The implementation confirms the cause. In `run_v11.py`, the violated branch
uses:

```python
project_out(full, attack)
```

where `project_out(source, onto)` removes the component of `source` along
`onto`. If the objective is “preserve Full while attacking FastV”, the attack
gradient should be projected away from the Full gradient:

```python
project_out(attack, full)
```

The current argument order instead projects the Full gradient away from the
attack gradient, producing a direction with Full-gradient scale. The
`full_primary` label is consistent with this behavior. This is a deterministic
algorithmic bug in the diagnostic implementation, not evidence that OPGD is
intrinsically unusable.

The smoke gate remains correctly blocked. The safe next experiment is a
two-sample, six-group rerun after correcting only this argument order, with a
post-update Full exact rollback/rejection check. Do not run the six-sample or
unseen cohorts before that amended smoke passes.
