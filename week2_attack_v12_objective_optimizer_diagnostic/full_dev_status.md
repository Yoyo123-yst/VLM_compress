# Full development run status

Status: `BLOCKED_DEV_OOM` (original implementation); sequential-ratio backward probe now passes.

The first full-development batch was started on samples 1025, 2008, 47450,
and 60700 for group H0. Sample 47450 failed at step 0 during the FastV
gradient path, before the post-update Full guard, with CUDA OOM on a 40-GiB
GPU. The traceback is in `full_dev2_47450_H0_gpu4.log` and ends in
`native_grad.py::_run_layers` while concatenating the transformer KV cache.
Peak allocation was approximately 39.48 GiB.

The other three jobs were stopped after this resource failure; they are not
reported as completed or failed attack trials. No six-sample aggregate and no
unseen run was started.

The post-update guard was optimized to perform only two Full generations and
explicitly release temporary inputs/caches. A one-step 47450 retry still OOMs
in the FastV backward path, so this is an intrinsic memory limit of the
current 40-GiB runtime for this sample/configuration, not merely guard-cache
retention.

Next action requires an explicit resource/protocol decision: use a compatible
larger-memory GPU/runtime, or define a documented lower-memory variant (for
example, a single-ratio diagnostic) and treat it as a different, non-comparable
condition. Do not mix that variant into the six-sample four-cell result.

After changing `attack_grads` to backpropagate the two training ratios one at a
time while accumulating into the same leaf gradient, a fresh one-step H0 run
on 47450 completed with Full exact/replay and peak memory 29,044 MiB. This is a
memory-preserving implementation of the same summed gradient, but it changes
the floating-point accumulation order; it therefore requires a new smoke gate
before restarting the official development run.
