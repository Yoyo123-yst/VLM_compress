# Native-cache margin admission

This directory checks whether a teacher-forced Full and FastV margin can use the same `model.generate()` cache path as the production call. It does not rerun Variant B, does not recompute the existing 52/60 strict generate result, and does not issue a retrospective certificate for B.

`FROZEN_PROTOCOL.json` fixes the ten conditions, the decode call, and the parent-directory hashes. `native_cache_margin.py` reads logits from the production model forward. The old manual `full_answer_logits` path is not a certificate or a loss.

CPU tests live in `tests/test_native_margin_cpu.py`. The ten-condition GPU health lives in `tests/test_native_margin_gpu.py`.
