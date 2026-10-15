# Target-independent surrogate runtime

Decision: `FAIL_FULL_BASELINE_REPLAY`.

Sample 1025 is the first id in the frozen v7d development list. Its frozen Full string is `Dr. C. Baur MD`. The current Full `generate()` returned one token, id 151645, which decodes to an empty string. The image-token count was 11097, equal to 3 times the frozen 3699-token page, so the page layout was recognized. The EOS logit was 28.875. The token for `Dr` ranked 66272. This is not a slicing error.

The protocol says to stop at the first Full mismatch. No surrogate `generate()`, disk replay, or gradient smoke was run. SpatialStridedDrop and LocalSpatialMerge exist as coordinate rules and were checked on a fake 3699-token page, where both reduce the token count and use different index sets. That CPU check is not admission. SeededBlockDrop was not run on the model.

Nothing here is an official compressor. No pixels were optimized. Target FastV was not called, and `target_fastv_access_count` is 0. There is no new attack success rate. The next 3–5 question transfer is not allowed. Parent directories were not modified.
