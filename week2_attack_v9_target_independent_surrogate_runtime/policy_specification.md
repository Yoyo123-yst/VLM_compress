# Surrogate policy specification

These three policies are local geometric rules. They are not official compressors, and they are not FastV.

The page layout is the frozen 768 packing: 3699 visual tokens. The first 729 are a 27 by 27 base grid in row-major order. The remaining tokens are 54 high-resolution rows. Each of those rows is 54 patches followed by one newline at local index `729 + row * 55 + 54`.

Every rule is applied independently on each page. Nothing in a rule reads pixel values, attention, the question, logits, or a target compressor.

## SpatialStridedDrop

This policy deletes tokens.

For a content grid with `n` cells and budget `r`, keep `k = max(1, round(n * r))` cells. The kept raster indices are `(j * n) // k` for `j = 0 .. k-1`. Raster order is row-major, so the index is a spatial coordinate. `r = 0.25` and `r = 0.35` are the only legal values. The same formula is used on the 27 by 27 base grid and the 54 by 54 high-resolution grid. A high-resolution newline is kept only when that row keeps at least one patch. Kept tokens stay in their original order.

## LocalSpatialMerge

This policy merges tokens. It does not use the drop index set.

`r = 0.25` uses groups of 4 consecutive columns. `r = 0.35` uses groups of 3. A short group at the end of a row is still one group. Each group becomes one token whose vector is the arithmetic mean of the source features. The recorded position is the first source index in the group. After every high-resolution row, that row's original newline is appended unchanged. Base rows have no newline.

## SeededBlockDrop

This policy is implemented only as a structured augmentation after the two policies above have entered `generate()` and reduced the token count. It is not an official compressor.

Seed `2026101109`. Content cells are tiled into 2 by 2 blocks from the top-left, with partial blocks on the odd edge. A block is kept when the first 8 hex digits of `sha256("{seed}|{page}|{grid}|{block_row}|{block_col}")`, read as an integer, are strictly below `round(r * 2^32)`. `grid` is `base` or `high`. Dropped blocks are removed. A high-resolution newline is kept only when its row keeps a patch. One seed is used for this smoke.

## Positions

The packed sequence passed to `generate()` contains text embeddings and the reduced visual embeddings, in the original page order. `generate()` assigns the positions of that shorter sequence. Each output token records the source local index it came from. A merged token records the first index of its group. The attention mask is all ones on the packed sequence; causality is the mask `generate()` builds for that sequence.
