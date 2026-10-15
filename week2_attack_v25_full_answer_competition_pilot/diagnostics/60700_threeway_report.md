# 60700 three-way comparison

Target: `Dr. H. S. SORBS` (frozen before attack optimization from the
clean-only candidate manifest). Clean answer: `PPBS`. Ratio: 0.25. Each run
used ten accepted updates and at most twenty attempts.

| method | wall seconds | first target/semantic failure | final FastV text | guard status |
|---|---:|---:|---|---|
| Native-FDP | 120.5 | none | `PPBS #6` | all 10 accepted updates passed |
| complete-target | 151.5 | step 10 | `Dr. H. S. SORBS` | all 10 accepted updates passed |
| complete-target + first-divergence | 176.0 | step 5 | `Dr. H. S. SORBS` | all 10 accepted updates passed |

The Native-FDP endpoint changes formatting/suffixes but does not replace the
core answer. Both targeted methods eventually produce the frozen complete
wrong entity. The first-divergence constraint reaches it five accepted steps
earlier than complete-target, at an additional wall-clock cost of about 16%.

Every accepted update passed Full text exact, Full token exact, Full replay,
FastV replay, and uint8 L-infinity <= 8 in the training guard. Independent
PNG replay was also run for all three trajectories.

This is a positive discriminator witness for 60700, but it is still one
additional sample. It does not establish general superiority or justify unseen
evaluation. 39204 remains blocked because no valid complete wrong target was
found.
