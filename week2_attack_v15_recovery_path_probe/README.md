# Recovery-path probe

This directory contains the bounded diagnostic for the proposed
compression-conditioned recovery-path attack. It consumes only the frozen
Native-FDP PNG checkpoints and does not modify the historical runs.

The first gate compares, at each checkpoint, the content-token escape loss
under clean teacher-forced prefixes with the same loss under the naturally
generated FastV prefixes. It also records content overlap, keep-set hashes,
Full exactness, and the cosine between the two image gradients.

The probe is diagnostic only. No new attack is started unless the report
shows a prefix-state mismatch on the registered failure samples.
