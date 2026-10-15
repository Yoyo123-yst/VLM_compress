# Native60 readonly remeasure

This directory remeasures the existing pristine60 Variant B images and their matched random-noise images with the admitted native-cache margin. It does not rerun the attack, change pixels, or replace the historical 52/60 strict generate result.

`FROZEN_PROTOCOL.json` fixes the 60 samples, PNG hashes, budgets, and the Delta definition. Four shards read those images and compare native teacher-forcing with real `generate`. `merge.py` computes the preregistered summaries only after all 600 paths align.
