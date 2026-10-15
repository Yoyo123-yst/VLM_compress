# r=0.15 source inventory

Both counts use the pristine60 manifest (`manifest_sha256` `62043dbdfb69c2de82fd9a149e8b260496c7f6571ec930bf7dcf6b9c4a86f327`) and the prospective selector in `week2_attack_v7_pristine60/attacks/selector/selected.csv`.

## 31/60

- Report: `week2_attack_v7_budget_conditioned/stage1_report.md` and `final_report.md`.
- Labels: `stage1/shard_s*/<id>.json`, field `images.B.by_r["0.15"].strict`.
- Writer: `stage1_diagnose.py`. Counter: `stage1_analyze.py`.
- Generate: `run_plain` / `run_fastv`, greedy, `max_new_tokens=16`, `use_cache=True`.
- Full exact: B full text equals the clean full text from that same stage1 call.
- FastV error: ANLS of the r=0.15 FastV text is below 0.5.
- Invalid: the full string is empty after stripping.
- Replay: the selected PNG is opened once. The strict bit does not compare a second reload.
- The strict bit also requires the clean page's own r=0.15 FastV ANLS to be at least 0.5.

## 45/60

- Report: `week2_attack_v7c_native60_readonly_remeasurement/statistical_summary.md`.
- Texts: `shards/shard_*/conditions.jsonl` at `image_kind=B`, `r=0.15`.
- Labels: `per_sample_native_summary.csv` column `strict_B_0.15`, written by `merge.py`.
- Generate: the same greedy call, plus the admitted native-cache replay. Alignment passed 600/600.
- Full exact: B full text equals `clean_full_pred` in the frozen manifest.
- FastV error: ANLS below 0.5.
- Invalid: either decoded string is empty.
- Replay: the published r=0.25 `replay_ok` bit. v7c does not replay again at r=0.15.
- The clean FastV gate uses the historical manifest ANLS, not a new clean generate at r=0.15.

Script hashes are in `source_inventory.json`. Neither tree was modified by this audit.
