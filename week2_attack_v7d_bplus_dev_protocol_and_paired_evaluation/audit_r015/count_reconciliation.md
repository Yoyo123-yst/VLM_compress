# r=0.15 count reconciliation

The 31/60 and 45/60 figures are labels on the same 60 B images and the same Full and FastV strings. They are not two attack runs and they must not be substituted for each other.

## What is the same

- Sample list: the pristine60 manifest, 60 ids, sorted as integers.
- B PNG: file SHA-256 matches `selected.csv`, stage1 `png.B`, and the v7c condition record for every id.
- Selector: pristine60 prospective selector. Stage1 and v7c both read that checkpoint. They do not reselect. The selected step matches on every id.
- Full text and r=0.15 FastV text match on all 60 ids.

## What the two counts mean

31/60 is `week2_attack_v7_budget_conditioned/stage1_report.md`. A row is strict only if the B image is inside the uint8 budget and its r=0.15 FastV answer is wrong, and the clean page generated in that same run is still right at r=0.15 (`clean anls_fastv >= 0.5`).

45/60 is `week2_attack_v7c_native60_readonly_remeasurement/statistical_summary.md`. A row is strict if the B image's new r=0.15 FastV answer is wrong and the historical clean FastV ANLS stored in the manifest is at least 0.5. That historical clean score is not a clean generate at r=0.15.

## The 14 disagreements

All 14 are old strict 0 and v7c strict 1. In every one, stage1's clean FastV ANLS at r=0.15 is below 0.5, while the stored texts match. 31 + 14 = 45.

Ids: 1023, 2288, 7937, 8122, 18736, 25716, 41828, 44779, 49432, 50803, 51298, 56403, 58827, 62650.

Reason code on those rows: `DIFFERENT_STRICT_DEFINITION`. The other 46 rows agree, so their reason cell is empty.

No row is `UNEXPLAINED`. No row has the same strict definition and a different label.
