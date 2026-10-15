# v9a Full baseline recovery

Decision: `ORIGINAL_OK_V9_WRAPPER_DIVERGED`

Sample 1025, frozen Full text `Dr. C. Baur MD`. The historical Full call reproduces that string. The v9 wrapper still emits one EOS token, `151645`, and decodes to empty text. Surrogate runtime admission stays blocked.

## What failed

This is a wrapper bug. The checkpoint, processor, tokenizer, page images, pixel values, and generation settings are the same on both paths.

The historical prompt is `build_prompt` in `week1_mechanism/run_eval.py`, used by `gate_experiment/run_causal_v2.py:prepare` and `week2_attack_v5b_expanded_pristine40/run_clean_screen.py`. It wraps the question as:

`<|im_start|>user\n` + body + `<|im_end|>\n<|im_start|>assistant\n`

`week2_attack_v9_target_independent_surrogate_runtime/run_admission.py:prompt_of` returns only the body. That is the earliest difference.

## Evidence

| field | original | v9 |
| --- | --- | --- |
| prompt SHA256 | differs | differs |
| input_ids | `(1, 11125)` | `(1, 11117)` |
| pixel_values SHA256 | equal | equal |
| page SHA256 | equal | equal |
| image tokens | 11097 | 11097 |
| `Dr` rank / logit | 0 / highest | 66272 / 0.122 |
| EOS rank / logit | 146683 | 0 / 28.875 |
| greedy text | `Dr. C. Baur MD` | empty |
| greedy ids | `8847, 13, 356, 13, 425, 4110, 13979, 151645` | `151645` |

Step 0 logits already differ, so no later-token trace was run. Both full greedy sequences are in `generation_trace.json`.

The original modules were not imported. `run_eval.py` imports the FastV adapter, and this audit is not allowed to load it. `audit_one.py` copies the Full prompt and the `generate(do_sample=False, max_new_tokens=16, use_cache=True)` call. No FastV module was loaded. No pixel update was run.

## Fix

Put the historical chat wrapper back into the v9 prompt. Leave EOS, `max_new_tokens`, sampling, and the image processor unchanged. That change was not applied here, because this directory is the audit record and the v9 wrapper still fails.

## Scope

Historical B and Native-FDP were not re-audited. No retrospective conclusion.

Surrogate runtime admission is not allowed.
