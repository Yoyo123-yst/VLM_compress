# Compressor-oblivious gray-box threat model

This file freezes the threat model. It does not authorize training.

The attacker knows the Full LLaVA-OneVision model, the processor, the question, the page order, the designated auxiliary page, the perturbation budget, and the Full native-cache decision-preservation term Native-FDP.

The attacker does not know, and the training process must not read, the target FastV source or hook, the target selector, the target ratio `r`, `K`, the keep-set, any compressed forward, compressed logits, compressed gradients, or target FastV text.

Training is legal only if `target_fastv_access_count == 0` and `target_fastv_modules_loaded == []`.

Real FastV is allowed only after surrogate optimization has finished, the PNGs and their hashes are sealed, and the training process has exited. A fresh evaluator process may then call FastV once. This admission stage does not run that evaluation.

The success endpoint stays the generate endpoint: nonzero uint8 perturbation, uint8 L∞ at most 8/255, Full text identical to the clean Full text, disk PNG reload matching the stored generate, and FastV ANLS below 0.5. A margin or a surrogate loss is not a strict success.

Native-FDP on the existing 12-question development set, 10/12 canonical strict, is a white-box upper bound. It is not a gray-box result and not an unseen-cohort result. B_PLUS remains the recorded ablation in which native Delta rose while real strict fell.
