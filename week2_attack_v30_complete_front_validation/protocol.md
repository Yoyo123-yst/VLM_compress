# v30 complete-front validation protocol

This is a new six-sample development validation cohort, selected before target generation and before any attack outcome is inspected. It is not an unseen evaluation.

For each frozen sample, first repeat the clean Full/FastV replay in a fresh process and review whether the clean FastV output answers the question correctly. Generate complete wrong targets from the clean input under the existing v29 cap and freeze the first semantically valid target. Do not replace a target-unavailable sample.

On samples with a valid target, compare Native-FDP, complete-target, and complete-front with the frozen v29 implementation, PNG budget, Full exact guard, final-selector rule, and independent PNG evaluation. The primary endpoint is question-specific `CONTENT_FAILURE` with Full text/token exact and independent replay. Exact target hit, invalid output, target coverage, and actual cost are reported separately.

The decision rule is fixed: retain complete-front only if it exceeds complete-target and Native-FDP on the new cohort's semantic endpoint or reaches the same qualifying endpoints with lower recorded cost. If the gain does not reproduce, return to complete-target as the simpler directional baseline.
