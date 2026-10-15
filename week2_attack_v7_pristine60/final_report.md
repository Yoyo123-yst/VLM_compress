# pristine60 final report

## 1. Decision

PASS_PRISTINE60_SOURCE_BOUNDED_COMPRESSION_ONLY

## 2. Primary endpoint

Prospective selector strict CSFR is 52/60. Wilson 95% CI is [0.758346, 0.930860]. Full exact is 60/60 and Full text changed is 0/60. Replay is 60/60, nonzero is 60/60, uint8 L∞ legal is 60/60, and invalid is 0/60. Selected steps: {75: 1, 100: 4, 150: 7, 200: 48}.

M=2 strict is 14/17 (Wilson [0.5897, 0.9381]). M=3 is 11/11 (Wilson [0.7412, 1.0000]). M=4 is 27/32 (Wilson [0.6825, 0.9314]).

## 3. Random controls

Single random noise strict is 1/60. Pairing: {'attack_only': 51, 'control_only': 0, 'both_strict': 1, 'both_nonstrict': 8}. Exact McNemar p is 8.881784197001252e-16.

Seven-candidate trajectory strict is 1/60. Pairing: {'attack_only': 51, 'control_only': 0, 'both_strict': 1, 'both_nonstrict': 8}. Exact McNemar p is 8.881784197001252e-16. ID 56084 has no feasible random candidate. It stays a control failure on the frozen 60-sample denominator and was not redrawn.

## 4. Repeatability

Strict-label agreement is 8/10. Success-endpoint reproduction (exact image match or strict endpoint with Full exact and replay) is 7/10. Labels: {'EXACT_REPRODUCTION': 0, 'ENDPOINT_REPRODUCTION': 7, 'ENDPOINT_MISMATCH': 3, 'INVALID_OR_INCOMPLETE': 0}. PNG hash agreement is 0/10 and keep-hash agreement is 0/10. A different selected step is not scored as a failure by itself.

## 5. Mechanism

CC reproduces clean FastV on 60/60 and AA reproduces the selected FastV text on 60/60. Among the 52 primary strict successes: CONTENT_SUFFICIENT 40/52 (Wilson [0.6387, 0.8628]), SELECTION_SUFFICIENT 0/52, BOTH_INDEPENDENTLY_SUFFICIENT 0/52, INTERACTION_REQUIRED 12/52 (Wilson [0.1372, 0.3613]), UNEXPLAINED_OR_INVALID 0/52, COUNTERFACTUAL_INVALID 0/52. These proportions use only the strict-success denominator. Non-strict samples are diagnosed separately: {'CONTENT_SUFFICIENT': 2, 'UNEXPLAINED_OR_INVALID': 6}.

Jaccard {'n': 60, 'mean': 0.833519091507378, 'median': 0.8437385754198377, 'p25': 0.817835774245401, 'p75': 0.8671211547725552}. Symmetric difference {'n': 60, 'mean': 520.4, 'median': 509.0, 'p25': 433.5, 'p75': 601.5}. Evidence-share change {'n': 60, 'mean': -0.002483606604831325, 'median': -0.0003604902667627974, 'p25': -0.003333658099339408, 'p75': 0.0033117058664503887}. Auxiliary-share change {'n': 60, 'mean': 0.002836516408068027, 'median': 0.00040551500405515417, 'p25': -0.00756978395727681, 'p75': 0.00601513922681806}.

## 6. Error quality

Among primary strict successes the auditable text classes are {'CORE_ANSWER_CHANGED': 32, 'TRUNCATION_OR_EOS': 5, 'UNRELATED_ANSWER_REPLACEMENT': 13, 'EXTRA_TEXT_POLLUTION': 2}. The label for each sample was assigned by reading clean FastV, adversarial FastV, and gold, and the reason is stored beside the three strings.

## 7. Protocol integrity

raw_immutability_check result is PASS. FROZEN.json sha256 is 884aa0117847b2b24373a8d101a545664b048119f51b4148b9e870e1027c5e15. Manifest canonical sha256 matches the frozen record: True. Code files changed: none. Parent files changed: none.

## 8. Allowed claim / prohibited claim

Allowed claim: source-bounded FastV r=0.25 compression-only attack on this frozen 60-sample draw. content-dominant / survivor-poisoning evidence is allowed for this FastV run.

Prohibited from these artifacts: selection-stable; all compressors or models; a universal visual-token compression vulnerability. The development 40-sample rate is not combined with this 60-sample rate.

## 9. Next step

Keep this 60-sample draw, the selected checkpoints, and the no-feasible trajectory case frozen. A later study can test another compressor or another model on a new unseen draw; that result is not implied by this file.
