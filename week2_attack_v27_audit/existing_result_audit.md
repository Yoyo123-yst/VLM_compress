# Existing result audit

This report separates final state, first content error, and exact target hit.

|sample|method|final|first content error|target hit|Full exact|seconds|
|---|---|---|---|---|---|---:|
|47567|native_fdp|2006 (CONTENT_FAILURE_NUMERIC)|2600|-|True|165.8|
|47567|complete_target|2600 (CONTENT_FAILURE_NUMERIC)|2600|-|True|183.0|
|47567|complete_front|900 (CONTENT_FAILURE_NUMERIC)|2600|900|True|135.7|
|50779|native_fdp|Dr. Donbaryx (AMBIGUOUS_PERSON_IDENTITY_OR_SPELLING_REVIEW)|Dr. Donbex|-|True|239.6|
|50779|complete_target|Dr. Donbary (AMBIGUOUS_PERSON_IDENTITY_OR_SPELLING_REVIEW)|-|-|True|220.1|
|50779|complete_front|Dr. Dorothy Darby (AMBIGUOUS_PERSON_IDENTITY_OR_SPELLING_REVIEW)|Dr. Dorothy|Dr. Dorothy Darby|True|172.8|
|51133|native_fdp|26 (CONTENT_FAILURE_NUMERIC)|30|-|True|144.8|
|51133|complete_target|30 (CONTENT_FAILURE_NUMERIC)|30|-|True|185.6|
|51133|complete_front|30 (CONTENT_FAILURE_NUMERIC)|30|-|True|140.2|
|6194|native_fdp|01017-X00-XX-XX (AMBIGUOUS_IDENTIFIER_FORMAT_REVIEW)|-|-|True|380.5|
|6194|complete_target|01017-X0X-XX (AMBIGUOUS_IDENTIFIER_FORMAT_REVIEW)|-|-|True|383.4|
|6194|complete_front|0107-XX-XX (ANSWER_PRESERVED)|-|-|True|310.7|
|62941|native_fdp|October 13, 1979 (CONTENT_FAILURE_DATE)|October 13, 1979|-|True|298.0|
|62941|complete_target|October 13, 1979 (CONTENT_FAILURE_DATE)|October 13, 1979|-|True|303.7|
|62941|complete_front|October 13, 1979 (CONTENT_FAILURE_DATE)|October 13, 1979|-|True|264.6|
|64506|native_fdp|E (ANSWER_PRESERVED)|-|-|True|286.2|
|64506|complete_target|J (CONTENT_FAILURE_LETTER)|J|-|True|219.7|
|64506|complete_front|C (CONTENT_FAILURE_LETTER)|J|-|True|194.6|

## Interpretation

A target miss is not counted as an attack miss when the final answer is a reviewed content error. 50779 remains ambiguous because adding a first name may or may not identify a different person. 64506's frozen target is invalid for a letter question, so its target-hit metric is not a valid method metric.

Unseen samples were not opened.
