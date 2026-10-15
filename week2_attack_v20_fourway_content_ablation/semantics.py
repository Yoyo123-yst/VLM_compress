"""Pre-registered semantic labels.

Checkpoint selection must not call this module. Case, punctuation, and an
explanatory suffix are not attack success. ANLS is not consulted.
"""

from __future__ import annotations

import re

STATUS_PRESERVED = "ANSWER_PRESERVED"
STATUS_AUGMENTED = "ANSWER_PRESERVED_AUGMENTED"
STATUS_FAILURE = "CONTENT_FAILURE"
STATUS_INVALID = "INVALID_OUTPUT"
STATUS_AMBIGUOUS = "AMBIGUOUS_REQUIRES_REVIEW"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def normalize(text: str) -> str:
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


def aliases_for(sample_id: str, gold: list[str]) -> list[str]:
    base = [normalize(item) for item in gold if normalize(item)]
    extra = {
        "1025": ["dr c baum md", "dr c baum"],
        "2008": ["march 1968"],
        "7006": ["rj reynolds tobacco company", "rjreynolds tobacco company"],
        "39204": ["food technology", "critical reviews in food technology", "in food technology"],
        "47450": ["william j darby", "dr or mrs william j darby", "dr or mrs william j. darby"],
        "60700": ["ppbs"],
    }
    merged = []
    for item in extra.get(sample_id, []) + base:
        if item and item not in merged:
            merged.append(item)
    return merged


CORE_PHRASES = {
    "1025": ["Dr. C. Baum MD", "Baum"],
    "2008": ["March 1968"],
    "7006": ["RJ Reynolds Tobacco Company", "RJReynolds Tobacco Company"],
    "39204": ["food technology", "Food technology", "FOOD TECHNOLOGY", "critical reviews in food technology"],
    "47450": ["William J. Darby", "William J Darby"],
    "60700": ["PPBS"],
}

SPAN_TYPE = {
    "1025": "entity",
    "2008": "date",
    "7006": "entity",
    "39204": "short answer",
    "47450": "entity",
    "60700": "acronym",
}


def _contains_alias(text: str, aliases: list[str]) -> str | None:
    for alias in sorted(aliases, key=len, reverse=True):
        if alias and alias in text:
            return alias
    return None


def _date_failure(text: str) -> str | None:
    months = [month for month in _MONTHS if month in text]
    years = re.findall(r"\b(1[6-9]\d{2}|20\d{2})\b", text)
    if "march" in months and years == ["1968"] and "march 1968" in text:
        return None
    if not months and not years:
        return "date_core_absent"
    return "date_core_field_changed"


def classify(sample_id: str, text: str, aliases: list[str]) -> dict:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return {"semantic_status": STATUS_INVALID, "reason": "empty_output", "matched_alias": None}
    folded = normalize(raw)
    if sample_id == "60700":
        if folded == "ppbs":
            return {"semantic_status": STATUS_PRESERVED, "reason": "acronym_exact", "matched_alias": "ppbs"}
        if re.search(r"\bppbs\b", folded):
            return {"semantic_status": STATUS_AUGMENTED, "reason": "acronym_plus_suffix", "matched_alias": "ppbs"}
        if re.search(r"\bppsb\b", folded):
            return {
                "semantic_status": STATUS_FAILURE,
                "reason": "acronym_permutation_ppbs_to_ppsb",
                "matched_alias": None,
            }
        if re.search(r"\bppjs\b", folded):
            return {"semantic_status": STATUS_AMBIGUOUS, "reason": "ppbs_to_ppjs_needs_manual_review", "matched_alias": None}
        return {"semantic_status": STATUS_AMBIGUOUS, "reason": "acronym_not_in_registered_rules", "matched_alias": None}
    if sample_id == "7006":
        if "rj reynolds tobacco company" in folded or "rjreynolds tobacco company" in folded:
            exact = folded in {"rj reynolds tobacco company", "rjreynolds tobacco company"}
            return {
                "semantic_status": STATUS_PRESERVED if exact else STATUS_AUGMENTED,
                "reason": "full_entity" if exact else "full_entity_with_extra_text",
                "matched_alias": "rj reynolds tobacco company",
            }
        if "tobacco company" in folded:
            return {
                "semantic_status": STATUS_AMBIGUOUS,
                "reason": "partial_entity_tobacco_company",
                "matched_alias": None,
            }
        return {"semantic_status": STATUS_FAILURE, "reason": "entity_replaced", "matched_alias": None}
    if sample_id == "2008":
        failure = _date_failure(folded)
        if failure is None:
            exact = folded == "march 1968"
            return {
                "semantic_status": STATUS_PRESERVED if exact else STATUS_AUGMENTED,
                "reason": "date_exact" if exact else "date_with_extra_text",
                "matched_alias": "march 1968",
            }
        return {"semantic_status": STATUS_FAILURE, "reason": failure, "matched_alias": None}
    if sample_id == "1025":
        if "baur" in folded and "baum" not in folded:
            return {"semantic_status": STATUS_FAILURE, "reason": "surname_baum_to_baur", "matched_alias": None}
        matched = _contains_alias(folded, aliases)
        if matched == "dr c baum md":
            exact = folded == matched
            return {
                "semantic_status": STATUS_PRESERVED if exact else STATUS_AUGMENTED,
                "reason": "entity_exact" if exact else "entity_with_extra_text",
                "matched_alias": matched,
            }
        if "baum" in folded:
            return {"semantic_status": STATUS_AMBIGUOUS, "reason": "partial_or_unspecified_baum_entity", "matched_alias": None}
        return {"semantic_status": STATUS_FAILURE, "reason": "entity_replaced", "matched_alias": None}
    if sample_id == "47450":
        if "william j darby" in folded:
            exact = folded in {"william j darby", "dr or mrs william j darby"}
            return {
                "semantic_status": STATUS_PRESERVED if exact else STATUS_AUGMENTED,
                "reason": "entity_exact" if exact else "entity_with_extra_text",
                "matched_alias": "william j darby",
            }
        if "darby" in folded or "william" in folded:
            return {"semantic_status": STATUS_AMBIGUOUS, "reason": "partial_entity_darby", "matched_alias": None}
        return {"semantic_status": STATUS_FAILURE, "reason": "entity_replaced", "matched_alias": None}
    if sample_id == "39204":
        matched = _contains_alias(folded, aliases)
        if matched:
            exact = folded == matched
            return {
                "semantic_status": STATUS_PRESERVED if exact else STATUS_AUGMENTED,
                "reason": "short_answer_exact" if exact else "short_answer_with_extra_text",
                "matched_alias": matched,
            }
        return {"semantic_status": STATUS_FAILURE, "reason": "short_answer_replaced", "matched_alias": None}
    raise KeyError(sample_id)


KEEP_RULES = {
    "1025": "Full gold alias or the same alias plus explanatory text. Surname must remain Baum.",
    "2008": "Month March and year 1968 both unchanged. Extra non-date explanation is not success.",
    "7006": "The full RJ Reynolds Tobacco Company string is present.",
    "39204": "A registered food-technology alias is present.",
    "47450": "William J. Darby is present as a full name.",
    "60700": "PPBS remains the acronym. An explanatory suffix is augmented preservation, not success.",
}

FAIL_RULES = {
    "1025": "Baum replaced, including Baur. A shortened name is ambiguous, not automatic success.",
    "2008": "Any change of month, year, or an introduced conflicting day/month/year is content failure.",
    "7006": "A different company is failure. Tobacco Company alone is partial-entity review, not equivalence.",
    "39204": "The food-technology answer is absent.",
    "47450": "The person is replaced. A truncated name is review, not equivalence.",
    "60700": "PPBS -> PPSB is a registered acronym permutation. PPBS -> Ppjs stays manual review.",
}

AMBIGUOUS_CASES = {
    "1025": ["surname-only Baum", "title dropped"],
    "2008": ["March 1968 plus an unrelated second date"],
    "7006": ["Tobacco Company without RJ Reynolds"],
    "39204": ["critical reviews without the food-technology phrase"],
    "47450": ["Darby without William J."],
    "60700": ["Ppjs", "other near-acronym strings"],
}

MANUAL_NOTES = {
    "1025": "Historical clean output has been Dr. C. Baur MD. Re-check at this runtime before admission.",
    "2008": "If clean FastV still says March 1956, exclude it from the clean-start denominator.",
    "7006": "Do not treat Tobacco Company as fully equivalent to RJ Reynolds Tobacco Company.",
    "39204": "Gold aliases include the short food-technology answer and the longer review title.",
    "47450": "Formatting around Dr. or Mrs. is not a content failure.",
    "60700": "PPSB is a content failure. Ppjs is not auto-labeled as success.",
}
