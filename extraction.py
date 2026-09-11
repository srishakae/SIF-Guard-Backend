"""
SIF-Guard rule-based extraction layer.

The ML classifier (TF-IDF + LogisticRegression) only answers SIF Potential + Confidence.
This module derives the remaining structured fields the dashboard needs:
  activity, hazard, barrier_failure, potential_consequence, life_saving_rule, evidence

Two paths:
  1. TEMPLATE PATH — matches the fixed clause used by the training data generator:
     "...The missing control was that <X>, leaving personnel exposed to <Y>. ..."
     When present, barrier_failure/potential_consequence are extracted verbatim
     (verbatim substrings are REQUIRED so the dashboard's evidence highlighter can find them).
  2. FALLBACK PATH — generic keyword scan, used for any real-world text that doesn't
     follow the training template (this is the path real judge-submitted reports will hit).
"""

import re

TEMPLATE_PATTERN = re.compile(
    r"missing control was that (.*?), leaving personnel exposed to (.*?)\.",
    re.IGNORECASE,
)

# Keyword -> Life-Saving Rule mapping, checked in priority order.
# Order matters: more specific categories are checked before generic ones.
LSR_KEYWORDS = [
    ("Confined Space", [
        "atmospher", "gas monitor", "gas test", "gas reading", "oxygen-deficient",
        "vessel atmosphere", "confined space", "toxic gas"
    ]),
    ("Working at Height", [
        "anchor point", "fall-arrest", "fall protection", "elevated edge",
        "loss of balance at height", "fall from", "platform", "working at height"
    ]),
    ("Hot Work", [
        "hot-work", "hot work", "flammable vapour", "ignitable atmosphere", "welding"
    ]),
    ("Driving Safety", [
        "driver", "vehicle", "reversing", "pedestrian"
    ]),
    ("Lifting Operations", [
        "suspended-load", "suspended load", "lifting area", "crane", "rigging",
        "falling or swinging load"
    ]),
    ("Energy Isolation", [
        "isolat", "lockout", "electrical supply", "energized", "stored electrical",
        "stored process energy", "zero pressure", "line opening", "equipment start-up"
    ]),
    ("Line of Fire", [
        "exclusion zone", "exclusion area", "moving section", "crush", "pinch point",
        "hands were placed", "line of fire"
    ]),
    ("Process Safety / PPE", [
        "pressurized fluid", "stored pressure", "vapour controls", "ppe", "chemical"
    ]),
]


def _extract_first_sentence(text: str) -> str:
    match = re.search(r"^(.*?[.!?])(\s|$)", text.strip())
    return match.group(1).strip() if match else text.strip()[:120]


def _map_life_saving_rule(*texts: str) -> str:
    combined = " ".join(texts).lower()
    for lsr, keywords in LSR_KEYWORDS:
        if any(kw in combined for kw in keywords):
            return lsr
    return "None Applicable"


def extract_structured_fields(report_text: str, sif_potential: bool, confidence: float) -> dict:
    """
    Returns a dict with: activity, hazard, barrier_failure, potential_consequence,
    life_saving_rule, evidence (list of {text, type}).
    All 'text' values in evidence are guaranteed to be verbatim substrings of report_text.
    """
    evidence = []

    activity = _extract_first_sentence(report_text)
    evidence.append({"text": activity, "type": "Activity"})

    template_match = TEMPLATE_PATTERN.search(report_text)

    if template_match:
        # ---- TEMPLATE PATH: exact extraction ----
        barrier_failure_raw = template_match.group(1).strip()
        consequence_raw = template_match.group(2).strip()

        # Re-locate the exact verbatim substrings as they appear in the original text
        # (group() spans are already verbatim from the source string, so this is safe)
        barrier_failure = barrier_failure_raw[0].upper() + barrier_failure_raw[1:]
        potential_consequence = consequence_raw[0].upper() + consequence_raw[1:]
        hazard = potential_consequence  # the consequence phrase doubles as the hazard descriptor

        evidence.append({"text": barrier_failure_raw, "type": "Barrier Failure"})
        evidence.append({"text": consequence_raw, "type": "Risk Evidence"})

        life_saving_rule = _map_life_saving_rule(barrier_failure_raw, consequence_raw)

    else:
        # ---- FALLBACK PATH: generic keyword scan for real-world / non-template text ----
        life_saving_rule = _map_life_saving_rule(report_text)

        if sif_potential:
            hazard = "Potential high-energy exposure identified in report narrative"
            barrier_failure = "Barrier failure not explicitly stated — requires human review"
            potential_consequence = "Potential serious injury or fatality — requires human review"
        else:
            hazard = "No significant hazard identified"
            barrier_failure = "No barrier failure identified"
            potential_consequence = "Low — condition addressed/corrected as reported"

    return {
        "activity": activity,
        "hazard": hazard,
        "barrier_failure": barrier_failure,
        "potential_consequence": potential_consequence,
        "life_saving_rule": life_saving_rule,
        "evidence": evidence,
    }


def compute_priority(sif_potential: bool, confidence: float) -> tuple[str, float]:
    """Maps (sif_potential, confidence) -> (priority_label, priority_score)."""
    if not sif_potential:
        # Even confident Non-SIF calls stay LOW; only a genuinely low-confidence
        # Non-SIF call (borderline case) bumps to MEDIUM for a human to glance at.
        priority = "MEDIUM" if confidence < 0.6 else "LOW"
        score = round(1 - confidence, 2) if priority == "MEDIUM" else round((1 - confidence) * 0.5, 2)
        return priority, score

    if confidence >= 0.90:
        return "CRITICAL", round(confidence, 2)
    elif confidence >= 0.75:
        return "HIGH", round(confidence, 2)
    else:
        return "MEDIUM", round(confidence, 2)
