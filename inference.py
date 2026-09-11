"""
Core inference pipeline for SIF-Guard.
Combines: TF-IDF+LogReg classifier -> extraction layer -> priority mapping
Produces a dict matching the dashboard's RawAnalysisResponse contract exactly:
  sif_potential, confidence, activity, hazard, barrier_failure,
  potential_consequence, life_saving_rule, priority, priority_score,
  evidence, model_attribution
"""

import pickle
import os
import numpy as np

from extraction import extract_structured_fields, compute_priority

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    _VECTORIZER = pickle.load(f)

with open(os.path.join(_MODEL_DIR, "sif_classifier.pkl"), "rb") as f:
    _CLASSIFIER = pickle.load(f)


def _top_attribution_tokens(text_vec, top_n: int = 8):
    """Builds a lightweight per-request token attribution list for the explainability panel."""
    coefs = _CLASSIFIER.coef_[0]
    row = text_vec.toarray()[0]
    present_idx = np.nonzero(row)[0]
    if len(present_idx) == 0:
        return []

    contributions = row[present_idx] * coefs[present_idx]
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    feature_names = _VECTORIZER.get_feature_names_out()

    tokens = []
    for idx in order:
        feat_idx = present_idx[idx]
        weight = float(contributions[idx])
        tokens.append({
            "token": feature_names[feat_idx],
            "weight": round(weight, 4),
            "class": "SIF" if weight > 0 else "Non-SIF",
        })
    return tokens


def analyze_report(report_text: str) -> dict:
    if not report_text or not report_text.strip():
        raise ValueError("report_text must not be empty")

    text_vec = _VECTORIZER.transform([report_text])
    proba = _CLASSIFIER.predict_proba(text_vec)[0]
    sif_potential = bool(_CLASSIFIER.predict(text_vec)[0] == 1)
    confidence = float(proba[1]) if sif_potential else float(proba[0])

    structured = extract_structured_fields(report_text, sif_potential, confidence)
    priority, priority_score = compute_priority(sif_potential, confidence)
    tokens = _top_attribution_tokens(text_vec)

    return {
        "sif_potential": sif_potential,
        "confidence": round(confidence, 4),
        "activity": structured["activity"],
        "hazard": structured["hazard"],
        "barrier_failure": structured["barrier_failure"],
        "potential_consequence": structured["potential_consequence"],
        "life_saving_rule": structured["life_saving_rule"],
        "priority": priority,
        "priority_score": priority_score,
        "evidence": structured["evidence"],
        "model_attribution": {
            "engine": "TF-IDF + Logistic Regression (SIF-Guard v1)",
            "tokens": tokens,
            "rationale": (
                f"Classified as {'SIF' if sif_potential else 'Non-SIF'} "
                f"with {confidence*100:.1f}% confidence based on weighted "
                f"keyword/phrase patterns in the report text."
            ),
        },
    }
