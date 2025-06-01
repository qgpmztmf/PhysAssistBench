"""
Phase 4 — Persona Characterisation (Primarily Hard Rules).

health_literacy  — Flesch-Kincaid grade level on patient quotes
adherence        — ADHERENCE_SIGNALS keyword mapping
anxiety_level    — emotional language detection
info_completeness— contradiction detection (hard flag + LLM confirmation in llm_generator)
self_diagnosis   — patient-initiated diagnosis language detection
"""
from __future__ import annotations
import re
import math
from statistics import mean
from typing import Optional

from .schema import Persona

# ---------------------------------------------------------------------------
# Flesch-Kincaid Grade Level (implemented without external library)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Approximate syllable count for an English word."""
    word = word.lower().strip(".,;:!?\"'")
    if not word:
        return 0
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Trailing silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float:
    """
    Compute Flesch-Kincaid Grade Level for a text.
    FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    """
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / sentences          # average sentence length
    asw = syllables / len(words)          # average syllables per word
    return 0.39 * asl + 11.8 * asw - 15.59


# Medical terminology vocabulary (used to detect high health literacy)
_MEDICAL_TERMS = {
    "cirrhosis", "hepatic", "encephalopathy", "ascites", "varices",
    "hyponatremia", "hypokalemia", "coagulopathy", "thrombocytopenia",
    "decompensated", "compensated", "lactulose", "furosemide", "spironolactone",
    "portal", "hypertension", "bilirubin", "albumin", "creatinine",
    "antiretroviral", "haart", "hiv", "hepatitis", "fibrosis",
    "diuretic", "paracentesis", "prognosis", "pathology",
}


def count_medical_terms(quotes: list[str]) -> int:
    words = " ".join(quotes).lower().split()
    return sum(1 for w in words if w.strip(".,;:!?\"'") in _MEDICAL_TERMS)


def estimate_literacy(patient_quotes: list[str]) -> str:
    """
    Estimate health literacy level from patient direct quotes.
    Returns 'high', 'medium', or 'low'.
    """
    if not patient_quotes:
        return "medium"
    fk_scores = [flesch_kincaid_grade(q) for q in patient_quotes if q.strip()]
    if not fk_scores:
        return "medium"
    avg_fk = mean(fk_scores)
    med_term_count = count_medical_terms(patient_quotes)

    if avg_fk < 6 and med_term_count < 2:
        return "low"
    elif avg_fk < 10 and med_term_count < 5:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Adherence classification
# ---------------------------------------------------------------------------

ADHERENCE_SIGNALS: dict[str, list[str]] = {
    "never_filled":       ["never filled", "did not fill", "hasn't filled", "never picked up"],
    "ran_out":            ["ran out of", "out of medication", "out of meds"],
    "self_discontinued":  ["self-discontinu", "stopped taking", "doesn't want to", "refused to take"],
    "poor":               ["not been taking", "not compliant", "non-compliant", "non compliant",
                           "missed doses", "not adherent", "non-adherent"],
    "good":               ["compliant", "taking all", "adherent", "takes all medications",
                           "taking her medications", "taking his medications"],
}


def classify_overall_adherence(hpi_text: str) -> str:
    """
    Classify overall medication adherence from HPI text.
    Returns one of: good / poor / never_filled / ran_out / self_discontinued / mixed / unknown
    """
    # Normalise whitespace so that line-wrapped phrases like "never \nfilled" still match
    text_lower = re.sub(r"\s+", " ", hpi_text.lower())
    found: dict[str, int] = {k: 0 for k in ADHERENCE_SIGNALS}
    for label, signals in ADHERENCE_SIGNALS.items():
        for sig in signals:
            if sig in text_lower:
                found[label] += 1

    hits = {k: v for k, v in found.items() if v > 0}
    if not hits:
        return "unknown"
    if len(hits) == 1:
        return list(hits.keys())[0]
    # Multiple signals found
    if "good" in hits and any(k in hits for k in ("poor", "never_filled", "ran_out", "self_discontinued")):
        return "mixed"
    # Return highest-count label
    return max(hits, key=hits.get)


def classify_drug_adherence(hpi_text: str, drug_name: str) -> str:
    """
    Classify adherence for a specific drug by searching a context window.
    """
    key = drug_name.lower().split()[0]
    text_lower = hpi_text.lower()

    # Find all positions where drug is mentioned and check nearby context
    positions = [m.start() for m in re.finditer(re.escape(key), text_lower)]
    if not positions:
        return "unknown"

    for pos in positions:
        window = text_lower[max(0, pos - 60): pos + 80]
        for label, signals in ADHERENCE_SIGNALS.items():
            for sig in signals:
                if sig in window:
                    return label
    return "unknown"


# ---------------------------------------------------------------------------
# Anxiety level estimation
# ---------------------------------------------------------------------------

_HIGH_ANXIETY_PHRASES = [
    "extremely worried", "very scared", "terrified", "panicking", "can't sleep",
    "constantly anxious", "overwhelmed", "desperate", "please help",
    "afraid to die", "fear of dying", "death", "dying",
    "can't cope", "breaking down",
]
_LOW_ANXIETY_PHRASES = [
    "feeling okay", "not worried", "doing well", "comfortable",
    "not concerned", "unconcerned", "let go", "gave up",
]


def estimate_anxiety(hpi_text: str, patient_quotes: list[str] | None = None) -> str:
    """Estimate anxiety level from HPI text and patient quotes."""
    combined = hpi_text + " " + " ".join(patient_quotes or [])
    low = combined.lower()

    high_score = sum(1 for p in _HIGH_ANXIETY_PHRASES if p in low)
    low_score  = sum(1 for p in _LOW_ANXIETY_PHRASES if p in low)

    if high_score > low_score and high_score >= 2:
        return "high"
    if low_score > high_score:
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# Self-diagnosis tendency
# ---------------------------------------------------------------------------

_SELF_DIAG_PHRASES = [
    "i think i have", "i believe i have", "maybe i have",
    "she thinks she has", "he thinks he has", "they think they have",
    "patient thinks", "patient believes",
    "sounds like", "looks like", "i read that", "i googled", "she googled", "he googled",
    "i diagnosed myself", "self-diagnosed", "i know what this is",
    "it must be", "probably have", "i researched",
]


def detect_self_diagnosis(hpi_text: str, patient_quotes: list[str] | None = None) -> str:
    """Detect whether the patient tends to self-diagnose."""
    combined = hpi_text.lower() + " " + " ".join(patient_quotes or []).lower()
    for phrase in _SELF_DIAG_PHRASES:
        if phrase in combined:
            return "present"
    return "absent"


# ---------------------------------------------------------------------------
# Top-level persona builder
# ---------------------------------------------------------------------------

def build_persona(
    hpi_text: str,
    patient_quotes: list[str],
    info_completeness: str = "full",   # determined by detect_info_withheld in llm_generator
) -> Persona:
    """
    Assemble the full Persona from extracted HPI text and patient quotes.
    info_completeness should be pre-determined by the caller (uses LLM confirmation).
    """
    return Persona(
        health_literacy=estimate_literacy(patient_quotes),
        anxiety_level=estimate_anxiety(hpi_text, patient_quotes),
        info_completeness=info_completeness,
        self_diagnosis_tendency=detect_self_diagnosis(hpi_text, patient_quotes),
        overall_adherence=classify_overall_adherence(hpi_text),
    )
