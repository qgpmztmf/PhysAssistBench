"""
Phase 3 — Semantically Generated Fields (LLM).

Only fields that genuinely require intent understanding or natural language
generation are handled here.  All inputs are already validated by hard rules.
"""
from __future__ import annotations
import json
import logging

from physassistbench.pipeline.agents.llm_client import llm_call, extract_json
from .section_parser import mask_deid

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common forbidden medical terms for patient_term generation
# ---------------------------------------------------------------------------
_FORBIDDEN_TERMS = [
    "hepatic encephalopathy", "cirrhosis", "decompensation", "hyponatremia",
    "ascites", "coagulopathy", "thrombocytopenia", "hepatorenal syndrome",
    "varices", "portal hypertension", "encephalopathy", "INR", "bilirubin",
    "creatinine", "albumin", "glomerular filtration rate", "GFR",
    "Child-Pugh", "MELD", "TIPS",
]


# ---------------------------------------------------------------------------
# patient_term generation
# ---------------------------------------------------------------------------

def generate_patient_term(
    medical_term: str,
    dc_instructions_excerpt: str = "",
    patient_quotes: list[str] | None = None,
    forbidden_terms: list[str] | None = None,
) -> str:
    """
    Convert a medical diagnosis term into a patient-friendly second-person
    explanation.

    Returns the patient_term string. Falls back to the raw medical_term on
    any LLM error.
    """
    forbidden = forbidden_terms or _FORBIDDEN_TERMS
    quotes_block = "\n".join(f'- "{q}"' for q in (patient_quotes or [])[:5])
    instructions_block = mask_deid(dc_instructions_excerpt[:800]) if dc_instructions_excerpt else "(none available)"

    prompt = f"""Convert the following medical diagnosis into patient-friendly language.

Medical term: {medical_term}

Reference — physician's discharge instructions to the patient (already plain language):
{instructions_block}

Patient's own words from the medical record:
{quotes_block or "(none available)"}

Requirements:
- Second person, beginning with "You have..." or "You were diagnosed with..."
- Must NOT use any of these medical terms: {', '.join(forbidden[:15])}
- Must include at least one functional impact the patient can perceive (e.g. symptoms, daily life effects)
- 1–2 sentences maximum
- Plain English, appropriate for someone who did not finish high school

Return ONLY the patient-friendly sentence(s), no preamble."""

    try:
        result = llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        return result.strip()
    except Exception as e:
        log.warning("generate_patient_term failed for '%s': %s", medical_term, e)
        return medical_term


# ---------------------------------------------------------------------------
# patient_explanation generation (adherence)
# ---------------------------------------------------------------------------

def generate_patient_explanation(
    extracted_quote: str,
    drug_name: str,
    adherence_type: str,
) -> str:
    """
    Generate a first-person patient explanation of their adherence behaviour,
    strictly grounded in an extracted HPI quote.

    Returns the explanation string. Falls back to the raw quote on error.
    """
    if not extracted_quote:
        return ""

    prompt = f"""Based on the following patient quote from a medical record, generate a first-person explanation of the patient's medication adherence behaviour.

Patient's exact words (from medical record): {extracted_quote}
Medication: {drug_name}
Adherence classification: {adherence_type}

Requirements:
- Must be grounded in the quote — do NOT add reasons not present in the original
- Write in first person ("I...")
- Colloquial, appropriate for someone with low health literacy
- Maximum two sentences
- Do NOT start with "I understand..." or "Based on..."

Return ONLY the first-person explanation, no preamble."""

    try:
        result = llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        return result.strip()
    except Exception as e:
        log.warning("generate_patient_explanation failed for '%s': %s", drug_name, e)
        return extracted_quote


# ---------------------------------------------------------------------------
# patient_signal generation for warning signs
# ---------------------------------------------------------------------------

def generate_patient_signal(
    condition: str,
    medical_condition: str,
    lab_context: str = "",
) -> str:
    """
    Convert a clinical warning-sign condition (numeric threshold or symptom combo)
    into a patient-perceivable subjective signal description.

    Returns the patient_signal string.
    """
    prompt = f"""A patient needs to know what warning signs to watch for at home.

Clinical condition: {medical_condition}
Trigger condition (from lab/clinical record): {condition}
{f'Lab context: {lab_context}' if lab_context else ''}

Describe what the PATIENT would physically feel or observe that should prompt them to seek care.

Requirements:
- Describe only what the patient can perceive (feelings, observable changes) — no lab values
- Plain English, no medical jargon
- 1–2 sentences
- Be specific and concrete (e.g. "your heartbeat feels irregular" not "cardiac symptoms")

Return ONLY the patient-perceivable signal description, no preamble."""

    try:
        result = llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        return result.strip()
    except Exception as e:
        log.warning("generate_patient_signal failed for '%s': %s", medical_condition, e)
        return f"Watch for any sudden worsening related to {medical_condition}."


# ---------------------------------------------------------------------------
# info_completeness detection (hard-rule flagging + LLM confirmation)
# ---------------------------------------------------------------------------

def detect_info_withheld(
    patient_statement: str,
    doctor_observation: str,
) -> dict:
    """
    Determine whether a patient statement in the HPI contradicts a physician
    observation in the same note — indicating critical_withheld behaviour.

    Returns {"is_withheld": bool, "evidence": str}.
    """
    if not patient_statement or not doctor_observation:
        return {"is_withheld": False, "evidence": ""}

    prompt = f"""Two passages from the same hospital discharge note are shown below.
Determine whether a contradiction exists where the patient claims compliance or denies a behaviour,
but the physician's observation indicates otherwise.

Passage A — patient self-report:
{mask_deid(patient_statement[:500])}

Passage B — physician observation:
{mask_deid(doctor_observation[:500])}

Return a JSON object with exactly these two keys:
  "is_withheld": true if a genuine contradiction exists, false otherwise
  "evidence": a one-sentence explanation of the contradiction (or "" if none)

Return ONLY the JSON object."""

    try:
        raw = llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        result = extract_json(raw)
        if isinstance(result, dict) and "is_withheld" in result:
            return result
        return {"is_withheld": False, "evidence": ""}
    except Exception as e:
        log.warning("detect_info_withheld failed: %s", e)
        return {"is_withheld": False, "evidence": ""}


# ---------------------------------------------------------------------------
# patient_impact generation for diagnoses
# ---------------------------------------------------------------------------

def generate_patient_impact(
    medical_term: str,
    patient_term: str,
    dc_instructions_excerpt: str = "",
) -> str:
    """
    Generate a description of how this diagnosis affects the patient's daily life.
    """
    instructions_block = mask_deid(dc_instructions_excerpt[:600]) if dc_instructions_excerpt else "(none)"

    prompt = f"""A patient has been diagnosed with: {medical_term}
Plain-language version already written: {patient_term}

Physician's discharge instructions excerpt:
{instructions_block}

Describe in 1–2 plain-English sentences how this condition affects the patient's DAILY LIFE
(e.g. activities they need to do, restrictions, symptoms they manage at home).
Write in second person ("You need to...").
No medical jargon. Return ONLY the impact description."""

    try:
        result = llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        return result.strip()
    except Exception as e:
        log.warning("generate_patient_impact failed for '%s': %s", medical_term, e)
        return ""
