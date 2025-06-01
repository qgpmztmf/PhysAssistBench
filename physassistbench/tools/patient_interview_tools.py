"""
Patient Interview Tools — 6 functions for clinical patient intake.

These tools simulate interviewing a patient by querying the PatientAgentRuntime.
Each function returns a natural language STRING (the patient's spoken response).
Structured annotations are stored as a side effect in PatientAgentRuntime._annotation_store
and are NOT returned to the Doctor Agent.

All functions take (subject_id, session_id, ...) — session_id is required to
route the call to the correct PatientAgentRuntime instance.

The session must be registered via register_session() before calling these tools.
"""

from physassistbench.phm.patient_agent_runtime import get_session


def patient_get_chief_complaint(subject_id: int, session_id: str) -> str:
    """
    Ask the patient what brought them in today.
    Returns the patient's natural-language chief complaint.
    """
    return get_session(session_id).respond("get_chief_complaint")


def patient_get_symptom_history(
    subject_id: int, session_id: str, query: str = ""
) -> str:
    """
    Ask the patient to describe their symptom history.
    Optionally focus on a specific symptom via the query parameter.
    Returns the patient's natural-language symptom description (OPQRST format).
    """
    return get_session(session_id).respond("get_symptom_history", query=query)


def patient_get_medication_adherence(
    subject_id: int, session_id: str, drug: str = ""
) -> str:
    """
    Ask the patient whether they are taking a specific medication as prescribed.
    Returns the patient's natural-language self-report of adherence.
    If info_completeness is 'critical_withheld' and drug matches a critical node,
    this call will reveal the withheld information.
    """
    return get_session(session_id).respond("get_medication_adherence", drug=drug)


def patient_get_social_history(subject_id: int, session_id: str) -> str:
    """
    Ask the patient about their social history (living situation, habits, support).
    Returns the patient's natural-language social history response.
    """
    return get_session(session_id).respond("get_social_history")


def patient_get_functional_status(subject_id: int, session_id: str) -> str:
    """
    Ask the patient about their functional status and daily activities.
    Returns the patient's natural-language description of functional limitations.
    """
    return get_session(session_id).respond("get_functional_status")


def patient_get_pain_assessment(subject_id: int, session_id: str) -> str:
    """
    Ask the patient to describe any pain they are experiencing (location, severity, character).
    Returns the patient's natural-language pain assessment.
    """
    return get_session(session_id).respond("get_pain_assessment")
