"""
FHIR write tool schemas for the discharge_planning Write/Update turn (T3).

These three tools simulate EHR write operations during benchmark generation
and are exposed to evaluated models at inference time so they can issue
proper tool calls for T3 Write/Update turns.
"""

WRITE_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "MedicationRequest.create",
            "description": (
                "Create a new medication order (prescription) for the patient. "
                "Use for discharge medication additions or dose changes identified during "
                "the encounter. All required fields must be specified explicitly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "medication": {"type": "string", "description": "Drug name (generic preferred, e.g. 'Furosemide')"},
                    "dose": {"type": "string", "description": "Dose amount and unit, e.g. '40 mg'"},
                    "route": {"type": "string", "description": "Route of administration: 'oral', 'IV', 'topical', etc."},
                    "frequency": {"type": "string", "description": "Dosing frequency: 'once daily', 'BID', 'TID', 'PRN', etc."},
                    "indication": {"type": "string", "description": "Clinical reason for the order (optional but recommended)"},
                    "duration_days": {"type": "integer", "description": "Duration in days (optional — leave blank for chronic medications)"},
                },
                "required": ["subject_id", "medication", "dose", "route", "frequency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ServiceRequest.create",
            "description": (
                "Create a referral or service request for the patient, such as home health, "
                "physical therapy, social work consultation, or specialist follow-up. "
                "Use for discharge planning services identified as necessary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "service_type": {
                        "type": "string",
                        "description": (
                            "Type of service: 'home-health', 'physical-therapy', "
                            "'occupational-therapy', 'social-work', 'cardiology-followup', "
                            "'primary-care-followup', 'palliative-care', 'wound-care', etc."
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["routine", "urgent", "asap"],
                        "description": "Request priority",
                    },
                    "note": {"type": "string", "description": "Clinical justification for the request"},
                    "target_date": {"type": "string", "description": "Target service start date (ISO date, optional)"},
                },
                "required": ["subject_id", "service_type", "priority"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Flag.create",
            "description": (
                "Create a clinical alert flag on the patient record. "
                "Use for safety-critical alerts: allergy updates, fall risk, "
                "medication reconciliation issues, or mandatory follow-up requirements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "category": {
                        "type": "string",
                        "enum": ["safety", "allergy", "administrative", "clinical"],
                        "description": "Flag category",
                    },
                    "code": {"type": "string", "description": "Short flag code or label, e.g. 'HIGH_FALL_RISK'"},
                    "detail": {"type": "string", "description": "Human-readable description of the flag"},
                },
                "required": ["subject_id", "category", "code", "detail"],
            },
        },
    },
]

WRITE_TOOL_NAMES: set[str] = {s["function"]["name"] for s in WRITE_TOOL_SCHEMAS}
