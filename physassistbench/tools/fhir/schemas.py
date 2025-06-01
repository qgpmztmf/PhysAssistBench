"""
tools/fhir/schemas.py — OpenAI function-calling JSON schemas for 12 FHIR tools.

Format mirrors WildToolBench / existing tool_schemas.py:
  {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

Tool naming convention: <ResourceType>.<operation>
  e.g. Patient.read, Observation.search, Patient.everything
"""

FHIR_TOOL_SCHEMAS = [
    # ── Patient ───────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "Patient.read",
            "description": (
                "Read demographic information for a patient as a FHIR Patient resource. "
                "Returns gender, anchor_age, and deceased date if applicable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── Encounter ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "Encounter.search",
            "description": (
                "Search hospital encounters (admissions) for a patient. "
                "Returns admission/discharge times, admission type, location, and "
                "discharge disposition. Optionally filter by hadm_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional — filters to one encounter)"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── Condition ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "Condition.search",
            "description": (
                "Search ICD diagnosis conditions for a patient. "
                "Results are ordered by seq_num (principal diagnosis first). "
                "Filter by hadm_id for a specific admission, by code prefix (e.g. 'K74'), "
                "or by clinical_status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional)"
                    },
                    "code": {
                        "type": "string",
                        "description": "ICD code prefix to filter by (e.g. 'K74', 'E11')"
                    },
                    "clinical_status": {
                        "type": "string",
                        "enum": ["active", "resolved", "inactive"],
                        "description": "Filter by clinical status"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── Observation ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "Observation.search",
            "description": (
                "Search clinical observations for a patient. Covers laboratory results, "
                "vital signs, and microbiology results. "
                "Use category='laboratory' for lab tests (e.g. Potassium, Creatinine, CBC), "
                "category='vital-signs' for ICU vitals (heart rate, BP, SpO2), "
                "category='microbiology' for culture results. "
                "The code parameter accepts a test name (e.g. 'Potassium', 'Hemoglobin'). "
                "Results are sorted newest-first. When code is specified, all matching "
                "records are returned (no count limit)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional)"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["laboratory", "vital-signs", "microbiology"],
                        "description": "Observation category"
                    },
                    "code": {
                        "type": "string",
                        "description": "Lab/vital test name (e.g. 'Potassium', 'Heart Rate')"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start of date range (ISO datetime, e.g. '2180-08-06')"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End of date range (ISO datetime)"
                    },
                    "_count": {
                        "type": "integer",
                        "description": "Maximum number of results (default 20, ignored when code is set)",
                        "default": 20
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── MedicationRequest ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "MedicationRequest.search",
            "description": (
                "Search prescription (medication order) records for a patient. "
                "When hadm_id is provided, returns ALL prescriptions for that admission. "
                "Use the medication parameter for drug name filtering (e.g. 'Furosemide'). "
                "Results are sorted by start time newest-first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional)"
                    },
                    "medication": {
                        "type": "string",
                        "description": "Drug name substring filter (e.g. 'Furosemide', 'Insulin')"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "completed", "stopped"],
                        "description": "Filter by medication order status"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── MedicationAdministration ──────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "MedicationAdministration.search",
            "description": (
                "Search medication administration (eMAR) records showing when medications "
                "were actually given to the patient. "
                "Use medication parameter to filter by drug name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional)"
                    },
                    "medication": {
                        "type": "string",
                        "description": "Drug name substring filter"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── Procedure ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "Procedure.search",
            "description": (
                "Search ICD procedure codes for a patient. "
                "Returns surgical and clinical procedures ordered by seq_num."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional)"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── DiagnosticReport ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "DiagnosticReport.search",
            "description": (
                "Search radiology and other diagnostic reports for a patient. "
                "Full report text is included in the response. "
                "Use report_type to filter by modality (e.g. 'CT', 'MRI', 'X-ray', 'Echo')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional)"
                    },
                    "report_type": {
                        "type": "string",
                        "description": "Report modality filter (e.g. 'CT', 'MRI', 'X-ray')"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── DocumentReference ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "DocumentReference.search",
            "description": (
                "Search clinical notes and documents for a patient. "
                "Covers discharge summaries and radiology reports. "
                "Use type_code to select document type, or keyword for full-text search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional)"
                    },
                    "type_code": {
                        "type": "string",
                        "enum": ["discharge-summary", "radiology"],
                        "description": "Document type"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search within note text"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── AllergyIntolerance ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "AllergyIntolerance.search",
            "description": (
                "Search allergy and intolerance records for a patient. "
                "Note: MIMIC-IV does not have a dedicated allergy table; allergies are "
                "extracted from the Allergies section of discharge notes. "
                "Returns an empty bundle when no allergy information is documented."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional)"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── CarePlan ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "CarePlan.search",
            "description": (
                "Search care plan records for a patient extracted from MIMIC-IV discharge notes. "
                "Returns sections such as Discharge Instructions, Followup Instructions, and "
                "free-text Plan paragraphs as FHIR CarePlan resources. "
                "Use this to retrieve post-discharge care plans, follow-up schedules, or "
                "inpatient treatment plans documented in the discharge summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional — scopes to one admission)"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["discharge-planning", "followup", "treatment"],
                        "description": (
                            "Filter by plan category. "
                            "'discharge-planning': discharge instructions and condition; "
                            "'followup': follow-up appointment instructions; "
                            "'treatment': inpatient treatment plan sections. "
                            "Omit to return all categories."
                        )
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── Patient.everything ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "Patient.everything",
            "description": (
                "Returns a summary Bundle of ALL available FHIR resources for a patient: "
                "demographics, encounters, diagnoses, recent lab results, vital signs, "
                "medications, and procedures. "
                "Use this for a quick clinical overview. "
                "Optionally scope to one admission with hadm_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "integer",
                        "description": "MIMIC-IV patient identifier"
                    },
                    "hadm_id": {
                        "type": "integer",
                        "description": "Hospital admission ID (optional — scopes to one admission)"
                    }
                },
                "required": ["subject_id"]
            }
        }
    },

    # ── prepare_to_answer ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "prepare_to_answer",
            "description": (
                "Signal that sufficient information has been gathered and the agent "
                "is ready to provide the final answer to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer_type": {
                        "type": "string",
                        "enum": ["tool", "chat"],
                        "description": "'tool' if tool calls were made, 'chat' for knowledge-only"
                    }
                },
                "required": []
            }
        }
    },
]

# Convenience lookup: tool name → schema dict
FHIR_SCHEMA_BY_NAME: dict[str, dict] = {
    s["function"]["name"]: s for s in FHIR_TOOL_SCHEMAS
}

# Tool names list (mirrors get_tools_for_task return shape)
FHIR_TOOL_NAMES: list[str] = [s["function"]["name"] for s in FHIR_TOOL_SCHEMAS]
