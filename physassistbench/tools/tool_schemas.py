"""
OpenAI function-calling schemas for all 28 EHR API tools.
Format mirrors WildToolBench: {"type": "function", "function": {...}}
"""

EHR_TOOL_SCHEMAS = [
    # ─── GROUP 1: Hospital / General ────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_patient_info",
            "description": "Returns demographic information for a patient, including gender, age, and death date if applicable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_admissions",
            "description": "Returns all hospital admissions for a patient with admission/discharge times, type, location, and insurance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_admission_details",
            "description": "Returns detailed information for a specific hospital admission including discharge location and expire flag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier"}
                },
                "required": ["subject_id", "hadm_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_diagnoses",
            "description": "Returns ICD diagnosis codes for a patient. If hadm_id is provided, returns diagnoses for that specific admission only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_results",
            "description": "Returns laboratory test results with reference ranges and abnormal flags. Can filter by admission, test name, or abnormal values only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"},
                    "item_name": {"type": "string", "description": "Name of the lab test to filter by (optional, e.g. 'Potassium', 'Creatinine')"},
                    "abnormal_only": {"type": "boolean", "description": "If true, return only abnormal results (default false)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_trends",
            "description": "Returns the N most recent results for a specific lab test to show trends over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "item_name": {"type": "string", "description": "Name of the lab test (e.g. 'Creatinine', 'Hemoglobin')"},
                    "n_recent": {"type": "integer", "description": "Number of most recent results to return (default 5)"}
                },
                "required": ["subject_id", "item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_microbiology_results",
            "description": "Returns microbiology culture results including organism identification, antibiotic susceptibility, and specimen type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_prescriptions",
            "description": "Returns medication prescriptions with drug name, dose, route, and frequency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_medication_administration",
            "description": "Returns medication administration records (eMAR) showing what was actually given, when, and any missed doses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"},
                    "medication": {"type": "string", "description": "Medication name to filter by (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_procedures",
            "description": "Returns ICD procedure codes performed during a hospital admission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_drg_info",
            "description": "Returns Diagnosis-Related Group (DRG) codes with severity and mortality weight for a hospital admission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_history",
            "description": "Returns the hospital service history (e.g., Medicine, Surgery, Cardiology) including transfer times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    # ─── GROUP 2: ICU ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_icu_stays",
            "description": "Returns all ICU stay records including care unit, admission/discharge times, and length of stay.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_icu_vitals",
            "description": "Returns ICU chart events (vital signs, ventilator settings, neurological assessments). Can filter by stay or vital name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "stay_id": {"type": "integer", "description": "ICU stay identifier (optional)"},
                    "vital_name": {"type": "string", "description": "Vital sign name filter (optional, e.g. 'Heart Rate', 'Blood Pressure')"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_icu_fluids_in",
            "description": "Returns ICU fluid and medication input events (IV fluids, vasopressors, nutrition).",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "stay_id": {"type": "integer", "description": "ICU stay identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_icu_output",
            "description": "Returns ICU fluid output events (urine output, drain output).",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "stay_id": {"type": "integer", "description": "ICU stay identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    # ─── GROUP 3: Emergency Department ──────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_ed_visits",
            "description": "Returns all emergency department visit records with arrival transport and disposition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ed_triage",
            "description": "Returns ED triage assessment including vital signs and chief complaint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "stay_id": {"type": "integer", "description": "ED stay identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ed_vital_signs",
            "description": "Returns time-series vital sign measurements during an ED visit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "stay_id": {"type": "integer", "description": "ED stay identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ed_diagnoses",
            "description": "Returns ICD diagnosis codes assigned during an ED visit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "stay_id": {"type": "integer", "description": "ED stay identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ed_medications",
            "description": "Returns ED medication reconciliation records and pyxis dispensing events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "stay_id": {"type": "integer", "description": "ED stay identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    # ─── GROUP 4: Clinical Notes ─────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_discharge_summary",
            "description": "Returns the full text of the discharge summary for a hospital admission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_discharge_section",
            "description": "Returns a specific section of the discharge summary (e.g., 'Assessment and Plan', 'Medications on Admission', 'Discharge Condition').",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"},
                    "section_name": {"type": "string", "description": "Section header to extract (e.g. 'Assessment and Plan', 'Physical Exam', 'Discharge Medications')"}
                },
                "required": ["subject_id", "section_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_radiology_report",
            "description": "Returns radiology report text (e.g., chest X-ray, CT scan findings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "hadm_id": {"type": "integer", "description": "Hospital admission identifier (optional)"},
                    "report_type": {"type": "string", "description": "Type of radiology report to filter by (optional, e.g. 'CHEST', 'CT', 'MRI')"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Searches across all clinical notes (discharge summaries and radiology reports) for a keyword or phrase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "keyword": {"type": "string", "description": "Keyword or phrase to search for in clinical notes"}
                },
                "required": ["subject_id", "keyword"]
            }
        }
    },
    # ─── GROUP 5: Utilities ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_vital_signs_outpatient",
            "description": "Returns outpatient vital signs and measurements (blood pressure, weight, BMI) from the OMR table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_timeline",
            "description": "Returns a chronological timeline of all clinical events (admissions, ICU stays, ED visits, diagnoses) for a patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "event_type": {"type": "string", "description": "Filter by event type: 'admission', 'discharge', 'icu_admit', 'ed_visit', 'diagnosis' (optional)"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_to_answer",
            "description": "Signal that all necessary tool calls have been completed and the agent is ready to provide the final answer to the user. Call this after all required data has been retrieved.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]

# ─── Patient Interview Tool Schemas ─────────────────────────────────────────

PATIENT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "patient.get_chief_complaint",
            "description": "Ask the patient what brought them in today. Returns the patient's natural-language chief complaint as a spoken response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "session_id": {"type": "string", "description": "Session identifier (provided in env_info)"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_symptom_history",
            "description": "Ask the patient to describe their symptom history. Optionally specify a symptom focus via the query parameter. Returns a natural-language OPQRST-style description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "session_id": {"type": "string", "description": "Session identifier (provided in env_info)"},
                    "query": {"type": "string", "description": "Optional symptom focus (e.g., 'abdominal pain', 'shortness of breath')"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_medication_adherence",
            "description": "Ask the patient whether they are taking a specific medication as prescribed. Returns the patient's natural-language self-report of adherence. For critical_withheld patients, naming the drug specifically may reveal withheld non-adherence information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "session_id": {"type": "string", "description": "Session identifier (provided in env_info)"},
                    "drug": {"type": "string", "description": "Medication name to ask about (e.g., 'Lactulose', 'Insulin', 'Metformin')"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_social_history",
            "description": "Ask the patient about their social history including living situation, occupation, habits (smoking, alcohol), and support system. Returns a natural-language response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "session_id": {"type": "string", "description": "Session identifier (provided in env_info)"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_functional_status",
            "description": "Ask the patient about their functional status and ability to perform daily activities. Returns a natural-language description of any limitations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "session_id": {"type": "string", "description": "Session identifier (provided in env_info)"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_pain_assessment",
            "description": "Ask the patient to describe any pain they are experiencing (location, severity 0-10, character, onset, radiation). Returns a natural-language pain assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV patient identifier"},
                    "session_id": {"type": "string", "description": "Session identifier (provided in env_info)"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
]

# Lookup: name → schema dict
TOOL_SCHEMA_BY_NAME = {s["function"]["name"]: s for s in EHR_TOOL_SCHEMAS}
# Add patient schemas to lookup
TOOL_SCHEMA_BY_NAME.update({s["function"]["name"]: s for s in PATIENT_TOOL_SCHEMAS})

# Subset for each task domain (tool names only)
TASK_TOOLS = {
    "LabInterp": [
        "get_patient_info", "get_admissions", "get_admission_details",
        "get_lab_results", "get_lab_trends", "get_microbiology_results",
        "get_vital_signs_outpatient", "get_radiology_report",
        "get_patient_timeline", "ask_user_for_required_parameters", "prepare_to_answer",
    ],
    "MedRecon": [
        "get_patient_info", "get_admissions", "get_admission_details",
        "get_prescriptions", "get_medication_administration",
        "get_drg_info", "get_service_history",
        "get_patient_timeline", "ask_user_for_required_parameters", "prepare_to_answer",
    ],
    "DiagCode": [
        "get_patient_info", "get_admissions", "get_admission_details",
        "get_diagnoses", "get_lab_results", "get_lab_trends",
        "get_microbiology_results", "get_procedures",
        "get_discharge_summary", "get_discharge_section",
        "get_radiology_report", "get_patient_timeline",
        "ask_user_for_required_parameters", "prepare_to_answer",
    ],
    "WorkflowQuery": [
        "get_patient_info", "get_admissions", "get_admission_details",
        "get_diagnoses", "get_lab_results", "get_lab_trends",
        "get_microbiology_results", "get_prescriptions",
        "get_medication_administration", "get_procedures",
        "get_drg_info", "get_service_history",
        "get_icu_stays", "get_icu_vitals",
        "get_ed_visits", "get_ed_triage",
        "get_discharge_summary", "get_radiology_report",
        "get_vital_signs_outpatient", "get_patient_timeline",
        "ask_user_for_required_parameters", "prepare_to_answer",
    ],
    "ICUReasoning": [
        "get_patient_info", "get_admissions", "get_admission_details",
        "get_icu_stays", "get_icu_vitals", "get_icu_fluids_in", "get_icu_output",
        "get_lab_results", "get_lab_trends",
        "get_prescriptions", "get_medication_administration",
        "get_patient_timeline", "ask_user_for_required_parameters", "prepare_to_answer",
    ],
    "DischargePlan": [
        "get_patient_info", "get_admissions", "get_admission_details",
        "get_diagnoses", "get_procedures", "get_drg_info",
        "get_service_history", "get_discharge_summary",
        "get_discharge_section", "get_radiology_report",
        "get_lab_results", "get_prescriptions",
        "get_patient_timeline", "ask_user_for_required_parameters", "prepare_to_answer",
    ],
    # ── Patient Interview domain ─────────────────────────────────────────────
    "PatientInterview": [
        "patient.get_chief_complaint",
        "patient.get_symptom_history",
        "patient.get_medication_adherence",
        "patient.get_social_history",
        "patient.get_functional_status",
        "patient.get_pain_assessment",
        "prepare_to_answer",
    ],
}


# Patient-interview tools excluded from the benchmark tool set (paper Appendix C
# exposes 5 patient tools; chief complaint is elicited via get_symptom_history).
_EXCLUDED_PATIENT_TOOLS = {"patient.get_chief_complaint"}


def _benchmark_patient_schemas() -> list:
    return [s for s in PATIENT_TOOL_SCHEMAS
            if s["function"]["name"] not in _EXCLUDED_PATIENT_TOOLS]


def get_fhir_tools_for_task(task_domain: str) -> list:
    """
    Return FHIR tool schemas for a given task domain.
    All clinical domains share the same FHIR tool set; patient interview
    tools are appended for PatientInterview domain.

    The exposed set matches the paper's tool inventory (Appendix C):
    9 EHR read tools + 5 patient-interview tools + prepare_to_answer
    (write tools are appended separately for Write/Update turns).
    """
    from physassistbench.tools.fhir.schemas import FHIR_TOOL_SCHEMAS
    from physassistbench.tools.fhir.schemas import FHIR_SCHEMA_BY_NAME

    # Core FHIR EHR read tools (all domains use the full set)
    fhir_ehr_names = [
        "Patient.read", "Encounter.search", "Condition.search",
        "Observation.search", "MedicationRequest.search",
        "MedicationAdministration.search",
        "DiagnosticReport.search", "DocumentReference.search",
        "CarePlan.search",
        "prepare_to_answer",
    ]

    if task_domain == "PatientInterview":
        return _benchmark_patient_schemas() + [FHIR_SCHEMA_BY_NAME["prepare_to_answer"]]

    schemas = [FHIR_SCHEMA_BY_NAME[n] for n in fhir_ehr_names if n in FHIR_SCHEMA_BY_NAME]

    # Append patient interview tools for mixed/workup turns
    schemas.extend(_benchmark_patient_schemas())

    return schemas


def get_tools_for_task(task_domain: str, language: str = "en") -> list:
    """
    Return the list of tool schemas for a given task domain.

    Args:
        task_domain: One of the six clinical domains or 'PatientInterview'.
        language: 'en' (default, English descriptions) or 'zh' (Chinese descriptions).
                  Loads from tools/locales/tool_schemas_en.py or tool_schemas_zh.py.
                  Falls back to the built-in English schemas if the locale file is missing.
    """
    if language == "en":
        # Load from locale file (identical to built-in schemas but kept separate for clarity)
        try:
            from physassistbench.tools.locales.tool_schemas_en import (
                EHR_TOOL_SCHEMAS as _EHR, PATIENT_TOOL_SCHEMAS as _PAT
            )
        except ImportError:
            _EHR, _PAT = EHR_TOOL_SCHEMAS, PATIENT_TOOL_SCHEMAS
    elif language == "zh":
        try:
            from physassistbench.tools.locales.tool_schemas_zh import (
                EHR_TOOL_SCHEMAS as _EHR, PATIENT_TOOL_SCHEMAS as _PAT
            )
        except ImportError:
            import warnings
            warnings.warn("Chinese tool schemas not found; falling back to English.")
            _EHR, _PAT = EHR_TOOL_SCHEMAS, PATIENT_TOOL_SCHEMAS
    else:
        _EHR, _PAT = EHR_TOOL_SCHEMAS, PATIENT_TOOL_SCHEMAS

    schema_by_name = {s["function"]["name"]: s for s in _EHR}
    schema_by_name.update({s["function"]["name"]: s for s in _PAT})

    names = TASK_TOOLS.get(task_domain, list(schema_by_name.keys()))
    return [schema_by_name[n] for n in names if n in schema_by_name]
