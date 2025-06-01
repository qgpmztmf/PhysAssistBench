"""
tools/locales/tool_schemas_en.py — English tool schemas.

All tool names (identifiers) are language-agnostic.
Only description strings are localised here.
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
            "name": "ask_user_for_required_parameters",
            "description": "Call this when the user's request is missing required information that cannot be inferred from the conversation history or EHR data. Ask the clinician to provide the missing details before proceeding with any tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the EHR tool that requires the missing parameter(s)"
                    },
                    "missing_required_parameters": {
                        "type": "array",
                        "description": "List of required parameter names that are missing from the user's request",
                        "items": {"type": "string"}
                    }
                },
                "required": ["tool_name", "missing_required_parameters"]
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
