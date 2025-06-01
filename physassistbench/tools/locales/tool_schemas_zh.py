"""
tools/locales/tool_schemas_zh.py — Chinese tool schemas.

Tool names (identifiers) are identical to the English version.
Only description strings are translated to Chinese.
"""

EHR_TOOL_SCHEMAS = [
    # ─── 第一组：住院 / 通用 ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_patient_info",
            "description": "返回患者的人口学信息，包括性别、年龄及死亡日期（如适用）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_admissions",
            "description": "返回患者的所有住院记录，包括入院/出院时间、住院类型、地点及保险信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_admission_details",
            "description": "返回特定住院记录的详细信息，包括出院去向及死亡标志。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符"}
                },
                "required": ["subject_id", "hadm_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_diagnoses",
            "description": "返回患者的ICD诊断编码。若提供hadm_id，则仅返回该次住院的诊断。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_results",
            "description": "返回实验室检测结果，含参考范围及异常标志。可按住院次、检测名称或仅异常值进行筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"},
                    "item_name": {"type": "string", "description": "筛选的检测名称（可选，如 'Potassium'、'Creatinine'）"},
                    "abnormal_only": {"type": "boolean", "description": "若为true，仅返回异常结果（默认false）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_trends",
            "description": "返回指定实验室检测项目最近N次结果，用于展示趋势变化。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "item_name": {"type": "string", "description": "实验室检测名称（如 'Creatinine'、'Hemoglobin'）"},
                    "n_recent": {"type": "integer", "description": "返回最近结果的数量（默认5）"}
                },
                "required": ["subject_id", "item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_microbiology_results",
            "description": "返回微生物培养结果，包括病原体鉴定、抗生素敏感性及标本类型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_prescriptions",
            "description": "返回药物处方记录，包含药物名称、剂量、给药途径及频次。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_medication_administration",
            "description": "返回药物给药记录（eMAR），显示实际给药时间、剂量及漏服情况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"},
                    "medication": {"type": "string", "description": "筛选的药物名称（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_procedures",
            "description": "返回住院期间执行的ICD手术操作编码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_drg_info",
            "description": "返回住院的诊断相关组（DRG）编码，含严重程度及死亡率权重。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_history",
            "description": "返回住院期间的科室转入记录（如内科、外科、心脏科），包含转科时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    # ─── 第二组：ICU ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_icu_stays",
            "description": "返回所有ICU住院记录，包括护理单元、入/出ICU时间及住院时长。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_icu_vitals",
            "description": "返回ICU图表事件（生命体征、呼吸机参数、神经系统评估）。可按住院次或生命体征名称筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "stay_id": {"type": "integer", "description": "ICU住院标识符（可选）"},
                    "vital_name": {"type": "string", "description": "生命体征名称筛选（可选，如 'Heart Rate'、'Blood Pressure'）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_icu_fluids_in",
            "description": "返回ICU液体及药物输入事件（静脉液体、血管加压药、营养液）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "stay_id": {"type": "integer", "description": "ICU住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_icu_output",
            "description": "返回ICU液体输出事件（尿量、引流量）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "stay_id": {"type": "integer", "description": "ICU住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    # ─── 第三组：急诊科 ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_ed_visits",
            "description": "返回所有急诊就诊记录，包含就诊转运方式及处置结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ed_triage",
            "description": "返回急诊分诊评估，包括生命体征及主诉。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "stay_id": {"type": "integer", "description": "急诊住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ed_vital_signs",
            "description": "返回急诊就诊期间的时序生命体征测量数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "stay_id": {"type": "integer", "description": "急诊住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ed_diagnoses",
            "description": "返回急诊就诊期间分配的ICD诊断编码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "stay_id": {"type": "integer", "description": "急诊住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ed_medications",
            "description": "返回急诊药物核对记录及Pyxis取药事件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "stay_id": {"type": "integer", "description": "急诊住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    # ─── 第四组：临床记录 ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_discharge_summary",
            "description": "返回住院出院小结的完整文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_discharge_section",
            "description": "返回出院小结中的特定章节（如'评估与计划'、'入院用药'、'出院情况'）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"},
                    "section_name": {"type": "string", "description": "需提取的章节标题（如 'Assessment and Plan'、'Physical Exam'、'Discharge Medications'）"}
                },
                "required": ["subject_id", "section_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_radiology_report",
            "description": "返回放射科报告文本（如胸部X光、CT扫描结果）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "hadm_id": {"type": "integer", "description": "住院标识符（可选）"},
                    "report_type": {"type": "string", "description": "筛选的放射科报告类型（可选，如 'CHEST'、'CT'、'MRI'）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "在所有临床记录（出院小结及放射科报告）中检索指定关键词或短语。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "keyword": {"type": "string", "description": "在临床记录中检索的关键词或短语"}
                },
                "required": ["subject_id", "keyword"]
            }
        }
    },
    # ─── 第五组：工具类 ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_vital_signs_outpatient",
            "description": "返回门诊生命体征及测量数据（血压、体重、BMI），来源于OMR表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_timeline",
            "description": "返回患者所有临床事件（住院、ICU入住、急诊就诊、诊断）的时间轴。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "event_type": {"type": "string", "description": "按事件类型筛选：'admission'（入院）、'discharge'（出院）、'icu_admit'（ICU入住）、'ed_visit'（急诊就诊）、'diagnosis'（诊断）（可选）"}
                },
                "required": ["subject_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_for_required_parameters",
            "description": "当用户请求缺少必要信息，且无法从对话历史或EHR数据中推断时，调用此工具。在执行任何工具调用前，先请临床医生提供缺失的信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "需要缺失参数的EHR工具名称"
                    },
                    "missing_required_parameters": {
                        "type": "array",
                        "description": "用户请求中缺失的必填参数名称列表",
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
            "description": "表示所有必要的工具调用已完成，准备向用户提供最终答案。在所有所需数据获取完毕后调用此工具。",
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
            "description": "询问患者今天就诊的原因。返回患者以自然语言口述的主诉。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "session_id": {"type": "string", "description": "会话标识符（在env_info中提供）"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_symptom_history",
            "description": "询问患者描述其症状史。可通过query参数指定症状重点。返回OPQRST格式的自然语言描述。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "session_id": {"type": "string", "description": "会话标识符（在env_info中提供）"},
                    "query": {"type": "string", "description": "可选的症状重点（如 'abdominal pain'、'shortness of breath'）"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_medication_adherence",
            "description": "询问患者是否按医嘱服用某种药物。返回患者关于依从性的自然语言自我报告。对于critical_withheld患者，具体说出药物名称可能揭示其隐瞒的不依从信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "session_id": {"type": "string", "description": "会话标识符（在env_info中提供）"},
                    "drug": {"type": "string", "description": "询问的药物名称（如 'Lactulose'、'Insulin'、'Metformin'）"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_social_history",
            "description": "询问患者的社会史，包括居住情况、职业、习惯（吸烟、饮酒）及支持系统。返回自然语言回答。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "session_id": {"type": "string", "description": "会话标识符（在env_info中提供）"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_functional_status",
            "description": "询问患者的功能状态及日常活动能力。返回对任何限制的自然语言描述。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "session_id": {"type": "string", "description": "会话标识符（在env_info中提供）"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patient.get_pain_assessment",
            "description": "询问患者描述其疼痛情况（位置、严重程度0-10分、性质、发作时间、放射方向）。返回自然语言疼痛评估。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer", "description": "MIMIC-IV患者标识符"},
                    "session_id": {"type": "string", "description": "会话标识符（在env_info中提供）"},
                },
                "required": ["subject_id", "session_id"],
            },
        },
    },
]
