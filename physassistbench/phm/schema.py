"""
PHM data schema — dataclasses matching the YAML structure described in patient_agent_en.md.
All fields use plain Python types so yaml.dump works without custom representers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Diagnosis:
    medical_term: str = ""
    patient_term: str = ""
    patient_impact: str = ""
    severity_self_assessment: str = ""
    source: str = ""
    trajectory: str = ""           # stable / worsening / rapidly_decompensating


@dataclass
class Medication:
    drug: str = ""
    drug_patient_term: str = ""
    indication: str = ""
    adherence: str = "unknown"     # good / poor / never_filled / ran_out / self_discontinued / unknown
    adherence_evidence: str = ""   # verbatim HPI quotes
    patient_explanation: str = ""  # first-person explanation (LLM-generated, quote-anchored)
    current_status: str = ""       # taking / not_taking / ran_out / unknown
    critical_flag: bool = False


@dataclass
class WarningSign:
    condition: str = ""            # numeric threshold or symptom combo (hard-rule derived)
    medical_condition: str = ""
    patient_signal: str = ""       # LLM-generated patient-perceivable description
    action: str = ""               # tiered response instruction
    urgency_level: int = 2         # 1=immediate ER, 2=same-day, 3=routine
    contributing_factor: str = ""
    critical_flag: bool = False


@dataclass
class LabTrend:
    item_name: str = ""
    itemid: Optional[int] = None
    unit: str = ""
    recent_values: list = field(default_factory=list)   # list of {charttime, value, flag}
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    trend_direction: str = ""      # improving / worsening / stable / fluctuating


@dataclass
class Persona:
    health_literacy: str = "medium"          # high / medium / low
    anxiety_level: str = "medium"            # high / medium / low
    info_completeness: str = "full"          # full / partial / critical_withheld
    self_diagnosis_tendency: str = "absent"  # present / absent
    # per-medication adherence is stored in Medication.adherence
    # overall adherence summary:
    overall_adherence: str = "unknown"       # good / poor / mixed / unknown


@dataclass
class PHM:
    subject_id: int = 0
    anchor_hadm_id: Optional[int] = None
    diagnoses: list[Diagnosis] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    warning_signs: list[WarningSign] = field(default_factory=list)
    lab_trends: list[LabTrend] = field(default_factory=list)
    symptom_log: list = field(default_factory=list)   # populated at runtime
    open_questions: list[str] = field(default_factory=list)
    persona: Persona = field(default_factory=Persona)

    def to_dict(self) -> dict:
        """Return a plain dict ready for yaml.dump."""
        import dataclasses
        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj
        return _convert(self)
