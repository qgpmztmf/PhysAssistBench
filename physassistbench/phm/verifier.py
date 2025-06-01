"""
Phase 5 — Fact Verification (Hard Rules only, fully deterministic).

Every LLM-generated PHM node is cross-validated against structured MIMIC-IV
tables. Nodes that cannot be verified are removed with an audit log entry.
No LLM is used in this phase — the goal is hallucination prevention.
"""
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from .schema import PHM, Medication, Diagnosis, WarningSign
from .extractor import fuzzy_drug_match, align_with_icd

log = logging.getLogger(__name__)


class VerificationLog:
    """Accumulates audit records for all verification decisions."""

    def __init__(self):
        self.removed: list[dict] = []
        self.verified: list[dict] = []
        self.warnings: list[dict] = []

    def log_removed(self, node_type: str, identifier: str, reason: str):
        entry = {"action": "removed", "type": node_type, "id": identifier, "reason": reason}
        self.removed.append(entry)
        log.info("PHM verification REMOVED %s '%s': %s", node_type, identifier, reason)

    def log_verified(self, node_type: str, identifier: str, evidence: str):
        entry = {"action": "verified", "type": node_type, "id": identifier, "evidence": evidence}
        self.verified.append(entry)

    def log_warning(self, node_type: str, identifier: str, message: str):
        entry = {"action": "warning", "type": node_type, "id": identifier, "message": message}
        self.warnings.append(entry)
        log.warning("PHM verification WARNING %s '%s': %s", node_type, identifier, message)

    def summary(self) -> dict:
        return {
            "removed_count": len(self.removed),
            "verified_count": len(self.verified),
            "warning_count": len(self.warnings),
            "removed": self.removed,
            "warnings": self.warnings,
        }


def verify_medications(
    medications: list[Medication],
    prescriptions_df: Optional[pd.DataFrame],
    vlog: VerificationLog,
) -> list[Medication]:
    """
    Check 1: Every medication must have a matching record in hosp_prescriptions.
    Medications without a match are removed.
    """
    if prescriptions_df is None or prescriptions_df.empty:
        vlog.log_warning("medication", "ALL", "prescriptions table unavailable — skipping medication verification")
        return medications

    verified: list[Medication] = []
    for med in medications:
        match = fuzzy_drug_match(med.drug, prescriptions_df)
        if match:
            vlog.log_verified("medication", med.drug, f"matched prescription: {match.get('drug', '')}")
            verified.append(med)
        else:
            vlog.log_removed("medication", med.drug, "not_in_prescriptions")
    return verified


def verify_diagnoses(
    diagnoses: list[Diagnosis],
    diagnoses_icd_df: Optional[pd.DataFrame],
    vlog: VerificationLog,
) -> list[Diagnosis]:
    """
    Check 2: Every diagnosis should align with hosp_diagnoses_icd.
    Diagnoses that cannot be aligned get a warning but are NOT removed
    (free-text diagnoses may not always have ICD codes in the table).
    """
    if diagnoses_icd_df is None or diagnoses_icd_df.empty:
        vlog.log_warning("diagnosis", "ALL", "diagnoses_icd table unavailable — skipping alignment")
        return diagnoses

    for diag in diagnoses:
        icd_title = align_with_icd(diag.medical_term, diagnoses_icd_df)
        if icd_title:
            vlog.log_verified("diagnosis", diag.medical_term, f"aligned to ICD: {icd_title}")
            if not diag.source:
                diag.source = f"icd_aligned:{icd_title}"
        else:
            vlog.log_warning("diagnosis", diag.medical_term, "no ICD alignment found — kept with warning")
    return diagnoses


def verify_warning_sign_thresholds(
    warning_signs: list[WarningSign],
    labevents_df: Optional[pd.DataFrame],
    vlog: VerificationLog,
) -> list[WarningSign]:
    """
    Check 3: Warning sign numeric thresholds should derive from actual observed
    lab values, not textbook values.
    Thresholds that cannot be traced to labevents get a warning flag.
    """
    if labevents_df is None or labevents_df.empty:
        vlog.log_warning("warning_sign", "ALL", "labevents table unavailable — skipping threshold verification")
        return warning_signs

    import re
    # Extract numeric values from condition strings, e.g. "K > 5.5 mEq/L"
    _num_pattern = re.compile(r"[\d.]+")

    for ws in warning_signs:
        nums = _num_pattern.findall(ws.condition)
        if not nums:
            continue  # symptom-based condition, no threshold to verify

        # Check whether any lab observation is near this threshold
        if "valuenum" not in labevents_df.columns:
            continue
        observed_vals = labevents_df["valuenum"].dropna()
        for num_str in nums:
            threshold = float(num_str)
            # Accept if any observed value is within 20% of the threshold
            if any(abs(v - threshold) / (threshold + 1e-9) < 0.20 for v in observed_vals):
                vlog.log_verified("warning_sign", ws.condition, f"threshold {threshold} traceable to labevents")
                break
        else:
            vlog.log_warning(
                "warning_sign", ws.condition,
                f"threshold(s) {nums} not traceable to actual lab observations — may be textbook value"
            )
    return warning_signs


def check_source_quotes(
    phm: PHM,
    sections: dict[str, str],
    vlog: VerificationLog,
) -> None:
    """
    Check 4: Verify that adherence_evidence quotes for medications actually
    appear verbatim (or near-verbatim) in the discharge note sections.
    """
    full_text = " ".join(sections.values()).lower()

    for med in phm.medications:
        if not med.adherence_evidence:
            continue
        # Take the first 60 chars of evidence as a probe
        probe = med.adherence_evidence[:60].lower().strip()
        # Remove leading quote chars
        probe = probe.lstrip('"').strip()
        if probe and probe not in full_text:
            vlog.log_warning(
                "medication", med.drug,
                f"adherence_evidence quote not found verbatim in note: '{probe[:40]}...'"
            )
        else:
            vlog.log_verified("medication", med.drug, "adherence_evidence traceable to note text")


def verify_phm(
    phm: PHM,
    mimic_tables: dict[str, Optional[pd.DataFrame]],
    sections: dict[str, str],
) -> tuple[PHM, VerificationLog]:
    """
    Run all five verification checks on a PHM instance.

    mimic_tables should contain:
        'prescriptions'  -> hosp_prescriptions DataFrame
        'diagnoses_icd'  -> hosp_diagnoses_icd DataFrame
        'labevents'      -> hosp_labevents DataFrame

    Returns (verified_phm, verification_log).
    """
    vlog = VerificationLog()

    phm.medications = verify_medications(
        phm.medications,
        mimic_tables.get("prescriptions"),
        vlog,
    )

    phm.diagnoses = verify_diagnoses(
        phm.diagnoses,
        mimic_tables.get("diagnoses_icd"),
        vlog,
    )

    phm.warning_signs = verify_warning_sign_thresholds(
        phm.warning_signs,
        mimic_tables.get("labevents"),
        vlog,
    )

    check_source_quotes(phm, sections, vlog)

    log.info(
        "PHM verification complete for subject_id=%s: %d removed, %d verified, %d warnings",
        phm.subject_id,
        len(vlog.removed),
        len(vlog.verified),
        len(vlog.warnings),
    )
    return phm, vlog
