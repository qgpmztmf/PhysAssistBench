"""
PHM Builder — orchestrates all five phases of the offline build pipeline.

Usage:
    from phm import PHMBuilder
    builder = PHMBuilder(data_root="/path/to/split_data_each_patient")
    phm = builder.build(subject_id=10000032)
    builder.save_yaml(phm, "PHM_10000032.yaml")
"""
from __future__ import annotations
import gzip
import logging
import os
from typing import Optional

import pandas as pd
import yaml

from .schema import PHM, Diagnosis, Medication, WarningSign, LabTrend, Persona
from .section_parser import parse_sections, extract_patient_quotes
from .extractor import (
    parse_medication_list,
    parse_diagnosis_list,
    extract_adherence_evidence,
    extract_all_adherence_evidence,
    build_lab_trends,
    fuzzy_drug_match,
)
from .persona import build_persona, classify_drug_adherence
from .llm_generator import (
    generate_patient_term,
    generate_patient_explanation,
    generate_patient_signal,
    generate_patient_impact,
    detect_info_withheld,
)
from .verifier import verify_phm

log = logging.getLogger(__name__)

# Critical lab panels that should always generate warning signs
_CRITICAL_LAB_WARNING_RULES: list[dict] = [
    {
        "item_keywords": ["sodium", "Na"],
        "medical_condition": "Hyponatremia",
        "low_threshold": 125,
        "direction": "low",
        "action": "Go to the emergency room the same day — very low sodium can cause seizures and confusion.",
        "urgency_level": 1,
    },
    {
        "item_keywords": ["potassium"],
        "medical_condition": "Hyperkalemia",
        "high_threshold": 5.5,
        "direction": "high",
        "action": "Call your doctor immediately or go to the ER — high potassium can affect your heartbeat.",
        "urgency_level": 1,
    },
    {
        "item_keywords": ["ammonia", "NH3"],
        "medical_condition": "Hepatic Encephalopathy",
        "high_threshold": 100,
        "direction": "high",
        "action": "Call 911 immediately if someone around you notices you are confused or not making sense.",
        "urgency_level": 1,
    },
    {
        "item_keywords": ["creatinine"],
        "medical_condition": "Acute Kidney Injury",
        "high_threshold": 2.0,
        "direction": "high",
        "action": "Contact your doctor the same day — your kidneys may be under stress.",
        "urgency_level": 2,
    },
]


class PHMBuilder:
    """
    Builds a Patient Health Memory YAML from MIMIC-IV per-patient data.

    Data root should be the split_data_each_patient directory where each
    subdirectory is named by subject_id and contains {table}.csv.csv.gz files.
    """

    def __init__(self, data_root: str):
        self.data_root = data_root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, subject_id: int) -> PHM:
        """Full five-phase pipeline. Returns a verified PHM instance."""
        log.info("Building PHM for subject_id=%d", subject_id)

        # Load all required MIMIC tables
        tables = self._load_tables(subject_id)

        # ── Phase 1: Parse all discharge notes ───────────────────────
        notes_df = tables.get("notes")
        if notes_df is None or notes_df.empty:
            log.warning("No discharge notes for subject_id=%d — PHM will be sparse", subject_id)
            notes_df = pd.DataFrame(columns=["hadm_id", "charttime", "text"])

        # Step 1a: Select anchor admission
        anchor_hadm_id = self._select_anchor(tables.get("admissions"), notes_df)
        log.info("Anchor hadm_id=%s", anchor_hadm_id)

        # Step 1b: Parse sections from all notes, using anchor note as primary
        all_sections_by_hadm = self._parse_all_notes(notes_df)
        primary_sections = all_sections_by_hadm.get(anchor_hadm_id, {})
        # Merge sections from all admissions for trajectory context
        merged_hpi = self._merge_hpi(all_sections_by_hadm)
        patient_quotes = extract_patient_quotes(merged_hpi)

        # ── Phase 2: Structured extraction ───────────────────────────
        raw_diagnoses = parse_diagnosis_list(primary_sections.get("dc_diagnosis", ""))
        raw_medications = parse_medication_list(primary_sections.get("medications_dc", ""))
        lab_trends = build_lab_trends(tables.get("labevents"))

        # ── Phase 3 + 4: LLM generation + persona ────────────────────
        dc_instructions = primary_sections.get("dc_instructions", "")

        diagnoses = self._build_diagnosis_nodes(
            raw_diagnoses, dc_instructions, patient_quotes,
            tables.get("diagnoses_icd"), all_sections_by_hadm,
        )

        medications = self._build_medication_nodes(
            raw_medications, merged_hpi, tables.get("prescriptions"),
        )

        warning_signs = self._build_warning_signs(lab_trends, diagnoses, dc_instructions)

        # Persona
        info_completeness = self._determine_info_completeness(merged_hpi)
        persona = build_persona(merged_hpi, patient_quotes, info_completeness)

        # Open questions (from patient quotes that contain question markers)
        open_questions = self._extract_open_questions(patient_quotes)

        phm = PHM(
            subject_id=subject_id,
            anchor_hadm_id=anchor_hadm_id,
            diagnoses=diagnoses,
            medications=medications,
            warning_signs=warning_signs,
            lab_trends=lab_trends,
            symptom_log=[],
            open_questions=open_questions,
            persona=persona,
        )

        # ── Phase 5: Fact verification ────────────────────────────────
        phm, vlog = verify_phm(phm, tables, primary_sections)
        log.info("Verification summary: %s", vlog.summary())

        return phm

    def save_yaml(self, phm: PHM, output_path: str) -> None:
        """Serialise a PHM instance to YAML."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(phm.to_dict(), f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        log.info("PHM saved to %s", output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _patient_dir(self, subject_id: int) -> str:
        return os.path.join(self.data_root, str(subject_id))

    def _read_table(self, subject_id: int, table: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self._patient_dir(subject_id), f"{table}.csv.csv.gz")
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, compression="gzip", low_memory=False)
        except Exception as e:
            log.warning("Could not read %s: %s", path, e)
            return None

    def _load_tables(self, subject_id: int) -> dict[str, Optional[pd.DataFrame]]:
        return {
            "admissions":   self._read_table(subject_id, "hosp_admissions"),
            "diagnoses_icd":self._read_table(subject_id, "hosp_diagnoses_icd"),
            "prescriptions":self._read_table(subject_id, "hosp_prescriptions"),
            "labevents":    self._read_table(subject_id, "hosp_labevents"),
            "notes":        self._read_table(subject_id, "note_discharge"),
        }

    def _select_anchor(
        self,
        admissions_df: Optional[pd.DataFrame],
        notes_df: pd.DataFrame,
    ) -> Optional[int]:
        """
        Select the anchor hadm_id.
        Priority: admission with the most notes / most text → most information-rich.
        Falls back to the most recent admission.
        """
        if notes_df.empty:
            if admissions_df is not None and not admissions_df.empty and "hadm_id" in admissions_df.columns:
                return int(admissions_df.sort_values("admittime").iloc[-1]["hadm_id"])
            return None

        # Score by text length per hadm_id
        if "hadm_id" in notes_df.columns and "text" in notes_df.columns:
            score = (
                notes_df.dropna(subset=["hadm_id", "text"])
                .assign(text_len=notes_df["text"].str.len())
                .groupby("hadm_id")["text_len"]
                .sum()
            )
            if not score.empty:
                return int(score.idxmax())

        # Fallback: most recent note
        if "charttime" in notes_df.columns:
            return int(notes_df.sort_values("charttime").iloc[-1].get("hadm_id", 0))
        return None

    def _parse_all_notes(self, notes_df: pd.DataFrame) -> dict[Optional[int], dict[str, str]]:
        """Parse sections from each discharge note, keyed by hadm_id."""
        result: dict[Optional[int], dict[str, str]] = {}
        if notes_df.empty or "text" not in notes_df.columns:
            return result
        for _, row in notes_df.iterrows():
            hadm_id = row.get("hadm_id")
            text = str(row.get("text", ""))
            if text:
                result[hadm_id] = parse_sections(text)
        return result

    def _merge_hpi(self, all_sections: dict) -> str:
        """Concatenate HPI sections from all admissions for adherence trajectory."""
        parts = []
        for hadm_id, sections in all_sections.items():
            hpi = sections.get("hpi", "")
            if hpi:
                parts.append(f"[hadm={hadm_id}] {hpi}")
        return "\n\n".join(parts)

    def _build_diagnosis_nodes(
        self,
        raw_diagnoses: list[str],
        dc_instructions: str,
        patient_quotes: list[str],
        diagnoses_icd_df: Optional[pd.DataFrame],
        all_sections: dict,
    ) -> list[Diagnosis]:
        nodes: list[Diagnosis] = []
        for raw in raw_diagnoses[:15]:   # cap at 15 to limit LLM calls
            patient_term = generate_patient_term(raw, dc_instructions, patient_quotes)
            patient_impact = generate_patient_impact(raw, patient_term, dc_instructions)
            nodes.append(Diagnosis(
                medical_term=raw,
                patient_term=patient_term,
                patient_impact=patient_impact,
                severity_self_assessment="",    # filled at runtime by Symptom Interview Engine
                source="discharge_note:dc_diagnosis",
                trajectory="",
            ))
        return nodes

    def _build_medication_nodes(
        self,
        raw_meds: list[dict],
        merged_hpi: str,
        prescriptions_df: Optional[pd.DataFrame],
    ) -> list[Medication]:
        nodes: list[Medication] = []
        # Pre-extract all adherence signals from HPI
        all_adh = extract_all_adherence_evidence(merged_hpi)

        for raw in raw_meds:
            drug_name = raw["drug"]
            dose_str = f"{raw['dose_value']} {raw['dose_unit']}".strip()
            full_drug = f"{drug_name} {dose_str} {raw['route']} {raw['frequency']}".strip()

            # Per-drug adherence from HPI
            adh_info = extract_adherence_evidence(merged_hpi, drug_name)
            adh_type = adh_info["adherence_type"]
            evidence_quotes = adh_info["evidence_quotes"]
            evidence_str = "; ".join(evidence_quotes[:3])

            # Generate patient explanation if we have evidence quotes
            explanation = ""
            if evidence_quotes:
                explanation = generate_patient_explanation(
                    extracted_quote=evidence_quotes[0],
                    drug_name=drug_name,
                    adherence_type=adh_type,
                )

            # Determine current_status
            status_map = {
                "good":             "taking",
                "poor":             "not_taking",
                "never_filled":     "not_taking",
                "ran_out":          "ran_out",
                "self_discontinued":"not_taking",
                "unknown":          "unknown",
            }
            current_status = status_map.get(adh_type, "unknown")

            # critical_flag: any confirmed non-adherence warrants clinical attention
            critical_flag = adh_type in ("never_filled", "self_discontinued", "poor")

            nodes.append(Medication(
                drug=full_drug,
                drug_patient_term="",   # LLM generation deferred — use drug name for now
                indication="",          # populated in future from drug-indication mapping
                adherence=adh_type,
                adherence_evidence=evidence_str,
                patient_explanation=explanation,
                current_status=current_status,
                critical_flag=critical_flag,
            ))
        return nodes

    def _build_warning_signs(
        self,
        lab_trends: list[LabTrend],
        diagnoses: list[Diagnosis],
        dc_instructions: str,
    ) -> list[WarningSign]:
        """
        Generate warning signs from:
        1. Observed abnormal lab values (hard rules on thresholds)
        2. Diagnosis-based symptom warning signs
        """
        signs: list[WarningSign] = []
        trend_by_name = {t.item_name.lower(): t for t in lab_trends}

        for rule in _CRITICAL_LAB_WARNING_RULES:
            for kw in rule["item_keywords"]:
                trend = trend_by_name.get(kw.lower())
                if trend is None:
                    continue

                # Check whether any observed value breaches the threshold
                values = [r["valuenum"] for r in trend.recent_values if r["valuenum"] is not None]
                if not values:
                    continue

                breached = False
                condition_str = ""
                if rule.get("direction") == "low":
                    thresh = rule["low_threshold"]
                    breached = min(values) < thresh
                    condition_str = f"{kw} < {thresh} {trend.unit} (observed min: {min(values):.1f})"
                elif rule.get("direction") == "high":
                    thresh = rule["high_threshold"]
                    breached = max(values) > thresh
                    condition_str = f"{kw} > {thresh} {trend.unit} (observed max: {max(values):.1f})"

                if not breached:
                    continue

                # Generate patient_signal via LLM
                patient_signal = generate_patient_signal(
                    condition=condition_str,
                    medical_condition=rule["medical_condition"],
                    lab_context=f"Recent values: {values[:3]}",
                )

                signs.append(WarningSign(
                    condition=condition_str,
                    medical_condition=rule["medical_condition"],
                    patient_signal=patient_signal,
                    action=rule["action"],
                    urgency_level=rule["urgency_level"],
                    contributing_factor="",
                    critical_flag=(rule["urgency_level"] == 1),
                ))
                break   # Only one warning per rule

        return signs

    def _determine_info_completeness(self, merged_hpi: str) -> str:
        """
        Hard-rule pre-detection of contradiction candidates, confirmed by LLM.
        Returns 'critical_withheld', 'partial', or 'full'.
        """
        import re
        # Hard-rule: look for juxtaposed compliance claim + non-adherence evidence
        compliance_claim = re.search(
            r"(patient states?|she states?|he states?|they state?)\s+.{0,60}(compliant|taking all|adherent)",
            merged_hpi, re.IGNORECASE
        )
        non_adherence = re.search(
            r"(never filled|self[- ]discontinu|not been taking|ran out|stopped taking)",
            merged_hpi, re.IGNORECASE
        )

        if compliance_claim and non_adherence:
            # Extract context around each match for LLM confirmation
            p_start = max(0, compliance_claim.start() - 20)
            d_start = max(0, non_adherence.start() - 20)
            patient_stmt = merged_hpi[p_start: compliance_claim.end() + 80]
            doctor_obs   = merged_hpi[d_start: non_adherence.end() + 80]
            result = detect_info_withheld(patient_stmt, doctor_obs)
            if result.get("is_withheld"):
                return "critical_withheld"

        if non_adherence:
            return "partial"

        return "full"

    def _extract_open_questions(self, patient_quotes: list[str]) -> list[str]:
        """Extract patient questions from their direct quotes."""
        questions = []
        for q in patient_quotes:
            if "?" in q or any(
                q.lower().startswith(w)
                for w in ("why", "what", "how", "when", "can i", "will i", "am i", "is it")
            ):
                questions.append(q)
        return questions[:5]   # cap at 5


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build PHM YAML for a MIMIC-IV patient")
    parser.add_argument("subject_id", type=int, help="MIMIC-IV subject_id")
    parser.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "MIMIC-IV-split-by-patient", "split_data_each_patient"),
        help="Path to split_data_each_patient directory",
    )
    parser.add_argument("--output-dir", default=".", help="Directory to write PHM YAML")
    args = parser.parse_args()

    builder = PHMBuilder(data_root=args.data_root)
    phm = builder.build(args.subject_id)
    out_path = os.path.join(args.output_dir, f"PHM_{args.subject_id}.yaml")
    builder.save_yaml(phm, out_path)
    print(f"PHM written to {out_path}")
