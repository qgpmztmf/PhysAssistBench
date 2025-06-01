"""
Phase 1 — Text Preprocessing (Hard Rules only).

Splits a MIMIC-IV discharge note into standard sections via regex.
MIMIC-IV section headers are fixed strings, so regex matching is sufficient.
All ___ de-identification placeholders are flagged and skipped downstream.
"""
from __future__ import annotations
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Section patterns
# Each captures everything after the header up to the next capitalised header.
# The (?s) flag makes . match newlines; the lookahead stops at the next header.
# ---------------------------------------------------------------------------
# Lookahead that detects the NEXT section header (including the blank line that precedes it).
# Using the full _HEADER (with leading \n) lets the lookahead fire immediately after the last
# content character, without needing to advance past blank lines.
_HEADER = r"\n\s*\n[A-Z][A-Za-z _/&-]+:\n"   # blank-line-separated header
_LOOKAHEAD = r"(?=" + _HEADER + r"|\Z)"        # stop at next header or end of string

SECTION_PATTERNS: dict[str, str] = {
    "chief_complaint":    r"Chief Complaint:\n(.*?)" + _LOOKAHEAD,
    "hpi":               r"History of Present Illness:\n(.*?)" + _LOOKAHEAD,
    "pmh":               r"Past Medical History:\n(.*?)" + _LOOKAHEAD,
    "medications_admit": r"Medications on Admission:\n(.*?)" + _LOOKAHEAD,
    "medications_dc":    r"Discharge Medications:\n(.*?)" + _LOOKAHEAD,
    "hospital_course":   r"Brief Hospital Course:\n(.*?)" + _LOOKAHEAD,
    "dc_instructions":   r"Discharge Instructions:\n(.*?)" + _LOOKAHEAD,
    "dc_diagnosis":      r"Discharge Diagnosis:\n(.*?)" + _LOOKAHEAD,
    "social_history":    r"Social History:\n(.*?)" + _LOOKAHEAD,
    "family_history":    r"Family History:\n(.*?)" + _LOOKAHEAD,
    "physical_exam":     r"(?:Pertinent Results|Physical Exam):\n(.*?)" + _LOOKAHEAD,
    "followup":          r"Followup Instructions:\n(.*?)" + _LOOKAHEAD,
}

# Regex that matches MIMIC-IV de-identification placeholders
DEID_PATTERN = re.compile(r"_{2,}")

# Compiled section regexes (DOTALL so . matches newlines)
_COMPILED: dict[str, re.Pattern] = {
    name: re.compile(pat, re.DOTALL)
    for name, pat in SECTION_PATTERNS.items()
}


def parse_sections(note_text: str) -> dict[str, str]:
    """
    Extract all standard sections from a discharge note.

    Returns a dict {section_name: section_text}.
    Missing sections are absent from the dict (not empty strings).
    """
    sections: dict[str, str] = {}
    for name, pattern in _COMPILED.items():
        m = pattern.search(note_text)
        if m:
            text = m.group(1).strip()
            if text:
                sections[name] = text
    return sections


def extract_patient_quotes(hpi_text: str) -> list[str]:
    """
    Pull out direct patient quotes from the HPI section.
    MIMIC-IV commonly wraps quotes in double quotes or after 'states/reports/denies'.
    De-identified placeholders are excluded from results.
    """
    quotes: list[str] = []

    # Explicit double-quoted strings
    for m in re.finditer(r'"([^"]{10,300})"', hpi_text):
        q = m.group(1).strip()
        if not DEID_PATTERN.search(q):
            quotes.append(q)

    # "patient states/reports/denies/says ..." clauses
    for m in re.finditer(
        r"(?:patient|pt|she|he|they)\s+(?:states?|reports?|denies?|says?|noted|admits?)\s+(?:that\s+)?(.{10,200}?)(?:[.;]|$)",
        hpi_text, re.IGNORECASE
    ):
        clause = m.group(1).strip()
        if not DEID_PATTERN.search(clause):
            quotes.append(clause)

    return quotes


def mask_deid(text: str, placeholder: str = "[REDACTED]") -> str:
    """Replace ___ placeholders with a readable token for LLM prompts."""
    return DEID_PATTERN.sub(placeholder, text)


def has_deid(text: str) -> bool:
    """Return True if the text contains a de-identification placeholder."""
    return bool(DEID_PATTERN.search(text))


def parse_all_notes(notes: list[str]) -> list[dict[str, str]]:
    """
    Parse a list of discharge note texts (one per admission).
    Returns a list of section dicts, preserving order.
    """
    return [parse_sections(n) for n in notes]
