"""
Central data-path resolution for PhysAssistBench.

Both roots can be overridden via environment variables; defaults point
inside the repository so a checkout with bundled sample data works out
of the box.

  MIMIC_PATIENT_ROOT  per-patient MIMIC-IV CSVs, one directory per subject_id:
                      <root>/<subject_id>/<table>.csv.csv.gz
  MIMIC_REF_ROOT      MIMIC-IV reference tables (d_labitems, d_icd_diagnoses,
                      d_items) in the standard layout: <root>/hosp/..., <root>/icu/...
"""

import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_PKG_DIR)

MIMIC_PATIENT_ROOT = os.environ.get(
    "MIMIC_PATIENT_ROOT",
    os.path.join(REPO_ROOT, "raw_data", "MIMIC-IV-split-by-patient", "split_data_each_patient"),
)

MIMIC_REF_ROOT = os.environ.get(
    "MIMIC_REF_ROOT",
    os.path.join(REPO_ROOT, "raw_data", "mimiciv"),
)

# Repo-root .env (DEEPSEEK_API_KEY etc.)
ENV_PATH = os.path.join(REPO_ROOT, ".env")
