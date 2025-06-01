# PhysAssistBench

Code for **"Are LLMs Ready to Assist Physicians? PhysAssistBench for Interactive
Doctor-Patient-EHR Assistance"**.

PhysAssistBench evaluates LLMs as physician assistants in multi-turn clinical
sessions built from real MIMIC-IV admissions: the model must resolve implicit
physician queries, call FHIR R4 EHR tools against real patient data, interview a
simulated patient, and integrate the evidence into physician-facing answers.

- 324 sessions × 4 turns, English + Chinese
- 4 clinical scenarios: diagnostic workup, medication safety, treatment response, discharge planning
- 4 task types: Information Lookup / Data Gathering / Clinical Reasoning / Write-Update
- 3 implicitness types: Nominal Anaphora / Predicate Ellipsis / Abstract Event Anaphora
- Rubric-based LLM-judge scoring: mean Rubric Score (mRS), Pass@Turn, Pass@Session

The complete evaluation dataset lives in `physassistbench/data/`, split into the
four clinical scenarios (one JSONL file each):

```
physassistbench/data/diagnostic_workup.jsonl
physassistbench/data/med_safety.jsonl
physassistbench/data/treatment_response.jsonl
physassistbench/data/discharge_planning.jsonl
```

Raw MIMIC-IV inputs used to generate and to execute tool calls against live at the
repository root under `raw_data/` (see [Data](#data)); they are distinct from the
`physassistbench/data/` evaluation benchmark above.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` at the repo root with the LLM credentials:

```
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
# Optional: GATEWAY_OPENAI_API_KEY=... # enables the GPT-5-mini rubric judge used in the paper
                                       # (via an Azure OpenAI gateway); without it the judge falls back to DeepSeek
```

### Data

MIMIC-IV is credentialed data. The four source modules come from three separate
PhysioNet datasets — download each (a credentialed PhysioNet account and its
data-use agreement are required for all):

| Module | Download | Path used below |
|--------|----------|-----------------|
| `hosp`, `icu` | [MIMIC-IV](https://physionet.org/content/mimiciv/) | `mimiciv/3.1/hosp`, `mimiciv/3.1/icu` |
| `ed` | [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) | `mimiciv/ed-2.2/ed` |
| `note` | [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/) | `mimic-iv-note/2.2/note` |

PhysAssistBench reads patient data as one directory per subject:

- `raw_data/MIMIC-IV-split-by-patient/split_data_each_patient/<subject_id>/<module>_<table>.csv.csv.gz`
  — per-patient MIMIC-IV tables (override location with `MIMIC_PATIENT_ROOT`)
- `raw_data/mimiciv/hosp/d_labitems.csv.gz`, `raw_data/mimiciv/hosp/d_icd_diagnoses.csv.gz`,
  `raw_data/mimiciv/icu/d_items.csv.gz` — reference tables (override with `MIMIC_REF_ROOT`)

**Splitting the raw download into the per-subject layout.** After downloading the
raw MIMIC-IV CSVs from PhysioNet, use `scripts/split_mimic_by_subject.py` to
produce the `split_data_each_patient/` structure:

```bash
uv run python scripts/split_mimic_by_subject.py \
    --out_root raw_data/MIMIC-IV-split-by-patient/split_data_each_patient \
    --source hosp /path/to/mimiciv/3.1/hosp \
    --source icu  /path/to/mimiciv/3.1/icu \
    --source ed   /path/to/mimiciv/ed-2.2/ed \
    --source note /path/to/mimic-iv-note/2.2/note

# Build only a small sample (fast) by restricting subjects:
uv run python scripts/split_mimic_by_subject.py ... --subjects 10000032 10000048
```

Each source table with a `subject_id` column is split into
`<module>_<table>.csv.csv.gz` per subject; dimension tables (d_labitems,
d_icd_diagnoses, d_items) have no `subject_id` and are skipped — copy those into
`MIMIC_REF_ROOT` yourself. Splitting the full dataset is I/O-heavy (labevents and
chartevents dominate); use `--subjects` to build a working sample quickly.

## Quick smoke test

Once dependencies are installed (`uv sync`) and `.env` has `DEEPSEEK_API_KEY`,
run the end-to-end smoke test — it generates one clinical session and then
evaluates a doctor-agent LLM on it, all via DeepSeek V4 Flash:

```bash
scripts/smoke_test.sh            # generate 1 med_safety session, then evaluate it
scripts/smoke_test.sh example    # skip generation; evaluate the bundled example session
```

`example` mode evaluates `physassistbench/examples/med_safety_example.jsonl`, a
ready-made session shipped with the repo, so it runs without generating new data
(only `DEEPSEEK_API_KEY` is needed). Results are written to
`physassistbench/results_smoke/` (`eval_report.txt`, `metrics_summary.json`).

## Generate benchmark data

```bash
# Pre-filter eligible patients (writes physassistbench/qualified_patients.json)
uv run python physassistbench/prefilter_patients.py

# Generate n sessions per scenario (writes physassistbench/data_<timestamp>/<scenario>.jsonl)
uv run python physassistbench/generate_all.py --scenarios med_safety --n 1
```

## Evaluate a model

Models are declared in `physassistbench/model_configs.yaml` (model id, API base,
API-key env var, thinking mode, token budgets).

```bash
uv run python physassistbench/run_eval.py \
    --model deepseek-v4-flash \
    --language en \
    --skip_generate \
    --data_dir physassistbench/data_<timestamp>
```

Useful flags: `--language zh`, `--use_explicit` (explicit-paraphrase ablation),
`--judge_only` (re-judge existing outputs), `--resume`, `--output_dir`.

Results land in `physassistbench/results_<lang>_<model>_<timestamp>/`
(`turn_results_final.json`, `metrics_summary.json`). Aggregate with:

```bash
uv run python physassistbench/fill_main_table.py   # mRS / Pass@Turn / Pass@Session table
uv run python physassistbench/results_to_excel.py  # per-turn Excel report
```

## Unit Tests

```bash
uv run pytest physassistbench/tests
```

## License and data use

The code is released under Apache 2.0. Benchmark data derived from MIMIC-IV is
distributed via PhysioNet under its credentialed data-use agreement and must not
be redistributed here.
