#!/usr/bin/env bash
#
# smoke_test.sh — End-to-end check: generate one benchmark session, then
# evaluate a doctor-agent LLM on it. All LLM calls use DeepSeek V4 Flash.
#
# Prerequisites:
#   1. uv sync                       (install dependencies)
#   2. A repo-root .env containing:
#          DEEPSEEK_API_KEY=sk-...
#          DEEPSEEK_BASE_URL=https://api.deepseek.com
#      (The rubric judge falls back to DeepSeek when GATEWAY_OPENAI_API_KEY is unset,
#       so DEEPSEEK_API_KEY alone is enough to run this smoke test.)
#   3. MIMIC-IV patient data under raw_data/MIMIC-IV-split-by-patient/ — a small
#      sample is bundled; see README and scripts/split_mimic_by_subject.py.
#
# Usage:
#   scripts/smoke_test.sh              # generate 1 session, then evaluate it
#   scripts/smoke_test.sh example      # skip generation; evaluate the bundled example
#
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

MODE="${1:-generate}"
MODEL="deepseek-v4-flash"
# Curated data-rich patient that reliably yields a coherent med_safety session
# (apixaban renal-dose-adjustment thread). Override with SMOKE_SUBJECT_ID=<id>.
SUBJECT_ID="${SMOKE_SUBJECT_ID:-10458345}"
DIFFICULTY="${SMOKE_DIFFICULTY:-2}"

# Load DEEPSEEK_API_KEY from .env if not already exported
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f .env ]; then
  set -a; . ./.env; set +a
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set. Add it to .env at the repo root." >&2
  exit 1
fi

if [ "$MODE" = "example" ]; then
  DATA_DIR="physassistbench/data_smoke"
  mkdir -p "$DATA_DIR"
  cp physassistbench/examples/med_safety_example.jsonl "$DATA_DIR/med_safety.jsonl"
  echo ">> Using bundled example: $DATA_DIR/med_safety.jsonl (generation skipped)"
else
  DATA_DIR="physassistbench/data_smoke"
  echo ">> Step 1/2: generating one med_safety session for subject $SUBJECT_ID (DeepSeek V4 Flash) ..."
  uv run python physassistbench/generate_all.py \
    --scenarios med_safety --n 1 --subject_id "$SUBJECT_ID" \
    --difficulty "$DIFFICULTY" --out_dir "$DATA_DIR"
fi

echo ">> Step 2/2: evaluating $MODEL on the session ..."
uv run python physassistbench/run_eval.py \
  --model "$MODEL" --language en --skip_generate \
  --data_dir "$DATA_DIR" --output_dir physassistbench/results_smoke

echo ""
echo ">> Smoke test complete."
echo "   Data:    $DATA_DIR/med_safety.jsonl"
echo "   Results: physassistbench/results_smoke/ (eval_report.txt, metrics_summary.json)"
