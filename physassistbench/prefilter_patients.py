"""
physassistbench/prefilter_patients.py — Offline patient pre-screening script.

Scans the full MIMIC-IV patient pool (364k patients) using two-stage filtering
for each scenario × difficulty combination, then saves a qualified candidate
list to physassistbench/qualified_patients.json.

Run this once before generation (or whenever filtering criteria change).
The generation pipeline loads qualified_patients.json directly — no online
filtering occurs during generation.

Usage
-----
    cd /path/to/PhysAssistBench
    uv run python physassistbench/prefilter_patients.py
    uv run python physassistbench/prefilter_patients.py --n 500 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physassistbench.pipeline.patient_prefilter import (
    DATA_ROOT,
    STATS_CSV,
    get_primary_hadm,
    load_stats_index,
    stage1_qualify,
    stage2_qualify,
)
from physassistbench.generate_all import SCENARIO_NAMES

# discharge_planning uses its own generator and is not in SCENARIO_NAMES
_ALL_SCENARIOS = SCENARIO_NAMES + ["discharge_planning"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_PKG_DIR, "qualified_patients.json")

DIFFICULTIES = [1, 2, 3]
N_TARGET = 500   # qualified candidates to collect per scenario × difficulty slot
SEED = 42


# ── Stage-2 worker ────────────────────────────────────────────────────────────

def _check_one(subject_id: int, scenario: str,
               difficulty: int) -> tuple[int, int | None, bool]:
    hadm_id = get_primary_hadm(subject_id)
    ok = stage2_qualify(subject_id, hadm_id, scenario, difficulty)
    return subject_id, hadm_id, ok


# ── Per-slot collection ───────────────────────────────────────────────────────

def collect_slot(
    scenario: str,
    difficulty: int,
    stats_index: dict[int, dict],
    all_pids: list[int],
    n_target: int,
    seed: int,
    max_workers: int,
) -> list[list]:
    """
    Return up to n_target [subject_id, hadm_id] pairs that pass both stages
    for the given scenario × difficulty slot.
    """
    rng = random.Random(seed)
    shuffled = all_pids[:]
    rng.shuffle(shuffled)

    # Stage 1 — instant, no I/O
    s1_pass = [
        pid for pid in shuffled
        if stage1_qualify(stats_index.get(pid, {}), scenario, difficulty)
    ]
    logger.info(
        f"  [Stage1] {len(s1_pass):>7,} / {len(shuffled):,} passed"
    )

    # Stage 2 — parallel CSV reads; stop early once n_target reached.
    # Scan all Stage-1 passers so the scan limit never caps the result.
    to_scan = s1_pass

    qualified: list[list] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_check_one, pid, scenario, difficulty): pid
            for pid in to_scan
        }
        for fut in as_completed(futs):
            if len(qualified) >= n_target:
                # Ask remaining futures to be ignored (they may still run,
                # but we stop collecting results).
                break
            try:
                sid, hid, ok = fut.result()
                if ok:
                    qualified.append([sid, hid])
            except Exception as exc:
                logger.debug(f"Worker error: {exc}")

    logger.info(
        f"  [Stage2] {len(qualified):>7,} qualified  (scanned ≤ {len(to_scan):,})"
    )
    if len(qualified) < n_target:
        logger.warning(
            f"  Only {len(qualified)}/{n_target} candidates found for "
            f"{scenario}/L{difficulty} — consider lowering thresholds or "
            f"increasing --n."
        )
    return qualified[:n_target]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_target: int = N_TARGET, seed: int = SEED, max_workers: int = 4,
         scenarios: list[str] | None = None):
    run_scenarios = scenarios if scenarios is not None else _ALL_SCENARIOS
    logger.info("=" * 60)
    logger.info("Patient pre-filter — PhysAssistBench physassistbench")
    logger.info(f"Data root : {DATA_ROOT}")
    logger.info(f"Stats CSV : {STATS_CSV}")
    logger.info(f"Scenarios : {run_scenarios}")
    logger.info(f"Target    : {n_target} candidates per scenario × difficulty")
    logger.info(f"Workers   : {max_workers}")
    logger.info(f"Output    : {OUT_PATH}")
    logger.info("=" * 60)

    stats_index = load_stats_index(STATS_CSV)
    all_pids = list(stats_index.keys())
    logger.info(f"Total patients in index: {len(all_pids):,}\n")

    # Load existing output and merge (keeps previously computed scenarios intact)
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            output = json.load(f)
        logger.info(f"Loaded existing output from {OUT_PATH}")
    else:
        output = {}
    output["generated_at"] = datetime.utcnow().isoformat() + "Z"
    output["n_target"] = n_target
    output["data_root"] = DATA_ROOT

    for scenario in run_scenarios:
        output[scenario] = {}
        for difficulty in DIFFICULTIES:
            logger.info(
                f"\n{'─'*50}\n"
                f"  Scenario: {scenario}   Difficulty: L{difficulty}\n"
                f"{'─'*50}"
            )
            candidates = collect_slot(
                scenario=scenario,
                difficulty=difficulty,
                stats_index=stats_index,
                all_pids=all_pids,
                n_target=n_target,
                seed=seed + difficulty,
                max_workers=max_workers,
            )
            output[scenario][str(difficulty)] = candidates

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved → {OUT_PATH}")

    # Summary table
    logger.info("\n=== Summary ===")
    for scenario in run_scenarios:
        for diff in DIFFICULTIES:
            n = len(output[scenario][str(diff)])
            status = "✓" if n >= n_target else f"⚠ only {n}"
            logger.info(f"  {scenario}/L{diff}: {n:4d} candidates  {status}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-filter MIMIC-IV patients for PhysAssistBench generation"
    )
    parser.add_argument(
        "--n", type=int, default=N_TARGET,
        help=f"Candidates to collect per scenario × difficulty (default: {N_TARGET})",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for Stage-2 content checks (default: 4)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        metavar="SCENARIO",
        help=(
            f"Scenarios to run (default: all). Choices: {_ALL_SCENARIOS}. "
            "Existing scenarios in output JSON are preserved."
        ),
    )
    args = parser.parse_args()
    main(n_target=args.n, seed=args.seed, max_workers=args.workers,
         scenarios=args.scenarios)
