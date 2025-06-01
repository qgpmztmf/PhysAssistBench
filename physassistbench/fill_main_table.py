"""
fill_main_table.py — Compute the main-results LaTeX table row for a model.

Per language, per difficulty (L1/L2/L3) and overall (All):
  Avg     = mean over sessions of (mean rubric_score over the session's turns)  [%]
  @T(τ)   = Pass@Turn    = fraction of turns with rubric_score ≥ τ              [%]
  @S(τ)   = Pass@Session = fraction of sessions where EVERY turn ≥ τ            [%]
with τ ∈ {0.60, 0.75}.

The LaTeX row matches tab:main_results column order:
  Model & EN[L1 L2 L3 @T60 @S60 @T75 @S75] & ZH[L1 L2 L3 @T60 @S60 @T75 @S75] \\

Usage:
    # one model, EN + ZH dirs
    uv run python physassistbench/fill_main_table.py --model "GPT-4.1-nano" \
        --en physassistbench/results_en_gpt_4_1_nano_20260521_0958 \
        --zh physassistbench/results_zh_gpt_4_1_nano_20260521_0958

    # EN only (ZH cells left blank)
    uv run python physassistbench/fill_main_table.py --model "GPT-4.1-nano" --en <dir>

    # print a readable table instead of LaTeX
    uv run python physassistbench/fill_main_table.py --model X --en <dir> --zh <dir> --table
"""

import argparse
import json
import os
import re
from collections import defaultdict

THRESHOLDS = (0.60, 0.75)


def _load_turns(results_dir: str) -> list:
    path = os.path.join(results_dir, "turn_results_final.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No turn_results_final.json in {results_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _difficulty(t: dict) -> str:
    d = t.get("difficulty_level")
    if d:
        return f"L{d}"
    m = re.search(r"_(L\d+)$", t.get("test_entry_id", ""))
    return m.group(1) if m else "?"


def compute(results_dir: str) -> dict:
    """Return {difficulty: {Avg, N, @T0.6, @S0.6, @T0.75, @S0.75}} including 'All'."""
    turns = _load_turns(results_dir)
    sessions = defaultdict(list)  # (difficulty, entry_id) -> [rubric_score, ...]
    for t in turns:
        r = t.get("rubric_score")
        if r is None:
            continue
        sessions[(_difficulty(t), t["test_entry_id"])].append(r)

    out = {}
    for diff in ["L1", "L2", "L3", "All"]:
        items = (list(sessions.items()) if diff == "All"
                 else [(k, v) for k, v in sessions.items() if k[0] == diff])
        session_avgs = [sum(v) / len(v) for _, v in items]
        row = {
            "N": len(session_avgs),
            "Avg": (sum(session_avgs) / len(session_avgs) * 100) if session_avgs else 0.0,
        }
        for thr in THRESHOLDS:
            turn_pass = [1 if r >= thr else 0 for _, v in items for r in v]
            sess_pass = [1 if all(r >= thr for r in v) else 0 for _, v in items]
            row[f"@T{thr}"] = (sum(turn_pass) / len(turn_pass) * 100) if turn_pass else 0.0
            row[f"@S{thr}"] = (sum(sess_pass) / len(sess_pass) * 100) if sess_pass else 0.0
        out[diff] = row
    return out


def _latex_cells(stats: dict | None) -> str:
    """7 cells: L1 L2 L3 @T60 @S60 @T75 @S75 (uses 'All' for the τ columns)."""
    if stats is None:
        return " & ".join([""] * 7)
    a = stats
    return (
        f"{a['L1']['Avg']:.1f} & {a['L2']['Avg']:.1f} & {a['L3']['Avg']:.1f} & "
        f"{a['All']['@T0.6']:.1f} & {a['All']['@S0.6']:.1f} & "
        f"{a['All']['@T0.75']:.1f} & {a['All']['@S0.75']:.1f}"
    )


def _print_table(label: str, stats: dict):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f'{"Diff":<6} {"N":>4} {"Avg":>7} | {"@T.60":>7} {"@S.60":>7} | {"@T.75":>7} {"@S.75":>7}')
    print("-" * 72)
    for diff in ["L1", "L2", "L3", "All"]:
        r = stats[diff]
        print(f'{diff:<6} {r["N"]:>4} {r["Avg"]:>6.1f}% | '
              f'{r["@T0.6"]:>6.1f}% {r["@S0.6"]:>6.1f}% | '
              f'{r["@T0.75"]:>6.1f}% {r["@S0.75"]:>6.1f}%')


def main():
    p = argparse.ArgumentParser(description="Fill main-results LaTeX table row")
    p.add_argument("--model", required=True, help="Model display name for the row label")
    p.add_argument("--en", default=None, help="English results dir")
    p.add_argument("--zh", default=None, help="Chinese results dir")
    p.add_argument("--table", action="store_true", help="Print readable tables too")
    args = p.parse_args()

    if not args.en and not args.zh:
        p.error("provide at least one of --en / --zh")

    en_stats = compute(args.en) if args.en else None
    zh_stats = compute(args.zh) if args.zh else None

    if args.table:
        if en_stats:
            _print_table(f"{args.model} [EN]", en_stats)
        if zh_stats:
            _print_table(f"{args.model} [ZH]", zh_stats)

    row = f"{args.model:<28} & {_latex_cells(en_stats)} & {_latex_cells(zh_stats)} \\\\"
    print("\n% LaTeX row (EN: L1 L2 L3 @T60 @S60 @T75 @S75 | ZH: same)")
    print(row)


if __name__ == "__main__":
    main()
