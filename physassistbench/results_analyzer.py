"""
results_analyzer.py — Aggregate eval results into paper table values.

Four tables:
  1. Main results: Avg@4 / Pass@4 / Pass^4  ×  difficulty (L1/L2/L3)
  2. Implicit subtype: rubric score  ×  subtype (Explicit/NA/PE/AE)  ×  language
  3. Health literacy: rubric score  ×  literacy (High/Med/Low)  ×  language
     (on patient-tool turns only; requires 3 separate eval runs with --health_literacy)
  4. Task type: rubric score + AP + OP  ×  task type (IL/DG/CR/WU)  ×  language

Usage:
    # Single result dir (tables 1, 2, 4):
    uv run python -m physassistbench.results_analyzer --results_dir physassistbench/results_en_20260507_1507

    # Multiple dirs for cross-language / cross-literacy (all 4 tables):
    uv run python -m physassistbench.results_analyzer \\
        --results_dir physassistbench/results_en_20260507_1507 physassistbench/results_zh_... \\
        --literacy_dirs physassistbench/results_en_lit_low_... physassistbench/results_en_lit_medium_... physassistbench/results_en_lit_high_...

    # Output LaTeX table snippets:
    uv run python -m physassistbench.results_analyzer --results_dir ... --latex
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Task type label mapping ───────────────────────────────────────────────────
# Paper labels: IL = Implicit Lookup (Information Lookup), DG = Diagnostic Data Gathering (Data Gathering),
#               CR = Clinical Reasoning (Clinical Reasoning), WU = Write/Action (Action)
_TASK_TYPE_LABEL = {
    "Information Lookup": "IL",
    "Data Gathering": "DG",
    "Clinical Reasoning": "CR",
    "Write/Update": "WU",
}

# Subtype: None → "Explicit" (fully specified query)
_SUBTYPE_LABEL = {
    None: "Explicit",
    "NA": "NA",
    "PE": "PE",
    "AE": "AE",
}

_PASS_THRESHOLD = 0.6   # rubric_score ≥ this → pass


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_turns(results_dir: str) -> list[dict]:
    """Load turn_results_final.json from a results directory."""
    path = os.path.join(results_dir, "turn_results_final.json")
    if not os.path.exists(path):
        # Fall back to pre-judge
        path = os.path.join(results_dir, "turn_results_pre_judge.json")
    with open(path, encoding="utf-8") as f:
        turns = json.load(f)

    # Back-fill metadata fields that older runs may lack
    for t in turns:
        if "difficulty_level" not in t or t.get("difficulty_level") is None:
            eid = t.get("test_entry_id", "")
            m = re.search(r"_(L\d+)$", eid)
            t["difficulty_level"] = int(m.group(1)[1:]) if m else None

        if "scenario" not in t or not t.get("scenario"):
            eid = t.get("test_entry_id", "")
            m = re.match(r"demo\d+_\w+_(.+?)_\d+_L\d+$", eid)
            t["scenario"] = m.group(1) if m else ""

        if "language" not in t or not t.get("language"):
            # Infer from directory name
            dname = os.path.basename(results_dir)
            t["language"] = "zh" if "_zh" in dname else "en"

        if "health_literacy" not in t:
            t["health_literacy"] = None

    return turns


# ── Per-session grouping ──────────────────────────────────────────────────────

def group_by_session(turns: list[dict]) -> dict[str, list[dict]]:
    """Group turns by test_entry_id (= one benchmark session = 4 turns)."""
    sessions: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        sessions[t["test_entry_id"]].append(t)
    # Sort each session by task_idx
    for sid in sessions:
        sessions[sid].sort(key=lambda t: t.get("task_idx", 0))
    return dict(sessions)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _rubric(t: dict) -> Optional[float]:
    return t.get("rubric_score")


def _ap(t: dict) -> Optional[float]:
    return t.get("ap_rate")


def _op(t: dict) -> Optional[float]:
    return t.get("op_rate")


def _mean(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _pct(vals: list) -> str:
    v = _mean(vals)
    return f"{v*100:.1f}" if v is not None else "—"


# ── Table 1: Main results — Avg@4 / Pass@4 / Pass^4  ×  difficulty ───────────

def table1_main_results(turns: list[dict], model: str = "", language: str = "") -> dict:
    """
    Per session:
      Avg@4  = mean rubric_score over 4 turns
      Pass@4 = fraction of turns with rubric_score ≥ threshold
      Pass^4 = 1 if ALL turns in session ≥ threshold

    Returns nested dict: difficulty → metric → value (0-100 scale).
    """
    sessions = group_by_session(turns)

    # difficulty → lists of per-session metric values
    per_diff: dict[str, dict[str, list]] = defaultdict(lambda: {"avg4": [], "pass4": [], "passall4": []})

    for sid, sess_turns in sessions.items():
        rubrics = [_rubric(t) for t in sess_turns]
        valid = [r for r in rubrics if r is not None]
        if not valid:
            continue

        diff = sess_turns[0].get("difficulty_level")
        diff_key = f"L{diff}" if diff else "?"

        avg4 = sum(valid) / len(valid)
        pass4 = sum(1 for r in valid if r >= _PASS_THRESHOLD) / len(valid)
        passall4 = 1.0 if all(r >= _PASS_THRESHOLD for r in valid) else 0.0

        per_diff[diff_key]["avg4"].append(avg4)
        per_diff[diff_key]["pass4"].append(pass4)
        per_diff[diff_key]["passall4"].append(passall4)

    result = {}
    for diff in ["L1", "L2", "L3"]:
        d = per_diff.get(diff, {})
        avg4_vals = d.get("avg4", [])
        pass4_vals = d.get("pass4", [])
        passall4_vals = d.get("passall4", [])
        result[diff] = {
            "n_sessions": len(avg4_vals),
            "Avg@4":  _mean(avg4_vals),
            "Pass@4": _mean(pass4_vals),
            "Pass^4": _mean(passall4_vals),
        }

    # Overall (all difficulties pooled)
    all_avg = [v for d in per_diff.values() for v in d.get("avg4", [])]
    all_pass = [v for d in per_diff.values() for v in d.get("pass4", [])]
    all_passall = [v for d in per_diff.values() for v in d.get("passall4", [])]
    result["All"] = {
        "n_sessions": len(all_avg),
        "Avg@4":  _mean(all_avg),
        "Pass@4": _mean(all_pass),
        "Pass^4": _mean(all_passall),
    }

    return result


# ── Table 2: Implicit subtype — rubric  ×  subtype  ×  language ──────────────

def table2_subtype(turns_by_lang: dict[str, list[dict]]) -> dict:
    """
    Returns: {subtype_label: {lang: rubric_mean}}
    Subtype labels: Explicit / NA / PE / AE
    """
    result: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for lang, turns in turns_by_lang.items():
        for t in turns:
            rubric = _rubric(t)
            if rubric is None:
                continue
            raw_sub = t.get("turn_subtype")
            sub_label = _SUBTYPE_LABEL.get(raw_sub, str(raw_sub))
            result[sub_label][lang].append(rubric)

    # Convert lists to means
    return {
        sub: {lang: _mean(vals) for lang, vals in lang_vals.items()}
        for sub, lang_vals in result.items()
    }


# ── Table 3: Health literacy — rubric  ×  literacy  ×  language ──────────────

def table3_health_literacy(
    lit_turns: dict[str, dict[str, list[dict]]]
) -> dict:
    """
    lit_turns: {literacy_level: {lang: [turns]}}
    Returns: {literacy_level: {lang: rubric_mean}}

    Only considers turns where a patient tool was used (paq is not None,
    OR task_type == 'Data Gathering' with patient tool calls).
    """
    result: dict[str, dict[str, Optional[float]]] = {}

    for literacy, lang_turns in lit_turns.items():
        result[literacy] = {}
        for lang, turns in lang_turns.items():
            # Patient-tool turns: paq is set OR any executed action uses patient.*
            patient_turns = []
            for t in turns:
                if t.get("paq") is not None:
                    patient_turns.append(t)
                    continue
                exec_actions = t.get("llm_executed_actions") or []
                if any(a.get("action", {}).get("name", "").startswith("patient.")
                       for a in exec_actions):
                    patient_turns.append(t)

            rubrics = [_rubric(t) for t in patient_turns if _rubric(t) is not None]
            result[literacy][lang] = _mean(rubrics)

    return result


# ── Table 4: Task type — rubric + AP + OP  ×  task type  ×  language ─────────

def table4_task_type(turns_by_lang: dict[str, list[dict]]) -> dict:
    """
    Returns: {task_type_label: {lang: {rubric, ap, op}}}
    Task type labels: IL / DG / CR / WU
    """
    result: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"rubric": [], "ap": [], "op": []})
    )

    for lang, turns in turns_by_lang.items():
        for t in turns:
            raw_type = t.get("task_type", "")
            label = _TASK_TYPE_LABEL.get(raw_type, raw_type)

            rubric = _rubric(t)
            if rubric is not None:
                result[label][lang]["rubric"].append(rubric)

            ap = _ap(t)
            if ap is not None:
                result[label][lang]["ap"].append(ap)

            op = _op(t)
            if op is not None:
                result[label][lang]["op"].append(op)

    # Convert to means
    return {
        label: {
            lang: {
                "rubric": _mean(metrics["rubric"]),
                "ap": _mean(metrics["ap"]),
                "op": _mean(metrics["op"]),
                "n": len(metrics["rubric"]),
            }
            for lang, metrics in lang_metrics.items()
        }
        for label, lang_metrics in result.items()
    }


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt(v: Optional[float], pct: bool = True, dec: int = 1) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v*100:.{dec}f}"
    return f"{v:.{dec}f}"


def print_table1(t1: dict, model: str = "", language: str = "") -> None:
    header = f"Table 1 — Main Results ({model or 'model'}, {language or 'lang'})"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"{'Difficulty':<12} {'N':<6} {'Avg@4':>8} {'Pass@4':>8} {'Pass^4':>8}")
    print(f"{'-'*44}")
    for diff in ["L1", "L2", "L3", "All"]:
        d = t1.get(diff, {})
        n = d.get("n_sessions", 0)
        avg = _fmt(d.get("Avg@4"))
        p4 = _fmt(d.get("Pass@4"))
        pa = _fmt(d.get("Pass^4"))
        print(f"{diff:<12} {n:<6} {avg:>8} {p4:>8} {pa:>8}")


def print_table2(t2: dict, langs: list[str]) -> None:
    print(f"\n{'='*60}")
    print("Table 2 — Rubric Score by Implicit Subtype")
    print(f"{'='*60}")
    header = f"{'Subtype':<12}"
    for lang in langs:
        header += f" {lang.upper():>10}"
    print(header)
    print("-" * (12 + 11 * len(langs)))
    for sub in ["Explicit", "NA", "PE", "AE"]:
        row = f"{sub:<12}"
        for lang in langs:
            v = t2.get(sub, {}).get(lang)
            row += f" {_fmt(v):>10}"
        print(row)


def print_table3(t3: dict, langs: list[str]) -> None:
    print(f"\n{'='*60}")
    print("Table 3 — Rubric Score by Health Literacy (patient turns)")
    print(f"{'='*60}")
    header = f"{'Literacy':<12}"
    for lang in langs:
        header += f" {lang.upper():>10}"
    print(header)
    print("-" * (12 + 11 * len(langs)))
    for lit in ["high", "medium", "low"]:
        row = f"{lit:<12}"
        for lang in langs:
            v = t3.get(lit, {}).get(lang)
            row += f" {_fmt(v):>10}"
        print(row)


def print_table4(t4: dict, langs: list[str]) -> None:
    print(f"\n{'='*60}")
    print("Table 4 — Rubric + AP + OP by Task Type")
    print(f"{'='*60}")
    header = f"{'Type':<8}"
    for lang in langs:
        header += f"  {lang.upper()}-Rubric  {lang.upper()}-AP  {lang.upper()}-OP  {lang.upper()}-N"
    print(header)
    print("-" * (8 + 35 * len(langs)))
    for label in ["IL", "DG", "CR", "WU"]:
        row = f"{label:<8}"
        for lang in langs:
            d = t4.get(label, {}).get(lang, {})
            row += (
                f"  {_fmt(d.get('rubric')):>10}"
                f"  {_fmt(d.get('ap')):>6}"
                f"  {_fmt(d.get('op')):>6}"
                f"  {d.get('n', 0):>5}"
            )
        print(row)


def format_latex_table1(t1: dict) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Difficulty & Avg@4 & Pass@4 & Pass\^{}4 \\",
        r"\midrule",
    ]
    for diff in ["L1", "L2", "L3"]:
        d = t1.get(diff, {})
        lines.append(
            f"{diff} & {_fmt(d.get('Avg@4'))} & {_fmt(d.get('Pass@4'))} "
            f"& {_fmt(d.get('Pass^4'))} \\\\"
        )
    lines += [r"\midrule"]
    d = t1.get("All", {})
    lines.append(
        f"All & {_fmt(d.get('Avg@4'))} & {_fmt(d.get('Pass@4'))} "
        f"& {_fmt(d.get('Pass^4'))} \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def format_latex_table2(t2: dict, langs: list[str]) -> str:
    col_spec = "l" + "c" * len(langs)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Subtype & " + " & ".join(l.upper() for l in langs) + r" \\",
        r"\midrule",
    ]
    for sub in ["Explicit", "NA", "PE", "AE"]:
        vals = " & ".join(_fmt(t2.get(sub, {}).get(l)) for l in langs)
        lines.append(f"{sub} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def format_latex_table3(t3: dict, langs: list[str]) -> str:
    col_spec = "l" + "c" * len(langs)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Health Literacy & " + " & ".join(l.upper() for l in langs) + r" \\",
        r"\midrule",
    ]
    for lit in ["High", "Medium", "Low"]:
        k = lit.lower()
        vals = " & ".join(_fmt(t3.get(k, {}).get(l)) for l in langs)
        lines.append(f"{lit} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def format_latex_table4(t4: dict, langs: list[str]) -> str:
    n_cols = 1 + 3 * len(langs)
    col_spec = "l" + "c" * (3 * len(langs))
    header_row = "Task Type"
    for l in langs:
        header_row += f" & \\multicolumn{{3}}{{c}}{{{l.upper()}}}"
    sub_header = " & " + " & ".join(
        ["Rubric & AP & OP"] * len(langs)
    ).replace(" & ", " & ")
    # Flatten: Rubric AP OP for each lang
    sub_cols = []
    for _ in langs:
        sub_cols += ["Rubric", "AP", "OP"]
    sub_header = " & ".join(sub_cols) + r" \\"

    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row + r" \\",
        r"\cmidrule{" + f"2-{n_cols}" + "}",
        "& " + sub_header,
        r"\midrule",
    ]
    for label in ["IL", "DG", "CR", "WU"]:
        cols = [label]
        for l in langs:
            d = t4.get(label, {}).get(l, {})
            cols += [_fmt(d.get("rubric")), _fmt(d.get("ap")), _fmt(d.get("op"))]
        lines.append(" & ".join(cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate eval results into paper table values"
    )
    parser.add_argument(
        "--results_dir", nargs="+", required=True,
        help="One or more results directories (turn_results_final.json must exist)",
    )
    parser.add_argument(
        "--literacy_dirs", nargs="*", default=None,
        help=(
            "Three result dirs for literacy table: low medium high "
            "(must be in that order, or use --literacy_labels to specify)"
        ),
    )
    parser.add_argument(
        "--literacy_labels", nargs="*", default=["low", "medium", "high"],
        help="Labels for literacy dirs (default: low medium high)",
    )
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table snippets")
    parser.add_argument("--out", default=None, help="Write tables to this JSON file")
    parser.add_argument("--model", default="", help="Model label for table header")
    args = parser.parse_args()

    # Load all result dirs
    all_turns_by_lang: dict[str, list[dict]] = defaultdict(list)
    all_turns_flat: list[dict] = []
    for d in args.results_dir:
        turns = load_turns(d)
        lang = turns[0].get("language", "en") if turns else "en"
        all_turns_by_lang[lang].extend(turns)
        all_turns_flat.extend(turns)
        print(f"Loaded {len(turns)} turns from {d!r} (lang={lang})")

    langs = sorted(all_turns_by_lang.keys())

    # ── Table 1 ──────────────────────────────────────────────────────────────
    t1_by_lang = {}
    for lang, turns in all_turns_by_lang.items():
        t1_by_lang[lang] = table1_main_results(turns, model=args.model, language=lang)

    # Use first language for display if single language
    primary_lang = langs[0] if langs else "en"
    t1 = t1_by_lang.get(primary_lang, {})
    print_table1(t1, model=args.model, language=primary_lang)

    if len(langs) > 1:
        for lang in langs[1:]:
            print_table1(t1_by_lang[lang], model=args.model, language=lang)

    # ── Table 2 ──────────────────────────────────────────────────────────────
    t2 = table2_subtype(all_turns_by_lang)
    print_table2(t2, langs)

    # ── Table 3 ──────────────────────────────────────────────────────────────
    t3 = {}
    if args.literacy_dirs:
        if len(args.literacy_dirs) != len(args.literacy_labels):
            print(f"Warning: --literacy_dirs count ({len(args.literacy_dirs)}) "
                  f"!= --literacy_labels count ({len(args.literacy_labels)})")
        lit_turns: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for lit_dir, lit_label in zip(args.literacy_dirs, args.literacy_labels):
            turns = load_turns(lit_dir)
            lang = turns[0].get("language", "en") if turns else "en"
            lit_turns[lit_label][lang].extend(turns)
            print(f"Literacy '{lit_label}' ({lang}): {len(turns)} turns from {lit_dir!r}")
        t3 = table3_health_literacy(dict(lit_turns))
        print_table3(t3, langs)
    else:
        # Try to extract from existing turns if health_literacy field is set
        lit_turns_inferred: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for t in all_turns_flat:
            hl = t.get("health_literacy")
            if hl:
                lang = t.get("language", "en")
                lit_turns_inferred[hl][lang].append(t)
        if lit_turns_inferred:
            t3 = table3_health_literacy(dict(lit_turns_inferred))
            print_table3(t3, langs)
        else:
            print("\n[Table 3] No health_literacy data found. "
                  "Run eval 3x with --health_literacy low/medium/high and pass via --literacy_dirs.")

    # ── Table 4 ──────────────────────────────────────────────────────────────
    t4 = table4_task_type(all_turns_by_lang)
    print_table4(t4, langs)

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    if args.latex:
        print(f"\n{'='*60}")
        print("LaTeX snippets")
        print(f"{'='*60}")
        print("\n% Table 1")
        print(format_latex_table1(t1))
        print("\n% Table 2")
        print(format_latex_table2(t2, langs))
        if t3:
            print("\n% Table 3")
            print(format_latex_table3(t3, langs))
        print("\n% Table 4")
        print(format_latex_table4(t4, langs))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "table1_by_lang": t1_by_lang,
        "table2_subtype": t2,
        "table3_literacy": t3,
        "table4_task_type": t4,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved to {args.out}")
    else:
        out_path = os.path.join(
            os.path.dirname(args.results_dir[0]),
            "tables_summary.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
