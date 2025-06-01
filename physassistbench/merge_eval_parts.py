"""
merge_eval_parts.py — Merge per-scenario eval result parts into one dir per language.

After the 8-way distributed eval (4 scenarios x 2 languages), results live in:
  physassistbench/results_dsv4pro_parts/{lang}_{scenario}/turn_results_final.json
This concatenates the 4 scenario parts per language into a single
turn_results_final.json so results_analyzer / fill_main_table can consume it.

Usage:
    uv run python physassistbench/merge_eval_parts.py --parts_dir physassistbench/results_dsv4pro_parts \
        --model_label deepseek_v4_pro
"""

import argparse
import glob
import json
import os
from datetime import datetime

SCENARIOS = ["diagnostic_workup", "discharge_planning", "med_safety", "treatment_response"]
PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def merge_lang(parts_dir: str, lang: str, model_label: str) -> str | None:
    merged = []
    found = 0
    for sc in SCENARIOS:
        p = os.path.join(parts_dir, f"{lang}_{sc}", "turn_results_final.json")
        if not os.path.exists(p):
            print(f"  ⚠️ missing: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            merged.extend(json.load(f))
        found += 1
    if not merged:
        return None
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join(PKG_DIR, f"results_{lang}_{model_label}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "turn_results_final.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  {lang}: merged {found}/4 scenarios, {len(merged)} turns → {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts_dir", required=True)
    p.add_argument("--model_label", required=True,
                   help="e.g. deepseek_v4_pro (used in merged dir name)")
    args = p.parse_args()

    dirs = {}
    for lang in ["en", "zh"]:
        d = merge_lang(args.parts_dir, lang, args.model_label)
        if d:
            dirs[lang] = d

    if dirs:
        print("\nNext: analyze with")
        en = dirs.get("en", "<en_dir>")
        zh = dirs.get("zh", "<zh_dir>")
        print(f"  uv run python physassistbench/fill_main_table.py --model '{args.model_label}' "
              f"--en {en} --zh {zh} --table")


if __name__ == "__main__":
    main()
