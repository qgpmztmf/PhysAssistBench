"""
Quick eval: compare DeepSeek-V4-Flash on explicit vs implicit queries.
Reads 10 sampled entries, runs eval twice (implicit / explicit), prints rubric comparison.

Usage:
    cd /path/to/PhysAssistBench
    uv run python physassistbench/run_explicit_vs_implicit.py
"""
import json
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physassistbench.eval_runner import run_evaluation, DOCTOR_AGENT_MODEL
from physassistbench.phm.patient_agent_runtime import reset_all_sessions, register_session, get_session
from physassistbench.tools.tool_registry import set_active_date

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

SAMPLE_PATH = "/tmp/eval_sample_10.jsonl"
OUTPUT_DIR  = "/tmp/eval_explicit_vs_implicit"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _run(entries: list, use_explicit: bool, label: str) -> list:
    """Patch entries to use explicit or implicit tasks, run eval, return turn_results."""
    import copy, re, importlib
    import physassistbench.eval_runner as _er
    from physassistbench.eval_runner import load_model_config, _build_client, run_agent_turn, compute_tool_metrics
    from physassistbench.run_eval import _SYSTEM_PROMPT_V2_EN
    from physassistbench.tools.tool_registry import call_tool, set_active_date
    from dotenv import load_dotenv
    from openai import OpenAI

    from physassistbench.paths import ENV_PATH as _ENV
    load_dotenv(dotenv_path=_ENV, override=False)

    model_cfg = load_model_config("deepseek-v4-flash")
    api_key   = os.environ.get(model_cfg["api_key_env"], "")
    client    = _build_client(model_cfg, api_key)
    _er.DOCTOR_AGENT_MODEL = model_cfg["model_id"]   # override global to deepseek-v4-flash

    reset_all_sessions()
    for entry in entries:
        sid = entry.get("session_id", "")
        if sid:
            try:
                register_session(sid, entry.get("subject_id", 0), entry.get("persona_config", {}))
                al = entry.get("answer_list", [])
                patient_actions = [a for turn in al for a in (turn or [])
                                   if isinstance(a.get("observation"), dict)
                                   and "patient_response" in a["observation"]]
                if patient_actions:
                    get_session(sid).preload_responses(patient_actions)
            except Exception:
                pass

    all_results = []
    system_prompt = _SYSTEM_PROMPT_V2_EN

    for entry in entries:
        entry_id  = entry["id"]
        subject_id = entry.get("subject_id", 0)
        hadm_id   = entry.get("hadm_id")
        session_id = entry.get("session_id", "")
        task_types = entry.get("task_types", [])
        tool_schema = entry.get("tools_en") or entry.get("tools", [])
        answer_list = entry.get("answer_list_en") or entry.get("answer_list", [])

        if use_explicit:
            tasks = entry.get("tasks_en_explicit") or entry.get("tasks_en") or []
        else:
            tasks = entry.get("tasks_en") or []

        import re as _re
        env_info = entry.get("env_info", "")
        current_date = entry.get("current_date") or (
            (m := _re.search(r"Current date: (\d{4}-\d{2}-\d{2})", env_info)) and m.group(1)
        ) or None
        set_active_date(current_date)

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content":
            f"[Environment Info]\n{env_info}\nsubject_id: {subject_id}\nhadm_id: {hadm_id}\n\n"
            "You are now assisting with a clinical case. I will give you tasks one at a time."})
        messages.append({"role": "assistant", "content":
            "Understood. I'm ready to assist. Please give me the first task."})

        for turn_idx, question in enumerate(tasks):
            task_type = task_types[turn_idx] if turn_idx < len(task_types) else "Information Lookup"
            gold_turn = answer_list[turn_idx] if turn_idx < len(answer_list) else []

            if task_type == "Intake":
                turn_tools = [t for t in tool_schema if t["function"]["name"].startswith("patient.")]
            elif task_type == "Protocol":
                turn_tools = []
            else:
                turn_tools = [t for t in tool_schema if not t["function"]["name"].startswith("patient.")]

            messages.append({"role": "user", "content": question})
            try:
                executed, answer, messages = run_agent_turn(
                    client=client, messages=messages, tools=turn_tools,
                    subject_id=subject_id, session_id=session_id,
                    task_type=task_type, gold_turn=gold_turn,
                )
            except Exception as e:
                executed = [{"action": {"name": "prepare_to_answer", "arguments": {}},
                             "observation": {"error": str(e)}, "idx": 0}]
                answer = f"[Error: {e}]"

            metrics = compute_tool_metrics(executed, gold_turn, task_type)
            subtype = (entry.get("turn_subtypes") or [None]*4)[turn_idx] \
                      if turn_idx < len(entry.get("turn_subtypes") or []) else None

            all_results.append({
                "entry_id": entry_id,
                "turn_idx": turn_idx,
                "task_type": task_type,
                "subtype": subtype,
                "question": question,
                "answer": answer,
                "ap_rate": metrics["ap_rate"],
                "gold_actions": gold_turn,
                "mode": label,
            })

        print(f"  [{label}] {entry_id} done")

    return all_results


def score_rubrics(results: list, entries: list) -> list:
    """Run rubric scoring on turn results."""
    from physassistbench.eval.rubric_eval import score_turn_rubric
    entry_map = {e["id"]: e for e in entries}

    for r in results:
        entry = entry_map.get(r["entry_id"], {})
        rubrics = entry.get("rubrics", [])
        ti = r["turn_idx"]
        rubric_items = rubrics[ti] if ti < len(rubrics) else []
        messages_en = entry.get("messages_en", [])
        # Filter out separator strings — keep only dict messages
        history = [m for m in messages_en if isinstance(m, dict)][:ti * 2]

        if rubric_items:
            res = score_turn_rubric(
                rubric_items=rubric_items,
                user_question=r["question"],
                model_answer=r["answer"],
                executed_actions=r["gold_actions"],
                history=history,
                language="en",
            )
            r["rubric_score"] = res["rubric_score"]
            r["rubric_items_passed"] = res["items_passed"]
            r["rubric_items_total"] = res["items_total"]
        else:
            r["rubric_score"] = None
        print(f"  scored {r['entry_id']} T{ti} [{r['mode']}] rubric={r.get('rubric_score')}")

    return results


def main():
    with open(SAMPLE_PATH) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    print(f"=== Running IMPLICIT eval ({len(entries)} entries) ===")
    implicit_results = _run(entries, use_explicit=False, label="implicit")

    print(f"\n=== Running EXPLICIT eval ({len(entries)} entries) ===")
    explicit_results = _run(entries, use_explicit=True, label="explicit")

    print("\n=== Scoring rubrics (implicit) ===")
    implicit_results = score_rubrics(implicit_results, entries)

    print("\n=== Scoring rubrics (explicit) ===")
    explicit_results = score_rubrics(explicit_results, entries)

    # Save
    all_results = implicit_results + explicit_results
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON: Implicit vs Explicit  (rubric score, higher=better)")
    print("="*80)
    print(f"{'Entry':<38} {'T':<2} {'Sub':<4} {'Impl':>6} {'Expl':>6} {'Δ':>6}")
    print("-"*70)

    entry_ids = [e["id"] for e in entries]
    for eid in entry_ids:
        im_turns = [r for r in implicit_results if r["entry_id"] == eid]
        ex_turns = [r for r in explicit_results if r["entry_id"] == eid]
        for ti in range(len(im_turns)):
            im = im_turns[ti]
            ex = ex_turns[ti] if ti < len(ex_turns) else {}
            rs_im = im.get("rubric_score")
            rs_ex = ex.get("rubric_score")
            delta = (rs_ex - rs_im) if rs_im is not None and rs_ex is not None else None
            st = str(im.get("subtype") or "—")[:4]
            im_s = f"{rs_im*100:.0f}" if rs_im is not None else "—"
            ex_s = f"{rs_ex*100:.0f}" if rs_ex is not None else "—"
            d_s  = f"{delta*100:+.0f}" if delta is not None else "—"
            print(f"{eid[-37:]:<38} {ti:<2} {st:<4} {im_s:>6} {ex_s:>6} {d_s:>6}")
        print()

    # Summary
    valid = [(r["entry_id"], r["turn_idx"], r.get("rubric_score"))
             for r in implicit_results if r.get("rubric_score") is not None]
    im_scores = {(r["entry_id"],r["turn_idx"]): r["rubric_score"]
                 for r in implicit_results if r.get("rubric_score") is not None}
    ex_scores = {(r["entry_id"],r["turn_idx"]): r["rubric_score"]
                 for r in explicit_results if r.get("rubric_score") is not None}
    common = set(im_scores) & set(ex_scores)
    if common:
        im_avg = sum(im_scores[k] for k in common) / len(common)
        ex_avg = sum(ex_scores[k] for k in common) / len(common)
        print("="*70)
        print(f"Avg rubric  Implicit: {im_avg*100:.1f}%   Explicit: {ex_avg*100:.1f}%   Δ={( ex_avg-im_avg)*100:+.1f}%")

        # By subtype
        from collections import defaultdict
        by_st: dict = defaultdict(lambda: {"im":[], "ex":[]})
        for r in implicit_results:
            k = (r["entry_id"], r["turn_idx"])
            if k in ex_scores and r.get("rubric_score") is not None:
                st = str(r.get("subtype") or "None")
                by_st[st]["im"].append(r["rubric_score"])
        for r in explicit_results:
            k = (r["entry_id"], r["turn_idx"])
            if k in im_scores and r.get("rubric_score") is not None:
                st_key = str(next((x["subtype"] for x in implicit_results
                                   if x["entry_id"]==r["entry_id"] and x["turn_idx"]==r["turn_idx"]), "None"))
                by_st[st_key]["ex"].append(r["rubric_score"])
        print()
        print("By subtype:")
        for st in ["None", "PE", "NA", "AE"]:
            d = by_st.get(st, {"im":[], "ex":[]})
            if d["im"] and d["ex"]:
                im_m = sum(d["im"])/len(d["im"])
                ex_m = sum(d["ex"])/len(d["ex"])
                print(f"  {st:<6}: Impl={im_m*100:.1f}%  Expl={ex_m*100:.1f}%  Δ={(ex_m-im_m)*100:+.1f}%  (n={len(d['im'])})")


if __name__ == "__main__":
    main()
