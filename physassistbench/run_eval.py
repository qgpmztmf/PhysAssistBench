"""
physassistbench/run_eval.py — Evaluate a Doctor Agent against the PhysAssistBench benchmark.

Uses IIRS (Implicit Information Recovery Score) decomposed into:
  AID (Antecedent ID) / ER (Ellipsis Recovery) / AC (Argument Completion) / AA (Answer Accuracy)

Additional metrics:
  FPTR (False Positive Tool Rate) / MTR (Missing Tool Rate)
  Cross-source Reconciliation (Mixed-source Data Gathering turns)

Usage:
    cd /path/to/PhysAssistBench

    # Evaluate with English questions (default)
    uv run python physassistbench/run_eval.py --language en [--skip_judge] [--verbose]

    # Evaluate with Chinese questions
    uv run python physassistbench/run_eval.py --language zh [--skip_judge] [--verbose]

    # Skip data generation (use existing JSONL files)
    uv run python physassistbench/run_eval.py --language en --skip_generate [--skip_judge]

    # Evaluate specific scenarios only
    uv run python physassistbench/run_eval.py --scenarios infection_management critical_care
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physassistbench.eval_runner import run_evaluation, compute_tool_metrics
from physassistbench.eval.metrics import calc_accuracy
from physassistbench.generate_all import SCENARIO_NAMES
from physassistbench.eval.iirs import (
    compute_iirs_turn,
    compute_iirs_entry,
    compute_benchmark_summary,
    score_cross_source_reconciliation,
    score_patient_answer_quality,
    _llm_score,
)
from physassistbench.eval.rule_checker import rule_based_is_correct
from physassistbench.eval.rubric_eval import score_turn_rubric
from physassistbench.eval.eval_discharge_planning import score_action_turn, _WRITE_TOOL_NAMES

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_PKG_DIR, "data")  # resolved at main() time to latest data_* dir

_IS_CORRECT_THRESHOLD = 0.5  # AA ≥ this → is_correct=True


def _latest_data_dir() -> str:
    """Return the most recently modified data_* directory, or fall back to 'data'."""
    import glob
    candidates = glob.glob(os.path.join(_PKG_DIR, "data_*"))
    if not candidates:
        return os.path.join(_PKG_DIR, "data")
    return max(candidates, key=os.path.getmtime)


def judge_is_correct(
    user_question: str,
    gold_answer: str,
    pred_answer: str,
    language: str = "en",
) -> bool:
    """
    LLM judge: is the predicted answer clinically correct relative to the gold?
    Returns True/False. Uses the same LLM scorer as IIRS AA.
    Falls back to False if either answer is empty.
    """
    if not pred_answer or not gold_answer:
        return False

    if language == "zh":
        prompt = f"""判断以下预测答案是否与标准答案临床等价。

用户问题："{user_question}"

标准答案：
"{gold_answer[:600]}"

预测答案：
"{pred_answer[:600]}"

评分标准：
- 如果预测答案与标准答案临床等价（即关键临床事实正确），返回 1。
- 如果预测答案错误、不相关或遗漏了关键临床事实，返回 0。

返回JSON：{{"is_correct": 0 or 1}}"""
    else:
        prompt = f"""Judge whether the predicted answer is clinically correct relative to the gold answer.

User question: "{user_question}"

Gold answer:
"{gold_answer[:600]}"

Predicted answer:
"{pred_answer[:600]}"

Scoring:
- Return 1 if the predicted answer is clinically equivalent (key clinical facts are correct).
- Return 0 if the predicted answer is wrong, irrelevant, or missing key clinical facts.

Return JSON: {{"is_correct": 0 or 1}}"""

    result = _llm_score(prompt)
    return bool(result.get("is_correct", 0))

_SYSTEM_PROMPT_V2_EN = """You are a clinical decision support agent assisting a physician in an ICU/hospital setting.
You have EHR tools (structured patient data) and patient interview tools available.

- Use the available tools to gather whatever information is needed to answer the
  physician's question. Do NOT ask the physician for data you can look up yourself.
- Decide autonomously which tool(s) to call and how many — call as many as the
  question requires, then stop.
- For patient interview tools, always pass session_id from env_info.
- For implicit or elliptic questions, resolve the implicit reference from the
  conversation history before acting.
- Answer concisely but clinically completely: state the relevant values with units
  and their normal/abnormal status, and give the clinical conclusion or recommendation
  the question calls for.
- Always call prepare_to_answer as your final action before giving your answer."""

_SYSTEM_PROMPT_V2_ZH = """你是一位临床决策支持助手，协助医生在ICU/医院环境中工作。
你可以使用EHR工具（结构化患者数据）和患者访谈工具。

- 使用可用的工具自主获取回答医生问题所需的信息。不要向医生索取你自己能查到的数据。
- 自主判断需要调用哪些工具、调用几个——问题需要多少就调多少，然后停止。
- 使用患者访谈工具时，始终传入env_info中的session_id。
- 对于隐含或省略性问题，在行动前从对话历史中解析隐含指代。
- 简洁但临床完整地回答：给出相关数值及单位、正常/异常判断，以及问题所需的临床结论或建议。
- 最后一个操作务必调用prepare_to_answer。"""


def _output_dir(language: str, timestamp: str, health_literacy: str | None = None,
                model: str | None = None, use_explicit: bool = False) -> str:
    parts = [f"results_{language}"]
    # "high" is the locked default — only tag the dir for non-default literacy
    # experiments (low/medium) to keep standard result dir names clean.
    if health_literacy and health_literacy != "high":
        parts.append(f"lit_{health_literacy}")
    if use_explicit:
        parts.append("explicit")
    if model:
        parts.append(model.replace("-", "_").replace(".", "_"))
    parts.append(timestamp)
    return os.path.join(_PKG_DIR, "_".join(parts))


def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def load_benchmark_entries(scenarios: list[str] | None = None) -> list:
    target = scenarios or SCENARIO_NAMES
    entries = []
    for scenario in target:
        path = os.path.join(DATA_DIR, f"{scenario}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"Missing benchmark file: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            domain_entries = [json.loads(line) for line in f if line.strip()]
        logger.info(f"  {scenario}: {len(domain_entries)} entries loaded")
        entries.extend(domain_entries)
    return entries


def _extract_gold_turn_info(gold_turn_actions: list) -> dict:
    """Extract structured gold info from a turn's answer_list entry."""
    tool_calls = [
        a["action"]["name"]
        for a in gold_turn_actions
        if a["action"]["name"] not in ("prepare_to_answer", "ask_user_for_required_parameters")
    ]
    tool_args = {}
    if tool_calls:
        # Use args from the first real tool call as primary
        first_real = next(
            (a for a in gold_turn_actions if a["action"]["name"] == tool_calls[0]), {}
        )
        tool_args = first_real.get("action", {}).get("arguments", {})
    answer = next(
        (str(a.get("observation", "")) for a in gold_turn_actions
         if a["action"]["name"] == "prepare_to_answer"), ""
    )
    # Extract fixed patient_response for PAQ scoring (only present on patient.* tool turns)
    patient_response = ""
    for a in gold_turn_actions:
        obs = a.get("observation", {})
        if isinstance(obs, dict) and "patient_response" in obs:
            patient_response = obs["patient_response"]
            break
    return {"tool_calls": tool_calls, "tool_args": tool_args, "answer": answer,
            "patient_response": patient_response}


def compute_iirs_for_entries(
    turn_results: list[dict],
    benchmark_entries: list[dict],
    language: str = "en",
    use_rule_based: bool = False,
) -> list[dict]:
    """
    Compute IIRS scores for all turns, annotating turn_results in-place.

    use_rule_based=True  → is_correct determined by rule-based checker
                           (tool name match + schema check + ROUGE-L/edit distance),
                           mirroring WildToolBench's ToolArgsChecker approach.
                           IIRS components (AID, ER, AC, AA) are still computed
                           and saved for analysis, but is_correct is NOT derived
                           from AA.

    use_rule_based=False → is_correct = (AA >= _IS_CORRECT_THRESHOLD)  [default]
                           AA is computed by an LLM judge.
    """
    # Build a lookup: entry_id → benchmark entry
    entry_map = {e["id"]: e for e in benchmark_entries}

    iirs_by_entry: dict[str, list[dict]] = defaultdict(list)

    for r in turn_results:
        entry_id = r.get("test_entry_id", "")
        turn_idx = r.get("task_idx", 0)
        entry = entry_map.get(entry_id, {})

        # Gold info
        answer_list = entry.get("answer_list", [])
        gold_turn_actions = answer_list[turn_idx] if turn_idx < len(answer_list) else []
        gold_info = _extract_gold_turn_info(gold_turn_actions)

        # Predicted tool calls + answer from eval result
        _exec = r.get("llm_executed_actions") or r.get("executed_actions", [])
        _CONTROL = {"prepare_to_answer", "ask_user_for_required_parameters"}
        pred_tools = [
            a["action"]["name"] for a in _exec
            if a["action"]["name"] not in _CONTROL
        ]
        pred_answer = r.get("predicted_answer", "") or r.get("final_answer", "") or r.get("llm_answer", "")
        pred_args = {}
        if pred_tools:
            first_pred = next(
                (a for a in _exec if a["action"]["name"] not in _CONTROL), {}
            )
            pred_args = first_pred.get("action", {}).get("arguments", {})

        pred_info = {"tool_calls": pred_tools, "tool_args": pred_args, "answer": pred_answer}

        # History up to this turn
        subtypes = entry.get("turn_subtypes", [])
        subtype = subtypes[turn_idx] if turn_idx < len(subtypes) else None
        task_type = r.get("task_type", "Information Lookup")

        messages_key = "messages_zh" if language == "zh" else "messages_en"
        messages = entry.get(messages_key) or entry.get("messages", [])
        history = [m for m in messages if isinstance(m, dict)][:turn_idx * 2]

        # Antecedent for EA/AE (from dep_graph)
        dep_graph = entry.get("dep_graph", {})
        antecedent = ""
        if subtype in ("EA", "AE") and dep_graph:
            nodes = dep_graph.get("nodes", {})
            # Use entity/event from the most recent prior turn
            if str(turn_idx - 1) in nodes:
                prior = nodes[str(turn_idx - 1)]
                antecedent = (
                    (prior.get("entities") or [None])[0]
                    or (prior.get("events") or [None])[0]
                    or ""
                )
        gold_info["antecedent"] = antecedent

        if use_rule_based:
            # Rule-based path: no LLM calls at all.
            # is_correct: tool names + schema + ROUGE-L/edit-distance
            tools = entry.get("tools", [])
            rb_ok, rb_reason = rule_based_is_correct(gold_info, pred_info, tools)
            r["is_correct"] = rb_ok
            r["is_correct_method"] = "rule_based"
            r["is_correct_reason"] = rb_reason

            # AC is rule-based (argument comparison) — compute without LLM
            from physassistbench.eval.iirs import score_AC
            ac = score_AC(r.get("user_question", ""), gold_info.get("tool_args", {}), pred_info.get("tool_args", {}))

            # Tool metrics
            gold_set = set(gold_info.get("tool_calls", []))
            pred_set = set(pred_info.get("tool_calls", []))
            iirs_scores = {
                "AID": 1.0, "ER": 1.0, "AC": ac, "AA": float(rb_ok),
                "IIRS_turn": ac * float(rb_ok),
                "FPTR": 1.0 if (pred_set - gold_set) else 0.0,
                "MTR": 1.0 if (gold_set - pred_set) else 0.0,
                "false_positive_tools": list(pred_set - gold_set),
                "missing_tools": list(gold_set - pred_set),
                "applicable_components": ["AC", "AA"],
                "component_mean": (ac + float(rb_ok)) / 2,
                "subtype": subtype,
                "task_type": task_type,
            }
        else:
            # LLM-judge path: full IIRS scoring (makes LLM calls for AID, AA)
            iirs_scores = compute_iirs_turn(
                user_question=r.get("user_question", ""),
                history=history,
                gold_turn=gold_info,
                pred_turn=pred_info,
                subtype=subtype,
                task_type=task_type,
                language=language,
            )
            r["is_correct"] = iirs_scores["AA"] >= _IS_CORRECT_THRESHOLD
            r["is_correct_method"] = "llm_judge"
            r["is_correct_reason"] = f"AA={iirs_scores['AA']:.3f} threshold={_IS_CORRECT_THRESHOLD}"

        r.update({
            "iirs_AID": iirs_scores["AID"],
            "iirs_ER": iirs_scores["ER"],
            "iirs_AC": iirs_scores["AC"],
            "iirs_AA": iirs_scores["AA"],
            "iirs_turn": iirs_scores["IIRS_turn"],
            "fptr": iirs_scores["FPTR"],
            "mtr": iirs_scores["MTR"],
        })

        # ── Rubric scoring ────────────────────────────────────────────────────
        if not use_rule_based:
            rubrics_key = "rubrics_zh" if language == "zh" else "rubrics"
            all_rubrics = entry.get(rubrics_key) or entry.get("rubrics", [])
            rubric_items = all_rubrics[turn_idx] if turn_idx < len(all_rubrics) else []

            if rubric_items and task_type == "Write/Update":
                # Write/Update turns are scored by parameter comparison against
                # the model's actual write tool-call arguments — NOT its free text.
                # (score_turn_rubric judges prose, which silently fails WU turns
                #  that call the tool but don't restate every field in the answer.)
                gold_write_params = [
                    {"tool": a["action"]["name"], "params": a["action"].get("arguments", {})}
                    for a in gold_turn_actions
                    if a.get("action", {}).get("name") in _WRITE_TOOL_NAMES
                ]
                _ref_ctx = next(
                    (str(a.get("observation", "")) for a in reversed(gold_turn_actions)
                     if a.get("action", {}).get("name") == "prepare_to_answer"),
                    "",
                )
                rubric_result = score_action_turn(
                    rubric_items=rubric_items,
                    user_question=r.get("user_question", ""),
                    model_write_calls=_exec,              # serializer filters to write tools
                    reference_context=_ref_ctx,
                    gold_write_params=gold_write_params,
                    history=history,
                    language=language,
                )
                r["rubric_score"]    = rubric_result["rubric_score"]
                r["rubric_items_passed"] = rubric_result["items_passed"]
                r["rubric_items_total"]  = rubric_result["items_total"]
                r["rubric_scores"]   = rubric_result["scores"]
                r["rubric_reasoning"] = rubric_result["reasoning"]
            elif rubric_items:
                rubric_result = score_turn_rubric(
                    rubric_items=rubric_items,
                    user_question=r.get("user_question", ""),
                    model_answer=pred_answer,
                    executed_actions=gold_turn_actions,
                    history=history,
                    language=language,
                )
                r["rubric_score"]    = rubric_result["rubric_score"]
                r["rubric_items_passed"] = rubric_result["items_passed"]
                r["rubric_items_total"]  = rubric_result["items_total"]
                r["rubric_scores"]   = rubric_result["scores"]
                r["rubric_reasoning"] = rubric_result["reasoning"]
            else:
                r["rubric_score"] = None
                r["rubric_items_passed"] = 0
                r["rubric_items_total"]  = 0
                r["rubric_scores"]   = []
                r["rubric_reasoning"] = []

        # ── Patient Answer Quality (PAQ) — only for patient tool turns ───────
        patient_response = gold_info.get("patient_response", "")
        gold_patient_tool = next(
            (t for t in gold_info.get("tool_calls", []) if t.startswith("patient.")), None
        )
        has_patient_tool = gold_patient_tool is not None
        if gold_patient_tool:
            r["gold_tool"] = gold_patient_tool
        if has_patient_tool and not use_rule_based:
            paq_scores = score_patient_answer_quality(
                user_question=r.get("user_question", ""),
                patient_response=patient_response,
                llm_answer=pred_answer,
                language=language,
            )
            r["paq_faithfulness"] = paq_scores["Faithfulness"]
            r["paq_coverage"]     = paq_scores["Coverage"]
            r["paq_relevance"]    = paq_scores["Relevance"]
            r["paq"]              = paq_scores["PAQ"]
            r["paq_reasoning"]    = paq_scores["reasoning"]
        else:
            r["paq_faithfulness"] = None
            r["paq_coverage"]     = None
            r["paq_relevance"]    = None
            r["paq"]              = None
            r["paq_reasoning"]    = None

        iirs_by_entry[entry_id].append(iirs_scores)

    # Compute entry-level IIRS
    entry_iirs_scores = []
    for entry in benchmark_entries:
        eid = entry["id"]
        if eid in iirs_by_entry:
            e_score = compute_iirs_entry(iirs_by_entry[eid])
            e_score["entry_id"] = eid
            entry_iirs_scores.append(e_score)

    return entry_iirs_scores


def format_report(
    metrics: dict,
    turn_results: list[dict],
    entry_iirs_scores: list[dict],
    language: str,
    scenarios: list[str],
) -> str:
    lang_label = "Chinese (zh)" if language == "zh" else "English (en)"
    summary = compute_benchmark_summary(entry_iirs_scores)
    n = len(turn_results)

    lines = [
        "=" * 70,
        f"PhysAssistBench Evaluation Report — Doctor Agent (New 4-Task-Type Framework)",
        f"Language: {lang_label}",
        f"Scenarios: {', '.join(scenarios)}",
        f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 70,
        "",
        "## Overall IIRS",
        f"  Entries:          {summary.get('n_entries', 0)}",
        f"  Turns:            {n}",
        f"  IIRS (mean):      {summary.get('IIRS_mean', 0):.3f}",
        f"  AA (mean):        {summary.get('AA_mean', 0):.3f}",
        f"  FPTR (mean):      {summary.get('FPTR_mean', 0):.3f}  ← false positive tool rate",
        f"  MTR (mean):       {summary.get('MTR_mean', 0):.3f}  ← missing tool rate",
        "",
        "## IIRS by Subtype",
    ]
    for st, score in sorted(summary.get("IIRS_by_subtype", {}).items()):
        lines.append(f"  {st or 'None':<8}  IIRS={score:.3f}")

    lines += ["", "## Classic Metrics"]
    lines.append(f"  Task Accuracy:    {metrics.get('total_accuracy', 0):.3f}  ← fraction of turns correct")
    lines.append(f"  Session Accuracy: {metrics.get('session_accuracy', 0):.3f}  ← fraction of entries where ALL 4 turns correct")
    lines.append(f"  Sessions:         {metrics.get('n_sessions', 0)}")

    lines += ["", "## IIRS by Task Type"]
    by_tt: dict[str, list[float]] = defaultdict(list)
    for r in turn_results:
        if r.get("iirs_turn") is not None:
            by_tt[r.get("task_type", "?")].append(r["iirs_turn"])
    for tt in ["Information Lookup", "Data Gathering", "Clinical Reasoning"]:
        vals = by_tt.get(tt, [])
        if vals:
            lines.append(f"  {tt:<22}  IIRS={sum(vals)/len(vals):.3f}  (n={len(vals)})")

    # ── Rubric scores ─────────────────────────────────────────────────────────
    rubric_turns = [r for r in turn_results if r.get("rubric_score") is not None]
    if rubric_turns:
        rubric_mean = sum(r["rubric_score"] for r in rubric_turns) / len(rubric_turns)
        items_passed = sum(r.get("rubric_items_passed", 0) for r in rubric_turns)
        items_total  = sum(r.get("rubric_items_total", 0) for r in rubric_turns)
        lines += [
            "",
            "## Rubric Score",
            f"  Rubric Score (mean):  {rubric_mean:.3f}  ← fraction of rubric items passed",
            f"  Items passed / total: {items_passed} / {items_total}",
            f"  Turns evaluated:      {len(rubric_turns)}",
        ]
        # Rubric by task type
        by_tt_rubric: dict[str, list[float]] = defaultdict(list)
        for r in rubric_turns:
            by_tt_rubric[r.get("task_type", "?")].append(r["rubric_score"])
        lines.append("  By task type:")
        for tt in ["Information Lookup", "Data Gathering", "Clinical Reasoning"]:
            vals = by_tt_rubric.get(tt, [])
            if vals:
                lines.append(f"    {tt:<22}  Rubric={sum(vals)/len(vals):.3f}  (n={len(vals)})")

    # ── PAQ section (only shown if any patient tool turns exist) ─────────────
    paq_turns = [r for r in turn_results if r.get("paq") is not None]
    if paq_turns:
        paq_mean       = sum(r["paq"] for r in paq_turns) / len(paq_turns)
        faith_mean     = sum(r["paq_faithfulness"] for r in paq_turns) / len(paq_turns)
        coverage_mean  = sum(r["paq_coverage"] for r in paq_turns) / len(paq_turns)
        relevance_mean = sum(r["paq_relevance"] for r in paq_turns) / len(paq_turns)
        lines += [
            "",
            "## Patient Answer Quality (PAQ)  ← patient tool turns only",
            f"  Turns with patient tool: {len(paq_turns)}",
            f"  PAQ (mean):              {paq_mean:.3f}",
            f"    Faithfulness (F):      {faith_mean:.3f}  ← no hallucination vs patient_response",
            f"    Coverage     (C):      {coverage_mean:.3f}  ← key facts from patient_response included",
            f"    Relevance    (R):      {relevance_mean:.3f}  ← answer addresses the user question",
        ]
        # Per-tool breakdown
        by_tool: dict[str, list[float]] = defaultdict(list)
        for r in paq_turns:
            tool = r.get("gold_tool", "patient.*")
            by_tool[tool].append(r["paq"])
        if len(by_tool) > 1:
            lines.append("")
            for tool, vals in sorted(by_tool.items()):
                lines.append(f"  {tool:<38}  PAQ={sum(vals)/len(vals):.3f}  (n={len(vals)})")

    # ── Tested-model token usage ──────────────────────────────────────────────
    tok_in  = sum(r.get("tokens_in", 0)  for r in turn_results)
    tok_out = sum(r.get("tokens_out", 0) for r in turn_results)
    n_tok_turns = sum(1 for r in turn_results
                      if r.get("tokens_in", 0) or r.get("tokens_out", 0))
    if tok_in or tok_out:
        lines += [
            "",
            "## Token Usage (tested model)",
            f"  Total input tokens:   {tok_in:,}",
            f"  Total output tokens:  {tok_out:,}",
            f"  Total tokens:         {tok_in + tok_out:,}",
            f"  Turns counted:        {n_tok_turns}",
            f"  Avg tokens / turn:    "
            f"{(tok_in + tok_out) / n_tok_turns:,.0f}" if n_tok_turns else
            "  Avg tokens / turn:    0",
        ]

    lines += ["", "## IIRS by Scenario"]
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for entry in entry_iirs_scores:
        eid = entry.get("entry_id", "")
        scenario = "_".join(eid.split("_")[2:-1]) if eid else "?"
        by_scenario[scenario].append(entry.get("IIRS_entry", 0))
    for sc, vals in sorted(by_scenario.items()):
        lines.append(f"  {sc:<28}  IIRS={sum(vals)/len(vals):.3f}  (n={len(vals)})")

    lines += ["", "## Per-Turn Detail", ""]
    lines.append(
        f"  {'Entry':<32} {'T':<3} {'Type':<22} {'Sub':<5} "
        f"{'AID':<6} {'ER':<6} {'AC':<6} {'AA':<6} {'IIRS':<6}"
    )
    lines.append("  " + "-" * 95)
    for r in turn_results:
        eid = r.get("test_entry_id", "?")[-22:]
        trn = str(r.get("task_idx", "?"))
        tt = (r.get("task_type", "?") or "?")[:21]
        st = str(r.get("subtype") or "—")[:4]
        aid = f"{r.get('iirs_AID', 0):.2f}"
        er = f"{r.get('iirs_ER', 0):.2f}"
        ac = f"{r.get('iirs_AC', 0):.2f}"
        aa = f"{r.get('iirs_AA', 0):.2f}"
        iirs = f"{r.get('iirs_turn', 0):.2f}"
        lines.append(
            f"  {eid:<32} {trn:<3} {tt:<22} {st:<5} "
            f"{aid:<6} {er:<6} {ac:<6} {aa:<6} {iirs:<6}"
        )

    lines += ["", "=" * 70, "End of Report", "=" * 70]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="PhysAssistBench: Evaluate Doctor Agent with IIRS metrics (new 4-task-type framework)"
    )
    parser.add_argument(
        "--language", default="en", choices=["en", "zh"],
        help="Evaluation language: en (English) or zh (Chinese)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None, choices=SCENARIO_NAMES,
        help="Scenarios to evaluate (default: all with existing data)",
    )
    parser.add_argument(
        "--skip_generate", action="store_true",
        help="Skip data generation (use existing JSONL files)",
    )
    parser.add_argument(
        "--skip_judge", action="store_true",
        help="Skip IIRS LLM-judge scoring (use heuristic AA only)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no_system_prompt", action="store_true",
        help="Evaluate without any system prompt (zero-shot, no task-type guidance)",
    )
    parser.add_argument(
        "--judge_only", action="store_true",
        help=(
            "Skip agent execution entirely. Load existing turn_results_pre_judge.json "
            "and run only the IIRS LLM judge. Requires a prior run to have saved pre-judge results."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=(
            "Path to benchmark data directory (default: auto-detect latest data_YYYYMMDD_HHMM/ "
            "under physassistbench/; falls back to physassistbench/data/ if none found)"
        ),
    )
    parser.add_argument(
        "--rule_based", action="store_true",
        help=(
            "Use rule-based is_correct instead of LLM judge. "
            "Mirrors WildToolBench: tool name set match + JSON schema check + "
            "ROUGE-L/edit-distance argument content check. No LLM call for correctness. "
            "IIRS components (AID, ER, AC, AA) are still computed for reference."
        ),
    )
    parser.add_argument(
        "--health_literacy", default="high", choices=["low", "medium", "high"],
        help=(
            "Patient health literacy level for evaluation (low/medium/high). "
            "Patient tool turns use the specified literacy variant from "
            "patient_responses_all_literacy. Default: high (locked for consistency, "
            "since the stored default patient_response is mixed across literacy levels)."
        ),
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name from model_configs.yaml to use for evaluation (overrides default).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Explicit output directory (overrides auto-generated name). "
             "Used for distributed per-scenario runs that are merged afterward.",
    )
    parser.add_argument(
        "--model_label", default=None,
        help="Label used in the output directory name (overrides --model for naming only). "
             "Useful when the served model name differs from the actual model.",
    )
    parser.add_argument(
        "--use_explicit", action="store_true",
        help="Use tasks_en_explicit instead of tasks_en (fully explicit paraphrase queries).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip entries already successfully completed in the output_dir checkpoint.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ── Judge model configuration ─────────────────────────────────────────────
    # If GATEWAY_OPENAI_API_KEY is set, use gpt-5-mini-2025-08-07 as rubric/IIRS judge.
    # Falls back to DeepSeek if the key is absent.
    _gateway_key = os.environ.get("GATEWAY_OPENAI_API_KEY", "")
    if _gateway_key:
        try:
            from physassistbench.eval_runner import load_model_config
            from physassistbench.pipeline.agents.llm_client import configure_judge_model
            _judge_cfg = load_model_config("gpt-5-mini")
            configure_judge_model(_judge_cfg, _gateway_key)
            logger.info("Judge model: gpt-5-mini-2025-08-07 (Azure Gateway)")
        except Exception as _je:
            logger.warning(f"Could not configure GPT-5-mini judge ({_je}); using DeepSeek fallback")
    else:
        logger.info("Judge model: deepseek-v4-flash (GATEWAY_OPENAI_API_KEY not set)")

    global DATA_DIR
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    DATA_DIR = args.data_dir if args.data_dir else _latest_data_dir()
    logger.info(f"Data directory: {DATA_DIR}")
    eval_model = args.model or None
    dir_label = args.model_label or eval_model
    if args.output_dir:
        # Resolve relative to CWD (repo root in SLURM), not _PKG_DIR, to avoid
        # physassistbench/physassistbench/ path doubling when caller passes "physassistbench/..." paths.
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = _output_dir(args.language, timestamp, args.health_literacy, dir_label,
                                 use_explicit=getattr(args, "use_explicit", False))
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Generate ──────────────────────────────────────────────────────
    if args.judge_only:
        logger.info("--judge_only: skipping generation and agent execution")
    elif not args.skip_generate:
        logger.info("=" * 60)
        logger.info("Step 1/4: Generating PhysAssistBench benchmark entries...")
        logger.info("=" * 60)
        from physassistbench.generate_all import main as gen_main
        gen_main(scenarios=args.scenarios)
    else:
        logger.info("Skipping generation (--skip_generate)")

    # ── Step 2: Load benchmark ────────────────────────────────────────────────
    benchmark_entries = load_benchmark_entries(args.scenarios)
    if not benchmark_entries:
        logger.error("No benchmark entries found in physassistbench/data/ — aborting.")
        sys.exit(1)
    logger.info(f"Total entries loaded: {len(benchmark_entries)}")

    # Write the combined file into the per-run output_dir (NOT DATA_DIR) so that
    # parallel jobs sharing the same DATA_DIR don't overwrite each other's combined
    # file mid-read (race condition → JSONDecodeError).
    combined_path = os.path.join(output_dir, "combined_benchmark.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for e in benchmark_entries:
            f.write(json.dumps(_sanitize(e), ensure_ascii=False) + "\n")
    logger.info(f"Combined benchmark → {combined_path}")

    # ── Step 3: Run agent evaluation (or load existing results) ──────────────
    if args.judge_only:
        logger.info("=" * 60)
        logger.info("Step 2/4: Loading existing agent results (--judge_only)...")
        logger.info("=" * 60)
        pre_judge_path = os.path.join(output_dir, "turn_results_pre_judge.json")
        if not os.path.exists(pre_judge_path):
            logger.error(
                f"--judge_only requires existing results at {pre_judge_path}. "
                "Run without --judge_only first to generate agent outputs."
            )
            sys.exit(1)
        with open(pre_judge_path, encoding="utf-8") as f:
            turn_results = json.load(f)
        # Filter to requested scenarios if specified
        if args.scenarios:
            scenario_set = set(args.scenarios)
            turn_results = [
                r for r in turn_results
                if any(sc in r.get("test_entry_id", "") for sc in scenario_set)
            ]
        logger.info(f"Loaded {len(turn_results)} turns from {pre_judge_path}")
    else:
        logger.info("=" * 60)
        logger.info(f"Step 2/4: Running Doctor Agent (language={args.language})...")
        logger.info("=" * 60)

        if args.no_system_prompt:
            system_prompt = ""
            logger.info("System prompt: DISABLED (zero-shot evaluation)")
        else:
            system_prompt = _SYSTEM_PROMPT_V2_ZH if args.language == "zh" else _SYSTEM_PROMPT_V2_EN
            logger.info("System prompt: ENABLED")

        run_kwargs = dict(
            benchmark_path=combined_path,
            output_dir=output_dir,
            verbose=args.verbose,
            language=args.language,
            system_prompt=system_prompt,
            health_literacy=args.health_literacy,
            use_explicit=getattr(args, "use_explicit", False),
            resume=getattr(args, "resume", False),
        )
        if eval_model:
            run_kwargs["model"] = eval_model
        turn_results, raw_outputs = run_evaluation(**run_kwargs)
        logger.info(f"Evaluation complete: {len(turn_results)} turns processed")

    # ── Step 4: IIRS scoring ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3/4: Computing IIRS scores...")
    logger.info("=" * 60)

    if not args.skip_judge:
        entry_iirs_scores = compute_iirs_for_entries(
            turn_results=turn_results,
            benchmark_entries=benchmark_entries,
            language=args.language,
            use_rule_based=args.rule_based,
        )
    else:
        logger.info("Skipping LLM-judge IIRS (--skip_judge): using heuristic AA + LLM is_correct judge")
        # Heuristic: AA = tool_coverage_correct as proxy
        # is_correct: use dedicated LLM judge (lightweight binary call)
        entry_map = {e["id"]: e for e in benchmark_entries}
        for r in turn_results:
            r["iirs_AID"] = 1.0
            r["iirs_ER"] = float(r.get("tool_coverage_correct", False))
            r["iirs_AC"] = float(r.get("tool_coverage_correct", False))
            r["iirs_AA"] = float(r.get("tool_coverage_correct", False))
            r["iirs_turn"] = r["iirs_AA"]
            r["fptr"] = 1.0 if r.get("extra_tools") else 0.0
            r["mtr"] = 1.0 if r.get("missed_tools") else 0.0
            # Judge is_correct via LLM
            entry = entry_map.get(r.get("test_entry_id", ""), {})
            turn_idx = r.get("task_idx", 0)
            answer_list = entry.get("answer_list", [])
            gold_actions = answer_list[turn_idx] if turn_idx < len(answer_list) else []
            gold_answer = next(
                (str(a.get("observation", "")) for a in gold_actions
                 if a.get("action", {}).get("name") == "prepare_to_answer"), ""
            )
            pred_answer = r.get("llm_answer", "") or r.get("predicted_answer", "")
            r["is_correct"] = judge_is_correct(
                user_question=r.get("user_question", ""),
                gold_answer=gold_answer,
                pred_answer=pred_answer,
                language=args.language,
            )
        # Build stub entry scores
        from physassistbench.eval.iirs import compute_iirs_entry
        iirs_by_entry: dict = defaultdict(list)
        for r in turn_results:
            st = r.get("subtype")
            iirs_by_entry[r.get("test_entry_id", "")].append({
                "subtype": st, "task_type": r.get("task_type"),
                "AID": r["iirs_AID"], "ER": r["iirs_ER"],
                "AC": r["iirs_AC"], "AA": r["iirs_AA"],
                "IIRS_turn": r["iirs_turn"],
                "FPTR": r["fptr"], "MTR": r["mtr"],
                "false_positive_tools": r.get("extra_tools", []),
                "missing_tools": r.get("missed_tools", []),
                "applicable_components": ["AC", "AA"],
                "component_mean": r["iirs_AA"],
            })
        entry_iirs_scores = []
        for entry in benchmark_entries:
            eid = entry["id"]
            if eid in iirs_by_entry:
                e_score = compute_iirs_entry(iirs_by_entry[eid])
                e_score["entry_id"] = eid
                entry_iirs_scores.append(e_score)

    # ── Step 5: Save + report ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 4/4: Saving results and generating report...")
    logger.info("=" * 60)

    final_path = os.path.join(output_dir, "turn_results_final.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(turn_results), f, ensure_ascii=False, indent=2)

    iirs_path = os.path.join(output_dir, "iirs_scores.json")
    with open(iirs_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(entry_iirs_scores), f, ensure_ascii=False, indent=2)

    metrics = calc_accuracy(turn_results)
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(metrics), f, ensure_ascii=False, indent=2)

    target_scenarios = args.scenarios or SCENARIO_NAMES
    correctness_method = "rule_based" if args.rule_based else "llm_judge"
    logger.info(f"is_correct method: {correctness_method}")
    report = format_report(metrics, turn_results, entry_iirs_scores, args.language, target_scenarios)
    report_path = os.path.join(output_dir, "eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
