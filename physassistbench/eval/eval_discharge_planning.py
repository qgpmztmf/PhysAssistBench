"""
physassistbench/eval/eval_discharge_planning.py

Evaluation pipeline for discharge_planning benchmark entries.

Extends rubric_eval.py with two new evaluation paths:

  Write/Update turns (T3, task_type=="Write/Update"):
    The evaluated model must produce write tool calls — not just a text answer.
    Evaluation: serialize model's write calls → judge against specific-value
    rubric items grounded in gold parameter values.

  Patient turns (T2, tool_source in {"patient", "mixed"}):
    Standard rubric evaluation applies, but the EHR context block for the
    judge is replaced with note-span grounded patient facts (not FHIR bundles).

  EHR turns (T0, T1, tool_source=="ehr"):
    Identical to rubric_eval.score_turn_rubric — no change.

Public API
----------
score_discharge_planning_entry(entry, model_outputs, language="en") -> dict
    Main entry point. Takes a benchmark entry (dict loaded from .jsonl) and
    a list of model output dicts (one per turn) and returns per-turn scores
    plus an overall entry score.

score_action_turn(rubric_items, user_question, model_write_calls,
                  reference_context, history, language) -> dict
    Score an Write/Update turn given the model's write tool call sequence.

score_patient_turn(rubric_items, user_question, model_answer,
                   patient_grounding, history, language) -> dict
    Score a patient-tool turn using note-span grounding as verification context.

Model output format (one dict per turn)
---------------------------------------
EHR turn:
  {"answer": "<model's text response>", "tool_calls": [...]}

Patient turn:
  {"answer": "<model's text response>", "tool_calls": [...]}

Write/Update turn — two supported formats:
  Format A: {"tool_calls": [{"name": "MedicationRequest.create", "arguments": {...}}, ...]}
  Format B: {"answer": "<text>", "tool_calls": [...]}
  (tool_calls are what the model proposed to write — not the actual EHR bundle)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physassistbench.pipeline.agents.llm_client import llm_call, extract_json
from physassistbench.eval.rubric_eval import score_turn_rubric

logger = logging.getLogger(__name__)

# ── Write tool names (must match generate_discharge_planning.py) ──────────────

_WRITE_TOOL_NAMES = frozenset({
    "MedicationRequest.create",
    "ServiceRequest.create",
    "Flag.create",
})


# ── Serializers ───────────────────────────────────────────────────────────────

def _serialize_write_for_judge(write_tool_calls: list[dict]) -> str:
    """
    Convert a model's proposed write tool call list into a readable string
    for the LLM judge to evaluate.

    Accepts both plan format:
      {"name": "MedicationRequest.create", "arguments": {...}}
    and executed format:
      {"action": {"name": ..., "arguments": ...}, "observation": ...}
    """
    lines = []
    for item in write_tool_calls:
        # Handle executed action format
        if "action" in item:
            name = item["action"].get("name", "")
            args = item["action"].get("arguments", {})
        else:
            name = item.get("name", "")
            args = item.get("arguments", item.get("parameters", {}))

        if name in ("prepare_to_answer", "ask_user_for_required_parameters"):
            continue
        if name not in _WRITE_TOOL_NAMES:
            continue

        lines.append(f"Tool called: {name}")
        for k, v in args.items():
            if k != "subject_id" and v:
                lines.append(f"  {k}: {v}")

    return "\n".join(lines) if lines else "(no write tool calls found)"


def _serialize_gold_write(gold_params: list[dict]) -> str:
    """Format gold write parameters for inclusion in the judge prompt."""
    lines = []
    for gp in gold_params:
        lines.append(f"Gold tool: {gp.get('tool', '?')}")
        for k, v in gp.get("params", {}).items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "(gold write params not available)"


def _serialize_patient_grounding(grounding: dict) -> str:
    """Format note-span grounded patient facts as verification context."""
    if not grounding:
        return "(no patient grounding available)"
    lines = []
    for k, v in grounding.items():
        if v and not k.startswith("_"):
            label = k.replace("_", " ").title()
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


# ── Write/Update turn scorer ────────────────────────────────────────────────────────

_ACTION_JUDGE_SYSTEM_EN = """\
You are a parameter-level evaluator for an EHR write action turn.

You will be given:
1. A clinical question (the physician's write request)
2. Conversation history (for context only)
3. Gold standard write parameters (reference)
4. The model's proposed write tool call(s) — STRUCTURED JSON only
5. A list of rubric items — each is a CHECK on a single field of
   `tool_call.arguments`

Your job: for each rubric item, decide YES (1) or NO (0).

STRICT EVALUATION RULES — read carefully:
- Judge each item SOLELY by inspecting the model's structured
  `tool_call.arguments`. IGNORE any free-text "answer" the model may
  have produced; only the tool call counts here.
- The model's free-text content MUST NOT influence the score. If the
  text says the model did something but the tool_call.arguments does
  not contain that field/value, it is a FAIL.
- For drug rubric items: accept clinically equivalent alternatives
  ONLY if the rubric explicitly lists them (e.g. "Metformin or
  Glucophage"). Otherwise require exact match (case-insensitive).
- For dose rubric items: parse the numeric value and unit from the
  field; mark YES only if the value falls in the stated acceptable
  range AND the unit matches.
- For route / frequency items: accept synonyms only if the rubric
  lists them (e.g. "oral, PO, by mouth"); otherwise require exact match.
- For service-type / priority items: same — exact match unless
  synonyms are stated.
- For safety items ("the dose is NOT >= X" / "no contraindicated drug
  is ordered"): mark YES if the unsafe value is ABSENT from the tool
  call arguments.
- mg/day vs mg/dose: treat differing units as FAIL.
- If the model produced no write tool call when one was requested,
  every parameter-level item is FAIL.

Return ONLY valid JSON: {"scores": [1, 0, 1, ...], "reasoning": ["...", "...", "..."]}
Both arrays must have the same length as the rubric items list.
Reasoning must cite the specific field value(s) checked.
"""

_ACTION_JUDGE_SYSTEM_ZH = """\
你是Action写操作轮次的参数级评估员。

你将收到：
1. 临床问题（医生的写操作请求）
2. 对话历史（仅作背景参考）
3. 金标准写操作参数（参考）
4. 模型提议的写工具调用——仅看结构化JSON
5. rubric条目列表——每条针对tool_call.arguments中的某个字段做检查

任务：对每条rubric判断是（1）或否（0）。

严格评测规则——请仔细阅读：
- 仅依据模型的结构化tool_call.arguments评判，**忽略**模型可能产出的任何
  自由文本"answer"；在Action轮只有tool call计分。
- 模型的自由文本内容不得影响打分。如果文字里说做了某事，但
  tool_call.arguments中没有对应字段/值，则FAIL。
- 药物条目：仅当rubric明确列出等效药物（如"Metformin 或 Glucophage"）
  时才接受替代；否则要求完全匹配（大小写不敏感）。
- 剂量条目：解析字段中的数值和单位；仅当数值落在rubric给定的可接受
  范围内且单位匹配时判YES。
- route / frequency 条目：仅在rubric列出同义词（如"oral, PO, by mouth"）
  时接受同义；否则要求完全匹配。
- service-type / priority 条目：同上，未列同义则要求完全匹配。
- 安全性条目（"`dose`不为≥X" / "未开具禁忌药物"）：危险值缺失则判YES。
- mg/日 vs mg/次：单位不同视为FAIL。
- 若模型未产出任何写工具调用而该轮需要写操作，所有参数级条目均FAIL。

只返回有效JSON：{"scores": [1, 0, 1, ...], "reasoning": ["...", "...", "..."]}
两数组长度必须与rubric条目数相同。reasoning应指明所检查的具体字段值。
"""


def score_action_turn(
    rubric_items: list[str],
    user_question: str,
    model_write_calls: list[dict],
    reference_context: str = "",
    gold_write_params: list[dict] | None = None,
    history: list[dict] | None = None,
    language: str = "en",
) -> dict:
    """
    Score an Write/Update turn by evaluating the model's write tool calls against rubric items.

    Args:
        rubric_items:      Pre-generated rubric items (specific values + ranges).
        user_question:     The discharge action question shown to the model.
        model_write_calls: The model's proposed write tool calls (list of dicts).
        reference_context: Brief clinical context for the judge.
        gold_write_params: Gold standard write parameters (for judge transparency).
        history:           Prior conversation turns.
        language:          "en" | "zh"

    Returns:
        {scores, reasoning, rubric_score, items_passed, items_total}
    """
    if not rubric_items:
        return {
            "scores": [], "reasoning": [],
            "rubric_score": None, "items_passed": 0, "items_total": 0,
        }

    is_zh = language == "zh"
    sys_prompt = _ACTION_JUDGE_SYSTEM_ZH if is_zh else _ACTION_JUDGE_SYSTEM_EN

    model_write_str = _serialize_write_for_judge(model_write_calls)
    gold_str = _serialize_gold_write(gold_write_params or [])

    hist_str = ""
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "").upper()
            content = str(msg.get("content", ""))[:300]
            hist_str += f"[{role}]: {content}\n"

    rubric_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(rubric_items))

    if is_zh:
        user_prompt = (
            f"临床问题：{user_question}\n\n"
            f"对话历史：\n{hist_str}\n"
            f"临床背景：{reference_context[:300]}\n\n"
            f"金标准写操作参数：\n{gold_str}\n\n"
            f"模型提议的写工具调用：\n{model_write_str}\n\n"
            f"Rubric条目：\n{rubric_str}\n\n"
            "对每条rubric条目评分（JSON）："
        )
    else:
        user_prompt = (
            f"Clinical question: {user_question}\n\n"
            f"Conversation history:\n{hist_str}\n"
            f"Clinical context: {reference_context[:300]}\n\n"
            f"Gold standard write parameters:\n{gold_str}\n\n"
            f"Model's proposed write tool calls:\n{model_write_str}\n\n"
            f"Rubric items:\n{rubric_str}\n\n"
            "Score each rubric item (JSON):"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = llm_call(messages, temperature=0.1, max_tokens=2000)
        result = extract_json(raw)

        scores = result.get("scores", [])
        reasoning = result.get("reasoning", [""] * len(scores))
        n = len(rubric_items)
        if len(scores) != n:
            logger.warning(f"score_action_turn: got {len(scores)} scores for {n} items — padding")
            scores = (scores + [0] * n)[:n]
            reasoning = (reasoning + [""] * n)[:n]

        scores = [int(bool(s)) for s in scores]
        items_passed = sum(scores)

        return {
            "scores": scores,
            "reasoning": reasoning,
            "rubric_score": items_passed / n,
            "items_passed": items_passed,
            "items_total": n,
        }

    except Exception as exc:
        logger.warning(f"score_action_turn failed ({exc})")
        n = len(rubric_items)
        return {
            "scores": [0] * n,
            "reasoning": [f"scoring error: {exc}"] * n,
            "rubric_score": None,
            "items_passed": 0,
            "items_total": n,
        }


# ── Patient turn scorer ───────────────────────────────────────────────────────

_PATIENT_JUDGE_SYSTEM_EN = """\
You are a clinical QA evaluator for an EHR discharge planning benchmark.

You will be given:
1. A clinical question (involving patient interview information)
2. Conversation history (prior turns, if any)
3. Note-span grounded patient facts (extracted from MIMIC-IV discharge notes — these are
   the verifiable ground-truth facts for the patient's adherence, functional status, etc.)
4. The model's response to evaluate
5. A list of rubric items

Your job: for each rubric item, decide YES (1) or NO (0).

EVALUATION RULES:
- Judge each item INDEPENDENTLY based on the model's response.
- Patient facts are the ground-truth. Use them to verify factual accuracy.
- Accept paraphrases of grounded facts — verbatim repetition is not required.
- For functional status items: equivalent clinical descriptions are acceptable.
- For recommendations based on patient info: accept clinically equivalent phrasing.

Return ONLY valid JSON: {"scores": [1, 0, 1, ...], "reasoning": ["...", "...", "..."]}
"""

_PATIENT_JUDGE_SYSTEM_ZH = """\
你是一个EHR出院规划基准测试的临床QA评估员。

你将收到：
1. 临床问题（涉及患者访谈信息）
2. 对话历史（如有前序轮次）
3. 笔记锚定的患者事实（从MIMIC-IV出院记录中提取——这些是患者依从性、功能状态等可核实的基本事实）
4. 待评测模型的回答
5. rubric条目列表

你的任务：对每条rubric条目判断是（1）或否（0）。

评测规则：
- 仅根据模型回答独立判断每条条目。
- 患者事实是基本事实，用于核实事实准确性。
- 接受对锚定事实的释义——不要求逐字重复。
- 功能状态条目：接受等效的临床描述。
- 基于患者信息的建议：接受临床等效的不同表述。

只返回有效JSON：{"scores": [1, 0, 1, ...], "reasoning": ["...", "...", "..."]}
"""


def score_patient_turn(
    rubric_items: list[str],
    user_question: str,
    model_answer: str,
    patient_grounding: dict,
    history: list[dict] | None = None,
    language: str = "en",
) -> dict:
    """
    Score a patient-tool turn using note-span grounded facts as verification context.

    Args:
        rubric_items:       Rubric items for this turn.
        user_question:      The question shown to the model.
        model_answer:       The model's text response.
        patient_grounding:  Note-span grounded facts dict (from entry["note_grounding"]).
        history:            Prior conversation turns.
        language:           "en" | "zh"
    """
    if not rubric_items:
        return {
            "scores": [], "reasoning": [],
            "rubric_score": None, "items_passed": 0, "items_total": 0,
        }

    is_zh = language == "zh"
    sys_prompt = _PATIENT_JUDGE_SYSTEM_ZH if is_zh else _PATIENT_JUDGE_SYSTEM_EN

    grounding_str = _serialize_patient_grounding(patient_grounding)

    hist_str = ""
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "").upper()
            content = str(msg.get("content", ""))[:300]
            hist_str += f"[{role}]: {content}\n"

    rubric_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(rubric_items))

    if is_zh:
        user_prompt = (
            f"临床问题：{user_question}\n\n"
            f"对话历史：\n{hist_str}\n"
            f"笔记锚定患者事实（可核实基本事实）：\n{grounding_str}\n\n"
            f"模型回答：\n{model_answer}\n\n"
            f"Rubric条目：\n{rubric_str}\n\n"
            "对每条rubric条目评分（JSON）："
        )
    else:
        user_prompt = (
            f"Clinical question: {user_question}\n\n"
            f"Conversation history:\n{hist_str}\n"
            f"Note-span grounded patient facts (verifiable ground truth):\n{grounding_str}\n\n"
            f"Model response:\n{model_answer}\n\n"
            f"Rubric items:\n{rubric_str}\n\n"
            "Score each rubric item (JSON):"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = llm_call(messages, temperature=0.1, max_tokens=2000)
        result = extract_json(raw)

        scores = result.get("scores", [])
        reasoning = result.get("reasoning", [""] * len(scores))
        n = len(rubric_items)
        if len(scores) != n:
            logger.warning(f"score_patient_turn: got {len(scores)} scores for {n} items — padding")
            scores = (scores + [0] * n)[:n]
            reasoning = (reasoning + [""] * n)[:n]

        scores = [int(bool(s)) for s in scores]
        items_passed = sum(scores)

        return {
            "scores": scores,
            "reasoning": reasoning,
            "rubric_score": items_passed / n,
            "items_passed": items_passed,
            "items_total": n,
        }

    except Exception as exc:
        logger.warning(f"score_patient_turn failed ({exc})")
        n = len(rubric_items)
        return {
            "scores": [0] * n,
            "reasoning": [f"scoring error: {exc}"] * n,
            "rubric_score": None,
            "items_passed": 0,
            "items_total": n,
        }


# ── Entry-level scorer ────────────────────────────────────────────────────────

def score_discharge_planning_entry(
    entry: dict,
    model_outputs: list[dict],
    language: str = "en",
) -> dict:
    """
    Score all 4 turns of a discharge_planning benchmark entry.

    Routes each turn to the appropriate scorer:
      T0 [EHR Information Lookup]  → score_turn_rubric (standard EHR path)
      T1 [EHR Data Gathering]     → score_turn_rubric (standard EHR path)
      T2 [Patient/mixed]  → score_patient_turn (note-span grounding path)
      T3 [Action]         → score_action_turn (write tool call path)

    Args:
        entry:         One benchmark entry loaded from the discharge_planning .jsonl
        model_outputs: List of model output dicts, one per turn. See module docstring
                       for the expected format.
        language:      "en" | "zh"

    Returns:
        {
          "entry_id":           str,
          "turn_results":       [per-turn result dicts],
          "entry_rubric_score": float,   # mean over turns with rubrics
          "turn_breakdown":     [{"turn": i, "type": ..., "source": ..., "score": ...}]
        }
    """
    rubrics = entry.get("rubrics", [])
    turn_data_list = entry.get("turn_data", [])
    note_grounding = entry.get("note_grounding", {})
    write_gold_params = entry.get("write_gold_params", [])
    task_sequence = entry.get("task_sequence", ["Information Lookup", "Data Gathering", "Data Gathering", "Write/Update"])
    tool_sources = entry.get("tool_sources", ["ehr", "ehr", "mixed", "write"])
    ehr_snapshot = entry.get("context", {}).get("ehr_snapshot", "")

    turn_results = []
    scored_scores = []
    history: list[dict] = []

    for turn_idx, model_out in enumerate(model_outputs):
        rubric_items = rubrics[turn_idx] if turn_idx < len(rubrics) else []
        turn_td = turn_data_list[turn_idx] if turn_idx < len(turn_data_list) else {}
        task_type = task_sequence[turn_idx] if turn_idx < len(task_sequence) else "Information Lookup"
        tool_source = tool_sources[turn_idx] if turn_idx < len(tool_sources) else "ehr"

        user_q = turn_td.get("user_question", model_out.get("question", ""))
        executed_actions = turn_td.get("executed_actions", [])

        if task_type == "Write/Update":
            # Write tool evaluation path
            model_write_calls = model_out.get("tool_calls", [])
            result = score_action_turn(
                rubric_items=rubric_items,
                user_question=user_q,
                model_write_calls=model_write_calls,
                reference_context=ehr_snapshot[:400],
                gold_write_params=write_gold_params,
                history=history,
                language=language,
            )
            result["eval_path"] = "action"

        elif tool_source in ("patient", "mixed"):
            # Patient interview evaluation path
            model_answer = model_out.get("answer", "")
            result = score_patient_turn(
                rubric_items=rubric_items,
                user_question=user_q,
                model_answer=model_answer,
                patient_grounding=note_grounding,
                history=history,
                language=language,
            )
            result["eval_path"] = "patient"

        else:
            # Standard EHR read evaluation path
            model_answer = model_out.get("answer", "")
            result = score_turn_rubric(
                rubric_items=rubric_items,
                user_question=user_q,
                model_answer=model_answer,
                executed_actions=executed_actions,
                history=history,
                language=language,
            )
            result["eval_path"] = "ehr"

        result["turn_idx"] = turn_idx
        result["task_type"] = task_type
        result["tool_source"] = tool_source
        turn_results.append(result)

        if result.get("rubric_score") is not None:
            scored_scores.append(result["rubric_score"])

        # Advance history with the model's response for subsequent turns
        if task_type == "Write/Update":
            # For action turns, represent the model's action as a text summary
            write_calls = model_out.get("tool_calls", [])
            summary = _serialize_write_for_judge(write_calls) or model_out.get("answer", "(action turn)")
            history.append({"role": "user", "content": user_q})
            history.append({"role": "assistant", "content": summary})
        else:
            history.append({"role": "user", "content": user_q})
            history.append({"role": "assistant", "content": model_out.get("answer", "")})

    entry_score = sum(scored_scores) / len(scored_scores) if scored_scores else None

    return {
        "entry_id": entry.get("entry_id", ""),
        "turn_results": turn_results,
        "entry_rubric_score": entry_score,
        "turn_breakdown": [
            {
                "turn": r["turn_idx"],
                "type": r["task_type"],
                "source": r["tool_source"],
                "eval_path": r.get("eval_path"),
                "score": r.get("rubric_score"),
                "items_passed": r.get("items_passed", 0),
                "items_total": r.get("items_total", 0),
            }
            for r in turn_results
        ],
    }


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_file(
    benchmark_file: str,
    model_output_file: str,
    output_file: str,
    language: str = "en",
) -> None:
    """
    Evaluate a full JSONL benchmark file against model outputs.

    benchmark_file:    Path to discharge_planning JSONL (from generate_discharge_planning.py)
    model_output_file: Path to model output JSONL.
                       Each line: {"entry_id": ..., "turns": [per-turn model output dicts]}
    output_file:       Where to write per-entry results JSONL.
    """
    # Index model outputs by entry_id
    model_out_by_id: dict[str, list[dict]] = {}
    with open(model_output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            eid = obj.get("entry_id", "")
            model_out_by_id[eid] = obj.get("turns", [])

    all_entry_scores = []

    with open(benchmark_file, encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            eid = entry.get("entry_id", "")
            model_outputs = model_out_by_id.get(eid, [])

            if not model_outputs:
                logger.warning(f"No model outputs for {eid} — skipping")
                continue

            result = score_discharge_planning_entry(
                entry=entry,
                model_outputs=model_outputs,
                language=language,
            )

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            score = result.get("entry_rubric_score")
            if score is not None:
                all_entry_scores.append(score)
            logger.info(
                f"  {eid}: score={score:.3f if score else 'N/A'}  "
                f"breakdown={[(r['turn'], r['score']) for r in result['turn_breakdown']]}"
            )

    if all_entry_scores:
        mean = sum(all_entry_scores) / len(all_entry_scores)
        print(f"\nEvaluated {len(all_entry_scores)} entries")
        print(f"Mean entry rubric score: {mean:.4f}  ({mean*100:.1f}%)")
    else:
        print("No entries scored.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate discharge_planning benchmark entries"
    )
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark JSONL (from generate_discharge_planning.py)")
    parser.add_argument("--model_outputs", required=True,
                        help="Model output JSONL (one line per entry, turns list)")
    parser.add_argument("--out", default="dp_eval_results.jsonl",
                        help="Output JSONL for per-entry results")
    parser.add_argument("--language", default="en", choices=["en", "zh"])
    args = parser.parse_args()

    evaluate_file(
        benchmark_file=args.benchmark,
        model_output_file=args.model_outputs,
        output_file=args.out,
        language=args.language,
    )


if __name__ == "__main__":
    _main()
