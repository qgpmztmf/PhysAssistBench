"""
physassistbench/eval/iirs.py — IIRS (Implicit Information Recovery Score) evaluation.

Decomposes per-turn evaluation into 4 sub-scores:
  AID : Antecedent Identification  (EA / AE subtypes)
  ER  : Ellipsis Recovery          (PE / AU subtypes)
  AC  : Argument Completion        (AU / Information Lookup / Data Gathering)
  AA  : Answer Accuracy            (all subtypes)

IIRS_turn  = mean(applicable_components) × AA
IIRS_entry = mean(IIRS_turn for all turns)

Additional metrics:
  - False Positive Tool Rate (FPTR): tool called when it should not be
  - Missing Tool Rate (MTR):         tool not called when it should be
  - Workup Mode Accuracy (WMA):      parallel vs adaptive correctly executed
  - Cross-source Reconciliation:     Mixed-source conflict detection (EHR vs Patient)

See docs/benchmark_redesign_integrated_v6.md §6.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physassistbench.pipeline.agents.llm_client import judge_llm_call
from physassistbench.eval.rule_checker import rule_based_is_correct  # noqa: F401 — re-exported for callers

# ── LLM-based sub-score evaluators ───────────────────────────────────────────

_SYSTEM_EVAL = """You are a clinical NLP evaluator for a multi-turn EHR question-answering benchmark.
Score each criterion on a scale of 0.0 to 1.0. Be strict but fair.
Return ONLY a JSON object with the requested keys and float values."""


def _llm_score(prompt: str) -> dict[str, float]:
    """Call the LLM evaluator and parse a JSON score dict."""
    messages = [
        {"role": "system", "content": _SYSTEM_EVAL},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = judge_llm_call(messages, temperature=0.0, max_tokens=4000)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        return {}


# ── Sub-score functions ───────────────────────────────────────────────────────

def score_AID(
    user_question: str,
    history: list[dict],
    gold_antecedent: str,
    predicted_answer: str,
    subtype: str,
) -> float:
    """
    Antecedent Identification (AID) — applicable to EA and AE subtypes.

    Measures whether the system correctly identified WHICH entity/event/situation
    the implicit reference in the user question points to.

    Returns 0.0–1.0.
    """
    if subtype not in ("EA", "AE"):
        return 1.0  # not applicable → neutral

    history_str = "\n".join(
        f"[{m['role'].upper()}]: {m['content'][:300]}"
        for m in history[-6:]
    )

    prompt = f"""Evaluate Antecedent Identification (AID).

User question (contains implicit reference):
"{user_question}"

Prior conversation (last few turns):
{history_str}

Gold antecedent (what the implicit reference SHOULD resolve to):
"{gold_antecedent}"

System response:
"{predicted_answer[:500]}"

Score AID (0.0–1.0):
- 1.0: The system's response clearly shows it identified the correct antecedent.
- 0.5: Partially correct (right entity type, wrong instance or partially identified).
- 0.0: Wrong antecedent identified, or the system ignored the implicit reference.

Return JSON: {{"AID": <float>}}"""
    result = _llm_score(prompt)
    return float(result.get("AID", 0.0))


def score_ER(
    user_question: str,
    history: list[dict],
    gold_tool_calls: list[str],
    predicted_tool_calls: list[str],
    subtype: str,
) -> float:
    """
    Ellipsis Recovery (ER) — applicable to PE and AU subtypes.

    Measures whether the system recovered the OMITTED PREDICATE (PE) or
    UNDERSPECIFIED ARGUMENT (AU) and called the correct tool(s).

    Returns 0.0–1.0.
    """
    if subtype not in ("PE", "AU"):
        return 1.0  # not applicable → neutral

    gold_set = set(gold_tool_calls)
    pred_set = set(predicted_tool_calls)

    if not gold_set:
        return 1.0

    overlap = gold_set & pred_set
    precision = len(overlap) / len(pred_set) if pred_set else 0.0
    recall = len(overlap) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)  # F1


def score_AC(
    user_question: str,
    gold_tool_args: dict[str, Any],
    predicted_tool_args: dict[str, Any],
) -> float:
    """
    Argument Completion (AC) — applicable to AU, Information Lookup, Data Gathering.

    Measures whether the system correctly filled in underspecified arguments
    (e.g., correct n_results, time_window, item_name).

    Returns 0.0–1.0 as fraction of correctly filled arguments.
    """
    if not gold_tool_args:
        return 1.0

    correct = 0
    total = 0
    for key, gold_val in gold_tool_args.items():
        if key in ("subject_id", "hadm_id", "session_id"):
            continue  # these are always known
        total += 1
        pred_val = predicted_tool_args.get(key)
        if pred_val is not None:
            # Flexible match: string normalization
            if str(gold_val).lower().strip() == str(pred_val).lower().strip():
                correct += 1
            elif isinstance(gold_val, (int, float)) and isinstance(pred_val, (int, float)):
                if abs(float(gold_val) - float(pred_val)) < 1e-6:
                    correct += 1

    return correct / total if total > 0 else 1.0


def score_AA(
    user_question: str,
    history: list[dict],
    gold_answer: str,
    predicted_answer: str,
    task_type: str,
    language: str = "en",
) -> float:
    """
    Answer Accuracy (AA) — applicable to all subtypes.

    For tool-calling turns: uses LLM judge to compare final answers.
    Returns 0.0–1.0.
    """
    if not predicted_answer or not gold_answer:
        return 0.0

    history_str = "\n".join(
        f"[{m['role'].upper()}]: {m['content'][:200]}"
        for m in history[-4:]
    )

    if language == "zh":
        prompt = f"""评估最终答案准确性（AA）。

用户问题："{user_question}"

对话历史（最近几轮）：
{history_str}

标准答案：
"{gold_answer[:600]}"

系统预测答案：
"{predicted_answer[:600]}"

评分标准（0.0–1.0）：
- 1.0：完全正确，临床意义等价
- 0.7–0.9：基本正确，有轻微遗漏或措辞差异
- 0.4–0.6：部分正确，关键信息正确但有误差或遗漏
- 0.1–0.3：相关但主要内容错误
- 0.0：完全错误或无法回答

返回JSON：{{"AA": <float>}}"""
    else:
        prompt = f"""Evaluate Answer Accuracy (AA).

User question: "{user_question}"

Conversation history (recent turns):
{history_str}

Gold answer:
"{gold_answer[:600]}"

Predicted answer:
"{predicted_answer[:600]}"

Score AA (0.0–1.0):
- 1.0: Fully correct, clinically equivalent
- 0.7–0.9: Mostly correct, minor omissions or phrasing differences
- 0.4–0.6: Partially correct, key information present but with errors or gaps
- 0.1–0.3: Relevant but mostly wrong
- 0.0: Completely wrong or refused to answer

Return JSON: {{"AA": <float>}}"""

    result = _llm_score(prompt)
    return float(result.get("AA", 0.0))


# ── IIRS composite ────────────────────────────────────────────────────────────

def compute_iirs_turn(
    user_question: str,
    history: list[dict],
    gold_turn: dict,
    pred_turn: dict,
    subtype: str | None,
    task_type: str,
    language: str = "en",
) -> dict:
    """
    Compute IIRS for a single turn.

    gold_turn: {
        "tool_calls": list[str],         # gold tool names called
        "tool_args": dict,               # gold args for primary tool call
        "answer": str,                   # gold final answer text
        "antecedent": str,               # gold antecedent (for EA/AE)
    }
    pred_turn: {
        "tool_calls": list[str],         # predicted tool names
        "tool_args": dict,               # predicted args
        "answer": str,                   # predicted final answer
    }

    Returns dict with component scores and IIRS_turn.
    """
    gold_tools = gold_turn.get("tool_calls", [])
    gold_args = gold_turn.get("tool_args", {})
    gold_answer = gold_turn.get("answer", "")
    gold_antecedent = gold_turn.get("antecedent", "")

    pred_tools = pred_turn.get("tool_calls", [])
    pred_args = pred_turn.get("tool_args", {})
    pred_answer = pred_turn.get("answer", "")

    # Compute sub-scores
    aid = score_AID(user_question, history, gold_antecedent, pred_answer, subtype or "")
    er = score_ER(user_question, history, gold_tools, pred_tools, subtype or "")
    ac = score_AC(user_question, gold_args, pred_args)
    aa = score_AA(user_question, history, gold_answer, pred_answer, task_type, language)

    # Select applicable components for this subtype
    if subtype == "EA":
        applicable = [aid, aa]
        applicable_names = ["AID", "AA"]
    elif subtype == "PE":
        applicable = [er, aa]
        applicable_names = ["ER", "AA"]
    elif subtype == "AU":
        applicable = [er, ac, aa]
        applicable_names = ["ER", "AC", "AA"]
    elif subtype == "AE":
        applicable = [aid, aa]
        applicable_names = ["AID", "AA"]
    else:
        # No subtype (turn 0): only AA + AC
        applicable = [ac, aa]
        applicable_names = ["AC", "AA"]

    component_mean = sum(applicable) / len(applicable) if applicable else 0.0
    iirs_turn = component_mean * aa  # AA as overall quality gate

    # Tool use metrics
    gold_set = set(gold_tools)
    pred_set = set(pred_tools)
    false_positive_tools = pred_set - gold_set
    missing_tools = gold_set - pred_set

    return {
        "subtype": subtype,
        "task_type": task_type,
        "AID": aid,
        "ER": er,
        "AC": ac,
        "AA": aa,
        "applicable_components": applicable_names,
        "component_mean": component_mean,
        "IIRS_turn": iirs_turn,
        "false_positive_tools": list(false_positive_tools),
        "missing_tools": list(missing_tools),
        "FPTR": 1.0 if false_positive_tools else 0.0,
        "MTR": 1.0 if missing_tools else 0.0,
    }


def compute_iirs_entry(turn_scores: list[dict]) -> dict:
    """Aggregate per-turn IIRS scores for one benchmark entry."""
    if not turn_scores:
        return {}

    n = len(turn_scores)
    iirs_turns = [t["IIRS_turn"] for t in turn_scores]
    aa_turns = [t["AA"] for t in turn_scores]
    fptr_turns = [t["FPTR"] for t in turn_scores]
    mtr_turns = [t["MTR"] for t in turn_scores]

    # Per-subtype IIRS breakdown
    by_subtype: dict[str, list[float]] = {}
    for t in turn_scores:
        st = t.get("subtype") or "None"
        by_subtype.setdefault(st, []).append(t["IIRS_turn"])

    return {
        "IIRS_entry": sum(iirs_turns) / n,
        "AA_avg": sum(aa_turns) / n,
        "FPTR_avg": sum(fptr_turns) / n,
        "MTR_avg": sum(mtr_turns) / n,
        "n_turns": n,
        "IIRS_by_subtype": {k: sum(v) / len(v) for k, v in by_subtype.items()},
        "turn_scores": turn_scores,
    }


# ── Patient Answer Quality (PAQ) ─────────────────────────────────────────────

def score_patient_answer_quality(
    user_question: str,
    patient_response: str,
    llm_answer: str,
    language: str = "en",
) -> dict:
    """
    RAG-style evaluation for patient tool turns.

    Evaluates the final generated answer against the fixed patient_response
    (the ground-truth context returned by the patient tool) on three dimensions:

      F  — Faithfulness   : every claim in llm_answer is supported by patient_response
      C  — Coverage       : key clinical facts in patient_response appear in llm_answer
      R  — Relevance      : the answer actually addresses the user's question

    PAQ (Patient Answer Quality) = (F + C + R) / 3

    Returns dict with keys: Faithfulness, Coverage, Relevance, PAQ, reasoning.
    Returns all zeros if patient_response or llm_answer is empty.
    """
    if not patient_response or not llm_answer:
        return {"Faithfulness": 0.0, "Coverage": 0.0, "Relevance": 0.0,
                "PAQ": 0.0, "reasoning": "empty input"}

    if language == "zh":
        prompt = f"""你是临床NLP评估专家，请评估AI系统对患者访谈工具结果的最终回答质量。

用户问题："{user_question}"

患者实际回答（固定ground truth上下文）：
"{patient_response}"

AI系统最终回答：
"{llm_answer}"

请从以下3个维度打分（0.0–1.0）：

1. FAITHFULNESS（忠实性 F）：AI回答中的每一个事实性陈述，是否都有患者原话的支撑？
   - 1.0：所有陈述均可从患者原话中找到依据
   - 0.5：部分陈述有依据，但有数值改动（如8/10→9/10）或位置描述偏差
   - 0.0：存在明显幻觉（编造了患者未说的症状、数值、时间等）

2. COVERAGE（覆盖率 C）：患者原话中的关键临床信息，有多少出现在AI回答里？
   - 关键信息包括：症状性质、严重程度（数字评分）、放射部位、加重/缓解因素、时间过程
   - 1.0：所有关键信息均被提及
   - 0.5：提及了主要信息，但遗漏了部分细节
   - 0.0：大量关键信息缺失

3. RELEVANCE（相关性 R）：AI回答是否真正回应了用户的问题？
   - 1.0：直接、完整地回答了问题
   - 0.5：提供了相关信息但未聚焦于问题核心
   - 0.0：答非所问

返回JSON：{{"F": <float>, "C": <float>, "R": <float>, "reasoning": "<一句话说明主要扣分原因>"}}"""
    else:
        prompt = f"""You are a clinical NLP evaluator. Assess the quality of an AI system's
final answer for a patient interview tool turn.

User question: "{user_question}"

Patient's actual response (fixed ground-truth context):
"{patient_response}"

AI system's final answer:
"{llm_answer}"

Score on 3 dimensions (0.0–1.0 each):

1. FAITHFULNESS (F): Are all factual claims in the AI answer supported by the patient response?
   - 1.0: Every claim is directly backed by the patient's words
   - 0.5: Most claims are backed but with minor alterations (e.g. 8/10 → 9/10, jaw → back)
   - 0.0: Hallucination present (symptoms, numbers, or timeline the patient did not mention)

2. COVERAGE (C): What fraction of key clinical facts in the patient response appear in the AI answer?
   - Key facts: symptom character, severity score, radiation site, aggravating/relieving factors, onset timing
   - 1.0: All key facts mentioned
   - 0.5: Main facts present, minor details omitted
   - 0.0: Major facts missing

3. RELEVANCE (R): Does the AI answer actually address the user's question?
   - 1.0: Directly and completely answers the question
   - 0.5: Provides related info but misses the core of what was asked
   - 0.0: Off-topic or non-responsive

Return JSON: {{"F": <float>, "C": <float>, "R": <float>, "reasoning": "<one sentence on main deduction>"}}"""

    result = _llm_score(prompt)
    f = float(result.get("F", 0.0))
    c = float(result.get("C", 0.0))
    r_val = float(result.get("R", 0.0))
    paq = (f + c + r_val) / 3.0
    return {
        "Faithfulness": round(f, 4),
        "Coverage": round(c, 4),
        "Relevance": round(r_val, 4),
        "PAQ": round(paq, 4),
        "reasoning": result.get("reasoning", ""),
    }


# ── Cross-source Reconciliation ───────────────────────────────────────────────

def score_cross_source_reconciliation(
    user_question: str,
    ehr_observation: str,
    patient_observation: str,
    predicted_answer: str,
    language: str = "en",
) -> dict:
    """
    Evaluate Mixed-source Data Gathering (EHR + Patient) for conflict detection.

    Checks:
      1. Source Coverage (SC): did agent call BOTH EHR and Patient tools?
      2. Conflict Detection (CD): did agent identify conflicting information?
      3. Source Prioritization (SP): did agent correctly prioritise EHR over patient self-report?
      4. Synthesis Quality (SQ): did agent integrate both sources in the final answer?

    Returns dict with SC / CD / SP / SQ scores (0.0–1.0) and composite.
    """
    if language == "zh":
        prompt = f"""评估跨来源信息矛盾处理（Cross-source Reconciliation）。

用户问题："{user_question}"

EHR工具返回结果：
"{ehr_observation[:400]}"

患者访谈返回结果：
"{patient_observation[:400]}"

系统最终回答：
"{predicted_answer[:600]}"

请对以下4个维度各打分（0.0–1.0）：
- SC（来源覆盖）：回答是否同时参考了EHR和患者两个来源？
- CD（矛盾识别）：回答是否识别出两来源之间的信息矛盾（如有）？
- SP（来源优先级）：发生冲突时，是否正确优先采信EHR（客观来源）？
- SQ（综合质量）：最终答案是否合理整合了两个来源的信息？

返回JSON：{{"SC": <float>, "CD": <float>, "SP": <float>, "SQ": <float>}}"""
    else:
        prompt = f"""Evaluate Cross-source Reconciliation for a Mixed-source Data Gathering turn.

User question: "{user_question}"

EHR tool observation:
"{ehr_observation[:400]}"

Patient interview observation:
"{patient_observation[:400]}"

System final answer:
"{predicted_answer[:600]}"

Score each dimension (0.0–1.0):
- SC (Source Coverage): Did the answer reference BOTH EHR and Patient sources?
- CD (Conflict Detection): Did the answer identify conflicting information between sources?
- SP (Source Prioritization): In case of conflict, did the agent correctly prioritize EHR (objective) over patient self-report?
- SQ (Synthesis Quality): Did the final answer integrate both sources into a coherent response?

Return JSON: {{"SC": <float>, "CD": <float>, "SP": <float>, "SQ": <float>}}"""

    result = _llm_score(prompt)
    sc = float(result.get("SC", 0.0))
    cd = float(result.get("CD", 0.0))
    sp = float(result.get("SP", 0.0))
    sq = float(result.get("SQ", 0.0))
    composite = (sc + cd + sp + sq) / 4.0

    return {"SC": sc, "CD": cd, "SP": sp, "SQ": sq, "CSR_composite": composite}


# ── Aggregate utilities ───────────────────────────────────────────────────────

def compute_benchmark_summary(all_entry_scores: list[dict]) -> dict:
    """
    Aggregate IIRS scores across all benchmark entries.

    all_entry_scores: list of dicts returned by compute_iirs_entry()
    """
    if not all_entry_scores:
        return {}

    n = len(all_entry_scores)
    iirs_vals = [e["IIRS_entry"] for e in all_entry_scores]
    aa_vals = [e["AA_avg"] for e in all_entry_scores]
    fptr_vals = [e["FPTR_avg"] for e in all_entry_scores]
    mtr_vals = [e["MTR_avg"] for e in all_entry_scores]

    # Per-subtype
    subtype_scores: dict[str, list[float]] = {}
    for entry in all_entry_scores:
        for st, score in entry.get("IIRS_by_subtype", {}).items():
            subtype_scores.setdefault(st, []).append(score)

    return {
        "n_entries": n,
        "IIRS_mean": sum(iirs_vals) / n,
        "IIRS_min": min(iirs_vals),
        "IIRS_max": max(iirs_vals),
        "AA_mean": sum(aa_vals) / n,
        "FPTR_mean": sum(fptr_vals) / n,
        "MTR_mean": sum(mtr_vals) / n,
        "IIRS_by_subtype": {
            k: sum(v) / len(v) for k, v in subtype_scores.items()
        },
    }
