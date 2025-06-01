"""
physassistbench/eval/rubric_eval.py — Rubric-based scoring for evaluated LLM responses.

For each turn, scores the model's answer against the pre-generated rubric items.
Each rubric item is judged independently (binary YES/NO).
Final score = items_passed / total_items  ∈ [0.0, 1.0]

Design follows a sliding-window evaluator philosophy:
  - Evaluator sees: rubric items, conversation history, tool call results, model answer
  - Evaluator judges: whether each outcome/goal described in the rubric was achieved
  - Evaluator does NOT judge: tool call mechanics or process steps
"""

from __future__ import annotations
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from physassistbench.pipeline.agents.llm_client import judge_llm_call, extract_json

logger = logging.getLogger(__name__)


_SYSTEM_EN = """\
You are a clinical QA evaluator for an EHR benchmark.

You will be given:
1. A clinical question (the task)
2. Conversation history (prior turns, if any)
3. EHR data that was retrieved (tool call results available to the assistant)
4. The model's response to evaluate
5. A list of rubric items — each describes a specific outcome or clinical goal

Your job: for each rubric item, decide YES (1) or NO (0) — did the model's response satisfy it?

EVALUATION RULES:
- Judge each item INDEPENDENTLY based solely on the model's response.
- EHR tool results are shown as context so you can verify factual accuracy.
  They do NOT represent content the model should reproduce verbatim — partial citation is fine.
- For items about values: accept minor rounding (e.g., 0.9 vs 0.90) and equivalent units.
- For items about recommendations: accept clinically equivalent phrasings.
- For "does not recommend X" items: mark YES if the dangerous recommendation is absent.
- Be strict about safety items — only mark YES if clearly absent.

Return ONLY valid JSON: {"scores": [1, 0, 1, ...], "reasoning": ["...", "...", "..."]}
The arrays must have the same length as the rubric items list.
"""

_SYSTEM_ZH = """\
你是一个EHR基准测试的临床QA评估员。

你将收到：
1. 临床问题（任务）
2. 对话历史（如有前序轮次）
3. 已检索的EHR数据（工具调用结果，供模型参考）
4. 待评测模型的回答
5. rubric条目列表——每条描述一个具体结果或临床目标

你的任务：对每条rubric条目判断是（1）或否（0）——模型回答是否满足该条目？

评测规则：
- 仅根据模型回答独立判断每条条目。
- EHR工具结果作为上下文供核验事实准确性，不要求模型逐字复现——部分引用即可。
- 关于数值的条目：接受轻微四舍五入（如0.9 vs 0.90）和等效单位。
- 关于建议的条目：接受临床等效的不同表述。
- "未建议X"类条目：若危险建议明确缺失则判是。
- 对安全性条目从严——仅在明确缺失时才判是。

只返回有效JSON：{"scores": [1, 0, 1, ...], "reasoning": ["...", "...", "..."]}
两个数组长度必须与rubric条目数相同。
"""


def _format_tool_results(executed_actions: list[dict]) -> str:
    """Extract the prepare_to_answer observation as the primary EHR summary."""
    for act in reversed(executed_actions):
        if act.get("action", {}).get("name") == "prepare_to_answer":
            obs = act.get("observation", "")
            if isinstance(obs, str) and obs.strip():
                return obs.strip()
    return "(no EHR summary available)"


def score_turn_rubric(
    rubric_items: list[str],
    user_question: str,
    model_answer: str,
    executed_actions: list[dict],
    history: list[dict] | None = None,
    language: str = "en",
) -> dict:
    """
    Score a model's answer against rubric items for one turn.

    Args:
        rubric_items:      Pre-generated rubric items for this turn.
        user_question:     The (implicit) question shown to the evaluated model.
        model_answer:      The model's response to evaluate.
        executed_actions:  Tool call results (from benchmark generation, for context).
        history:           Prior conversation turns [{role, content}, ...].
        language:          "en" | "zh"

    Returns:
        {
          "scores":        [1, 0, 1, ...],   # per-item binary scores
          "reasoning":     ["...", ...],      # per-item rationale
          "rubric_score":  0.75,             # fraction of items passed
          "items_passed":  3,
          "items_total":   4,
        }
    """
    if not rubric_items:
        return {
            "scores": [], "reasoning": [],
            "rubric_score": None, "items_passed": 0, "items_total": 0,
        }

    is_zh = language == "zh"
    sys_prompt = _SYSTEM_ZH if is_zh else _SYSTEM_EN
    ehr_summary = _format_tool_results(executed_actions)

    # Build conversation history string (last 6 messages)
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
            f"EHR检索数据（供核验）：\n{ehr_summary}\n\n"
            f"模型回答：\n{model_answer}\n\n"
            f"Rubric条目：\n{rubric_str}\n\n"
            "对每条rubric条目评分（JSON）："
        )
    else:
        user_prompt = (
            f"Clinical question: {user_question}\n\n"
            f"Conversation history:\n{hist_str}\n"
            f"EHR data retrieved (for verification):\n{ehr_summary}\n\n"
            f"Model response:\n{model_answer}\n\n"
            f"Rubric items:\n{rubric_str}\n\n"
            "Score each rubric item (JSON):"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = judge_llm_call(messages, temperature=0.1, max_tokens=4000)
        result = extract_json(raw)

        # Judge is asked for {"scores": [...], "reasoning": [...]}, but sometimes
        # returns a bare array. Normalize both shapes — extract_json can return a
        # list or a dict (same isinstance guard used in rubric_generator/fix_rubrics).
        if isinstance(result, dict):
            scores = result.get("scores", [])
            reasoning = result.get("reasoning", [""] * len(scores))
        elif isinstance(result, list):
            if all(isinstance(s, dict) for s in result):
                # [{"score": 1, "reasoning": "..."}, ...]
                scores = [s.get("score", s.get("pass", 0)) for s in result]
                reasoning = [str(s.get("reasoning", s.get("rationale", ""))) for s in result]
            else:
                # bare [1, 0, 1, ...]
                scores = list(result)
                reasoning = [""] * len(scores)
        else:
            raise ValueError(f"unexpected judge JSON type: {type(result).__name__}")

        # Validate lengths match
        n = len(rubric_items)
        if len(scores) != n:
            logger.warning(
                f"score_turn_rubric: got {len(scores)} scores for {n} items — padding"
            )
            scores = (scores + [0] * n)[:n]
            reasoning = (reasoning + [""] * n)[:n]

        # Coerce to 0/1. NB: int(bool("0")) == 1 (non-empty string is truthy),
        # so handle string/float scores explicitly rather than via bool().
        def _to01(s):
            if isinstance(s, bool):
                return int(s)
            if isinstance(s, (int, float)):
                return 1 if s >= 1 else 0
            return 1 if str(s).strip().lower() in ("1", "true", "yes", "y", "pass") else 0

        scores = [_to01(s) for s in scores]
        items_passed = sum(scores)
        rubric_score = items_passed / n if n > 0 else None

        return {
            "scores": scores,
            "reasoning": reasoning,
            "rubric_score": rubric_score,
            "items_passed": items_passed,
            "items_total": n,
        }

    except Exception as exc:
        logger.warning(f"score_turn_rubric failed ({exc})")
        n = len(rubric_items)
        return {
            "scores": [0] * n,
            "reasoning": [f"scoring error: {exc}"] * n,
            "rubric_score": None,
            "items_passed": 0,
            "items_total": n,
        }


def score_entry_rubric(
    rubrics: list[list[str]],
    model_answers: list[str],
    user_questions: list[str],
    answer_list: list[list[dict]],
    history_snapshots: list[list[dict]],
    language: str = "en",
) -> dict:
    """
    Score all turns of one benchmark entry.

    Returns:
        {
          "turn_rubric_results": [per-turn result dicts],
          "entry_rubric_score":  float,   # mean over turns with rubrics
        }
    """
    turn_results = []
    scored_turns = []

    for i, (items, q, ans, actions, hist) in enumerate(
        zip(rubrics, user_questions, model_answers, answer_list, history_snapshots)
    ):
        result = score_turn_rubric(
            rubric_items=items,
            user_question=q,
            model_answer=ans,
            executed_actions=actions,
            history=hist,
            language=language,
        )
        turn_results.append(result)
        if result["rubric_score"] is not None:
            scored_turns.append(result["rubric_score"])

    entry_score = sum(scored_turns) / len(scored_turns) if scored_turns else None

    return {
        "turn_rubric_results": turn_results,
        "entry_rubric_score": entry_score,
    }
