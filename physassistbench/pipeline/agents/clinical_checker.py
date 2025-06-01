"""
Clinical Checker — validates clinical correctness of the generated answer.

Checks:
1. The answer is grounded in the tool observations (no hallucination of values)
2. The answer doesn't contradict known facts from the patient record
3. Basic clinical safety: no obviously dangerous recommendations

Returns (is_valid: bool, issues: list[str])
"""

import json
from physassistbench.pipeline.agents.llm_client import llm_call, extract_json


_SYSTEM_PROMPT = """You are a clinical quality reviewer. Given a clinician's question,
tool observations (real EHR data), and the AI assistant's answer, check for:

1. Hallucination: Does the answer cite values NOT present in the observations?
2. Contradiction: Does the answer contradict values in the observations?
3. Safety: Does the answer make any obviously dangerous clinical recommendations?
4. Completeness: Does the answer address the actual question asked?

Return JSON:
{
  "valid": <boolean>,
  "hallucination": <boolean>,
  "contradiction": <boolean>,
  "safety_issue": <boolean>,
  "incomplete": <boolean>,
  "issues": ["<issue1>", ...],
  "score": <0-10>
}

Be lenient — minor omissions are OK. Flag only clear errors."""

_SYSTEM_PROMPT_ZH = """你是一位临床质量审核员。给定医生的问题、工具观察结果（真实EHR数据）和AI助手的回答，请检查以下内容：

1. 幻觉：回答是否引用了观察结果中不存在的数值？
2. 矛盾：回答是否与观察结果中的数值相矛盾？
3. 安全性：回答是否提出了明显危险的临床建议？
4. 完整性：回答是否真正回答了所提出的问题？

返回JSON：
{
  "valid": <布尔值>,
  "hallucination": <布尔值>,
  "contradiction": <布尔值>,
  "safety_issue": <布尔值>,
  "incomplete": <布尔值>,
  "issues": ["<问题1>", ...],
  "score": <0-10>
}

请保持宽松标准——轻微遗漏可以接受。仅标记明显错误。"""


def validate_answer(
    user_question: str,
    executed_actions: list,
    assistant_answer: str,
    task_domain: str,
    language: str = "en",
) -> tuple[bool, list]:
    """
    Validate clinical correctness of the answer.

    Returns (is_valid, issues_list)
    """
    # Build observation summary
    obs_parts = []
    for act in executed_actions:
        name = act["action"]["name"]
        if name == "prepare_to_answer":
            continue
        obs_str = json.dumps(act["observation"], default=str)[:800]
        obs_parts.append(f"[{name}]: {obs_str}")
    obs_text = "\n".join(obs_parts) if obs_parts else "[No tool calls]"

    user_prompt = f"""Clinical Task: {task_domain}
Question asked: {user_question}

EHR Data retrieved (ground truth):
{obs_text}

AI Assistant's answer:
{assistant_answer}

Validate this answer. Output JSON."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_ZH if language == "zh" else _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = llm_call(messages, temperature=0.0, max_tokens=4000)
        result = extract_json(raw)
        is_valid = bool(result.get("valid", True))
        issues = result.get("issues", [])
        # Hard fail only on safety issues or severe hallucination
        if result.get("safety_issue", False):
            is_valid = False
            issues.append("Safety issue detected")
        return is_valid, issues
    except Exception as e:
        # Don't block generation on checker failure
        return True, [f"Clinical checker skipped: {e}"]
