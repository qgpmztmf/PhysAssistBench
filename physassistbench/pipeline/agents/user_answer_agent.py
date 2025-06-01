"""
user_answer_agent.py — Simulate the clinician's response to a clarification question.

Called after clarify_agent generates a clarification question, to produce a realistic
user response that provides the requested information.

Equivalent to WildToolBench's multi-agent-framework/agent/user_answer_ask.py.
"""

from physassistbench.pipeline.agents.llm_client import llm_call

_SYSTEM_PROMPT_EN = """You are simulating a clinician (doctor, nurse, or pharmacist) interacting
with a clinical AI assistant.

The AI assistant asked you for missing information needed to retrieve patient data from the EHR.
Provide a realistic, natural-language response that supplies the requested information.

Requirements:
- Base your response on the patient context provided.
- Use natural clinical language — not overly formal, not overly casual.
- Be direct and concise (1-3 sentences).
- Do NOT add any prefix like "User:" — just write the response directly.
- If the required information is genuinely ambiguous, make a plausible clinical choice."""

_SYSTEM_PROMPT_VAGUE_EN = """You are simulating a busy clinician (doctor, nurse, or pharmacist)
quickly replying to a clarification question from a clinical AI assistant.

Your reply must be BRIEF and use INDIRECT reference — do not spell out the exact technical value.
Use anaphoric reference, ordinal reference, or partial information, just like a busy clinician
would type in real life.

Examples of the vague reply style:
  AI asked: "Which lab value — tacrolimus or creatinine?"
  → Reply: "The immunosuppressant" / "The first one" / "The tacro level"

  AI asked: "Which admission are you referring to?"
  → Reply: "The most recent one" / "Last week's" / "The second one"

  AI asked: "What INR target are you using — standard AF range or mechanical valve?"
  → Reply: "Valve target" / "The higher range" / "It's a mechanical valve"

  AI asked: "Which culture result — blood or urine?"
  → Reply: "Blood" / "The one that came back positive"

  AI asked: "Which medication — metformin or the SGLT2?"
  → Reply: "The diabetes one we started recently" / "Second one"

Rules:
- 1-5 words is ideal. 10 words maximum.
- Do NOT use full sentences like "I am referring to tacrolimus."
- Do NOT add any prefix — just write the reply directly.
- Stay consistent with the patient context provided."""

_SYSTEM_PROMPT_ZH = """你正在模拟一位临床医生（医生、护士或药剂师），与临床AI助手进行交互。

AI助手向你询问了从EHR中检索患者数据所需的缺失信息。
请根据提供的患者背景信息，给出真实、自然的语言回复，提供所请求的信息。

要求：
- 回复应基于所提供的患者背景信息。
- 使用自然的临床语言——不要过于正式，也不要过于随意。
- 简洁直接（1-3句话）。
- 不要添加任何前缀，如"用户："——直接写回复内容。
- 如果所需信息确实模糊，做出合理的临床选择。"""

_SYSTEM_PROMPT_VAGUE_ZH = """你正在模拟一位繁忙的临床医生，正在快速回复临床AI助手的澄清问题。

你的回复必须简短且使用间接指代——不要直接说出完整的技术参数值。
使用回指、序数指代或不完整信息，就像忙碌的医生在真实对话中那样。

回复风格示例：
  AI问："您指的是他克莫司还是肌酐？"
  → 回复："免疫抑制剂那个" / "第一个" / "他克莫司的那个"

  AI问："您指的是哪次住院？"
  → 回复："最近那次" / "上周的" / "第二次"

  AI问："INR目标是标准房颤范围还是机械瓣膜？"
  → 回复："瓣膜那个" / "高一点的那个" / "她是机械瓣"

规则：
- 1-10个字最理想。
- 不要用完整句子，如"我指的是他克莫司。"
- 不要加任何前缀——直接写回复。
- 与提供的患者背景保持一致。"""


def generate_user_response(
    context: dict,
    history: list,
    user_question: str,
    clarification_question: str,
    available_tools: list,
    language: str = "en",
    vague: bool = False,
) -> str:
    """
    Simulate the clinician's response to the clarification question.

    Returns: user response string (plain text, no prefix).
    """
    if vague:
        system_prompt = _SYSTEM_PROMPT_VAGUE_ZH if language == "zh" else _SYSTEM_PROMPT_VAGUE_EN
    else:
        system_prompt = _SYSTEM_PROMPT_ZH if language == "zh" else _SYSTEM_PROMPT_EN

    context_str = context.get("context_str", "")
    subject_id = context.get("subject_id", "")
    hadm_id = context.get("hadm_id", "")

    # Recent conversation context
    history_str = ""
    for m in history[-4:]:
        role = m.get("role", "")
        content = str(m.get("content", ""))[:300]
        history_str += f"[{role.upper()}]: {content}\n"

    user_prompt = (
        f"Patient context:\n{context_str}\n"
        f"subject_id: {subject_id}  hadm_id: {hadm_id}\n\n"
        f"Original clinical question: {user_question}\n\n"
        f"The AI assistant asked: {clarification_question}\n\n"
        f"Recent conversation:\n{history_str}\n"
        f"Provide a brief, realistic response as the clinician, "
        f"supplying the requested information:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result = llm_call(messages)

    # Strip any role prefix the LLM may add
    for prefix in ("User:", "Clinician:", "Doctor:", "用户：", "医生：", "临床医生："):
        if result.startswith(prefix):
            result = result[len(prefix):].strip()

    return result.strip()
