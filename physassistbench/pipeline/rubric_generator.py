"""
physassistbench/pipeline/rubric_generator.py — Auto-generate per-turn evaluation rubrics.

Called after each turn's tool execution and answer validation.
Uses actual EHR values from tool observations to produce outcome-based rubric items
following an outcome-based design philosophy:
  - Items describe OUTCOMES/GOALS, not tool calls or process steps
  - Items are grounded in actual patient values retrieved from the EHR
  - Items are independently evaluable (binary yes/no) by an LLM judge

Write/Update turns use a separate parameter-level generator (_generate_action_rubric)
that produces rubric items tied directly to FHIR write tool_call arguments
(medication, dose, route, frequency, ...), not free-text reasoning.
"""

from __future__ import annotations
import json
import logging

from physassistbench.pipeline.agents.llm_client import llm_call, extract_json

logger = logging.getLogger(__name__)

_WRITE_TOOL_NAMES = frozenset({
    "MedicationRequest.create",
    "ServiceRequest.create",
    "Flag.create",
})


# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_EN = """\
You are a clinical benchmark rubric designer for an EHR-based QA evaluation.

Given a clinical question, the EHR data retrieved, and a reference answer,
generate 3–6 atomic rubric criteria to evaluate another LLM's response.

DESIGN RULES:
1. Each item describes an OUTCOME or CLINICAL GOAL — never a tool call, API name, or process step.
2. GROUND items in actual values from the EHR data.
   Write: "The answer correctly cites creatinine as 0.9 mg/dL"
   NOT:  "The answer mentions the creatinine value"
3. Each item must be independently evaluable as YES or NO.
4. Include at least one reasoning/recommendation item (not just fact retrieval).
5. For safety-critical decisions, include one item checking a dangerous recommendation is absent.
6. Do NOT mention tool names, function names, or any system internals.
7. CLINICAL ACCURACY: Do NOT simply echo the reference answer's conclusion as a rubric item.
   If the reference answer makes a clinical claim (e.g. "risk is X not Y"), verify it is medically
   correct before including it. If the claim is debatable, write the rubric to check the reasoning
   process, not just the specific conclusion.
8. MIXED/PATIENT turns: When the answer incorporates BOTH EHR data AND patient-reported information,
   rubric items MUST cover both dimensions:
   (a) EHR data: correct value citation and interpretation
   (b) Patient interview: patient's reported symptom/adherence correctly quoted and clinically interpreted
   Do NOT write rubric items saying "EHR data not available" for a mixed turn — those items are
   irrelevant since the data came from patient interview, not EHR lookup.
9. ACTION turns: Always include one safety item checking that no contraindicated intervention
   was ordered (e.g. wrong dose range, drug-disease contraindication, missing monitoring plan).

Item count by task type:
  Information Lookup:          3 items  (value cited correctly, interpreted correctly, conclusion given)
  Data Gathering:             4–5 items (each value cited, relationship stated, conclusion)
  Clinical Reasoning: 5–6 items (value cited, threshold applied, reasoning chain, recommendation, safety)
  Mixed/Patient turn: 5–6 items — 2–3 on EHR findings, 2–3 on patient-reported findings and their interpretation
  Write/Update turn:        5–6 items — specific parameters ordered, indication documented, safety check

Output format: Return ONLY a valid JSON array of strings. No prose, no markdown fences.
Example: ["The answer cites creatinine as 0.9 mg/dL", "The answer concludes no metformin adjustment is needed", "..."]
"""

_SYSTEM_ZH = """\
你是一个基于EHR的临床QA基准测试的rubric设计者。

给定一个临床问题、检索到的EHR数据和参考答案，为评测另一个LLM对同一问题的回答生成3–6条原子化评测标准。

设计规则：
1. 每条描述结果或临床目标——绝不描述工具调用、函数名称或过程步骤。
2. 用EHR数据中的实际数值来表达。
   写：「回答正确引用了肌酐值0.9 mg/dL」
   不写：「回答提到了肌酐数值」
3. 每条标准可被LLM法官独立判断为是或否。
4. 至少包含一条关于推理或临床建议的条目（不只是数值引用）。
5. 对安全关键决策，包含一条检查危险建议不存在的条目。
6. 不得提及工具名称、函数名称或任何系统内部细节。
7. 临床准确性：不要简单地把参考答案的结论复制为rubric条目。
   若参考答案作出临床判断（如"风险是X而非Y"），应验证其医学正确性。
   若该判断有争议，rubric应检查推理过程，而非特定结论。
8. Mixed/病人访谈轮：当答案同时整合了EHR数据和病人自述信息，
   rubric必须同时覆盖两个维度：
   (a) EHR数据：数值正确引用和解读
   (b) 病人访谈：患者自述的症状/依从性被正确引用并纳入临床解读
   不要写"EHR中未发现数据"这类条目——mixed轮的数据来自病人访谈，不是EHR查询。
9. Action轮：必须包含一条安全性条目，检查未开具禁忌医嘱
   （如剂量范围错误、药物-疾病禁忌、缺少监测计划）。

各任务类型的条目数：
  Information Lookup：          3条（数值正确引用、正确解读、给出结论）
  Data Gathering：             4–5条（各数值引用、关联描述、结论）
  Clinical Reasoning：5–6条（数值引用、阈值应用、推理链、建议、安全性）
  Mixed/病人访谈轮：  5–6条——2–3条关于EHR发现，2–3条关于病人自述及其临床解读
  Action轮：          5–6条——具体参数、适应证记录、安全性检查

输出格式：只返回有效的JSON字符串数组，不含任何散文或代码块。
示例：["回答正确引用了肌酐值0.9 mg/dL", "回答得出无需调整二甲双胍的结论", "..."]
"""


# ── Action-turn (Write/Update) rubric prompts ─────────────────────────────────

_SYSTEM_ACTION_EN = """\
You design rubric items for evaluating a model's FHIR WRITE tool call
(MedicationRequest.create, ServiceRequest.create, or Flag.create).

The model is expected to output a structured tool call whose `arguments`
object contains specific parameter fields (e.g. medication, dose, route,
frequency, indication, service_type, priority, note, ...). EACH RUBRIC
ITEM MUST BE A CHECK ON A SINGLE PARAMETER FIELD — never on free-text
reasoning or prose.

DESIGN RULES:
1. Each item names a specific field of `tool_call.arguments`.
2. Each item has a clear PASS / FAIL criterion checkable from the field value.
3. Explicitly allow clinically equivalent values (drug synonyms,
   route synonyms, frequency synonyms, dose ranges) inside the rubric text.
4. Include exactly one NEGATIVE SAFETY item that FAILS when a dangerous
   value is present (e.g. dose >= contraindicated threshold, a
   contraindicated drug class is ordered, priority is `stat` when not
   warranted).
5. Do NOT write items about clinical reasoning, indication-justification
   prose, or any text outside the tool call. The model's free-text answer
   is NOT evaluated in Write/Update turns.
6. Do NOT reference EHR values from prior turns (e.g. "kidney function
   is normal so dose is fine"). Those are evaluated in CR/DG turns.
7. Item count: 5–6.

Output format: a JSON array of strings, each clearly naming the field
being checked. Example (gold: medication=Metformin, dose=250 mg,
route=oral, frequency=once daily, indication="dose reduction for GI
intolerance"):

[
  "The write call invokes `MedicationRequest.create` (no other write tool is called).",
  "The `medication` field equals 'Metformin' (case-insensitive; brand-name 'Glucophage' is also acceptable).",
  "The `dose` field equals exactly '250 mg' (not 500 mg, not 1000 mg).",
  "The `route` field is 'oral' (or one of: 'PO', 'by mouth', 'per os').",
  "The `frequency` field is 'once daily' (or one of: 'QD', 'daily', 'qd', 'once a day').",
  "Safety: the `dose` is NOT >= 500 mg (which would defeat the reduction strategy)."
]
"""

_SYSTEM_ACTION_ZH = """\
你为模型的FHIR写工具调用（MedicationRequest.create、ServiceRequest.create或
Flag.create）设计rubric条目。

模型应输出结构化的tool call，其`arguments`对象包含具体字段（如medication、
dose、route、frequency、indication、service_type、priority、note等）。
每一条rubric必须针对tool call中的某一个参数字段——绝不评测自由文本推理或
散文式说明。

设计规则：
1. 每条明确指出tool_call.arguments中的某个字段。
2. 每条具备清晰的通过/失败判定标准，可从字段值直接检查。
3. 在rubric文本中显式列出临床等效的可接受值（药物同义词、给药途径同义词、
   频次同义词、剂量范围等）。
4. 必须包含恰好一条负向安全条目：当存在危险值时判FAIL
   （如剂量≥禁忌阈值、开具禁忌药物类、不当地将priority设为stat等）。
5. 不得评测临床推理、适应证说明散文或tool call之外的任何文字。
   Action轮的自由文本回答不参与评分。
6. 不得引用前序轮次的EHR数值（如"肾功能正常所以剂量合适"）。
   那些应在CR/DG轮评测。
7. 条目数：5–6条。

输出格式：JSON字符串数组，每条明确指出所检查的字段。示例
（金标准：medication=Metformin，dose=250 mg，route=oral，
frequency=once daily，indication="dose reduction for GI intolerance"）：

[
  "写工具调用为`MedicationRequest.create`（且没有调用其他写工具）。",
  "`medication`字段等于'Metformin'（大小写不敏感；可接受品牌名'Glucophage'）。",
  "`dose`字段严格等于'250 mg'（不是500 mg、不是1000 mg）。",
  "`route`字段为'oral'（或'PO'、'by mouth'、'per os'之一）。",
  "`frequency`字段为'once daily'（或'QD'、'daily'、'qd'、'once a day'之一）。",
  "安全性：`dose`不为≥500 mg（否则违背减量初衷）。"
]
"""


def _extract_gold_write_calls(executed_actions: list[dict]) -> list[dict]:
    """Pick the gold FHIR write tool calls (skip read/prepare_to_answer)."""
    out = []
    for act in executed_actions:
        a = act.get("action", {}) if isinstance(act, dict) else {}
        if a.get("name") in _WRITE_TOOL_NAMES:
            out.append({"name": a.get("name"), "arguments": a.get("arguments", {})})
    return out


def _format_gold_writes(write_calls: list[dict]) -> str:
    if not write_calls:
        return "(no write tool call recorded)"
    parts = []
    for wc in write_calls:
        args_str = json.dumps(wc.get("arguments", {}), ensure_ascii=False, indent=2)
        parts.append(f"{wc['name']}:\n{args_str}")
    return "\n\n".join(parts)


def _generate_action_rubric(
    executed_actions: list[dict],
    user_question_explicit: str,
    language: str = "en",
) -> list[str]:
    """Action-turn rubric: tied strictly to write tool_call parameter fields."""
    gold_writes = _extract_gold_write_calls(executed_actions)
    if not gold_writes:
        logger.warning("_generate_action_rubric: no gold write tool call found "
                       "in executed_actions — returning empty rubric")
        return []

    is_zh = language == "zh"
    sys_prompt = _SYSTEM_ACTION_ZH if is_zh else _SYSTEM_ACTION_EN
    gold_str = _format_gold_writes(gold_writes)

    if is_zh:
        user_prompt = (
            f"临床问题（医生的写操作请求）：{user_question_explicit}\n\n"
            f"金标准写工具调用：\n{gold_str}\n\n"
            "生成参数级rubric条目（JSON数组）："
        )
    else:
        user_prompt = (
            f"Clinical question (physician's write request): "
            f"{user_question_explicit}\n\n"
            f"Gold write tool call:\n{gold_str}\n\n"
            "Generate parameter-level rubric items (JSON array):"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = llm_call(messages, temperature=0.2, max_tokens=2000)
        items = extract_json(raw)
        if isinstance(items, list) and all(isinstance(i, str) for i in items):
            return [i.strip() for i in items if i.strip()]
        logger.warning("_generate_action_rubric: unexpected JSON — returning empty")
        return []
    except Exception as exc:
        logger.warning(f"_generate_action_rubric failed ({exc}) — returning empty")
        return []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_summary(executed_actions: list[dict]) -> str:
    """
    Extract the prepare_to_answer observation as the primary EHR summary.
    Falls back to raw observations if no prepare_to_answer action exists.
    """
    for act in reversed(executed_actions):
        if act.get("action", {}).get("name") == "prepare_to_answer":
            obs = act.get("observation", "")
            if isinstance(obs, str) and obs.strip():
                return obs.strip()

    # Fallback: collect non-empty raw observations
    lines = []
    for act in executed_actions:
        name = act.get("action", {}).get("name", "")
        if name == "prepare_to_answer":
            continue
        obs = act.get("observation", "")
        args = act.get("action", {}).get("arguments", {})
        if isinstance(obs, dict):
            # Summarise FHIR bundle: extract value quantities
            entries = obs.get("entry", [])
            for entry in entries[:3]:
                res = entry.get("resource", {})
                vq = res.get("valueQuantity", {})
                code = res.get("code", {}).get("text", "")
                val = vq.get("value", "")
                unit = vq.get("unit", "")
                if code and val != "":
                    lines.append(f"{code}: {val} {unit}".strip())
            if obs.get("total", 0) == 0:
                lines.append(f"{name}: no results found")
        elif isinstance(obs, str) and obs.strip():
            lines.append(f"{name}: {obs.strip()[:200]}")
    return "\n".join(lines) if lines else "(no EHR data retrieved)"


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_rubric(
    task_type: str,
    user_question_explicit: str,
    executed_actions: list[dict],
    assistant_answer: str,
    language: str = "en",
    tool_source: str = "ehr",
) -> list[str]:
    """
    Generate outcome-based rubric items grounded in actual EHR values.

    Args:
        task_type:              "Information Lookup" | "Data Gathering" | "Clinical Reasoning"
        user_question_explicit: The Stage-1 explicit question (before ellipsis transform).
        executed_actions:       Tool call results for this turn.
        assistant_answer:       The validated reference answer.
        language:               "en" | "zh"
        tool_source:            "ehr" | "patient" | "mixed" | "write" — informs rubric design.
    """
    # Route Action / Write turns to the parameter-level generator.
    # These rubrics evaluate FHIR write tool_call arguments only, not free
    # text reasoning.
    if task_type == "Write/Update" or tool_source == "write":
        return _generate_action_rubric(
            executed_actions=executed_actions,
            user_question_explicit=user_question_explicit,
            language=language,
        )

    is_zh = language == "zh"
    sys_prompt = _SYSTEM_ZH if is_zh else _SYSTEM_EN
    ehr_summary = _extract_summary(executed_actions)

    # Extract patient interview responses for mixed/patient turns
    patient_summary = ""
    if tool_source in ("patient", "mixed"):
        patient_parts = []
        for act in executed_actions:
            name = act.get("action", {}).get("name", "")
            if name.startswith("patient."):
                obs = act.get("observation", {})
                resp = obs.get("patient_response", "") if isinstance(obs, dict) else str(obs)
                tool_label = name.replace("patient.", "").replace("_", " ").title()
                if resp:
                    patient_parts.append(f"[{tool_label}]: {resp[:300]}")
        patient_summary = "\n".join(patient_parts)

    # Annotate task_type with tool_source context for clearer rubric generation
    task_type_label = task_type
    if tool_source == "mixed":
        task_type_label = f"{task_type} (Mixed: EHR + Patient Interview)"
    elif tool_source == "patient":
        task_type_label = f"{task_type} (Patient Interview only)"
    elif tool_source == "write":
        task_type_label = f"{task_type} (Action/Write)"

    if is_zh:
        patient_section = (f"\n病人访谈回答：\n{patient_summary}\n" if patient_summary else "")
        user_prompt = (
            f"任务类型：{task_type_label}\n\n"
            f"临床问题：{user_question_explicit}\n\n"
            f"EHR检索到的数据：\n{ehr_summary}\n"
            f"{patient_section}\n"
            f"参考答案：\n{assistant_answer[:600]}\n\n"
            "生成评测rubric条目（JSON数组）："
        )
    else:
        patient_section = (f"\nPatient interview responses:\n{patient_summary}\n" if patient_summary else "")
        user_prompt = (
            f"Task type: {task_type_label}\n\n"
            f"Clinical question: {user_question_explicit}\n\n"
            f"EHR data retrieved:\n{ehr_summary}\n"
            f"{patient_section}\n"
            f"Reference answer:\n{assistant_answer[:600]}\n\n"
            "Generate rubric items (JSON array):"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = llm_call(messages, temperature=0.2, max_tokens=4000)
        items = extract_json(raw)
        if isinstance(items, list) and all(isinstance(i, str) for i in items):
            return [i.strip() for i in items if i.strip()]
        logger.warning("generate_rubric: unexpected JSON structure — returning empty")
        return []
    except Exception as exc:
        logger.warning(f"generate_rubric failed ({exc}) — returning empty")
        return []
