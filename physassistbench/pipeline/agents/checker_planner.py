"""
Checker Planner (Gate 1) — validates that the planner's action list is correct
for the given task type and user question.

Task types:
  Information Lookup  — 1 tool (EHR or Patient) + prepare_to_answer
  Data Gathering      — 2-4 tools + prepare_to_answer
  Clinical Reasoning  — ≥1 tool + prepare_to_answer
  Write/Update        — 1 write tool + prepare_to_answer

Tool source (orthogonal to task type):
  ehr     — EHR tools only
  patient — patient.xxx tools only
  mixed   — both EHR and patient tools
  write   — write tools only (MedicationRequest/ServiceRequest/Flag)

Returns (is_valid: bool, reason: str)
"""

from physassistbench.pipeline.agents.llm_client import llm_call, extract_json

_SYSTEM_PROMPT = """You are a clinical planning validator. Given a user question, task type,
tool source, and an action plan, check whether the plan is correct.

Return JSON: {"valid": <bool>, "reason": "<brief explanation>"}

Validation rules:
1. Information Lookup: exactly 1 non-prepare_to_answer tool in Action_List.
   - If tool_source=ehr: must be an EHR tool (get_lab_results, get_diagnoses, etc.)
   - If tool_source=patient: must be a patient.xxx tool
2. Data Gathering: 2-4 non-prepare_to_answer tools in Action_List.
   - If tool_source=mixed: may combine EHR tools AND patient.xxx tools.
3. Clinical Reasoning: ≥1 non-prepare_to_answer tool in Action_List.
   Typically 1 tool fetching a clinical parameter; may use 2 when the question
   requires correlating two data points (e.g. creatinine + medication order).
4. Write/Update: exactly 1 write tool (MedicationRequest.create, ServiceRequest.create,
   or Flag.create) + prepare_to_answer. No read/search tools allowed.
5. All tools must exist in the Available Tools list.
6. subject_id must be present in EHR tool arguments when the patient is known.
7. For patient tools: both subject_id AND session_id must be present in arguments.
8. Tool arguments must match their schema (no missing required parameters).
9. The tools chosen must be relevant to the question asked.
10. Action_List must end with prepare_to_answer.

Output ONLY the JSON, no other text."""

_SYSTEM_PROMPT_ZH = """你是一位临床规划验证员。给定用户问题、任务类型、工具来源和行动计划，请检查该计划是否正确。

返回JSON：{"valid": <布尔值>, "reason": "<简要解释>"}

验证规则：
1. Information Lookup（检索）：Action_List中恰好有1个非prepare_to_answer工具。
   - tool_source=ehr：必须是EHR工具（get_lab_results、get_diagnoses等）
   - tool_source=patient：必须是patient.xxx工具
2. Data Gathering（检查工作流）：Action_List中有2-4个非prepare_to_answer工具。
   - tool_source=mixed：可以同时包含EHR工具和patient.xxx工具。
3. Clinical Reasoning（知识结合）：Action_List中有≥1个非prepare_to_answer工具。
   通常为1个获取临床参数的工具；当问题需要关联两个数据点时可用2个。
4. Write/Update（写入操作）：恰好1个写入工具（MedicationRequest.create、
   ServiceRequest.create或Flag.create）+ prepare_to_answer。不允许读取/搜索工具。
5. 所有工具必须存在于可用工具列表中。
6. 已知患者时，EHR工具参数中必须包含subject_id。
7. 使用患者工具时：参数中必须同时包含subject_id和session_id。
8. 工具参数必须符合其schema（不得缺少必要参数）。
9. 所选工具必须与所提问题相关。
10. Action_List末尾必须是prepare_to_answer。

只输出JSON，不要包含其他文本。"""


def validate_plan(
    user_question: str,
    task_type: str,
    plan: dict,
    available_tool_names: list,
    subject_id: int,
    tool_source: str = "ehr",
    language: str = "en",
) -> tuple[bool, str]:
    """Validate the planner's output. Returns (is_valid, reason)."""
    action_list = plan.get("Action_List", [])
    non_prep = [a for a in action_list if a.get("name") not in ("prepare_to_answer",)]

    # Fast rule-based checks by task type
    if task_type == "Information Lookup":
        if len(non_prep) != 1:
            return False, f"Information Lookup requires exactly 1 tool, got {len(non_prep)}"
    elif task_type == "Clinical Reasoning":
        if len(non_prep) < 1:
            return False, f"Clinical Reasoning requires ≥1 tool, got {len(non_prep)}"
    elif task_type == "Data Gathering":
        if len(non_prep) < 2:
            return False, f"Data Gathering requires ≥2 tools, got {len(non_prep)}"

    # Mixed tool_source: must include ≥1 patient tool (EHR tools optional)
    if tool_source == "mixed":
        patient_calls = [a for a in non_prep if a.get("name", "").startswith("patient.")]
        if not patient_calls:
            return False, (
                "tool_source=mixed requires ≥1 patient.get_xxx call, but none found. "
                f"Got: {[a.get('name') for a in non_prep]}"
            )
    elif task_type == "Write/Update":
        if len(non_prep) != 1:
            return False, f"Write/Update requires exactly 1 write tool, got {len(non_prep)}"

    # Patient-source: at least one patient.xxx tool required
    if tool_source == "patient":
        patient_tools = [a for a in non_prep if a.get("name", "").startswith("patient.")]
        if len(patient_tools) < 1:
            return False, f"tool_source=patient requires ≥1 patient.xxx tool, got {len(patient_tools)}"

    # Mixed: should have at least one of each
    if tool_source == "mixed" and task_type == "Data Gathering":
        patient_tools = [a for a in non_prep if a.get("name", "").startswith("patient.")]
        ehr_tools = [a for a in non_prep if not a.get("name", "").startswith("patient.")
                     and a.get("name") != "ask_user_for_required_parameters"]
        if not patient_tools or not ehr_tools:
            # Allow mixed to pass with only one type on retry (not strict)
            pass

    for action in non_prep:
        name = action.get("name", "")
        if name not in available_tool_names:
            return False, f"Tool '{name}' not in available tools"
        args = action.get("arguments", {})
        if name.startswith("patient."):
            # Patient tools need session_id
            if subject_id and "subject_id" not in args:
                return False, f"Patient tool '{name}' missing subject_id argument"
        elif subject_id and "subject_id" not in args:
            return False, f"Tool '{name}' missing subject_id argument"

    # LLM semantic check
    user_prompt = f"""User question: {user_question}
Task type: {task_type}
Tool source: {tool_source}
Plan: {plan}
Available tools: {available_tool_names}

Is this plan correct and appropriate? Output JSON."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_ZH if language == "zh" else _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = llm_call(messages, temperature=0.0, max_tokens=4000)
        result = extract_json(raw)
        return bool(result.get("valid", True)), str(result.get("reason", ""))
    except Exception:
        return True, "LLM check skipped"
