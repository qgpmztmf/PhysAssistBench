"""
Planner Agent — decides which tools to call and in what order.

Task types (PhysAssistBench / new design):
  Information Lookup         — 1 tool (EHR or Patient), fetch one data point
  Data Gathering            — ≥2 tools, parallel or conditional branching
  Clinical Reasoning — 1 tool to fetch a patient parameter; answer agent applies knowledge
  Write/Update — call write tool (MedicationRequest/ServiceRequest/Flag) to place a clinical order

Tool source (orthogonal to task type):
  ehr     — use EHR tools only
  patient — use patient interview tools (patient.xxx)
  mixed   — use both EHR and patient tools

Output JSON format:
{
  "Task_Finish": false,
  "Thought": "...",
  "Plan": "...",
  "Action_List": [
    {"name": "get_lab_results", "arguments": {"subject_id": 10000032, "hadm_id": 22595853}},
    {"name": "prepare_to_answer", "arguments": {}}
  ]
}
"""

import json
from physassistbench.pipeline.agents.llm_client import llm_call, extract_json

_SYSTEM_PROMPT = """You are a clinical planning agent. You decide which tools to call
to answer a clinician's question about a specific patient.

You must output a JSON object with these exact fields:
{
  "Task_Finish": <boolean — always false; tools are always needed>,
  "Thought": "<one-sentence reasoning about what data is needed>",
  "Plan": "<brief description of the tool call sequence>",
  "Action_List": [
    {"name": "<tool_name>", "arguments": {<tool_arguments>}},
    ...
    {"name": "prepare_to_answer", "arguments": {}}
  ]
}

Task type rules:
1. Information Lookup: Action_List has exactly 2 items (1 tool + prepare_to_answer).
   - If tool_source=ehr: use one EHR tool (get_lab_results, get_diagnoses, etc.)
   - If tool_source=patient: use one patient tool (patient.get_symptom_history, etc.)

2. Data Gathering: Action_List has 3-5 items (2-4 tools + prepare_to_answer).
   Parallel mode: call independent tools together (no dependency between them).
   Adaptive mode: first tool result determines whether/which second tool to call.
   - If tool_source=mixed: MANDATORY — the Action_List MUST include at least one
     patient.get_xxx tool call. EHR tools may also be included but are optional.
     Do NOT produce a plan with only EHR tools when tool_source=mixed.
   CLINICAL SCORING — tool call recipe per score (retrieve ALL components in parallel):
     SOFA (organ dysfunction):
       Observation.search(platelet) + Observation.search(bilirubin) +
       Observation.search(creatinine) + MedicationAdministration.search(vasopressor)
       [PaO₂/FiO₂ and GCS not in labevents — note as missing]
     SIRS (systemic inflammation):
       Observation.search(WBC) + Observation.search(temperature) [if available]
     MELD (liver disease prognosis):
       Observation.search(bilirubin) + Observation.search(INR) + Observation.search(creatinine)
     Child-Pugh (cirrhosis severity):
       Observation.search(bilirubin) + Observation.search(albumin) + Observation.search(INR)
       [+ patient.get_symptom_history for ascites/encephalopathy if tool_source=mixed]
     CURB-65 (pneumonia severity):
       Observation.search(BUN) + Condition.search [to confirm pneumonia diagnosis]
     CHA₂DS₂-VASc (afib stroke risk):
       Condition.search [single call retrieves CHF/HTN/DM/stroke/vascular diagnoses]
     HAS-BLED (bleeding risk):
       Condition.search + MedicationRequest.search + Observation.search(creatinine/INR)
     Ranson (pancreatitis — admission criteria):
       Observation.search(WBC) + Observation.search(glucose) +
       Observation.search(LDH) + Observation.search(AST)
     Cockcroft-Gault / eGFR (renal dosing):
       Observation.search(creatinine) [apply formula with age from patient context]
     Wells PE (pulmonary embolism probability):
       Condition.search [prior PE/DVT, malignancy] + Observation.search(D-dimer)
       [+ patient.get_symptom_history for HR/immobilization if tool_source=mixed]
   The answer agent will MAP each retrieved value to a subscore and SUM them.

3. Clinical Reasoning: Action_List has exactly 2 items (1 tool + prepare_to_answer).
   Fetch ONE specific patient parameter (e.g. eGFR, INR, weight). The answer agent
   will combine this with clinical knowledge to give personalised advice.
   Do NOT call multiple tools — one parameter is sufficient.
   Note: Clinical score COMPUTATION happens in Data Gathering turns (step 2 above), not here.

4. Write/Update: The question requests a clinical write action (order, flag, or referral).
   Action_List has exactly 2 items (1 write tool + prepare_to_answer).
   Only write tools are available in this turn (MedicationRequest.create,
   ServiceRequest.create, Flag.create). Do NOT call read/search tools.

General rules:
5. Always end Action_List with {"name": "prepare_to_answer", "arguments": {}}.
6. For patient tools, ALWAYS include both subject_id AND session_id (from env_info):
   {"name": "patient.get_symptom_history", "arguments": {"subject_id": 10000032, "session_id": "session_00"}}
7. Only use tools from the provided Available Tools list.
8. Always include subject_id in EHR tool arguments when known.
9. CRITICAL: Use EXACTLY the parameter names shown in the tool definitions.
   - get_lab_results uses "item_name" (NOT "test_name", "lab_name")
   - get_lab_trends uses "item_name" and "n_recent"
   - patient tools require both "subject_id" and "session_id"

Output ONLY the JSON object, no other text."""

_SYSTEM_PROMPT_ZH = """你是一个临床规划代理，负责决定调用哪些工具来回答医生关于特定患者的问题。

你必须输出一个包含以下确切字段的JSON对象：
{
  "Task_Finish": <布尔值 — 始终为false；始终需要调用工具>,
  "Thought": "<一句话说明需要什么数据的推理>",
  "Plan": "<工具调用顺序的简要描述>",
  "Action_List": [
    {"name": "<工具名称>", "arguments": {<工具参数>}},
    ...
    {"name": "prepare_to_answer", "arguments": {}}
  ]
}

任务类型规则：
1. Information Lookup（检索）：Action_List恰好包含2项（1个工具 + prepare_to_answer）。
   - tool_source=ehr：使用一个EHR工具（get_lab_results、get_diagnoses等）
   - tool_source=patient：使用一个患者工具（patient.get_symptom_history等）

2. Data Gathering（检查工作流）：Action_List包含3-5项（2-4个工具 + prepare_to_answer）。
   并行模式：同时调用相互独立的工具。
   自适应模式：第一个工具的结果决定是否/调用哪个第二个工具。
   - tool_source=mixed：强制要求——Action_List 中必须至少包含一个 patient.get_xxx 工具调用。
     EHR 工具可选择性地同时调用。tool_source=mixed 时不得只使用 EHR 工具。
   临床评分——各评分工具调用方案（并行获取所有组件）：
     SOFA（器官功能障碍）：
       Observation.search(血小板) + Observation.search(胆红素) +
       Observation.search(肌酐) + MedicationAdministration.search(升压药)
       [PaO₂/FiO₂和GCS不在labevents中——注明缺失]
     SIRS（全身炎症反应）：
       Observation.search(WBC) + Observation.search(体温)[如可用]
     MELD（肝病预后）：
       Observation.search(胆红素) + Observation.search(INR) + Observation.search(肌酐)
     Child-Pugh（肝硬化严重程度）：
       Observation.search(胆红素) + Observation.search(白蛋白) + Observation.search(INR)
       [+ patient.get_symptom_history 获取腹水/脑病信息，当tool_source=mixed]
     CURB-65（肺炎严重程度）：
       Observation.search(BUN) + Condition.search [确认肺炎诊断]
     CHA₂DS₂-VASc（房颤卒中风险）：
       Condition.search [单次调用获取CHF/高血压/糖尿病/卒中/血管病诊断]
     HAS-BLED（出血风险）：
       Condition.search + MedicationRequest.search + Observation.search(肌酐/INR)
     Ranson（胰腺炎——入院标准）：
       Observation.search(WBC) + Observation.search(血糖) +
       Observation.search(LDH) + Observation.search(AST)
     Cockcroft-Gault / eGFR（肾脏剂量）：
       Observation.search(肌酐) [结合患者上下文中的年龄应用公式]
     Wells PE（肺栓塞概率）：
       Condition.search [既往PE/DVT、恶性肿瘤] + Observation.search(D-二聚体)
       [+ patient.get_symptom_history 获取HR/制动信息，当tool_source=mixed]
   回答代理会将每个值映射为子分值并求和。

3. Clinical Reasoning（知识结合）：Action_List恰好包含2项（1个工具 + prepare_to_answer）。
   获取一个特定患者参数（如eGFR、INR、体重）。回答代理将结合临床知识给出个体化建议。
   不要调用多个工具——一个参数即可。
   注意：临床评分的计算在Workup轮（上面第2步）完成，不在此处。

4. Write/Update（写入操作）：问题要求执行临床写入操作（医嘱、标记或转诊）。
   Action_List恰好包含2项（1个写入工具 + prepare_to_answer）。
   本轮仅提供写入工具（MedicationRequest.create、ServiceRequest.create、Flag.create）。
   不要调用读取/搜索工具。

通用规则：
5. Action_List末尾始终以prepare_to_answer结尾。
6. 使用患者工具时，必须同时包含subject_id和session_id（来自env_info）。
7. 仅使用提供的可用工具列表中的工具。
8. 已知患者时，EHR工具参数中始终包含subject_id。
9. 严格使用工具定义中所示的参数名称，不得创造替代名称。

只输出JSON对象，不要包含其他文本。"""


def plan_actions(
    context: dict,
    history: list,
    user_question: str,
    available_tools: list,
    task_type: str,
    tool_source: str = "ehr",
    attempt: int = 0,
    prev_obs_failure: str = "",
    language: str = "en",
    tool_hint: str = "",
) -> dict:
    """
    Plan tool calls for the current user question.

    Args:
        task_type:    Information Lookup / Data Gathering / Clinical Reasoning / Write/Update
        tool_source:  ehr / patient / mixed
        attempt:      Outer retry count (0=first try). Raises temperature on retries.
        prev_obs_failure: Reason from validate_observations() on previous attempt.
        tool_hint:    Suggested tool call(s) from the session planner (e.g. "DiagnosticReport.search(CT)
                      + Observation.search(WBC)"). The planner SHOULD follow this hint unless the
                      data is unavailable.

    Returns dict with Task_Finish, Thought, Plan, Action_List.
    """
    subject_id = context.get("subject_id")
    hadm_id = context.get("hadm_id") or context.get("admission", {}).get("hadm_id")
    session_id = context.get("session_id", "")

    def _fmt_tool(t):
        fn = t["function"]
        props = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        param_lines = []
        for pname, pinfo in props.items():
            req_tag = "required" if pname in required else "optional"
            param_lines.append(
                f"      {pname} ({pinfo.get('type','any')}, {req_tag}): {pinfo.get('description','')}"
            )
        params_str = "\n".join(param_lines) if param_lines else "      (no parameters)"
        return f"  - {fn['name']}: {fn['description']}\n    Parameters:\n{params_str}"

    tool_descriptions = "\n".join(_fmt_tool(t) for t in available_tools)

    history_str = ""
    if history:
        history_str = "\nConversation history:\n"
        for m in history[-6:]:
            role = m.get("role", "")
            content = str(m.get("content", ""))[:400]
            history_str += f"[{role.upper()}]: {content}\n"

    failure_hint = ""
    if prev_obs_failure:
        failure_hint = (
            f"\nWARNING: Previous attempt failed: \"{prev_obs_failure}\". "
            "Choose DIFFERENT tools or arguments.\n"
        )

    tool_hint_section = ""
    if tool_hint:
        tool_hint_section = (
            f"\nSESSION PLANNER TOOL HINT: {tool_hint}\n"
            "You SHOULD include the tools listed above in your Action_List. "
            "Only omit them if the required data is genuinely absent from the EHR.\n"
        )

    ehr_snapshot = context.get("ehr_snapshot", "")
    ehr_snapshot_section = ""
    if ehr_snapshot:
        ehr_snapshot_section = (
            f"\nAvailable EHR Data (pre-fetched — use these EXACT item names in tool arguments):\n"
            f"{ehr_snapshot[:3000]}\n"
        )

    user_prompt = f"""Patient Information:
- subject_id: {subject_id}
- hadm_id: {hadm_id}
- session_id: {session_id}
- {context.get('context_str', '')}
{ehr_snapshot_section}
{history_str}
{failure_hint}
{tool_hint_section}
Current User Question: {user_question}
Task Type: {task_type}
Tool Source: {tool_source}

Available Tools:
{tool_descriptions}

Plan the tool calls needed to answer this question.
Use the EXACT lab test names from the EHR Data section above as item_name arguments.
Output the JSON action plan:"""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_ZH if language == "zh" else _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    base_temp = 0.0 if attempt == 0 else 0.4
    for inner_attempt in range(3):
        try:
            raw = llm_call(
                messages,
                temperature=base_temp if inner_attempt == 0 else base_temp + 0.2,
                max_tokens=4000,
            )
            plan = extract_json(raw)
            assert "Action_List" in plan, "Missing Action_List"
            return plan
        except Exception:
            if inner_attempt == 2:
                # Fallback: minimal plan based on tool_source
                if tool_source == "patient":
                    fallback_actions = [
                        {"name": "patient.get_symptom_history",
                         "arguments": {"subject_id": subject_id, "session_id": session_id}},
                        {"name": "prepare_to_answer", "arguments": {}},
                    ]
                else:
                    fallback_actions = [
                        {"name": "get_patient_info",
                         "arguments": {"subject_id": subject_id}},
                        {"name": "prepare_to_answer", "arguments": {}},
                    ]
                return {
                    "Task_Finish": False,
                    "Thought": "Fallback plan",
                    "Plan": "Get basic patient info",
                    "Action_List": fallback_actions,
                }
    return {"Task_Finish": False, "Thought": "", "Plan": "", "Action_List": []}
