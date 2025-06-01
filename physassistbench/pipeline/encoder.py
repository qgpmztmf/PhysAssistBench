"""
Encoder — formats a completed multi-turn generation into WildToolBench JSONL format.
"""

import json
from physassistbench.tools.tool_schemas import get_tools_for_task, get_fhir_tools_for_task
from physassistbench.tools.write_tool_schemas import WRITE_TOOL_SCHEMAS, WRITE_TOOL_NAMES

SEP = "=" * 60  # Turn separator (mirrors WildToolBench)


def encode_entry(
    entry_id: str,
    task_domain: str,
    subject_id: int,
    hadm_id: int | None,
    context: dict,
    turn_data: list,  # list of dicts per turn
    session_id: str | None = None,
    tasks_zh: list | None = None,        # parallel Chinese user questions (bilingual mode)
    messages_zh: list | None = None,     # parallel Chinese messages (bilingual mode)
    tools_zh: list | None = None,        # parallel Chinese tool schemas (bilingual mode)
    answer_list_zh: list | None = None,  # parallel Chinese ground-truth answers (bilingual mode)
    tool_set: str = "legacy",            # "fhir" or "legacy" — controls which tool schemas go in `tools`
) -> dict:
    """
    Build a WildToolBench-format JSONL entry from completed turn data.

    turn_data: list of {
        "user_question": str,
        "task_type": str,       # Single-Tool / Multi-Tool / Clarify / Chat
        "subtype": str | None,  # None for turn 0
        "executed_actions": list,  # from tool_executor
        "assistant_answer": str,
    }

    Returns dict ready for json.dumps(...) + newline.
    """
    tools = get_fhir_tools_for_task(task_domain) if tool_set == "fhir" else get_tools_for_task(task_domain)

    # If any turn is an Write/Update turn, merge write tool schemas into the tool list
    # so evaluated models can issue proper write tool calls at T3.
    if any(t.get("task_type") == "Write/Update" for t in turn_data):
        existing_names = {t["function"]["name"] for t in tools}
        tools = tools + [t for t in WRITE_TOOL_SCHEMAS if t["function"]["name"] not in existing_names]

    tasks = [t["user_question"] for t in turn_data]
    tasks_explicit = [t.get("user_question_explicit", t["user_question"]) for t in turn_data]
    tasks_explicit_zh = [t["user_question_explicit_zh"] for t in turn_data
                         if "user_question_explicit_zh" in t]
    task_types = [t["task_type"] for t in turn_data]
    turn_subtypes = [t["subtype"] for t in turn_data if t.get("subtype")]
    # Mirrors WildToolBench: turn_types[0]=False (first turn), True for subsequent turns
    turn_types = [i != 0 for i in range(len(turn_data))]

    # Build messages: user + assistant per turn, separated by SEP strings.
    # For Clarify (MDV) turns, insert intermediate clarification exchange messages
    # between the user question and the final answer.
    messages = []
    for i, td in enumerate(turn_data):
        if i > 0:
            messages.append(SEP)
        messages.append({"role": "user", "content": td["user_question"]})
        for exchange in td.get("clarify_exchanges", []):
            messages.append({"role": "assistant", "content": exchange["clarify_q"]})
            messages.append({"role": "user", "content": exchange["user_reply"]})
        messages.append({"role": "assistant", "content": td["assistant_answer"]})

    # Build answer_list: one list per turn with actions + observations
    answer_list = []
    for td in turn_data:
        turn_actions = []
        for act in td["executed_actions"]:
            entry_act = {
                "action": act["action"],
                "observation": act["observation"],
                "dependency_list": act["dependency_list"],
                "idx": act["idx"],
            }
            # Preserve user_input for ask_user_for_required_parameters actions
            if act.get("user_input") is not None:
                entry_act["user_input"] = act["user_input"]
            turn_actions.append(entry_act)
        # Ensure prepare_to_answer is last and carries the answer text as observation
        # (mirrors WildToolBench: prepare_to_answer.observation = assistant answer text)
        answer_text = td.get("assistant_answer", "")
        # Remove any existing prepare_to_answer (we'll re-add with correct observation)
        turn_actions = [a for a in turn_actions if a["action"]["name"] != "prepare_to_answer"]
        prep_dep = list(range(len(turn_actions)))
        turn_actions.append({
            "action": {"name": "prepare_to_answer",
                       "arguments": {"answer_type": "tool" if prep_dep else "chat"}},
            "observation": answer_text,
            "dependency_list": prep_dep,
            "idx": len(turn_actions),
        })
        answer_list.append(turn_actions)

    # env_info: inject session_id so Doctor Agent can pass it to patient tools
    env_info = context.get("env_info", "")
    if session_id:
        env_info = f"session_id: {session_id}\n{env_info}" if env_info else f"session_id: {session_id}"

    # For entries with Intake turns, include PatientInterview tools in the full tool list
    has_intake = "Intake" in task_types
    if has_intake:
        from physassistbench.tools.tool_schemas import get_tools_for_task as _get
        patient_tools = _get("PatientInterview")
        patient_tool_names = {t["function"]["name"] for t in patient_tools}
        existing_names = {t["function"]["name"] for t in tools}
        tools = tools + [t for t in patient_tools if t["function"]["name"] not in existing_names]

    entry = {
        "id": entry_id,
        "env_info": env_info,
        # Bilingual tool schemas — tools_en / tools_zh store parallel language versions.
        # `tools` and `tools_en` are always the English schemas (backward compat).
        "tools": tools,
        "tools_en": tools,
        # Bilingual question/message fields.
        # `tasks` and `messages` remain for backward compatibility (= English versions).
        "tasks": tasks,
        "tasks_en": tasks,
        "tasks_explicit": tasks_explicit,
        "tasks_en_explicit": tasks_explicit,
        "messages": messages,
        "messages_en": messages,
        "answer_list": answer_list,
        "turn_types": turn_types,
        "task_types": task_types,
        "turn_subtypes": turn_subtypes,
        # Clinical extensions
        "clinical_task_domain": task_domain,
        "subject_id": subject_id,
        "hadm_id": hadm_id,
    }
    if tasks_zh is not None:
        entry["tasks_zh"] = tasks_zh
    if tasks_explicit_zh:
        entry["tasks_zh_explicit"] = tasks_explicit_zh
    if messages_zh is not None:
        entry["messages_zh"] = messages_zh
    if tools_zh is not None:
        entry["tools_zh"] = tools_zh
    if answer_list_zh is not None:
        entry["answer_list_zh"] = answer_list_zh
    if session_id:
        entry["session_id"] = session_id
    return entry
