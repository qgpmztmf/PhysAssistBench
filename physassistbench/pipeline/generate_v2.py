"""
physassistbench/pipeline/generate_v2.py — Main benchmark data generation pipeline (v2, FHIR edition).

Extends physassistbench/pipeline/generate_v2.py with HL7 FHIR R4 tool support.

Changes from PhysAssistBench:
  - Default tool_set="fhir": agents see FHIR-named tools (Observation.search, etc.)
  - _get_turn_tools() dispatches to get_fhir_tools_for_task() when tool_set="fhir"
  - execute_actions() receives tool_set="fhir" so FHIR registry is used for execution
  - tool_set is stored in each turn_data entry and the final benchmark entry

All other logic (sequences, subtypes, dep_graph, bilingual) is unchanged from PhysAssistBench.

See docs/fhir_migration_plan.md for the full migration design.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physassistbench.pipeline.patient_selector import select_patients
from physassistbench.pipeline.context_builder import build_context
from physassistbench.tools.tool_registry import set_active_date
from physassistbench.pipeline.agents.planner_agent import plan_actions
from physassistbench.pipeline.agents.checker_planner import validate_plan
from physassistbench.pipeline.agents.tool_executor import execute_actions
from physassistbench.pipeline.agents.checker_tool import validate_observations
from physassistbench.pipeline.agents.answer_agent import generate_answer
from physassistbench.pipeline.agents.clinical_checker import validate_answer
from physassistbench.pipeline.agents.user_answer_agent import generate_user_response
from physassistbench.pipeline.agents.llm_client import llm_call, extract_json
from physassistbench.pipeline.agents.translator import translate_to_zh, translate_messages
from physassistbench.pipeline.encoder import encode_entry
from physassistbench.tools.tool_schemas import get_tools_for_task, get_fhir_tools_for_task

from physassistbench.pipeline.sequences import get_arc, pick_subtype, TASK_TYPES
from physassistbench.pipeline.scenarios import sample_grounding_facts_v2, SCENARIO_NAMES
from physassistbench.pipeline.dep_graph import TurnDependencyGraph
from physassistbench.pipeline.user_agent_v2 import generate_user_turn_v2, _apply_ellipsis_transform
from physassistbench.pipeline.ehr_prefetch import build_ehr_snapshot, has_queryable_data
from physassistbench.pipeline.session_planner import plan_session
from physassistbench.pipeline.rubric_generator import generate_rubric
from physassistbench.tools.write_tool_schemas import WRITE_TOOL_SCHEMAS, WRITE_TOOL_NAMES
from physassistbench.tools.fhir.schemas import FHIR_SCHEMA_BY_NAME

# physassistbench default: use FHIR tools
DEFAULT_TOOL_SET = "fhir"

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# ── Task-type name mapping ────────────────────────────────────────────────────
# The legacy pipeline agents (planner, checker, answer_agent, etc.) use the old
# task type names.  Map new PhysAssistBench names → legacy names when calling those agents.
# The new names are preserved in the stored entry (task_types / turn_data).
#
# "Intake" does NOT appear in PhysAssistBench benchmark entries — patient-source is a
# tool_source dimension, not a task type.  However, the legacy pipeline agents
# (planner prompt, checker LLM) need a name they understand.  When
# tool_source=patient we pass "Intake" to those agents so the LLM does not flag
# patient tools as wrong for "Lookup".  This label is internal only; the stored
# JSONL entry always records the PhysAssistBench task type (Information Lookup/Data Gathering/…).
#
# tool_source mapping:
#   patient → "Intake"  (legacy: accepts 1-4 patient.xxx tools without complaint)
#   mixed   → "Data Gathering"  (legacy: multi-tool, ehr+patient combined)
#   ehr     → structural equivalent per task type

_NEW_TO_LEGACY_EHR: dict[str, str] = {
    "Information Lookup":           "Lookup",
    "Data Gathering":              "Data Gathering",
    "Clinical Reasoning":  "Lookup",   # one tool fetch + knowledge reasoning in answer
}


def _legacy(task_type: str, tool_source: str = "ehr") -> str:
    """Return the legacy task type name for use with legacy pipeline agents only.

    NEVER stored in the benchmark entry — only used when calling plan_actions,
    validate_plan, validate_observations, generate_answer, validate_answer.
    """
    if tool_source == "patient":
        return "Intake"
    if tool_source == "mixed":
        return "Data Gathering"
    return _NEW_TO_LEGACY_EHR.get(task_type, task_type)


# ── Write/Update turn helpers ───────────────────────────────────────────────────────

_ACTION_Q_SYSTEM_EN = """\
You are generating the FINAL turn of a 4-turn EHR benchmark session.
This is an ACTION turn — the clinician asks the assistant to WRITE something to the EHR:
create a medication order, a service referral, or a clinical safety flag.

Based on the clinical findings from the prior conversation, write ONE concise clinical
instruction requesting a specific write action.

Scenario-specific guidance:
  lab_trend:          Flag a critical lab value (Flag.create) OR order a follow-up
                      service/lab referral (ServiceRequest.create).
  med_safety:         Adjust or create a medication order (MedicationRequest.create)
                      OR create a drug-lab safety flag (Flag.create).
  treatment_response: Change/de-escalate antibiotic (MedicationRequest.create) OR
                      create a service referral for step-down/escalation (ServiceRequest.create).

Rules:
- The action MUST be clinically justified by the prior turns' findings.
- Request exactly ONE write operation (do not combine multiple write actions in the question).
- Keep it concise (1-2 sentences). No preamble.
- Do NOT ask to retrieve or look up data — write action only.

Return ONLY the question string, no JSON, no explanation.
"""

_ACTION_Q_SYSTEM_ZH = """\
你正在为4轮EHR基准会话生成最后一轮（ACTION轮）的用户问题。
这一轮是操作轮——临床医生要求助手向EHR写入内容：
创建用药医嘱、服务申请或临床安全标记。

根据前序对话中的临床发现，写一条简洁的临床指令，要求执行一个具体的写操作。

场景指引：
  lab_trend：         标记危急化验值（Flag.create）或开具随访服务/复查申请（ServiceRequest.create）。
  med_safety：        调整或新建用药医嘱（MedicationRequest.create）或创建药-化验安全标记（Flag.create）。
  treatment_response：更换/降阶梯抗生素（MedicationRequest.create）或创建降级/升级护理转介（ServiceRequest.create）。

规则：
- 操作必须有前序轮次的临床发现作为依据。
- 只请求一个写操作（不要在问题中组合多个写操作）。
- 简洁（1-2句），无前言。
- 不要要求检索数据——纯写操作。

只返回问题字符串，不含JSON或解释。
"""


def _generate_action_question_v2(
    session_plan: dict | None,
    history: list[dict],
    ehr_snapshot: str,
    scenario: str,
    language: str = "en",
) -> tuple[str, str]:
    """Generate the Write/Update turn write-request question. Returns (user_q, explicit)."""
    is_zh = language == "zh"
    sys_prompt = _ACTION_Q_SYSTEM_ZH if is_zh else _ACTION_Q_SYSTEM_EN

    tool_hint = ""
    if session_plan and "turns" in session_plan:
        last = next((t for t in reversed(session_plan["turns"]) if t.get("task_type") == "Write/Update"), None)
        if last:
            tool_hint = last.get("tool_hint", "")

    hist_str = ""
    for msg in history[-6:]:
        role = msg.get("role", "").upper()
        hist_str += f"[{role}]: {str(msg.get('content', ''))[:250]}\n"

    if is_zh:
        user_prompt = (
            f"临床场景：{scenario}\n"
            f"患者EHR摘要：{ehr_snapshot[:400]}\n\n"
            f"前序对话：\n{hist_str}\n"
            f"建议的写操作：{tool_hint}\n\n生成Action轮用户问题："
        )
    else:
        user_prompt = (
            f"Clinical scenario: {scenario}\n"
            f"Patient EHR summary: {ehr_snapshot[:400]}\n\n"
            f"Prior conversation:\n{hist_str}\n"
            f"Suggested write action: {tool_hint}\n\nGenerate the Write/Update turn question:"
        )

    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}]
    try:
        raw = llm_call(messages, temperature=0.4, max_tokens=200)
        q = raw.strip().strip('"').strip("`").strip()
        if not q:
            raise ValueError("empty response")
        return q, q
    except Exception as exc:
        logger.warning(f"  _generate_action_question_v2 failed: {exc}")
        # Grounded fallback from tool_hint
        _verbs_en = {"MedicationRequest.create": "create a medication order",
                     "ServiceRequest.create": "create a service referral",
                     "Flag.create": "create a clinical safety flag"}
        _verbs_zh = {"MedicationRequest.create": "创建一条用药医嘱",
                     "ServiceRequest.create": "创建一条服务转介申请",
                     "Flag.create": "创建一条临床安全标记"}
        tool_name = tool_hint.split("(")[0].strip() if tool_hint else ""
        if is_zh:
            verb = _verbs_zh.get(tool_name, "执行一个写操作")
            return f"请根据上述发现为该患者{verb}。", f"请根据上述发现为该患者{verb}。"
        verb = _verbs_en.get(tool_name, "perform a write action")
        fb = f"Based on the findings above, please {verb} for this patient."
        return fb, fb


def _exec_write_tool_v2(name: str, args: dict, subject_id: int) -> dict:
    """Simulate a write tool call. Returns a confirmation dict; no real DB writes."""
    from datetime import datetime as _dt
    ts = _dt.utcnow().isoformat()
    if "subject_id" not in args:
        args["subject_id"] = subject_id
    if name == "MedicationRequest.create":
        return {"resourceType": "MedicationRequest", "id": f"MR-sim-{subject_id}-{ts[:10]}",
                "status": "active", "intent": "order",
                "medicationCodeableConcept": {"text": args.get("medication", "")},
                "subject": {"reference": f"Patient/{subject_id}"},
                "dosageInstruction": [{"text": f"{args.get('dose','')} {args.get('route','')} {args.get('frequency','')}".strip()}],
                "reasonCode": [{"text": args.get("indication", "")}] if args.get("indication") else [],
                "_simulated": True, "_created_at": ts}
    if name == "ServiceRequest.create":
        return {"resourceType": "ServiceRequest", "id": f"SR-sim-{subject_id}-{ts[:10]}",
                "status": "active", "intent": "order",
                "priority": args.get("priority", "routine"),
                "code": {"text": args.get("service_type", "")},
                "subject": {"reference": f"Patient/{subject_id}"},
                "note": [{"text": args.get("note", "")}] if args.get("note") else [],
                "_simulated": True, "_created_at": ts}
    if name == "Flag.create":
        return {"resourceType": "Flag", "id": f"FL-sim-{subject_id}-{ts[:10]}",
                "status": "active",
                "category": [{"text": args.get("category", "")}],
                "code": {"text": args.get("code", ""), "coding": [{"display": args.get("detail", "")}]},
                "subject": {"reference": f"Patient/{subject_id}"},
                "_simulated": True, "_created_at": ts}
    return {"error": f"Unknown write tool: {name}"}


def _validate_action_plan_v2(plan: dict, available_tool_names: list[str]) -> tuple[bool, str]:
    """Accept if ≥1 write tool (not prepare_to_answer) is in the action list."""
    action_list = plan.get("Action_List", [])
    write_calls = [a for a in action_list
                   if a.get("name") in WRITE_TOOL_NAMES and a.get("name") != "prepare_to_answer"]
    if not write_calls:
        return False, f"Action plan must include ≥1 write tool, got: {[a.get('name') for a in action_list]}"
    for a in action_list:
        if a.get("name") not in available_tool_names:
            return False, f"Unknown tool in action plan: {a.get('name')}"
    return True, ""


# ── Tool-source decision ──────────────────────────────────────────────────────

def decide_tool_source(
    task_type: str,
    grounding_facts: str,
    scenario: str,
    turn_plan: dict | None = None,
) -> str:
    """
    Return tool_source for this turn.
    Reads tool_source from the session plan's per-turn data when available;
    otherwise defaults to 'ehr'.  Valid values: 'ehr' | 'patient' | 'mixed'.
    """
    if turn_plan and "tool_source" in turn_plan:
        src = turn_plan["tool_source"]
        if src in ("ehr", "patient", "mixed"):
            return src
    return "ehr"


def _get_turn_tools(task_type: str, tool_source: str, task_domain: str,
                    language: str = "en", tool_set: str = DEFAULT_TOOL_SET):
    """Return the appropriate tool list given task type, tool_source, and tool_set."""
    # Write/Update turns always expose write tools + prepare_to_answer
    if task_type == "Write/Update" or tool_source == "write":
        prepare = FHIR_SCHEMA_BY_NAME.get("prepare_to_answer")
        return WRITE_TOOL_SCHEMAS + ([prepare] if prepare else [])

    if tool_set == "fhir":
        ehr_tools = get_fhir_tools_for_task(task_domain)
        patient_tools = get_tools_for_task("PatientInterview", language=language)
    else:
        ehr_tools = get_tools_for_task(task_domain, language=language)
        patient_tools = get_tools_for_task("PatientInterview", language=language)

    if tool_source == "patient":
        return patient_tools
    elif tool_source == "mixed":
        existing = {t["function"]["name"] for t in ehr_tools}
        combined = ehr_tools + [t for t in patient_tools if t["function"]["name"] not in existing]
        return combined
    else:
        return ehr_tools


def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj



# ── Main generation function ──────────────────────────────────────────────────

def generate_one_entry_v2(
    task_domain: str,
    subject_id: int,
    hadm_id: int | None,
    entry_index: int,
    scenario: str,
    sequence_idx: int | None = None,
    n_turns: int = 4,
    session_id: str | None = None,
    persona: dict | None = None,
    language: str = "en",
    bilingual: bool = True,
    tool_set: str = DEFAULT_TOOL_SET,
    scenario_constraints: dict | None = None,
    difficulty: int = 1,
    difficulty_constraints: dict | None = None,
    subtype_counter: dict[str, int] | None = None,
    tools_to_prioritize: list[str] | None = None,
    require_patient_turn: bool = False,
) -> dict | None:
    """
    Generate one benchmark entry under the new 4-task-type / 4-subtype framework.

    Args:
        task_domain: EHR domain for tool selection (LabInterp / MedRecon / ...)
        subject_id:  MIMIC-IV patient identifier
        hadm_id:     Hospital admission ID (may be None)
        entry_index: Index for unique ID construction
        scenario:    Clinical scenario from SCENARIO_NAMES (11 options)
        sequence_idx: Arc index 0-35 (T0=Information Lookup fixed; T1-T3 from R/W/KG; or T3=Action)
        n_turns:     Number of turns (default 4)
        require_patient_turn: If True, guarantee at least one mixed/patient turn per session.
            When the session planner does not designate any patient turn, the last eligible
            EHR turn (non-T0, non-Action) is forced to tool_source="mixed".
        session_id:  Patient Agent session ID
        persona:     Patient persona config
        language:    'en' | 'zh' (primary generation language)
        bilingual:   If True, generate in `language` then translate to the other

    Returns None on failure.
    """
    entry_id = f"physassistbench_{task_domain}_{scenario}_{entry_index}_L{difficulty}"
    logger.info(f"Generating {entry_id}  subject={subject_id}  hadm={hadm_id}  scenario={scenario}")

    # Build patient context
    context = build_context(subject_id, hadm_id, task_domain)
    if session_id:
        context["session_id"] = session_id
    if persona:
        context["persona"] = persona

    # Gate all FHIR time-ordered queries to the admission's discharge date so
    # observations from future admissions cannot pollute gold answers or rubrics.
    set_active_date(context.get("current_date"))

    # Pre-fetch all EHR data from all tables with human-readable item names.
    # Always extract everything — maximum grounding for user agent and planner.
    ehr_snapshot = build_ehr_snapshot(subject_id, hadm_id or context.get("hadm_id"))
    context["ehr_snapshot"] = ehr_snapshot
    logger.info(f"  EHR snapshot built ({len(ehr_snapshot)} chars)")

    # Skip patients with no EHR data at all
    if not has_queryable_data(subject_id, hadm_id or context.get("hadm_id")):
        logger.warning(f"  Skipping {entry_id}: no EHR data available")
        return None

    # Select task-type arc (T0 always Information Lookup, T1–T3 from arc definition)
    if sequence_idx is None:
        sequence_idx = random.randint(0, 26)
    task_sequence = get_arc(sequence_idx)[:n_turns]
    logger.info(f"  sequence_idx={sequence_idx}  path={task_sequence}  (subtypes: dynamic)")

    # Dependency graph — tracks antecedents turn-by-turn
    dep_graph = TurnDependencyGraph()

    # Per-entry subtype increments — merged into subtype_counter only on full
    # entry success so that a partial failure doesn't bias the global counter.
    local_subtype_increments: dict[str, int] = {}

    gen_lang = language

    # Generate session-level plan before entering the turn loop
    session_plan = plan_session(
        ehr_snapshot=ehr_snapshot,
        task_sequence=task_sequence,
        scenario=scenario,
        n_turns=n_turns,
        language=gen_lang,
        scenario_constraints=scenario_constraints,
        difficulty_constraints=difficulty_constraints,
        tools_to_prioritize=tools_to_prioritize,
    )
    turn_data = []
    history: list[dict] = []
    tasks_zh: list[str] = []
    messages_zh: list[dict] = []
    _patient_turn_used = False  # enforce at most one patient/mixed turn per session

    # Pre-compute the forced patient turn index when require_patient_turn=True.
    # Choose the last non-T0, non-Action eligible EHR turn so that EHR context
    # is already built before the patient interview happens.
    _forced_patient_turn_idx: int | None = None
    if require_patient_turn:
        for _ti in range(n_turns - 1, 0, -1):   # T0 excluded (no context yet)
            if task_sequence[_ti] != "Write/Update":
                _forced_patient_turn_idx = _ti
                break
        if _forced_patient_turn_idx is not None:
            logger.info(
                f"  require_patient_turn=True: will force tool_source='mixed' at T{_forced_patient_turn_idx} "
                "if session planner does not designate a patient turn first"
            )

    for turn_idx in range(n_turns):
        task_type = task_sequence[turn_idx]
        # Dynamic subtype selection: pick the least-used feasible subtype.
        # T0 always returns None (no prior context). T1+ query dep_graph for
        # available antecedents, then balance across NA/PE/AE via counter.
        subtype = pick_subtype(turn_idx, dep_graph, task_type, subtype_counter)

        # Sample grounding facts for this scenario
        grounding_facts = sample_grounding_facts_v2(
            subject_id=subject_id,
            hadm_id=hadm_id or context.get("hadm_id"),
            scenario=scenario,
        )

        # Extract per-turn plan first — tool_source decision reads from it
        turn_plan: dict | None = None
        turn_intent: str | None = None
        if session_plan and "turns" in session_plan:
            planned = session_plan["turns"]
            if turn_idx < len(planned):
                turn_plan = planned[turn_idx]
        if session_plan and "turn_intents" in session_plan:
            intents = session_plan["turn_intents"]
            if turn_idx < len(intents):
                turn_intent = intents[turn_idx]

        # Decide tool source — reads tool_source field from session plan if present.
        # Guard: downgrade to 'ehr' if a patient/mixed turn has already been used
        # this session (LLM may assign tool_source='mixed' to multiple turns).
        tool_source = decide_tool_source(task_type, grounding_facts, scenario,
                                         turn_plan=turn_plan)
        if tool_source in ("patient", "mixed"):
            if _patient_turn_used:
                logger.info(
                    f"  turn {turn_idx}: downgrading tool_source '{tool_source}' → 'ehr' "
                    "(patient turn already used this session)"
                )
                tool_source = "ehr"
            else:
                _patient_turn_used = True

        # require_patient_turn enforcement: if the switch is on and the session planner
        # did not voluntarily assign a patient turn, force the designated turn to "mixed".
        if (
            require_patient_turn
            and not _patient_turn_used
            and turn_idx == _forced_patient_turn_idx
            and task_type != "Write/Update"
        ):
            logger.info(
                f"  turn {turn_idx}: require_patient_turn — forcing tool_source 'ehr' → 'mixed'"
            )
            tool_source = "mixed"
            _patient_turn_used = True

        # Get tools for this turn
        turn_tools = _get_turn_tools(task_type, tool_source, task_domain,
                                     language=gen_lang, tool_set=tool_set)
        turn_tool_names = [t["function"]["name"] for t in turn_tools]

        # Fetch antecedents for the chosen subtype (pick_subtype already verified
        # they are non-empty, so effective_subtype == subtype unconditionally).
        antecedents = []
        if subtype and turn_idx > 0:
            antecedents = dep_graph.get_antecedents(turn_idx, subtype)
        effective_subtype = subtype

        success = False
        _max_attempts = MAX_RETRIES
        obs_reason = ""
        failed_questions: list[dict] = []

        for attempt in range(_max_attempts):
            try:
                # ── ACTION turn ───────────────────────────────────────────────
                if task_type == "Write/Update":
                    # 1. Generate write-request question
                    user_q, user_q_explicit = _generate_action_question_v2(
                        session_plan=session_plan,
                        history=history,
                        ehr_snapshot=ehr_snapshot,
                        scenario=scenario,
                        language=gen_lang,
                    )
                    logger.info(
                        f"  turn {turn_idx} [Write/Update/write] Q: {user_q[:80]}"
                    )

                    # 2. Plan write tool calls
                    plan = plan_actions(
                        context=context,
                        history=history,
                        user_question=user_q,
                        available_tools=turn_tools,
                        task_type="Write/Update",
                        tool_source="write",
                        attempt=attempt,
                        prev_obs_failure="",
                        language=gen_lang,
                    )

                    # 3. Validate: must include ≥1 write tool
                    is_valid, reason = _validate_action_plan_v2(plan, turn_tool_names)
                    if not is_valid:
                        if attempt < _max_attempts - 1:
                            logger.warning(f"  Action plan invalid: {reason}. Retrying...")
                            continue
                        else:
                            logger.warning(f"  Action plan invalid (final): {reason}. Discarding.")
                            break

                    # 4. Execute: route write tools to simulator, prepare_to_answer as-is
                    action_list = plan.get("Action_List", [])
                    if not action_list:
                        action_list = [{"name": "prepare_to_answer", "arguments": {}}]
                    executed_actions = []
                    for i, act in enumerate(action_list):
                        name = act.get("name", "")
                        args = dict(act.get("arguments", {}))
                        if name in WRITE_TOOL_NAMES and name != "prepare_to_answer":
                            obs = _exec_write_tool_v2(name, args, subject_id)
                        elif name == "prepare_to_answer":
                            obs = ""
                        else:
                            obs = {"error": f"Non-write tool in Write/Update turn: {name}"}
                        executed_actions.append({
                            "action": {"name": name, "arguments": args},
                            "observation": obs,
                            "dependency_list": list(range(i)) if name == "prepare_to_answer" else [],
                            "idx": i,
                        })

                    # 5. Skip observation validation (write confirmations always succeed)

                    # 6. Generate answer
                    assistant_answer = generate_answer(
                        context=context,
                        history=history,
                        user_question=user_q,
                        executed_actions=executed_actions,
                        task_type="Data Gathering",
                        tool_source="write",
                        language=gen_lang,
                    )

                    # 7. Skip answer validation for write turns

                    # 8. Generate action rubric
                    from physassistbench.pipeline.generate_discharge_planning import (
                        generate_action_rubric, _serialize_write_actions,
                    )
                    rubric_en = generate_action_rubric(
                        user_question=user_q_explicit,
                        executed_write_actions=executed_actions,
                        reference_answer=assistant_answer,
                        clinical_context=ehr_snapshot[:600],
                        language="en",
                    )

                    turn_data.append({
                        "user_question": user_q,
                        "user_question_explicit": user_q_explicit,
                        "task_type": task_type,
                        "subtype": None,
                        "tool_source": "write",
                        "tool_set": tool_set,
                        "executed_actions": executed_actions,
                        "assistant_answer": assistant_answer,
                        "sequence_idx": sequence_idx,
                        "workup_mode": None,
                        "rubric": rubric_en,
                        "write_gold_params": [
                            {"tool": a["action"]["name"],
                             "params": {k: v for k, v in a["action"]["arguments"].items() if k != "subject_id"}}
                            for a in executed_actions if a["action"]["name"] in WRITE_TOOL_NAMES
                            and a["action"]["name"] != "prepare_to_answer"
                        ],
                    })

                else:
                # ── EHR / patient turn ────────────────────────────────────────
                    plan_fail_reason = ""  # carries plan validation failure across retries

                # 1. Generate user question (two-stage: explicit → transform)
                    user_q, user_q_explicit = generate_user_turn_v2(
                        context=context,
                        scenario=scenario,
                        task_type=task_type,
                        subtype=effective_subtype,
                        tool_source=tool_source,
                        history=history,
                        language=gen_lang,
                        grounding_facts=grounding_facts,
                        antecedents=antecedents if effective_subtype else None,
                        ehr_snapshot=ehr_snapshot,
                        failed_questions=failed_questions if failed_questions else None,
                        turn_plan=turn_plan,
                        turn_intent=turn_intent,
                    )

                    # PE fallback: when Stage-2 transform was skipped (antecedent check
                    # failed), try a PE transform as a last resort.
                    if (
                        user_q == user_q_explicit
                        and effective_subtype
                        and effective_subtype != "PE"
                    ):
                        pe_ants = dep_graph.get_antecedents(turn_idx, "PE")
                        if pe_ants:
                            fallback_q = _apply_ellipsis_transform(
                                explicit_q=user_q_explicit,
                                subtype="PE",
                                history=history,
                                antecedents=pe_ants,
                                language=gen_lang,
                                ehr_snapshot=ehr_snapshot,
                                task_type=task_type,
                            )
                            if fallback_q != user_q_explicit:
                                logger.info(f"  turn {turn_idx}: {effective_subtype} → PE fallback")
                                user_q = fallback_q
                                effective_subtype = "PE"
                                # fallback_q is a PE abbreviation of user_q_explicit.
                                # Verify they share content words; if Stage-1 is unrelated
                                # to the PE form (LLM drift), rebuild explicit from the PE form.
                                _stop = {
                                    'what','the','is','are','a','an','and','or','for','with',
                                    'that','this','in','of','to','how','does','did','any','do',
                                    'it','was','were','on','at','by','from','has','have','been',
                                    'be','so','now','look','we','her','his','their','?','s','t',
                                    'its','our','pull','check','still','ordered','active',
                                }
                                fq_words = set(fallback_q.lower().split()) - _stop
                                eq_words = set(user_q_explicit.lower().split()) - _stop
                                if fq_words and eq_words and not (fq_words & eq_words):
                                    # No content-word overlap: Stage-1 drifted to a different
                                    # clinical topic. Reconstruct explicit as the full form of
                                    # the PE-fallback question so the conversation chain is
                                    # preserved (every turn must depend on prior turns).
                                    from physassistbench.pipeline.user_agent_v2 import (
                                        _expand_pe_to_explicit,
                                    )
                                    rebuilt = _expand_pe_to_explicit(
                                        pe_q=fallback_q,
                                        history=history,
                                        language=gen_lang,
                                    )
                                    if rebuilt and rebuilt != fallback_q:
                                        user_q_explicit = rebuilt
                                        logger.info(
                                            f"  turn {turn_idx}: rebuilt explicit from PE: {rebuilt[:80]}"
                                        )

                    logger.info(
                        f"  turn {turn_idx} [{task_type}/{effective_subtype}/{tool_source}] "
                        f"Q: {user_q[:80]}"
                    )

                    # 2. Plan tool calls
                    # Combine obs failure and plan validation failure as hints for retries
                    _retry_hint = obs_reason if attempt > 0 else ""
                    if plan_fail_reason:
                        _retry_hint = plan_fail_reason if not _retry_hint else f"{_retry_hint}; {plan_fail_reason}"
                    # Extract tool_hint from session plan for this turn
                    _tool_hint = turn_plan.get("tool_hint", "") if turn_plan else ""

                    plan = plan_actions(
                        context=context,
                        history=history,
                        user_question=user_q,
                        available_tools=turn_tools,
                        task_type=task_type,
                        tool_source=tool_source,
                        attempt=attempt,
                        prev_obs_failure=_retry_hint,
                        language=gen_lang,
                        tool_hint=_tool_hint,
                    )

                    # 3. Validate plan
                    is_valid, reason = validate_plan(
                        user_question=user_q,
                        task_type=task_type,
                        plan=plan,
                        available_tool_names=turn_tool_names,
                        subject_id=subject_id,
                        tool_source=tool_source,
                        language=gen_lang,
                    )
                    if not is_valid:
                        plan_fail_reason = reason  # pass to next attempt
                        if attempt < _max_attempts - 1:
                            logger.warning(f"  Plan check failed: {reason}. Retrying...")
                            continue
                        else:
                            logger.warning(f"  Plan check failed on final attempt: {reason}. Discarding turn.")
                            break

                    # 4. Execute tools
                    action_list = plan.get("Action_List", [])
                    if not action_list:
                        action_list = [{"name": "prepare_to_answer", "arguments": {}}]

                    executed_actions = execute_actions(
                        action_list=action_list,
                        subject_id=subject_id,
                        hadm_id=hadm_id,
                        tool_set=tool_set,
                    )

                    # 5. Validate observations
                    obs_ok, obs_reason = validate_observations(executed_actions)
                    if not obs_ok:
                        failed_questions.append({"question": user_q, "reason": obs_reason})
                        if attempt < _max_attempts - 1:
                            logger.warning(f"  Obs check failed: {obs_reason}. Retrying...")
                            continue
                        else:
                            logger.warning(
                                f"  Obs check failed on final attempt: {obs_reason}. "
                                "Discarding turn — EHR has no data for this question."
                            )
                            break

                    # 6. Generate assistant answer
                    assistant_answer = generate_answer(
                        context=context,
                        history=history,
                        user_question=user_q,
                        executed_actions=executed_actions,
                        task_type=task_type,
                        tool_source=tool_source,
                        language=gen_lang,
                    )

                    # 7. Validate answer
                    ans_ok, ans_reason = validate_answer(
                        user_question=user_q,
                        executed_actions=executed_actions,
                        assistant_answer=assistant_answer,
                        task_domain=task_domain,
                        language=gen_lang,
                    )
                    if not ans_ok:
                        if attempt < _max_attempts - 1:
                            logger.warning(f"  Answer check failed: {ans_reason}. Retrying...")
                            continue
                        else:
                            logger.warning(f"  Answer check failed on final attempt: {ans_reason}. Discarding turn.")
                            break

                    # 8. Generate rubric grounded in actual EHR values
                    rubric_en = generate_rubric(
                        task_type=task_type,
                        user_question_explicit=user_q_explicit,
                        executed_actions=executed_actions,
                        assistant_answer=assistant_answer,
                        language="en",
                        tool_source=tool_source,
                    )

                    turn_data.append({
                        "user_question": user_q,
                        "user_question_explicit": user_q_explicit,
                        "task_type": task_type,
                        "subtype": effective_subtype,
                        "tool_source": tool_source,
                        "tool_set": tool_set,
                        "executed_actions": executed_actions,
                        "assistant_answer": assistant_answer,
                        "sequence_idx": sequence_idx,
                        "workup_mode": (
                            _infer_workup_mode(executed_actions)
                            if task_type == "Data Gathering" else None
                        ),
                        "rubric": rubric_en,
                    })

                dep_graph.add_turn(
                    turn_idx=turn_idx,
                    task_type=task_type,
                    tool_source=tool_source,
                    subtype=effective_subtype,
                    executed_actions=executed_actions,
                    assistant_answer=assistant_answer,
                )

                history.append({"role": "user", "content": user_q})
                history.append({"role": "assistant", "content": assistant_answer})

                if bilingual and gen_lang == "en":
                    zh_q = translate_to_zh(user_q)
                    zh_ans = translate_to_zh(assistant_answer)
                    tasks_zh.append(zh_q)
                    messages_zh.extend([
                        {"role": "user", "content": zh_q},
                        {"role": "assistant", "content": zh_ans},
                    ])
                    # Also translate the explicit (pre-transform) question
                    turn_data[-1]["user_question_explicit_zh"] = (
                        translate_to_zh(user_q_explicit)
                        if user_q_explicit != user_q else zh_q
                    )
                elif bilingual and gen_lang == "zh":
                    pass

                # Accumulate locally; merge to global only when entry fully succeeds.
                if subtype_counter is not None and effective_subtype:
                    local_subtype_increments[effective_subtype] = (
                        local_subtype_increments.get(effective_subtype, 0) + 1
                    )

                success = True
                break

            except Exception as exc:
                logger.warning(f"  Turn {turn_idx} attempt {attempt+1} error: {exc}")
                if attempt >= _max_attempts - 1:
                    logger.error(f"  Turn {turn_idx} failed after {_max_attempts} attempts.")
                    return None

        if not success:
            logger.error(f"  [FAILED] turn {turn_idx} did not succeed.")
            return None

    # All turns succeeded — merge local increments into global counter now.
    if subtype_counter is not None:
        for st, cnt in local_subtype_increments.items():
            subtype_counter[st] = subtype_counter.get(st, 0) + cnt

    # ── Encode entry ─────────────────────────────────────────────────────────
    entry = encode_entry(
        entry_id=entry_id,
        task_domain=task_domain,
        subject_id=subject_id,
        hadm_id=hadm_id,
        context=context,
        turn_data=turn_data,
        session_id=session_id,
        tasks_zh=tasks_zh if bilingual and tasks_zh else None,
        messages_zh=messages_zh if bilingual and messages_zh else None,
        tool_set=tool_set,
    )

    # ── Scenario-specific session-level validation ────────────────────────────
    # After all turns succeed, verify mandatory tool coverage per scenario.
    # Return None to discard the entry and retry with a different patient.
    _used_tools: set[str] = {
        act["action"]["name"]
        for td in turn_data
        for act in td["executed_actions"]
    }
    _MANDATORY_TOOLS: dict[str, str] = {
        "diagnostic_workup": "DiagnosticReport.search",
    }
    _required = _MANDATORY_TOOLS.get(scenario)
    if _required and _required not in _used_tools:
        logger.warning(
            f"  {entry_id}: {scenario} ideally requires {_required} but it was never called "
            f"(tools used: {sorted(_used_tools - {'prepare_to_answer','ask_user_for_required_parameters'})}). "
            "Keeping entry — DiagnosticReport data may be absent for this patient/admission."
        )

    # Inject PhysAssistBench-specific annotations
    entry["clinical_scenario"] = scenario
    entry["sequence_idx"] = sequence_idx
    entry["task_sequence"] = task_sequence
    entry["turn_subtypes"] = [td["subtype"] for td in turn_data]
    entry["tool_sources"] = [td["tool_source"] for td in turn_data]
    entry["workup_modes"] = [td.get("workup_mode") for td in turn_data]
    entry["dep_graph"] = dep_graph.summary()
    entry["session_plan"] = session_plan  # None if planning failed
    entry["tool_set"] = tool_set      # "fhir" or "legacy"
    entry["difficulty_level"] = difficulty
    entry["current_date"] = context.get("current_date")  # admission-anchored date upper bound
    entry["generated_at"] = datetime.utcnow().isoformat()

    # Per-turn rubrics (English, grounded in actual EHR values)
    entry["rubrics"] = [td.get("rubric", []) for td in turn_data]

    # Chinese rubrics: translate each item
    if bilingual:
        from physassistbench.pipeline.agents.translator import translate_to_zh as _t
        entry["rubrics_zh"] = [
            [_t(item) for item in turn_rubric]
            for turn_rubric in entry["rubrics"]
        ]

    return _sanitize(entry)


def _infer_workup_mode(executed_actions: list[dict]) -> str:
    """
    Heuristic: if all non-prepare_to_answer actions have dependency_list == [],
    they are parallel. Otherwise adaptive (conditional branching).
    """
    real_actions = [
        a for a in executed_actions
        if a["action"]["name"] not in ("prepare_to_answer", "ask_user_for_required_parameters")
    ]
    if len(real_actions) < 2:
        return "single"
    all_independent = all(
        not a.get("dependency_list") for a in real_actions
    )
    return "parallel" if all_independent else "adaptive"
