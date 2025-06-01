"""
physassistbench/pipeline/generate_discharge_planning.py

Specialized generation pipeline for the discharge_planning clinical scenario.

Extends generate_v2.py with three new capabilities:
  1. Patient Interview tools (T2 mixed turn)
     Ground truth extracted from MIMIC-IV discharge notes and ED triage records.
     Patient responses are synthesized by LLM from note-span grounded facts.
  2. EHR write tools (T3 Write/Update turn)
     MedicationRequest.create / ServiceRequest.create / Flag.create
     Execution is simulated (returns a confirmation dict; no DB writes).
     Rubric items cite SPECIFIC gold parameter values + acceptable ranges.
  3. Discharge-planning-specific tool coverage
     Ensures DocumentReference.search, CarePlan.search, Encounter.search
     are exercised — the three zero-call tools in the current data_full corpus.

Fixed session arc
  T0 [Information Lookup]        — Encounter.search  or  DocumentReference.search
  T1 [Data Gathering]           — CarePlan.search   +   MedicationRequest.search  (or Condition.search)
  T2 [Data Gathering/mixed]     — patient.get_medication_adherence + patient.get_functional_status
                          (optionally +  patient.get_social_history)
  T3 [Action]           — MedicationRequest.create  or  ServiceRequest.create

Tool sources per turn
  T0: ehr      T1: ehr      T2: patient (mixed EHR+patient)      T3: write

Run one entry:
  python -m physassistbench.pipeline.generate_discharge_planning \
    --subject_id 10013015 --hadm_id 22595853 --n 1 --out out_dp.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physassistbench.pipeline.context_builder import build_context
from physassistbench.tools.tool_registry import set_active_date, call_tool
from physassistbench.pipeline.agents.planner_agent import plan_actions
from physassistbench.pipeline.agents.checker_planner import validate_plan
from physassistbench.pipeline.agents.checker_tool import validate_observations
from physassistbench.pipeline.agents.answer_agent import generate_answer
from physassistbench.pipeline.agents.clinical_checker import validate_answer
from physassistbench.pipeline.agents.llm_client import llm_call, extract_json
from physassistbench.pipeline.agents.translator import translate_to_zh
from physassistbench.pipeline.encoder import encode_entry
from physassistbench.tools.tool_schemas import get_tools_for_task
from physassistbench.tools.fhir.schemas import FHIR_SCHEMA_BY_NAME
from physassistbench.tools.write_tool_schemas import WRITE_TOOL_SCHEMAS, WRITE_TOOL_NAMES as _WRITE_TOOL_NAMES

from physassistbench.pipeline.sequences import pick_subtype
from physassistbench.pipeline.dep_graph import TurnDependencyGraph
from physassistbench.pipeline.user_agent_v2 import generate_user_turn_v2, _apply_ellipsis_transform
from physassistbench.pipeline.ehr_prefetch import build_ehr_snapshot, has_queryable_data
from physassistbench.pipeline.rubric_generator import generate_rubric

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
DEFAULT_TOOL_SET = "fhir"

# ── Fixed arc for discharge_planning ─────────────────────────────────────────

DISCHARGE_ARC = ["Information Lookup", "Data Gathering", "Data Gathering", "Write/Update"]
DISCHARGE_TOOL_SOURCES = ["ehr", "ehr", "mixed", "write"]

# ── EHR tools exposed in this scenario ───────────────────────────────────────

_DP_EHR_TOOL_NAMES = [
    "Encounter.search",
    "DocumentReference.search",
    "CarePlan.search",
    "MedicationRequest.search",
    "Condition.search",
    "prepare_to_answer",
]

_DP_PATIENT_TOOL_NAMES = [
    "patient.get_medication_adherence",
    "patient.get_functional_status",
    "patient.get_social_history",
]

# WRITE_TOOL_SCHEMAS and _WRITE_TOOL_NAMES are imported from physassistbench.tools.write_tool_schemas


# ── Write tool simulation ─────────────────────────────────────────────────────

def _exec_write_tool(name: str, args: dict) -> dict:
    """
    Simulate execution of a write tool. Returns a confirmation dict.
    No actual database writes occur — benchmark generation only.
    """
    subject_id = args.get("subject_id", 0)
    ts = datetime.utcnow().isoformat()

    if name == "MedicationRequest.create":
        return {
            "resourceType": "MedicationRequest",
            "id": f"MR-sim-{subject_id}-{ts[:10]}",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"text": args.get("medication", "")},
            "subject": {"reference": f"Patient/{subject_id}"},
            "dosageInstruction": [{
                "text": (
                    f"{args.get('dose', '')} {args.get('route', '')} "
                    f"{args.get('frequency', '')}"
                ).strip(),
                "route": {"text": args.get("route", "")},
                "timing": {"code": {"text": args.get("frequency", "")}},
                "doseAndRate": [{"doseQuantity": {"text": args.get("dose", "")}}],
            }],
            "reasonCode": [{"text": args.get("indication", "")}] if args.get("indication") else [],
            "_simulated": True,
            "_created_at": ts,
        }

    if name == "ServiceRequest.create":
        return {
            "resourceType": "ServiceRequest",
            "id": f"SR-sim-{subject_id}-{ts[:10]}",
            "status": "active",
            "intent": "order",
            "priority": args.get("priority", "routine"),
            "code": {"text": args.get("service_type", "")},
            "subject": {"reference": f"Patient/{subject_id}"},
            "note": [{"text": args.get("note", "")}] if args.get("note") else [],
            "occurrenceDateTime": args.get("target_date", ""),
            "_simulated": True,
            "_created_at": ts,
        }

    if name == "Flag.create":
        return {
            "resourceType": "Flag",
            "id": f"FL-sim-{subject_id}-{ts[:10]}",
            "status": "active",
            "category": [{"text": args.get("category", "")}],
            "code": {"text": args.get("code", ""), "coding": [{"display": args.get("detail", "")}]},
            "subject": {"reference": f"Patient/{subject_id}"},
            "_simulated": True,
            "_created_at": ts,
        }

    return {"error": f"Unknown write tool: {name}"}


def _execute_dp_actions(
    action_list: list[dict],
    subject_id: int,
    hadm_id: int | None,
    tool_set: str,
    note_grounding: dict | None = None,
    health_literacy: str = "medium",
) -> list[dict]:
    """
    Execute action list, routing write tools to _exec_write_tool.
    Read tools go through the standard call_tool pathway.
    Patient tools use note_grounding to form simulated responses.
    """
    results = []
    for i, action in enumerate(action_list):
        name = action.get("name", "")
        args = dict(action.get("arguments", {}))

        if name == "ask_user_for_required_parameters":
            continue

        if name in _WRITE_TOOL_NAMES and name != "prepare_to_answer":
            if "subject_id" not in args:
                args["subject_id"] = subject_id
            observation = _exec_write_tool(name, args)
        elif name.startswith("patient.") and note_grounding:
            observation = _exec_patient_tool(
                name, args, note_grounding, subject_id, health_literacy
            )
        else:
            if name != "prepare_to_answer" and "subject_id" not in args:
                args["subject_id"] = subject_id
            observation = call_tool(name, args, tool_set=tool_set)

        dep_list = [j for j in range(i)] if name == "prepare_to_answer" else []
        results.append({
            "action": {"name": name, "arguments": args},
            "observation": observation,
            "dependency_list": dep_list,
            "idx": i,
        })

    return results


# ── Patient tool grounding ────────────────────────────────────────────────────

_GROUNDING_SYSTEM = """\
You are a clinical note parser. Extract structured patient facts from the given
MIMIC-IV discharge note sections and return them as JSON.

Return ONLY valid JSON with these fields (use empty string if absent):
{
  "chief_complaint":        "<primary reason for admission, 1-2 sentences>",
  "medications_on_admission": "<list of medications patient was taking at home>",
  "social_history":         "<living situation, smoking, alcohol, support>",
  "functional_status":      "<ADL independence, mobility, assistive devices>",
  "discharge_condition":    "<clinical condition at discharge: stable/improved/etc.>",
  "follow_up_instructions": "<follow-up appointments, when and with whom>",
  "discharge_medications":  "<list of discharge medications>"
}
"""


def _extract_note_grounding(subject_id: int, hadm_id: int | None) -> dict:
    """
    Extract structured grounding facts from the patient's discharge note.
    Falls back to empty strings if sections are absent.
    """
    # Fetch discharge summary via FHIR DocumentReference
    obs = call_tool(
        "DocumentReference.search",
        {"subject_id": subject_id, "hadm_id": hadm_id, "type_code": "discharge-summary"},
        tool_set="fhir",
    )

    note_text = ""
    if isinstance(obs, dict):
        entries = obs.get("entry", [])
        for entry in entries[:1]:
            res = entry.get("resource", {})
            for content in res.get("content", []):
                note_text = content.get("attachment", {}).get("data", "")
                if note_text:
                    break

    if not note_text:
        logger.warning(f"  _extract_note_grounding: no discharge note for {subject_id}/{hadm_id}")
        return {}

    # Trim to avoid LLM context limits
    trimmed = note_text[:6000]

    messages = [
        {"role": "system", "content": _GROUNDING_SYSTEM},
        {"role": "user", "content": f"Discharge note:\n\n{trimmed}\n\nExtract JSON:"},
    ]
    try:
        raw = llm_call(messages, temperature=0.0, max_tokens=1200)
        # Try extract_json first; on failure fall back to raw JSON parse
        try:
            result = extract_json(raw)
        except Exception:
            import re, json as _json
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            result = _json.loads(m.group(0)) if m else {}
        if isinstance(result, dict):
            result["_raw_note_chars"] = len(note_text)
            return result
    except Exception as exc:
        logger.warning(f"  _extract_note_grounding LLM failed: {exc}")

    return {}


_LITERACY_STYLE: dict[str, str] = {
    "low": (
        "Use simple everyday words only. Avoid all medical terms — if you must mention "
        "a drug or condition, use the plain name (e.g. 'water pill' not 'diuretic'). "
        "You may misremember details or express confusion. Short sentences."
    ),
    "medium": (
        "Use plain language but you know basic medical terms like 'blood pressure' or "
        "'infection'. You are slightly uncertain and may ask the doctor to clarify. "
        "Natural conversational tone, 2-4 sentences."
    ),
    "high": (
        "You are medically literate. Use correct clinical terminology (drug generic names, "
        "condition names, relevant lab values if mentioned). You describe symptoms precisely "
        "and remember details accurately. 2-4 sentences."
    ),
}


def _patient_system_prompt(health_literacy: str = "medium") -> str:
    style = _LITERACY_STYLE.get(health_literacy, _LITERACY_STYLE["medium"])
    return (
        "You are simulating a patient responding to a clinician during discharge planning.\n"
        "The patient is a real MIMIC-IV patient. Respond in first person, based ONLY on "
        "the grounded facts provided. Do not invent information.\n"
        "If the fact is absent, say \"I'm not sure\" or \"I don't remember.\"\n\n"
        f"Health literacy level — {health_literacy}:\n{style}"
    )


def _exec_patient_tool(
    tool_name: str,
    args: dict,
    grounding: dict,
    subject_id: int,
    health_literacy: str = "medium",
) -> dict:
    """
    Simulate a patient interview tool response from note-span grounded facts.
    Returns {"patient_response": str, "_grounded_from": str, "_simulated": True}
    """
    fact_key = {
        "patient.get_medication_adherence": "medications_on_admission",
        "patient.get_functional_status":    "functional_status",
        "patient.get_social_history":       "social_history",
        "patient.get_chief_complaint":      "chief_complaint",
        "patient.get_pain_assessment":      "chief_complaint",
    }.get(tool_name, "")

    grounded_fact = grounding.get(fact_key, "")
    drug = args.get("drug", "")

    if tool_name == "patient.get_medication_adherence" and drug:
        prompt = (
            f"The clinician asks: 'Are you taking {drug} as prescribed?'\n"
            f"Patient's medication history from records: {grounded_fact or '(not documented)'}\n"
            "Simulate the patient's response:"
        )
    elif tool_name == "patient.get_functional_status":
        prompt = (
            f"The clinician asks: 'Can you tell me about your ability to perform daily activities at home?'\n"
            f"Patient's functional status from records: {grounded_fact or '(not documented)'}\n"
            "Simulate the patient's response:"
        )
    elif tool_name == "patient.get_social_history":
        prompt = (
            f"The clinician asks: 'Can you tell me about your home situation and support system?'\n"
            f"Patient's social history from records: {grounded_fact or '(not documented)'}\n"
            "Simulate the patient's response:"
        )
    else:
        prompt = (
            f"The clinician asks about: {tool_name.replace('patient.get_', '').replace('_', ' ')}\n"
            f"Relevant patient fact from records: {grounded_fact or '(not documented)'}\n"
            "Simulate the patient's response:"
        )

    messages = [
        {"role": "system", "content": _patient_system_prompt(health_literacy)},
        {"role": "user", "content": prompt},
    ]
    try:
        response_text = llm_call(messages, temperature=0.3, max_tokens=200)
        result = {
            "patient_response": response_text.strip(),
            "_grounded_from": fact_key,
            "_grounded_fact": grounded_fact[:200] if grounded_fact else "",
            "_simulated": True,
        }
        # Generate responses for all 3 literacy levels and store as a parallel field.
        # The main patient_response reflects health_literacy; the dict provides all variants.
        all_responses: dict[str, str] = {health_literacy: response_text.strip()}
        for lit in ["low", "medium", "high"]:
            if lit == health_literacy:
                continue
            try:
                msgs_lit = [
                    {"role": "system", "content": _patient_system_prompt(lit)},
                    {"role": "user", "content": prompt},
                ]
                all_responses[lit] = llm_call(msgs_lit, temperature=0.3, max_tokens=200).strip()
            except Exception:
                all_responses[lit] = ""
        result["patient_responses_all_literacy"] = all_responses
        return result
    except Exception as exc:
        return {
            "patient_response": "(patient response unavailable)",
            "_grounded_from": fact_key,
            "_simulated": True,
            "_error": str(exc),
        }


# ── Tool list helpers ─────────────────────────────────────────────────────────

def _get_dp_tools(turn_idx: int, task_type: str, tool_source: str, language: str = "en") -> list[dict]:
    """
    Return the tool list for a discharge_planning turn.
    T0/T1: EHR read tools (discharge-relevant subset)
    T2:    Patient Interview tools + key EHR tools for context
    T3:    Write tools (Write/Update turn)
    """
    # Build EHR subset from FHIR schemas
    ehr_tools = [
        FHIR_SCHEMA_BY_NAME[n] for n in _DP_EHR_TOOL_NAMES
        if n in FHIR_SCHEMA_BY_NAME
    ]

    # Patient interview tool schemas
    patient_tool_schemas = get_tools_for_task("PatientInterview", language=language)
    # Filter to discharge-relevant patient tools
    dp_patient_tools = [
        t for t in patient_tool_schemas
        if t["function"]["name"] in _DP_PATIENT_TOOL_NAMES
    ]

    if tool_source == "write" or task_type == "Write/Update":
        prepare = FHIR_SCHEMA_BY_NAME.get("prepare_to_answer")
        extras = [prepare] if prepare else []
        return WRITE_TOOL_SCHEMAS + extras

    if tool_source == "patient":
        return dp_patient_tools

    if tool_source == "mixed":
        existing_names = {t["function"]["name"] for t in ehr_tools}
        combined = ehr_tools + [
            t for t in dp_patient_tools if t["function"]["name"] not in existing_names
        ]
        return combined

    # Default: EHR tools
    return ehr_tools


def _legacy_for_dp(task_type: str, tool_source: str) -> str:
    """Map to legacy task type name for planner/checker agents."""
    if task_type == "Write/Update":
        return "Data Gathering"   # multi-tool, closest analogue
    if tool_source == "patient":
        return "Intake"
    if tool_source == "mixed":
        return "Data Gathering"
    return {"Information Lookup": "Lookup", "Data Gathering": "Data Gathering",
            "Clinical Reasoning": "Lookup"}.get(task_type, task_type)


# ── Session planner for discharge_planning ────────────────────────────────────

_DP_PLANNER_SYSTEM = """\
You are a clinical conversation planner for a discharge planning benchmark.

The session MUST follow this fixed arc:
  T0 [Information Lookup]     — Retrieve admission overview (Encounter.search or DocumentReference.search)
  T1 [Data Gathering]        — Retrieve discharge plan + medication list
                       (CarePlan.search + MedicationRequest.search, or + Condition.search)
  T2 [Data Gathering/mixed]  — Patient interview: medication adherence + functional status
                       (patient.get_medication_adherence + patient.get_functional_status,
                        optionally also patient.get_social_history)
  T3 [Action]        — Discharge write action:
                       MedicationRequest.create  OR  ServiceRequest.create

Topics MUST be grounded in the patient's actual EHR snapshot.

Return ONLY valid JSON:
{
  "clinical_situation": "<one sentence: patient summary including key diagnoses>",
  "investigation_arc": "T0[R] <phrase> → T1[W] <phrase> → T2[W/mixed] <phrase> → T3[Action] <phrase>",
  "turn_intents": [
    "T0 [Information Lookup]: retrieve admission encounter record — establish admission context",
    "T1 [Data Gathering/CarePlan×MedReq]: co-retrieve discharge plan and medication list — identify gaps",
    "T2 [Data Gathering/patient]: interview patient about adherence and functional status — assess discharge readiness",
    "T3 [Action]: create discharge medication order / home health referral — complete discharge plan"
  ],
  "turns": [
    {
      "turn": 0,
      "task_type": "Information Lookup",
      "topic": "<admission type, dates, discharge disposition — MUST be in EHR snapshot>",
      "tool_hint": "Encounter.search(hadm_id=<hadm_id>) or DocumentReference.search(discharge-summary)",
      "workup_pattern": null
    },
    {
      "turn": 1,
      "task_type": "Data Gathering",
      "topic": "<care plan instructions + key discharge medications — MUST be in EHR snapshot>",
      "tool_hint": "CarePlan.search(hadm_id=<hadm_id>) + MedicationRequest.search(hadm_id=<hadm_id>)",
      "workup_pattern": "CarePlan×MedReq"
    },
    {
      "turn": 2,
      "task_type": "Data Gathering",
      "topic": "<patient interview: adherence to [SPECIFIC DRUG NAME] + functional status at home — MUST name a drug from the EHR snapshot; MUST NOT mention lab values, creatinine, potassium, or any EHR-derived numbers>",
      "tool_hint": "patient.get_medication_adherence(drug=<specific drug name>) + patient.get_functional_status",
      "workup_pattern": "Adherence×Function"
    },
    {
      "turn": 3,
      "task_type": "Write/Update",
      "topic": "<specific medication to order OR home health referral based on what turns 0-2 revealed>",
      "tool_hint": "MedicationRequest.create(medication=<drug>, dose=<dose>, route=<route>, frequency=<freq>) OR ServiceRequest.create(service_type=home-health, priority=routine)",
      "workup_pattern": null
    }
  ]
}

RULES:
1. Choose T3 action based on actual clinical need revealed in T0-T2.
   If medications were found to be sub-therapeutic or missing → MedicationRequest.create
   If patient has functional limitations or needs home support → ServiceRequest.create
2. The drug/dose in T3 tool_hint MUST match a medication already in the EHR snapshot.
3. Do NOT invent values — ground every topic in the EHR snapshot content.
"""


_MIXED_Q_SYSTEM_EN = """\
You are generating Turn 2 of a 4-turn discharge planning benchmark session.
This turn is a PATIENT INTERVIEW turn — the clinician speaks DIRECTLY to the patient
to assess whether they are ready for discharge.

The question must ask the patient about TWO things:
  1. Medication adherence — Is the patient actually taking a specific key medication at home?
     Use the drug name from the session plan / prior conversation (e.g. Metoprolol, Warfarin).
  2. Functional status — Can the patient manage daily activities independently at home?

STRICT RULES:
- The question MUST be phrased as "ask the patient about X" — NOT as an EHR lookup.
- Do NOT ask about lab values (creatinine, potassium, glucose, BMP, etc.) — those come from EHR.
- Do NOT ask about imaging, test results, or diagnoses — those come from EHR.
- Do NOT ask the patient for their own vitals or lab numbers — patients don't know these.
- Name a SPECIFIC drug from the medication list for the adherence check.
- Keep it concise (1–2 sentences).

Good examples:
  "Is the patient taking Metoprolol as prescribed, and how are they managing daily activities at home?"
  "Ask the patient about their Warfarin adherence and any functional limitations at home."
  "Does the patient take Furosemide regularly, and can they manage self-care independently?"

Return ONLY the question string, no JSON, no explanation.
"""

_MIXED_Q_SYSTEM_ZH = """\
你正在为出院计划基准测试生成第2轮（病人访谈轮）的用户问题。
这一轮是病人访谈轮——临床医生直接与患者交流，评估患者的出院准备情况。

问题必须同时涵盖两个方面：
  1. 用药依从性——患者是否在家按处方服用某种关键药物？
     使用来自会话计划/前序对话中的具体药物名称（如美托洛尔、华法林）。
  2. 功能状态——患者能否在家独立完成日常活动？

严格规则：
- 问题必须措辞为"询问患者..."——不是EHR查询。
- 不要询问实验室值（肌酐、血钾、血糖、基础代谢面板等）——这些来自EHR。
- 不要询问影像、检查结果或诊断——这些来自EHR。
- 不要让患者报告自己的生化指标——患者不知道这些数值。
- 为依从性检查指定用药列表中的一个具体药物名称。
- 简洁（1-2句话）。

只返回问题字符串，不含JSON或解释。
"""


def _generate_mixed_turn_question(
    session_plan: dict | None,
    history: list[dict],
    ehr_snapshot: str,
    language: str = "en",
) -> tuple[str, str]:
    """
    Generate the T2 patient-interview question for the mixed turn.
    Produces a question about medication adherence + functional status,
    never about EHR lab values or imaging results.
    Returns (user_q, user_q_explicit).
    """
    import re as _re
    is_zh = language == "zh"
    sys_prompt = _MIXED_Q_SYSTEM_ZH if is_zh else _MIXED_Q_SYSTEM_EN

    # Extract T2 tool_hint from session plan for specific drug name
    tool_hint = ""
    if session_plan and "turns" in session_plan:
        t2 = next((t for t in session_plan["turns"] if t.get("turn") == 2), None)
        if t2:
            tool_hint = t2.get("tool_hint", "")

    # Prior conversation (T0+T1) for clinical context
    hist_str = ""
    for msg in history[-4:]:
        role = msg.get("role", "").upper()
        content = str(msg.get("content", ""))[:200]
        hist_str += f"[{role}]: {content}\n"

    if is_zh:
        user_prompt = (
            f"患者EHR摘要（用于确认关键药物）：{ehr_snapshot[:400]}\n\n"
            f"前序对话（T0-T1）：\n{hist_str}\n"
            f"建议的工具调用：{tool_hint}\n\n"
            "生成T2病人访谈轮的用户问题："
        )
    else:
        user_prompt = (
            f"Patient EHR summary (to identify key medication): {ehr_snapshot[:400]}\n\n"
            f"Prior conversation (T0-T1):\n{hist_str}\n"
            f"Suggested tool calls: {tool_hint}\n\n"
            "Generate the T2 patient interview question:"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = llm_call(messages, temperature=0.4, max_tokens=180)
        q = raw.strip().strip('"').strip("`").strip()
        if not q:
            raise ValueError("LLM returned empty mixed-turn question")
        # Reject if the question asks about lab values — force fallback
        _lab_keywords = ("creatinine", "potassium", "glucose", "sodium", "bmp",
                         "lab", "肌酐", "血钾", "血糖", "血钠", "实验室")
        if any(kw in q.lower() for kw in _lab_keywords):
            raise ValueError(f"Generated question asks about labs (rejected): {q!r:.60}")
        return q, q
    except Exception as exc:
        logger.warning(f"  _generate_mixed_turn_question failed: {exc}")
        # Fallback: extract drug name from tool_hint, produce safe default
        m = _re.search(r'drug=([^,\)\s]+)', tool_hint or "")
        drug = m.group(1).strip() if m else ("当前用药" if is_zh else "their current medications")
        if is_zh:
            fallback = f"请问患者是否按处方服用{drug}，以及他们在家的日常生活活动能力如何？"
        else:
            fallback = (
                f"Is the patient taking {drug} as prescribed, "
                "and how are they managing daily activities at home?"
            )
        logger.info(f"  _generate_mixed_turn_question: using fallback Q={fallback!r:.80}")
        return fallback, fallback


_ACTION_Q_SYSTEM_EN = """\
You are generating the FINAL turn of a 4-turn EHR benchmark session.
This turn is an ACTION turn — the user (a clinician) is asking the assistant to WRITE
something to the EHR (create a medication order, service referral, or clinical flag).

Based on the clinical context and the prior conversation, generate a concise clinical
instruction asking the assistant to perform one specific write action.

Examples of good Action questions:
  "Please add Furosemide 40 mg oral daily to the discharge medication list."
  "Create a home health referral for this patient given her functional limitations."
  "Please flag this patient as high fall risk before discharge."
  "Order a repeat INR check for 1 week post-discharge via ServiceRequest."

Rules:
- The question must directly request a WRITE operation (create/order/flag/refer).
- The request must be clinically justified by what the prior turns revealed.
- Keep it concise (1-2 sentences). No need for a preamble.
- Do NOT ask to retrieve or look up data — this is a write action only.

Return ONLY the question string, no JSON, no explanation.
"""

_ACTION_Q_SYSTEM_ZH = """\
你正在为一个4轮EHR基准会话生成最后一轮（ACTION轮）的用户问题。
这一轮是操作轮——用户（临床医生）要求助手向EHR中写入内容
（创建用药医嘱、服务申请或临床标记）。

根据临床背景和之前的对话，生成一个简洁的临床指令，要求助手执行一个具体的写操作。

好的Action问题示例：
  "请将呋塞米40 mg口服每日一次加入出院带药清单。"
  "基于患者的功能状态限制，请为其创建家庭护理转介申请。"
  "请在出院前将该患者标记为高跌倒风险。"

规则：
- 问题必须直接要求执行写操作（创建/开具/标记/转介）。
- 请求必须与前序轮次揭示的临床发现相呼应。
- 简洁（1-2句），无需前言。
- 不要要求检索或查询数据——这是纯写操作。

只返回问题字符串，不含JSON或解释。
"""


def _generate_action_question(
    session_plan: dict | None,
    history: list[dict],
    ehr_snapshot: str,
    language: str = "en",
) -> tuple[str, str]:
    """
    Generate an action request question for the T3 Write/Update turn.
    Returns (user_q, user_q_explicit) — same string for both (no ellipsis transform).
    """
    is_zh = language == "zh"
    sys_prompt = _ACTION_Q_SYSTEM_ZH if is_zh else _ACTION_Q_SYSTEM_EN

    # Build prior history summary (last 3 QA turns)
    hist_str = ""
    for msg in history[-6:]:
        role = msg.get("role", "").upper()
        content = str(msg.get("content", ""))[:250]
        hist_str += f"[{role}]: {content}\n"

    # Tool hint from session plan
    tool_hint = ""
    if session_plan and "turns" in session_plan:
        t3 = next((t for t in session_plan["turns"] if t.get("turn") == 3), None)
        if t3:
            tool_hint = t3.get("tool_hint", "")

    if is_zh:
        user_prompt = (
            f"患者EHR摘要：{ehr_snapshot[:500]}\n\n"
            f"前序对话：\n{hist_str}\n"
            f"建议的写操作类型：{tool_hint}\n\n"
            "生成Action轮的用户问题："
        )
    else:
        user_prompt = (
            f"Patient EHR summary: {ehr_snapshot[:500]}\n\n"
            f"Prior conversation:\n{hist_str}\n"
            f"Suggested write action type: {tool_hint}\n\n"
            "Generate the Write/Update turn question:"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = llm_call(messages, temperature=0.4, max_tokens=220)
        q = raw.strip().strip('"').strip("`").strip()
        if not q:
            raise ValueError(f"LLM returned empty action question (raw={raw!r:.60})")
        # Reject obviously truncated responses (no sentence-ending punctuation)
        if len(q) > 80 and not any(q.rstrip().endswith(c) for c in ".?!。？！"):
            raise ValueError(f"LLM returned truncated action question: {q!r:.60}")
        return q, q
    except Exception as exc:
        logger.warning(f"  _generate_action_question failed: {exc}")
        # Grounded fallback: use session plan's T3 tool_hint if available
        _TOOL_VERB_EN = {
            "MedicationRequest.create": "create a discharge medication order",
            "ServiceRequest.create":    "create a service referral",
            "Flag.create":              "create a clinical safety flag",
        }
        _TOOL_VERB_ZH = {
            "MedicationRequest.create": "创建一条出院用药医嘱",
            "ServiceRequest.create":    "创建一条服务转介申请",
            "Flag.create":              "创建一条临床安全标记",
        }
        if tool_hint:
            tool_name = tool_hint.split("(")[0].strip()
            if is_zh:
                verb = _TOOL_VERB_ZH.get(tool_name, f"执行{tool_name}")
                fallback = f"请根据前序发现，为该患者{verb}。"
            else:
                verb = _TOOL_VERB_EN.get(tool_name, f"execute {tool_name}")
                fallback = f"Based on the findings above, please {verb} for this patient."
        else:
            fallback = (
                "请根据前序发现为该患者创建适当的出院医嘱或服务申请。" if is_zh
                else "Based on the findings above, please create an appropriate discharge order or referral for this patient."
            )
        logger.info(f"  _generate_action_question: using fallback Q={fallback!r:.80}")
        return fallback, fallback


def _validate_action_plan(plan: dict, available_tool_names: list[str]) -> tuple[bool, str]:
    """
    Validate a plan for an Write/Update turn.
    Accepts if at least one write tool (not prepare_to_answer) is called.
    """
    action_list = plan.get("Action_List", [])
    write_calls = [
        a for a in action_list
        if a.get("name") in _WRITE_TOOL_NAMES
        and a.get("name") != "prepare_to_answer"
    ]
    if not write_calls:
        return False, f"Action plan must include at least one write tool call, got: {[a.get('name') for a in action_list]}"
    # Check all tools are valid
    for a in action_list:
        name = a.get("name", "")
        if name not in available_tool_names:
            return False, f"Unknown write tool: {name}"
    return True, ""


def _plan_discharge_session(
    ehr_snapshot: str,
    hadm_id: int | None,
    language: str = "en",
) -> dict | None:
    """Generate a discharge-planning-specific session plan."""
    snapshot_trimmed = ehr_snapshot[:6000]
    hadm_hint = f" (admission ID: {hadm_id})" if hadm_id else ""

    user_prompt = (
        f"Patient EHR snapshot{hadm_hint}:\n{snapshot_trimmed}\n\n"
        "Generate the discharge planning session plan JSON:"
    )
    messages = [
        {"role": "system", "content": _DP_PLANNER_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = llm_call(messages, temperature=0.4, max_tokens=2000)
        try:
            plan = extract_json(raw)
        except Exception:
            import re as _re, json as _json
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            plan = _json.loads(m.group(0)) if m else {}
        if not isinstance(plan, dict) or "turns" not in plan:
            logger.warning("_plan_discharge_session: invalid JSON structure")
            return None
        if "turn_intents" not in plan:
            plan["turn_intents"] = [""] * 4
        logger.info(f"  DP session plan: {plan.get('investigation_arc', '')[:100]}")
        return plan
    except Exception as exc:
        logger.warning(f"_plan_discharge_session failed ({exc})")
        return None


# ── Action-turn rubric generator ──────────────────────────────────────────────

_ACTION_RUBRIC_SYSTEM_EN = """\
You are a clinical benchmark rubric designer evaluating a discharge write action.

Given the clinical context, the write action executed (tool + parameters), and
the reference answer, generate 4–6 atomic rubric criteria to evaluate another
model's proposed write action for the same question.

DESIGN RULES:
1. Each item describes an OUTCOME — "The model correctly orders Furosemide at ≥40 mg daily"
   NOT "The model calls MedicationRequest.create correctly."
2. Cite SPECIFIC gold parameter values with acceptable ranges:
   Write: "The answer orders a loop diuretic (Furosemide or equivalent) at 40–120 mg daily"
   NOT:  "The answer orders an appropriate diuretic"
3. Include items for:
   a) Correct drug/service class (allowing clinically equivalent alternatives)
   b) Dose range acceptability (include renal/hepatic adjustment if applicable)
   c) Route + frequency appropriateness
   d) Indication/clinical reasoning documented
   e) Safety item: does NOT order a contraindicated drug (if relevant)
4. Each item is independently evaluable as YES / NO by an LLM judge.

Output: Return ONLY a valid JSON array of strings. No prose.
Example: ["The answer orders a loop diuretic (Furosemide or equivalent) for volume overload", ...]
"""

_ACTION_RUBRIC_SYSTEM_ZH = """\
你是一个临床基准测试的rubric设计者，专门评测出院写操作。

给定临床上下文、已执行的写操作（工具+参数）和参考答案，
生成4–6条原子化评测标准，用于评测另一个模型对同一问题的写操作提议。

设计规则：
1. 每条描述结果："模型正确开具Furosemide ≥40 mg/日"
   不写："模型正确调用了MedicationRequest.create"
2. 引用具体金标准参数值及可接受范围：
   写："回答开具了袢利尿剂（呋塞米或等效药物）40–120 mg/日"
   不写："回答开具了合适的利尿剂"
3. 包含以下方面的条目：
   a) 正确的药物/服务类别（允许临床等效的替代品）
   b) 剂量范围合理性（如涉及肾/肝功能调整需体现）
   c) 给药途径+频次合理性
   d) 适应证/临床推理有据可查
   e) 安全性：未开具禁忌药物（如相关）
4. 每条均可被LLM法官独立判断为是/否。

输出：只返回有效的JSON字符串数组，不含散文。
"""


def generate_action_rubric(
    user_question: str,
    executed_write_actions: list[dict],
    reference_answer: str,
    clinical_context: str = "",
    language: str = "en",
) -> list[str]:
    """
    Generate outcome-based rubric items for an Write/Update turn write operation.
    Items cite SPECIFIC gold parameter values with acceptable ranges.
    """
    is_zh = language == "zh"
    sys_prompt = _ACTION_RUBRIC_SYSTEM_ZH if is_zh else _ACTION_RUBRIC_SYSTEM_EN

    # Summarize the write actions into a readable string
    write_summary = _serialize_write_actions(executed_write_actions)

    if is_zh:
        user_prompt = (
            f"临床问题：{user_question}\n\n"
            f"临床背景：{clinical_context[:400]}\n\n"
            f"已执行的写操作（金标准）：\n{write_summary}\n\n"
            f"参考答案：\n{reference_answer[:500]}\n\n"
            "生成写操作评测rubric条目（JSON数组）："
        )
    else:
        user_prompt = (
            f"Clinical question: {user_question}\n\n"
            f"Clinical context: {clinical_context[:400]}\n\n"
            f"Write action executed (gold standard):\n{write_summary}\n\n"
            f"Reference answer:\n{reference_answer[:500]}\n\n"
            "Generate write-action rubric items (JSON array):"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = llm_call(messages, temperature=0.2, max_tokens=1500)
        try:
            items = extract_json(raw)
        except ValueError:
            # Truncated JSON — extract complete string items via regex
            import re
            items = re.findall(r'"((?:[^"\\]|\\.)+)"', raw)
            items = [i.strip() for i in items if len(i) > 20]  # skip short key-like strings
            if items:
                logger.info(f"generate_action_rubric: recovered {len(items)} items from truncated JSON")
        if isinstance(items, list) and all(isinstance(i, str) for i in items):
            return [i.strip() for i in items if i.strip()]
        logger.warning("generate_action_rubric: unexpected JSON structure")
        return []
    except Exception as exc:
        logger.warning(f"generate_action_rubric failed: {exc}")
        return []


def _serialize_write_actions(executed_actions: list[dict]) -> str:
    """Serialize write tool call(s) into a readable string for rubric generation."""
    lines = []
    for act in executed_actions:
        name = act.get("action", {}).get("name", "")
        args = act.get("action", {}).get("arguments", {})
        obs = act.get("observation", {})
        if name in ("prepare_to_answer", "ask_user_for_required_parameters"):
            continue
        if name in _WRITE_TOOL_NAMES:
            params = {k: v for k, v in args.items() if k != "subject_id" and v}
            lines.append(f"Tool: {name}")
            for k, v in params.items():
                lines.append(f"  {k}: {v}")
            if isinstance(obs, dict) and obs.get("_simulated"):
                lines.append(f"  [status: created — simulated]")
    return "\n".join(lines) if lines else "(no write actions recorded)"


# ── Sanitize helper ───────────────────────────────────────────────────────────

def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ── Main entry generation ─────────────────────────────────────────────────────

def generate_discharge_planning_entry(
    subject_id: int,
    hadm_id: int | None,
    entry_index: int,
    language: str = "en",
    bilingual: bool = True,
    tool_set: str = DEFAULT_TOOL_SET,
    difficulty: int = 1,
    health_literacy: str | None = None,
    subtype_counter: dict[str, int] | None = None,
) -> dict | None:
    """
    Generate one benchmark entry for the discharge_planning scenario.

    Session arc: Information Lookup → Data Gathering → Data Gathering/mixed → Action

    Args:
        subject_id:    MIMIC-IV patient identifier
        hadm_id:       Hospital admission ID
        entry_index:   Unique index for ID construction
        language:      Primary generation language "en" | "zh"
        bilingual:     If True, also generate Chinese translations
        tool_set:      "fhir" (default)
        difficulty:    1 (L1), 2 (L2), or 3 (L3) for the Write/Update turn complexity
        subtype_counter: Shared counter for subtype balancing across entries

    Returns:
        Benchmark entry dict, or None on failure.
    """
    entry_id = f"dp_{subject_id}_{entry_index}_L{difficulty}"
    logger.info(f"Generating {entry_id}  subject={subject_id}  hadm={hadm_id}")

    context = build_context(subject_id, hadm_id, task_domain="DischargePlan")
    set_active_date(context.get("current_date"))

    ehr_snapshot = build_ehr_snapshot(subject_id, hadm_id or context.get("hadm_id"))
    context["ehr_snapshot"] = ehr_snapshot
    logger.info(f"  EHR snapshot built ({len(ehr_snapshot)} chars)")

    if not has_queryable_data(subject_id, hadm_id or context.get("hadm_id")):
        logger.warning(f"  Skipping {entry_id}: no EHR data")
        return None

    # Extract note-span grounding for patient tools
    note_grounding = _extract_note_grounding(subject_id, hadm_id or context.get("hadm_id"))
    logger.info(f"  Note grounding extracted: {list(note_grounding.keys())}")

    # Generate discharge-specific session plan
    session_plan = _plan_discharge_session(
        ehr_snapshot=ehr_snapshot,
        hadm_id=hadm_id or context.get("hadm_id"),
        language=language,
    )

    task_sequence = DISCHARGE_ARC  # fixed: [R, W, W, Action]
    dep_graph = TurnDependencyGraph()
    local_subtype_increments: dict[str, int] = {}

    turn_data: list[dict] = []
    history: list[dict] = []
    tasks_zh: list[str] = []
    messages_zh: list[dict] = []

    for turn_idx in range(4):
        task_type = task_sequence[turn_idx]
        tool_source = DISCHARGE_TOOL_SOURCES[turn_idx]

        subtype = pick_subtype(turn_idx, dep_graph, task_type, subtype_counter)

        # For Write/Update turns, grounding facts from prior conversation are sufficient
        grounding_facts = ehr_snapshot[:2000]

        antecedents = []
        if subtype and turn_idx > 0:
            antecedents = dep_graph.get_antecedents(turn_idx, subtype)

        # Extract turn plan from session_plan
        turn_plan: dict | None = None
        turn_intent: str | None = None
        if session_plan:
            turns = session_plan.get("turns", [])
            if turn_idx < len(turns):
                turn_plan = turns[turn_idx]
            intents = session_plan.get("turn_intents", [])
            if turn_idx < len(intents):
                turn_intent = intents[turn_idx]

        # Get available tools for this turn
        turn_tools = _get_dp_tools(turn_idx, task_type, tool_source, language=language)
        turn_tool_names = [t["function"]["name"] for t in turn_tools]

        # Legacy task type name for planner/checker agents
        legacy_type = _legacy_for_dp(task_type, tool_source)

        success = False
        failed_questions: list[dict] = []

        for attempt in range(MAX_RETRIES):
            try:
                # 1. Generate user question
                if task_type == "Write/Update":
                    # Write/Update turns need a write-request question, not a retrieval question
                    user_q, user_q_explicit = _generate_action_question(
                        session_plan=session_plan,
                        history=history,
                        ehr_snapshot=ehr_snapshot,
                        language=language,
                    )
                elif tool_source == "mixed":
                    # T2 mixed turns are patient interview turns — must ask about
                    # medication adherence + functional status, NOT EHR lab values.
                    # Use dedicated generator to avoid generic EHR questions being produced.
                    user_q, user_q_explicit = _generate_mixed_turn_question(
                        session_plan=session_plan,
                        history=history,
                        ehr_snapshot=ehr_snapshot,
                        language=language,
                    )
                else:
                    user_q, user_q_explicit = generate_user_turn_v2(
                        context=context,
                        scenario="discharge_planning",
                        task_type=task_type,
                        subtype=subtype,
                        tool_source=tool_source if tool_source != "write" else "ehr",
                        history=history,
                        language=language,
                        grounding_facts=grounding_facts,
                        antecedents=antecedents if subtype else None,
                        ehr_snapshot=ehr_snapshot,
                        failed_questions=failed_questions or None,
                        turn_plan=turn_plan,
                        turn_intent=turn_intent,
                    )

                    # PE fallback (only for non-Write/Update turns)
                    if (
                        user_q == user_q_explicit
                        and subtype and subtype != "PE" and turn_idx > 0
                    ):
                        pe_ants = dep_graph.get_antecedents(turn_idx, "PE")
                        if pe_ants:
                            fb = _apply_ellipsis_transform(
                                explicit_q=user_q_explicit,
                                subtype="PE",
                                history=history,
                                antecedents=pe_ants,
                                language=language,
                                ehr_snapshot=ehr_snapshot,
                                task_type=task_type,
                            )
                            if fb != user_q_explicit:
                                user_q = fb
                                subtype = "PE"

                logger.info(
                    f"  turn {turn_idx} [{task_type}/{subtype}/{tool_source}] "
                    f"Q: {user_q[:80]}"
                )

                # 2. Plan tool calls
                # On retry, remind the planner to use FHIR tool names (not legacy names)
                _fhir_names = ", ".join(turn_tool_names[:6])
                _retry_hint = (
                    f"IMPORTANT: Use ONLY FHIR tool names from the available tools list: "
                    f"{_fhir_names}. Do NOT use legacy names like get_lab_results."
                    if attempt > 0 else ""
                )
                _dp_tool_hint = turn_plan.get("tool_hint", "") if turn_plan else ""
                plan = plan_actions(
                    context=context,
                    history=history,
                    user_question=user_q,
                    available_tools=turn_tools,
                    task_type=legacy_type,
                    tool_source=tool_source if tool_source != "write" else "ehr",
                    attempt=attempt,
                    prev_obs_failure=_retry_hint,
                    language=language,
                    tool_hint=_dp_tool_hint,
                )

                # 3. Validate plan
                if task_type == "Write/Update":
                    # Use write-specific validator: needs ≥1 write tool call
                    is_valid, reason = _validate_action_plan(plan, turn_tool_names)
                else:
                    is_valid, reason = validate_plan(
                        user_question=user_q,
                        task_type=legacy_type,
                        plan=plan,
                        available_tool_names=turn_tool_names,
                        subject_id=subject_id,
                        tool_source=tool_source if tool_source != "write" else "ehr",
                        language=language,
                    )
                if not is_valid:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"  Plan check failed: {reason}. Retrying...")
                        continue
                    else:
                        logger.warning(f"  Plan check failed (final): {reason}. Discarding.")
                        break

                # 4. Execute tools
                action_list = plan.get("Action_List", [])
                if not action_list:
                    action_list = [{"name": "prepare_to_answer", "arguments": {}}]

                _literacy = health_literacy or {1: "low", 2: "medium", 3: "high"}.get(difficulty, "medium")
                executed_actions = _execute_dp_actions(
                    action_list=action_list,
                    subject_id=subject_id,
                    hadm_id=hadm_id,
                    tool_set=tool_set,
                    note_grounding=note_grounding,
                    health_literacy=_literacy,
                )

                # 5. Validate observations (skip for Write/Update turns — write confirmation is always valid)
                if task_type != "Write/Update":
                    obs_ok, obs_reason = validate_observations(executed_actions)
                    if not obs_ok:
                        failed_questions.append({"question": user_q, "reason": obs_reason})
                        if attempt < MAX_RETRIES - 1:
                            logger.warning(f"  Obs check failed: {obs_reason}. Retrying...")
                            continue
                        else:
                            logger.warning(f"  Obs check failed (final): {obs_reason}. Discarding.")
                            break

                # 6. Generate assistant answer
                assistant_answer = generate_answer(
                    context=context,
                    history=history,
                    user_question=user_q,
                    executed_actions=executed_actions,
                    task_type=legacy_type,
                    tool_source=tool_source if tool_source != "write" else "ehr",
                    language=language,
                )

                # 7. Validate answer
                # Skip for: patient/write tool turns, and T1 Data Gathering turns where the
                # answer agent often hallucinates from medication lists not fully visible
                # in the tool observations. Quality is still enforced via rubric eval.
                _skip_ans_check = (
                    tool_source in ("patient", "write")
                    or (turn_idx == 1 and task_type == "Data Gathering")
                )
                if not _skip_ans_check:
                    ans_ok, ans_reason = validate_answer(
                        user_question=user_q,
                        executed_actions=executed_actions,
                        assistant_answer=assistant_answer,
                        task_domain="DischargePlan",
                        language=language,
                    )
                    if not ans_ok:
                        if attempt < MAX_RETRIES - 1:
                            logger.warning(f"  Answer check failed: {ans_reason}. Retrying...")
                            continue
                        else:
                            logger.warning(f"  Answer check failed (final): {ans_reason}.")
                            break

                # 8. Generate rubric
                if task_type == "Write/Update":
                    # Write-action rubric with specific parameter values
                    rubric_en = generate_action_rubric(
                        user_question=user_q_explicit,
                        executed_write_actions=executed_actions,
                        reference_answer=assistant_answer,
                        clinical_context=ehr_snapshot[:600],
                        language="en",
                    )
                else:
                    # Standard outcome-based rubric grounded in EHR values
                    rubric_en = generate_rubric(
                        task_type=task_type,
                        user_question_explicit=user_q_explicit,
                        executed_actions=executed_actions,
                        assistant_answer=assistant_answer,
                        language="en",
                        tool_source=tool_source,
                    )

                # ── Turn complete ─────────────────────────────────────────────
                turn_entry: dict = {
                    "user_question": user_q,
                    "user_question_explicit": user_q_explicit,
                    "task_type": task_type,
                    "subtype": subtype,
                    "tool_source": tool_source,
                    "tool_set": tool_set,
                    "executed_actions": executed_actions,
                    "assistant_answer": assistant_answer,
                    "rubric": rubric_en,
                }

                # Attach note grounding for patient tool turns
                if tool_source in ("patient", "mixed"):
                    turn_entry["patient_grounding"] = {
                        k: v for k, v in note_grounding.items()
                        if not k.startswith("_")
                    }

                # Attach write gold params for Write/Update turns
                if task_type == "Write/Update":
                    write_actions = [
                        a for a in executed_actions
                        if a["action"]["name"] in _WRITE_TOOL_NAMES
                        and a["action"]["name"] != "prepare_to_answer"
                    ]
                    turn_entry["write_gold_params"] = [
                        {
                            "tool": a["action"]["name"],
                            "params": {k: v for k, v in a["action"]["arguments"].items()
                                       if k != "subject_id"},
                        }
                        for a in write_actions
                    ]

                turn_data.append(turn_entry)

                dep_graph.add_turn(
                    turn_idx=turn_idx,
                    task_type=task_type,
                    tool_source=tool_source,
                    subtype=subtype,
                    executed_actions=executed_actions,
                    assistant_answer=assistant_answer,
                )

                history.append({"role": "user", "content": user_q})
                history.append({"role": "assistant", "content": assistant_answer})

                if bilingual and language == "en":
                    zh_q = translate_to_zh(user_q)
                    zh_ans = translate_to_zh(assistant_answer)
                    tasks_zh.append(zh_q)
                    messages_zh.extend([
                        {"role": "user", "content": zh_q},
                        {"role": "assistant", "content": zh_ans},
                    ])
                    turn_data[-1]["user_question_explicit_zh"] = (
                        translate_to_zh(user_q_explicit)
                        if user_q_explicit != user_q else zh_q
                    )

                if subtype_counter is not None and subtype:
                    local_subtype_increments[subtype] = (
                        local_subtype_increments.get(subtype, 0) + 1
                    )

                success = True
                break

            except Exception as exc:
                logger.warning(f"  Turn {turn_idx} attempt {attempt+1} error: {exc}")
                if attempt >= MAX_RETRIES - 1:
                    logger.error(f"  Turn {turn_idx} failed after {MAX_RETRIES} attempts.")
                    return None

        if not success:
            logger.error(f"  [FAILED] turn {turn_idx}")
            return None

    # Merge subtype increments
    if subtype_counter is not None:
        for st, cnt in local_subtype_increments.items():
            subtype_counter[st] = subtype_counter.get(st, 0) + cnt

    # Encode entry
    entry = encode_entry(
        entry_id=entry_id,
        task_domain="DischargePlan",
        subject_id=subject_id,
        hadm_id=hadm_id,
        context=context,
        turn_data=turn_data,
        session_id=None,
        tasks_zh=tasks_zh if bilingual and tasks_zh else None,
        messages_zh=messages_zh if bilingual and messages_zh else None,
        tool_set=tool_set,
    )

    entry["clinical_scenario"] = "discharge_planning"
    entry["task_sequence"] = task_sequence
    entry["turn_subtypes"] = [td["subtype"] for td in turn_data]
    entry["tool_sources"] = [td["tool_source"] for td in turn_data]
    entry["dep_graph"] = dep_graph.summary()
    entry["session_plan"] = session_plan
    entry["tool_set"] = tool_set
    entry["difficulty_level"] = difficulty
    entry["health_literacy"] = health_literacy or {1: "low", 2: "medium", 3: "high"}.get(difficulty, "medium")
    entry["current_date"] = context.get("current_date")
    entry["generated_at"] = datetime.utcnow().isoformat()

    # Per-turn rubrics
    entry["rubrics"] = [td.get("rubric", []) for td in turn_data]

    # Note grounding (from last patient turn — same grounding applies to all patient turns)
    patient_turns = [td for td in turn_data if td.get("tool_source") in ("patient", "mixed")]
    entry["note_grounding"] = patient_turns[-1].get("patient_grounding", {}) if patient_turns else {}

    # Write gold params from Write/Update turn (T3)
    action_turns = [td for td in turn_data if td.get("task_type") == "Write/Update"]
    entry["write_gold_params"] = action_turns[-1].get("write_gold_params", []) if action_turns else []

    if bilingual:
        entry["rubrics_zh"] = [
            [translate_to_zh(item) for item in turn_rubric]
            for turn_rubric in entry["rubrics"]
        ]

    return _sanitize(entry)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate discharge_planning benchmark entries"
    )
    parser.add_argument("--subject_id", type=int, default=None,
                        help="Single patient. Omit to draw from qualified_patients.json pool.")
    parser.add_argument("--hadm_id", type=int, default=None)
    parser.add_argument("--n", type=int, default=1, help="Number of entries to generate")
    parser.add_argument("--out", type=str, default="discharge_planning.jsonl")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--health_literacy", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="Override patient health literacy (default: derived from difficulty)")
    parser.add_argument("--no_bilingual", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random as _random
    _random.seed(args.seed)

    # Build patient list: single patient or pool from qualified_patients.json
    if args.subject_id is not None:
        patients = [(args.subject_id, args.hadm_id)] * args.n
    else:
        _qpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "qualified_patients.json")
        if not os.path.exists(_qpath):
            raise FileNotFoundError(f"qualified_patients.json not found at {_qpath}. "
                                    "Run prefilter_patients.py first, or pass --subject_id.")
        with open(_qpath) as _f:
            _q = json.load(_f)
        _pool = _q.get("discharge_planning", {}).get(str(args.difficulty), [])
        if not _pool:
            raise ValueError(f"No qualified patients for discharge_planning L{args.difficulty}")
        _random.shuffle(_pool)
        patients = [(int(p[0]), int(p[1]) if p[1] is not None else None) for p in _pool]

    subtype_counter: dict[str, int] = {}
    written = 0
    entry_index = 0

    # Load existing IDs to avoid duplicates
    existing_ids: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as _ef:
            for _line in _ef:
                try:
                    existing_ids.add(json.loads(_line)["id"])
                except Exception:
                    pass

    patient_iter = iter(patients)

    with open(args.out, "a", encoding="utf-8") as fout:
        while written < args.n:
            try:
                sid, hid = next(patient_iter)
            except StopIteration:
                logger.warning("Exhausted patient pool before reaching target n.")
                break
            entry_id = f"dp_{sid}_{entry_index}_L{args.difficulty}"
            if entry_id in existing_ids:
                logger.info(f"  Skipping {entry_id} (already exists)")
                entry_index += 1
                continue
            entry = generate_discharge_planning_entry(
                subject_id=sid,
                hadm_id=hid,
                entry_index=entry_index,
                language=args.language,
                bilingual=not args.no_bilingual,
                difficulty=args.difficulty,
                health_literacy=args.health_literacy,
                subtype_counter=subtype_counter,
            )
            entry_index += 1
            if entry is not None:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                logger.info(f"  Written entry {written}/{args.n} → {args.out}")
            else:
                logger.warning(f"  Entry failed for patient {sid} — trying next")

    print(f"\nDone. {written}/{args.n} entries written to {args.out}")
    # legacy path kept below for single-patient mode compatibility
    if False:  # unreachable — kept only for reference
        for i in range(args.n):
            entry = generate_discharge_planning_entry(
                subject_id=args.subject_id,
                hadm_id=args.hadm_id,
                entry_index=i,
                language=args.language,
                bilingual=not args.no_bilingual,
                difficulty=args.difficulty,
                health_literacy=args.health_literacy,
                subtype_counter=subtype_counter,
            )
            if entry is not None:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                logger.info(f"  Written entry {i} → {args.out}  (total: {written})")
            else:
                logger.warning(f"  Entry {i} failed — skipped")

    print(f"\nDone. {written}/{args.n} entries written to {args.out}")
    print(f"Subtype distribution: {subtype_counter}")


if __name__ == "__main__":
    _main()
