"""
physassistbench/pipeline/session_planner.py — Session-level conversation planner.

Before individual turns are generated, produces a coherent clinical narrative
and per-turn topic + tool-pattern plan grounded in the patient's EHR snapshot.

Returns a dict with:
  clinical_situation  — one-sentence patient summary
  investigation_arc   — one-sentence description of what the 4-turn session investigates
  turns               — list of per-turn dicts:
      turn            — int index
      task_type       — matches the task_sequence
      topic           — specific data item to ask about (must exist in EHR snapshot)
      tool_hint       — FHIR tool(s) to call, e.g. "Observation.search(creatinine)"
      workup_pattern  — (Data Gathering turns only) e.g. "MedReq×MedReq"
"""

from __future__ import annotations

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physassistbench.pipeline.agents.llm_client import llm_call, extract_json

logger = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_EN = """\
You are a clinical conversation planner for an EHR benchmark dataset.

Given a patient's EHR snapshot and a task-type sequence, generate a session plan
that tells ONE coherent clinical story across all turns.

OUTPUT: Return ONLY valid JSON — no prose, no markdown fences.
IMPORTANT: Generate turn_intents and investigation_arc following the ACTUAL task sequence
provided in the user message — do NOT copy the example sequence shown in the schema below.
{
  "clinical_situation": "<one sentence: patient's main problem>",
  "investigation_arc": "T0[R/W/KG] <arc phrase> → T1[R/W/KG] <arc phrase> → T2[R/W/KG] <arc phrase> → T3[R/W/KG] <arc phrase>",
  "turn_intents": [
    "<full intent sentence for turn 0 — see format rules below>",
    "<full intent sentence for turn 1>",
    "<full intent sentence for turn 2>",
    "<full intent sentence for turn 3>"
  ],
  "turns": [
    {
      "turn": 0,
      "task_type": "<must match the sequence exactly>",
      "topic": "<specific data item(s) — MUST exist in the EHR snapshot>",
      "tool_hint": "<FHIR tool(s) to call, e.g. Observation.search(creatinine)>",
      "tool_source": "ehr",
      "workup_pattern": "<Data Gathering turns only — see patterns below>"
    }
  ]
}

PATIENT INTERVIEW OPTION (use when clinically meaningful):
In AT MOST ONE turn per session, you may integrate patient interview tools.
Set tool_source="mixed" (EHR + patient) or tool_source="patient" (patient only) for that turn.
All other turns must have tool_source="ehr".

Available patient tools:
  patient.get_symptom_history(query=<symptom>)  — ask patient to describe symptoms
  patient.get_medication_adherence(drug=<drug>) — ask if patient is taking medication as prescribed
  patient.get_functional_status()               — ask about daily activity limitations
  patient.get_social_history()                  — ask about living situation and support

When to use patient tools (scenario-specific guidance):
  lab_trend:           patient.get_symptom_history() to correlate a lab trend with reported symptoms
                       (e.g. rising WBC + patient reports fever/chills; falling Hgb + fatigue)
  med_safety:          patient.get_medication_adherence(drug=<drug>) to verify the patient is
                       actually taking the medication whose lab safety is being monitored
  treatment_response:  patient.get_symptom_history() or patient.get_functional_status() to
                       assess whether the patient subjectively perceives improvement

Rules for patient turns:
- Use patient tools in AT MOST ONE turn. Do NOT add patient tools if no clear clinical value.
- For "mixed" turns: tool_hint combines EHR tool + patient tool, e.g.
    "Observation.search(WBC) + patient.get_symptom_history(query=fever)"
- For "patient" turns: tool_hint lists only patient tools.
- The topic for a mixed/patient turn must describe BOTH what EHR provides AND what the
  patient is asked — NOT just a lab value or a drug name alone.

RULES:
1. The EHR snapshot starts with a [QUERYABLE ITEMS] block. Every topic you choose MUST
   appear in that block. Labs listed under "1 result (Information Lookup turns only)" may ONLY be
   used in Information Lookup turns — never in Data Gathering (trend) or Clinical Reasoning turns.
   Labs listed under "≥2 results" may be used in any turn type.
   Do NOT choose any lab, drug, or resource that is absent from the [QUERYABLE ITEMS] block.
2. Topics must not be redundant across turns — each turn adds new information.
3. The turns should read as one progressive clinical investigation (not random questions).
4. The item(s) named in turn_intents[i] MUST match the topic and tool_hint in turns[i].
5. TOOL DIVERSITY (within this session):
   - Each FHIR resource type (Observation, MedicationRequest, Condition, etc.) may appear
     in AT MOST 2 turns. Do NOT call the same tool in every turn.
   - The session MUST use at least 2 distinct FHIR resource types across all turns.
   - Avoid patterns like Observation.search in T0, T1, T2, and T3 simultaneously.
6. CLINICAL SCORING PRIORITY: If the EHR snapshot contains a "CLINICAL SCORING
   OPPORTUNITIES" section, you MUST plan at least one Data Gathering or Clinical Reasoning turn
   that computes the listed score. The tool_hint for that turn should retrieve ALL
   required components in parallel (e.g. for SOFA: Observation.search(platelet) +
   Observation.search(bilirubin) + Observation.search(creatinine)). The topic must
   explicitly state the score name and what clinical decision it drives.

investigation_arc format (REQUIRED — one arc phrase per turn, joined by "→"):
  — Information Lookup:          "T{i}[R] retrieve [item] — [clinical purpose]"
  — Data Gathering:             "T{i}[W] [item A] × [item B] — [clinical question]"
  — Clinical Reasoning: "T{i}[KG] interpret [item] — [clinical decision]"
  — Action:             "T{i}[A] [write operation] — [clinical justification]"

turn_intents format (REQUIRED — one full sentence per turn):
  — Information Lookup:          "T{i} [Information Lookup]: retrieve [specific item] to [establish/confirm/track/...]"
  — Data Gathering:             "T{i} [Data Gathering/{pattern}]: co-retrieve [item A] × [item B] — [clinical question per pattern]"
  — Clinical Reasoning: "T{i} [Clinical Reasoning]: interpret [item] to assess [clinical decision]"
  — Action:             "T{i} [Action]: [MedicationRequest.create/ServiceRequest.create/Flag.create] — [clinical justification from prior turns]"

For ACTION turns (T3 only):
  — task_type: "Write/Update"
  — tool_hint: exactly ONE write tool call with concrete parameters drawn from the EHR snapshot.
    Examples:
      "Flag.create(category=clinical, code=CRITICAL_ANEMIA, detail=Hgb 6.2 g/dL — transfusion threshold met)"
      "MedicationRequest.create(medication=Metformin, dose=500mg, route=oral, frequency=once daily, indication=dose reduction for eGFR 35)"
      "ServiceRequest.create(service_type=nephrology-followup, priority=urgent, note=eGFR 28 with active nephrotoxin exposure)"
  — The write action MUST be clinically justified by findings from T0–T2.
  — tool_source: "write"
  — NO workup_pattern field

  Data Gathering clinical question phrasing BY PATTERN:
    Obs×Obs:         what comparing the two lab values reveals, or what the trend indicates
    Obs×MedReq:      dosing appropriateness, safety, or drug-lab interaction
    MedReq×MedReq:   drug-drug interaction, redundancy, or overlap
    Obs×Condition:   how the lab contextualizes or confirms the diagnosis
    MedReq×MedAdmin: whether the ordered drug was actually administered as prescribed
    Obs×MedAdmin:    how the administered dose relates to the observed lab result

For RETRIEVAL turns:
  — topic: one specific lab, medication, diagnosis, or vital sign
  — tool_hint: single FHIR tool, e.g. "Observation.search(potassium)"

For WORKUP turns:
  — topic: the TWO (or three) specific data items being co-retrieved, e.g. "creatinine and metformin order"
  — tool_hint: the two (or three) tools combined, e.g. "MedicationRequest.search(warfarin) + MedicationRequest.search(aspirin)"
  — workup_pattern: MUST be one of the following (pick the best fit from the EHR data):

  TIER 1 — preferred (data always reliable):
    Obs×Obs           two lab values together, OR one lab trend over time
    Obs×MedReq        abnormal lab + active drug order (safety/dose monitoring)
    MedReq×MedReq     two co-prescribed drugs (interaction or overlap check)
    Obs×Condition     lab value + ICD diagnosis context
    MedReq×MedAdmin   drug order vs. actual eMAR administration records

  TIER 2 — use when Tier 1 doesn't fit:
    Obs×MedAdmin      lab value + actual admin records
    MedReq×Condition  drug order + diagnosis appropriateness
    3-tool            Obs+Obs+MedReq | Obs+MedReq+MedAdmin | Obs+MedReq+Condition

For KNOWLEDGE-GROUNDED turns:
  — topic: ONE patient parameter + the clinical knowledge question it triggers
  — tool_hint: single FHIR tool, e.g. "Observation.search(eGFR)"
  — NO workup_pattern field\
"""

_SYSTEM_ZH = """\
你是一个EHR基准数据集的临床对话规划器。

给定患者EHR快照和任务类型序列，生成一个覆盖所有轮次的、讲述同一临床故事的会话计划。

输出：只返回有效JSON，不含散文或markdown代码块。
{
  "clinical_situation": "<一句话：患者的主要问题>",
  "investigation_arc": "T0[R] <弧短语> → T1[W] <弧短语> → T2[KG] <弧短语> → ...",
  "turn_intents": [
    "T0 [Information Lookup]：获取[具体指标]——[临床目的]",
    "T1 [Data Gathering/Obs×Obs]：并行查询[指标A] × [指标B]——[比较两者揭示的内容]",
    "T2 [Clinical Reasoning]：解读[指标]以评估[临床决策]",
    "T3 [Information Lookup]：获取[具体指标]——[临床目的]"
  ],
  "turns": [
    {
      "turn": 0,
      "task_type": "<必须与序列完全匹配>",
      "topic": "<具体数据项——必须存在于EHR快照中>",
      "tool_hint": "<调用的FHIR工具，例如 Observation.search(creatinine)>",
      "tool_source": "ehr",
      "workup_pattern": "<仅Workup轮填写——见下方模式>"
    }
  ]
}

病人访谈选项（仅在有明确临床价值时使用）：
每个会话最多只有一轮可以融入病人访谈工具。
对该轮设置 tool_source="mixed"（EHR+病人）或 tool_source="patient"（纯病人访谈）。
其余所有轮次必须为 tool_source="ehr"。

可用病人工具：
  patient.get_symptom_history(query=<症状>)   — 询问患者描述症状
  patient.get_medication_adherence(drug=<药名>) — 询问患者是否按处方服药
  patient.get_functional_status()              — 询问日常活动能力
  patient.get_social_history()                 — 询问生活状况和支持体系

各场景使用时机：
  lab_trend：         patient.get_symptom_history() 将化验趋势与患者自述症状相关联
                      （例如：WBC升高 + 患者报告发热/寒战；Hgb下降 + 乏力）
  med_safety：        patient.get_medication_adherence(drug=<药名>) 验证患者是否确实在服用
                      被监测安全性的药物
  treatment_response：patient.get_symptom_history() 或 patient.get_functional_status()
                      评估患者主观上是否感到好转

病人轮规则：
- 最多只有一轮使用病人工具。若无明确临床价值则不添加。
- "mixed"轮：tool_hint 同时包含EHR工具和病人工具，例如：
    "Observation.search(WBC) + patient.get_symptom_history(query=发热)"
- "patient"轮：tool_hint 只列病人工具。
- mixed/patient轮的 topic 必须同时描述EHR数据和病人被询问的内容，而不是单纯列化验值或药名。

规则：
1. EHR快照开头有一个[QUERYABLE ITEMS]块。你选择的每个topic必须出现在该块中。
   标注为"1 result (Information Lookup turns only)"的化验只能用于Retrieval轮，
   不得用于Workup（趋势）或Knowledge-Grounded轮。
   标注为"≥2 results"的化验可用于任意轮次类型。
   不得选择[QUERYABLE ITEMS]块中未列出的化验、药物或资源。
2. 各轮topic不得重复——每轮添加新信息。
3. 各轮应构成一个递进的临床调查（不是随机问题）。
4. turn_intents[i]中提到的指标必须与turns[i]的topic和tool_hint一致。
5. 工具多样性（本session内）：
   - 每种FHIR资源类型（Observation、MedicationRequest、Condition等）在所有轮次中
     最多出现2次。不得在每一轮都调用同一个工具。
   - 整个session必须至少使用2种不同的FHIR资源类型。
   - 避免T0、T1、T2、T3同时全部使用Observation.search的模式。
6. 临床评分优先级：若EHR快照包含"CLINICAL SCORING OPPORTUNITIES"段，
   必须规划至少一个Workup或Knowledge-Grounded轮来计算其中列出的评分。
   该轮的tool_hint应并行检索所有所需组件（如SOFA：
   Observation.search(血小板) + Observation.search(胆红素) + Observation.search(肌酐)）。
   topic必须明确写出评分名称及其驱动的临床决策。

investigation_arc格式（必填——每轮一个弧短语，用"→"连接）：
  — Information Lookup：         "T{i}[R] 获取[指标]——[临床目的]"
  — Data Gathering：            "T{i}[W] [指标A] × [指标B]——[临床问题]"
  — Clinical Reasoning："T{i}[KG] 解读[指标]——[临床决策]"

turn_intents格式（必填——每轮一个完整句子）：
  — Information Lookup：         "T{i} [Information Lookup]：获取[具体指标]以[建立基线/确认/追踪/...]"
  — Data Gathering：            "T{i} [Data Gathering/{模式}]：并行查询[指标A] × [指标B]——[按模式确定的临床问题]"
  — Clinical Reasoning："T{i} [Clinical Reasoning]：解读[指标]以评估[临床决策]"

  各Workup模式的临床问题措辞：
    Obs×Obs：         两个化验值比较揭示了什么，或趋势说明什么
    Obs×MedReq：      剂量合理性、用药安全性或药-化验交互
    MedReq×MedReq：   药物相互作用、重复用药或重叠
    Obs×Condition：   化验值如何印证或解释诊断
    MedReq×MedAdmin： 医嘱药物是否按处方实际给药
    Obs×MedAdmin：    实际给药剂量与观测化验值的关系

Retrieval轮：
  — topic：一个具体的化验、用药、诊断或生命体征
  — tool_hint：单个FHIR工具，例如 "Observation.search(potassium)"

Workup轮：
  — topic：正在调查的临床问题
  — tool_hint：两个（或三个）工具的组合，例如 "MedicationRequest.search(warfarin) + MedicationRequest.search(aspirin)"
  — workup_pattern：必须是以下之一（从EHR数据中选择最合适的）：

  第一优先级（优先选择，数据可靠）：
    Obs×Obs           两个化验值并行，或单个化验随时间趋势
    Obs×MedReq        异常化验值 + 当前药物医嘱（安全性/剂量监测）
    MedReq×MedReq     两种同时处方的药物（相互作用或重叠核查）
    Obs×Condition     化验值 + ICD诊断背景
    MedReq×MedAdmin   药物医嘱 vs 实际eMAR给药记录

  第二优先级（第一优先级不适用时）：
    Obs×MedAdmin      化验值 + 实际给药记录
    MedReq×Condition  药物医嘱 + 诊断合理性
    三工具            Obs+Obs+MedReq | Obs+MedReq+MedAdmin | Obs+MedReq+Condition

Knowledge-Grounded轮：
  — topic：一个患者参数 + 它触发的临床知识问题
  — tool_hint：单个FHIR工具，例如 "Observation.search(eGFR)"
  — 不填 workup_pattern 字段\
"""


# ── Public API ─────────────────────────────────────────────────────────────────

def plan_session(
    ehr_snapshot: str,
    task_sequence: list[str],
    scenario: str,
    n_turns: int = 4,
    language: str = "en",
    scenario_constraints: dict | None = None,
    difficulty_constraints: dict | None = None,
    tools_to_prioritize: list[str] | None = None,
) -> dict | None:
    """
    Generate a coherent session plan before turn-level question generation.

    Args:
        ehr_snapshot:          Full EHR snapshot string (trimmed to 6000 chars).
        task_sequence:         List of task type strings, e.g. ["Information Lookup", "Data Gathering", ...].
        scenario:              Clinical scenario name.
        n_turns:               Number of turns (default 4).
        language:              "en" | "zh".
        scenario_constraints:  Per-scenario constraint dict from SCENARIO_CONSTRAINTS.
        difficulty_constraints: Per-difficulty constraint dict from DIFFICULTY_CONSTRAINTS.
        tools_to_prioritize:   Tools underused in the current dataset; inject as a coverage
                               hint so the planner exercises them when clinically appropriate.

    Returns:
        dict with keys clinical_situation, investigation_arc, turns — or None on failure.
    """
    is_zh = language == "zh"
    sys_prompt = _SYSTEM_ZH if is_zh else _SYSTEM_EN

    # Append scenario-specific structural constraints to system prompt
    if scenario_constraints:
        constraint_key = "constraint_text_zh" if is_zh else "constraint_text_en"
        constraint_block = scenario_constraints.get(constraint_key, "")
        if constraint_block:
            sys_prompt = sys_prompt + "\n\n" + constraint_block

    # Append difficulty-level constraints to system prompt
    if difficulty_constraints:
        constraint_key = "constraint_text_zh" if is_zh else "constraint_text_en"
        diff_block = difficulty_constraints.get(constraint_key, "")
        if diff_block:
            sys_prompt = sys_prompt + "\n\n" + diff_block

    # Append coverage hint — steer planner toward underused tools (Solution B)
    if tools_to_prioritize:
        tool_list = ", ".join(tools_to_prioritize)
        if is_zh:
            coverage_hint = (
                f"工具覆盖提示：以下工具在当前数据集中调用次数偏少，"
                f"如在临床上合适，请优先选用它们：{tool_list}"
            )
        else:
            coverage_hint = (
                f"COVERAGE HINT: The following tools are underused in the current dataset. "
                f"Prefer them over frequently-used tools when clinically appropriate: {tool_list}"
            )
        sys_prompt = sys_prompt + "\n\n" + coverage_hint

    seq_str = "  →  ".join(
        f"Turn {i} [{t}]" for i, t in enumerate(task_sequence[:n_turns])
    )
    # Extract the CLINICAL SCORING OPPORTUNITIES section first (it's appended at the end)
    # so it doesn't get lost in truncation.
    _scoring_section = ""
    if "CLINICAL SCORING OPPORTUNITIES" in ehr_snapshot:
        _idx = ehr_snapshot.index("CLINICAL SCORING OPPORTUNITIES")
        _scoring_section = "\n" + ehr_snapshot[_idx:_idx + 1500]

    # Truncate the main EHR body, then append scoring section to ensure it's visible
    snapshot_trimmed = ehr_snapshot[:5500] + _scoring_section

    if is_zh:
        user_prompt = (
            f"临床场景：{scenario}\n\n"
            f"任务序列：{seq_str}\n\n"
            f"患者EHR快照：\n{snapshot_trimmed}\n\n"
            "生成会话计划JSON："
        )
    else:
        user_prompt = (
            f"Clinical scenario: {scenario}\n\n"
            f"Task sequence: {seq_str}\n\n"
            f"Patient EHR snapshot:\n{snapshot_trimmed}\n\n"
            "Generate the session plan JSON:"
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = llm_call(messages, temperature=0.4, max_tokens=2000)
        plan = extract_json(raw)
        if not isinstance(plan, dict) or "turns" not in plan:
            logger.warning("plan_session: LLM returned invalid structure — skipping plan")
            return None
        if len(plan["turns"]) < n_turns:
            logger.warning(
                f"plan_session: only {len(plan['turns'])} turns in plan, expected {n_turns}"
            )
        if "turn_intents" not in plan:
            logger.warning("plan_session: turn_intents missing — filling with empty strings")
            plan["turn_intents"] = [""] * n_turns
        elif len(plan["turn_intents"]) < n_turns:
            logger.warning(
                f"plan_session: only {len(plan['turn_intents'])} turn_intents, expected {n_turns} — padding"
            )
            plan["turn_intents"] += [""] * (n_turns - len(plan["turn_intents"]))
        logger.info(
            f"  session plan: {plan.get('investigation_arc', '')[:100]}"
        )
        return plan
    except Exception as exc:
        logger.warning(f"plan_session failed ({exc}) — proceeding without plan")
        return None
