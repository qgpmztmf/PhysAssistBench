"""
physassistbench/generate_all.py — Generate benchmark entries using HL7 FHIR R4 tools.

Each entry:
  - Uses FHIR R4 tool set (Observation.search, Condition.search, etc.) by default
  - Follows one of 81 task-type sequences (3^4 = Information Lookup/Data Gathering/Clinical Reasoning)
  - Uses one of 11 clinical scenarios for grounding fact sampling
  - Assigns subtypes EA/PE/AU/AE to turns 1-3 dynamically (feasibility-first, counter-balanced)
  - Stores tool_source annotation (ehr / patient / mixed) per turn
  - Stores tool_set="fhir" in the entry
  - Stores TurnDependencyGraph in the entry for analysis
  - Optionally produces bilingual output (tasks_en + tasks_zh)

Output: physassistbench/data/<scenario>.jsonl

Usage:
    cd /path/to/PhysAssistBench
    uv run python physassistbench/generate_all.py [--scenarios infection_management critical_care] [--n 2]
    uv run python physassistbench/generate_all.py --tool_set legacy   # use legacy tools (same as PhysAssistBench)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physassistbench.pipeline.patient_selector import select_patients
from physassistbench.phm.patient_agent_runtime import register_session, get_session, reset_all_sessions

from physassistbench.pipeline.generate_v2 import generate_one_entry_v2
from physassistbench.pipeline.sequences import get_eligible_arc_indices, SUBTYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Qualified-patient index ───────────────────────────────────────────────────
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_QUALIFIED_PATH = os.path.join(_PKG_DIR, "qualified_patients.json")

# Loaded once per process; None means the JSON does not exist yet.
_qualified_index: dict | None | bool = False   # False = not yet attempted


def _load_qualified_index() -> dict | None:
    """Return the qualified-patient index, or None if not available."""
    global _qualified_index
    if _qualified_index is not False:
        return _qualified_index  # type: ignore[return-value]
    if not os.path.exists(_QUALIFIED_PATH):
        logger.info(
            f"qualified_patients.json not found at {_QUALIFIED_PATH}. "
            "Run physassistbench/prefilter_patients.py to generate it. "
            "Falling back to select_patients()."
        )
        _qualified_index = None
        return None
    with open(_QUALIFIED_PATH, encoding="utf-8") as f:
        _qualified_index = json.load(f)
    logger.info(f"Loaded qualified patient index from {_QUALIFIED_PATH}")
    return _qualified_index  # type: ignore[return-value]


def _get_patient_pool(
    scenario: str,
    difficulty: int | None,
    task_domain: str,
    pool_size: int,
) -> list[tuple[int, int | None]]:
    """
    Return a patient pool for the given scenario and difficulty.

    Priority:
      1. qualified_patients.json (pre-filtered, full MIMIC-IV dataset)
      2. select_patients() fallback (local 2k-patient subset)

    When difficulty is None (cycling mode), the pools for L1/L2/L3 are merged
    and deduplicated so the caller can iterate freely.
    """
    index = _load_qualified_index()
    if index and scenario in index:
        scenario_index = index[scenario]
        if difficulty is not None:
            pool = scenario_index.get(str(difficulty), [])
        else:
            # Merge all difficulty pools; keep insertion order, deduplicate
            seen: set[int] = set()
            pool = []
            for diff in ["1", "2", "3"]:
                for pair in scenario_index.get(diff, []):
                    sid = pair[0]
                    if sid not in seen:
                        seen.add(sid)
                        pool.append(pair)

        if pool:
            result = [(int(p[0]), int(p[1]) if p[1] is not None else None)
                      for p in pool]
            logger.info(
                f"[{scenario}] Using pre-filtered pool: "
                f"{len(result)} candidates (difficulty={difficulty})"
            )
            return result
        logger.warning(
            f"[{scenario}/L{difficulty}] Qualified pool is empty — "
            "falling back to select_patients()"
        )

    # Fallback: legacy patient selector
    return select_patients(task_domain, n=pool_size, seed=SEED)

# ── Configuration ─────────────────────────────────────────────────────────────

# 3 benchmark scenarios, each anchored to a distinct FHIR resource combination
SCENARIO_NAMES = [
    "diagnostic_workup",  # Observation + DiagnosticReport + Condition: differential diagnosis
    "med_safety",         # Observation + MedicationRequest: lab-drug safety review
    "treatment_response", # Observation + MedicationAdministration + Condition: treatment monitoring
    "discharge_planning", # MedicationRequest + Condition + Observation: discharge reconciliation
]

# EHR domain per scenario — controls patient pool selection
SCENARIO_TO_DOMAIN: dict[str, str] = {
    "diagnostic_workup":   "LabInterp",
    "med_safety":          "MedRecon",
    "treatment_response":  "ICUReasoning",
    "discharge_planning":  "MedRecon",
}

# Per-scenario structural constraints injected into the session planner prompt.
# Each scenario has a forbidden resource set and an allowed workup_pattern set
# so the generated tool sequences are structurally distinct across scenarios.
SCENARIO_CONSTRAINTS: dict[str, dict] = {
    "diagnostic_workup": {
        "allowed_resources":   ["Observation", "DiagnosticReport", "Condition",
                                "Encounter", "MedicationRequest"],
        "forbidden_resources": ["MedicationAdministration"],
        "workup_patterns":     ["Obs×DiagReport", "Obs×Condition",
                                "Obs×Obs", "DiagReport×Condition"],
        "constraint_text_en": (
            "SCENARIO CONSTRAINTS (diagnostic_workup — Multi-System Diagnostic Reasoning):\n"
            "- Clinical focus: multi-system pattern recognition and differential diagnosis generation.\n"
            "- REQUIRED: At least one Data Gathering turn MUST use DiagnosticReport.search() alongside "
            "Observation.search() or Condition.search() (pattern Obs×DiagReport or DiagReport×Condition). "
            "This forces use of radiology/pathology reports — the key tool differentiating this scenario.\n"
            "- Also use Condition.search() to retrieve the problem list and DiagnosticReport.search() "
            "to check prior imaging/reports (avoid duplicate testing).\n"
            "- Encounter.search() and MedicationRequest.search() may be used to identify timeline "
            "patterns and drug-induced disease as a differential.\n"
            "- Do NOT use MedicationAdministration.search() — that is for treatment_response.\n"
            "- Clinical Reasoning turns MUST reason about differential diagnosis: which diagnoses "
            "are consistent with the multi-system findings, and what distinguishes them.\n"
            "- Write/Update turns (when present): use ServiceRequest.create() to order targeted diagnostic "
            "tests that address identified information gaps (e.g. ADAMTS13 for TMA, complement for HUS, "
            "AFP for hepatoma). Cite specific lab values that justify each ordered test.\n"
            "- PATIENT INTERVIEW (required at L3, otherwise optional): patient.get_symptom_history() is highly relevant — "
            "symptom timeline and character are critical for differential diagnosis narrowing. "
            "Set tool_source='mixed' when a Data Gathering or KG turn benefits from patient-reported symptoms."
        ),
        "constraint_text_zh": (
            "场景约束（diagnostic_workup — 多系统诊断推理）：\n"
            "- 临床重点：多系统模式识别与鉴别诊断生成。\n"
            "- 必须：至少一个Workup轮必须将DiagnosticReport.search()与Observation.search()或"
            "Condition.search()配合使用（模式Obs×DiagReport或DiagReport×Condition）。"
            "这是本场景区别于其他场景的核心工具。\n"
            "- 使用Condition.search()获取问题列表，使用DiagnosticReport.search()查阅既往影像/报告（避免重复检查）。\n"
            "- 可使用Encounter.search()识别时间线模式，MedicationRequest.search()排查药物诱发疾病。\n"
            "- 禁止使用MedicationAdministration.search()——该工具属于treatment_response场景。\n"
            "- Knowledge-Grounded轮必须进行鉴别诊断推理：哪些诊断与多系统发现一致，各有何区别。\n"
            "- Action轮（如有）：使用ServiceRequest.create()开具针对信息缺口的诊断检查，"
            "引用具体化验值和影像发现作为每项检查的依据。\n"
            "- 病人访谈（可选）：patient.get_symptom_history()高度相关——"
            "症状时间线与特征是鉴别诊断的核心依据。"
        ),
    },
    "lab_trend": {
        "allowed_resources":   ["Observation"],
        "forbidden_resources": ["MedicationRequest", "MedicationAdministration", "Condition"],
        "workup_patterns":     ["Obs×Obs"],
        "constraint_text_en": (
            "SCENARIO CONSTRAINTS (lab_trend — Pure Lab Trend Analysis):\n"
            "- Use ONLY Observation.search() in ALL EHR turns. "
            "Do NOT call MedicationRequest, MedicationAdministration, or Condition.\n"
            "- Choose biomarkers from: CBC (WBC/Hgb/Hct/platelets), BMP/CMP "
            "(Na/K/Cr/BUN/glucose/bicarbonate), liver panel (ALT/AST/ALP/bilirubin), "
            "cardiac enzymes (troponin/CK/CK-MB), coagulation (PT/INR).\n"
            "- Each turn MUST ask about a DIFFERENT biomarker — no repeats across turns.\n"
            "- Data Gathering turns MUST use Obs×Obs: two distinct lab values queried together.\n"
            "- Clinical Reasoning turns MUST interpret a trend "
            "(rising/falling values over time), not just a single data point.\n"
            "- PATIENT INTERVIEW (required at L3, otherwise optional): In AT MOST ONE turn, set tool_source='mixed' "
            "and add patient.get_symptom_history(query=<symptom>) alongside an "
            "Observation.search() call. Use only when a specific lab trend has a clear "
            "symptomatic correlate (e.g. falling Hgb + fatigue, rising WBC + fever/chills, "
            "elevated bilirubin + jaundice). Do NOT use if no such correlate exists.\n"
            "- ACTION turn (T3 only, when arc ends in Action): Use Flag.create to flag a "
            "critical lab value (e.g. CRITICAL_ANEMIA, HYPERKALEMIA_ALERT) OR "
            "ServiceRequest.create to order a follow-up lab/referral based on the trend. "
            "Parameters MUST be grounded in actual lab values from the EHR snapshot."
        ),
        "constraint_text_zh": (
            "场景约束（lab_trend — 纯实验室趋势分析）：\n"
            "- 所有EHR轮次只能使用 Observation.search()，"
            "禁止调用 MedicationRequest、MedicationAdministration 或 Condition。\n"
            "- 生物标志物选自：CBC（WBC/Hgb/Hct/血小板）、BMP/CMP（Na/K/Cr/BUN/葡萄糖/碳酸氢根）、"
            "肝功能（ALT/AST/ALP/胆红素）、心肌酶谱（肌钙蛋白/CK/CK-MB）、凝血（PT/INR）。\n"
            "- 每轮必须询问不同的生物标志物，各轮不得重复。\n"
            "- Workup轮必须使用 Obs×Obs 模式：同时查询两个不同的化验指标。\n"
            "- Knowledge-Grounded轮必须解读趋势（随时间上升/下降），而非单点数值。\n"
            "- 病人访谈（可选）：最多一轮可设置 tool_source='mixed'，"
            "在 Observation.search() 旁附加 patient.get_symptom_history(query=<症状>)。"
            "仅当化验趋势有明确症状关联时才使用（如Hgb下降+乏力、WBC升高+发热/寒战、"
            "胆红素升高+黄疸）。若无此类关联则不使用。"
        ),
    },
    "med_safety": {
        "allowed_resources":   ["Observation", "MedicationRequest"],
        "forbidden_resources": ["MedicationAdministration"],
        "workup_patterns":     ["Obs×MedReq", "MedReq×MedReq"],
        "constraint_text_en": (
            "SCENARIO CONSTRAINTS (med_safety — Medication Safety Review):\n"
            "- Use Observation.search() and MedicationRequest.search() ONLY. "
            "Do NOT call MedicationAdministration.\n"
            "- ALL Data Gathering turns MUST use Obs×MedReq or MedReq×MedReq pattern "
            "(never Obs×Obs alone).\n"
            "- Clinical focus: renal function (Cr/eGFR/BUN) + nephrotoxic or "
            "renally-dosed medications, OR metabolic labs (HbA1c/glucose) + "
            "diabetes medications, OR electrolytes (K/Mg) + related drugs.\n"
            "- Information Lookup turns may fetch either a lab value OR a medication order.\n"
            "- Clinical Reasoning turns MUST reason about a drug safety implication: "
            "dose adjustment, contraindication, or drug-lab interaction.\n"
            "- PATIENT INTERVIEW (required at L3, otherwise optional): In AT MOST ONE turn, set tool_source='mixed' "
            "and add patient.get_medication_adherence(drug=<drug>) alongside an "
            "Observation.search() or MedicationRequest.search() call. Use only when "
            "verifying whether the patient is actually taking the specific drug whose "
            "lab safety is being assessed (e.g. Metformin + creatinine check, "
            "Warfarin + INR, ACE-inhibitor + potassium). Do NOT use if adherence is "
            "not clinically relevant to the safety question.\n"
            "- ACTION turn (T3 only, when arc ends in Action): Use MedicationRequest.create "
            "to adjust a drug dose based on the safety finding (e.g. reduce Metformin dose "
            "for low eGFR) OR Flag.create to alert on a drug-lab safety concern "
            "(e.g. NEPHROTOXIN_ALERT, HIGH_INR_FLAG). Parameters MUST cite specific "
            "lab values and drug names from the EHR snapshot."
        ),
        "constraint_text_zh": (
            "场景约束（med_safety — 药物安全性审查）：\n"
            "- 只能使用 Observation.search() 和 MedicationRequest.search()，"
            "禁止调用 MedicationAdministration。\n"
            "- 所有 Data Gathering 轮必须使用 Obs×MedReq 或 MedReq×MedReq 模式（不得单独使用 Obs×Obs）。\n"
            "- 临床重点：肾功能（Cr/eGFR/BUN）+ 肾毒性或需调剂量药物，"
            "或代谢指标（HbA1c/血糖）+ 降糖方案，或电解质（K/Mg）+ 相关药物。\n"
            "- Retrieval轮可以单独查化验值或药物医嘱。\n"
            "- Knowledge-Grounded轮必须推断药物安全含义：剂量调整、禁忌症或药-化验交互。\n"
            "- 病人访谈（可选）：最多一轮可设置 tool_source='mixed'，"
            "在 Observation.search() 或 MedicationRequest.search() 旁附加 "
            "patient.get_medication_adherence(drug=<药名>)。"
            "仅当需要核实患者是否确实在服用被评估安全性的特定药物时才使用"
            "（如二甲双胍+肌酐检查、华法林+INR、ACE抑制剂+血钾）。"
            "若依从性与安全问题无临床关联则不使用。"
        ),
    },
    "treatment_response": {
        "allowed_resources":   ["Observation", "MedicationAdministration", "Condition"],
        "forbidden_resources": ["MedicationRequest"],
        "workup_patterns":     ["Obs×MedAdmin", "Obs×Condition", "MedReq×MedAdmin"],
        "constraint_text_en": (
            "SCENARIO CONSTRAINTS (treatment_response — Treatment Response Monitoring):\n"
            "- Focus exclusively on infection/treatment trajectory. "
            "Do NOT call MedicationRequest — use MedicationAdministration (actual eMAR records) instead.\n"
            "- Primary biomarkers: WBC, neutrophil count, CRP, PCT, lactate, temperature.\n"
            "- At least one Data Gathering turn MUST use MedicationAdministration.search() "
            "(actual antibiotic/treatment administration records).\n"
            "- Data Gathering patterns: Obs×MedAdmin (infection marker + antibiotic given), "
            "Obs×Condition (lab + active diagnosis), or MedReq×MedAdmin.\n"
            "- Clinical Reasoning turns MUST assess a clinical decision: "
            "antibiotic de-escalation, discharge readiness, or treatment escalation.\n"
            "- PATIENT INTERVIEW (required at L3, otherwise optional): In AT MOST ONE turn, set tool_source='mixed' "
            "and add patient.get_symptom_history(query=<symptom>) or "
            "patient.get_functional_status() alongside an Observation.search() call. "
            "Use when assessing whether the patient subjectively perceives improvement "
            "(e.g. WBC falling + patient reports less fever/pain, improving functional "
            "status alongside improving infection markers). Do NOT use unless there is "
            "a clear clinical reason to corroborate objective data with patient report.\n"
            "- ACTION turn (T3 only, when arc ends in Action): Use MedicationRequest.create "
            "to adjust/de-escalate antibiotic based on infection marker trends OR "
            "ServiceRequest.create to arrange step-down care, ID consult, or follow-up. "
            "Parameters MUST be grounded in actual drug names and lab values from the EHR snapshot."
        ),
        "constraint_text_zh": (
            "场景约束（treatment_response — 治疗反应监测）：\n"
            "- 专注于感染/治疗轨迹，禁止调用 MedicationRequest，"
            "改用 MedicationAdministration（实际eMAR给药记录）。\n"
            "- 主要生物标志物：WBC、中性粒细胞、CRP、PCT、乳酸、体温。\n"
            "- 至少一个 Data Gathering 轮必须使用 MedicationAdministration.search()（实际抗生素给药记录）。\n"
            "- Data Gathering 模式：Obs×MedAdmin（感染指标+抗生素给药）、"
            "Obs×Condition（化验+活跃诊断）或 MedReq×MedAdmin。\n"
            "- Knowledge-Grounded轮必须评估临床决策：抗生素降阶梯、出院时机或治疗升级。\n"
            "- 病人访谈（可选）：最多一轮可设置 tool_source='mixed'，"
            "在 Observation.search() 旁附加 patient.get_symptom_history(query=<症状>) 或 "
            "patient.get_functional_status()。"
            "用于评估患者是否主观感到好转时使用"
            "（如WBC下降+患者报告发热/疼痛减轻、感染指标改善+功能状态恢复）。"
            "若无明确临床理由将客观数据与患者自述相佐证则不使用。"
        ),
    },
}

N_PER_SCENARIO = 1
N_TURNS = 4
SEED = 42

# ── Difficulty level constraints injected into session_planner ────────────────
DIFFICULTY_CONSTRAINTS: dict[int, dict] = {
    1: {
        "constraint_text_en": (
            "DIFFICULTY LEVEL 1 — FOUNDATIONAL:\n"
            "Generate a straightforward clinical arc. Each turn retrieves or interprets "
            "a single data point. No formula calculations, no multi-drug interactions, "
            "no conflicting evidence. Suitable for testing basic EHR retrieval and "
            "single-step clinical reasoning."
        ),
        "constraint_text_zh": (
            "难度等级1——基础：\n"
            "生成简单直接的临床弧线。每轮检索或解读单个数据点。"
            "无需公式计算、多药相互作用或冲突证据。"
            "适用于测试基础EHR检索和单步临床推理。"
        ),
    },
    2: {
        "constraint_text_en": (
            "DIFFICULTY LEVEL 2 — INTERMEDIATE:\n"
            "The arc MUST require non-trivial reasoning in at least one turn. Choose ONE:\n"
            "  A) TREND: A Data Gathering turn compares ≥2 time-stamped values of the same lab "
            "to characterise a trend (rising/falling/stable over days).\n"
            "  B) FORMULA: A Clinical Reasoning turn applies a clinical formula or "
            "dosing table (e.g., Cockcroft-Gault for CrCl, eGFR-based dose adjustment, "
            "INR therapeutic range) — interpretation requires more than a single lookup.\n"
            "  C) DRUG-LAB THRESHOLD: A Data Gathering turn pairs an abnormal lab with an active "
            "drug that has a monitoring threshold (e.g., K⁺ + spironolactone, "
            "INR + warfarin, creatinine + metformin). The turn must assess safety.\n"
            "The arc must NOT be answerable by simple direct lookups alone."
        ),
        "constraint_text_zh": (
            "难度等级2——中级：\n"
            "弧线至少有一轮需要非平凡推理。选择以下之一：\n"
            "  A) 趋势：Workup轮比较同一化验的≥2个时间戳值，描述趋势（数天内上升/下降/稳定）。\n"
            "  B) 公式：Knowledge-Grounded轮应用临床公式或剂量表（如Cockcroft-Gault计算CrCl、"
            "基于eGFR的剂量调整、INR治疗范围）——解读需要超过单次查询。\n"
            "  C) 药-化验阈值：Workup轮将异常化验与有监测阈值的活跃药物配对"
            "（如K⁺+螺内酯、INR+华法林、肌酐+二甲双胍），该轮必须评估用药安全性。\n"
            "弧线不能仅靠简单直接查询来回答。"
        ),
    },
    3: {
        "constraint_text_en": (
            "DIFFICULTY LEVEL 3 — ADVANCED:\n"
            "The arc MUST feature complex clinical reasoning requiring prioritisation "
            "or synthesis. Choose ONE:\n"
            "  A) MULTI-DRUG CONFLICT: ≥2 active medications with competing effects or "
            "a significant interaction. A KG turn must reason which drug to adjust first "
            "and why — the answer requires weighing risks "
            "(e.g., ACE-I + spironolactone + NSAID causing hyperkalemia).\n"
            "  B) CONFLICTING EVIDENCE: ≥2 findings pointing to different management "
            "decisions. The KG or Data Gathering turn must acknowledge and reconcile the conflict "
            "(e.g., INR supratherapeutic + active GI bleed vs. mechanical valve → "
            "cannot simply stop anticoagulation).\n"
            "  C) SEQUENTIAL DEPENDENCY: A turn's clinical question is only determinable "
            "after a prior turn's result is known — the actual value (not just context) "
            "drives the next step (e.g., if eGFR < 30 → stop drug A; 30–60 → reduce "
            "dose; > 60 → continue). Plan the arc so T0 result determines T1 action.\n"
            "The arc must span ≥3 different FHIR resource types across the 4 turns."
        ),
        "constraint_text_zh": (
            "难度等级3——高级：\n"
            "弧线必须包含需要优先级判断或综合分析的复杂临床推理。选择以下之一：\n"
            "  A) 多药冲突：≥2种活跃药物存在竞争效应或显著相互作用。"
            "KG轮必须推理首先调整哪种药物及原因——答案需权衡风险"
            "（如ACE-I+螺内酯+NSAID导致高钾血症）。\n"
            "  B) 冲突证据：≥2个发现指向不同的管理决策。KG或Workup轮必须承认并调和冲突"
            "（如INR超治疗范围+活动性GI出血vs.机械瓣→不能简单停用抗凝药）。\n"
            "  C) 序列依赖：某轮的临床问题只有在前一轮结果已知后才能确定——"
            "实际数值（而非仅上下文）驱动下一步"
            "（如eGFR<30→停药A；30–60→减量；>60→继续）。"
            "规划弧线使T0结果决定T1行动。\n"
            "弧线在4轮中必须跨越≥3种不同的FHIR资源类型。"
        ),
    },
}

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_PKG_DIR, "data")  # overridden at main() time with timestamp

# ── Scenario-specific difficulty constraints ──────────────────────────────────
# These override DIFFICULTY_CONSTRAINTS for their respective scenarios.
# Each entry maps difficulty level → constraint dict with constraint_text_en/zh.
SCENARIO_DIFFICULTY_CONSTRAINTS: dict[str, dict[int, dict]] = {

    "diagnostic_workup": {
        1: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 1 — FOUNDATIONAL DIFFERENTIAL (diagnostic_workup):\n"
                "Generate a simple two-candidate differential diagnosis. Requirements:\n"
                "- Involvement of ONE or TWO organ systems only.\n"
                "- T3 (final turn) should present exactly 2 competing diagnoses where the "
                "available lab/imaging data clearly favours one over the other.\n"
                "- No conflicting evidence between data sources. The differential is "
                "resolved by a single decisive finding (e.g., one lab value or one imaging result).\n"
                "- The KG turn states which diagnosis fits the pattern and briefly states why "
                "the alternative is less likely."
            ),
            "constraint_text_zh": (
                "难度等级1——基础鉴别诊断（diagnostic_workup）：\n"
                "生成简单的双候选鉴别诊断。要求：\n"
                "- 仅涉及一至两个器官系统。\n"
                "- T3（最后一轮）呈现两个竞争诊断，现有化验/影像数据明确倾向其中之一。\n"
                "- 数据来源之间无冲突证据，鉴别由单一决定性发现解决（如一项化验值或一份影像报告）。\n"
                "- KG轮说明哪个诊断与数据模式吻合，并简要说明为何另一个可能性较低。"
            ),
        },
        2: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 2 — MULTI-SOURCE DIFFERENTIAL WITH SCORING (diagnostic_workup):\n"
                "Generate a three-candidate differential requiring cross-referencing of "
                "two distinct data sources AND one validated clinical scoring system. Requirements:\n"
                "- At least 2-3 organ systems involved.\n"
                "- A Data Gathering turn MUST pair DiagnosticReport.search() with "
                "Observation.search() or Condition.search() — imaging findings must "
                "interact with lab values to narrow the differential.\n"
                "- The KG turn MUST apply one of the following validated scoring systems "
                "and cite the computed or estimated score:\n"
                "  * SIRS criteria (temp/HR/RR/WBC) — if infection vs. SIRS differential\n"
                "  * CURB-65 (BUN/RR/BP/age/confusion) — if pneumonia severity\n"
                "  * Child-Pugh score (bilirubin/albumin/PT/ascites/encephalopathy) — if liver disease\n"
                "  * Wells score (clinical features + D-dimer) — if DVT/PE differential\n"
                "- The answer must state the score value (or estimated range) and explain "
                "which diagnosis it supports. Do NOT use the scoring system name without computing it."
            ),
            "constraint_text_zh": (
                "难度等级2——多源鉴别诊断（diagnostic_workup）：\n"
                "生成需要交叉参考两类数据源的三候选鉴别诊断。要求：\n"
                "- 涉及至少2-3个器官系统。\n"
                "- 一个Workup轮必须将DiagnosticReport.search()与Observation.search()或"
                "Condition.search()配合——影像发现必须与化验值相互作用以缩小鉴别范围。\n"
                "- KG轮应用已知诊断标准或评分系统（如SIRS标准、Child-Pugh、Wells评分）"
                "对三个候选进行排序。\n"
                "- 单一数据点不足以解答，答案需综合至少两类来源。"
            ),
        },
        3: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 3 — CONFLICTING EVIDENCE DIFFERENTIAL (diagnostic_workup):\n"
                "Generate a case where evidence points in different directions. Requirements:\n"
                "- At least 4 organ systems or 4+ distinct FHIR resource types involved.\n"
                "- CONFLICTING EVIDENCE: some findings support Diagnosis A, others support "
                "Diagnosis B, and neither can be confirmed without additional targeted testing.\n"
                "- The KG turn MUST explicitly acknowledge the conflict, explain why both "
                "diagnoses remain plausible, and reason about which test would best discriminate.\n"
                "- The Write/Update turn (if present) orders 2-3 targeted diagnostic tests that "
                "directly address the identified information gaps, with specific clinical "
                "justification for each.\n"
                "- Patient interview MUST be used: set tool_source='mixed' on the KG or "
                "Data Gathering turn that assesses conflicting evidence, and call "
                "patient.get_symptom_history(). The patient's reported symptoms must either "
                "support or contradict the objective findings, adding a layer of diagnostic "
                "complexity. Do NOT leave tool_source='ehr' on all turns."
            ),
            "constraint_text_zh": (
                "难度等级3——证据冲突鉴别诊断（diagnostic_workup）：\n"
                "生成证据指向不同方向的案例。要求：\n"
                "- 至少涉及4个器官系统或4种以上不同FHIR资源类型。\n"
                "- 冲突证据：部分发现支持诊断A，其他发现支持诊断B，"
                "无法在不进行额外针对性检查的情况下确认任一诊断。\n"
                "- KG轮必须明确承认冲突，解释为何两个诊断都仍然可能，"
                "并推理哪种检查最能鉴别。\n"
                "- Action轮（如有）开具2-3项针对信息缺口的诊断检查，"
                "每项均有具体临床依据。\n"
                "- 必须使用病人访谈（get_symptom_history）——患者自述症状或支持或"
                "与客观发现矛盾，增加诊断复杂性。"
            ),
        },
    },

    "med_safety": {
        1: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 1 — SINGLE DRUG-LAB PAIR (med_safety):\n"
                "Generate a straightforward single drug-lab monitoring scenario. Requirements:\n"
                "- Exactly ONE drug-lab monitoring pair (e.g., warfarin + INR, "
                "metformin + creatinine, digoxin + potassium).\n"
                "- The lab value either clearly exceeds the safety threshold or is "
                "clearly within range — no ambiguity.\n"
                "- The KG turn states whether the drug is safe to continue, hold, or "
                "dose-adjust, with one-sentence justification.\n"
                "- No need for clinical formulae or multi-drug reasoning."
            ),
            "constraint_text_zh": (
                "难度等级1——单药-化验对（med_safety）：\n"
                "生成直接的单药监测场景。要求：\n"
                "- 恰好一个药-化验监测对（如华法林+INR、二甲双胍+肌酐、地高辛+血钾）。\n"
                "- 化验值要么明确超过安全阈值，要么明确在正常范围内——无歧义。\n"
                "- KG轮说明药物是否可以继续、暂停或调整剂量，并给出一句话依据。\n"
                "- 无需临床公式或多药推理。"
            ),
        },
        2: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 2 — FORMULA-BASED DOSE ADJUSTMENT (med_safety):\n"
                "Generate a scenario requiring a clinical formula to determine safety. "
                "Requirements:\n"
                "- The safety assessment MUST apply one of: Cockcroft-Gault equation "
                "(CrCl for renal dosing), eGFR-based dose adjustment, or "
                "Child-Pugh score (hepatic dosing).\n"
                "- A Data Gathering turn retrieves the required inputs "
                "(e.g., creatinine + age + weight for CrCl, or LFTs + albumin + INR "
                "for Child-Pugh).\n"
                "- The KG turn calculates or estimates the formula result and maps it "
                "to a specific dose adjustment recommendation (e.g., 'CrCl ~35 mL/min "
                "→ reduce metformin to 500 mg BID, monitor renal function monthly').\n"
                "- The answer must cite the calculated value, not just the raw lab."
            ),
            "constraint_text_zh": (
                "难度等级2——公式驱动的剂量调整（med_safety）：\n"
                "生成需要临床公式判断安全性的场景。要求：\n"
                "- 安全评估必须应用以下之一：Cockcroft-Gault公式（CrCl用于肾脏剂量调整）、"
                "基于eGFR的剂量调整，或Child-Pugh评分（肝功能剂量调整）。\n"
                "- 一个Workup轮检索所需输入值（如肌酐+年龄+体重用于CrCl，"
                "或LFTs+白蛋白+INR用于Child-Pugh）。\n"
                "- KG轮计算或估算公式结果，并映射到具体剂量调整建议"
                "（如'CrCl约35 mL/min→减少二甲双胍至500 mg BID，每月监测肾功能'）。\n"
                "- 答案必须引用计算值，而非仅引用原始化验值。"
            ),
        },
        3: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 3 — COMPETING SAFETY CONCERNS (med_safety):\n"
                "Generate a polypharmacy scenario where ≥2 drugs have conflicting "
                "safety requirements. Requirements:\n"
                "- At least 2 active drugs with monitoring thresholds that interact or "
                "compete (e.g., ACE-I raises K⁺ while the patient also takes "
                "spironolactone and NSAIDs — triple whammy for hyperkalemia).\n"
                "- CONFLICTING SAFETY: adjusting Drug A to address Lab A's abnormality "
                "worsens the safety concern for Drug B (or vice versa).\n"
                "- The KG turn must explicitly prioritise: which drug to address first, "
                "why, and what the acceptable risk tradeoff is.\n"
                "- Patient interview MUST be used: set tool_source='mixed' on the Data Gathering "
                "or KG turn that assesses drug safety, and call "
                "patient.get_medication_adherence(drug=<one of the conflicting drugs>). "
                "The patient must reveal NOT taking one drug as prescribed — this changes "
                "the safety assessment. Do NOT leave tool_source='ehr' on all turns.\n"
                "- The Write/Update turn (if present) addresses the highest-priority drug "
                "safety issue with a specific order."
            ),
            "constraint_text_zh": (
                "难度等级3——竞争性安全顾虑（med_safety）：\n"
                "生成≥2种药物存在冲突安全要求的多药并用场景。要求：\n"
                "- 至少2种有监测阈值的活跃药物相互作用或竞争"
                "（如ACE-I升高K⁺，同时患者服用螺内酯和NSAIDs——高钾血症三重打击）。\n"
                "- 冲突安全性：调整药物A以处理化验A的异常会加重药物B的安全顾虑（反之亦然）。\n"
                "- KG轮必须明确优先级：首先处理哪种药物，原因是什么，"
                "以及可接受的风险权衡是什么。\n"
                "- 病人访谈（get_medication_adherence）必须揭示患者未按处方服用其中一种药物——"
                "这改变了安全评估和建议行动。\n"
                "- Action轮（如有）用具体医嘱处理最高优先级的药物安全问题。"
            ),
        },
    },

    "treatment_response": {
        1: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 1 — SINGLE MARKER RESPONSE (treatment_response):\n"
                "Generate a clear-cut treatment response assessment using one infection "
                "marker. Requirements:\n"
                "- Track ONE primary infection marker (WBC or temperature) over 2+ time "
                "points.\n"
                "- The trend is unambiguous: clearly improving (normalising) or clearly "
                "worsening (rising despite treatment).\n"
                "- One antibiotic class is involved. The KG turn states whether to "
                "continue, de-escalate, or escalate — with direct reference to the "
                "trend direction and the current value vs. normal range.\n"
                "- No conflicting markers; no multi-drug decisions."
            ),
            "constraint_text_zh": (
                "难度等级1——单指标治疗反应（treatment_response）：\n"
                "使用单一感染指标生成清晰的治疗反应评估。要求：\n"
                "- 追踪一个主要感染指标（WBC或体温）≥2个时间点。\n"
                "- 趋势明确：清晰改善（正常化）或清晰恶化（治疗下仍上升）。\n"
                "- 涉及一类抗生素。KG轮说明是否继续、降阶梯或升级治疗——"
                "直接参考趋势方向及当前值与正常范围的比较。\n"
                "- 无冲突指标；无多药决策。"
            ),
        },
        2: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 2 — DIVERGING MARKERS (treatment_response):\n"
                "Generate a scenario where 2 infection markers diverge, requiring "
                "interpretation of which to trust. Requirements:\n"
                "- Track 2 infection markers (e.g., WBC + CRP, or neutrophils + "
                "temperature) that move in DIFFERENT directions (one improving, "
                "one still elevated or worsening).\n"
                "- A Data Gathering turn pairs both markers with MedicationAdministration.search() "
                "to verify the antibiotic was actually administered.\n"
                "- The KG turn must reason about the discordance: is the patient "
                "improving (use the normalising marker) or still failing (use the "
                "elevated marker)? Cite a clinical rationale for weighting one marker "
                "over the other (e.g., 'CRP lags 24–48 h behind WBC normalisation').\n"
                "- Decision: continue vs. extend duration vs. narrow spectrum."
            ),
            "constraint_text_zh": (
                "难度等级2——指标背离（treatment_response）：\n"
                "生成两个感染指标向不同方向变化的场景，需要解读哪个指标更可信。要求：\n"
                "- 追踪2个感染指标（如WBC+CRP，或中性粒细胞+体温）向不同方向变化"
                "（一个改善，一个仍升高或恶化）。\n"
                "- 一个Workup轮将两个指标与MedicationAdministration.search()配对，"
                "验证抗生素是否确实已给药。\n"
                "- KG轮必须推理矛盾：患者是在改善（依赖正常化的指标）还是治疗失败"
                "（依赖升高的指标）？引用临床依据说明为何偏重某一指标"
                "（如'CRP滞后WBC正常化24-48小时'）。\n"
                "- 决策：继续/延长疗程/缩窄抗菌谱。"
            ),
        },
        3: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 3 — CONFLICTING EVIDENCE WITH PATIENT CONTEXT "
                "(treatment_response):\n"
                "Generate a complex case where objective markers conflict with patient-"
                "reported symptoms, requiring synthesis across data sources. Requirements:\n"
                "- ≥3 infection markers involved with mixed trends (some improving, "
                "some not).\n"
                "- PATIENT INTERVIEW is MANDATORY: get_symptom_history() reveals that "
                "the patient STILL reports fever/pain/weakness despite labs suggesting "
                "improvement — or vice versa (labs worsening but patient feels better).\n"
                "- At least 2 antibiotic classes are involved; assessment must address "
                "whether the current coverage is adequate given the clinical picture.\n"
                "- The KG turn MUST apply a validated organ dysfunction scoring system "
                "to quantify treatment response:\n"
                "  * SOFA score trend (if platelets, bilirubin, creatinine available): "
                "compute or estimate the change from admission to current to show "
                "organ recovery or deterioration.\n"
                "  * qSOFA (RR>22, GCS<15, SBP<100): assess whether sepsis criteria "
                "are still met despite antibiotic treatment.\n"
                "  * SIRS criteria: determine whether the patient still meets systemic "
                "inflammatory response criteria.\n"
                "- The answer must cite the score numerically and explain whether it "
                "supports de-escalation, maintenance, or escalation of treatment."
            ),
            "constraint_text_zh": (
                "难度等级3——证据冲突与病人背景（treatment_response）：\n"
                "生成客观指标与患者自述症状冲突的复杂案例，需要跨数据源综合。要求：\n"
                "- 涉及≥3个感染指标，趋势不一（部分改善，部分未改善）。\n"
                "- 病人访谈必须使用：get_symptom_history()揭示患者在化验提示改善的情况下"
                "仍报告发热/疼痛/乏力——或反之（化验恶化但患者感觉好转）。\n"
                "- 涉及至少2类抗生素；评估必须结合临床表现判断当前覆盖范围是否足够。\n"
                "- KG轮必须调和客观与主观数据：解释差异并说明哪方驱动临床决策及原因。\n"
                "- 决策需在以下选项中选择：降阶梯（化验好）、升级（症状持续）"
                "或维持观察（等待培养结果）。"
            ),
        },
    },

    "discharge_planning": {
        1: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 1 — STRAIGHTFORWARD DISCHARGE (discharge_planning):\n"
                "Generate a simple discharge plan for a patient with a single primary "
                "diagnosis and stable condition. Requirements:\n"
                "- One primary discharge diagnosis; medications at discharge closely "
                "match admission medications (no complex reconciliation needed).\n"
                "- Patient is cognitively intact and understands their condition "
                "(high health literacy expected from patient interview).\n"
                "- The Write/Update turn creates exactly ONE write order: either a medication "
                "continuation order or a single routine follow-up referral.\n"
                "- No medication safety flags, no functional limitations that complicate "
                "discharge."
            ),
            "constraint_text_zh": (
                "难度等级1——简单出院计划（discharge_planning）：\n"
                "为单一主要诊断、病情稳定的患者生成简单出院计划。要求：\n"
                "- 一个主要出院诊断；出院用药与入院用药基本相同（无需复杂核对）。\n"
                "- 患者认知完整，理解自身病情（病人访谈中预期高医学素养）。\n"
                "- Action轮创建恰好一条写入医嘱：要么是药物继续医嘱，要么是单次常规随访转介。\n"
                "- 无药物安全警告，无功能限制复杂化出院。"
            ),
        },
        2: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 2 — MEDICATION RECONCILIATION WITH CLINICAL SCORING "
                "(discharge_planning):\n"
                "Generate a discharge plan requiring identification of medication gaps "
                "AND application of a clinical risk score to guide treatment decisions. Requirements:\n"
                "- At least ONE medication change between admission and discharge "
                "(new drug added, drug stopped, or dose changed) that requires "
                "reconciliation and patient education.\n"
                "- A Data Gathering turn pairs MedicationRequest.search() with CarePlan.search() "
                "or DocumentReference.search() to identify the gap.\n"
                "- Patient interview (get_medication_adherence) reveals partial "
                "adherence to one medication — this must be addressed in the plan.\n"
                "- The KG turn MUST apply one of the following scoring systems to justify "
                "the discharge medication decision:\n"
                "  * CHA₂DS₂-VASc score — if the patient has atrial fibrillation and "
                "anticoagulation is being started/continued/stopped at discharge. "
                "Compute the score and state the resulting stroke risk category.\n"
                "  * eGFR / Cockcroft-Gault — if renal dose adjustment is needed for "
                "discharge medications. Compute CrCl and map to dose recommendation.\n"
                "  * Child-Pugh score — if the patient has liver disease affecting "
                "drug metabolism. Estimate the score and its dosing implications.\n"
                "- The Write/Update turn creates 2 write orders driven by the score result."
            ),
            "constraint_text_zh": (
                "难度等级2——含缺口的用药核对（discharge_planning）：\n"
                "生成需要识别和解决用药缺口的出院计划。要求：\n"
                "- 入院至出院期间至少一项用药变化（新增、停用或剂量调整），"
                "需要核对和患者教育。\n"
                "- 一个Workup轮将MedicationRequest.search()与CarePlan.search()或"
                "DocumentReference.search()配对以识别缺口。\n"
                "- 病人访谈（get_medication_adherence）揭示对某一药物的部分依从——"
                "计划中必须解决。\n"
                "- Action轮创建2条写入医嘱：一条药物医嘱AND一条服务转介"
                "（如药房辅导或家庭护理）。\n"
                "- KG轮必须解释每项用药变化的临床依据。"
            ),
        },
        3: {
            "constraint_text_en": (
                "DIFFICULTY LEVEL 3 — COMPLEX POLYPHARMACY DISCHARGE WITH SOCIAL BARRIERS "
                "(discharge_planning):\n"
                "Generate a complex discharge scenario with medication safety concerns, "
                "functional limitations, and social/adherence barriers. Requirements:\n"
                "- At least 5 discharge medications with ≥1 safety flag needed "
                "(drug interaction, renal dose adjustment, or high-risk drug).\n"
                "- Patient interview MUST be used: get_functional_status() reveals "
                "functional limitations (e.g., cannot manage stairs, cannot self-"
                "administer injections), AND get_medication_adherence() or "
                "get_social_history() reveals a social barrier (lives alone, no "
                "caregiver, limited English).\n"
                "- The KG turn synthesises clinical findings with social context to "
                "determine appropriate discharge disposition (home vs. skilled nursing "
                "vs. rehabilitation).\n"
                "- The Write/Update turn creates ALL THREE write tool types: "
                "MedicationRequest.create (medication order or adjustment) + "
                "ServiceRequest.create (home health or SNF referral) + "
                "Flag.create (safety alert for receiving care team).\n"
                "- The flag must cite specific clinical values and the specific risk "
                "they represent."
            ),
            "constraint_text_zh": (
                "难度等级3——含社会障碍的复杂多药出院（discharge_planning）：\n"
                "生成含药物安全顾虑、功能限制和社会/依从障碍的复杂出院场景。要求：\n"
                "- 至少5种出院药物，需要≥1条安全警告"
                "（药物相互作用、肾功能剂量调整或高风险药物）。\n"
                "- 必须使用病人访谈：get_functional_status()揭示功能限制"
                "（如无法上楼梯、无法自行注射），"
                "AND get_medication_adherence()或get_social_history()揭示社会障碍"
                "（独居、无照护者、语言障碍）。\n"
                "- KG轮综合临床发现与社会背景，确定适当的出院去向"
                "（回家/技术护理机构/康复机构）。\n"
                "- Action轮创建全部三种写入工具：MedicationRequest.create（药物医嘱或调整）+"
                "ServiceRequest.create（居家护理或技术护理转介）+"
                "Flag.create（接收护理团队的安全警报）。\n"
                "- 标记必须引用具体临床数值及其代表的具体风险。"
            ),
        },
    },
}


def get_difficulty_constraints(scenario: str, difficulty: int) -> dict | None:
    """
    Return difficulty constraints for a given scenario and difficulty level.
    Scenario-specific constraints take precedence over generic DIFFICULTY_CONSTRAINTS.
    """
    scenario_specific = SCENARIO_DIFFICULTY_CONSTRAINTS.get(scenario, {})
    if difficulty in scenario_specific:
        return scenario_specific[difficulty]
    return DIFFICULTY_CONSTRAINTS.get(difficulty)


# ── Tool coverage tracker (Solution B) ───────────────────────────────────────

# Tools never counted as "underused" — system calls, write tools, patient tools
_COVERAGE_EXCLUDE = {
    "prepare_to_answer",
    "ask_user_for_required_parameters",
    "MedicationRequest.create",
    "ServiceRequest.create",
    "Flag.create",
    "patient.get_medication_adherence",
    "patient.get_symptom_history",
    "patient.get_functional_status",
    "patient.get_social_history",
    "patient.get_chief_complaint",
    "patient.get_pain_assessment",
}


def compute_tool_coverage(
    out_dir: str,
    scenario: str,
    available_tool_names: list[str],
    underuse_ratio: float = 0.25,
) -> list[str]:
    """
    Scan existing JSONL output for the scenario and return EHR tools whose call
    count is below `underuse_ratio * max_tool_count`.  These are passed to the
    session planner as a coverage hint so it exercises them more often.

    Args:
        out_dir:             Directory containing <scenario>.jsonl
        scenario:            Scenario name
        available_tool_names: Full list of tool names for this scenario/domain
        underuse_ratio:      Tools called < ratio * max_calls are flagged (default 0.25)

    Returns:
        Sorted list of underused EHR tool names (empty if no existing data).
    """
    from collections import Counter
    ehr_tools = [t for t in available_tool_names if t not in _COVERAGE_EXCLUDE]
    counts: Counter = Counter({t: 0 for t in ehr_tools})

    jsonl_path = os.path.join(out_dir, f"{scenario}.jsonl")
    if not os.path.exists(jsonl_path):
        # No existing data — all tools are equally "unseen"; return nothing
        # (don't bias first entries before any data is generated)
        return []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                for turn_actions in entry.get("answer_list", []):
                    for act in turn_actions:
                        name = act.get("action", {}).get("name", "")
                        if name in counts:
                            counts[name] += 1
            except Exception:
                pass

    if not counts:
        return []

    max_count = max(counts.values(), default=0)
    if max_count == 0:
        return []

    threshold = max(1, int(max_count * underuse_ratio))
    underused = sorted(t for t, c in counts.items() if c < threshold)
    return underused

PERSONAS = [
    {
        "health_literacy": "low",
        "adherence": "poor",
        "anxiety_level": "high",
        "info_completeness": "critical_withheld",
    },
    {
        "health_literacy": "high",
        "adherence": "good",
        "anxiety_level": "low",
        "info_completeness": "full",
    },
]


def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def generate_scenario(
    scenario: str,
    n: int = N_PER_SCENARIO,
    subject_id: int | None = None,
    sequence_idx: int | None = None,
    preassigned_seq_indices: list[int] | None = None,
    bilingual: bool = True,
    tool_set: str = "fhir",
    scenario_constraints: dict | None = None,
    difficulty: int | None = None,
    start_index: int = 0,
    subtype_counter: dict[str, int] | None = None,
    require_patient_turn: bool = False,
) -> list:
    """Generate n entries for one clinical scenario. Returns list of entries."""
    task_domain = SCENARIO_TO_DOMAIN.get(scenario, "LabInterp")
    out_path = os.path.join(OUT_DIR, f"{scenario}.jsonl")

    # Load existing IDs to avoid duplication
    existing_ids: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    # Select patients
    if subject_id is not None:
        all_patients = select_patients(task_domain, split="test", n=500, seed=SEED)
        patients = [(sid, hid) for sid, hid in all_patients if sid == subject_id]
        if not patients:
            logger.warning(
                f"subject_id {subject_id} not found in test split for {task_domain}, "
                "trying train split"
            )
            all_patients = select_patients(task_domain, split="train", n=500, seed=SEED)
            patients = [(sid, hid) for sid, hid in all_patients if sid == subject_id]
        if not patients:
            raise ValueError(f"subject_id {subject_id} not found for domain {task_domain}")
        logger.info(
            f"[{scenario}] Using subject_id={subject_id}, hadm_id={patients[0][1]}"
        )
    else:
        pool_size = max(n * 20, 50)
        this_difficulty = difficulty  # may be None (cycling)
        patients = _get_patient_pool(scenario, this_difficulty, task_domain, pool_size)
    logger.info(f"[{scenario}] {len(patients)} candidate patients in pool")

    # Compute tool coverage hint once before the generation loop (Solution B).
    # Re-computed each time generate_scenario() is called so it reflects newly
    # written entries within the same run (progressive coverage improvement).
    from physassistbench.tools.tool_schemas import get_fhir_tools_for_task as _get_fhir
    _all_tool_names = [t["function"]["name"] for t in _get_fhir(task_domain)]
    tools_to_prioritize = compute_tool_coverage(OUT_DIR, scenario, _all_tool_names)
    if tools_to_prioritize:
        logger.info(f"[{scenario}] Underused tools → will hint planner: {tools_to_prioritize}")
    else:
        logger.info(f"[{scenario}] Tool coverage balanced — no priority hint injected")

    entries = []
    generated = 0
    entry_index = start_index

    # Assign arc indices: pre-assigned > pinned > sample from eligible pool
    if preassigned_seq_indices is not None:
        seq_indices = preassigned_seq_indices
    elif sequence_idx is not None:
        seq_indices = [sequence_idx] * n
    else:
        eligible = get_eligible_arc_indices(difficulty)
        seq_indices = random.sample(eligible, min(n, len(eligible)))
        if n > len(eligible):
            seq_indices = seq_indices + random.choices(eligible, k=n - len(eligible))

    # Max patient attempts per entry slot — prevents infinite loops when EHR data
    # quality is low for a scenario (default: try up to 20 different patients per slot)
    _MAX_ATTEMPTS_PER_SLOT = int(os.environ.get("EHR_MAX_PATIENTS_PER_SLOT", "20"))
    _slot_attempts: dict[int, int] = {}   # entry_index → attempts so far

    for sid, hid in patients:
        if generated >= n:
            break

        # Cycle difficulty 1→2→3→1→2→3… unless pinned by caller (needed for ID before generate call)
        this_difficulty = difficulty if difficulty is not None else (entry_index % 3 + 1)
        entry_id = f"physassistbench_{task_domain}_{scenario}_{entry_index}_L{this_difficulty}"
        if entry_id in existing_ids:
            logger.info(f"  Skipping {entry_id} (already exists)")
            generated += 1
            entry_index += 1
            continue

        persona = PERSONAS[entry_index % len(PERSONAS)]
        session_id = f"physassistbench_{scenario}_{entry_index}"
        this_seq_idx = seq_indices[entry_index % len(seq_indices)]

        logger.info(
            f"  [{scenario}] entry {entry_index}: "
            f"subject={sid} hadm={hid} seq={this_seq_idx} session={session_id}"
        )

        reset_all_sessions()
        register_session(session_id, sid, persona, language="en")

        difficulty_constraints = get_difficulty_constraints(scenario, this_difficulty)

        try:
            entry = generate_one_entry_v2(
                task_domain=task_domain,
                subject_id=sid,
                hadm_id=hid,
                entry_index=entry_index,
                scenario=scenario,
                sequence_idx=this_seq_idx,
                n_turns=N_TURNS,
                session_id=session_id,
                persona=persona,
                language="en",
                bilingual=bilingual,
                tool_set=tool_set,
                scenario_constraints=scenario_constraints,
                difficulty=this_difficulty,
                difficulty_constraints=difficulty_constraints,
                subtype_counter=subtype_counter,
                tools_to_prioritize=tools_to_prioritize or None,
                require_patient_turn=require_patient_turn,
            )
        except Exception as exc:
            logger.error(
                f"  [FAILED] {entry_id}: {exc} "
                f"(patient {sid} — trying next candidate)"
            )
            _slot_attempts[entry_index] = _slot_attempts.get(entry_index, 0) + 1
            if _slot_attempts[entry_index] >= _MAX_ATTEMPTS_PER_SLOT:
                logger.warning(
                    f"  [SKIP] {entry_id}: reached {_MAX_ATTEMPTS_PER_SLOT} patient attempts "
                    "— giving up on this slot and moving on."
                )
                generated += 1
                entry_index += 1
            continue

        if entry is None:
            logger.error(
                f"  [FAILED] {entry_id}: generate_one_entry_v2 returned None "
                f"(patient {sid} has no suitable EHR data — trying next candidate)"
            )
            _slot_attempts[entry_index] = _slot_attempts.get(entry_index, 0) + 1
            if _slot_attempts[entry_index] >= _MAX_ATTEMPTS_PER_SLOT:
                logger.warning(
                    f"  [SKIP] {entry_id}: reached {_MAX_ATTEMPTS_PER_SLOT} patient attempts "
                    "— giving up on this slot and moving on."
                )
                generated += 1
                entry_index += 1
            continue

        rt = get_session(session_id)
        entry["patient_agent_annotations"] = rt.get_annotation_store()
        entry["persona_config"] = persona
        entry["session_id"] = session_id
        entry["generated_at"] = datetime.utcnow().isoformat()

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(entry), ensure_ascii=False) + "\n")

        # Save session plan to a dedicated JSON file
        session_plan = entry.get("session_plan")
        if session_plan:
            plans_dir = os.path.join(OUT_DIR, "session_plans")
            os.makedirs(plans_dir, exist_ok=True)
            plan_record = {
                "entry_id":     entry.get("id", entry_id),
                "subject_id":   sid,
                "hadm_id":      hid,
                "scenario":     scenario,
                "task_sequence": entry.get("task_sequence", []),
                "session_plan": session_plan,
            }
            plan_path = os.path.join(plans_dir, f"{entry.get('id', entry_id)}_plan.json")
            with open(plan_path, "w", encoding="utf-8") as pf:
                json.dump(plan_record, pf, ensure_ascii=False, indent=2)

        n_turns = len(entry.get("answer_list", []))
        has_zh = "tasks_zh" in entry
        sources = entry.get("tool_sources", [])
        logger.info(
            f"  [OK] {entry['id']}  turns={n_turns}  bilingual={has_zh}  "
            f"tool_sources={sources}  tool_set={tool_set}  → {out_path}"
        )

        entries.append(entry)
        generated += 1
        entry_index += 1

    logger.info(f"[{scenario}] Done: {generated}/{n} entries written")
    return entries


def main(
    scenarios: list[str] | None = None,
    n: int = N_PER_SCENARIO,
    subject_id: int | None = None,
    sequence_idx: int | None = None,
    bilingual: bool = True,
    seed: int = SEED,
    tool_set: str = "fhir",
    difficulty: int | None = None,
    out_dir: str | None = None,
    require_patient_turn: bool = False,
    no_constraints: bool = False,
):
    global OUT_DIR
    if out_dir:
        OUT_DIR = out_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        OUT_DIR = os.path.join(_PKG_DIR, f"data_{timestamp}")
    random.seed(seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    target_scenarios = scenarios or SCENARIO_NAMES

    logger.info("=" * 60)
    logger.info(
        f"physassistbench — Generating {n} entries × {len(target_scenarios)} scenarios "
        f"= {n * len(target_scenarios)} total"
    )
    logger.info(f"Tool set:   {tool_set.upper()} ({'FHIR R4 tools' if tool_set == 'fhir' else 'Legacy custom tools'})")
    logger.info(f"Task types: Information Lookup / Data Gathering / Clinical Reasoning / Action")
    logger.info(f"Subtypes:   Dynamic (feasibility-first, counter-balanced EA/PE/AU/AE)")
    logger.info(f"Sequences:  36 arcs (27 EHR + 9 Action-final)")
    logger.info(f"Bilingual:  {bilingual}")
    logger.info(f"Constraints: {'DISABLED (free LLM tool selection)' if no_constraints else 'ENABLED (scenario-specific)'}")
    logger.info(f"Patient turn: {'REQUIRED (forced if planner omits)' if require_patient_turn else 'OPTIONAL (LLM-driven)'}")
    logger.info("=" * 60)

    # Pre-sample arc indices globally — restricted to difficulty-eligible arcs
    total = n * len(target_scenarios)
    eligible = get_eligible_arc_indices(difficulty)
    global_seq_indices = random.sample(eligible, min(total, len(eligible)))
    if total > len(eligible):
        global_seq_indices += random.choices(eligible, k=total - len(eligible))

    # Shared counter across all scenarios so the EA/PE/AU/AE distribution is
    # balanced over the entire generated dataset, not just within one scenario.
    subtype_counter: dict[str, int] = {st: 0 for st in SUBTYPES}

    all_entries = []
    for s_idx, scenario in enumerate(target_scenarios):
        logger.info(f"\n{'='*60}\nScenario: {scenario}\n{'='*60}")
        # Slice this scenario's chunk from the global pool
        scenario_seqs = (
            global_seq_indices[s_idx * n : (s_idx + 1) * n]
            if sequence_idx is None
            else None
        )
        entries = generate_scenario(
            scenario=scenario,
            n=n,
            subject_id=subject_id,
            sequence_idx=sequence_idx,
            preassigned_seq_indices=scenario_seqs,
            bilingual=bilingual,
            tool_set=tool_set,
            scenario_constraints=None if no_constraints else SCENARIO_CONSTRAINTS.get(scenario),
            difficulty=difficulty,
            subtype_counter=subtype_counter,
            require_patient_turn=require_patient_turn,
        )
        all_entries.extend(entries)
        logger.info(
            f"  Subtype counts so far: "
            + ", ".join(f"{st}={subtype_counter[st]}" for st in SUBTYPES)
        )

    logger.info(f"\n{'='*60}")
    logger.info(
        f"Generation complete: {len(all_entries)} entries "
        f"across {len(target_scenarios)} scenarios"
    )
    logger.info(f"Output directory: {OUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="physassistbench: Generate benchmark entries with FHIR R4 tools"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=SCENARIO_NAMES,
        help="Clinical scenarios to generate (default: all 3)",
    )
    parser.add_argument(
        "--n", type=int, default=N_PER_SCENARIO,
        help="Entries per scenario (default: 2)",
    )
    parser.add_argument(
        "--subject_id", type=int, default=None,
        help="Pin generation to one MIMIC-IV patient subject_id",
    )
    parser.add_argument(
        "--sequence_idx", type=int, default=None,
        help="Pin arc index (0-26, T0=Information Lookup fixed). Default: sample from eligible pool.",
    )
    parser.add_argument(
        "--no_bilingual", action="store_true",
        help="Disable bilingual output (English only)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--tool_set", type=str, default="fhir", choices=["fhir", "legacy"],
        help="Tool set to use: 'fhir' (default, FHIR R4) or 'legacy' (original custom tools)",
    )
    parser.add_argument(
        "--difficulty", type=int, default=None, choices=[1, 2, 3],
        help="Pin all entries to a fixed difficulty level (1/2/3). Default: cycle 1→2→3.",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: auto-generated data_<timestamp>/).",
    )
    parser.add_argument(
        "--require_patient", action="store_true", default=False,
        help=(
            "Guarantee at least one patient interview (mixed) turn per session. "
            "When the session planner does not voluntarily assign a patient turn, "
            "the last eligible EHR turn is forced to tool_source='mixed'. "
            "Default: off (patient turns are LLM-driven soft decisions only)."
        ),
    )
    parser.add_argument(
        "--generation_model", type=str, default=None,
        help=(
            "Override the LLM used for data generation (defined in physassistbench/model_configs.yaml). "
            "Default: deepseek-v4-flash. Example: --generation_model gpt-4.1-mini"
        ),
    )
    parser.add_argument(
        "--no_constraints", action="store_true", default=False,
        help=(
            "Disable all scenario-specific tool constraints (allowed/forbidden resources, "
            "workup patterns, clinical focus). The session planner uses only the EHR snapshot "
            "and general diversity rules to freely select tools. "
            "Improves tool coverage at the cost of scenario structural distinctiveness."
        ),
    )
    args = parser.parse_args()
    # Configure generation model if overridden
    if args.generation_model:
        from physassistbench.eval_runner import load_model_config, _build_client
        from physassistbench.pipeline.agents.llm_client import configure_generation_model
        model_cfg = load_model_config(args.generation_model)
        api_key = os.environ.get(model_cfg["api_key_env"], "")
        if not api_key:
            parser.error(f"API key env var '{model_cfg['api_key_env']}' not set for model '{args.generation_model}'")
        configure_generation_model(model_cfg, api_key)
        logger.info(f"Generation model overridden: {model_cfg['description']} (model_id={model_cfg['model_id']})")

    main(
        scenarios=args.scenarios,
        n=args.n,
        subject_id=args.subject_id,
        sequence_idx=args.sequence_idx,
        bilingual=not args.no_bilingual,
        seed=args.seed,
        tool_set=args.tool_set,
        difficulty=args.difficulty,
        out_dir=args.out_dir,
        require_patient_turn=args.require_patient,
        no_constraints=args.no_constraints,
    )
