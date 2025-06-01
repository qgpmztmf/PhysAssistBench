"""
physassistbench/pipeline/scenarios.py — 11 clinical scenarios replacing the old 6 task domains.

Each scenario specifies:
  - description: clinical context for grounding fact sampling
  - grounding_domains: which original data_sampler domains to delegate to
    (reuses existing sample_grounding_facts() infrastructure)

See docs/benchmark_redesign_integrated.md §2.3 for full definitions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physassistbench.pipeline.data_sampler import sample_grounding_facts as _original_sample

# ── Scenario definitions ──────────────────────────────────────────────────────

CLINICAL_SCENARIOS: dict[str, dict] = {
    "infection_management": {
        "description": (
            "感染性疾病管理；"
            "疾病：脓毒症（肺源/泌尿源/腹腔源/导管相关）、社区获得性肺炎、院内肺炎、"
            "尿路感染、腹腔感染、皮肤软组织感染、骨髓炎、感染性心内膜炎、脑膜炎/脑炎；"
            "关注：血培养/PCT/抗生素选择与降阶梯/感染源控制/疗程"
        ),
        "grounding_domains": ["LabInterp", "DiagCode"],
    },
    "critical_care": {
        "description": (
            "ICU 危重症管理；"
            "疾病：脓毒症休克、ARDS、多器官功能障碍（MODS）、心源性休克、"
            "失血性休克、梗阻性休克（张力性气胸/大面积 PE）、机械通气依赖；"
            "关注：血管活性药滴定/液体复苏/器官保护/SOFA 评分/镇静镇痛方案"
        ),
        "grounding_domains": ["ICUReasoning", "LabInterp"],
    },
    "acute_cardiac": {
        "description": (
            "急性心血管事件；"
            "疾病：ACS（STEMI/NSTEMI/UA）、急性心衰失代偿、恶性心律失常（新发房颤/室速/室颤）、"
            "高血压急症/亚急症、大面积肺栓塞、主动脉夹层、心包积液/心脏压塞；"
            "关注：再灌注时机/抗栓方案/血流动力学稳定性/启动靶向治疗时机"
        ),
        "grounding_domains": ["ICUReasoning", "LabInterp"],
    },
    "metabolic_electrolyte": {
        "description": (
            "代谢与电解质紊乱；"
            "疾病：DKA、HHS、低钾/高钾血症、低钠/高钠血症、低镁/低磷血症、"
            "代谢性酸中毒/碱中毒、甲状腺危象、肾上腺皮质功能不全、乳酸酸中毒；"
            "关注：纠正速度/补充方案/病因溯源/复查频率/并发症预防"
        ),
        "grounding_domains": ["LabInterp"],
    },
    "gi_hepatic": {
        "description": (
            "消化与肝胆疾病；"
            "疾病：上消化道出血（消化性溃疡/食管胃底静脉曲张破裂）、下消化道出血、"
            "急性胰腺炎（轻中重度分级）、IBD 急性发作（克罗恩病/溃疡性结肠炎）、"
            "药物性肝损伤、急性肝衰竭、胆道感染（急性胆囊炎/胆管炎）；"
            "关注：出血风险评分/内镜时机/液体复苏/营养支持/外科会诊指征"
        ),
        "grounding_domains": ["LabInterp", "DiagCode"],
    },
    "respiratory": {
        "description": (
            "呼吸系统疾病；"
            "疾病：COPD 急性加重、哮喘急性发作、急性呼吸衰竭（Ⅰ型/Ⅱ型）、"
            "肺栓塞（次大面积/大面积）、气胸、胸腔积液、间质性肺疾病急性加重；"
            "关注：氧疗目标/无创-有创通气时机/支气管扩张剂/抗凝决策"
        ),
        "grounding_domains": ["ICUReasoning", "LabInterp"],
    },
    "neurology": {
        "description": (
            "神经系统疾病；"
            "疾病：缺血性卒中/TIA、脑出血、蛛网膜下腔出血、癫痫持续状态、"
            "ICU 谵妄/代谢性脑病、吉兰-巴雷综合征、重症肌无力危象；"
            "关注：溶栓/取栓时间窗/神经功能评分/抗癫痫药物选择/镇静深度管理"
        ),
        "grounding_domains": ["ICUReasoning", "DiagCode"],
    },
    "chronic_disease_mgmt": {
        "description": (
            "慢性病综合管理；"
            "疾病：T2DM（血糖目标/胰岛素方案/低血糖处理/并发症筛查）、"
            "CKD（分期评估/肾毒性药物调整/透析指征）、慢性心衰（容量管理/利尿剂滴定/GDMT）、"
            "COPD 稳定期（吸入药物阶梯/肺康复）、肝硬化（腹水/肝性脑病/食管静脉曲张二级预防）、"
            "高血压（达标评估/联合用药）、冠心病（二级预防/抗栓方案）、"
            "房颤（心率/心律控制/抗凝）；"
            "关注：靶值达标/用药优化/并发症预防/患者教育"
        ),
        "grounding_domains": ["DiagCode", "MedRecon"],
    },
    "medication_review": {
        "description": (
            "用药核查与药学评估；"
            "场景：多重用药（≥5 种）、高危药物监测（抗凝/胰岛素/地高辛/氨基糖苷类）、"
            "肾功能/肝功能剂量调整（eGFR 分级）、药物相互作用（P450/QT 延长/出血风险叠加）、"
            "用药依从性评估、入院带药核对（medication reconciliation）、停药指征；"
            "关注：处方核对/替代方案/监测参数/患者教育"
        ),
        "grounding_domains": ["MedRecon"],
    },
    "lab_interpretation": {
        "description": (
            "化验解读与趋势分析；"
            "场景：单指标危急值处理、多指标联合解读（感染 panel/肝功/肾功/凝血/血气/电解质）、"
            "动态趋势判断（好转/恶化/平台期）、化验与临床不一致的鉴别、"
            "肿瘤标志物/自身抗体/培养结果解读；"
            "关注：异常程度分级/趋势方向/鉴别诊断/复查时机/下一步检查"
        ),
        "grounding_domains": ["LabInterp"],
    },
    "discharge_planning": {
        "description": (
            "出院规划与过渡期管理；"
            "场景：出院诊断核对（主诊断/并发症/合并症）、出院带药核查（新增/停用/剂量调整）、"
            "随访安排（科室/时间/监测指标）、患者教育（用药/饮食/活动限制/警示症状）、"
            "高风险再入院因素评估（社会支持/依从性/经济因素）；"
            "关注：出院核查清单/过渡期用药安全/患者理解度评估"
        ),
        "grounding_domains": ["DischargePlan", "MedRecon"],
    },
    # ── generate_all.py primary scenarios ─────────────────────────────────────
    "diagnostic_workup": {
        "description": (
            "Multi-system diagnostic workup: integrate heterogeneous data (labs, "
            "radiology/pathology reports, prior procedures, diagnosis history) to "
            "generate differential diagnoses and recommend targeted diagnostic testing. "
            "Core clinical reasoning: hypothesis generation and evidence evaluation "
            "across multiple organ systems."
        ),
        "description_zh": (
            "多系统诊断推理：整合异质性数据（化验、影像/病理报告、既往操作、诊断历史），"
            "生成鉴别诊断并推荐有针对性的诊断检查。"
            "核心临床推理：跨多器官系统的假说生成与证据评估。"
        ),
        "grounding_domains": ["LabInterp", "DiagCode"],
    },
    "lab_trend": {
        "description": (
            "Pure lab trend analysis: retrieve and interpret time-series lab values "
            "(CBC, BMP, liver panel, cardiac enzymes, coagulation) to identify rising, "
            "falling, or stable trends and their clinical significance."
        ),
        "description_zh": (
            "纯实验室趋势分析：检索并解读时序化验值（CBC、BMP、肝功能、心肌酶谱、凝血），"
            "识别上升、下降或稳定趋势及其临床意义。"
        ),
        "grounding_domains": ["LabInterp"],
    },
    "med_safety": {
        "description": (
            "Medication safety review: correlate abnormal lab values with active drug orders "
            "to identify dose adjustment needs, contraindications, or drug-lab interactions "
            "(e.g. renal function + nephrotoxins, electrolytes + related drugs)."
        ),
        "description_zh": (
            "药物安全审查：将异常化验值与活跃药物医嘱相关联，"
            "识别剂量调整需求、禁忌证或药-化验交互（如肾功能+肾毒性药物、电解质+相关药物）。"
        ),
        "grounding_domains": ["MedRecon", "LabInterp"],
    },
    "treatment_response": {
        "description": (
            "Treatment response monitoring: track infection markers (WBC, CRP, PCT, lactate) "
            "alongside antibiotic administration records to assess treatment efficacy and "
            "guide de-escalation, escalation, or discharge decisions."
        ),
        "description_zh": (
            "治疗反应监测：跟踪感染指标（WBC、CRP、PCT、乳酸）与抗生素给药记录，"
            "评估治疗效果并指导降阶梯、升级或出院决策。"
        ),
        "grounding_domains": ["ICUReasoning", "LabInterp"],
    },
}

SCENARIO_NAMES = list(CLINICAL_SCENARIOS.keys())


def sample_grounding_facts_v2(
    subject_id: int,
    hadm_id: int | None,
    scenario: str,
) -> str:
    """
    Sample real patient grounding facts for a given clinical scenario.
    Delegates to the original data_sampler using the scenario's primary domain.
    """
    cfg = CLINICAL_SCENARIOS.get(scenario)
    if cfg is None:
        raise ValueError(f"Unknown scenario: {scenario!r}. Valid: {SCENARIO_NAMES}")

    primary_domain = cfg["grounding_domains"][0]
    facts = _original_sample(
        subject_id=subject_id,
        hadm_id=hadm_id,
        task_domain=primary_domain,
    )

    # Prepend scenario context so the user agent knows the clinical theme
    header = f"[Clinical scenario: {scenario}]\n{cfg['description']}\n\n"
    return header + facts
