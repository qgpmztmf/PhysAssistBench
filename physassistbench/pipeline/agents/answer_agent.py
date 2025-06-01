"""
Answer Agent — generates the AI assistant's final response to the user.

Given the user question, tool observations (REAL MIMIC data), and conversation
history, generates a clinically accurate, grounded answer.

Task types (new PhysAssistBench design):
  Information Lookup         — report retrieved EHR or patient data directly
  Data Gathering            — synthesize multi-source findings
  Clinical Reasoning — combine ONE fetched patient parameter with clinical knowledge
"""

import json
from physassistbench.pipeline.agents.llm_client import llm_call

# ── Information Lookup ─────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_RETRIEVAL = """You are an AI clinical assistant reporting EHR data to a clinician.

FORMAT RULES — follow exactly:
- Report ONLY the directly retrieved value(s).
- Format each item as: "[Item]: [Value] [Unit] (↑/↓/normal)" — one line per item.
- No introductory sentences. No closing remarks. No clinical commentary unless the question explicitly asks for interpretation.
- If data is missing: "[Item]: not found in EHR"
- Maximum 2 lines total."""

_SYSTEM_PROMPT_RETRIEVAL_ZH = """你是一位向临床医生报告EHR数据的AI临床助手。

格式规则——严格遵守：
- 只报告直接检索到的数值。
- 每项格式：「[项目]：[数值] [单位]（↑/↓/正常）」——每项占一行。
- 无引导句，无结束语，无临床评注（除非问题明确要求解读）。
- 如数据缺失：「[项目]：EHR中未找到」
- 最多2行。"""

# ── Data Gathering ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_WORKUP = """You are an AI clinical assistant synthesizing multi-source EHR findings.

FORMAT RULES — follow exactly:
- Respond as a bullet list, maximum 3 bullets.
- Each bullet: one key finding + its direct clinical significance.
- Format: "• [Finding]: [clinical implication]"
- No introductory sentence. No closing sentence.
- If a tool returned no data: "• [item]: not available in EHR"

CLINICAL SCORING — subscore tables (use when Data Gathering retrieves multiple components):
When the retrieved data contains components of a scoring system, compute it inline:

  FORMAT:
  • [Score Name] calculation:
      [Component] = [value] [unit] → [X] pts  (threshold: ...)
      ...
      Missing: [component list] — assumed 0 (note in answer)
      TOTAL: [X]/[max] → [category]
  • Clinical implication: [one actionable recommendation]

  CRITICAL SCORING RULE: For EVERY component, you MUST look up the EXACT value
  in the threshold table below and assign the corresponding score. Do NOT estimate
  or default to 2 pts for "abnormal". A value of Cr=5.1 is NOT 2 pts — it is 4 pts
  (≥5→4). Bilirubin=7.1 is NOT 2 pts — it is 3 pts (6.0-11.9→3). Check each
  component individually against its specific threshold range.

  SCORING TABLES:
  SOFA (max 24, each organ 0-4):
    Platelets K/μL: ≥150→0, <150→1, <100→2, <50→3, <20→4
    Bilirubin mg/dL: <1.2→0, 1.2-1.9→1, 2.0-5.9→2, 6.0-11.9→3, ≥12→4
    Creatinine mg/dL: <1.2→0, 1.2-1.9→1, 2.0-3.4→2, 3.5-4.9→3, ≥5→4
    Vasopressors: none→0, dopamine≤5/dobutamine→1, dopamine>5/epi≤0.1/norepi≤0.1→2,
                  dopamine>15/epi>0.1/norepi>0.1→3, ≥highest doses→4
    [PaO₂/FiO₂ and GCS: note as unavailable from labevents]
    Severity: 0-6=low risk, 7-9=moderate, ≥10=high (>40% mortality)

  SIRS (≥2 criteria = SIRS met):
    Temp: >38°C or <36°C = 1 criterion
    HR: >90 bpm = 1 criterion (from vital signs if available)
    RR: >20/min or PaCO₂<32 = 1 criterion
    WBC K/μL: >12 or <4 or bands >10% = 1 criterion

  MELD = 3.78×ln(Bili) + 11.2×ln(INR) + 9.57×ln(Cr) + 6.43 (round to integer)
    <10=low, 10-19=moderate, 20-29=high, 30-39=very high, ≥40=critical

  Child-Pugh (each 1-3 pts, total 5-15):
    Bilirubin mg/dL: <2→1, 2-3→2, >3→3
    Albumin g/dL: >3.5→1, 2.8-3.5→2, <2.8→3
    INR: <1.7→1, 1.7-2.3→2, >2.3→3
    Ascites: none→1, mild→2, moderate-severe→3 (from patient interview)
    Encephalopathy: none→1, grade 1-2→2, grade 3-4→3 (from patient interview)
    Class: 5-6=A(compensated), 7-9=B(significant), 10-15=C(decompensated)

  CURB-65 (1pt each, max 5):
    Confusion (GCS<15 or new confusion) = 1 pt
    Urea/BUN: BUN>19 mg/dL (or urea>7 mmol/L) = 1 pt
    RR ≥30/min = 1 pt (from vital signs if available)
    BP: SBP<90 or DBP≤60 = 1 pt (from vital signs)
    Age ≥65 = 1 pt (from patient context)
    Score: 0-1=low(outpatient), 2=moderate(hospitalize), ≥3=severe(consider ICU)

  CHA₂DS₂-VASc (from Condition.search diagnoses):
    CHF=1, Hypertension=1, Age≥75=2, Diabetes=1, Stroke/TIA=2,
    Vascular disease=1, Age 65-74=1, Female sex=1
    Score ≥2(male) or ≥3(female): anticoagulation recommended

  HAS-BLED:
    Hypertension(SBP>160)=1, Abnormal renal(Cr>2.3)/liver(cirrhosis)=1each,
    Stroke=1, Bleeding history=1, Labile INR=1, Elderly>65=1,
    Drugs(antiplatelet/NSAID)/alcohol=1each
    Score ≥3: high bleeding risk — anticoagulation with caution

  Ranson admission criteria (pancreatitis, 1pt each):
    Age>55, WBC>16K, Glucose>200 mg/dL, LDH>350 IU/L, AST>250 IU/L
    Score: 0-2=mild, 3-4=moderate, ≥5=severe

  Cockcroft-Gault CrCl (mL/min) = (140-age) × weight(kg) / (72 × Cr) × (0.85 if female)
    Use for renal dose adjustment: <15=ESRD, 15-29=severe, 30-59=moderate, 60-89=mild"""

_SYSTEM_PROMPT_WORKUP_ZH = """你是一位综合多来源EHR发现的AI临床助手。

格式规则——严格遵守：
- 以项目符号列表回答，最多3条。
- 每条：一个关键发现 + 其直接临床意义。
- 格式：「• [发现]：[临床意义]」
- 无引导句，无结束句。
- 如某工具未返回数据：「• [项目]：EHR中不可用」

临床评分——子分值对照表（当Workup获取多个组件时使用）：
若检索数据包含某评分系统的组件，按以下格式内联计算：

  关键规则：每个组件必须对照下方对照表的具体数值区间查找分值。
  严禁"异常=2分"的简化处理。例如：Cr=5.1 mg/dL → 4分（≥5→4，非2分）；
  胆红素=7.1 mg/dL → 3分（6.0-11.9→3，非2分）。必须逐一核查每个区间。

  格式：
  • [评分名称]计算：
      [组件] = [数值] [单位] → [X]分  (阈值: ...)
      ...
      缺失：[组件列表]——默认为0（在答案中注明）
      总分：[X]/[最高分] → [分级]
  • 临床意义：[一条具体可行建议]

  各评分子分值对照表：
  SOFA（最高24分，每个器官0-4分）：
    血小板 K/μL: ≥150→0, <150→1, <100→2, <50→3, <20→4
    胆红素 mg/dL: <1.2→0, 1.2-1.9→1, 2.0-5.9→2, 6.0-11.9→3, ≥12→4
    肌酐 mg/dL: <1.2→0, 1.2-1.9→1, 2.0-3.4→2, 3.5-4.9→3, ≥5→4
    升压药: 无→0, 多巴胺≤5/多巴酚丁胺→1, 多巴胺>5/肾上腺素≤0.1/去甲肾≤0.1→2,
            多巴胺>15/肾上腺素>0.1/去甲肾>0.1→3, 最高剂量→4
    [PaO₂/FiO₂和GCS：来自ICU记录，labevents中不可用，注明缺失]
    严重程度: 0-6=低风险, 7-9=中度, ≥10=高危(病死率>40%)

  SIRS（≥2项=满足）：
    体温: >38°C或<36°C = 1项
    心率: >90次/分 = 1项（来自生命体征）
    呼吸: >20次/分或PaCO₂<32 = 1项
    WBC K/μL: >12或<4或杆状核>10% = 1项

  MELD = 3.78×ln(胆红素) + 11.2×ln(INR) + 9.57×ln(肌酐) + 6.43（取整）
    <10=低, 10-19=中, 20-29=高, 30-39=极高, ≥40=危重

  Child-Pugh（每项1-3分，总分5-15）：
    胆红素 mg/dL: <2→1, 2-3→2, >3→3
    白蛋白 g/dL: >3.5→1, 2.8-3.5→2, <2.8→3
    INR: <1.7→1, 1.7-2.3→2, >2.3→3
    腹水: 无→1, 轻度→2, 中重度→3（来自病人访谈）
    脑病: 无→1, 1-2级→2, 3-4级→3（来自病人访谈）
    分级: 5-6=A级(代偿), 7-9=B级(显著), 10-15=C级(失代偿)

  CURB-65（每项1分，最高5分）：
    意识模糊(GCS<15) = 1分
    尿素/BUN: BUN>19 mg/dL = 1分
    呼吸率≥30次/分 = 1分（来自生命体征）
    血压: SBP<90或DBP≤60 = 1分
    年龄≥65岁 = 1分
    评分: 0-1=低危(门诊), 2=中危(住院), ≥3=重危(考虑ICU)

  CHA₂DS₂-VASc（来自Condition.search诊断）：
    心衰=1, 高血压=1, 年龄≥75=2, 糖尿病=1, 卒中/TIA=2,
    血管病=1, 年龄65-74=1, 女性=1
    评分≥2(男)或≥3(女): 建议抗凝

  HAS-BLED：
    高血压(SBP>160)=1, 肾功异常(Cr>2.3)/肝功异常(肝硬化)=各1,
    卒中=1, 出血史=1, INR不稳定=1, 年龄>65=1,
    药物(抗血小板/NSAIDs)/酒精=各1
    评分≥3: 高出血风险——谨慎抗凝

  Ranson入院标准（胰腺炎，各1分）：
    年龄>55, WBC>16K, 血糖>200 mg/dL, LDH>350 IU/L, AST>250 IU/L
    评分: 0-2=轻度, 3-4=中度, ≥5=重度

  Cockcroft-Gault CrCl (mL/min) = (140-年龄) × 体重(kg) / (72 × 肌酐) × (0.85若女性)
    用于肾脏剂量调整: <15=终末期, 15-29=重度, 30-59=中度, 60-89=轻度"""

# ── Clinical Reasoning ────────────────────────────────────────────────────────
_SYSTEM_PROMPT_KNOWLEDGE = """You are an AI clinical assistant combining a retrieved patient value with clinical knowledge.

FORMAT RULES — follow exactly:
- Respond in EXACTLY 2 sentences. No more.
- Sentence 1: State the retrieved patient value with units and whether it is normal/abnormal.
- Sentence 2: Give ONE specific, actionable clinical recommendation based on that value.
- Do NOT give generic advice. Do NOT add a third sentence."""

_SYSTEM_PROMPT_KNOWLEDGE_ZH = """你是一位将检索到的患者数值与临床知识相结合的AI临床助手。

格式规则——严格遵守：
- 恰好回答2句话，不多不少。
- 第1句：说明检索到的患者数值（含单位及是否正常/异常）。
- 第2句：基于该数值给出一条具体可行的临床建议。
- 不给泛泛建议，不写第3句话。"""

# ── Patient-source ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_PATIENT = """You are an AI clinical assistant summarizing information gathered from a patient interview.

FORMAT RULES — follow exactly:
- Respond as a bullet list, maximum 3 bullets.
- Each bullet: one key finding from the interview + its clinical significance.
- Format: "• [Finding]: [clinical implication]"
- No introductory sentence. No closing sentence."""

_SYSTEM_PROMPT_PATIENT_ZH = """你是一位总结患者访谈信息的AI临床助手。

格式规则——严格遵守：
- 以项目符号列表回答，最多3条。
- 每条：访谈中的一个关键发现 + 其临床意义。
- 格式：「• [发现]：[临床意义]」
- 无引导句，无结束句。"""


def generate_answer(
    context: dict,
    history: list,
    user_question: str,
    executed_actions: list,
    task_type: str,
    tool_source: str = "ehr",
    language: str = "en",
) -> str:
    """
    Generate the assistant's response to the user question.

    Returns: answer string (plain text / markdown)
    """
    zh = language == "zh"

    # Format observations
    obs_text = ""
    for act in executed_actions:
        name = act["action"]["name"]
        if name == "prepare_to_answer":
            continue
        obs = act["observation"]

        # Special handling for DiagnosticReport FHIR bundles:
        # extract presentedForm.data (report text) directly instead of
        # JSON-dumping the whole bundle, which buries the findings in noise.
        if isinstance(obs, dict) and obs.get("resourceType") == "Bundle":
            extracted_parts = []
            for entry in obs.get("entry", []):
                res = entry.get("resource", {})
                if res.get("resourceType") == "DiagnosticReport":
                    code_text = res.get("code", {}).get("text", "DiagnosticReport")
                    report_text = ""
                    for pf in res.get("presentedForm", []):
                        report_text = pf.get("data", "")
                        if report_text:
                            break
                    eff_dt = res.get("effectiveDateTime", "")
                    entry_str = f"Report: {code_text}"
                    if eff_dt:
                        entry_str += f"  Date: {eff_dt[:10]}"
                    if report_text:
                        entry_str += f"\n{report_text[:2000]}"
                    extracted_parts.append(entry_str)
                else:
                    extracted_parts.append(
                        json.dumps(res, default=str, ensure_ascii=False)[:800]
                    )
            obs_str = "\n---\n".join(extracted_parts) if extracted_parts else "(no results)"
        else:
            obs_str = json.dumps(obs, default=str, ensure_ascii=False)[:1500]

        obs_text += f"\n[{name}] returned:\n{obs_str}\n"

    # Format history (last 4 messages)
    history_str = ""
    if history:
        history_str = "Previous conversation:\n"
        for m in history[-4:]:
            role = m.get("role", "")
            content = str(m.get("content", ""))[:300]
            history_str += f"[{role.upper()}]: {content}\n"

    ctx = context.get("context_str", "")[:300]

    # ── Patient-source turns ──────────────────────────────────────────────────
    is_patient_only = tool_source == "patient" or (
        tool_source == "mixed"
        and any(a["action"]["name"].startswith("patient.") for a in executed_actions)
        and not any(
            not a["action"]["name"].startswith("patient.")
            and a["action"]["name"] not in ("prepare_to_answer", "ask_user_for_required_parameters")
            for a in executed_actions
        )
    )
    if is_patient_only:
        patient_info = ""
        for act in executed_actions:
            name = act["action"]["name"]
            if name.startswith("patient."):
                obs = act["observation"]
                patient_response = obs.get("patient_response", str(obs))
                tool_label = name.replace("patient.", "").replace("_", " ").title()
                patient_info += f"\n[{tool_label}]: {patient_response}\n"

        user_prompt = (
            f"Patient context: {ctx}\n\n"
            f"{history_str}\n"
            f"Clinician's question: {user_question}\n\n"
            f"Information from the patient:\n"
            f"{patient_info.strip() or '[No patient interview data collected]'}\n\n"
            f"Summarize in ≤3 bullets:"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_PATIENT_ZH if zh else _SYSTEM_PROMPT_PATIENT},
            {"role": "user", "content": user_prompt},
        ]
        return llm_call(messages, temperature=0.2, max_tokens=4000).strip()

    # ── Clinical Reasoning ────────────────────────────────────────────────────
    if task_type == "Clinical Reasoning":
        user_prompt = (
            f"Patient context: {ctx}\n\n"
            f"{history_str}\n"
            f"Clinician's question: {user_question}\n\n"
            f"Retrieved patient data:\n"
            f"{obs_text.strip() or '[No data retrieved]'}\n\n"
            f"Respond in exactly 2 sentences:"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_KNOWLEDGE_ZH if zh else _SYSTEM_PROMPT_KNOWLEDGE},
            {"role": "user", "content": user_prompt},
        ]
        return llm_call(messages, temperature=0.2, max_tokens=4000).strip()

    # ── Data Gathering ────────────────────────────────────────────────────────────────
    if task_type == "Data Gathering":
        user_prompt = (
            f"Patient context: {ctx}\n\n"
            f"{history_str}\n"
            f"Clinician's question: {user_question}\n\n"
            f"Retrieved EHR data:\n"
            f"{obs_text.strip() or '[No tool calls]'}\n\n"
            f"Summarize in ≤3 bullets:"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_WORKUP_ZH if zh else _SYSTEM_PROMPT_WORKUP},
            {"role": "user", "content": user_prompt},
        ]
        return llm_call(messages, temperature=0.2, max_tokens=4000).strip()

    # ── Information Lookup with DiagnosticReport text ─────────────────────────────────
    # When the retrieved data contains a full radiology/pathology report
    # (DiagnosticReport.presentedForm.data), the standard 2-line Information Lookup
    # format is insufficient. Switch to the Data Gathering bullet-list prompt so the
    # answer summarises key FINDINGS rather than just echoing the report title.
    _has_report_text = any(
        act["action"]["name"] == "DiagnosticReport.search"
        and isinstance(act.get("observation"), dict)
        and act["observation"].get("total", 0) > 0
        for act in executed_actions
    )
    if _has_report_text:
        user_prompt = (
            f"Patient context: {ctx}\n\n"
            f"{history_str}\n"
            f"Clinician's question: {user_question}\n\n"
            f"Radiology / Diagnostic report text:\n"
            f"{obs_text.strip() or '[No report text]'}\n\n"
            f"Summarise the key FINDINGS and IMPRESSION in ≤4 bullets. "
            f"If the report explicitly states absence of a finding (e.g. 'no PE'), include that. "
            f"Do NOT quote the full report — extract the clinically relevant conclusions."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_WORKUP_ZH if zh else _SYSTEM_PROMPT_WORKUP},
            {"role": "user", "content": user_prompt},
        ]
        return llm_call(messages, temperature=0.2, max_tokens=4000).strip()

    # ── Information Lookup (default) ───────────────────────────────────────────────────
    user_prompt = (
        f"Patient context: {ctx}\n\n"
        f"{history_str}\n"
        f"Clinician's question: {user_question}\n\n"
        f"Retrieved EHR data:\n"
        f"{obs_text.strip() or '[No data retrieved]'}\n\n"
        f"Report the retrieved value(s) only:"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT_RETRIEVAL_ZH if zh else _SYSTEM_PROMPT_RETRIEVAL},
        {"role": "user", "content": user_prompt},
    ]
    return llm_call(messages, temperature=0.2, max_tokens=4000).strip()
