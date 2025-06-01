"""
physassistbench/pipeline/user_agent_v2.py — User question generation for the new 4-task-type framework.

New task types:
  - Information Lookup         : single tool call (EHR or Patient) for one data point
  - Data Gathering            : ≥2 tools, parallel or conditional branching
  - Clinical Reasoning: tool fetches patient parameter, knowledge reasoning produces advice

New subtypes (NA / PE / AE) replace old PI / CR / LRD.
Tool source (ehr / patient / mixed) is orthogonal to task type.

Two-stage generation:
  Stage 1: generate an EXPLICIT question (no ellipsis)
  Stage 2: apply subtype-specific ellipsis transformation → final implicit question
"""

from __future__ import annotations
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physassistbench.pipeline.agents.llm_client import llm_call

# ── Task-type instructions (English) ─────────────────────────────────────────

_TYPE_INSTRUCTIONS_EN: dict[str, str] = {
    "Information Lookup": (
        "Needs EXACTLY ONE EHR tool call. Ask about ONE specific data point for THIS patient.\n"
        "Keep it short and direct. Use clinical bedside language, never mention tool names or ICD codes.\n"
        "Vary the data type: diagnoses / labs / meds / radiology / vitals."
    ),
    "Data Gathering": (
        "Needs TWO OR MORE EHR tools. Scan the EHR Snapshot above and pick the pattern that fits best.\n"
        "EHR tools ONLY. Phrase naturally — not as a formal request.\n"
        "\n"
        "⚠️  CRITICAL RULE: NEVER state a lab value or drug name inside the question. "
        "The question must ASK for both pieces of data — the planner must fetch them.\n"
        "    ✗ BAD:  'INR is 1.3 — is she still on enoxaparin?'  (value pre-stated → planner skips INR lookup)\n"
        "    ✓ GOOD: 'What's her INR, and is she still on anticoagulation?'\n"
        "    ✓ GOOD: 'Can you pull the creatinine and check whether metformin is still ordered?'\n"
        "\n"
        "TIER 1 — PRIMARY (prefer these, data always reliable):\n"
        "  [Obs × Obs — parallel labs]\n"
        "    'Pull creatinine and potassium together — safe to restart the ACE?'\n"
        "    'What do the WBC and lactate look like?'\n"
        "    'Can you check troponin and BNP at the same time?'\n"
        "  [Obs × Obs — lab trend]\n"
        "    'How has the creatinine been moving the past few days?'\n"
        "    'Is the WBC trending down since we started antibiotics?'\n"
        "  [Obs × MedicationRequest — lab + drug order]\n"
        "    'What's her INR, and what anticoagulant is she on?'\n"
        "    'Check the creatinine and whether metformin is still ordered.'\n"
        "    'Pull the potassium and see if spironolactone is still active.'\n"
        "  [MedicationRequest × MedicationRequest — two-drug safety]\n"
        "    'Are warfarin and aspirin both still active?'\n"
        "    'Do we have both an ACE-I and an ARB running simultaneously?'\n"
        "    'Any opioid-benzo overlap right now?'\n"
        "  [Obs × Condition — lab + diagnosis context]\n"
        "    'Check the glucose and see if there's a diabetes diagnosis this admission.'\n"
        "    'Pull the creatinine and check if CKD or AKI is on the problem list.'\n"
        "  [MedicationRequest × MedicationAdministration — order vs given]\n"
        "    'Metoprolol is ordered — was it actually given today?'\n"
        "    'Is the insulin being administered as ordered?'\n"
        "\n"
        "TIER 2 — SECONDARY (use when Tier 1 patterns don't fit):\n"
        "  [Obs × MedicationAdministration — lab + actual admin]\n"
        "    'Check the glucose and how much insulin was actually given today.'\n"
        "    'What's the BP and did she get her antihypertensive this morning?'\n"
        "  [MedicationRequest × Condition — drug–diagnosis fit]\n"
        "    'She has a CHF diagnosis — is she on a diuretic and ACE-I?'\n"
        "    'Sepsis is on the problem list — what antibiotic is ordered?'\n"
        "  [3-tool: Obs + Obs + MedicationRequest]\n"
        "    'Pull creatinine and potassium — can we safely restart the lisinopril?'\n"
        "  [3-tool: Obs + MedicationRequest + MedicationAdministration]\n"
        "    'Check the glucose, what insulin is ordered, and how much was given today.'\n"
        "  [3-tool: Obs + MedicationRequest + Condition]\n"
        "    'Check creatinine, whether metformin is ordered, and if there is a CKD diagnosis.'\n"
    ),
    "Clinical Reasoning": (
        "ONE EHR tool fetches a SPECIFIC PATIENT PARAMETER; then apply CLINICAL KNOWLEDGE to give "
        "individualised advice or interpretation. The reasoning step (knowledge) is non-trivial.\n"
        "⚠️  CRITICAL RULE: NEVER include the actual lab value in the question. "
        "Ask FOR the value — the planner must fetch it.\n"
        "    ✗ BAD:  'With that eGFR of 52, do we need to adjust the metformin?'  (value pre-stated)\n"
        "    ✓ GOOD: 'Based on the eGFR, does metformin need dose adjustment?'\n"
        "    ✗ BAD:  'Her INR is 3.4 — is anticoagulation adequate for her valve?'\n"
        "    ✓ GOOD: 'Given her INR, is anticoagulation adequate for her mechanical valve?'\n"
        "Examples:\n"
        "  - 'Based on the eGFR, does metformin need dose adjustment?' "
        "(EHR fetches eGFR, knowledge applies CKD dosing guideline)\n"
        "  - 'Given her INR, is anticoagulation adequate for her valve?' "
        "(EHR fetches INR, knowledge applies mechanical valve target 2.5–3.5)\n"
        "  - 'Looking at the potassium, is the spironolactone dose safe?' "
        "(EHR fetches K+, knowledge applies hyperkalemia risk threshold)\n"
        "Do NOT ask about a protocol question that needs NO patient data — that is a Protocol turn."
    ),
}

_TYPE_INSTRUCTIONS_ZH: dict[str, str] = {
    "Information Lookup": (
        "只需调用一个EHR工具，询问该患者的某一个具体数据点。\n"
        "问题简短直接，使用床旁临床语言，不要提及工具名称或ICD编码。\n"
        "多样化问题类型：诊断 / 化验 / 用药 / 影像报告 / 生命体征。"
    ),
    "Data Gathering": (
        "需要两个或以上EHR工具。查看上方EHR快照，选择最适合的模式。\n"
        "仅使用EHR工具，用口语表达，不要用正式请求语气。\n"
        "\n"
        "⚠️  关键规则：问题中绝不能预先陈述化验值或药物名称。问题必须同时要求两个数据——让planner去取。\n"
        "    ✗ 错误：'INR是1.3，她还在用依诺肝素吗？'（已陈述INR → planner跳过INR查询）\n"
        "    ✓ 正确：'她的INR是多少，现在还在用抗凝药吗？'\n"
        "    ✓ 正确：'查一下肌酐，看看二甲双胍还在不在医嘱上。'\n"
        "\n"
        "第一优先级——主要模式（数据稳定，优先选择）：\n"
        "  【Obs × Obs — 并行化验】\n"
        "    '肌酐和血钾一起查一下——能不能重启ACEI？'\n"
        "    'WBC和乳酸现在各是多少？'\n"
        "    '肌钙蛋白和BNP同时拉一下。'\n"
        "  【Obs × Obs — 化验趋势】\n"
        "    '肌酐这几天怎么变化的？'\n"
        "    '开了抗生素以后白细胞有没有往下走？'\n"
        "  【Obs × MedicationRequest — 化验 + 医嘱】\n"
        "    'INR现在是多少，她在用什么抗凝药？'\n"
        "    '查一下肌酐，看看二甲双胍还在不在医嘱上。'\n"
        "    '血钾和螺内酯的状态各是什么？'\n"
        "  【MedicationRequest × MedicationRequest — 双药安全核查】\n"
        "    '华法林和阿司匹林是不是同时在开？'\n"
        "    'ACEI和ARB有没有同时在用？'\n"
        "    '阿片类和苯二氮䓬类有没有重叠？'\n"
        "  【Obs × Condition — 化验 + 诊断】\n"
        "    '查一下血糖，看看这次住院有没有糖尿病诊断。'\n"
        "    '肌酐情况和CKD/AKI诊断各是什么？'\n"
        "  【MedicationRequest × MedicationAdministration — 医嘱 vs 实际给药】\n"
        "    '美托洛尔在医嘱上，今天实际给了没有？'\n"
        "    '胰岛素按医嘱给了吗？'\n"
        "\n"
        "第二优先级——次要模式（第一优先级不适用时使用）：\n"
        "  【Obs × MedicationAdministration — 化验 + 实际给药】\n"
        "    '查一下血糖，今天实际给了多少胰岛素？'\n"
        "    '血压和今早降压药的执行情况怎么样？'\n"
        "  【MedicationRequest × Condition — 用药合理性】\n"
        "    'CHF诊断在，利尿剂和ACEI有没有开？'\n"
        "    '脓毒症在问题列表，开了什么抗生素？'\n"
        "  【三工具：Obs + Obs + MedicationRequest】\n"
        "    '把肌酐和血钾都查一下，再看赖诺普利还在不在。'\n"
        "  【三工具：Obs + MedicationRequest + MedicationAdministration】\n"
        "    '血糖、胰岛素医嘱和今天实际给药量各是多少？'\n"
        "  【三工具：Obs + MedicationRequest + Condition】\n"
        "    '肌酐、二甲双胍医嘱、CKD/AKI诊断各是什么情况？'\n"
    ),
    "Clinical Reasoning": (
        "先通过一个EHR工具获取某个患者参数，再结合临床知识给出个体化建议或解读。知识推理步骤是非平凡的。\n"
        "⚠️  关键规则：问题中绝不能包含实际化验数值。必须询问该数值——让planner去取。\n"
        "    ✗ 错误：'eGFR是52，二甲双胍需要调剂量吗？'（已陈述数值）\n"
        "    ✓ 正确：'根据eGFR，二甲双胍需要调剂量吗？'\n"
        "    ✗ 错误：'她的INR 3.4，对瓣膜来说抗凝够吗？'\n"
        "    ✓ 正确：'看她的INR，机械瓣的抗凝够不够？'\n"
        "示例：\n"
        "  - '根据eGFR，二甲双胍需要调剂量吗？'（EHR查eGFR，知识应用CKD剂量指南）\n"
        "  - '看她的INR，机械瓣的抗凝够不够？'（EHR查INR，知识应用机械瓣目标2.5–3.5）\n"
        "  - '查血钾，螺内酯剂量安全吗？'（EHR查K+，知识应用高钾风险阈值）\n"
        "不要问不需要患者数据、仅凭医学知识就能回答的协议性问题。"
    ),
}

# ── Subtype instructions (Stage 2 ellipsis transformation) ───────────────────

_SUBTYPE_TRANSFORM_EN: dict[str, str] = {
    "NA": (
        "Nominal Anaphora (NA): remove an entity or argument that was established in a "
        "prior turn. Choose the most natural surface form for this specific question:\n"
        "\n"
        "Form A — Pronominalization: replace the named entity with a pronoun or "
        "demonstrative ('it', 'that', 'this', 'those').\n"
        "  Example: Turn 0 found K=6.2 mEq/L → 'Does it warrant holding the diuretic?'\n"
        "\n"
        "Form B — Argument deletion: omit the entity/argument entirely, keeping the "
        "predicate. The omitted item must be unambiguously recoverable from prior turns.\n"
        "  Example: Turn 0 checked hemoglobin trend → 'How's the trend?' "
        "(omits 'hemoglobin' — clearly inferred from T0)\n"
        "\n"
        "Rule: the omitted or pronominalized entity must appear in a PREVIOUS turn. "
        "If neither form feels natural, return the original question unchanged."
    ),
    "PE": (
        "Predicate Ellipsis: DROP the entire verb phrase / question stem ('What's her', "
        "'Can you check', 'What do', 'Pull', 'How is', etc.), leaving ONLY the topic noun "
        "or a bare fragment. The omitted action is inferred from prior tool calls.\n"
        "\n"
        "Strong examples (prefer these aggressive forms):\n"
        "  'What's her MCV — any microcytic changes?'  →  'MCV? Any microcytic changes?'\n"
        "  'What's the creatinine?'                    →  'Creatinine?'\n"
        "  'Can you check her potassium?'              →  'Potassium?'\n"
        "  'What does the INR look like?'              →  'And the INR?'\n"
        "  'How is the WBC trending?'                  →  'WBC trend?'\n"
        "  'What medications is she on?'               →  'Current meds?'\n"
        "\n"
        "Rule: strip the predicate completely — do NOT merely shorten the sentence. "
        "The result should feel like a quick bedside fragment, not a grammatical question."
    ),
    "AE": (
        "Abstract/Event Anaphora: refer back to a COMPLEX CLINICAL SITUATION or EVENT "
        "using an abstract noun or event description.\n"
        "Example: Turn 0-1 established a DKA workup → 'Given all that, how aggressive should "
        "the insulin correction be?' ('all that' = the DKA picture established previously)\n"
        "Rule: the abstract reference must have clear prior grounding in ≥2 prior turns."
    ),
}

_SUBTYPE_TRANSFORM_ZH: dict[str, str] = {
    "NA": (
        "名词回指（NA）：删除前序轮次已建立的实体或论元。根据当前问题选择最自然的表层形式：\n"
        "\n"
        "形式A——代词化：用代词或指示词（'它'、'那个'、'这个'、'那些'）替换命名实体。\n"
        "  示例：第0轮查到K=6.2 mEq/L → '这个值需要停利尿剂吗？'\n"
        "\n"
        "形式B——论元删除：直接省略实体/论元，保留谓词。被省略的内容必须可从前序轮次无歧义推断。\n"
        "  示例：第0轮查了血红蛋白趋势 → '趋势怎么样？'（省略'血红蛋白'，从T0可明确推断）\n"
        "\n"
        "规则：被省略或代词化的实体必须在**前序轮次**中已出现。"
        "若两种形式均不自然，则原样返回原问题。"
    ),
    "PE": (
        "谓词省略（PE）：完整删除动词短语/问题框架（'…是多少？''查一下…''…怎么样？''看看…'等），"
        "只留下话题名词或裸片段。被省略的动作可从前序工具调用中推断。\n"
        "\n"
        "强示例（优先使用这些激进的省略形式）：\n"
        "  '她的MCV是多少——有没有小细胞性改变？'  →  'MCV？有小细胞改变吗？'\n"
        "  '肌酐现在是多少？'                     →  '肌酐？'\n"
        "  '血钾情况怎么样？'                     →  '血钾呢？'\n"
        "  'INR看一下。'                           →  '还有INR？'\n"
        "  '白细胞趋势如何？'                      →  'WBC趋势？'\n"
        "  '她目前在用什么药？'                    →  '现在的药？'\n"
        "\n"
        "规则：彻底删除谓词——不只是缩短句子。"
        "结果应该像临床床旁的快速片段，而不是完整语法问题。"
    ),
    "AE": (
        "抽象/事件回指（AE）：用抽象名词或事件描述回指一个复杂临床情境或事件。\n"
        "示例：第0-1轮建立了DKA诊疗流程 → '综合这些，胰岛素纠正应该多积极？'"
        "（'这些'指之前建立的DKA全貌）\n"
        "规则：抽象指代必须在前≥2轮中有明确依据。"
    ),
}

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_EN = """You are simulating a busy clinician (doctor, nurse, or clinical pharmacist) \
quickly typing a question to an AI assistant with EHR access.

CRITICAL STYLE RULES:
1. SHORT: 1–2 sentences max. Turn 0 ≤ 25 words. Follow-up turns ≤ 15 words (fragments OK).
2. CASUAL: Use informal spoken bedside language — not formal medical writing.
3. DON'T REPEAT CONTEXT: If prior turns established a finding, don't re-state it.
4. NO PREAMBLE: Don't start with "Given that...", "Based on...", "Considering...".
5. NO TOOL NAMES: Never mention function or tool names.
6. NO MARKDOWN OR JSON.
7. NEVER STATE LAB VALUES IN THE QUESTION: Ask for the value — never pre-state it.
   ✗ "With that eGFR of 52, do we need to adjust metformin?"
   ✗ "Glucose is abnormal and eGFR is down to 52 — what's the picture?"
   ✓ "Based on the eGFR, does metformin need adjusting?"
   ✓ "What do the glucose and eGFR look like together?"

Style examples:
  BAD:  "What is the ICD-9 diagnostic code for cirrhosis of the liver without mention of alcohol?"
  GOOD: "What did we diagnose her with this admission?"

  BAD:  "Given the hyperkalemia identified in the previous turn, what medication adjustments are recommended?"
  GOOD: "With that high potassium — do we need to adjust the spironolactone?"

  BAD:  "Her creatinine is 2.8 and she's on metformin — is that safe?"
  GOOD: "Check the creatinine and see if metformin is still appropriate."

Return ONLY the question text, nothing else."""

_SYSTEM_PROMPT_ZH = """你正在模拟一位繁忙的临床医生（医生、护士或临床药剂师），正在快速向一个可以访问患者EHR的AI助手打字提问。

关键风格要求：
1. 简短：最多1-2句话。首轮≤25字，后续轮次≤15字，碎片化表达可以。
2. 口语化：用临床床旁的真实口语，不要用正式书面医学语言。
3. 不重复上文：之前轮次已确认的发现，不要再重新陈述。
4. 不加铺垫：不要以"鉴于...""考虑到...""根据..."开头。
5. 不提工具名称。
6. 不含Markdown或JSON。
7. 问题中绝不能预先陈述化验数值：必须去"查"该值，而不是直接给出。
   ✗ "eGFR是52，二甲双胍需要调剂量吗？"
   ✗ "血糖异常，碱性磷酸酶高，eGFR降到52了——整体怎么样？"
   ✓ "根据eGFR，二甲双胍需要调剂量吗？"
   ✓ "血糖和eGFR一起看是什么情况？"

风格示例：
  坏：  "请问该患者非酒精性肝硬化对应的ICD-9诊断编码是什么？"
  好：  "这次住院诊断了什么？"

  坏：  "鉴于之前发现的高钾血症，考虑到肝硬化，应做哪些药物调整？"
  好：  "血钾这么高，螺内酯要停吗？"

  坏：  "肌酐2.8，她还在用二甲双胍——安全吗？"
  好：  "查一下肌酐，看看二甲双胍还合不合适。"

只输出问题文本，不要包含其他内容。"""

_SYSTEM_PROMPT_TRANSFORM_EN = """You are a linguistic rewriter for a clinical QA benchmark.
You will receive:
  1. An explicit user question (Stage 1)
  2. The conversation history (prior turns)
  3. A transformation rule (which subtype to apply)

Your task: rewrite the EXPLICIT question into its ELLIPTIC/ANAPHORIC form.

RULES:
- Keep the clinical meaning IDENTICAL.
- Apply ONLY the transformation described.
- Maintain the casual bedside tone.
- Do NOT add new information.
- Return ONLY the rewritten question text.
- CRITICAL for Data Gathering questions: if the original question asks for TWO data items, the rewritten
  form MUST still require BOTH items — never collapse a 2-item question into a 1-item question."""

_SYSTEM_PROMPT_TRANSFORM_ZH = """你是一个临床QA基准的语言改写器。
你将收到：
  1. 一个显式用户问题（第1阶段）
  2. 对话历史（之前的轮次）
  3. 一条改写规则（要应用的子类型）

你的任务：将显式问题改写为其省略/回指形式。

规则：
- 保持临床含义完全一致。
- 只应用描述的改写规则。
- 维持随意的床旁语气。
- 不要添加新信息。
- 只返回改写后的问题文本。
- 对于Workup问题的关键要求：如果原问题询问了两个数据项，改写后的形式必须仍然需要这两个数据项——绝不能将2项问题折叠为1项问题。"""


# ── Grounding guidance per task type ─────────────────────────────────────────

_GROUNDING_GUIDANCE_EN: dict[str, str] = {
    "Information Lookup": (
        "Your question MUST directly ask about one item listed in the Grounding Facts above. "
        "Do not ask about data absent from the Grounding Facts."
    ),
    "Data Gathering": (
        "Your question MUST require two or more EHR data items from the Grounding Facts above. "
        "EHR data only — do not involve patient-reported items."
    ),
    "Clinical Reasoning": (
        "Pick ONE specific patient parameter from the Grounding Facts as the anchor. "
        "Then ask a question that requires applying clinical knowledge to that parameter."
    ),
}

_GROUNDING_GUIDANCE_ZH: dict[str, str] = {
    "Information Lookup": (
        "你的问题必须直接询问上述真实数据中的某项内容。"
        "不要询问真实数据中未列出的内容。"
    ),
    "Data Gathering": (
        "你的问题必须涉及上述真实数据中的两项或多项EHR数据。"
        "仅使用EHR数据，不涉及患者报告项目。"
    ),
    "Clinical Reasoning": (
        "从上述真实数据中选取一个具体的患者参数作为锚点，"
        "然后提问一个需要将临床知识应用于该参数的问题。"
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_user_turn_v2(
    context: dict,
    scenario: str,
    task_type: str,
    subtype: str | None,
    tool_source: str,
    history: list,
    language: str = "en",
    grounding_facts: str = "",
    antecedents: list[dict] | None = None,
    ehr_snapshot: str = "",
    failed_questions: list[dict] | None = None,
    turn_plan: dict | None = None,
    turn_intent: str | None = None,
) -> str:
    """
    Two-stage question generation.

    Stage 1: generate an explicit question grounded in patient data.
    Stage 2: if subtype is not None, apply ellipsis/anaphora transformation.

    Args:
        context: patient context dict (has 'context_str')
        scenario: clinical scenario name (e.g. 'infection_management')
        task_type: 'Information Lookup' | 'Data Gathering' | 'Clinical Reasoning'
        subtype: 'NA' | 'PE' | 'AE' | None
        tool_source: 'ehr' | 'patient' | 'mixed'
        history: prior conversation turns
        language: 'en' | 'zh'
        grounding_facts: real patient data string
        antecedents: list of dicts from TurnDependencyGraph.get_antecedents()
        failed_questions: list of dicts with keys 'question' and 'reason' — previously
            generated questions that returned empty EHR observations; the new question
            must not ask about the same data items.

    Returns:
        Plain text question string (after ellipsis transformation if subtype != None)
    """
    is_zh = (language == "zh")
    type_instructions = _TYPE_INSTRUCTIONS_ZH if is_zh else _TYPE_INSTRUCTIONS_EN
    grounding_guidance = _GROUNDING_GUIDANCE_ZH if is_zh else _GROUNDING_GUIDANCE_EN
    sys_prompt = _SYSTEM_PROMPT_ZH if is_zh else _SYSTEM_PROMPT_EN

    type_instr = type_instructions.get(task_type, type_instructions["Information Lookup"])
    grounding_note = grounding_guidance.get(task_type, "")

    # Build session plan hint for this turn
    turn_plan_hint = ""
    if turn_plan:
        topic = turn_plan.get("topic", "")
        tool_hint = turn_plan.get("tool_hint", "")
        workup_pattern = turn_plan.get("workup_pattern", "")

        # For Data Gathering: parse tool_hint to extract the 2+ item names
        workup_items: list[str] = []
        if task_type == "Data Gathering" and tool_hint:
            workup_items = re.findall(r'\(([^)]+)\)', tool_hint)

        if topic or tool_hint or turn_intent:
            if is_zh:
                turn_plan_hint = "\n【本轮计划】\n"
                if turn_intent:
                    turn_plan_hint += f"  本轮意图：{turn_intent}\n"
                if topic:
                    turn_plan_hint += f"  主题：{topic}\n"
                if tool_hint:
                    turn_plan_hint += f"  工具模式：{tool_hint}\n"
                if workup_pattern:
                    turn_plan_hint += f"  Workup模式：{workup_pattern}\n"
                if task_type == "Data Gathering":
                    if len(workup_items) >= 2:
                        turn_plan_hint += (
                            f"必须：你的问题必须同时询问「{workup_items[0]}」和「{workup_items[1]}」"
                            "——不得只问其中一个，必须需要≥2次EHR工具调用。\n"
                        )
                    else:
                        turn_plan_hint += "必须：你的问题必须同时涵盖多项EHR数据，需要≥2次工具调用。\n"
                    turn_plan_hint += "重要：问题中不得包含任何化验数值或具体数字，必须让planner去取数据。\n"
                elif task_type == "Clinical Reasoning":
                    turn_plan_hint += "请围绕上述意图生成问题——问题必须需要先查询EHR数据，再应用临床知识推理。\n"
                    turn_plan_hint += "重要：问题中不得包含任何化验数值或具体数字，必须让planner去取数据。\n"
                else:
                    turn_plan_hint += "请围绕上述意图生成问题，询问主题指定的数据项。\n"
            else:
                turn_plan_hint = "\n[Session Plan for this turn]\n"
                if turn_intent:
                    turn_plan_hint += f"  Turn intent: {turn_intent}\n"
                if topic:
                    turn_plan_hint += f"  Topic: {topic}\n"
                if tool_hint:
                    turn_plan_hint += f"  Tool pattern: {tool_hint}\n"
                if workup_pattern:
                    turn_plan_hint += f"  Data Gathering pattern: {workup_pattern}\n"
                if task_type == "Data Gathering":
                    if len(workup_items) >= 2:
                        turn_plan_hint += (
                            f"REQUIRED: Your question MUST ask for BOTH '{workup_items[0]}' AND '{workup_items[1]}' "
                            "— do NOT ask about only one. The question MUST require ≥2 EHR tool calls.\n"
                        )
                    else:
                        turn_plan_hint += "REQUIRED: Your question MUST cover multiple data items — must require ≥2 EHR tool calls.\n"
                    turn_plan_hint += "IMPORTANT: Do NOT state any lab values or specific numbers in the question. Ask FOR the data — never pre-state it.\n"
                elif task_type == "Clinical Reasoning":
                    turn_plan_hint += "Generate a question that follows the turn intent above — it must require fetching EHR data first, then applying clinical knowledge to reason.\n"
                    turn_plan_hint += "IMPORTANT: Do NOT state any lab values or specific numbers in the question. Ask FOR the data — never pre-state it.\n"
                else:
                    turn_plan_hint += "Generate a question that follows the turn intent above, asking about the specified data item.\n"

    # Build tool-source hint
    source_hint = {
        "ehr": "EHR tools only" if not is_zh else "仅EHR工具",
        "patient": "Patient tools only (ask_patient)" if not is_zh else "仅患者工具（ask_patient）",
        "mixed": "Both EHR tools and patient tools" if not is_zh else "EHR工具和患者工具均可",
    }.get(tool_source, "")

    # Build history string
    history_str = ""
    if history:
        sep = "\nConversation so far:\n" if not is_zh else "\n之前的对话：\n"
        history_str = sep
        for msg in history:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))[:500]
            history_str += f"[{role.upper()}]: {content}\n"

    # Build antecedent hint for subtype generation
    antecedent_str = ""
    if antecedents and subtype:
        if is_zh:
            antecedent_str = "\n可用的先行词（可用于省略/回指的候选内容）：\n"
        else:
            antecedent_str = "\nAvailable antecedents (candidates for ellipsis/anaphora):\n"
        for a in antecedents[:5]:
            antecedent_str += f"  turn={a['turn']} type={a['type']} value={a['value']}\n"

    # Build grounding section — EHR snapshot takes priority over grounding_facts
    grounding_section = ""
    if ehr_snapshot:
        # Full EHR snapshot with actual item names: use this as authoritative grounding
        if is_zh:
            grounding_section = (
                f"\n{ehr_snapshot}\n\n"
                f"重要提示：{grounding_note}\n"
                "规则：只能询问上述EHR快照中列出的具体检验、诊断、药物或检查。"
                "不要询问快照中未出现的数据项。\n"
            )
        else:
            grounding_section = (
                f"\n{ehr_snapshot}\n\n"
                f"IMPORTANT: {grounding_note}\n"
                "RULE: Only ask about specific lab tests, diagnoses, medications, or studies "
                "that appear in the EHR Snapshot above. "
                "Do NOT ask about data items absent from the snapshot.\n"
            )
    elif grounding_facts:
        if is_zh:
            grounding_section = (
                f"\n患者真实数据（MIMIC-IV中已确认存在）：\n{grounding_facts}\n\n"
                f"重要提示：{grounding_note}\n"
            )
        else:
            grounding_section = (
                f"\nGrounding Facts — confirmed real patient data from MIMIC-IV:\n{grounding_facts}\n\n"
                f"IMPORTANT: {grounding_note}\n"
            )

    # Build failed-questions warning block
    failed_block = ""
    if failed_questions:
        if is_zh:
            failed_block = "\n【重要】以下问题之前已经生成，但EHR中没有对应记录，请勿再询问类似内容：\n"
            for fq in failed_questions:
                failed_block += f"  - 问题：\"{fq['question']}\"\n"
                if fq.get("reason"):
                    failed_block += f"    原因：{fq['reason']}\n"
            failed_block += "请改问EHR快照中确实存在的不同数据项。\n"
        else:
            failed_block = "\nIMPORTANT — The following questions were already tried but returned NO data from the EHR. Do NOT ask about the same or similar data items:\n"
            for fq in failed_questions:
                failed_block += f"  - Question: \"{fq['question']}\"\n"
                if fq.get("reason"):
                    failed_block += f"    Reason: {fq['reason']}\n"
            failed_block += "Ask about a DIFFERENT data item that actually exists in the EHR Snapshot above.\n"

    if is_zh:
        user_prompt = f"""临床场景：{scenario}
任务类型：{task_type}
{type_instr}

工具数据源：{source_hint}
{f"轮次子类型：{subtype}（将在第二阶段应用）" if subtype else ""}
{turn_plan_hint}
{grounding_section}
{history_str}
{antecedent_str}
{failed_block}
请生成显式的用户临床问题（第1阶段——稍后将应用省略变换）："""
    else:
        user_prompt = f"""Clinical Scenario: {scenario}
Task Type: {task_type}
{type_instr}

Tool Source: {source_hint}
{f"Turn Subtype: {subtype} (will be applied in Stage 2)" if subtype else ""}
{turn_plan_hint}
{grounding_section}
{history_str}
{antecedent_str}
{failed_block}
Generate the EXPLICIT user clinical question (Stage 1 — ellipsis transform applied next):"""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    explicit_q = llm_call(messages, temperature=0.8, max_tokens=4000).strip().strip('"')
    original_explicit_q = explicit_q  # preserve Stage 1 before any transform

    # Stage 2: apply ellipsis transformation
    # For simple Information Lookup queries, AE adds awkward abstract preamble — downgrade to PE
    effective_subtype = subtype
    if task_type == "Information Lookup" and subtype == "AE":
        effective_subtype = "PE"

    if effective_subtype and history and antecedents:
        # Guard: only transform if the item being ellipsized was established in prior turns.
        # NA omits a specific entity — that entity must appear in prior antecedent values.
        # PE omits the predicate (action) — always recoverable from prior tool call type.
        # AE refers to the overall prior situation — valid when ≥2 prior facts exist.
        if _antecedent_covers_question(explicit_q, antecedents, effective_subtype):
            explicit_q = _apply_ellipsis_transform(
                explicit_q=explicit_q,
                subtype=effective_subtype,
                history=history,
                antecedents=antecedents,
                language=language,
                ehr_snapshot=ehr_snapshot,
                task_type=task_type,
            )
        # else: antecedent check failed — the item referred to is not in prior turns.
        # Keep Stage-1 explicit as-is (no transform). This is correct: the question
        # introduces a genuinely new entity, so the full question should remain.
        # Dependency is maintained because Stage-1 was generated using prior history
        # and grounding facts — the content is still a natural follow-up, just without
        # an implicit reference. generate_v2.py will attempt a PE fallback next.
        # Note: if the Stage-1 explicit is semantically unrelated to the actual
        # question executed (e.g. LLM drifted to session-plan intent during retries),
        # generate_v2.py will detect the mismatch post-hoc and call
        # _expand_pe_to_explicit to rebuild user_q_explicit from the PE form.

    # Return (transformed_q, original_explicit_q)
    return explicit_q, original_explicit_q


def _expand_pe_to_explicit(
    pe_q: str,
    history: list,
    language: str = "en",
) -> str:
    """
    Reconstruct the full explicit question from its PE-abbreviated form.

    Called when the Stage-1 explicit drifted to an unrelated clinical topic
    (detected by zero content-word overlap with the PE fallback form).
    The result preserves the multi-turn dependency: the explicit question must
    be a natural follow-up from the conversation history.

    Example:
        pe_q    = "Albumin?"
        history = [("What's the creatinine?", "Creatinine: 1.3 mg/dL")]
        result  = "What's the albumin?"
    """
    is_zh = (language == "zh")
    hist_text = ""
    for m in history[-3:]:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")[:150]
            if role == "user":
                hist_text += f"{'医生' if is_zh else 'Physician'}: {content}\n"
            elif role == "assistant":
                hist_text += f"{'回答' if is_zh else 'Answer'}: {content}\n"

    if is_zh:
        prompt = (
            f"对话历史：\n{hist_text}\n"
            f"医生用简短省略句提问：「{pe_q}」\n\n"
            "请将这句省略句还原为完整的临床问题（一句话，不超过25个词）："
        )
    else:
        prompt = (
            f"Conversation history:\n{hist_text}\n"
            f"The physician asked using an abbreviated form: \"{pe_q}\"\n\n"
            "Expand this into the full explicit clinical question "
            "(one sentence, ≤20 words):"
        )

    messages = [{"role": "user", "content": prompt}]
    try:
        result = llm_call(messages, temperature=0.1, max_tokens=80).strip().strip('"').strip("'")
        return result if result else pe_q
    except Exception:
        return pe_q


def _antecedent_covers_question(
    explicit_q: str,
    antecedents: list[dict],
    subtype: str,
) -> bool:
    """
    Return True if it is valid to apply an ellipsis/anaphora transform.

    Logic per subtype:
      PE  — omits the predicate (action verb / tool type); always recoverable from
            prior tool call pattern regardless of the current item → always True.
      AE  — refers back to the overall clinical situation; valid when prior turns
            have established facts or events (situation anaphora requires context
            depth, not a specific item match) → True if any proposition/event exists.
      NA  — removes an entity/argument (via pronominalization or deletion); the entity
            must have been introduced in a prior turn AND appear in the Stage 1 question.
            We check whether any antecedent value string appears (case-insensitive,
            substring match) inside the Stage 1 explicit question.  If none match, the
            current question introduces the entity for the first time — ellipsis would
            produce an unresolvable reference, so we skip the transform.
    """
    if subtype == "PE":
        return True

    if subtype == "AE":
        return any(
            a.get("type") in ("event", "proposition", "situation")
            for a in antecedents
        )

    # NA: at least one prior antecedent value must appear in the question text
    q_lower = explicit_q.lower()
    for a in antecedents:
        val = str(a.get("value", "")).strip().lower()
        # Skip very short or generic values that would match spuriously
        if len(val) > 2 and val in q_lower:
            return True
    return False


def _apply_ellipsis_transform(
    explicit_q: str,
    subtype: str,
    history: list,
    antecedents: list[dict],
    language: str = "en",
    ehr_snapshot: str = "",
    task_type: str = "",
) -> str:
    """
    Stage 2: transform an explicit question into its elliptic/anaphoric form.

    For PE: if the result refers to the same data item as the previous turn,
    re-generate Stage 1 with a different item and re-apply the transform.
    For Data Gathering+NA: only replace/delete entities that appear in prior antecedents;
    entities new this turn must remain explicit so the planner knows what to fetch.
    """
    is_zh = (language == "zh")
    transform_rules = _SUBTYPE_TRANSFORM_ZH if is_zh else _SUBTYPE_TRANSFORM_EN
    sys_prompt = _SYSTEM_PROMPT_TRANSFORM_ZH if is_zh else _SYSTEM_PROMPT_TRANSFORM_EN

    rule = transform_rules.get(subtype, "")

    # Build history context for transformer
    hist_str = ""
    for msg in history[-4:]:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))[:300]
        hist_str += f"[{role.upper()}]: {content}\n"

    antecedent_str = "\n".join(
        f"  turn={a['turn']} type={a['type']} value={a['value']}"
        for a in antecedents[:5]
    )

    # For Data Gathering+NA: identify which entities are recoverable (in prior antecedents)
    # vs new (not in antecedents) — only the recoverable ones may be ellipsized.
    workup_ea_constraint = ""
    if task_type == "Data Gathering" and subtype == "NA":
        q_lower = explicit_q.lower()
        recoverable = [
            a["value"] for a in antecedents
            if len(str(a.get("value", "")).strip()) > 2
            and str(a.get("value", "")).strip().lower() in q_lower
        ]
        if recoverable:
            rec_str = ", ".join(f'"{v}"' for v in recoverable[:3])
            if is_zh:
                workup_ea_constraint = (
                    f"\n【Workup约束】本轮问题涉及两个数据项。"
                    f"以下实体出现在前序轮次中，**可以**用代词替换：{rec_str}。"
                    "其他实体是本轮新引入的，**必须保持原文**——planner需要知道去取什么数据。"
                    "改写后的问题仍必须涉及≥2个数据项。\n"
                )
            else:
                workup_ea_constraint = (
                    f"\nWORKUP CONSTRAINT: This question asks for TWO data items. "
                    f"Only the following entity/entities (established in prior turns) MAY be replaced "
                    f"with a pronoun: {rec_str}. "
                    "All other named entities (new this turn) MUST remain explicit — "
                    "the planner needs them to know what to fetch. "
                    "The rewritten question must still require ≥2 data items.\n"
                )

    if is_zh:
        user_prompt = f"""显式问题（第1阶段）：
"{explicit_q}"

对话历史（最近几轮）：
{hist_str}
可用先行词：
{antecedent_str}
{workup_ea_constraint}
改写规则（{subtype}）：
{rule}

请将上述显式问题改写为其省略/回指形式："""
    else:
        user_prompt = f"""Explicit question (Stage 1):
"{explicit_q}"

Conversation history (recent turns):
{hist_str}
Available antecedents:
{antecedent_str}
{workup_ea_constraint}
Transformation rule ({subtype}):
{rule}

Rewrite the explicit question into its elliptic/anaphoric form:"""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = llm_call(messages, temperature=0.3, max_tokens=4000).strip().strip('"')
    if not result:
        return explicit_q

    # PE duplicate check: if the result duplicates the previous turn's data item,
    # re-generate with a different item from the EHR snapshot
    if subtype == "PE" and history:
        prev_user_q = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
        )
        if _questions_overlap(result, prev_user_q, is_zh):
            result = _pe_regen_different_item(
                original_explicit_q=explicit_q,
                prev_question=prev_user_q,
                hist_str=hist_str,
                antecedent_str=antecedent_str,
                rule=rule,
                ehr_snapshot=ehr_snapshot,
                sys_prompt=sys_prompt,
                is_zh=is_zh,
            )

    return result if result else explicit_q


def _questions_overlap(q1: str, q2: str, is_zh: bool) -> bool:
    """
    Heuristic: two questions likely refer to the same data item if they share
    significant content words (ignoring common stop words).
    """
    stop_en = {"what", "is", "the", "how", "any", "are", "was", "her", "his",
               "does", "do", "it", "that", "this", "and", "or", "a", "an", "of"}
    stop_zh = {"是", "多少", "什么", "怎", "有", "吗", "呢", "她", "他", "的",
               "了", "在", "还", "最近", "一次", "那", "这", "和", "或"}

    def tokens(s: str) -> set:
        if is_zh:
            words = set(s) - stop_zh
        else:
            words = set(w.lower() for w in re.findall(r'\w+', s)) - stop_en
        return {w for w in words if len(w) > 1}

    t1, t2 = tokens(q1), tokens(q2)
    if not t1 or not t2:
        return False
    overlap = len(t1 & t2) / min(len(t1), len(t2))
    return overlap >= 0.5


def _pe_regen_different_item(
    original_explicit_q: str,
    prev_question: str,
    hist_str: str,
    antecedent_str: str,
    rule: str,
    ehr_snapshot: str,
    sys_prompt: str,
    is_zh: bool,
) -> str:
    """Ask the LLM to reformulate PE with a DIFFERENT data item."""
    snap_hint = ehr_snapshot[:800] if ehr_snapshot else ""
    if is_zh:
        user_prompt = f"""显式问题（第1阶段）：
"{original_explicit_q}"

问题：变换后的问题与上一轮（"{prev_question}"）询问的是同一个数据项，产生了重复。

请**选择一个不同的数据项**（从下方EHR快照中选取），重新生成一个新的显式问题，
然后应用谓词省略（PE）变换，使其与前序对话衔接自然。

对话历史：
{hist_str}
EHR快照（部分）：
{snap_hint}

改写规则（PE）：
{rule}

输出格式：直接输出改写后的省略形式问题，不要任何解释。"""
    else:
        user_prompt = f"""Explicit question (Stage 1):
"{original_explicit_q}"

Problem: the transformed question duplicates the previous turn ("{prev_question}") — same data item.

Please SELECT A DIFFERENT data item from the EHR snapshot below, generate a new explicit
question, then apply PE (Predicate Ellipsis) so it connects naturally to the prior dialogue.

Conversation history:
{hist_str}
EHR snapshot (partial):
{snap_hint}

Transformation rule (PE):
{rule}

Output: just the rewritten elliptic question, no explanation."""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = llm_call(messages, temperature=0.5, max_tokens=4000).strip().strip('"')
    return result
