"""
PatientAgentRuntime — session-scoped patient simulation engine.

Design principles:
- respond() returns STR only (natural language → Doctor Agent)
- Structured Direction B annotation stored as side effect in _annotation_store
- WithheldFlags: critical_withheld info hidden on general queries; revealed on
  specific drug follow-up; stays revealed once triggered
- symptom_log grows each turn to prevent multi-turn contradictions

Usage:
    register_session("mvp_s00", 10000032, persona_dict)
    rt = get_session("mvp_s00")
    response: str = rt.respond("get_chief_complaint")
    response: str = rt.respond("get_medication_adherence", drug="Lactulose")
    annotations = rt.get_annotation_store()   # eval system only
"""

import json
import logging
import os
import sys
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default PHM directory: PhysAssistBench/output/
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_PHM_DIR = os.path.join(_REPO_ROOT, "output")


# ─── PHM loader ──────────────────────────────────────────────────────────────

def _load_phm(subject_id: int, phm_dir: str = DEFAULT_PHM_DIR) -> dict:
    """Load PHM for subject_id. Tries YAML first, falls back to PHMBuilder."""
    yaml_path = os.path.join(phm_dir, f"PHM_{subject_id}.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Fallback: attempt to build on-the-fly (requires MIMIC-IV access)
    try:
        sys.path.insert(0, _REPO_ROOT)
        from physassistbench.phm.builder import PHMBuilder
        data_root = os.path.join(
            _REPO_ROOT, "data", "MIMIC-IV-split-by-patient", "split_data_each_patient"
        )
        logger.info(f"Building PHM for subject_id={subject_id} (YAML not found at {yaml_path})")
        builder = PHMBuilder(data_root=data_root)
        phm_obj = builder.build(subject_id)
        # Cache to disk so next call is instant
        try:
            builder.save_yaml(phm_obj, yaml_path)
            logger.info(f"PHM cached to {yaml_path}")
        except Exception as save_err:
            logger.warning(f"Could not cache PHM YAML: {save_err}")
        return phm_obj if isinstance(phm_obj, dict) else vars(phm_obj)
    except Exception as e:
        logger.warning(f"PHM build failed for {subject_id}: {e}. Returning empty PHM.")
        return {
            "subject_id": subject_id,
            "diagnoses": [],
            "medications": [],
            "warning_signs": [],
            "lab_trends": [],
            "persona": {},
        }


# ─── PatientAgentRuntime ─────────────────────────────────────────────────────

class PatientAgentRuntime:
    """
    Session-scoped patient simulation engine.
    One instance per (subject_id, session_id) pair.
    """

    def __init__(self, phm: dict, persona: dict, session_id: str, language: str = "en"):
        self.phm = phm
        self.persona = persona              # Fixed for the entire session
        self.session_id = session_id
        self.language = language
        self.symptom_log: list[dict] = []   # grows each turn; provides continuity
        self.withheld_flags: dict[str, bool] = {}   # node_id → has been revealed
        self._annotation_store: list[dict] = []     # Direction B (eval system only)
        # Preloaded responses from gold generation — keyed by tool name.
        # When set, respond() returns stored responses instead of calling the LLM.
        self._preloaded: dict[str, list[str]] = {}   # tool_name → [responses...]
        self._preloaded_idx: dict[str, int] = {}     # tool_name → next index

    def preload_responses(self, gold_actions: list) -> None:
        """
        Preload patient responses from gold generation data.

        Call this before evaluation to avoid LLM API calls during patient tool use.
        Responses are stored per tool name and replayed in order on each call.

        Args:
            gold_actions: answer_list entry for the Intake turn (list of action dicts)
        """
        from collections import defaultdict
        buckets: dict[str, list[str]] = defaultdict(list)
        for action in gold_actions:
            name = action.get("action", {}).get("name", "")
            obs = action.get("observation", {})
            if isinstance(obs, dict) and "patient_response" in obs:
                buckets[name].append(obs["patient_response"])
        self._preloaded = dict(buckets)
        self._preloaded_idx = {k: 0 for k in buckets}
        logger.info(
            f"Session {self.session_id!r}: preloaded responses for "
            f"{list(self._preloaded.keys())}"
        )

    def _get_preloaded(self, query_type: str) -> str | None:
        """Return next preloaded response for query_type, or None if not available."""
        tool_name = f"patient.{query_type}"
        responses = self._preloaded.get(tool_name, [])
        if not responses:
            return None
        idx = self._preloaded_idx.get(tool_name, 0)
        response = responses[idx % len(responses)]
        self._preloaded_idx[tool_name] = idx + 1
        return response

    # ── Public API ────────────────────────────────────────────────────────────

    def respond(self, query_type: str, query: str = "", drug: str = "") -> str:
        """
        Simulate a patient response to a clinical query.

        Returns: natural language patient response (str only → Doctor Agent)
        Side effects:
            - appends to self._annotation_store (eval system only)
            - appends to self.symptom_log (continuity across turns)
        """
        # Step 1: Retrieve relevant PHM nodes
        nodes = self._retrieve_nodes(query_type, query, drug)

        # Step 2: Apply WithheldFlags filter
        nodes, newly_revealed = self._apply_withheld_filter(nodes, query_type, drug)

        # Step 3: Return preloaded response if available (evaluation mode — no LLM call)
        preloaded = self._get_preloaded(query_type)
        if preloaded is not None:
            response = preloaded
        else:
            # Step 3b: Build LLM prompts and generate (generation mode)
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query_type, query, drug, nodes)
            response = self._generate_with_guardrails(system_prompt, user_prompt, query_type)

        # Step 5: Extract annotation (side effect — eval system only)
        annotation = self._extract_annotation(response, query_type, nodes, newly_revealed)
        self._annotation_store.append({
            "turn": len(self.symptom_log),
            "query_type": query_type,
            "query": query,
            "drug": drug,
            "patient_response": response,
            "annotation": annotation,
        })

        # Step 6: Update symptom_log for continuity
        self.symptom_log.append({
            "query_type": query_type,
            "query": query,
            "drug": drug,
            "summary": annotation.get("medical_entities", []),
        })

        return response  # STR only — never dict

    def get_annotation_store(self) -> list[dict]:
        """For evaluation system only. NOT exposed to Doctor Agent."""
        return self._annotation_store

    # ── PHM node retrieval ────────────────────────────────────────────────────

    @staticmethod
    def _med_name(m: dict) -> str:
        """Extract medication name from PHM medication dict (handles 'drug' or 'name' key)."""
        return str(m.get("drug", m.get("name", m.get("drug_name", ""))))

    @staticmethod
    def _is_critical_med(m: dict) -> bool:
        """
        Returns True if a medication should be treated as critical withheld info.
        Criteria: critical_flag=True, or adherence in (poor, never_filled, not_taking),
        or current_status in (not_taking, never_filled).
        """
        if m.get("critical_flag"):
            return True
        adh = str(m.get("adherence", "")).lower()
        status = str(m.get("current_status", "")).lower()
        return adh in ("poor", "never_filled", "not_taking") or status in ("not_taking", "never_filled")

    def _retrieve_nodes(self, query_type: str, query: str, drug: str) -> list[dict]:
        """Return PHM nodes relevant to the query type."""
        phm = self.phm
        nodes: list[dict] = []

        if query_type == "get_chief_complaint":
            # Chief complaint: diagnoses + warning signs
            nodes += [{"type": "diagnosis", **d} for d in phm.get("diagnoses", [])[:3]]
            nodes += [{"type": "warning_sign", **w} for w in phm.get("warning_signs", [])]

        elif query_type == "get_symptom_history":
            # Symptom history: diagnoses + lab trends (filtered by query keyword)
            all_dx = [{"type": "diagnosis", **d} for d in phm.get("diagnoses", [])]
            all_labs = [{"type": "lab_trend", **l} for l in phm.get("lab_trends", [])]
            if query:
                q_lower = query.lower()
                nodes += [n for n in all_dx if q_lower in str(n).lower()]
                nodes += [n for n in all_labs if q_lower in str(n).lower()]
                if not nodes:
                    nodes = all_dx[:2] + all_labs[:2]
            else:
                nodes = all_dx[:3] + all_labs[:2]

        elif query_type == "get_medication_adherence":
            # Medication: filter by drug name if provided
            all_meds = phm.get("medications", [])
            if drug:
                drug_lower = drug.lower()
                nodes = [{"type": "medication", **m} for m in all_meds
                         if drug_lower in self._med_name(m).lower()]
                if not nodes:
                    nodes = [{"type": "medication", **m} for m in all_meds[:2]]
            else:
                nodes = [{"type": "medication", **m} for m in all_meds[:3]]

        elif query_type == "get_social_history":
            # Social history from persona + diagnoses
            nodes = [{"type": "persona", **phm.get("persona", {})}]
            nodes += [{"type": "diagnosis", **d} for d in phm.get("diagnoses", [])[:2]]

        elif query_type == "get_functional_status":
            # Functional status: warning signs + diagnoses
            nodes = [{"type": "warning_sign", **w} for w in phm.get("warning_signs", [])]
            nodes += [{"type": "diagnosis", **d} for d in phm.get("diagnoses", [])[:2]]

        elif query_type == "get_pain_assessment":
            # Pain: warning signs + symptom-related diagnoses
            nodes = [{"type": "warning_sign", **w} for w in phm.get("warning_signs", [])]
            nodes += [{"type": "diagnosis", **d} for d in phm.get("diagnoses", [])
                      if any(kw in str(d).lower() for kw in ["pain", "acute", "abdo"])][:3]
            if not nodes:
                nodes = [{"type": "diagnosis", **d} for d in phm.get("diagnoses", [])[:2]]

        return nodes

    # ── WithheldFlags filter ──────────────────────────────────────────────────

    def _apply_withheld_filter(
        self, nodes: list[dict], query_type: str, drug: str
    ) -> tuple[list[dict], list[str]]:
        """
        Apply critical_withheld persona behavior.

        Rules:
        - If info_completeness != "critical_withheld" → pass all nodes through
        - On general queries: filter out critical_flag=True nodes not yet revealed
        - On get_medication_adherence with specific drug → force reveal critical nodes
        - Once revealed, stays revealed (withheld_flags[node_id] = True)

        Returns: (filtered_nodes, list_of_newly_revealed_node_ids)
        """
        info_completeness = self.persona.get("info_completeness", "full")
        newly_revealed: list[str] = []

        if info_completeness != "critical_withheld":
            return nodes, newly_revealed

        # Determine if this is a specific drug follow-up that should force reveal
        force_reveal = (
            query_type == "get_medication_adherence"
            and bool(drug)
        )

        filtered: list[dict] = []
        for node in nodes:
            node_id = str(node.get("drug", node.get("name", node.get("drug_name",
                          node.get("condition", node.get("description", id(node)))))))
            node_type = node.get("type", "")
            if node_type == "medication":
                is_critical = self._is_critical_med(node)
            else:
                is_critical = node.get("critical_flag", False)
            already_revealed = self.withheld_flags.get(node_id, False)

            if not is_critical:
                filtered.append(node)
            elif already_revealed:
                # Previously revealed — always include
                filtered.append(node)
            elif force_reveal:
                # Drug-specific follow-up triggers reveal
                self.withheld_flags[node_id] = True
                newly_revealed.append(node_id)
                filtered.append(node)
            else:
                # Withhold: don't include in context (patient hasn't mentioned it yet)
                pass

        return filtered, newly_revealed

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        persona = self.persona
        literacy = persona.get("health_literacy", "medium")
        adherence = persona.get("adherence", "uncertain")
        anxiety = persona.get("anxiety_level", "medium")

        if self.language == "zh":
            literacy_desc = {
                "low": "使用简单的日常用语，避免医学术语，可能误解专业词汇",
                "medium": "了解基本医学概念，但遇到复杂术语时会要求解释",
                "high": "具备医学素养，使用正确的专业术语，能准确描述症状",
            }.get(literacy, "使用日常语言")

            adherence_desc = {
                "good": "按处方服用所有药物，严格遵从医嘱",
                "uncertain": "有时忘记服药，或对用药时间安排不确定",
                "poor": "经常漏服，已停用部分药物，或从未取药",
            }.get(adherence, "对用药依从性不确定")

            anxiety_desc = {
                "low": "对症状态度平静，陈述客观",
                "medium": "略有焦虑，有时使用带情绪色彩的语言",
                "high": "明显焦虑，对病情感到担忧，可能会强调最严重的症状",
            }.get(anxiety, "对病情有些担忧")

            return f"""你正在模拟一名正在接受医生临床访谈的患者。

患者性格：
- 健康素养：{literacy} — {literacy_desc}
- 用药依从性：{adherence} — {adherence_desc}
- 焦虑程度：{anxiety} — {anxiety_desc}

你将接收临床PHM数据和之前的对话背景。
你的任务是按照该患者的性格特点，以自然的方式回应医生的提问。

规则：
1. 以患者的口吻用自然的中文口语作答
2. 严格保持与上述性格设定一致
3. 你的回答必须仅基于所提供的PHM数据节点
4. 不得编造PHM数据中没有的症状或药物
5. 若健康素养较低，不得使用医学术语
6. 若被问及从未取过的药物，请自然地表达出来
7. 对于症状史（OPQRST）：描述起病时间、诱发因素、性质、放射、严重程度、时间规律
8. 保持简洁（2-5句话），除非被追问细节
9. 与之前已告知医生的内容保持一致"""

        literacy_desc = {
            "low": "uses simple everyday words, avoids medical terms, may misunderstand jargon",
            "medium": "understands basic medical concepts but asks for clarification on complex terms",
            "high": "medically literate, uses correct terminology, describes symptoms precisely",
        }.get(literacy, "uses everyday language")

        adherence_desc = {
            "good": "takes all medications as prescribed and follows medical advice closely",
            "uncertain": "sometimes forgets doses or is unsure about medication schedules",
            "poor": "often misses doses, has stopped some medications, or never filled some prescriptions",
        }.get(adherence, "is uncertain about medication compliance")

        anxiety_desc = {
            "low": "calm and matter-of-fact about their symptoms",
            "medium": "somewhat anxious and sometimes uses emotional language",
            "high": "visibly anxious, worried about their condition, may emphasize worst symptoms",
        }.get(anxiety, "is somewhat concerned about their condition")

        return f"""You are simulating a patient in a clinical interview with a doctor.

Patient personality:
- Health literacy: {literacy} — {literacy_desc}
- Medication adherence: {adherence} — {adherence_desc}
- Anxiety level: {anxiety} — {anxiety_desc}

You will receive clinical PHM data and prior conversation context.
Your job is to respond as this patient would — naturally, in character.

Rules:
1. Respond in natural spoken English as the patient
2. Stay strictly in character based on the personality above
3. Base your response ONLY on the provided PHM data nodes
4. Do NOT invent symptoms or medications not in the PHM data
5. Do NOT use medical jargon if health_literacy is low
6. If asked about a medication you never filled, express this naturally
7. For symptom history (OPQRST): describe Onset, Provocation, Quality, Radiation, Severity, Timing
8. Keep responses concise (2-5 sentences) unless probed for details
9. Stay consistent with prior conversation turns shown in context"""

    def _build_user_prompt(
        self, query_type: str, query: str, drug: str, nodes: list[dict]
    ) -> str:
        if self.language == "zh":
            query_desc = {
                "get_chief_complaint": "医生正在询问您今天就诊的主诉。",
                "get_symptom_history": f"医生正在询问您的症状史{f'（相关症状：{query}）' if query else ''}。",
                "get_medication_adherence": f"医生正在询问您是否按医嘱服用{drug or '您的药物'}。",
                "get_social_history": "医生正在询问您的生活情况、生活习惯和社会背景。",
                "get_functional_status": "医生正在询问您日常活动的状况以及症状是否影响您的功能。",
                "get_pain_assessment": "医生正在请您描述您所经历的疼痛。",
            }.get(query_type, f"医生正在询问：{query}")

            nodes_text = ""
            if nodes:
                node_lines = []
                for n in nodes:
                    ntype = n.get("type", "")
                    if ntype == "diagnosis":
                        desc = n.get("description", n.get("icd_description", ""))
                        node_lines.append(f"  [诊断] {desc}")
                    elif ntype == "medication":
                        name = self._med_name(n)
                        adh = n.get("adherence", n.get("current_status", "unknown"))
                        patient_expl = n.get("patient_explanation", "")
                        is_crit = self._is_critical_med(n)
                        node_lines.append(
                            f"  [药物] {name} — 依从性：{adh}{' （严重：患者未按处方服药）' if is_crit else ''}"
                            + (f"\n    患者本人的说法：{patient_expl}" if patient_expl else "")
                        )
                    elif ntype == "warning_sign":
                        desc = n.get("description", "")
                        urgency = n.get("urgency_level", "")
                        node_lines.append(f"  [预警信号] {desc}（紧急程度：{urgency}）")
                    elif ntype == "lab_trend":
                        item = n.get("item_name", "")
                        trend = n.get("trend_direction", "")
                        node_lines.append(f"  [化验趋势] {item}：{trend}")
                    elif ntype == "persona":
                        for k in ["occupation", "living_situation", "smoking", "alcohol", "family_support"]:
                            if k in n:
                                node_lines.append(f"  [社会信息] {k}：{n[k]}")
                nodes_text = "您的相关医疗信息：\n" + "\n".join(node_lines)

            prior_context = ""
            if self.symptom_log:
                prior_context = "\n之前与医生交流的内容（您已告诉医生的信息）：\n"
                for entry in self.symptom_log[-3:]:
                    prior_context += f"  - 您讨论了 {entry['query_type']}（{entry.get('query', '')}）\n"

            return f"""{query_desc}

{nodes_text}

{prior_context}

请以该患者的口吻自然地回答："""

        query_desc = {
            "get_chief_complaint": "The doctor is asking you what brought you in today (chief complaint).",
            "get_symptom_history": f"The doctor is asking about your symptom history{f' related to: {query}' if query else ''}.",
            "get_medication_adherence": f"The doctor is asking whether you are taking {drug or 'your medications'} as prescribed.",
            "get_social_history": "The doctor is asking about your living situation, habits, and social background.",
            "get_functional_status": "The doctor is asking how you're managing daily activities and whether symptoms limit your function.",
            "get_pain_assessment": "The doctor is asking you to describe any pain you are experiencing.",
        }.get(query_type, f"The doctor is asking: {query}")

        # Format PHM nodes for context
        nodes_text = ""
        if nodes:
            node_lines = []
            for n in nodes:
                ntype = n.get("type", "")
                if ntype == "diagnosis":
                    desc = n.get("description", n.get("icd_description", ""))
                    node_lines.append(f"  [Diagnosis] {desc}")
                elif ntype == "medication":
                    name = self._med_name(n)
                    adh = n.get("adherence", n.get("current_status", "unknown"))
                    patient_expl = n.get("patient_explanation", "")
                    is_crit = self._is_critical_med(n)
                    node_lines.append(
                        f"  [Medication] {name} — adherence: {adh}{' (CRITICAL: patient not taking as prescribed)' if is_crit else ''}"
                        + (f"\n    Patient's own words: {patient_expl}" if patient_expl else "")
                    )
                elif ntype == "warning_sign":
                    desc = n.get("description", "")
                    urgency = n.get("urgency_level", "")
                    node_lines.append(f"  [Warning Sign] {desc} (urgency: {urgency})")
                elif ntype == "lab_trend":
                    item = n.get("item_name", "")
                    trend = n.get("trend_direction", "")
                    node_lines.append(f"  [Lab Trend] {item}: {trend}")
                elif ntype == "persona":
                    # Extract relevant social history fields
                    for k in ["occupation", "living_situation", "smoking", "alcohol", "family_support"]:
                        if k in n:
                            node_lines.append(f"  [Social] {k}: {n[k]}")
            nodes_text = "Your relevant medical information:\n" + "\n".join(node_lines)

        # Prior conversation context
        prior_context = ""
        if self.symptom_log:
            prior_context = "\nPrior conversation context (what you've already told the doctor):\n"
            for entry in self.symptom_log[-3:]:
                prior_context += f"  - You discussed {entry['query_type']} ({entry.get('query', '')})\n"

        return f"""{query_desc}

{nodes_text}

{prior_context}

Respond naturally as this patient:"""

    # ── LLM generation ────────────────────────────────────────────────────────

    def _generate_with_guardrails(
        self, system_prompt: str, user_prompt: str, query_type: str
    ) -> str:
        """Call the LLM and apply post-generation guardrails."""
        # Import here to avoid circular imports
        from physassistbench.pipeline.agents.llm_client import llm_call

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = llm_call(messages, temperature=0.7, max_tokens=300)
            response = response.strip().strip('"')

            # Guardrail: remove any JSON artifacts
            if response.startswith("{") or response.startswith("["):
                # Extract just the text content if there's a "response" key
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        response = (
                            parsed.get("response")
                            or parsed.get("patient_response")
                            or parsed.get("text")
                            or str(parsed)
                        )
                except Exception:
                    pass

            return response
        except Exception as e:
            logger.warning(f"PatientAgentRuntime LLM call failed: {e}. Using fallback response.")
            return self._fallback_response(query_type)

    def rewrite_literacy(self, high_text: str, target_level: str) -> str:
        """Style-rewrite a high-literacy patient response to a lower literacy level.

        Lower-literacy patients describe the SAME clinical reality but report it
        less precisely: they may forget exact numbers, give vague descriptions, or
        express uncertainty. Crucially they NEVER state a DIFFERENT value than the
        high-literacy version — they only drop precision or omit details. This keeps
        the underlying clinical facts consistent across variants while realistically
        modelling information loss at lower health literacy.
        """
        import random
        from physassistbench.pipeline.agents.llm_client import llm_call

        if not high_text or not high_text.strip():
            return high_text
        if target_level == "high":
            return high_text

        # Probabilistically apply "forgetfulness" — lower literacy = more likely to
        # be vague / unable to recall specific values.
        vague_prob = {"medium": 0.30, "low": 0.60}.get(target_level, 0.0)
        be_vague = random.random() < vague_prob

        is_zh = (self.language == "zh")
        if is_zh:
            style = {
                "medium": "了解基本医学概念但不太用专业术语，措辞偏日常",
                "low": "只用最朴素的日常口语，不用任何医学术语",
            }.get(target_level, "日常口语")
            if be_vague:
                vague_rule = (
                    "3. 这位患者记不太清具体数值——请把精确的数字/度数模糊化或省略，"
                    "改成口语化的模糊表述（如「好像有点发烧」「具体多少度我没记住」"
                    "「记不太清了」「就觉得不太舒服」）。但绝不能说出与原文不同的数字。"
                )
            else:
                vague_rule = (
                    "3. 数值可以保留，但用更口语的方式说（如「一百度左右」）；绝不改变实际数字"
                )
            prompt = (
                f"下面是一位高健康素养患者的回答。请将它改写成「{target_level}」素养水平的说法：\n"
                f"风格要求：{style}\n\n"
                f"严格规则：\n"
                f"1. 描述的是同一个临床事实，绝不能编造或改变与原文矛盾的信息\n"
                f"2. 只改变用词和表达方式（专业度、口语化、精确度）\n"
                f"{vague_rule}\n"
                f"4. 保持患者第一人称口吻，2-4句话\n"
                f"5. 只输出改写后的回答，不要解释\n\n"
                f"高素养原文：\n{high_text}\n\n"
                f"{target_level}素养改写："
            )
        else:
            style = {
                "medium": "understands basic medical concepts but uses few technical terms, "
                          "phrases things casually",
                "low": "uses only plain everyday language, NO medical jargon",
            }.get(target_level, "everyday language")
            if be_vague:
                vague_rule = (
                    "3. This patient does NOT recall the exact figures — replace precise "
                    "numbers/temperatures with vague colloquial descriptions or express "
                    "uncertainty (e.g. 'I think I had a bit of a fever', \"I didn't check "
                    "the exact number\", \"I don't really remember\", 'I just felt off'). "
                    "But NEVER state a number that differs from the original."
                )
            else:
                vague_rule = (
                    "3. Keep the numbers but phrase them colloquially (e.g. 'about a "
                    "hundred degrees'); never change the actual values"
                )
            prompt = (
                f"Below is a high-health-literacy patient's response. Rewrite it at the "
                f"'{target_level}' literacy level.\nStyle: {style}\n\n"
                f"STRICT RULES:\n"
                f"1. It describes the SAME clinical reality — never invent or state info "
                f"that contradicts the original\n"
                f"2. Only change vocabulary, phrasing, and precision\n"
                f"{vague_rule}\n"
                f"4. Keep first-person patient voice, 2-4 sentences\n"
                f"5. Output ONLY the rewritten response, no explanation\n\n"
                f"High-literacy original:\n{high_text}\n\n"
                f"{target_level}-literacy rewrite:"
            )
        try:
            out = llm_call(
                [{"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=300,
            ).strip().strip('"')
            return out or high_text
        except Exception as e:
            logger.warning(f"rewrite_literacy failed: {e}")
            return high_text

    def _fallback_response(self, query_type: str) -> str:
        """Simple rule-based fallback when LLM is unavailable."""
        phm = self.phm
        if self.language == "zh":
            if query_type == "get_chief_complaint":
                dx = phm.get("diagnoses", [])
                if dx:
                    desc = dx[0].get("description", dx[0].get("icd_description", "一些健康问题"))
                    return f"我一直有{desc}的问题，这是我今天来的主要原因。"
                return "我最近感觉不太好，所以医生让我过来检查一下。"
            elif query_type == "get_medication_adherence":
                return "我尽量按时服药，但说实话，我确实漏服过几次。"
            elif query_type == "get_pain_assessment":
                ws = phm.get("warning_signs", [])
                if ws:
                    return f"我一直有{ws[0].get('description', '一些不适').lower()}。"
                return "我一直有些疼痛，大概是十分中的五分吧。"
            else:
                return "我一直在尽力坚持，但确实有些困难。"
        if query_type == "get_chief_complaint":
            dx = phm.get("diagnoses", [])
            if dx:
                desc = dx[0].get("description", dx[0].get("icd_description", "some health issues"))
                return f"I've been having problems with {desc.lower()}. That's mainly why I came in."
            return "I haven't been feeling well lately, so the doctor sent me in."
        elif query_type == "get_medication_adherence":
            return "I try to take my medications but I'll be honest, I've missed some doses."
        elif query_type == "get_pain_assessment":
            ws = phm.get("warning_signs", [])
            if ws:
                return f"I've been having {ws[0].get('description', 'some discomfort').lower()}."
            return "I've been having some pain, maybe a 5 out of 10."
        else:
            return "I've been managing as best I can. It's been difficult."

    # ── Direction B annotation ─────────────────────────────────────────────────

    def _extract_annotation(
        self,
        response: str,
        query_type: str,
        nodes: list[dict],
        newly_revealed: list[str],
    ) -> dict:
        """
        Extract structured annotation from the patient response.
        This is the Direction B annotation — stored for eval system only.
        """
        # Identify critical nodes that were part of this turn
        critical_nodes = []
        for n in nodes:
            ntype = n.get("type", "")
            if ntype == "medication" and self._is_critical_med(n):
                critical_nodes.append(self._med_name(n))
            elif n.get("critical_flag", False):
                critical_nodes.append(str(n.get("condition", n.get("description", ""))))

        # Simple keyword extraction for medical entities
        medical_entities: list[str] = []
        response_lower = response.lower()
        for node in nodes:
            ntype = node.get("type", "")
            if ntype == "medication":
                name = self._med_name(node).split()[0]  # First word of drug name
            else:
                name = str(node.get("condition", node.get("description", "")))
            if name and name.lower() in response_lower:
                medical_entities.append(name)

        def _node_label(n: dict) -> str:
            ntype = n.get("type", "")
            if ntype == "medication":
                return self._med_name(n)
            return str(n.get("condition", n.get("description", n.get("drug", ""))))

        return {
            "medical_entities": medical_entities,
            "phm_nodes_used": [_node_label(n) for n in nodes],
            "critical_flags_in_context": critical_nodes,
            "newly_revealed": newly_revealed,
            "critical_flags_triggered": bool(newly_revealed),
        }


# ─── Session registry ─────────────────────────────────────────────────────────

# session_id → PatientAgentRuntime
_sessions: dict[str, PatientAgentRuntime] = {}

# session_id → {subject_id, persona}
SESSION_CONFIG: dict[str, dict] = {}


def register_session(
    session_id: str,
    subject_id: int,
    persona: dict,
    phm_dir: str = DEFAULT_PHM_DIR,
    language: str = "en",
) -> None:
    """
    Register a new benchmark session. Must be called before any patient tool calls.

    Args:
        session_id: Unique session identifier (e.g., "mvp_s00")
        subject_id: MIMIC-IV patient identifier
        persona: Persona config dict with keys:
                 health_literacy, adherence, anxiety_level, info_completeness
        phm_dir: Directory containing PHM_<subject_id>.yaml files
        language: Prompt language ("en" or "zh")
    """
    phm = _load_phm(subject_id, phm_dir)
    _sessions[session_id] = PatientAgentRuntime(phm, persona, session_id, language=language)
    SESSION_CONFIG[session_id] = {"subject_id": subject_id, "persona": persona}
    logger.info(f"Registered session {session_id!r} for subject_id={subject_id}")


def get_session(session_id: str) -> PatientAgentRuntime:
    """Retrieve a registered session by ID."""
    if session_id not in _sessions:
        raise KeyError(
            f"Session {session_id!r} not registered. "
            "Call register_session() before using patient tools."
        )
    return _sessions[session_id]


def reset_all_sessions() -> None:
    """Clear all registered sessions (useful between test runs)."""
    _sessions.clear()
    SESSION_CONFIG.clear()
