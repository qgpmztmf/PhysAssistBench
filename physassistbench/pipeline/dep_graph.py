"""
physassistbench/pipeline/dep_graph.py — Turn Dependency Graph.

Tracks entities, facts, and events introduced in each turn so that
Subtype generation can anchor ellipsis to real antecedents from prior turns.

See docs/benchmark_redesign_integrated.md §5.3
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TurnNode:
    turn_idx: int
    task_type: str
    tool_source: str                        # "ehr" | "patient" | "mixed"
    subtype: str | None                     # "NA" | "PE" | "AE" | None
    entities_introduced: list[str] = field(default_factory=list)
    facts_established: list[str] = field(default_factory=list)
    events_described: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class DependencyEdge:
    src_turn: int
    tgt_turn: int
    dep_type: Literal["entity", "predicate", "argument", "situation"]
    antecedent: str


class TurnDependencyGraph:
    """
    Incrementally built during generation.  After each turn completes,
    call add_turn() to register what was introduced.  Before generating
    the next turn's question, call get_antecedents() to find what can
    be omitted/referred back to.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, TurnNode] = {}
        self.edges: list[DependencyEdge] = []

    # ── Building ──────────────────────────────────────────────────────────────

    def add_turn(
        self,
        turn_idx: int,
        task_type: str,
        tool_source: str,
        subtype: str | None,
        executed_actions: list[dict],
        assistant_answer: str,
    ) -> TurnNode:
        """
        Register a completed turn.  Extracts entities / facts / events from
        tool call names and the assistant answer (heuristic extraction).
        """
        tool_calls = [
            a["action"]["name"]
            for a in executed_actions
            if a["action"]["name"] != "prepare_to_answer"
        ]

        entities = _extract_entities(executed_actions)
        facts = _extract_facts(executed_actions, assistant_answer)
        events = _extract_events(executed_actions, assistant_answer)

        node = TurnNode(
            turn_idx=turn_idx,
            task_type=task_type,
            tool_source=tool_source,
            subtype=subtype,
            entities_introduced=entities,
            facts_established=facts,
            events_described=events,
            tool_calls=tool_calls,
        )
        self.nodes[turn_idx] = node
        return node

    def add_edge(
        self,
        src_turn: int,
        tgt_turn: int,
        dep_type: Literal["entity", "predicate", "argument", "situation"],
        antecedent: str,
    ) -> None:
        self.edges.append(
            DependencyEdge(src_turn, tgt_turn, dep_type, antecedent)
        )

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_antecedents(
        self,
        current_turn_idx: int,
        subtype: str,
    ) -> list[dict]:
        """
        Return a list of available antecedent candidates for the given subtype.
        Each item: {"turn": int, "type": str, "value": str}

        NA → entities + facts from previous turns (covers pronominalization and deletion)
        PE → tool call names (predicates) from previous turns
        AE → events / facts / complex situations from previous turns
        """
        prior_turns = [
            self.nodes[i]
            for i in range(current_turn_idx)
            if i in self.nodes
        ]
        if not prior_turns:
            return []

        candidates: list[dict] = []

        if subtype == "NA":
            for node in prior_turns:
                for ent in node.entities_introduced:
                    candidates.append({"turn": node.turn_idx, "type": "entity", "value": ent})
                for fact in node.facts_established:
                    candidates.append({"turn": node.turn_idx, "type": "argument_fact", "value": fact})

        elif subtype == "PE":
            for node in prior_turns:
                for tool in node.tool_calls:
                    candidates.append({"turn": node.turn_idx, "type": "predicate", "value": tool})

        elif subtype == "AE":
            for node in prior_turns:
                for evt in node.events_described:
                    candidates.append({"turn": node.turn_idx, "type": "event", "value": evt})
                for fact in node.facts_established:
                    candidates.append({"turn": node.turn_idx, "type": "proposition", "value": fact})
                # Situation: combine all facts across all prior turns
            if len(prior_turns) >= 2:
                situation = "; ".join(
                    f for node in prior_turns for f in node.facts_established
                )
                if situation:
                    candidates.append({
                        "turn": prior_turns[-1].turn_idx,
                        "type": "situation",
                        "value": situation[:200],
                    })

        return candidates

    def summary(self) -> dict:
        """Serialisable summary for JSONL annotation."""
        return {
            "nodes": {
                str(k): {
                    "task_type": v.task_type,
                    "tool_source": v.tool_source,
                    "subtype": v.subtype,
                    "entities": v.entities_introduced,
                    "facts": v.facts_established,
                    "events": v.events_described,
                    "tool_calls": v.tool_calls,
                }
                for k, v in self.nodes.items()
            },
            "edges": [
                {
                    "src": e.src_turn,
                    "tgt": e.tgt_turn,
                    "dep_type": e.dep_type,
                    "antecedent": e.antecedent,
                }
                for e in self.edges
            ],
        }


# ── Heuristic extractors ──────────────────────────────────────────────────────

def _extract_entities(executed_actions: list[dict]) -> list[str]:
    """Extract entity names from tool call arguments (items, drugs, diagnoses…)."""
    entities: list[str] = []
    for act in executed_actions:
        args = act.get("action", {}).get("arguments", {})
        for key in ("item", "item_name", "code", "drug", "items", "diagnosis"):
            val = args.get(key)
            if val and isinstance(val, str):
                entities.append(val)
            elif val and isinstance(val, list):
                entities.extend(str(v) for v in val)
    return list(dict.fromkeys(entities))  # deduplicate, preserve order


def _extract_facts(executed_actions: list[dict], answer: str) -> list[str]:
    """
    Extract key clinical facts.  Uses the prepare_to_answer observation
    (the final answer text) as a proxy — we take the first 2 sentences.
    """
    facts: list[str] = []
    for act in executed_actions:
        if act.get("action", {}).get("name") == "prepare_to_answer":
            obs = act.get("observation", "")
            if isinstance(obs, str) and obs.strip():
                sentences = [s.strip() for s in obs.replace("。", ".").split(".") if s.strip()]
                facts.extend(sentences[:2])
    if not facts and answer:
        sentences = [s.strip() for s in answer.replace("。", ".").split(".") if s.strip()]
        facts.extend(sentences[:2])
    return facts


def _extract_events(executed_actions: list[dict], answer: str) -> list[str]:
    """
    Events are clinical actions / recommendations described in the answer.
    Heuristic: sentences containing action verbs (给/用/补/开/调整/启动…).
    """
    EVENT_VERBS_ZH = ["给", "用", "补", "开", "调整", "启动", "建议", "推荐", "使用", "给予"]
    EVENT_VERBS_EN = ["give", "administer", "start", "initiate", "recommend", "adjust", "order"]

    events: list[str] = []
    text = answer or ""
    for act in executed_actions:
        if act.get("action", {}).get("name") == "prepare_to_answer":
            obs = act.get("observation", "")
            if isinstance(obs, str) and obs.strip():
                text = obs

    sentences = [s.strip() for s in text.replace("。", ".").split(".") if s.strip()]
    for sent in sentences:
        if any(v in sent for v in EVENT_VERBS_ZH + EVENT_VERBS_EN):
            events.append(sent[:100])
    return events[:3]
