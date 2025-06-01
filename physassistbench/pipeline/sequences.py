"""
physassistbench/pipeline/sequences.py — Task type sequences and subtype definitions.

Core design (physassistbench: 27-arc system):
  - T0 always = Information Lookup (anchors each session in a concrete data request)
  - T1, T2 each from {Information Lookup, Data Gathering, Clinical Reasoning}  → 3 × 3 = 9 prefix combinations
  - T3 from {Data Gathering, Clinical Reasoning, Action}              → 3 options (Information Lookup excluded)
  - Total: 3 × 3 × 3 = 27 unique arcs, indexed 0–26

  Rationale for excluding Information Lookup at T3:
    The final turn should always carry reasoning depth — either multi-tool analysis (Data Gathering),
    knowledge-grounded clinical decision (Clinical Reasoning), or a write operation (Action).
    A single-point retrieval as the closing turn adds no analytical value.

  Arc eligibility by difficulty:
    L1 (or None): all 27 arcs
    L2: 24 arcs — T1/T2 contain ≥1 Data Gathering or Clinical Reasoning
                  (excludes R→R→{W,KG,Action} where T1=T2=Information Lookup)
    L3: 18 arcs — T1/T2 contain ≥1 Clinical Reasoning

  3 linguistic subtypes (NA/PE/AE) assigned dynamically per turn.
  Priority: always trigger a subtype if feasible; balance counts via counter.
"""

from itertools import product
import random

# ── Task types ────────────────────────────────────────────────────────────────

TASK_TYPES = ["Information Lookup", "Data Gathering", "Clinical Reasoning"]

TASK_TYPE_DESCRIPTIONS = {
    "Information Lookup": "Single tool call (EHR or Patient) to fetch one specific data point.",
    "Data Gathering": "Two or more tools, parallel or conditional branching.",
    "Clinical Reasoning": "One tool fetches a patient parameter; knowledge reasoning produces individualised advice.",
    "Write/Update": "Write operation: create a medication order, service referral, or clinical flag.",
}

# ── Subtypes ──────────────────────────────────────────────────────────────────

SUBTYPES = ["NA", "PE", "AE"]

SUBTYPE_FULL = {
    "NA": "Nominal Anaphora",
    "PE": "Predicate Ellipsis",
    "AE": "Abstract/Event Anaphora",
}

# Probability weights for subtype assignment (turns 1-3); used by legacy gen_subtypes()
_SUBTYPE_WEIGHTS = [1/3, 1/3, 1/3]   # NA, PE, AE (equal)

# ── 27-arc system ─────────────────────────────────────────────────────────────
# T0=R (fixed), T1/T2 ∈ {R,W,KG}, T3 ∈ {W,KG,Action}
# Information Lookup is intentionally excluded from T3: the final turn must carry reasoning depth.
_T3_TYPES = ["Data Gathering", "Clinical Reasoning", "Write/Update"]

ALL_ARCS: list[tuple[str, ...]] = [
    ("Information Lookup",) + t1t2 + (t3,)
    for t1t2 in product(TASK_TYPES, repeat=2)
    for t3 in _T3_TYPES
]
# len(ALL_ARCS) == 27, indexed 0–26

# Convenience views
_ACTION_ARCS:     list[tuple[str, ...]] = [a for a in ALL_ARCS if a[-1] == "Write/Update"]
_NON_ACTION_ARCS: list[tuple[str, ...]] = [a for a in ALL_ARCS if a[-1] != "Write/Update"]


def get_arc(arc_idx: int) -> list[str]:
    """Return the 4-turn task-type sequence for the given arc index (0–26)."""
    return list(ALL_ARCS[arc_idx])


def get_eligible_arc_indices(difficulty: int | None) -> list[int]:
    """Return arc indices eligible for the given difficulty level.

    Eligibility is assessed on T1 and T2 (the middle turns):

    L1 (or None): all 27 arcs
    L2: 24 arcs — T1/T2 contain ≥1 Data Gathering or Clinical Reasoning
                  (excludes 3 arcs where T1=T2=Information Lookup)
    L3: 18 arcs — T1/T2 contain ≥1 Clinical Reasoning
    """
    if difficulty is None or difficulty == 1:
        return list(range(len(ALL_ARCS)))
    eligible = []
    for i, arc in enumerate(ALL_ARCS):
        inner = arc[1:3]  # T1, T2 only
        if difficulty == 2:
            if any(t in ("Data Gathering", "Clinical Reasoning") for t in inner):
                eligible.append(i)
        else:  # difficulty >= 3
            if any(t == "Clinical Reasoning" for t in inner):
                eligible.append(i)
    return eligible


# ── Legacy 81-sequence system (kept for backward compatibility) ───────────────

ALL_SEQUENCES: list[tuple[str, ...]] = list(product(TASK_TYPES, repeat=4))
# len(ALL_SEQUENCES) == 81, indexed 0–80


def get_sequence(sequence_idx: int) -> list[str]:
    """Return the task-type sequence for the given index (0–80). Legacy use only."""
    return list(ALL_SEQUENCES[sequence_idx])


def gen_subtypes(n_turns: int = 4) -> list[str | None]:
    """Legacy random subtype assignment. Kept for backward compatibility only.
    Use pick_subtype() in new code."""
    result: list[str | None] = [None]
    for _ in range(n_turns - 1):
        result.append(random.choices(SUBTYPES, weights=_SUBTYPE_WEIGHTS, k=1)[0])
    return result


def pick_subtype(
    turn_idx: int,
    dep_graph,  # TurnDependencyGraph — duck-typed to avoid circular import
    task_type: str,
    subtype_counter: dict[str, int] | None = None,
) -> str | None:
    """
    Dynamically select a subtype for the current turn.

    Strategy:
      1. Check all 4 subtypes for feasibility via dep_graph.get_antecedents().
         A subtype is feasible iff it returns at least one antecedent candidate.
      2. AE is skipped for Information Lookup turns (user_agent_v2 downgrades AE→PE for
         Information Lookup anyway; skipping here avoids counting AE while applying PE).
      3. Among feasible subtypes, pick the one with the lowest count in
         subtype_counter so that triggered subtypes stay balanced across a run.
         Falls back to uniform random when subtype_counter is None.

    Returns None only if turn_idx == 0 (no prior context) or dep_graph has no
    prior turns registered yet.
    """
    if turn_idx == 0:
        return None

    feasible: list[str] = []
    for st in SUBTYPES:
        if task_type == "Information Lookup" and st == "AE":
            continue  # would be silently downgraded to PE; count under PE instead
        if dep_graph.get_antecedents(turn_idx, st):
            feasible.append(st)

    if not feasible:
        return None

    if subtype_counter is None:
        return random.choice(feasible)

    return min(feasible, key=lambda s: subtype_counter.get(s, 0))
