"""
Metrics computation for PhysAssistBench evaluation.

Extends WildToolBench's 6-dimension metrics with 3 clinical dimensions:
  - hallucination_rate
  - safety_violation_rate
  - temporal_accuracy
"""

from collections import defaultdict
from typing import List, Dict, Any


def calc_accuracy(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute accuracy across all evaluation dimensions.

    results: list of per-turn result dicts from eval_runner, each containing:
        - test_entry_id
        - task_idx
        - task_type (Single-Tool / Multi-Tool / Clarify / Chat)
        - turn_subtype (Partial Information / Coreferential Reference / Long-Range Dependency / None)
        - clinical_task_domain (LabInterp / MedRecon / etc.)
        - is_correct (bool)
        - is_optimal (bool | None)  — for parallel multi-tool
        - complete_steps (int)
        - total_steps (int)
        - hallucination (bool)
        - safety_violation (bool)
        - temporal_correct (bool | None)
    """
    total = len(results)
    if total == 0:
        return {"error": "No results"}

    # Helper accumulators
    def acc(items, key=None):
        if not items:
            return None
        if key:
            return sum(1 for x in items if x.get(key)) / len(items)
        return sum(1 for x in items if x.get("is_correct")) / len(items)

    # 1. Total accuracy
    metrics = {
        "total_accuracy": acc(results),
        "n_total": total,
    }

    # 2. Accuracy by task type
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("task_type", "Unknown")].append(r)
    metrics["task_type_accuracy"] = {
        t: acc(items) for t, items in by_type.items()
    }

    # 3. Accuracy by clinical domain
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r.get("clinical_task_domain", "Unknown")].append(r)
    metrics["domain_accuracy"] = {
        d: acc(items) for d, items in by_domain.items()
    }

    # 4. Accuracy by turn subtype
    by_subtype = defaultdict(list)
    for r in results:
        st = r.get("turn_subtype")
        if st:
            by_subtype[st].append(r)
    metrics["subtype_accuracy"] = {
        s: acc(items) for s, items in by_subtype.items()
    }

    # 5. Turn-layer accuracy (turn 0=first, 1=second, 2=third)
    by_layer = defaultdict(list)
    for r in results:
        by_layer[r.get("task_idx", 0)].append(r)
    metrics["layer_accuracy"] = {
        f"turn_{i}": acc(items) for i, items in sorted(by_layer.items())
    }

    # 6. Optimality (for Multi-Tool tasks)
    mt_results = [r for r in results if "Multi-Tool" in r.get("task_type", "")
                  and r.get("is_optimal") is not None]
    metrics["optimality"] = acc(mt_results, "is_optimal") if mt_results else None

    # 7. Progress ratio (for sequential Multi-Tool)
    progress_items = [r for r in results if r.get("total_steps", 0) > 0]
    if progress_items:
        metrics["progress_ratio"] = sum(
            r.get("complete_steps", 0) / r.get("total_steps", 1)
            for r in progress_items
        ) / len(progress_items)

    # 8. Clinical dimensions
    metrics["hallucination_rate"] = acc(results, "hallucination")
    metrics["safety_violation_rate"] = acc(results, "safety_violation")
    temporal_items = [r for r in results if r.get("temporal_correct") is not None]
    metrics["temporal_accuracy"] = acc(temporal_items, "temporal_correct") if temporal_items else None

    # 9. Session-level accuracy (all turns in a session must be correct)
    sessions = defaultdict(list)
    for r in results:
        sessions[r.get("test_entry_id", "")].append(r)
    session_correct = sum(
        1 for items in sessions.values() if all(r.get("is_correct") for r in items)
    )
    metrics["session_accuracy"] = session_correct / len(sessions) if sessions else None
    metrics["n_sessions"] = len(sessions)

    # 10. Intake coverage: did the LLM call at least one patient.xxx tool on Intake turns?
    intake_turns = [r for r in results if r.get("task_type") == "Intake"]
    if intake_turns:
        metrics["intake_coverage"] = acc(intake_turns, key="tool_coverage_correct")
    else:
        metrics["intake_coverage"] = None

    # 11. Critical symptom coverage: proportion of critical PHM nodes asked about
    # Fields set by eval_runner.py: critical_symptoms_total, critical_symptoms_covered
    critical_turns = [r for r in results if "critical_symptoms_total" in r]
    if critical_turns:
        total = sum(r.get("critical_symptoms_total", 0) for r in critical_turns)
        covered = sum(r.get("critical_symptoms_covered", 0) for r in critical_turns)
        metrics["critical_symptom_coverage"] = covered / total if total > 0 else 0.0
    else:
        metrics["critical_symptom_coverage"] = None

    return metrics


def format_metrics_report(metrics: Dict) -> str:
    """Format metrics dict as a human-readable report."""
    lines = [
        "=" * 60,
        "PhysAssistBench Evaluation Results",
        "=" * 60,
        f"Total turns: {metrics.get('n_total', 0)}",
        f"Total sessions: {metrics.get('n_sessions', 0)}",
        "",
        f"Overall turn accuracy:    {metrics.get('total_accuracy', 0):.3f}",
        f"Session accuracy:          {metrics.get('session_accuracy', 0):.3f}",
        "",
        "── By Clinical Domain ──",
    ]
    for domain, acc in sorted((metrics.get("domain_accuracy") or {}).items()):
        lines.append(f"  {domain:<20} {acc:.3f}")

    lines += ["", "── By Task Type ──"]
    for tt, acc in sorted((metrics.get("task_type_accuracy") or {}).items()):
        lines.append(f"  {tt:<25} {acc:.3f}")

    lines += ["", "── By Turn Subtype ──"]
    for st, acc in sorted((metrics.get("subtype_accuracy") or {}).items()):
        lines.append(f"  {st:<30} {acc:.3f}")

    lines += ["", "── By Turn Layer ──"]
    for layer, acc in sorted((metrics.get("layer_accuracy") or {}).items()):
        lines.append(f"  {layer:<10} {acc:.3f}")

    lines += ["", "── Clinical Dimensions ──"]
    if metrics.get("hallucination_rate") is not None:
        lines.append(f"  Hallucination rate:      {metrics['hallucination_rate']:.3f}")
    if metrics.get("safety_violation_rate") is not None:
        lines.append(f"  Safety violation rate:   {metrics['safety_violation_rate']:.3f}")
    if metrics.get("temporal_accuracy") is not None:
        lines.append(f"  Temporal accuracy:       {metrics['temporal_accuracy']:.3f}")
    if metrics.get("optimality") is not None:
        lines.append(f"  Optimality (MT tasks):   {metrics['optimality']:.3f}")
    if metrics.get("progress_ratio") is not None:
        lines.append(f"  Progress ratio:          {metrics['progress_ratio']:.3f}")

    lines += ["", "── Patient Interview ──"]
    if metrics.get("intake_coverage") is not None:
        lines.append(f"  Intake coverage:         {metrics['intake_coverage']:.3f}")
    if metrics.get("critical_symptom_coverage") is not None:
        lines.append(f"  Critical symptom cov.:   {metrics['critical_symptom_coverage']:.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)
