"""
Tool Executor — executes planned tool calls against real MIMIC-IV data.

This is the KEY innovation vs WildToolBench: observations come from REAL
patient records, not LLM-generated synthetic values.

Returns a list of (action_dict, observation_dict) pairs, with dependency tracking.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from physassistbench.tools.tool_registry import call_tool


def execute_actions(
    action_list: list,
    subject_id: int,
    hadm_id: int | None = None,
    tool_set: str = "auto",
) -> list:
    """
    Execute all actions in the list (respecting dependencies via sequential execution).
    Returns list of dicts:
      {"action": {"name": ..., "arguments": ...},
       "observation": <real MIMIC data or error>,
       "dependency_list": [...],
       "idx": i}
    """
    results = []
    observations = {}  # idx → observation, for dependency injection

    for i, action in enumerate(action_list):
        name = action.get("name", "")
        args = dict(action.get("arguments", {}))

        # ask_user_for_required_parameters is handled by the clarify loop in generate.py,
        # not by tool_executor. Skip it silently if it appears here.
        if name == "ask_user_for_required_parameters":
            continue

        # Inject subject_id if missing and action expects it
        if name != "prepare_to_answer" and "subject_id" not in args:
            args["subject_id"] = subject_id

        # Execute the real tool
        observation = call_tool(name, args, tool_set=tool_set)
        observations[i] = observation

        # Compute dependency list:
        # - prepare_to_answer depends on all prior non-prepare actions
        # - other actions depend on nothing (parallel by default)
        #   unless an argument references a prior observation
        dep_list = []
        if name == "prepare_to_answer":
            dep_list = [j for j in range(i)]
        # Check if any arg values reference prior observation keys
        # (simple heuristic: if action is after another and uses hadm_id from prior result)

        results.append({
            "action": {"name": name, "arguments": args},
            "observation": observation,
            "dependency_list": dep_list,
            "idx": i,
        })

    return results
