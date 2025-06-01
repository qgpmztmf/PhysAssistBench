"""
Checker Tool — validates that tool execution results are non-empty and meaningful.

Returns (is_valid: bool, reason: str)
"""


def validate_observations(executed_actions: list) -> tuple[bool, str]:
    """
    Check that tool observations are non-empty and don't all contain errors.
    Also checks that prepare_to_answer fired last.

    Returns (is_valid, reason).
    """
    if not executed_actions:
        return False, "No actions were executed"

    # Check last action is prepare_to_answer
    last = executed_actions[-1]
    if last["action"]["name"] != "prepare_to_answer":
        return False, "Action list must end with prepare_to_answer"

    # Check that at least one non-prepare action returned real data
    real_actions = [a for a in executed_actions if a["action"]["name"] != "prepare_to_answer"]

    if not real_actions:
        # Chat task — OK
        return True, "Chat task (no tool calls)"

    def _is_error_obs(obs) -> bool:
        """Return True if the observation represents a tool error / no-data response."""
        if not isinstance(obs, dict):
            return True
        # Legacy EHR tools use top-level "error" key
        if "error" in obs:
            return True
        # FHIR OperationOutcome: {"resourceType": "OperationOutcome", "issue": [...]}
        if obs.get("resourceType") == "OperationOutcome":
            return True
        return False

    all_errors = all(_is_error_obs(a["observation"]) for a in real_actions)
    if all_errors:
        return False, "All tool calls returned errors"

    # Check that at least one action returned non-trivial data
    for a in real_actions:
        obs = a["observation"]
        if _is_error_obs(obs):
            continue
        # Patient interview tools: patient_response key with str content
        if "patient_response" in obs and isinstance(obs["patient_response"], str) and len(obs["patient_response"]) > 0:
            return True, "OK (patient response)"
        # EHR tools: any non-empty list, long string, or non-empty dict
        for v in obs.values():
            if isinstance(v, list) and len(v) > 0:
                return True, "OK"
            if isinstance(v, str) and len(v) > 50:
                return True, "OK (text content)"
            if isinstance(v, dict) and v:
                return True, "OK (dict content)"

    return False, "Tool observations appear empty or trivial"
