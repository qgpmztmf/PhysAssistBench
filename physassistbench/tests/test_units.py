"""
Unit tests for physassistbench — pure/deterministic functions only (no network, no DB).

Covers:
  - eval_runner:    compute_tool_metrics, _api_name/_call_name,
                    _sanitize_tools_for_api, _normalize_patient_tool_args,
                    _compact_fhir_bundle, _compact_observation_str,
                    _filter_tools_for_task, _messages_to_responses_input,
                    _tools_to_responses, _thinking_extra_body
  - eval/metrics:   calc_accuracy, format_metrics_report
  - tools/tool_registry: _normalize_args, set_active_date/get_active_date

Run from the PhysAssistBench repo root:
    uv run python -m pytest physassistbench/tests/test_physassistbench_units.py -v
"""

import sys
import os

# Make sure the PhysAssistBench root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# eval_runner imports (no network; imported functions are pure utilities)
# ─────────────────────────────────────────────────────────────────────────────
from physassistbench.eval_runner import (
    compute_tool_metrics,
    _api_name,
    _call_name,
    _TOOL_API_NAME_MAP,
    _TOOL_CALL_NAME_MAP,
    _sanitize_tools_for_api,
    _normalize_patient_tool_args,
    _compact_fhir_bundle,
    _compact_observation_str,
    _filter_tools_for_task,
    _messages_to_responses_input,
    _tools_to_responses,
    _thinking_extra_body,
    _WRITE_TOOL_NAMES,
)
from physassistbench.eval.metrics import calc_accuracy, format_metrics_report
from physassistbench.tools.tool_registry import (
    _normalize_args,
    set_active_date,
    get_active_date,
)


# ═════════════════════════════════════════════════════════════════════════════
# compute_tool_metrics
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeToolMetrics:
    def _wrap(self, names):
        return [{"action": {"name": n}, "observation": {}} for n in names]

    def test_exact_match_single_tool(self):
        executed = self._wrap(["get_lab_results", "prepare_to_answer"])
        gold     = self._wrap(["get_lab_results", "prepare_to_answer"])
        m = compute_tool_metrics(executed, gold, "Information Lookup")
        assert m["ap_rate"] == 1.0
        assert m["tool_names_correct"] is True
        assert m["tool_coverage_correct"] is True
        assert m["missed_tools"] == []
        assert m["extra_tools"] == []

    def test_missed_tool(self):
        executed = self._wrap(["prepare_to_answer"])
        gold     = self._wrap(["get_lab_results", "prepare_to_answer"])
        m = compute_tool_metrics(executed, gold, "Information Lookup")
        assert m["ap_rate"] == 0.0
        assert m["tool_names_correct"] is False
        assert m["tool_coverage_correct"] is False
        assert "get_lab_results" in m["missed_tools"]

    def test_extra_tool(self):
        executed = self._wrap(["get_lab_results", "get_diagnoses", "prepare_to_answer"])
        gold     = self._wrap(["get_lab_results", "prepare_to_answer"])
        m = compute_tool_metrics(executed, gold, "Information Lookup")
        assert m["ap_rate"] == 1.0          # all gold tools present
        assert m["tool_names_correct"] is False
        assert "get_diagnoses" in m["extra_tools"]

    def test_partial_coverage(self):
        executed = self._wrap(["get_lab_results", "prepare_to_answer"])
        gold     = self._wrap(["get_lab_results", "get_diagnoses", "prepare_to_answer"])
        m = compute_tool_metrics(executed, gold, "Data Gathering")
        assert m["ap_rate"] == pytest.approx(0.5)
        assert m["tool_coverage_correct"] is True
        assert "get_diagnoses" in m["missed_tools"]

    def test_protocol_no_tools_expected(self):
        executed = self._wrap(["prepare_to_answer"])
        gold     = self._wrap(["prepare_to_answer"])
        m = compute_tool_metrics(executed, gold, "Protocol")
        assert m["ap_rate"] == 1.0
        assert m["tool_names_correct"] is True
        assert m["tool_coverage_correct"] is True
        assert m["op_rate"] is None

    def test_protocol_llm_called_extra_tools(self):
        executed = self._wrap(["get_lab_results", "prepare_to_answer"])
        gold     = self._wrap(["prepare_to_answer"])
        m = compute_tool_metrics(executed, gold, "Protocol")
        assert m["ap_rate"] == 0.0
        assert "get_lab_results" in m["extra_tools"]

    def test_op_rate_data_gathering_all_called(self):
        executed = self._wrap(["get_lab_results", "get_diagnoses", "prepare_to_answer"])
        gold     = self._wrap(["get_lab_results", "get_diagnoses", "prepare_to_answer"])
        m = compute_tool_metrics(executed, gold, "Data Gathering")
        assert m["op_rate"] == 1.0

    def test_op_rate_data_gathering_missed_tool(self):
        executed = self._wrap(["get_lab_results", "prepare_to_answer"])
        gold     = self._wrap(["get_lab_results", "get_diagnoses", "prepare_to_answer"])
        m = compute_tool_metrics(executed, gold, "Data Gathering")
        assert m["op_rate"] == 0.0

    def test_empty_gold_and_executed(self):
        m = compute_tool_metrics([], [], "Information Lookup")
        assert m["ap_rate"] == 1.0
        assert m["tool_names_correct"] is True


# ═════════════════════════════════════════════════════════════════════════════
# _api_name / _call_name  (dot → __ conversion for function names)
# ═════════════════════════════════════════════════════════════════════════════

class TestApiNameConversion:
    def setup_method(self):
        # Clear maps before each test to avoid cross-test state
        _TOOL_API_NAME_MAP.clear()
        _TOOL_CALL_NAME_MAP.clear()

    def test_no_dots_unchanged(self):
        assert _api_name("get_lab_results") == "get_lab_results"

    def test_dot_converted_to_double_underscore(self):
        safe = _api_name("patient.get_chief_complaint")
        assert safe == "patient__get_chief_complaint"

    def test_reverse_lookup_registered(self):
        safe = _api_name("patient.get_symptom_history")
        assert _call_name(safe) == "patient.get_symptom_history"

    def test_unknown_api_name_returned_as_is(self):
        assert _call_name("some_unknown_tool") == "some_unknown_tool"


# ═════════════════════════════════════════════════════════════════════════════
# _sanitize_tools_for_api
# ═════════════════════════════════════════════════════════════════════════════

class TestSanitizeToolsForApi:
    def _make_tool(self, name):
        return {"function": {"name": name, "description": "desc",
                             "parameters": {"type": "object", "properties": {}}}}

    def test_plain_name_unchanged(self):
        tools = [self._make_tool("get_lab_results")]
        out = _sanitize_tools_for_api(tools)
        assert out[0]["function"]["name"] == "get_lab_results"

    def test_dot_name_sanitized(self):
        _TOOL_API_NAME_MAP.clear()
        _TOOL_CALL_NAME_MAP.clear()
        tools = [self._make_tool("patient.get_chief_complaint")]
        out = _sanitize_tools_for_api(tools)
        assert out[0]["function"]["name"] == "patient__get_chief_complaint"

    def test_original_tool_not_mutated(self):
        _TOOL_API_NAME_MAP.clear()
        _TOOL_CALL_NAME_MAP.clear()
        orig = self._make_tool("patient.get_pain_assessment")
        out = _sanitize_tools_for_api([orig])
        assert orig["function"]["name"] == "patient.get_pain_assessment"  # deep copy
        assert out[0]["function"]["name"] == "patient__get_pain_assessment"


# ═════════════════════════════════════════════════════════════════════════════
# _normalize_patient_tool_args
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalizePatientToolArgs:
    def test_injects_subject_id_and_session_id(self):
        args = {"query": "chest pain"}
        result = _normalize_patient_tool_args(
            "patient.get_symptom_history", args, subject_id=12345, session_id="sess-abc"
        )
        assert result["subject_id"] == 12345
        assert result["session_id"] == "sess-abc"
        assert result["query"] == "chest pain"

    def test_non_patient_tool_unchanged(self):
        args = {"subject_id": 99, "hadm_id": 1}
        result = _normalize_patient_tool_args("get_lab_results", args, 12345, "sess-abc")
        assert result == {"subject_id": 99, "hadm_id": 1}

    def test_overwrites_existing_ids(self):
        args = {"subject_id": 0, "session_id": "wrong"}
        result = _normalize_patient_tool_args(
            "patient.get_social_history", args, 99, "correct-session"
        )
        assert result["subject_id"] == 99
        assert result["session_id"] == "correct-session"


# ═════════════════════════════════════════════════════════════════════════════
# _compact_fhir_bundle
# ═════════════════════════════════════════════════════════════════════════════

class TestCompactFhirBundle:
    def _obs_entry(self, code, value, unit, date, interp_code=None):
        entry = {
            "resource": {
                "resourceType": "Observation",
                "code": {"text": code},
                "valueQuantity": {"value": value, "unit": unit},
                "effectiveDateTime": date,
            }
        }
        if interp_code:
            entry["resource"]["interpretation"] = [
                {"coding": [{"code": interp_code, "display": interp_code}]}
            ]
        return entry

    def test_empty_bundle_returns_zero_results(self):
        bundle = {"resourceType": "Bundle", "total": 0, "entry": []}
        out = _compact_fhir_bundle(bundle)
        assert "(0 results)" in out

    def test_observation_formatted_correctly(self):
        bundle = {
            "resourceType": "Bundle",
            "total": 1,
            "entry": [self._obs_entry("Hemoglobin", 12.5, "g/dL", "2024-01-15")],
        }
        out = _compact_fhir_bundle(bundle)
        assert "Hemoglobin" in out
        assert "12.5" in out
        assert "g/dL" in out
        assert "2024-01-15" in out

    def test_non_bundle_returns_none(self):
        result = _compact_fhir_bundle({"resourceType": "Patient", "id": "123"})
        assert result is None

    def test_non_dict_returns_truncated_string(self):
        result = _compact_fhir_bundle("not a dict")
        assert isinstance(result, str)

    def test_truncates_beyond_max_entries(self):
        entries = [self._obs_entry(f"Test{i}", i, "units", "2024-01-01") for i in range(20)]
        bundle = {"resourceType": "Bundle", "total": 20, "entry": entries}
        out = _compact_fhir_bundle(bundle, max_entries=5)
        assert "more" in out  # truncation notice

    def test_condition_resource(self):
        bundle = {
            "resourceType": "Bundle", "total": 1,
            "entry": [{
                "resource": {
                    "resourceType": "Condition",
                    "code": {"text": "Sepsis"},
                    "clinicalStatus": {"text": "active"},
                    "recordedDate": "2024-02-01",
                }
            }]
        }
        out = _compact_fhir_bundle(bundle)
        assert "Sepsis" in out
        assert "active" in out


# ═════════════════════════════════════════════════════════════════════════════
# _compact_observation_str
# ═════════════════════════════════════════════════════════════════════════════

class TestCompactObservationStr:
    def test_fhir_bundle_compacted(self):
        bundle = {
            "resourceType": "Bundle", "total": 1,
            "entry": [{
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Creatinine"},
                    "valueQuantity": {"value": 1.2, "unit": "mg/dL"},
                    "effectiveDateTime": "2024-03-10",
                }
            }]
        }
        out = _compact_observation_str(bundle)
        assert "Creatinine" in out
        assert "1.2" in out

    def test_patient_response_extracted(self):
        obs = {"patient_response": "I have chest pain", "other_key": "ignored"}
        out = _compact_observation_str(obs)
        assert out == "I have chest pain"

    def test_created_resource_compact(self):
        obs = {"resourceType": "Flag", "id": "flag-001", "status": "active"}
        out = _compact_observation_str(obs)
        assert "Flag" in out
        assert "flag-001" in out

    def test_plain_dict_json_encoded(self):
        obs = {"error": "patient not found"}
        out = _compact_observation_str(obs)
        assert "patient not found" in out

    def test_non_dict_json_encoded(self):
        out = _compact_observation_str(["a", "b"])
        assert "a" in out


# ═════════════════════════════════════════════════════════════════════════════
# _filter_tools_for_task
# ═════════════════════════════════════════════════════════════════════════════

class TestFilterToolsForTask:
    def _tool(self, name):
        return {"function": {"name": name}}

    def _schema(self):
        return [
            self._tool("get_lab_results"),
            self._tool("get_diagnoses"),
            self._tool("patient.get_chief_complaint"),
            self._tool("patient.get_symptom_history"),
            self._tool("MedicationRequest.create"),
            self._tool("ServiceRequest.create"),
            self._tool("Flag.create"),
            self._tool("prepare_to_answer"),
        ]

    def test_intake_returns_only_patient_tools(self):
        tools = _filter_tools_for_task(self._schema(), "Intake")
        names = [t["function"]["name"] for t in tools]
        assert all(n.startswith("patient.") for n in names)
        assert "get_lab_results" not in names
        assert "MedicationRequest.create" not in names

    def test_protocol_returns_empty(self):
        tools = _filter_tools_for_task(self._schema(), "Protocol")
        assert tools == []

    def test_write_update_returns_only_write_tools(self):
        tools = _filter_tools_for_task(self._schema(), "Write/Update")
        names = set(t["function"]["name"] for t in tools)
        assert names == _WRITE_TOOL_NAMES

    def test_lookup_returns_ehr_read_tools(self):
        tools = _filter_tools_for_task(self._schema(), "Information Lookup")
        names = [t["function"]["name"] for t in tools]
        assert "get_lab_results" in names
        assert "get_diagnoses" in names
        # no patient or write tools
        assert all(not n.startswith("patient.") for n in names)
        assert all(n not in _WRITE_TOOL_NAMES for n in names)


# ═════════════════════════════════════════════════════════════════════════════
# _messages_to_responses_input
# ═════════════════════════════════════════════════════════════════════════════

class TestMessagesToResponsesInput:
    def test_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        out = _messages_to_responses_input(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        assert out[0]["content"] == "Hello"

    def test_tool_result_becomes_function_call_output(self):
        msgs = [{"role": "tool", "tool_call_id": "tc-1", "content": '{"result": 42}'}]
        out = _messages_to_responses_input(msgs)
        assert len(out) == 1
        assert out[0]["type"] == "function_call_output"
        assert out[0]["call_id"] == "tc-1"

    def test_assistant_with_tool_calls(self):
        msgs = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "get_lab_results", "arguments": "{}"}
            }]
        }]
        out = _messages_to_responses_input(msgs)
        fc = [x for x in out if x.get("type") == "function_call"]
        assert len(fc) == 1
        assert fc[0]["name"] == "get_lab_results"
        assert fc[0]["call_id"] == "call-1"

    def test_none_content_excluded(self):
        msgs = [{"role": "user", "content": None}]
        out = _messages_to_responses_input(msgs)
        assert out == []

    def test_system_role_becomes_user_content(self):
        msgs = [{"role": "system", "content": "You are a doctor."}]
        out = _messages_to_responses_input(msgs)
        assert len(out) == 1
        assert out[0]["content"] == "You are a doctor."


# ═════════════════════════════════════════════════════════════════════════════
# _tools_to_responses
# ═════════════════════════════════════════════════════════════════════════════

class TestToolsToResponses:
    def test_none_returns_none(self):
        assert _tools_to_responses(None) is None
        assert _tools_to_responses([]) is None

    def test_converts_function_tool(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_lab_results",
                "description": "Get lab results",
                "parameters": {"type": "object", "properties": {"subject_id": {"type": "integer"}}}
            }
        }]
        out = _tools_to_responses(tools)
        assert len(out) == 1
        assert out[0]["type"] == "function"
        assert out[0]["name"] == "get_lab_results"
        assert "parameters" in out[0]

    def test_preserves_description(self):
        tools = [{"function": {"name": "foo", "description": "bar desc", "parameters": {}}}]
        out = _tools_to_responses(tools)
        assert out[0]["description"] == "bar desc"


# ═════════════════════════════════════════════════════════════════════════════
# _thinking_extra_body
# ═════════════════════════════════════════════════════════════════════════════

class TestThinkingExtraBody:
    def test_deepseek_enabled(self):
        body = _thinking_extra_body("deepseek-v3", True)
        assert body == {"thinking": {"type": "enabled"}}

    def test_deepseek_disabled(self):
        body = _thinking_extra_body("deepseek-v3", False)
        assert body == {"thinking": {"type": "disabled"}}

    def test_qwen_enabled(self):
        body = _thinking_extra_body("qwen3-5-35b", True)
        assert body == {"enable_thinking": True}

    def test_qwen_disabled(self):
        body = _thinking_extra_body("qwen3-5-35b", False)
        assert body == {"enable_thinking": False}

    def test_openai_no_thinking_param(self):
        body = _thinking_extra_body("gpt-5-4-high", True)
        assert body == {}

    def test_doubao_enabled(self):
        body = _thinking_extra_body("doubao-seed-1-8", True)
        assert body == {"thinking": {"type": "enabled"}}

    def test_glm_disabled(self):
        body = _thinking_extra_body("glm-5-plus", False)
        assert body == {"thinking": {"type": "disabled"}}


# ═════════════════════════════════════════════════════════════════════════════
# calc_accuracy
# ═════════════════════════════════════════════════════════════════════════════

class TestCalcAccuracy:
    def _result(self, entry_id, task_idx, task_type, is_correct,
                domain="LabInterp", hallucination=False,
                safety_violation=False, temporal_correct=None,
                total_steps=1, complete_steps=1):
        return {
            "test_entry_id": entry_id,
            "task_idx": task_idx,
            "task_type": task_type,
            "clinical_task_domain": domain,
            "is_correct": is_correct,
            "is_optimal": None,
            "complete_steps": complete_steps,
            "total_steps": total_steps,
            "hallucination": hallucination,
            "safety_violation": safety_violation,
            "temporal_correct": temporal_correct,
        }

    def test_empty_returns_error(self):
        assert calc_accuracy([]) == {"error": "No results"}

    def test_all_correct(self):
        results = [
            self._result("e1", 0, "Information Lookup", True),
            self._result("e1", 1, "Data Gathering", True),
        ]
        m = calc_accuracy(results)
        assert m["total_accuracy"] == 1.0
        assert m["session_accuracy"] == 1.0

    def test_half_correct(self):
        results = [
            self._result("e1", 0, "Information Lookup", True),
            self._result("e1", 1, "Information Lookup", False),
        ]
        m = calc_accuracy(results)
        assert m["total_accuracy"] == pytest.approx(0.5)
        assert m["session_accuracy"] == 0.0  # not all turns correct

    def test_session_accuracy_multi_entry(self):
        results = [
            self._result("e1", 0, "Information Lookup", True),
            self._result("e2", 0, "Information Lookup", False),
        ]
        m = calc_accuracy(results)
        assert m["n_sessions"] == 2
        assert m["session_accuracy"] == pytest.approx(0.5)

    def test_by_task_type(self):
        results = [
            self._result("e1", 0, "Information Lookup", True),
            self._result("e2", 0, "Data Gathering", False),
        ]
        m = calc_accuracy(results)
        assert m["task_type_accuracy"]["Information Lookup"] == 1.0
        assert m["task_type_accuracy"]["Data Gathering"] == 0.0

    def test_hallucination_rate(self):
        results = [
            self._result("e1", 0, "Information Lookup", True, hallucination=True),
            self._result("e2", 0, "Information Lookup", True, hallucination=False),
        ]
        m = calc_accuracy(results)
        assert m["hallucination_rate"] == pytest.approx(0.5)

    def test_temporal_accuracy_none_when_no_temporal_results(self):
        results = [self._result("e1", 0, "Information Lookup", True)]
        m = calc_accuracy(results)
        assert m["temporal_accuracy"] is None

    def test_temporal_accuracy_computed(self):
        results = [
            self._result("e1", 0, "Information Lookup", True, temporal_correct=True),
            self._result("e2", 0, "Information Lookup", True, temporal_correct=False),
        ]
        m = calc_accuracy(results)
        assert m["temporal_accuracy"] == pytest.approx(0.5)

    def test_intake_coverage(self):
        r1 = self._result("e1", 0, "Intake", True)
        r1["tool_coverage_correct"] = True
        r2 = self._result("e2", 0, "Intake", False)
        r2["tool_coverage_correct"] = False
        m = calc_accuracy([r1, r2])
        assert m["intake_coverage"] == pytest.approx(0.5)


# ═════════════════════════════════════════════════════════════════════════════
# format_metrics_report
# ═════════════════════════════════════════════════════════════════════════════

class TestFormatMetricsReport:
    def test_output_contains_accuracy(self):
        metrics = {
            "total_accuracy": 0.75, "n_total": 4, "n_sessions": 2,
            "session_accuracy": 0.5,
            "domain_accuracy": {"LabInterp": 0.8},
            "task_type_accuracy": {"Information Lookup": 0.75},
            "subtype_accuracy": {},
            "layer_accuracy": {"turn_0": 1.0},
            "hallucination_rate": 0.0,
            "safety_violation_rate": 0.0,
            "temporal_accuracy": None,
            "optimality": None,
            "progress_ratio": None,
            "intake_coverage": None,
            "critical_symptom_coverage": None,
        }
        report = format_metrics_report(metrics)
        assert "0.750" in report
        assert "LabInterp" in report
        assert "Information Lookup" in report


# ═════════════════════════════════════════════════════════════════════════════
# tool_registry: _normalize_args
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalizeArgs:
    def test_known_alias_remapped(self):
        args = {"test_name": "Creatinine", "subject_id": 1}
        out = _normalize_args("get_lab_results", args)
        assert "item_name" in out
        assert out["item_name"] == "Creatinine"
        assert "test_name" not in out

    def test_none_alias_dropped(self):
        args = {"limit": 10, "subject_id": 1}
        out = _normalize_args("get_lab_results", args)
        assert "limit" not in out

    def test_valid_arg_kept(self):
        args = {"subject_id": 42, "item_name": "Hemoglobin"}
        out = _normalize_args("get_lab_results", args)
        assert out.get("subject_id") == 42
        assert out.get("item_name") == "Hemoglobin"

    def test_unknown_arg_silently_dropped(self):
        args = {"totally_unknown_kwarg": "value", "subject_id": 1}
        out = _normalize_args("get_lab_results", args)
        assert "totally_unknown_kwarg" not in out

    def test_fhir_observation_alias(self):
        args = {"test_name": "BP", "subject_id": 10}
        out = _normalize_args("Observation.search", args)
        assert "code" in out
        assert out["code"] == "BP"

    def test_fhir_medication_alias(self):
        args = {"drug": "Aspirin", "subject_id": 5}
        out = _normalize_args("MedicationRequest.search", args)
        assert "medication" in out
        assert out["medication"] == "Aspirin"

    def test_lab_trends_n_recent_alias(self):
        args = {"n_results": 5, "item_name": "Sodium", "subject_id": 1}
        out = _normalize_args("get_lab_trends", args)
        assert "n_recent" in out
        assert out["n_recent"] == 5


# ═════════════════════════════════════════════════════════════════════════════
# tool_registry: set_active_date / get_active_date
# ═════════════════════════════════════════════════════════════════════════════

class TestActiveDateGate:
    def teardown_method(self):
        set_active_date(None)  # clean up after each test

    def test_initial_state_is_none(self):
        set_active_date(None)
        assert get_active_date() is None

    def test_set_and_get(self):
        set_active_date("2024-06-15")
        assert get_active_date() == "2024-06-15"

    def test_reset_to_none(self):
        set_active_date("2024-01-01")
        set_active_date(None)
        assert get_active_date() is None

    def test_overwrite(self):
        set_active_date("2023-01-01")
        set_active_date("2024-12-31")
        assert get_active_date() == "2024-12-31"
