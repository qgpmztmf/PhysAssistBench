"""
physassistbench/eval/rule_checker.py — Rule-based is_correct checker for PhysAssistBench.

Mirrors WildToolBench's ToolArgsChecker approach:
  1. Tool name check  : predicted tool set must exactly match gold tool set
  2. Schema check     : predicted args must conform to the tool's JSON schema
                        (required fields, types, enum constraints)
  3. Content check    : predicted arg values must match gold values
                        - short strings (< 10 chars) : edit distance ≥ 0.8
                        - long strings               : ROUGE-L ≥ 0.7
                          (falls back to edit distance if rouge not installed)
                        - numbers / booleans         : exact equality
                        - arrays                     : length + element-wise
                        - objects                    : recursive key+value match

EHR-specific: subject_id, hadm_id, session_id are always skipped in the
content check because they are injected automatically and are never part of
the agent's decision.

Install optional deps for full fidelity:
    pip install rouge jieba
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

# Optional: rouge + jieba (graceful degradation if absent)
try:
    from rouge import Rouge as _Rouge
    _rouge = _Rouge()
    _ROUGE_AVAILABLE = True
except ImportError:
    _rouge = None
    _ROUGE_AVAILABLE = False

try:
    import jieba as _jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _jieba = None
    _JIEBA_AVAILABLE = False

# Arguments that are always injected / always correct — skip in content check
_SKIP_ARGS = {"subject_id", "hadm_id", "session_id"}

# Control-flow tools that carry no real arguments
_CONTROL_TOOLS = {"prepare_to_answer", "ask_user_for_required_parameters"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return "".join(s.split()).lower()


def _edit_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", text))


def _rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1.  Falls back to edit distance if rouge is unavailable."""
    if not _ROUGE_AVAILABLE or not pred.strip() or not ref.strip():
        return _edit_similarity(pred, ref)
    try:
        if _JIEBA_AVAILABLE and (_contains_chinese(pred) or _contains_chinese(ref)):
            pred = " ".join(_jieba.cut(pred))
            ref = " ".join(_jieba.cut(ref))
        scores = _rouge.get_scores(pred, ref)
        return scores[0]["rouge-l"]["f"]
    except Exception:
        return _edit_similarity(pred, ref)


# ── JSON-schema type map ──────────────────────────────────────────────────────

_JSON_TO_PY: dict[str, list] = {
    "string":  [str],
    "integer": [int],
    "float":   [float],
    "number":  [int, float],
    "boolean": [bool],
    "array":   [list],
    "object":  [dict],
    "null":    [type(None)],
}


# ── RuleBasedChecker ─────────────────────────────────────────────────────────

class RuleBasedChecker:
    """
    Deterministic correctness checker.

    Usage::

        checker = RuleBasedChecker()
        ok, reason = checker.is_correct(gold_turn, pred_turn, tools)

    gold_turn / pred_turn have the shape produced by _extract_gold_turn_info /
    the pred_info dict built in compute_iirs_for_entries::

        {
            "tool_calls": list[str],   # tool names (excl. control tools)
            "tool_args":  dict,        # args of the *first* real tool call
            "answer":     str,         # final answer text (unused here)
        }

    tools: the OpenAI-format tool schema list from the benchmark entry.
    """

    # ── result tokens ─────────────────────────────────────────────────────────
    CORRECT = "correct"
    ERR_JSON        = "error: args invalid json format"
    ERR_MISSING     = "error_schema: required args missing"
    ERR_UNDEFINED   = "error_schema: args not defined"
    ERR_TYPE        = "error_schema: args type inconsistent"
    ERR_ENUM        = "error_schema: args value not in enum"
    ERR_KEYS        = "error_match: args keys mismatch"
    ERR_ARRAY_LEN   = "error_match: array length mismatch"
    ERR_TYPE_MM     = "error_match: value type inconsistent"
    ERR_VALUE       = "error_match: value mismatch"
    ERR_SIMILARITY  = "error_match: string similarity too low"
    ERR_TOOLS       = "error: tool name mismatch"

    # ── public entry point ────────────────────────────────────────────────────

    def is_correct(
        self,
        gold_turn: dict[str, Any],
        pred_turn: dict[str, Any],
        tools: list[dict],
    ) -> tuple[bool, str]:
        """
        Returns (True, "correct") or (False, <reason>).

        Checks in order:
          1. Tool names match
          2. Schema validity of predicted args
          3. Content match of predicted args vs gold args
        """
        gold_tools = [t for t in gold_turn.get("tool_calls", []) if t not in _CONTROL_TOOLS]
        pred_tools = [t for t in pred_turn.get("tool_calls", []) if t not in _CONTROL_TOOLS]

        # 1. Tool name set must match exactly
        if set(gold_tools) != set(pred_tools):
            return False, (
                f"{self.ERR_TOOLS}: gold={sorted(gold_tools)} pred={sorted(pred_tools)}"
            )

        # No real tools to check further (e.g. Clinical Reasoning / Protocol)
        if not gold_tools:
            return True, self.CORRECT

        # Build schema lookup
        schema_map = self._parse_tools(tools)

        # Check each tool (matched by name — order-independent)
        gold_args = gold_turn.get("tool_args", {})
        pred_args = pred_turn.get("tool_args", {})

        for tool_name in gold_tools:
            if tool_name not in schema_map:
                # Tool not in schema — skip deep checks
                continue

            tool_schema = schema_map[tool_name]

            # 2. Schema check
            schema_err = self._schema_check(pred_args, tool_schema)
            if schema_err:
                return False, schema_err

            # 3. Content check
            content_err = self._content_check(pred_args, gold_args, tool_schema)
            if content_err:
                return False, content_err

        return True, self.CORRECT

    # ── schema parsing ────────────────────────────────────────────────────────

    def _parse_tools(self, tools: list[dict]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name")
            if not name:
                continue
            params = fn.get("parameters", {})
            result[name] = {
                "required": set(params.get("required", [])),
                "properties": params.get("properties", {}),
            }
        return result

    # ── schema check ─────────────────────────────────────────────────────────

    def _schema_check(self, args: dict, tool_schema: dict) -> str | None:
        """Return error string or None if valid."""
        required = tool_schema["required"]
        properties = tool_schema["properties"]

        missing = required - set(args.keys()) - _SKIP_ARGS
        if missing:
            return f"{self.ERR_MISSING}: {sorted(missing)}"

        for key, val in args.items():
            if key in _SKIP_ARGS:
                continue
            if key not in properties:
                return f"{self.ERR_UNDEFINED}: '{key}'"
            err = self._check_value_schema(val, properties[key])
            if err:
                return err
        return None

    def _check_value_schema(self, val: Any, schema: dict) -> str | None:
        if "anyOf" in schema:
            for sub in schema["anyOf"]:
                if self._check_value_schema(val, sub) is None:
                    return None
            return self.ERR_TYPE

        allowed = []
        for tname in ([schema["type"]] if "type" in schema else []):
            allowed.extend(_JSON_TO_PY.get(tname, []))
        if allowed and not isinstance(val, tuple(allowed)):
            return self.ERR_TYPE

        if "enum" in schema and val not in schema["enum"]:
            return self.ERR_ENUM

        if isinstance(val, dict) and "properties" in schema:
            req = set(schema.get("required", []))
            missing = req - set(val.keys())
            if missing:
                return self.ERR_MISSING
            for k, v in val.items():
                if k not in schema["properties"]:
                    return self.ERR_UNDEFINED
                err = self._check_value_schema(v, schema["properties"][k])
                if err:
                    return err

        if isinstance(val, list) and "items" in schema:
            for item in val:
                err = self._check_value_schema(item, schema["items"])
                if err:
                    return err

        return None

    # ── content check ─────────────────────────────────────────────────────────

    def _content_check(
        self,
        pred: dict,
        gold: dict,
        tool_schema: dict,
        path: str = "",
    ) -> str | None:
        """Return error string or None if values match."""
        properties = tool_schema.get("properties", {})

        # Only compare keys present in gold (ignore EHR-injected keys)
        for key, gold_val in gold.items():
            if key in _SKIP_ARGS:
                continue
            if key not in pred:
                # Missing key — already caught by schema check; skip here
                continue
            pred_val = pred[key]
            sub_path = f"{path}.{key}" if path else key
            sub_schema = properties.get(key, {})
            err = self._compare_values(pred_val, gold_val, sub_schema, sub_path)
            if err:
                return err
        return None

    def _compare_values(
        self,
        pred: Any,
        gold: Any,
        schema: dict,
        path: str,
    ) -> str | None:
        path_tag = f" at '{path}'" if path else ""

        if type(pred) != type(gold):
            return f"{self.ERR_TYPE_MM}{path_tag}"

        if isinstance(gold, dict):
            if set(pred.keys()) != set(gold.keys()):
                return f"{self.ERR_KEYS}{path_tag}"
            props = schema.get("properties", {})
            for k in gold:
                err = self._compare_values(pred[k], gold[k], props.get(k, {}), f"{path}.{k}")
                if err:
                    return err
            return None

        if isinstance(gold, list):
            if len(pred) != len(gold):
                return f"{self.ERR_ARRAY_LEN}{path_tag}"
            item_schema = schema.get("items", {})
            for i, (p, g) in enumerate(zip(pred, gold)):
                err = self._compare_values(p, g, item_schema, f"{path}[{i}]")
                if err:
                    return err
            return None

        if isinstance(gold, str):
            if pred == gold:
                return None
            if _normalize(pred) == _normalize(gold):
                return None
            if len(gold) < 10:
                sim = _edit_similarity(pred, gold)
                if sim < 0.8:
                    return f"{self.ERR_SIMILARITY}{path_tag} (edit={sim:.2f})"
                return None
            sim = _rouge_l(pred, gold)
            if sim < 0.7:
                return f"{self.ERR_SIMILARITY}{path_tag} (rouge-l={sim:.2f})"
            return None

        # numbers, booleans, None
        if pred != gold:
            return f"{self.ERR_VALUE}{path_tag} (pred={pred!r} gold={gold!r})"
        return None


# ── module-level convenience ──────────────────────────────────────────────────

_checker = RuleBasedChecker()


def rule_based_is_correct(
    gold_turn: dict[str, Any],
    pred_turn: dict[str, Any],
    tools: list[dict],
) -> tuple[bool, str]:
    """
    Module-level convenience wrapper around RuleBasedChecker.is_correct().

    Returns (is_correct: bool, reason: str).
    """
    return _checker.is_correct(gold_turn, pred_turn, tools)
