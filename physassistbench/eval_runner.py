"""
eval_runner.py — Run DeepSeek LLM as a Doctor Agent against mvp_test.jsonl.

For each benchmark entry, the agent receives:
  - env_info (patient context, session_id)
  - tools (EHR tools + patient interview tools)
  - tasks (user questions, one per turn)

The agent must call tools, receive observations, then produce a clinical answer.
Results are compared against gold answer_list for metric computation.

Usage:
    cd /path/to/PhysAssistBench
    uv run python physassistbench/run_eval.py
"""

import json
import logging
import math
import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional

import httpx
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from physassistbench.tools.tool_registry import call_tool, set_active_date
from physassistbench.phm.patient_agent_runtime import register_session, reset_all_sessions, get_session

# Load credentials from the repo-root .env
from physassistbench.paths import ENV_PATH as _ENV_PATH
load_dotenv(dotenv_path=_ENV_PATH, override=False)

logger = logging.getLogger(__name__)

# ── Model config registry ──────────────────────────────────────────────────────

_MODEL_CONFIGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs.yaml")


def load_model_config(model_name: str) -> dict:
    """Load API config for a named model from physassistbench/model_configs.yaml."""
    with open(_MODEL_CONFIGS_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for m in cfg.get("models", []):
        if m["name"] == model_name:
            return m
    available = [m["name"] for m in cfg.get("models", [])]
    raise ValueError(f"Unknown model '{model_name}'. Available: {available}")


# ── Responses API adapter (GPT-5 series via gateway) ────────────────────
# The Responses API uses a different request/response shape than chat
# completions. This adapter exposes a `.chat.completions.create()` interface so
# the existing agent loop works unchanged.

class _RFunc:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class _RToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = _RFunc(name, arguments)

class _RMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls or None

class _RChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"

class _RUsage:
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion

class _RCompletion:
    def __init__(self, choices, usage):
        self.choices = choices
        self.usage = usage


def _messages_to_responses_input(messages: list) -> list:
    """Convert chat-completions messages → Responses API input items."""
    items = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id", ""),
                "output": str(m.get("content", "")),
            })
        elif role == "assistant" and m.get("tool_calls"):
            if m.get("content"):
                items.append({"role": "assistant",
                              "content": str(m["content"])})
            for tc in m["tool_calls"]:
                fn = tc["function"] if isinstance(tc, dict) else tc.function
                cid = tc["id"] if isinstance(tc, dict) else tc.id
                name = fn["name"] if isinstance(fn, dict) else fn.name
                args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                items.append({
                    "type": "function_call",
                    "call_id": cid,
                    "name": name,
                    "arguments": args,
                })
        else:
            content = m.get("content")
            if content is not None:
                items.append({"role": role, "content": str(content)})
    return items


def _tools_to_responses(tools: Optional[list]) -> Optional[list]:
    """Flatten chat-completions tool schema → Responses API tool schema."""
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return out


class _ResponsesCompletions:
    def __init__(self, parent: "ResponsesAPIClient"):
        self._p = parent

    def create(self, model=None, messages=None, tools=None,
               tool_choice=None, temperature=None, max_tokens=None,
               extra_body=None, **kwargs):
        body = {
            "model": self._p.model_id,
            "input": _messages_to_responses_input(messages or []),
        }
        rtools = _tools_to_responses(tools)
        if rtools:
            body["tools"] = rtools
            if tool_choice:
                body["tool_choice"] = tool_choice
        if max_tokens:
            body["max_output_tokens"] = max_tokens
        # GPT-5 series cannot fully disable reasoning, but we minimise it to stay
        # consistent with the "thinking off" eval policy (faster, cheaper, less
        # internal analysis). Configurable via ResponsesAPIClient.reasoning_effort.
        if self._p.reasoning_effort:
            body["reasoning"] = {"effort": self._p.reasoning_effort}

        # Proactive RPM throttle + reactive retry on rate-limit / transient 5xx.
        # Azure returns 429 when over the per-minute quota; honor Retry-After,
        # else exponential backoff. Without this, raise_for_status() throws and
        # the agentic loop aborts the turn.
        max_retries = 6
        for attempt in range(max_retries + 1):
            self._p._throttle()
            r = self._p._http.post(
                self._p.url,
                headers={"Ocp-Apim-Subscription-Key": self._p.api_key,
                         "Content-Type": "application/json"},
                json=body,
                timeout=120.0,
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                ra = r.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else min(2 ** attempt, 60)
                except ValueError:
                    wait = min(2 ** attempt, 60)
                logger.warning(f"[rpm] {r.status_code} from gateway; retry "
                               f"{attempt + 1}/{max_retries} in {wait:.1f}s")
                time.sleep(wait)
                continue
            break
        r.raise_for_status()
        j = r.json()

        text_parts, tool_calls = [], []
        for item in j.get("output", []):
            itype = item.get("type")
            if itype == "function_call":
                tool_calls.append(_RToolCall(
                    item.get("call_id", item.get("id", "")),
                    item.get("name", ""),
                    item.get("arguments", "{}"),
                ))
            elif itype == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        text_parts.append(c.get("text", ""))

        msg = _RMessage("".join(text_parts) or None, tool_calls)
        u = j.get("usage", {}) or {}
        usage = _RUsage(u.get("input_tokens", 0), u.get("output_tokens", 0))
        return _RCompletion([_RChoice(msg)], usage)


class _ResponsesChat:
    def __init__(self, parent):
        self.completions = _ResponsesCompletions(parent)


class ResponsesAPIClient:
    """Duck-typed OpenAI client backed by the Gateway Responses API."""
    def __init__(self, api_base: str, model_id: str, api_key: str,
                 reasoning_effort: str = "minimal", rpm: int = 0):
        self.url = api_base.rstrip("/") + "/v1/openai/responses"
        self.model_id = model_id
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort   # minimal|low|medium|high
        self._http = httpx.Client(timeout=120.0)
        self.chat = _ResponsesChat(self)
        # Client-side request-per-minute throttle (0 = disabled). Sliding 60s
        # window: block before a request would exceed `rpm` in the last minute.
        self.rpm = int(rpm or 0)
        self._req_times: deque = deque()
        self._rpm_lock = threading.Lock()

    def _throttle(self) -> None:
        """Block until issuing one more request stays within `rpm` per 60s."""
        if self.rpm <= 0:
            return
        while True:
            with self._rpm_lock:
                now = time.monotonic()
                while self._req_times and now - self._req_times[0] >= 60.0:
                    self._req_times.popleft()
                if len(self._req_times) < self.rpm:
                    self._req_times.append(now)
                    return
                sleep_for = 60.0 - (now - self._req_times[0]) + 0.05
            logger.info(f"[rpm] throttling {sleep_for:.1f}s "
                        f"({len(self._req_times)}/{self.rpm} in last 60s)")
            time.sleep(max(sleep_for, 0.05))


class AnthropicChatAdapter:
    """Wraps anthropic.Anthropic to expose an OpenAI-compatible chat.completions.create() interface."""

    def __init__(self, api_key: str, base_url: str, max_retries: int = 2):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key, base_url=base_url, max_retries=max_retries)
        self.chat = _AnthropicCompletionsProxy(self)

    def _convert_messages(self, messages: list) -> tuple[str | None, list]:
        """Convert OpenAI message list → (system_str, anthropic_messages)."""
        system = None
        out: list = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system = msg.get("content", "") or ""
                continue
            if role == "tool":
                # tool result → merge into preceding user message as tool_result block
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": str(msg.get("content", "")),
                }
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue
            if role == "assistant":
                blocks: list = []
                text = msg.get("content") or ""
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in (msg.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args_raw = fn.get("arguments", "{}")
                    else:
                        fn = tc.function
                        name = fn.name
                        args_raw = fn.arguments
                    try:
                        inp = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        inp = {}
                    tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                    blocks.append({"type": "tool_use", "id": tc_id, "name": name, "input": inp})
                if not blocks:
                    blocks = [{"type": "text", "text": ""}]
                out.append({"role": "assistant", "content": blocks})
                continue
            # user message
            content = msg.get("content", "")
            if isinstance(content, list):
                out.append({"role": "user", "content": content})
            else:
                out.append({"role": "user", "content": content})
        return system, out

    @staticmethod
    def _convert_tools(tools) -> list | None:
        if not tools:
            return None
        result = []
        for t in tools:
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    @staticmethod
    def _wrap_response(resp):
        """Wrap an anthropic.types.Message into a fake OpenAI response object."""
        from types import SimpleNamespace
        text_content = ""
        tool_calls = []
        for block in (resp.content or []):
            if getattr(block, "type", None) == "text":
                text_content = block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(SimpleNamespace(
                    id=block.id,
                    type="function",
                    function=SimpleNamespace(
                        name=block.name,
                        arguments=json.dumps(block.input, ensure_ascii=False),
                    ),
                    model_extra={},
                ))
        usage = SimpleNamespace(
            prompt_tokens=getattr(resp.usage, "input_tokens", 0),
            completion_tokens=getattr(resp.usage, "output_tokens", 0),
            total_tokens=getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0),
        )
        message = SimpleNamespace(
            role="assistant",
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
            model_extra={},
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
            usage=usage,
            model=getattr(resp, "model", ""),
        )

    def _create(self, model, messages, tools=None, tool_choice=None,
                temperature=0.2, max_tokens=1024, extra_body=None, **kwargs):
        system, anth_messages = self._convert_messages(messages)
        anth_tools = self._convert_tools(tools)
        params: dict = dict(
            model=model,
            messages=anth_messages,
            max_tokens=max_tokens,
        )
        if system:
            params["system"] = system
        if anth_tools:
            params["tools"] = anth_tools
            params["tool_choice"] = {"type": "auto"}
        if temperature is not None:
            params["temperature"] = temperature
        resp = self._client.messages.create(**params)
        return self._wrap_response(resp)


class _AnthropicCompletionsProxy:
    def __init__(self, adapter: "AnthropicChatAdapter"):
        self._adapter = adapter
        self.completions = self

    def create(self, **kwargs):
        return self._adapter._create(**kwargs)


def _build_client(model_cfg: dict, api_key: str):
    """Build an OpenAI-compatible client, handling the Azure gateway."""
    # max_retries: OpenAI SDK retries 429/5xx with exponential backoff and honors
    # Retry-After. Bump it (via config `max_retries`) for providers with tight
    # rate limits (e.g. Gemini compat endpoint under N-way parallelism) so a
    # rate-limit burst doesn't fall through to the "[Error generating final
    # answer]" fallback. Default 2 = OpenAI SDK default (unchanged for others).
    max_retries = int(model_cfg.get("max_retries", 2))

    # Doctor-agent endpoint override for local serving (e.g. a vLLM server).
    # Uses a DEDICATED env var, NOT a provider var like DEEPSEEK_BASE_URL —
    # llm_client.load_dotenv(override=True) would overwrite the provider vars
    # from the WildToolBench .env, silently sending traffic to the cloud.
    # When set, talk plain OpenAI to that URL regardless of the model's api_type
    # (a local vLLM is OpenAI-compatible).
    override_url = os.environ.get("EVAL_DOCTOR_BASE_URL")
    if override_url:
        return OpenAI(api_key=api_key, base_url=override_url, max_retries=max_retries)

    api_type = model_cfg.get("api_type")

    if api_type == "anthropic":
        return AnthropicChatAdapter(
            api_key=api_key,
            base_url=model_cfg["api_base"],
            max_retries=max_retries,
        )

    if api_type == "gateway_responses":
        # RPM cap: env EVAL_RPM overrides the model_config `rpm` (0 = no limit).
        rpm = int(os.environ.get("EVAL_RPM") or model_cfg.get("rpm", 0) or 0)
        return ResponsesAPIClient(
            api_base=model_cfg["api_base"],
            model_id=model_cfg["model_id"],
            api_key=api_key,
            # default to minimal reasoning; override per-model via config
            reasoning_effort=model_cfg.get("reasoning_effort", "minimal"),
            rpm=rpm,
        )

    if api_type == "gateway":
        endpoint_path = model_cfg["endpoint_path"]

        def _rewrite_url(request: httpx.Request) -> None:
            if request.url.path == "/chat/completions":
                request.url = request.url.copy_with(
                    path=f"{endpoint_path}/chat/completions"
                )

        return OpenAI(
            base_url=model_cfg["api_base"],
            api_key="unused",
            default_headers={"Ocp-Apim-Subscription-Key": api_key},
            http_client=httpx.Client(event_hooks={"request": [_rewrite_url]}),
        )

    return OpenAI(api_key=api_key, base_url=model_cfg["api_base"], max_retries=max_retries)


# Agentic loop limits
MAX_TOOL_STEPS = 8
MAX_TOKENS_ANSWER = 1024              # default output budget (non-thinking models)
DOCTOR_AGENT_MODEL = "deepseek-v3"   # default eval model (non-reasoning, no thinking issue)
# Thinking / output-budget state for the model-under-test. Set from model_config
# in run_evaluation(); DO NOT guess from the model name (see _thinking_extra_body).
DOCTOR_THINKING = False              # whether the tested model runs with thinking ON
DOCTOR_MAX_TOKENS = MAX_TOKENS_ANSWER  # effective output budget (config-overridable)


def _thinking_extra_body(model_name: str, enabled: bool) -> dict:
    """Provider-specific extra_body to toggle 'thinking' mode ON/OFF.

    Decided by the model config's `thinking` flag, not by guessing from the
    model name. Param name differs per provider:
      DeepSeek      : {"thinking": {"type": "enabled"|"disabled"}}
      Qwen/DashScope: {"enable_thinking": True|False}
      OpenAI/Azure  : no chat-completions param (reasoning_effort handled in the
                      Responses adapter) → send nothing to avoid a 400.
    """
    m = str(model_name).lower()
    if "deepseek" in m:
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}
    if "glm" in m:
        # Zhipu GLM (verified on vectorengine proxy): {"thinking": {"type": "disabled"}}
        # collapses reasoning to ~0 tokens; the enabled form keeps it on.
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}
    if "doubao" in m:
        # ByteDance Doubao-Seed: same DeepSeek-style thinking toggle.
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}
    if "qwen" in m:
        return {"enable_thinking": bool(enabled)}
    return {}

_SYSTEM_PROMPT = """You are a clinical decision support agent assisting a physician in an ICU/hospital setting.
You have access to a set of EHR tools to query real patient data, and patient interview tools to conduct patient intake.

Guidelines:
1. Carefully read the user's clinical question and determine which tools to call.
2. For Lookup tasks: call exactly 1 EHR tool, then prepare_to_answer.
3. For Data Gathering tasks: call 2-4 EHR tools (use parallel calls when independent), then prepare_to_answer.
4. For Protocol tasks: no tool calls needed — answer from clinical knowledge directly.
5. For Verify tasks: if the question lacks required parameters that cannot be inferred from
   context, call ask_user_for_required_parameters with the tool_name and missing parameter
   names. Do NOT call any EHR tools before receiving the user's clarification.
6. For Intake tasks: call 1-4 patient.xxx tools to conduct intake interview, then prepare_to_answer.
7. Always call prepare_to_answer as your last action before giving a final answer.
8. For patient interview tools (patient.get_xxx), always pass the session_id from env_info.

Respond concisely and clinically. Your final answer should include:
- Key findings from tool observations
- Clinical interpretation
- Recommended next steps (if applicable)"""

_SYSTEM_PROMPT_ZH = """你是一位临床决策支持助手，协助医生在ICU/医院环境中工作。
你可以使用一套EHR工具查询真实患者数据，以及患者访谈工具进行患者入院评估。

操作指南：
1. 仔细阅读用户的临床问题，判断需要调用哪些工具。
2. Lookup类任务：调用恰好1个EHR工具，然后调用prepare_to_answer。
3. Data Gathering类任务：调用2-4个EHR工具（独立的工具可并行调用），然后调用prepare_to_answer。
4. Protocol类任务：无需调用工具，直接依据临床知识回答。
5. Verify类任务：若问题缺少无法从上下文推断的必要参数，调用ask_user_for_required_parameters，
   传入tool_name和缺失的参数名称。在收到用户澄清回复之前，不得调用任何EHR工具。
6. Intake类任务：调用1-4个patient.xxx工具进行患者入院访谈，然后调用prepare_to_answer。
7. 在给出最终答案前，务必将prepare_to_answer作为最后一个操作调用。
8. 对于患者访谈工具（patient.get_xxx），请始终传入env_info中的session_id。

请简洁专业地回答，最终答案应包含：
- 工具观察结果中的关键发现
- 临床解读
- 建议的后续步骤（如适用）"""


def _build_prepare_to_answer_schema():
    return {
        "type": "function",
        "function": {
            "name": "prepare_to_answer",
            "description": "Signal that you have gathered all needed information and are ready to answer.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }


# DeepSeek/OpenAI API does not allow dots in function names (must match ^[a-zA-Z0-9_-]+$).
# Patient tools use dot notation (patient.get_xxx). We normalize dots→double-underscores
# for the API and reverse when executing via call_tool.
_TOOL_API_NAME_MAP: dict[str, str] = {}  # api_name → call_tool name
_TOOL_CALL_NAME_MAP: dict[str, str] = {}  # call_tool name → api_name


def _api_name(tool_name: str) -> str:
    """Convert call_tool name to API-safe name (dots → __)."""
    if "." not in tool_name:
        return tool_name
    safe = tool_name.replace(".", "__")
    _TOOL_API_NAME_MAP[safe] = tool_name
    _TOOL_CALL_NAME_MAP[tool_name] = safe
    return safe


def _call_name(api_name: str) -> str:
    """Reverse: API-safe name → call_tool name."""
    return _TOOL_API_NAME_MAP.get(api_name, api_name)


def _sanitize_tools_for_api(tools: list) -> list:
    """Return a copy of tools with API-safe function names."""
    result = []
    for t in tools:
        orig_name = t["function"]["name"]
        safe_name = _api_name(orig_name)
        if safe_name == orig_name:
            result.append(t)
        else:
            import copy
            t2 = copy.deepcopy(t)
            t2["function"]["name"] = safe_name
            result.append(t2)
    return result


def _normalize_patient_tool_args(tool_name: str, args: dict,
                                  subject_id: int, session_id: str) -> dict:
    """Always inject correct subject_id and session_id for patient tools."""
    if tool_name.startswith("patient."):
        args = dict(args)
        args["subject_id"] = subject_id
        args["session_id"] = session_id
    return args


def _accum_usage(resp, acc: dict) -> None:
    """Add a response's token usage into the accumulator dict (in/out)."""
    try:
        u = getattr(resp, "usage", None)
        if u is not None:
            acc["in"] += int(getattr(u, "prompt_tokens", 0) or 0)
            acc["out"] += int(getattr(u, "completion_tokens", 0) or 0)
            # Track cache hit tokens if API reports them
            cache_hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
            if cache_hit:
                acc["cache_hit"] = acc.get("cache_hit", 0) + int(cache_hit)
    except Exception:
        pass


# ── FHIR observation compaction (Optimization 3) ────────────────────────────
# Compact full FHIR Bundles → "Code: value unit (interp) @ date" lines.
# Reduces obs token cost by ~80% without losing clinical content.

def _compact_fhir_bundle(bundle: dict, max_entries: int = 15) -> str:
    """Convert FHIR search Bundle to a compact human-readable summary."""
    if not isinstance(bundle, dict):
        return str(bundle)[:500]
    if bundle.get("resourceType") != "Bundle":
        return None  # not a bundle — fall back to raw

    entries = bundle.get("entry", []) or []
    total = bundle.get("total", len(entries))
    if not entries:
        return f"(0 results)"

    lines = [f"({total} results)"]
    for en in entries[:max_entries]:
        r = en.get("resource", {}) or {}
        rt = r.get("resourceType", "")

        if rt == "Observation":
            code = (r.get("code") or {}).get("text") or ""
            vq = r.get("valueQuantity") or {}
            val = vq.get("value", "")
            unit = vq.get("unit", "")
            date = (r.get("effectiveDateTime") or r.get("issued") or "")[:10]
            interp = ""
            ilist = r.get("interpretation") or []
            if ilist:
                ic = (ilist[0].get("coding") or [{}])[0]
                interp = f" ({ic.get('display') or ic.get('code') or ''})"
            value_str = r.get("valueString") or r.get("valueCodeableConcept", {}).get("text", "")
            if val != "":
                lines.append(f"  - {code}: {val} {unit}{interp} @ {date}")
            elif value_str:
                lines.append(f"  - {code}: {value_str}{interp} @ {date}")
            else:
                lines.append(f"  - {code} @ {date}")

        elif rt == "MedicationRequest":
            med = (r.get("medicationCodeableConcept") or {}).get("text", "")
            dose = (r.get("dosageInstruction") or [{}])[0].get("text", "")
            status = r.get("status", "")
            auth = (r.get("authoredOn") or "")[:10]
            lines.append(f"  - {med} [{status}]: {dose} @ {auth}")

        elif rt == "MedicationAdministration":
            med = (r.get("medicationCodeableConcept") or {}).get("text", "")
            dose_obj = (r.get("dosage") or {})
            dose = dose_obj.get("text") or (
                f"{(dose_obj.get('dose') or {}).get('value','')} {(dose_obj.get('dose') or {}).get('unit','')}"
            )
            date = (r.get("effectiveDateTime") or "")[:16]
            lines.append(f"  - {med}: {dose} @ {date}")

        elif rt == "Condition":
            code = (r.get("code") or {}).get("text", "")
            clinical = (r.get("clinicalStatus") or {}).get("text") or \
                       ((r.get("clinicalStatus") or {}).get("coding", [{}]) or [{}])[0].get("code", "")
            date = (r.get("recordedDate") or "")[:10]
            lines.append(f"  - {code} [{clinical}] @ {date}")

        elif rt == "DiagnosticReport":
            code = (r.get("code") or {}).get("text", "")
            date = (r.get("effectiveDateTime") or "")[:10]
            # presentedForm contains the full report text
            pf = r.get("presentedForm") or []
            report_text = ""
            for p in pf:
                if isinstance(p.get("data"), str):
                    report_text = p["data"]
                    break
            concl = r.get("conclusion") or report_text
            if concl:
                lines.append(f"  - {code} @ {date}: {concl[:600]}")
            else:
                lines.append(f"  - {code} @ {date}")

        elif rt == "Encounter":
            ec = (r.get("class") or {})
            etype = (r.get("type") or [{}])[0].get("text", "") if r.get("type") else ""
            period = r.get("period") or {}
            start = (period.get("start") or "")[:10]
            end = (period.get("end") or "")[:10]
            disp = ""
            for h in r.get("hospitalization", {}).get("dischargeDisposition", {}).get("coding", []):
                disp = h.get("display", "") or h.get("code", "")
                break
            lines.append(f"  - {ec.get('code','')} {etype} {start}–{end} {disp}")

        elif rt == "AllergyIntolerance":
            substance = (r.get("code") or {}).get("text", "")
            crit = r.get("criticality", "")
            reactions = []
            for rx in r.get("reaction") or []:
                for m in rx.get("manifestation") or []:
                    reactions.append(m.get("text", ""))
            lines.append(f"  - {substance} [{crit}]: {', '.join(reactions)}")

        elif rt == "Procedure":
            code = (r.get("code") or {}).get("text", "")
            status = r.get("status", "")
            date = (r.get("performedDateTime") or "")[:10]
            lines.append(f"  - {code} [{status}] @ {date}")

        elif rt == "CarePlan":
            title = r.get("title") or ""
            status = r.get("status", "")
            cat = (r.get("category") or [{}])[0].get("text", "") if r.get("category") else ""
            lines.append(f"  - {title or cat} [{status}]")

        elif rt == "DocumentReference":
            doc_type = (r.get("type") or {}).get("text", "")
            date = (r.get("date") or "")[:10]
            content = r.get("content") or []
            text = ""
            for c in content:
                att = c.get("attachment") or {}
                if att.get("data"):
                    text = att["data"]
                    break
            if text:
                lines.append(f"  - {doc_type} @ {date}: {text[:800]}")
            else:
                lines.append(f"  - {doc_type} @ {date}")

        else:
            # Unknown type — keep minimal info
            lines.append(f"  - {rt}: {json.dumps(r, default=str)[:200]}")

    if total > max_entries:
        lines.append(f"  ... ({total - max_entries} more)")
    return "\n".join(lines)


def _compact_observation_str(observation) -> str:
    """Return a compact string representation of a tool observation.
    Falls back to JSON for non-Bundle observations."""
    if isinstance(observation, dict):
        if observation.get("resourceType") == "Bundle":
            compact = _compact_fhir_bundle(observation)
            if compact is not None:
                return compact
        # Patient interview responses: keep patient_response only, drop the literacy variants
        if "patient_response" in observation:
            return str(observation.get("patient_response", ""))[:1500]
        # Created resources (write tools): keep id + key fields
        if observation.get("resourceType") in ("MedicationRequest", "ServiceRequest", "Flag"):
            return (f'{observation["resourceType"]} created: id={observation.get("id","")} '
                    f'status={observation.get("status","")}')
    return json.dumps(observation, ensure_ascii=False, default=str)


# ── Per-task-type tool filtering (Optimization 2) ───────────────────────────
# Only pass tools relevant to the task type to reduce schema overhead in every call.
# Write tools are completely hidden from non-Write/Update turns and vice versa.

_WRITE_TOOL_NAMES = {
    "MedicationRequest.create", "ServiceRequest.create", "Flag.create",
}


def _filter_tools_for_task(tools: list, task_type: str) -> list:
    """Return the subset of tools appropriate for the given task_type."""
    if task_type == "Intake":
        return [t for t in tools if t["function"]["name"].startswith("patient.")]
    if task_type == "Protocol":
        return []
    if task_type == "Write/Update":
        # only write-tools (no read, no patient)
        return [t for t in tools if t["function"]["name"] in _WRITE_TOOL_NAMES]
    # Information Lookup / Clinical Reasoning / Data Gathering — read EHR tools only
    return [t for t in tools
            if (not t["function"]["name"].startswith("patient."))
            and t["function"]["name"] not in _WRITE_TOOL_NAMES]


def run_agent_turn(
    client: OpenAI,
    messages: list,
    tools: list,
    subject_id: int,
    session_id: str,
    task_type: str,
    gold_turn: list | None = None,
    language: str = "en",
) -> tuple[list, str, list, dict]:
    """
    Run one agentic turn: the LLM calls tools until it signals prepare_to_answer.

    Returns:
        executed_actions: list of {action, observation} dicts
        final_answer: the LLM's final response text
        messages: updated message history
        token_usage: {"in": prompt_tokens, "out": completion_tokens} for the
                     tested model across all LLM calls in this turn
    """
    tok = {"in": 0, "out": 0}
    # Add prepare_to_answer to tools (always available)
    all_tools = list(tools) + [_build_prepare_to_answer_schema()]
    # Deduplicate
    seen = set()
    deduped_tools = []
    for t in all_tools:
        name = t["function"]["name"]
        if name not in seen:
            seen.add(name)
            deduped_tools.append(t)
    # Sanitize tool names for the API (dots not allowed in function names)
    deduped_tools = _sanitize_tools_for_api(deduped_tools)

    executed_actions = []
    messages = list(messages)

    # Toggle provider-specific "thinking" mode based on the model config
    # (DOCTOR_THINKING), not by guessing from the model name. Default OFF keeps
    # the deterministic non-streaming eval policy; ON requires a larger
    # DOCTOR_MAX_TOKENS so reasoning tokens don't truncate the answer.
    _thinking_eb = _thinking_extra_body(DOCTOR_AGENT_MODEL, DOCTOR_THINKING)

    # Protocol turns: no tool calls, just answer
    if task_type == "Protocol":
        resp = client.chat.completions.create(
            model=DOCTOR_AGENT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=DOCTOR_MAX_TOKENS,
            extra_body=_thinking_eb,
        )
        _accum_usage(resp, tok)
        answer = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": answer})
        # Still record prepare_to_answer as action
        executed_actions.append({
            "action": {"name": "prepare_to_answer", "arguments": {}},
            "observation": {"ready": True},
            "idx": 0,
        })
        return executed_actions, answer, messages, tok

    # Agentic tool-calling loop
    answer = ""  # initialise so nudge-skip check is always safe
    for step in range(MAX_TOOL_STEPS):
        try:
            resp = client.chat.completions.create(
                model=DOCTOR_AGENT_MODEL,
                messages=messages,
                tools=deduped_tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=DOCTOR_MAX_TOKENS,
                extra_body=_thinking_eb,
            )
        except Exception as e:
            logger.error(f"LLM call failed at step {step}: {e}")
            break

        _accum_usage(resp, tok)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # LLM produced a text response — treat as final answer
            answer = msg.content or ""
            asst = {"role": "assistant", "content": answer}
            # DeepSeek thinking models REQUIRE reasoning_content to be echoed
            # back in subsequent assistant messages (else 400
            # "The reasoning_content in the thinking mode must be passed back
            # to the API"). Preserve it conditionally — other providers don't
            # populate model_extra['reasoning_content'], so this is a no-op.
            extras = getattr(msg, "model_extra", None) or {}
            if extras.get("reasoning_content"):
                asst["reasoning_content"] = extras["reasoning_content"]
            messages.append(asst)
            # Add prepare_to_answer if not already done
            if not executed_actions or executed_actions[-1]["action"]["name"] != "prepare_to_answer":
                executed_actions.append({
                    "action": {"name": "prepare_to_answer", "arguments": {}},
                    "observation": {"ready": True},
                    "idx": len(executed_actions),
                })
            # Path A fix: Gemini sometimes returns no tool_calls AND empty content.
            # The nudge block below is only reached after the for-loop, but this
            # branch returns early — so fall through to the nudge instead of
            # returning an empty answer immediately.
            if answer.strip():
                return executed_actions, answer, messages, tok
            # answer is empty — fall through to the nudge block below
            break

        # Build assistant message with tool_calls
        tool_calls_data = []
        for tc in msg.tool_calls:
            d = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            # Preserve provider-specific extras (only present for some providers).
            # Notably, Gemini 3.x via the OpenAI-compat endpoint puts the required
            # thought_signature in tool_calls[*].extra_content.google.thought_signature;
            # it MUST be echoed back in subsequent turns or the API returns
            # 400 "Function call is missing a thought_signature". Other providers
            # don't populate model_extra here, so this is a no-op for them.
            extras = getattr(tc, "model_extra", None) or {}
            if "extra_content" in extras:
                d["extra_content"] = extras["extra_content"]
            tool_calls_data.append(d)
        asst = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": tool_calls_data,
        }
        # Same DeepSeek-thinking reasoning_content echo requirement (see above).
        extras = getattr(msg, "model_extra", None) or {}
        if extras.get("reasoning_content"):
            asst["reasoning_content"] = extras["reasoning_content"]
        messages.append(asst)

        # Execute each tool call
        done = False
        for tc in msg.tool_calls:
            api_tool_name = tc.function.name
            # Reverse the API-safe name back to call_tool name
            tool_name = _call_name(api_tool_name)
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            args = _normalize_patient_tool_args(tool_name, args, subject_id, session_id)

            # ── ask_user_for_required_parameters: inject gold user_input ──────────
            # Supports multiple consecutive ask rounds (matching WildToolBench pattern).
            # Each call consumes the next ask action from the gold turn in order.
            if tool_name == "ask_user_for_required_parameters":
                # Find the next unused gold ask action (by idx order)
                n_asks_so_far = sum(
                    1 for a in executed_actions
                    if a["action"]["name"] == "ask_user_for_required_parameters"
                )
                gold_ask_actions = [
                    a for a in (gold_turn or [])
                    if a.get("action", {}).get("name") == "ask_user_for_required_parameters"
                ]
                gold_act = gold_ask_actions[n_asks_so_far] if n_asks_so_far < len(gold_ask_actions) else {}
                gold_clarify_obs = str(gold_act.get("observation", ""))
                gold_user_input = str(gold_act.get("user_input", ""))

                executed_actions.append({
                    "action": {"name": tool_name, "arguments": args},
                    "observation": gold_clarify_obs,
                    "user_input": gold_user_input,
                    "idx": len(executed_actions),
                })
                # Return the clarification Q as the tool result, then inject user response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"clarification": gold_clarify_obs}, ensure_ascii=False),
                })
                if gold_user_input:
                    messages.append({"role": "user", "content": gold_user_input})
                continue
            # ── End ask_user_for_required_parameters handling ─────────────────────

            # Execute tool
            try:
                observation = call_tool(tool_name, args)
            except Exception as e:
                observation = {"error": str(e)}

            executed_actions.append({
                "action": {"name": tool_name, "arguments": args},
                "observation": observation,
                "idx": len(executed_actions),
            })

            # Add tool result to messages (Optimization 3: compact FHIR Bundles).
            # _compact_observation_str collapses verbose FHIR JSON to "code: value @ date"
            # lines, cutting obs tokens ~80% while keeping all clinical content.
            obs_str = _compact_observation_str(observation)
            MAX_OBS_CHARS = 4000   # tighter cap; compaction already removes most bulk
            if len(obs_str) > MAX_OBS_CHARS:
                obs_str = obs_str[:MAX_OBS_CHARS] + (
                    f"\n... [truncated: {len(obs_str)} chars total]"
                )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": obs_str,
            })

            if tool_name == "prepare_to_answer":
                done = True

        if done:
            # Path B fix: Gemini sometimes embeds the final answer in msg.content
            # of the same response that contains prepare_to_answer (parallel tool
            # calls). The code stores msg.content in the assistant message but
            # doesn't capture it as `answer`, so the nudge below gets a confusing
            # history and returns empty. Extract it here before the nudge runs.
            if msg.content and msg.content.strip():
                answer = msg.content.strip()
            break

    # After loop: force a final text answer when the step budget was exhausted
    # without the model emitting one. Some providers (notably Gemini 3.x via the
    # OpenAI-compat endpoint) return EMPTY content here if they still "want" to
    # call a tool — this silently produced ~44% blank answers. To prevent that:
    #   1. append an explicit instruction to answer in plain text, no tools;
    #   2. retry up to 3x while content comes back empty.
    # The instruction message is dropped afterward so it doesn't pollute the
    # cross-turn history that the next turn inherits.
    if language == "zh":
        _final_nudge = ("请基于已获取的信息，现在用纯文本给出你的最终答复。"
                        "不要再调用任何工具。")
    else:
        _final_nudge = ("Based on the information gathered, provide your final "
                        "answer now in plain text. Do not call any tools.")
    # Skip the nudge if Path B already captured a non-empty answer from msg.content.
    if answer.strip():
        messages.append({"role": "assistant", "content": answer})
        return executed_actions, answer, messages, tok
    answer = ""
    for _attempt in range(3):
        _msgs = messages + [{"role": "user", "content": _final_nudge}]
        try:
            resp = client.chat.completions.create(
                model=DOCTOR_AGENT_MODEL,
                messages=_msgs,
                tools=deduped_tools,
                tool_choice="none",
                temperature=0.2,
                max_tokens=DOCTOR_MAX_TOKENS,
                extra_body=_thinking_eb,
            )
        except Exception as e:
            logger.error(f"Final answer LLM call failed (attempt {_attempt}): {e}")
            answer = "[Error generating final answer]"
            break
        _accum_usage(resp, tok)
        answer = resp.choices[0].message.content or ""
        if answer.strip():
            break
        logger.warning(f"Final answer empty (attempt {_attempt + 1}/3); retrying")
    if answer.strip():
        messages.append({"role": "assistant", "content": answer})

    return executed_actions, answer, messages, tok


def compute_tool_metrics(
    executed: list,
    gold: list,
    task_type: str,
) -> dict:
    """
    Compare LLM's tool calls against gold answer_list for one turn.

    Returns dict with:
        ap_rate: fraction of gold tools the LLM called (Action Precision Rate)
        tool_names_correct: bool — did LLM call exactly the right tool names?
        tool_coverage_correct: bool — did LLM call at least 1 correct tool?
        op_rate: bool — did LLM call parallel tools in one step (optimal)?
        extra_tools: list of tool names LLM called that weren't in gold
        missed_tools: list of gold tools LLM didn't call
    """
    # Extract tool names (exclude prepare_to_answer for comparison)
    _EXCLUDE = {"prepare_to_answer"}
    llm_tools = [a["action"]["name"] for a in executed
                 if a["action"]["name"] not in _EXCLUDE]
    gold_tools = [a["action"]["name"] for a in gold
                  if a["action"]["name"] not in _EXCLUDE]

    if not gold_tools:
        # Protocol type — no tool calls expected
        return {
            "ap_rate": 1.0 if not llm_tools else 0.0,
            "tool_names_correct": not bool(llm_tools),
            "tool_coverage_correct": True,  # no tools needed = OK
            "op_rate": None,
            "extra_tools": llm_tools,
            "missed_tools": [],
        }

    gold_tool_set = set(gold_tools)
    llm_tool_set = set(llm_tools)

    matched = gold_tool_set & llm_tool_set
    missed = list(gold_tool_set - llm_tool_set)
    extra = list(llm_tool_set - gold_tool_set)

    ap_rate = len(matched) / len(gold_tool_set) if gold_tool_set else 1.0
    tool_names_correct = (gold_tool_set == llm_tool_set)
    tool_coverage_correct = len(matched) > 0

    # OP Rate: did LLM call all parallel tools in a single API step?
    # Proxy: for Data Gathering/Intake, check if LLM called multiple tools before prepare_to_answer
    op_rate = None
    if task_type in ("Data Gathering", "Intake") and len(gold_tool_set) >= 2:
        # Check if all gold tools appear in executed (without interleaving prepare_to_answer)
        # Simple proxy: did LLM make all calls before any prepare_to_answer?
        prep_idx = next((i for i, a in enumerate(executed)
                         if a["action"]["name"] == "prepare_to_answer"), len(executed))
        actual_tool_names = [a["action"]["name"] for a in executed[:prep_idx]]
        # Optimal = called all needed tools (regardless of order for this MVP)
        all_called = all(t in actual_tool_names for t in gold_tools)
        op_rate = 1.0 if (all_called and len(extra) == 0) else 0.0

    return {
        "ap_rate": ap_rate,
        "tool_names_correct": tool_names_correct,
        "tool_coverage_correct": tool_coverage_correct,
        "op_rate": op_rate,
        "extra_tools": extra,
        "missed_tools": missed,
    }


def count_critical_symptoms(
    executed: list,
    annotations: list,
    session_id: str,
) -> dict:
    """
    For Intake turns: count how many critical symptoms the LLM probed.

    Uses the benchmark's patient_agent_annotations to identify what topics
    were explored during gold generation, then checks if LLM also asked about them
    via medication_adherence or symptom_history calls with similar queries.

    Returns {critical_symptoms_total, critical_symptoms_covered}
    """
    # Extract critical medication probes from annotations (gold)
    # A medication adherence call with a specific drug = critical probe
    critical_probes = []
    for ann in annotations:
        if ann.get("query_type") == "get_medication_adherence" and ann.get("query", "").strip():
            critical_probes.append(ann["query"].lower().strip())
        elif ann.get("query_type") == "get_symptom_history" and ann.get("query", "").strip():
            # Symptom queries that were explored in gold
            critical_probes.append(ann["query"].lower().strip())

    if not critical_probes:
        # No structured critical symptoms in annotations — use a fixed count of 1
        # (at least the chief complaint was asked about)
        return {"critical_symptoms_total": 1, "critical_symptoms_covered": 1}

    # Check which of these the LLM also asked about
    llm_queries = []
    for a in executed:
        name = a["action"]["name"]
        args = a["action"].get("arguments", {})
        if name == "patient.get_symptom_history":
            q = str(args.get("query", "")).lower().strip()
            if q:
                llm_queries.append(q)
        elif name == "patient.get_medication_adherence":
            q = str(args.get("drug", "")).lower().strip()
            if q:
                llm_queries.append(q)

    # Fuzzy match: a critical probe is covered if any LLM query shares keywords
    covered = 0
    for probe in critical_probes:
        probe_words = set(probe.split())
        for lq in llm_queries:
            lq_words = set(lq.split())
            if probe_words & lq_words:  # any overlap
                covered += 1
                break

    return {
        "critical_symptoms_total": len(critical_probes),
        "critical_symptoms_covered": covered,
    }


def run_evaluation(
    benchmark_path: str,
    output_dir: str,
    model: str = DOCTOR_AGENT_MODEL,
    verbose: bool = False,
    system_prompt: str | None = None,
    language: str = "en",
    health_literacy: str | None = "high",
    use_explicit: bool = False,
    resume: bool = False,
) -> list:
    """
    Run full evaluation of DeepSeek against all benchmark entries.

    health_literacy: if set ('low'/'medium'/'high'), patient tool turns use the
        specified literacy variant from patient_responses_all_literacy.
        When None, uses the default patient response (as generated).

    Returns list of per-turn result dicts for metrics computation.
    """
    # system_prompt="" means explicitly no system prompt (zero-shot).
    # system_prompt=None means caller didn't specify → use built-in default.
    if system_prompt is None:
        system_prompt = _SYSTEM_PROMPT_ZH if language == "zh" else _SYSTEM_PROMPT
    if verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")

    model_cfg = load_model_config(model)
    api_key = os.environ.get(model_cfg["api_key_env"], "")
    # Doctor-agent key override (paired with EVAL_DOCTOR_BASE_URL). Dedicated var
    # so local-serving scripts can inject a key without it being clobbered by
    # llm_client.load_dotenv(override=True) loading the provider key from .env.
    api_key = os.environ.get("EVAL_DOCTOR_API_KEY") or api_key
    client = _build_client(model_cfg, api_key)
    if os.environ.get("EVAL_DOCTOR_BASE_URL"):
        logger.info(f"Doctor Agent endpoint override → {os.environ['EVAL_DOCTOR_BASE_URL']}")
    # Override the module-level constants so run_agent_turn picks them up.
    global DOCTOR_AGENT_MODEL, DOCTOR_THINKING, DOCTOR_MAX_TOKENS
    DOCTOR_AGENT_MODEL = model_cfg["model_id"]
    DOCTOR_THINKING = bool(model_cfg.get("thinking", False))
    # Thinking models need a bigger output budget (reasoning shares it); fall back
    # to a 10x default if the config enables thinking without specifying max_tokens.
    DOCTOR_MAX_TOKENS = int(model_cfg.get(
        "max_tokens", 10240 if DOCTOR_THINKING else MAX_TOKENS_ANSWER))
    model = model_cfg["model_id"]   # use actual model_id for API calls
    logger.info(f"Doctor Agent: {model_cfg['description']} (model_id={model}) "
                f"thinking={DOCTOR_THINKING} max_tokens={DOCTOR_MAX_TOKENS}")

    with open(benchmark_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(entries)} benchmark entries from {benchmark_path}")

    # Reset patient sessions then re-register all
    reset_all_sessions()
    for entry in entries:
        sid = entry.get("session_id", "")
        subject_id = entry.get("subject_id", 0)
        persona = entry.get("persona_config", {})
        if sid:
            try:
                register_session(sid, subject_id, persona)
                logger.info(f"Registered session {sid} for subject {subject_id}")
                # Preload gold patient responses so evaluation never calls the LLM
                # for patient simulation — responses are replayed from generation data.
                # Scan ALL turns (not just "Intake") because PhysAssistBench has patient tools
                # in Information Lookup/Data Gathering/etc. turns.
                answer_list = entry.get("answer_list", [])
                all_patient_actions: list = []
                for turn_actions in answer_list:
                    for a in (turn_actions or []):
                        obs = a.get("observation", {})
                        if isinstance(obs, dict) and "patient_response" in obs:
                            all_patient_actions.append(a)
                if all_patient_actions and health_literacy:
                    # Rewrite patient_response to use the specified literacy variant
                    rewritten = []
                    for a in all_patient_actions:
                        obs = a.get("observation", {})
                        if isinstance(obs, dict) and "patient_responses_all_literacy" in obs:
                            literacy_variants = obs["patient_responses_all_literacy"]
                            lit_response = literacy_variants.get(
                                health_literacy, obs.get("patient_response", "")
                            )
                            import copy
                            a2 = copy.deepcopy(a)
                            a2["observation"]["patient_response"] = lit_response
                            rewritten.append(a2)
                        else:
                            rewritten.append(a)
                    get_session(sid).preload_responses(rewritten)
                elif all_patient_actions:
                    get_session(sid).preload_responses(all_patient_actions)
            except Exception as e:
                logger.warning(f"Could not register session {sid}: {e}")

    all_turn_results = []
    raw_outputs = []  # for raw storage

    # Resume: load existing checkpoint and skip entries where all turns succeeded.
    _good_keys: set = set()
    if resume:
        _pre_path = os.path.join(output_dir, "turn_results_pre_judge.json")
        _raw_path_ck = os.path.join(output_dir, "raw_agent_outputs.json")
        if os.path.exists(_pre_path) and os.path.exists(_raw_path_ck):
            _existing_turns = json.load(open(_pre_path, encoding="utf-8"))
            _existing_raw   = json.load(open(_raw_path_ck, encoding="utf-8"))
            # Group turns by entry_id; an entry is "good" if all its turns have
            # a non-error llm_answer (no '[Error' placeholder).
            from collections import defaultdict as _dd
            _entry_turns = _dd(list)
            for _t in _existing_turns:
                _entry_turns[_t["test_entry_id"]].append(_t)
            for _eid, _turns in _entry_turns.items():
                if all("[Error" not in str(_t.get("llm_answer", "")) for _t in _turns):
                    _good_keys.add(_eid)
            # Pre-populate results with the good entries so they appear in output.
            all_turn_results = [_t for _t in _existing_turns if _t["test_entry_id"] in _good_keys]
            raw_outputs      = [_r for _r in _existing_raw   if _r["entry_id"]      in _good_keys]
            logger.info(f"[resume] {len(_good_keys)}/{len(_entry_turns)} entries already good, "
                        f"re-running {len(_entry_turns)-len(_good_keys)} entries with errors.")
        else:
            logger.info("[resume] No existing checkpoint found, starting fresh.")

    for entry_idx, entry in enumerate(entries):
        entry_id = entry.get("id", f"entry_{entry_idx}")

        if entry_id in _good_keys:
            logger.info(f"  [resume] skip {entry_id} (already complete)")
            continue

        session_id = entry.get("session_id", "")
        subject_id = entry.get("subject_id", 0)
        hadm_id = entry.get("hadm_id")
        task_domain = entry.get("clinical_task_domain", "LabInterp")
        scenario = entry.get("clinical_scenario", "")
        difficulty_level = entry.get("difficulty_level")
        # Fall back to parsing from entry_id if not present
        if not scenario or difficulty_level is None:
            import re as _reid
            _m = _reid.match(r'demo\d+_\w+_(.+?)_(\d+)_(L\d+)$', entry_id)
            if _m:
                if not scenario:
                    scenario = _m.group(1)
                if difficulty_level is None:
                    difficulty_level = int(_m.group(3)[1:])
        task_types = entry.get("task_types", [])
        tool_sources = entry.get("tool_sources", [])
        # Language-aware field selection: prefer *_zh fields for Chinese,
        # *_en fields for English; fall back to original fields for old entries.
        if language == "zh":
            tasks = entry.get("tasks_zh") or entry.get("tasks", [])
            tools_schema = entry.get("tools_zh") or entry.get("tools", [])
            answer_list = entry.get("answer_list_zh") or entry.get("answer_list", [])
        else:
            if use_explicit:
                tasks = entry.get("tasks_en_explicit") or entry.get("tasks_en") or entry.get("tasks", [])
            else:
                tasks = entry.get("tasks_en") or entry.get("tasks", [])
            tools_schema = entry.get("tools_en") or entry.get("tools", [])
            answer_list = entry.get("answer_list_en") or entry.get("answer_list", [])
        annotations = entry.get("patient_agent_annotations", [])
        env_info = entry.get("env_info", "")

        # Gate FHIR time-ordered queries to the admission's date upper bound.
        # Prefer the machine-readable field written during generation; fall back
        # to parsing from env_info for older entries that lack it.
        import re as _re
        current_date = entry.get("current_date") or (
            (m := _re.search(r"Current date: (\d{4}-\d{2}-\d{2})", env_info))
            and m.group(1)
        ) or None
        set_active_date(current_date)

        logger.info(f"\n{'='*60}")
        logger.info(f"Entry {entry_idx+1}/{len(entries)}: {entry_id}")
        logger.info(f"  session_id={session_id}  subject={subject_id}  current_date={current_date}")
        logger.info(f"  task_types={task_types}")

        # Conversation history (grows across turns)
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        # Inject env_info as first user message — include subject_id and hadm_id
        # so the agent can pass them directly to EHR tool arguments.
        if language == "zh":
            messages.append({
                "role": "user",
                "content": (
                    f"[环境信息]\n{env_info}\n"
                    f"subject_id: {subject_id}\n"
                    f"hadm_id: {hadm_id}\n\n"
                    "您现在正在协助处理一个临床病例。我将逐一给您布置任务。"
                )
            })
            messages.append({
                "role": "assistant",
                "content": "好的。我已准备好协助处理该临床病例。请给我第一个任务。"
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    f"[Environment Info]\n{env_info}\n"
                    f"subject_id: {subject_id}\n"
                    f"hadm_id: {hadm_id}\n\n"
                    "You are now assisting with a clinical case. I will give you tasks one at a time."
                )
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I'm ready to assist with the clinical case. Please give me the first task."
            })

        entry_raw = {
            "entry_id": entry_id,
            "session_id": session_id,
            "subject_id": subject_id,
            "model": model,
            "turns": []
        }

        for turn_idx, task_question in enumerate(tasks):
            task_type = task_types[turn_idx] if turn_idx < len(task_types) else "Lookup"
            gold_turn = answer_list[turn_idx] if turn_idx < len(answer_list) else []

            logger.info(f"  Turn {turn_idx} [{task_type}]: {task_question[:80]}...")

            # Pass the FULL tool set every turn. This is a tool-use benchmark:
            # the model must autonomously decide which tools (read / write / patient)
            # to call. Per-task-type filtering would leak the turn's nature (e.g. only
            # showing write tools on a Write/Update turn), so it is disabled.
            turn_tools = list(tools_schema)

            # Add task question to messages
            messages.append({"role": "user", "content": task_question})

            # Run agentic loop
            try:
                executed, answer, messages, turn_tokens = run_agent_turn(
                    client=client,
                    messages=messages,
                    tools=turn_tools,
                    subject_id=subject_id,
                    session_id=session_id,
                    task_type=task_type,
                    gold_turn=gold_turn,
                    language=language,
                )
            except Exception as e:
                logger.error(f"  Turn {turn_idx} failed: {e}")
                logger.debug(traceback.format_exc())
                executed = [{"action": {"name": "prepare_to_answer", "arguments": {}},
                             "observation": {"error": str(e)}, "idx": 0}]
                answer = f"[Error: {e}]"
                turn_tokens = {"in": 0, "out": 0}

            # Accumulate token usage globally for cost reporting
            if "total_token_usage" not in locals():
                pass
            if not hasattr(run_evaluation, "_tok_total"):
                run_evaluation._tok_total = {"in": 0, "out": 0, "cache_hit": 0}
            for k in ("in", "out", "cache_hit"):
                run_evaluation._tok_total[k] = run_evaluation._tok_total.get(k, 0) + turn_tokens.get(k, 0)

            # Compute tool metrics
            tool_metrics = compute_tool_metrics(executed, gold_turn, task_type)

            # Critical symptom coverage (Intake only)
            crit_metrics = {}
            if task_type == "Intake":
                crit_metrics = count_critical_symptoms(executed, annotations, session_id)

            # Gold answer (for judge evaluation later)
            gold_answer_text = ""
            if gold_turn:
                # Last action in gold turn has no "answer" field directly
                # Find it via entry messages if available
                pass
            gold_messages = (
                entry.get("messages_zh") if language == "zh"
                else entry.get("messages_en")
            ) or entry.get("messages", [])

            # Turn result record
            turn_result = {
                "test_entry_id": entry_id,
                "task_idx": turn_idx,
                "task_type": task_type,
                "turn_subtype": (entry.get("turn_subtypes") or [None] * (turn_idx + 1))[
                    turn_idx] if turn_idx < len(entry.get("turn_subtypes") or []) else None,
                "clinical_task_domain": task_domain,
                "scenario": scenario,
                "difficulty_level": difficulty_level,
                "language": language,
                "health_literacy": health_literacy,
                "tokens_in":   turn_tokens.get("in", 0),
                "tokens_out":  turn_tokens.get("out", 0),
                "tokens_cache_hit": turn_tokens.get("cache_hit", 0),
                "user_question": task_question,
                "llm_executed_actions": executed,
                "llm_answer": answer,
                "gold_actions": gold_turn,
                # Tested-model token usage for this turn
                "tokens_in": turn_tokens.get("in", 0),
                "tokens_out": turn_tokens.get("out", 0),
                # Tool metrics
                "ap_rate": tool_metrics["ap_rate"],
                "tool_names_correct": tool_metrics["tool_names_correct"],
                "tool_coverage_correct": tool_metrics["tool_coverage_correct"],
                "op_rate": tool_metrics["op_rate"],
                "extra_tools": tool_metrics["extra_tools"],
                "missed_tools": tool_metrics["missed_tools"],
                # Critical symptoms (Intake only)
                **crit_metrics,
                # Placeholder — filled by eval_judge
                "is_correct": None,
                "is_optimal": tool_metrics["op_rate"],
                "complete_steps": round(tool_metrics["ap_rate"] * max(len(gold_turn) - 1, 1)),
                "total_steps": max(len(gold_turn) - 1, 1),
                "hallucination": False,  # filled by judge
                "safety_violation": False,  # filled by judge
                "temporal_correct": None,
            }
            all_turn_results.append(turn_result)

            entry_raw["turns"].append({
                "turn_idx": turn_idx,
                "task_type": task_type,
                "question": task_question,
                "executed": executed,
                "answer": answer,
                "tool_metrics": tool_metrics,
            })

            logger.info(
                f"    AP={tool_metrics['ap_rate']:.2f}  "
                f"coverage={tool_metrics['tool_coverage_correct']}  "
                f"missed={tool_metrics['missed_tools']}"
            )

        raw_outputs.append(entry_raw)

        # Incremental checkpoint: flush after every entry so partial results
        # survive job cancellation or quota exhaustion mid-run.
        raw_path = os.path.join(output_dir, "raw_agent_outputs.json")
        turns_path = os.path.join(output_dir, "turn_results_pre_judge.json")
        for _path, _obj in [(raw_path, raw_outputs), (turns_path, all_turn_results)]:
            _tmp = _path + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as _f:
                json.dump(_obj, _f, ensure_ascii=False, indent=2, default=str)
            os.replace(_tmp, _path)
        logger.info(f"  [checkpoint] {entry_idx+1}/{len(entries)} entries saved")

    logger.info(f"\nRaw outputs saved to {raw_path}")
    logger.info(f"Turn results (pre-judge) saved to {turns_path}")

    return all_turn_results, raw_outputs
