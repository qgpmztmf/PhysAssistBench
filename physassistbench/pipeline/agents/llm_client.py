"""
Shared DeepSeek LLM client (mirrors WildToolBench's request pattern).
Reads credentials from wild-tool-bench/.env (already configured).
"""

import json
import os
import time
from typing import Optional, Union

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Load credentials from the repo-root .env (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL).
# Falls back silently to the ambient environment if the file does not exist.
from physassistbench.paths import ENV_PATH as _ENV_PATH
load_dotenv(dotenv_path=_ENV_PATH, override=False)

_client: Optional[OpenAI] = None
_default_model: str = "deepseek-v4-flash"
_use_thinking_disabled: bool = True   # DeepSeek-specific; disabled for other providers

_judge_client: Optional[OpenAI] = None
_judge_model: str = "deepseek-v4-flash"
_judge_use_thinking_disabled: bool = True  # mirrors main client default


def configure_generation_model(model_cfg: dict, api_key: str) -> None:
    """
    Reconfigure llm_client to use a different model for generation.
    Call this before starting generation when --generation_model is set.
    model_cfg: entry from model_configs.yaml (has name, model_id, api_base, api_type, etc.)
    """
    import httpx as _httpx
    global _client, _default_model, _use_thinking_disabled

    _default_model = model_cfg["model_id"]

    # Disable DeepSeek thinking-mode override for non-DeepSeek providers
    _use_thinking_disabled = "deepseek" in model_cfg.get("api_base", "").lower()

    if model_cfg.get("api_type") == "gateway":
        endpoint_path = model_cfg["endpoint_path"]

        def _rewrite_url(request: _httpx.Request) -> None:
            if request.url.path == "/chat/completions":
                request.url = request.url.copy_with(
                    path=f"{endpoint_path}/chat/completions"
                )

        _client = OpenAI(
            base_url=model_cfg["api_base"],
            api_key="unused",
            default_headers={"Ocp-Apim-Subscription-Key": api_key},
            http_client=_httpx.Client(event_hooks={"request": [_rewrite_url]}),
        )
    else:
        _client = OpenAI(api_key=api_key, base_url=model_cfg["api_base"])


def configure_judge_model(model_cfg: dict, api_key: str) -> None:
    """
    Configure a dedicated judge LLM client used by rubric_eval and iirs scoring.
    Supports Azure Gateway and standard OpenAI-compatible endpoints.
    Call this once at startup before any scoring begins.
    """
    import httpx as _httpx
    global _judge_client, _judge_model, _judge_use_thinking_disabled

    _judge_model = model_cfg["model_id"]
    _judge_use_thinking_disabled = "deepseek" in model_cfg.get("api_base", "").lower()

    if model_cfg.get("api_type") == "gateway":
        endpoint_path = model_cfg["endpoint_path"]

        def _rewrite_url(request: _httpx.Request) -> None:
            if request.url.path == "/chat/completions":
                request.url = request.url.copy_with(
                    path=f"{endpoint_path}/chat/completions"
                )

        _judge_client = OpenAI(
            base_url=model_cfg["api_base"],
            api_key="unused",
            default_headers={"Ocp-Apim-Subscription-Key": api_key},
            http_client=_httpx.Client(event_hooks={"request": [_rewrite_url]}),
        )
    elif model_cfg.get("api_type") == "gateway_responses":
        # GPT-5 series via Gateway Responses API — reuse the duck-typed client
        # from eval_runner. judge_llm_call's chat.completions.create() interface
        # is satisfied by ResponsesAPIClient.
        from physassistbench.eval_runner import ResponsesAPIClient
        _judge_client = ResponsesAPIClient(
            api_base=model_cfg["api_base"],
            model_id=model_cfg["model_id"],
            api_key=api_key,
            reasoning_effort=model_cfg.get("reasoning_effort", "minimal"),
        )
    else:
        _judge_client = OpenAI(api_key=api_key, base_url=model_cfg["api_base"])


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def llm_call(
    messages: list,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
) -> str:
    """Single LLM call. Returns the assistant message content string.

    model: if None, uses the currently configured default model (_default_model).
    Thinking mode is disabled automatically for DeepSeek models to prevent
    reasoning tokens from consuming the output budget.
    """
    client = _get_client()
    effective_model = model if model is not None else _default_model
    kwargs = dict(
        model=effective_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if _use_thinking_disabled:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    if msg.tool_calls:
        # Return JSON representation of tool calls
        return json.dumps([
            {"name": tc.function.name,
             "arguments": json.loads(tc.function.arguments)}
            for tc in msg.tool_calls
        ])
    return msg.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def judge_llm_call(
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 4000,
) -> str:
    """LLM call using the configured judge model (set via configure_judge_model).
    Falls back to the main DeepSeek client if no judge has been configured.
    """
    client = _judge_client if _judge_client is not None else _get_client()
    kwargs = dict(
        model=_judge_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if _judge_use_thinking_disabled:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    return msg.content or ""


def extract_json(text: str) -> "Union[dict, list]":
    """Extract the first JSON object or array from a string."""
    import re

    def _try_parse(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Fix invalid \' escapes inside JSON strings (LLM sometimes emits these)
        cleaned = re.sub(r"\\'", "'", s)
        if cleaned != s:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        return None

    # Try direct parse
    result = _try_parse(text)
    if result is not None:
        return result
    # Try extracting ```json ... ``` block (closed fence)
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        result = _try_parse(m.group(1).strip())
        if result is not None:
            return result
    # Try extracting from unclosed ```json ... (truncated before closing fence)
    m = re.search(r"```(?:json)?\s*([\s\S]+)$", text)
    if m:
        result = _try_parse(m.group(1).strip())
        if result is not None:
            return result
    # Try extracting first { ... } or [ ... ]
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        result = _try_parse(m.group(1))
        if result is not None:
            return result
    raise ValueError(f"Could not extract JSON from: {text[:300]}")
