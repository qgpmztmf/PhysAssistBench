"""
translator.py — Lightweight LLM-based translation utility for bilingual benchmark generation.

Translates clinical text between English and Chinese.
Uses the shared llm_client so that any --generation_model override also applies here.
"""

import logging

from physassistbench.pipeline.agents.llm_client import llm_call

logger = logging.getLogger(__name__)


_TRANSLATE_TO_ZH_PROMPT = (
    "You are a professional medical translator specializing in clinical EHR queries. "
    "Your ONLY task is to translate the given text from English to Chinese. "
    "NEVER answer questions, explain concepts, or add any content not present in the original. "
    "Preserve all medical terminology and clinical meaning exactly. "
    "Output ONLY the translated Chinese text, nothing else.\n\n"
    "CRITICAL RULE for clinical value queries:\n"
    "English 'What is [lab/vital/drug]?' in a clinical EHR context means asking for the VALUE, "
    "NOT asking for a definition. Translate using the pattern '[item]是多少？' or '[item]的值是多少？', "
    "NOT '什么是[item]？'.\n"
    "Examples:\n"
    "  'What is the creatinine?' → '肌酐是多少？'\n"
    "  'What is the hemoglobin A1c?' → '糖化血红蛋白是多少？'\n"
    "  'What is the potassium?' → '血钾是多少？'\n"
    "  'What is the latest WBC?' → '最新的白细胞计数是多少？'"
)

_TRANSLATE_TO_EN_PROMPT = (
    "You are a professional medical translator. "
    "Your ONLY task is to translate the given text from Chinese to English word-for-word. "
    "NEVER answer questions, explain concepts, or add any content not present in the original. "
    "If the input is a question, output the English translation of that question. "
    "If the input is a statement, output the English translation of that statement. "
    "Preserve all medical terminology and clinical meaning exactly. "
    "Output ONLY the translated English text, nothing else."
)


def translate_to_zh(text: str) -> str:
    """Translate English clinical text to Chinese."""
    if not text or not text.strip():
        return text
    try:
        return llm_call(
            messages=[
                {"role": "system", "content": _TRANSLATE_TO_ZH_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=4000,
        ) or text
    except Exception as e:
        logger.warning(f"translate_to_zh failed: {e} — returning original text")
        return text


def translate_to_en(text: str) -> str:
    """Translate Chinese clinical text to English."""
    if not text or not text.strip():
        return text
    try:
        return llm_call(
            messages=[
                {"role": "system", "content": _TRANSLATE_TO_EN_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=4000,
        ) or text
    except Exception as e:
        logger.warning(f"translate_to_en failed: {e} — returning original text")
        return text


def translate_messages(messages: list, target_language: str) -> list:
    """
    Translate user AND assistant message content to target_language.
    SEP markers (plain strings) are passed through unchanged.
    """
    translate_fn = translate_to_zh if target_language == "zh" else translate_to_en
    result = []
    for msg in messages:
        if isinstance(msg, str):
            result.append(msg)
        elif isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            content = msg.get("content") or ""
            translated = translate_fn(content) if content.strip() else content
            result.append({**msg, "content": translated})
        else:
            result.append(msg)
    return result
