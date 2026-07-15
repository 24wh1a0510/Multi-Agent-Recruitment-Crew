"""
utils/helpers.py
-----------------
Small shared utilities: LLM client construction, robust JSON extraction
from LLM responses, and retry/backoff wrapper for LLM calls.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from langchain_openai import ChatOpenAI

from config import settings
from utils.logger import get_logger

logger = get_logger("helpers")

T = TypeVar("T")


class LLMCallError(Exception):
    """Raised when an LLM call fails after all retries are exhausted."""


def get_llm(temperature: Optional[float] = None) -> ChatOpenAI:
    """Construct a ChatOpenAI client using either OpenAI or OpenRouter,
    depending on configuration."""
    kwargs = settings.llm_kwargs()
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw_text: str) -> Dict[str, Any]:
    """Extract the first top-level JSON object from an LLM response.

    Handles the common cases of the model wrapping JSON in markdown code
    fences or adding a short preamble before the object.
    """
    if not raw_text:
        raise ValueError("Empty response from LLM; cannot extract JSON.")

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from LLM output: {exc}") from exc

    raise ValueError("No JSON object could be located in the LLM response.")


def call_with_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    base_delay_seconds: float = 1.5,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """Generic retry wrapper with exponential backoff for flaky LLM/network calls."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - intentionally broad, this wraps arbitrary calls
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, max_attempts, exc)
            if on_retry:
                on_retry(attempt, exc)
            if attempt < max_attempts:
                time.sleep(base_delay_seconds * attempt)
    raise LLMCallError(f"LLM call failed after {max_attempts} attempts: {last_exc}") from last_exc


def truncate(text: str, max_chars: int = 6000) -> str:
    if text is None:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"
