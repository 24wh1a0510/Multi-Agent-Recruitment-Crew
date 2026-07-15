"""
utils/logger.py
----------------
Structured logging helpers shared by every agent. Writes both to the
shared LangGraph state (so the UI can render a timeline) and to the
standard Python logging module (so the FastAPI/uvicorn console shows
what's happening).
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def make_log_entry(
    agent: str,
    message: str,
    duration_ms: Optional[float] = None,
    tokens_used: Optional[int] = None,
    level: str = "INFO",
) -> Dict[str, Any]:
    from datetime import datetime

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent,
        "message": message,
        "duration_ms": duration_ms,
        "tokens_used": tokens_used,
        "level": level,
    }


@contextmanager
def timed_step(agent_name: str) -> Generator[Dict[str, Any], None, None]:
    """Context manager that measures wall-clock time for an agent step.

    Usage:
        with timed_step("Resume Analyst") as timer:
            ... do work ...
        # timer["duration_ms"] is populated after the block exits
    """
    logger = get_logger(agent_name)
    start = time.perf_counter()
    timer: Dict[str, Any] = {"duration_ms": 0.0}
    logger.info("Starting step")
    try:
        yield timer
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        timer["duration_ms"] = round(elapsed, 2)
        logger.info("Finished step in %.2fms", elapsed)


def estimate_tokens(*texts: str) -> int:
    """Very rough token estimate (chars / 4) used when the LLM response
    does not carry usage metadata (e.g. some OpenRouter models)."""
    total_chars = sum(len(t or "") for t in texts)
    return max(1, total_chars // 4)
