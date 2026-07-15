"""
config.py
---------
Centralized configuration for the Multi-Agent Recruitment Crew.
Loads environment variables and exposes typed settings used across
agents, the LangGraph orchestration layer, and the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- LLM provider settings (OpenRouter only) --------------------------------
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "openai/gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: _get_float("MODEL_TEMPERATURE", 0.2))
    request_timeout: int = field(default_factory=lambda: _get_int("LLM_TIMEOUT_SECONDS", 60))

    # --- Verification / borderline routing --------------------------------------
    borderline_low: float = field(default_factory=lambda: _get_float("BORDERLINE_LOW", 2.8))
    borderline_high: float = field(default_factory=lambda: _get_float("BORDERLINE_HIGH", 3.4))
    max_retries: int = field(default_factory=lambda: _get_int("MAX_RETRIES", 3))

    # --- App / server settings ---------------------------------------------------
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _get_int("API_PORT", 8000))
    backend_url: str = field(default_factory=lambda: os.getenv("BACKEND_URL", "http://localhost:8000"))

    # --- Misc ---------------------------------------------------------------------
    max_pdf_size_mb: int = field(default_factory=lambda: _get_int("MAX_PDF_SIZE_MB", 10))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def llm_kwargs(self) -> dict:
        """Return kwargs for constructing a LangChain ChatOpenAI client via OpenRouter."""
        return {
            "model": self.model_name,
            "api_key": self.openrouter_api_key,
            "base_url": self.openrouter_base_url,
            "temperature": self.temperature,
            "timeout": self.request_timeout,
        }


settings = Settings()
