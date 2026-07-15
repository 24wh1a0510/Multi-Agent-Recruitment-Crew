"""
models.py
---------
Pydantic models used to validate every handoff between agents in the
recruitment crew. Each agent reads/writes these structures into the
shared LangGraph state so that malformed output is caught immediately
rather than silently propagating through the graph.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Resume Analyst output
# ---------------------------------------------------------------------------
class ParsedProfile(BaseModel):
    """Structured candidate profile extracted from a resume PDF."""

    name: str = Field(..., min_length=1, description="Candidate full name")
    email: Optional[str] = Field(None, description="Candidate email address")
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience_years: float = Field(0.0, ge=0)
    experience_summary: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text_excerpt: str = Field("", description="First N chars of parsed resume text")

    @field_validator("email")
    @classmethod
    def _basic_email_check(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        if "@" not in v:
            # Don't hard fail the whole pipeline over a slightly malformed
            # email extraction -- just null it out.
            return None
        return v


# ---------------------------------------------------------------------------
# Scoring Agent output
# ---------------------------------------------------------------------------
class ScoreCard(BaseModel):
    """Rubric-based candidate score compared against a job description."""

    skills_score: float = Field(..., ge=0, le=5)
    experience_score: float = Field(..., ge=0, le=5)
    education_score: float = Field(..., ge=0, le=5)
    projects_score: float = Field(..., ge=0, le=5)
    communication_score: float = Field(..., ge=0, le=5)
    overall_score: float = Field(..., ge=0, le=5)
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("overall_score")
    @classmethod
    def _sanity_check_overall(cls, v: float) -> float:
        return round(v, 2)


# ---------------------------------------------------------------------------
# Verification Agent output
# ---------------------------------------------------------------------------
class VerificationStatus(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    SKIPPED = "skipped"  # score was outside the borderline band


class VerificationResult(BaseModel):
    status: VerificationStatus
    adjusted_score: Optional[float] = Field(None, ge=0, le=5)
    prompt_injection_detected: bool = False
    bias_check_passed: bool = True
    consistency_notes: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# Decision Agent output
# ---------------------------------------------------------------------------
class DecisionLabel(str, Enum):
    HIRE = "Hire"
    INTERVIEW = "Interview"
    HOLD = "Hold"
    REJECT = "Reject"


class HiringDecision(BaseModel):
    decision: DecisionLabel
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str
    recommendations: List[str] = Field(default_factory=list)
    requires_human_review: bool = False


# ---------------------------------------------------------------------------
# Observability / logging
# ---------------------------------------------------------------------------
class LogEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    agent: str
    message: str
    duration_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    level: str = "INFO"
