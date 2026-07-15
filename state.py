"""
state.py
--------
Defines the shared state object that flows through the LangGraph
recruitment pipeline. All agents read from and write to this single
TypedDict -- they never communicate directly with one another.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from models import HiringDecision, ParsedProfile, ScoreCard, VerificationResult


def _append(a: List[Any], b: List[Any]) -> List[Any]:
    """Reducer used so that concurrent/sequential node writes to `logs`
    are appended rather than overwritten, which is required by LangGraph
    when multiple nodes touch the same state key across steps."""
    return (a or []) + (b or [])


class RecruitmentState(TypedDict, total=False):
    # --- Inputs ---------------------------------------------------------
    job_description: str
    candidate_resume: str  # raw extracted text from the uploaded PDF
    resume_filename: str

    # --- Agent outputs (validated Pydantic models, stored as dicts) -----
    parsed_profile: Optional[Dict[str, Any]]
    scorecard: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]
    decision: Optional[Dict[str, Any]]

    # --- Control flow -----------------------------------------------------
    revision_count: int
    needs_verification: bool
    escalated: bool
    current_agent: str
    execution_path: Annotated[List[str], _append]

    # --- Observability ------------------------------------------------------
    logs: Annotated[List[Dict[str, Any]], _append]
    errors: Annotated[List[Dict[str, Any]], _append]
    timings_ms: Annotated[List[Dict[str, Any]], _append]
    tokens_used: Annotated[List[Dict[str, Any]], _append]


def new_initial_state(job_description: str, candidate_resume: str, resume_filename: str = "") -> RecruitmentState:
    """Factory for a clean initial state dict."""
    return RecruitmentState(
        job_description=job_description,
        candidate_resume=candidate_resume,
        resume_filename=resume_filename,
        parsed_profile=None,
        scorecard=None,
        verification_result=None,
        decision=None,
        revision_count=0,
        needs_verification=False,
        escalated=False,
        current_agent="",
        execution_path=[],
        logs=[],
        errors=[],
        timings_ms=[],
        tokens_used=[],
    )


def parsed_profile_model(state: RecruitmentState) -> Optional[ParsedProfile]:
    data = state.get("parsed_profile")
    return ParsedProfile(**data) if data else None


def scorecard_model(state: RecruitmentState) -> Optional[ScoreCard]:
    data = state.get("scorecard")
    return ScoreCard(**data) if data else None


def verification_model(state: RecruitmentState) -> Optional[VerificationResult]:
    data = state.get("verification_result")
    return VerificationResult(**data) if data else None


def decision_model(state: RecruitmentState) -> Optional[HiringDecision]:
    data = state.get("decision")
    return HiringDecision(**data) if data else None
