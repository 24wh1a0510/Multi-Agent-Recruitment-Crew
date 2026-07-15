"""
agents/supervisor.py
---------------------
Supervisor: the entry node of the graph. It validates that required
inputs are present, initializes run-level bookkeeping, and hands off to
the Resume Analyst. It does not call an LLM -- it's a lightweight
orchestration/validation gate, mirroring the architecture diagram where
the Supervisor sits above the specialist agents.
"""

from __future__ import annotations

from state import RecruitmentState
from utils.logger import get_logger, make_log_entry, timed_step
from utils.parser import MissingInputError, validate_job_description

logger = get_logger("Supervisor")

AGENT_NAME = "Supervisor"


def run_supervisor(state: RecruitmentState) -> RecruitmentState:
    """LangGraph entry node. Validates inputs before dispatching to the crew."""
    with timed_step(AGENT_NAME) as timer:
        errors = []
        logs = [make_log_entry(AGENT_NAME, "Initializing recruitment crew run.")]

        try:
            validate_job_description(state.get("job_description"))
            if not state.get("candidate_resume"):
                raise MissingInputError("Candidate resume text is missing.")
            logs.append(make_log_entry(AGENT_NAME, "Inputs validated. Dispatching to Resume Analyst."))
        except MissingInputError as exc:
            logger.error("Input validation failed: %s", exc)
            errors.append(make_log_entry(AGENT_NAME, str(exc), level="ERROR"))

    return {
        "current_agent": AGENT_NAME,
        "execution_path": [AGENT_NAME],
        "revision_count": state.get("revision_count", 0),
        "logs": logs,
        "errors": errors,
        "timings_ms": [{"agent": AGENT_NAME, "duration_ms": timer["duration_ms"]}],
    }
