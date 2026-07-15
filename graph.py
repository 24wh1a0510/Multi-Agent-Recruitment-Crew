"""
graph.py
--------
Builds and compiles the LangGraph StateGraph that orchestrates the
recruitment crew:

    START -> Supervisor -> Resume Analyst -> Scoring Agent
                -> [conditional] -> Verification Agent (if borderline)
                                 -> Decision Agent (if not borderline)
             Verification Agent -> [conditional]
                    -> Scoring Agent (retry, if rejected & retries remain)
                    -> Decision Agent (if confirmed, skipped, or escalated)
                -> END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.analyst import run_resume_analyst
from agents.decision import run_decision_agent
from agents.scorer import run_scoring_agent
from agents.supervisor import run_supervisor
from agents.verifier import run_verification_agent
from config import settings
from models import VerificationStatus
from state import RecruitmentState

# --- Conditional routing functions ------------------------------------------


def route_after_supervisor(state: RecruitmentState) -> str:
    """If the supervisor recorded a validation error, short-circuit to END."""
    if state.get("errors"):
        return "end"
    return "continue"


def route_after_scoring(state: RecruitmentState) -> str:
    """Send to Verification Agent only if the score is borderline."""
    if state.get("needs_verification"):
        return "verify"
    return "decide"


def route_after_verification(state: RecruitmentState) -> str:
    """Decide whether to retry scoring, proceed to decision, or stop for
    human escalation (escalation still proceeds to Decision Agent, but the
    decision is flagged as requiring human review)."""
    verification = state.get("verification_result", {}) or {}
    status = verification.get("status")
    revision_count = state.get("revision_count", 0)

    if status == VerificationStatus.REJECTED.value and revision_count < settings.max_retries:
        return "retry"
    # confirmed, skipped, escalated, or rejected-but-out-of-retries -> proceed
    return "decide"


# --- Graph construction -------------------------------------------------------


def build_graph():
    graph = StateGraph(RecruitmentState)

    graph.add_node("supervisor", run_supervisor)
    graph.add_node("resume_analyst", run_resume_analyst)
    graph.add_node("scoring_agent", run_scoring_agent)
    graph.add_node("verification_agent", run_verification_agent)
    graph.add_node("decision_agent", run_decision_agent)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"continue": "resume_analyst", "end": END},
    )

    graph.add_edge("resume_analyst", "scoring_agent")

    graph.add_conditional_edges(
        "scoring_agent",
        route_after_scoring,
        {"verify": "verification_agent", "decide": "decision_agent"},
    )

    graph.add_conditional_edges(
        "verification_agent",
        route_after_verification,
        {"retry": "scoring_agent", "decide": "decision_agent"},
    )

    graph.add_edge("decision_agent", END)

    return graph.compile()


# Compiled singleton graph reused across requests.
recruitment_graph = build_graph()
