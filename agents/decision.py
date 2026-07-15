"""
agents/decision.py
-------------------
Decision Agent: reads the (possibly verified) score and produces a final
hiring recommendation with reasoning.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import ValidationError

from models import DecisionLabel, HiringDecision, VerificationStatus
from state import RecruitmentState
from utils.helpers import call_with_retry, extract_json, get_llm, truncate
from utils.logger import estimate_tokens, get_logger, make_log_entry, timed_step

logger = get_logger("DecisionAgent")

AGENT_NAME = "Decision Agent"

_SYSTEM_PROMPT = """You are the final decision-maker in a recruitment pipeline.
Given a candidate's verified scorecard and the job description, produce a
final hiring recommendation.

Decision must be exactly one of: "Hire", "Interview", "Hold", "Reject".
Guidance: overall_score >= 4.2 -> Hire; 3.4-4.19 -> Interview;
2.5-3.39 -> Hold; below 2.5 -> Reject. Use judgment but stay close to this rubric.

Return ONLY valid JSON with exactly these keys:
{
  "decision": "Hire" | "Interview" | "Hold" | "Reject",
  "confidence": number (0-1),
  "reasoning": "2-4 sentence explanation",
  "recommendations": ["short actionable next steps"]
}
"""


def _build_user_prompt(scorecard: Dict[str, Any], jd: str, escalated: bool) -> str:
    extra = (
        "\nNOTE: This candidate's score required human escalation after verification "
        "retries were exhausted. Reflect appropriate caution in your reasoning."
        if escalated
        else ""
    )
    return (
        f"Job Description:\n---\n{truncate(jd)}\n---\n\n"
        f"Final Scorecard (JSON):\n---\n{scorecard}\n---\n{extra}\n\n"
        "Return the JSON hiring decision now."
    )


def run_decision_agent(state: RecruitmentState) -> RecruitmentState:
    """LangGraph node function for the Decision Agent."""
    with timed_step(AGENT_NAME) as timer:
        errors = []
        logs = [make_log_entry(AGENT_NAME, "Generating final hiring recommendation...")]

        scorecard = dict(state.get("scorecard", {}) or {})
        verification = state.get("verification_result", {}) or {}
        jd = state.get("job_description", "")
        escalated = state.get("escalated", False)

        # Use the verification-adjusted score if verification ran and confirmed/rejected an adjustment.
        if verification.get("adjusted_score") is not None:
            scorecard["overall_score"] = verification["adjusted_score"]

        decision_data: Dict[str, Any] = {}
        try:
            llm = get_llm(temperature=0.1)

            def _invoke():
                return llm.invoke(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(scorecard, jd, escalated)},
                    ]
                )

            response = call_with_retry(_invoke, max_attempts=3)
            content = response.content if hasattr(response, "content") else str(response)
            data = extract_json(content)

            decision = HiringDecision(
                decision=DecisionLabel(data["decision"]),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "")),
                recommendations=list(data.get("recommendations", [])),
                requires_human_review=escalated
                or verification.get("status") == VerificationStatus.ESCALATED.value,
            )
            decision_data = decision.model_dump()

            tokens = getattr(response, "response_metadata", {}).get("token_usage", {}).get(
                "total_tokens"
            ) or estimate_tokens(str(scorecard), jd, content)

            logs.append(
                make_log_entry(
                    AGENT_NAME,
                    f"Final decision: {decision.decision.value} (confidence {decision.confidence:.2f})",
                    tokens_used=tokens,
                )
            )

        except (ValidationError, KeyError, ValueError) as exc:
            logger.error("Validation failed for hiring decision: %s", exc)
            errors.append(
                make_log_entry(AGENT_NAME, f"Malformed agent output: {exc}", level="ERROR")
            )
            decision_data = HiringDecision(
                decision=DecisionLabel.HOLD,
                confidence=0.0,
                reasoning=f"Decision generation failed due to malformed output: {exc}",
                requires_human_review=True,
            ).model_dump()

        except Exception as exc:  # noqa: BLE001
            logger.error("Decision Agent failed: %s", exc)
            errors.append(make_log_entry(AGENT_NAME, f"LLM failure: {exc}", level="ERROR"))
            decision_data = HiringDecision(
                decision=DecisionLabel.HOLD,
                confidence=0.0,
                reasoning=f"Decision generation failed due to an LLM error: {exc}",
                requires_human_review=True,
            ).model_dump()

    logs.append(
        make_log_entry(AGENT_NAME, "Decision complete.", duration_ms=timer["duration_ms"])
    )

    return {
        "decision": decision_data,
        "current_agent": AGENT_NAME,
        "execution_path": [AGENT_NAME],
        "logs": logs,
        "errors": errors,
        "timings_ms": [{"agent": AGENT_NAME, "duration_ms": timer["duration_ms"]}],
    }
