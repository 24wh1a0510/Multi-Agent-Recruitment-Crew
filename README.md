# Multi-Agent Recruitment Crew

A production-quality, multi-agent candidate evaluation system built with **LangGraph**, **FastAPI**, and **Streamlit**. Specialized agents — Resume Analyst, Scoring Agent, Verification Agent, and Decision Agent — collaborate through a shared state graph, coordinated by a Supervisor node, to turn a resume PDF and a job description into a transparent, auditable hiring recommendation.

This is a standalone project with its own architecture, dependencies, and UI.

---

## Architecture

```
                    Supervisor
                        │
                Resume Analyst
                        │
                 Scoring Agent
                        │
              (borderline score?)
                 ┌──────┴──────┐
                 │             │
                 ▼             │
       Verification Agent      │
                 │             │
        (rejected & retries    │
         remaining?)           │
                 │             │
        ┌────────┴────────┐    │
        ▼                 ▼    ▼
  back to Scoring    Decision Agent
     (retry)                │
                            END
```

- **Supervisor** — validates inputs (resume + job description present) and dispatches the run.
- **Resume Analyst** — extracts a structured `ParsedProfile` (name, email, skills, experience, education, projects, certifications) from resume text via an LLM call, with prompt-injection-aware instructions.
- **Scoring Agent** — produces a rubric-based `ScoreCard` (skills, experience, education, projects, communication, overall) comparing the candidate to the job description.
- **Verification Agent** — runs **only** when the overall score falls in a configurable borderline band (default `2.8–3.4`). It independently re-scores the candidate on an **anonymized** profile (name-swapped for bias checking), scans for prompt-injection heuristics, and either confirms, rejects (triggering a retry back to the Scoring Agent, up to `MAX_RETRIES`), or escalates to a human after retries are exhausted.
- **Decision Agent** — reads the final (possibly verified) score and produces a `HiringDecision`: Hire / Interview / Hold / Reject, with reasoning and recommendations.

All agents communicate **only** through the shared `RecruitmentState` (a `TypedDict`) — never directly with one another — and every handoff is validated with Pydantic models (`ParsedProfile`, `ScoreCard`, `VerificationResult`, `HiringDecision`).

---

## Tech Stack

- Python 3.12+
- FastAPI + Uvicorn (backend API)
- LangGraph + LangChain (agent orchestration)
- OpenAI or OpenRouter (LLM provider, OpenAI-compatible)
- Pydantic (structured validation)
- Streamlit (frontend UI)
- python-dotenv (config)

---

## Project Structure

```
multi-agent-recruitment-crew/
├── app.py                   # FastAPI backend, exposes /run pipeline endpoint
├── graph.py                 # LangGraph StateGraph wiring + conditional routing
├── state.py                 # Shared RecruitmentState TypedDict
├── config.py                # Env-driven settings (Settings dataclass)
├── models.py                # Pydantic models for every agent handoff
├── requirements.txt
├── README.md
├── .env.example
│
├── agents/
│   ├── supervisor.py         # Entry node: input validation & dispatch
│   ├── analyst.py             # Resume Analyst
│   ├── scorer.py               # Scoring Agent
│   ├── verifier.py             # Verification Agent (bias/injection/retry/escalation)
│   └── decision.py             # Decision Agent
│
├── utils/
│   ├── parser.py             # PDF text extraction, JD validation
│   ├── logger.py              # Structured logging + timing helpers
│   ├── scoring.py              # Borderline detection, injection heuristics, name-swap
│   └── helpers.py               # LLM client factory, JSON extraction, retry/backoff
│
├── frontend/
│   └── streamlit_ui.py        # Glassmorphism multi-page Streamlit UI
│
├── data/
│   ├── sample_resume.pdf     # Bundled demo resume
│   └── sample_jd.txt          # Bundled demo job description
│
└── screenshots/
```

---

## Installation

```bash
git clone <this-repo-url>
cd multi-agent-recruitment-crew

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENAI_API_KEY (or USE_OPENROUTER=true + OPENROUTER_API_KEY)
```

---

## Running the App

You need **two processes**: the FastAPI backend (runs the LangGraph pipeline) and the Streamlit frontend (talks to it over HTTP).

**Terminal 1 — backend:**
```bash
uvicorn app:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
streamlit run frontend/streamlit_ui.py
```

Then open the Streamlit URL (typically `http://localhost:8501`). Use the sidebar to navigate: **Home → Upload Resume → Job Description → Run Crew → Execution Logs → Final Report**. You can load the bundled sample resume/JD with one click on their respective pages.

Interactive API docs (Swagger) are available at `http://localhost:8000/docs`.

---

## Configuration

All configuration lives in `.env` (see `.env.example`):

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `USE_OPENROUTER` | Use OpenRouter instead of OpenAI | `false` |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | OpenRouter credentials | — |
| `MODEL_NAME` | Chat model to use | `gpt-4o-mini` |
| `MODEL_TEMPERATURE` | Sampling temperature | `0.2` |
| `BORDERLINE_LOW` / `BORDERLINE_HIGH` | Score band that triggers verification | `2.8` / `3.4` |
| `MAX_RETRIES` | Max verification retries before human escalation | `3` |
| `API_HOST` / `API_PORT` | FastAPI bind address | `0.0.0.0` / `8000` |
| `BACKEND_URL` | Backend URL the Streamlit UI calls | `http://localhost:8000` |
| `MAX_PDF_SIZE_MB` | Upload size limit | `10` |
| `LOG_LEVEL` | Python logging level | `INFO` |

---

## Observability

Every pipeline run returns the full shared state, which the UI renders as:

- **Execution Path** — ordered list of agents that ran (animated timeline)
- **Current Active Agent** — surfaced during/after each step
- **Execution Time per Agent** — bar chart from `timings_ms`
- **Tokens Used** — per-step estimate/actual from `logs`
- **Full Log Stream** — timestamped, leveled log entries
- **Shared State Viewer** — raw JSON of the entire `RecruitmentState`
- **Retry Counter / Escalation Flag** — visible in metrics and the Verification Summary

---

## Error Handling

| Scenario | Handling |
|---|---|
| Invalid / corrupt PDF | `PDFParsingError` → HTTP 422 with a clear message |
| Missing resume or JD | `MissingInputError` → HTTP 422, Supervisor also logs and short-circuits the graph |
| Malformed agent JSON output | Caught via Pydantic `ValidationError`; falls back to a safe default and logs an `ERROR` entry rather than crashing the graph |
| LLM call failure / timeout | `call_with_retry` retries with exponential backoff (3 attempts); final failure logged and a safe fallback value is used |
| Verification rejects the score | Routed back to the Scoring Agent, `revision_count` incremented |
| Max retries reached | `VerificationStatus.ESCALATED`; Decision Agent flags `requires_human_review=True` |

---

## Security Notes

- Resume and job description content is always treated as **data**, never as instructions — every LLM system prompt explicitly tells the model to ignore embedded instructions.
- The Verification Agent runs an additional static regex-based prompt-injection scan independent of the LLM's own judgment.
- Bias mitigation: the Verification Agent re-scores using a **name-anonymized** copy of the profile and compares against the original score for consistency.

---

## License

MIT — use freely, adapt for your own recruitment workflows.
