"""
app.py
------
FastAPI backend for the Multi-Agent Recruitment Crew. Exposes a single
primary endpoint that runs the full LangGraph pipeline (PDF resume +
job description in, full shared state with decision out), plus a health
check and sample-data endpoints used by the Streamlit frontend.

Also exposes /transcribe — local Whisper transcription for the Voice
Interview page (faster-whisper, no API key needed, runs on CPU).
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import recruitment_graph
from state import new_initial_state
from utils.logger import get_logger
from utils.parser import (
    MissingInputError,
    PDFParsingError,
    extract_text_from_pdf,
    validate_job_description,
)

logger = get_logger("API")

app = FastAPI(
    title="Multi-Agent Recruitment Crew API",
    description="LangGraph-orchestrated multi-agent candidate evaluation pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunResponse(BaseModel):
    success: bool
    total_duration_ms: float
    state: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run_pipeline(
    job_description: str = Form(...),
    resume: UploadFile = File(...),
) -> RunResponse:
    """Run the full multi-agent recruitment pipeline for one candidate."""
    start = time.perf_counter()

    # --- Validate & extract inputs -----------------------------------------
    try:
        jd_text = validate_job_description(job_description)
    except MissingInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        file_bytes = await resume.read()
        parsed_pdf = extract_text_from_pdf(file_bytes, filename=resume.filename or "resume.pdf")
    except (PDFParsingError, MissingInputError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # --- Run the LangGraph pipeline -----------------------------------------
    initial_state = new_initial_state(
        job_description=jd_text,
        candidate_resume=parsed_pdf.text,
        resume_filename=parsed_pdf.filename,
    )

    try:
        final_state = recruitment_graph.invoke(
            initial_state, config={"recursion_limit": 50}
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline execution failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

    total_duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return RunResponse(success=True, total_duration_ms=total_duration_ms, state=dict(final_state))


@app.get("/sample-jd")
def sample_jd() -> Dict[str, str]:
    """Return the bundled sample job description text for demo purposes."""
    try:
        with open("data/sample_jd.txt", "r", encoding="utf-8") as f:
            return {"job_description": f.read()}
    except FileNotFoundError:
        return {"job_description": ""}


# ---------------------------------------------------------------------------
# TRANSCRIPTION ENDPOINT  (local Whisper via faster-whisper — no API key)
# ---------------------------------------------------------------------------

_whisper_model = None  # loaded lazily on first call


def _get_whisper_model():
    """Load faster-whisper model once and cache it in the module."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            # "base" is fast enough on CPU and accurate for speech
            _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("faster-whisper 'base' model loaded on CPU.")
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="faster-whisper is not installed. Run: pip install faster-whisper",
            )
    return _whisper_model


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    duration_s: float


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (webm/wav/mp3/ogg/flac)"),
) -> TranscribeResponse:
    """
    Transcribe an audio file locally using faster-whisper.
    Accepts any audio format supported by ffmpeg (webm, wav, mp3, ogg, flac, m4a).
    Returns the transcript text, detected language, and processing duration.
    """
    start = time.perf_counter()

    # Determine file extension from content-type or filename
    fname = audio.filename or "audio.webm"
    ext = os.path.splitext(fname)[-1].lower() or ".webm"

    # Write uploaded bytes to a temp file
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file received.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = _get_whisper_model()
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language=None,          # auto-detect language
            vad_filter=True,        # filter silence / non-speech
            vad_parameters={"min_silence_duration_ms": 300},
        )
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        duration_s = round(time.perf_counter() - start, 2)

        logger.info(
            "Transcribed %.1fs audio | lang=%s | %d chars in %.2fs",
            info.duration,
            info.language,
            len(transcript),
            duration_s,
        )
        return TranscribeResponse(
            transcript=transcript or "(no speech detected)",
            language=info.language,
            duration_s=duration_s,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    from config import settings

    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=True)
