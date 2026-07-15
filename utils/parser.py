"""
utils/parser.py
----------------
Low-level helpers for extracting raw text from uploaded resume PDFs and
validating job description input. Kept separate from the LLM-based
Resume Analyst agent so that parsing failures (corrupt PDF, empty file,
etc.) can be handled deterministically before any LLM call is made.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from pypdf import PdfReader

from config import settings


class PDFParsingError(Exception):
    """Raised when a resume PDF cannot be read or contains no extractable text."""


class MissingInputError(Exception):
    """Raised when a required input (resume or job description) is missing."""


@dataclass
class ParsedPDFResult:
    text: str
    num_pages: int
    filename: str


def validate_pdf_size(file_bytes: bytes) -> None:
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_pdf_size_mb:
        raise PDFParsingError(
            f"PDF file is {size_mb:.1f}MB which exceeds the "
            f"{settings.max_pdf_size_mb}MB limit."
        )


def extract_text_from_pdf(file_bytes: bytes, filename: str = "resume.pdf") -> ParsedPDFResult:
    """Extract raw text from a PDF's bytes.

    Raises PDFParsingError on corrupt files or when no text could be
    extracted at all (e.g. a purely scanned/image-only resume).
    """
    if not file_bytes:
        raise MissingInputError("No resume file was provided.")

    validate_pdf_size(file_bytes)

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001 - surface as domain error
        raise PDFParsingError(f"Could not open PDF: {exc}") from exc

    if len(reader.pages) == 0:
        raise PDFParsingError("The PDF has no pages.")

    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - skip unreadable page, don't kill whole doc
            continue

    text = "\n".join(chunks).strip()
    if not text:
        raise PDFParsingError(
            "No extractable text found in the PDF. It may be a scanned "
            "image without OCR text -- please upload a text-based PDF."
        )

    return ParsedPDFResult(text=text, num_pages=len(reader.pages), filename=filename)


def validate_job_description(jd_text: Optional[str]) -> str:
    if not jd_text or not jd_text.strip():
        raise MissingInputError("Job description is required and cannot be empty.")
    if len(jd_text.strip()) < 30:
        raise MissingInputError(
            "Job description looks too short to be meaningful. Please provide more detail."
        )
    return jd_text.strip()
