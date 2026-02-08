"""
PDF Text Extraction + LLM Causal Claim Extraction.

Extracts text from academic PDFs and feeds it through the LLM
for richer causal claim extraction than abstract-only parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max chars to send to LLM (context limit safety)
MAX_TEXT_CHARS = 50_000


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file.

    Tries pymupdf (fitz) first, falls back to pypdf.
    Truncates to MAX_TEXT_CHARS.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text (may be truncated).
    """
    text = ""

    # Try pymupdf first (higher quality extraction)
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n\n".join(pages)
    except ImportError:
        logger.debug("pymupdf not available, trying pypdf")
    except Exception as e:
        logger.warning(f"pymupdf extraction failed: {e}, trying pypdf")

    # Fallback to pypdf
    if not text:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "No PDF reader available. Install pymupdf or pypdf: "
                "pip install pymupdf  # or: pip install pypdf"
            )

    # Truncate to limit
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        logger.info(f"PDF text truncated to {MAX_TEXT_CHARS} chars")

    return text


def extract_claims_from_pdf(
    pdf_path: Path,
    client: Any,  # LLMClient
) -> list[dict]:
    """Extract causal claims from a PDF using LLM.

    Args:
        pdf_path: Path to the PDF file.
        client: An LLMClient instance.

    Returns:
        List of extracted causal claim dicts.
    """
    from shared.llm.prompts import CAUSAL_CLAIM_EXTRACTION_SYSTEM

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        logger.warning(f"No text extracted from {pdf_path}")
        return []

    user_prompt = (
        f"Full paper text (may be truncated):\n\n{text}\n\n"
        "Extract all causal claims from this paper."
    )

    schema = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "treatment": {"type": "string"},
                        "outcome": {"type": "string"},
                        "mechanism": {"type": "string"},
                        "direction": {"type": "string"},
                        "identification": {"type": "string"},
                        "confidence": {"type": "string"},
                        "quote": {"type": "string"},
                        "edge_type": {"type": "string"},
                    },
                },
            },
        },
    }

    try:
        result = client.complete_structured(
            CAUSAL_CLAIM_EXTRACTION_SYSTEM, user_prompt, schema,
        )
        claims = result.get("claims", [])
        logger.info(f"Extracted {len(claims)} causal claims from {pdf_path.name}")
        return claims
    except Exception as e:
        logger.warning(f"LLM claim extraction failed for {pdf_path}: {e}")
        return []
