"""
PDF paper parser with multi-backend extraction.

Tries pymupdf (fitz) first, falls back to pypdf, then plain text.
"""

import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PaperParser:
    def parse_pdf(self, pdf_path: str) -> Dict:
        """Parse a PDF file, extracting metadata, abstract, and full text.

        Tries pymupdf first, then pypdf, then plain-text fallback.
        """
        text = ""
        metadata: Dict = {}

        # Strategy 1: pymupdf (highest quality)
        text, metadata = self._try_pymupdf(pdf_path)

        # Strategy 2: pypdf fallback
        if not text:
            text, metadata = self._try_pypdf(pdf_path)

        # Strategy 3: plain text fallback (for non-PDF text files)
        if not text:
            text = self._try_plain_text(pdf_path)

        abstract = self._extract_abstract_from_text(text)

        return {
            "metadata": metadata,
            "abstract": abstract,
            "full_text": text.strip(),
        }

    def _try_pymupdf(self, pdf_path: str) -> tuple[str, Dict]:
        """Try extracting with pymupdf (fitz)."""
        try:
            import fitz

            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()
            return "\n".join(pages), metadata
        except ImportError:
            logger.debug("pymupdf not available")
            return "", {}
        except Exception as e:
            logger.debug(f"pymupdf failed: {e}")
            return "", {}

    def _try_pypdf(self, pdf_path: str) -> tuple[str, Dict]:
        """Try extracting with pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            metadata = {}
            if reader.metadata:
                metadata = {
                    k.lstrip("/"): v
                    for k, v in (reader.metadata or {}).items()
                    if isinstance(v, str)
                }
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            return "\n".join(pages), metadata
        except ImportError:
            logger.debug("pypdf not available")
            return "", {}
        except Exception as e:
            logger.debug(f"pypdf failed: {e}")
            return "", {}

    def _try_plain_text(self, pdf_path: str) -> str:
        """Last resort: read file as plain text."""
        try:
            with open(pdf_path, "r", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Plain text fallback failed: {e}")
            return ""

    def _extract_abstract_from_text(self, text: str) -> str:
        """Extract abstract section from full text."""
        lower = text.lower()
        if "abstract" not in lower:
            return ""
        start = lower.find("abstract")
        # Look for double newline or next section header
        end = lower.find("\n\n", start)
        if end == -1:
            end = len(text)
        return text[start:end].strip()

    def save_parsed(self, parsed_data: Dict, json_path: str):
        with open(json_path, "w") as f:
            json.dump(parsed_data, f, indent=2)


# Standalone usage (decoupled)
if __name__ == "__main__":
    parser = PaperParser()
    parsed = parser.parse_pdf("downloads/paper.pdf")
    parser.save_parsed(parsed, "downloads/parsed_paper.json")