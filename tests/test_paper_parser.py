"""Tests for PaperParser PDF extraction."""

import pytest
from src.data.paper_parser import PaperParser


@pytest.fixture
def parser():
    return PaperParser()


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a plain text file with .pdf extension (tests fallback path)."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("Dummy PDF content with Abstract: This is a test.")
    return str(pdf_path)


@pytest.fixture
def sample_real_pdf(tmp_path):
    """Create a minimal valid PDF using pymupdf if available."""
    try:
        import fitz

        pdf_path = tmp_path / "real.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Abstract: Oil shocks cause welfare losses.")
        page.insert_text((72, 100), "This paper studies causal effects.")
        doc.save(str(pdf_path))
        doc.close()
        return str(pdf_path)
    except ImportError:
        pytest.skip("pymupdf not installed")


def test_parse_pdf(sample_text_file, parser):
    """Test parsing a non-PDF file falls back to plain text."""
    parsed = parser.parse_pdf(sample_text_file)
    assert "abstract" in parsed
    assert "full_text" in parsed
    assert "test" in parsed["full_text"].lower()


def test_parse_real_pdf(sample_real_pdf, parser):
    """Test parsing a valid PDF file."""
    parsed = parser.parse_pdf(sample_real_pdf)
    assert "full_text" in parsed
    assert "oil shocks" in parsed["full_text"].lower()
    assert "abstract" in parsed
    assert parsed["abstract"] != ""


def test_parse_nonexistent_file(tmp_path, parser):
    """Test parsing a file that doesn't exist returns empty."""
    parsed = parser.parse_pdf(str(tmp_path / "nonexistent.pdf"))
    assert parsed["full_text"] == ""
    assert parsed["abstract"] == ""
