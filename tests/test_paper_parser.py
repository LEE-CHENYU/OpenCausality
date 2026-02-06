import pytest
import os
from src.data.paper_parser import PaperParser

@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    # Create minimal PDF (or use real sample)
    with open(pdf_path, 'w') as f:
        f.write("Dummy PDF content with Abstract: This is a test.")
    return str(pdf_path)

def test_parse_pdf(sample_pdf):
    parser = PaperParser()
    parsed = parser.parse_pdf(sample_pdf)
    assert "abstract" in parsed
    assert "full_text" in parsed
    assert "test" in parsed["full_text"].lower()