import pytest
from src.data.annas_archive import AnnasArchiveFetcher
import os
import tempfile

def test_search_annas_archive():
    fetcher = AnnasArchiveFetcher()
    results = fetcher.search("machine learning", limit=1)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "title" in results[0]

def test_download_paper(mock_download_url):
    fetcher = AnnasArchiveFetcher()
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, "test.pdf")
        fetcher.download(mock_download_url, save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

# Fixture for mock URL (replace with a real test URL if needed)
@pytest.fixture
def mock_download_url():
    return "https://www.africau.edu/images/default/sample.pdf"  # Small public PDF for testing
