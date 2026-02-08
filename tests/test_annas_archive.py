import pytest
from src.data.annas_archive import AnnasArchiveFetcher
import os
import tempfile

def test_search_annas_archive():
    fetcher = AnnasArchiveFetcher()
    results = fetcher.search("test", limit=1)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "title" in results[0]

def test_get_download_url():
    fetcher = AnnasArchiveFetcher()
    # Use a known real detail URL from a public result (replace with actual if needed)
    test_detail_url = f"{fetcher.BASE_URL}/md5/af33b573961b7687bd5bfb6acd7171e1"  # From previous Kazakhstan example
    dl_url = fetcher.get_download_url(test_detail_url)
    assert dl_url is not None
    assert dl_url.startswith(fetcher.BASE_URL)

def test_download_paper(tmpdir):
    fetcher = AnnasArchiveFetcher()
    test_dl_url = "https://www.africau.edu/images/default/sample.pdf"  # Real small PDF
    save_path = tmpdir / "test.pdf"
    success = fetcher.download(test_dl_url, str(save_path))
    assert success
    assert os.path.exists(str(save_path))
    assert os.path.getsize(str(save_path)) > 0

# Fixture for mock URL (replace with a real test URL if needed)
@pytest.fixture
def mock_download_url():
    return "https://www.africau.edu/images/default/sample.pdf"  # Small public PDF for testing
