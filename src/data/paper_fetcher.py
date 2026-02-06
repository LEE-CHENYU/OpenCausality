"""
Central paper fetching component that treats multiple sources equally.
"""

from typing import List, Dict, Optional
import random  # For equal priority random selection

from .base import DataSource
from .annas_archive import AnnasArchiveFetcher

class PaperFetcher:
    def __init__(self):
        # List of available fetchers (equal priority)
        self.fetchers = [
            AnnasArchiveFetcher(),
            # Add other fetchers here, e.g., ArxivFetcher(), SciHubFetcher(), etc.
            # For now, only Anna's as per user request
        ]

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search across all fetchers equally."""
        all_results = []
        for fetcher in self.fetchers:
            try:
                results = fetcher.search(query, limit)
                all_results.extend(results)
            except Exception as e:
                print(f"Error with {fetcher.__class__.__name__}: {e}")
        
        # Shuffle to treat equally (random order)
        random.shuffle(all_results)
        return all_results[:limit]  # Return top N

    def download(self, download_info: Dict, save_path: str) -> bool:
        """Download using the source's method."""
        # Assuming download_info has 'source' and 'url'
        source_name = download_info.get('source')
        fetcher = next((f for f in self.fetchers if f.source_name == source_name), None)
        if fetcher:
            fetcher.download(download_info['url'], save_path)
            return True
        return False

# Example usage
if __name__ == "__main__":
    fetcher = PaperFetcher()
    query = input("Enter search query: ")
    results = fetcher.search(query)
    if results:
        # Add source info to results
        for res in results:
            res['source'] = "AnnasArchive"  # Adjust based on actual fetcher
        first_result = results[0]
        fetcher.download({'url': first_result.get('download_urls', [])[0], 'source': first_result['source']}, "downloads/paper.pdf")
