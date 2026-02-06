import requests
import json
import os

class AnnasArchiveFetcher:
    @property
    def source_name(self):
        return "annas_archive"

    def search(self, query, limit=5):
        base_url = "https://annas-archive.org/api/v1/search"
        params = {
            "q": query,
            "limit": limit,
            "format": "json"
        }
        headers = {
            "User-Agent": "EconometricResearchFetcher/1.0"
        }
        
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code == 200:
            results = response.json()
            return results.get("results", [])
        else:
            print(f"Error: {response.status_code}")
            return []

    def download(self, download_url, save_path):
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded to {save_path}")
        else:
            print(f"Download failed: {response.status_code}")

# Example usage (can be integrated into CLI or pipeline)
if __name__ == "__main__":
    fetcher = AnnasArchiveFetcher()
    query = input("Enter search query: ")
    results = fetcher.search(query)
    if results:
        first_result = results[0]
        download_urls = first_result.get("download_urls", [])
        if download_urls:
            fetcher.download(download_urls[0], "downloads/paper.pdf")
