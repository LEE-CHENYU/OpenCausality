import fitz  # PyMuPDF
import json
from typing import Dict

class PaperParser:
    def parse_pdf(self, pdf_path: str) -> Dict:
        doc = fitz.open(pdf_path)
        text = ""
        metadata = doc.metadata or {}
        abstract = self._extract_abstract(doc)
        
        for page in doc:
            text += page.get_text("text") + "\n"

        parsed = {
            "metadata": metadata,
            "abstract": abstract,
            "full_text": text.strip(),
        }
        doc.close()
        return parsed

    def _extract_abstract(self, doc) -> str:
        # Heuristic: Look for "Abstract" section in first few pages
        for page in doc[:5]:  # Check first 5 pages
            text = page.get_text("text").lower()
            if "abstract" in text:
                start = text.find("abstract")
                end = text.find("\n\n", start) or len(text)
                return text[start:end].strip()
        return ""  # Fallback if not found

    def save_parsed(self, parsed_data: Dict, json_path: str):
        with open(json_path, 'w') as f:
            json.dump(parsed_data, f, indent=2)

# Standalone usage (decoupled)
if __name__ == "__main__":
    parser = PaperParser()
    parsed = parser.parse_pdf("downloads/paper.pdf")
    parser.save_parsed(parsed, "downloads/parsed_paper.json")