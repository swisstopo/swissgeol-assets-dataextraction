import fitz
import regex
from typing import Union
from text import TextLine

from keyword_finding import find_keywords_in_lines


class TOC:
    def __init__(self, entries: list[dict[str, Union[str, int]]]):
        """
        Initializes a TOC instance.

        :param entries: A list of TOC entries. Each entry is a dictionary with keys:
            - "heading" (str): The title of the TOC entry.
            - "page" (int): The page number where the heading points.
            - Optional: Additional metadata (e.g., level, section hierarchy).
        """
        self.entries = entries

    def __repr__(self):
        return f"TOC(entries={self.entries})"

    def to_dict(self) -> dict:
        """Converts the TOC to a dictionary format for JSON serialization."""
        return {"entries": self.entries}



class TOCExtractor:
    def __init__(self, config):
        """
        Initializes the TOCExtractor with patterns for pattern-based TOC extraction.

        Args:
            config_path (str): Path to the YAML file containing the configurations.
        """
        self.table_of_contents = config["table_of_contents"]
        self.toc_patterns = [
            {
                "pattern": regex.compile(
                    r"^(\d+(\.\d+)*)\s+([\p{L}\p{M}\p{N}\p{P}\-\s]+)\s*(\.+|\s{2,})\s*([ivxlcdm]+|\d+)$",
                    flags=regex.VERBOSE | regex.IGNORECASE,
                ),
                "header_group": 3,
                "page_group": 5,
            },
            {
                "pattern": regex.compile(
                    r"^([\p{L}\p{M}\p{N}\p{P}\-\s]+)\s*(\.+|\s{2,})\s*([ivxlcdm]+|\d+)$",
                    flags=regex.VERBOSE | regex.IGNORECASE,
                ),
                "header_group": 1,
                "page_group": 3,
            },
            {
                "pattern": regex.compile(
                    r"^(\d+(\.\d+)*)\s+([\p{L}\p{M}\p{N}\p{P}\-\s]+)\s+([ivxlcdm]+|\d+)$",
                    flags=regex.VERBOSE | regex.IGNORECASE,
                ),
                "header_group": 3,
                "page_group": 4,
            },
            {
                "pattern": regex.compile(
                    r"^([\p{L}\p{M}\p{N}\p{P}\-\s]+)\s+([ivxlcdm]+|\d+)$",
                    flags=regex.VERBOSE | regex.IGNORECASE,
                ),
                "header_group": 1,
                "page_group": 2,
            },
        ]
    
    def extract_toc(self, doc: fitz.Document, text_lines: list["TextLine"], language: str) -> TOC | None:
        """
        Extracts the TOC using three approaches: bookmarks, keywords, or patterns.

        Args:
            text_lines (list[TextLine]): Text lines extracted from the document.
            language (str): Detected language of the document.

        Returns:
            TOC | None: The extracted TOC or None if not found.
        """
        toc = self.extract_by_bookmarks(doc)
        if toc:
            return toc
        # Try extracting by keywords
        keywords = self.table_of_contents.get(language, [])
        found_keywords = find_keywords_in_lines(text_lines, keywords)
        if found_keywords:
            toc = self.extract_by_keywords(text_lines, found_keywords)
            if toc:
                return toc
        
        print("No table of contents found")
        return None

    @staticmethod
    def extract_by_bookmarks(doc: fitz.Document) -> TOC | None:
        """Extracts TOC using PDF bookmarks."""
        toc = doc.get_toc()
        if toc:
            entries = [{"header": title, "page": page} for _,title, page in toc]
            return TOC(entries=entries)
        return None
    
    def extract_by_keywords(self, text_lines: list[TextLine], found_keywords: list[dict]) -> TOC | None:
        """Extracts TOC using found keywords and subsequent matching."""
        toc_entries = []

        for keyword_entry in found_keywords:
            line_index = text_lines.index(keyword_entry["line"])
            break  # Only first keyword for now

        # Process subsequent lines starting from  keyword
        for line in text_lines[line_index + 1:]:
            line_matched = False

            for entry in self.toc_patterns:
                toc_pattern = entry["pattern"]
                match = toc_pattern.match(line.line_text())
                if match:
                    if entry["header_group"] > len(match.groups()) or entry["page_group"] > len(match.groups()):
                        print(f"Invalid group index for pattern: {entry}")
                        continue
                    header = match.group(entry["header_group"]).strip()
                    page = match.group(entry["page_group"]).strip()
                    cleaned_header = clean_header(header)
                    toc_entries.append({"header": cleaned_header, "page": page})
                    line_matched = True
                    break

            if not line_matched:
                print(f"First unmatched line: {line.line_text()}")
                break

        return TOC(entries=toc_entries) if toc_entries else None
    
def clean_header(header: str) -> str:
    return regex.sub(r"\.{2,}", "", header).strip()
