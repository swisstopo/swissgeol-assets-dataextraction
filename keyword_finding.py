from utils import TextWord, TextLine
import regex
from typing import Union
import fitz



def find_keyword(word: TextWord, keywords: list[str]) -> TextWord:
    for keyword in keywords:
        pattern = regex.compile(r"(\b" + regex.escape(keyword) + r"\b)", flags=regex.IGNORECASE)
        match = pattern.search(word.text)
        if match:
            return match.group(1)
    return None

def find_keywords_in_lines(text_lines: list[TextLine],keywords : list[str]):
    found_keywords =[]

    for line in text_lines:
        for word in line.words:
            matched_keyword =find_keyword(word, keywords)
            if matched_keyword:
                found_keywords.append({"key": matched_keyword,
                                       "word":word, 
                                       "line": line})
    
    return found_keywords

def get_keywords_by_language(language, params):
    return params.get("table_of_contents", {}).get(language, [])
    

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


def extract_by_bookmarks(doc: fitz.Document) :
        """Extracts TOC using PDF bookmarks."""
        toc = doc.get_toc()
        if toc:
            entries = [{"heading": title, "page": page, "level": level} for level, title, page in toc]
            return TOC(entries=entries)
        return None