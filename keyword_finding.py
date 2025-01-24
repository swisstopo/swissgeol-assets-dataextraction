from utils import TextWord
import regex
from typing import Union


def find_keywords(word: TextWord, keywords: list[str]) -> TextWord:
    for keyword in keywords:
        pattern = regex.compile(r"(\b" + regex.escape(keyword) + r"\b)", flags=regex.IGNORECASE)
        match = pattern.search(word.text)
        if match:
            return word
    return None

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
