import regex

from .text import TextWord, TextLine


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