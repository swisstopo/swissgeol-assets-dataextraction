import fitz
import os
from collections import defaultdict

def text_from_document(doc) -> dict:
    """ Retrieve text per page from a single pdf file
    Returns dictionary with pagenumber as key and all text on that page as item"""

    page_text= {}
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        text = page.get_text()
        
        page_text[page_number]= text
    
    return(page_text)


def process_documents(input_path):
    """ Retrieves text from input file or folder and returns dictionary"""
    
    results = {}
    
    if os.path.isfile(input_path):
        with fitz.Document(input_path) as doc:

            results[os.path.basename(input_path)] = text_from_document(doc)

    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):

            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(input_path, filename)
                with fitz.Document(file_path) as doc:
                    results[filename] = text_from_document(doc)
    else:
        print(f"Input path is invalid: {input_path}")
    
    return results


class TextWord:

    def __init__(self, rect: fitz.Rect, text: str, page: int):
        self.rect = rect
        self.text = text
        self.page_number = page

    def __repr__(self) -> str:
        return f"TextWord({self.rect}, {self.text})"


class TextLine:

    def __init__(self, words: list[TextWord]):
        
        if not words:
            raise ValueError("Cannot create an empty TextLine.")

        self.rect = words[0].rect
        for word in words[1:]:
            self.rect.include_rect(word.rect)
        self.words = words
        self.page_number = words[0].page_number


    
    def __repr__(self) -> str:
        return f"TextLine({self.rect},{self.line_text()})"

    def line_text(self):
        return ' '.join([word.text for word in self.words])
    

def create_text_lines(doc: fitz.Document) -> list[TextLine]:
 
    lines = [] 

    for page_index, page in enumerate(doc):
        page_number = page_index + 1

        words =[]
        words_by_line = defaultdict(list)

        for x0, y0, x1, y1, word, block_no, line_no, _word_no in page.get_text("words"):
            rect = fitz.Rect(x0, y0, x1, y1) * page.rotation_matrix
            text_word = TextWord(rect=rect, text=word, page=page_number)
            words.append(text_word)


            key = f"{block_no}_{line_no}"
            words_by_line[key].append(text_word)

        text_lines = [TextLine(words) for words in words_by_line.values() if words]
        merged_text_lines = merge_text_lines(text_lines)
        lines.extend(merged_text_lines)
    return lines


def merge_text_lines(naive_lines: list[TextLine]) -> list[TextLine]:
    """
    Merges raw lines into logical lines if PyMuPDF splits them unnecessarily.
    """
    merged_lines = []
    current_words = []

    for naive_line in naive_lines:
        for word in naive_line.words:
            if current_words:
                previous_word = current_words[-1]
                if not is_same_line(word, previous_word):
                    merged_lines.append(TextLine(current_words))
                    current_words = []

            current_words.append(word)

    return merged_lines


def is_same_line(previous_word: TextWord, current_word: TextWord) -> bool: ## dont use threshold but maybe height -> how much they intersect
    """
    Determines whether two words belong to the same line based on their y-coordinates.
    """
    return abs(previous_word.rect.y0 - current_word.rect.y0) <= 2.0
