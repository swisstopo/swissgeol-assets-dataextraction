import fitz
import os

def text_from_document(file_path) -> dict:
    """ Retrieve text per page from a single pdf file
    Returns dictionary with pagenumber as key and all text on that page as item"""
    doc = fitz.open(file_path)

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
        results[os.path.basename(input_path)] = text_from_document(input_path)

    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(input_path, filename)
                results[filename] = text_from_document(file_path)
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
