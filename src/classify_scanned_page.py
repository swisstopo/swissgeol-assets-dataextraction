import os
import pymupdf
import numpy as np
import pandas as pd
import regex
import logging
import logging
logger = logging.getLogger(__name__)

from .text import extract_words, create_text_lines, create_text_blocks, TextLine
from .utils import TextWord, closest_word_distances, y0_word_cluster
from .keyword_finding import find_keywords_in_lines
from .title_page import title_page_type
from .detect_language import detect_language_of_page


pattern_maps = [
    regex.compile(r"1\s*:\s*[125](25|5)?000+"),
    regex.compile(r"1\s*:\s*[125]((0{1,2})?([',]000)+)")
]

def find_maps_pattern(words: list[TextWord]) -> regex.Match | None:
    return next((match 
                 for pattern in pattern_maps 
                 for word in words 
                 if (match := pattern.search(word.text))), None)


def is_description(line: TextLine, material_description):
    """Check if the line is a material description."""
    line_text = line.line_text().lower()
    return any(
        line_text.find(word) > -1 for word in material_description["including_expressions"]
    ) and not any(line_text.find(word) > -1 for word in material_description["excluding_expressions"])

def detect_material_description(lines: list[TextLine], material_description: dict) -> bool:
    """Detects if the page contains a material description."""
    material_description = [
            line
            for line in lines
            if is_description(line, material_description)
        ]

    ## bbox of material description
    if material_description:
        start_line = min(material_description, key=lambda line: line.rect.y0)
        end_line = max(material_description, key=lambda line: line.rect.y1)
        start = (start_line.rect.x0, start_line.rect.y0)
        end = (end_line.rect.x1, end_line.rect.y1)
        material_description_bbox = pymupdf.Rect(start[0], start[1], end[0], end[1])

        ##TODO: check for lines in bbox not in material description, check valid size of bbox to be a material description
        
    return material_description

def classify_on_keywords(lines: list[str], words: list[TextWord], matching_params: dict, language:str) -> str | None:
    """    Classifies a page based on keywords or patterns defined in matching_params for the given language"""

    if language not in matching_params["material_description"]:
        logging.warning(f"Language '{language}' not supported. Using default german language.")
        language = "de"

    if detect_material_description(lines, matching_params["material_description"].get(language, {})):
        pass
    
    if find_keywords_in_lines(lines, matching_params["boreprofile"].get(language, [])):
        return "Boreprofile"
    if find_maps_pattern(words):
        return "Map"  
    return None


def classify_page(page, page_number, filename, matching_params, language) -> dict:

    text = page.get_text()
    words = extract_words(page, page_number)
    if not words:
        return {"Filename": filename, "Page Number": page_number, "Classification": "Unknown"}

    # Compute word distances and line attributes
    distances = closest_word_distances(words)
    median_distance = np.median(distances) if distances else None
    lines = create_text_lines(page, page_number)
    words_per_line = [len(line.words) for line in lines]
    mean_words_per_line = np.mean(words_per_line) if words_per_line else 0

    # Compute text block attributes
    text_blocks = create_text_blocks(lines)
    block_area = sum(block.rect.get_area() for block in text_blocks)
    word_area = sum(word.rect.get_area()
                    for block in text_blocks
                    for line in block.lines
                    for word in line.words if len(line.words) > 1)

    classification = "Unknown"

    # Rule-based classification
    if block_area > 0 and word_area / block_area > 1 and mean_words_per_line > 3:
        classification = "Title Page" if title_page_type(text) else "Text"
    else:
        classify_keywords = classify_on_keywords(lines, words, matching_params, language)
        if classify_keywords:
            classification = classify_keywords
        else:
            clusters = y0_word_cluster(lines)
            filtered_clusters = [cluster for cluster in clusters if len(cluster) > 1]
            longest_cluster = max(map(len, filtered_clusters), default=0)

            if median_distance is not None and median_distance < 20 and longest_cluster > 4:
                classification = "Boreprofile"
            else:
                classification = "Map"

    return {"Filename": filename,
            "Page Number": page_number,
            "Classification": classification}

def classify_pdf(file_path, matching_params)-> pd.DataFrame:
    """Processes a pdf File, classfies each page"""
    classification_data = []

    if not os.path.isfile(file_path) or not file_path.lower().endswith('.pdf'):
        logging.error(f"Invalid file path: {file_path}. Must be a valid PDF file.")
        return pd.DataFrame()
    
    filename = os.path.basename(file_path)
    with pymupdf.Document(file_path) as doc:
        for page_number, page in enumerate(doc, start = 1):
            
            ##detect language
            language = detect_language_of_page(page)

            ##classify page
            page_classification = classify_page(page,
                                                page_number,
                                                filename,
                                                matching_params, 
                                                language)   

            ##update classification count
            classification_data.append(page_classification)
  
    return classification_data