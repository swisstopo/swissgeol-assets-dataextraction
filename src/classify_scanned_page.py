import pymupdf
import numpy as np
import regex
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from .text import extract_words, create_text_lines, create_text_blocks, TextLine, TextWord, TextBlock
from .utils import cluster_text_elements, is_description
from .title_page import sparse_title_page
from .detect_language import detect_language_of_page
from .material_description import detect_material_description
from .bounding_box import merge_bounding_boxes

pattern_maps = [
    regex.compile(r"1\s*:\s*[125](25|5)?000+"),
    regex.compile(r"1\s*:\s*[125]((0{1,2})?([',]000)+)")
]

def find_map_scales(line: TextLine) -> regex.Match | None:
    return next((match 
                 for pattern in pattern_maps 
                 for word in line.words
                 if (match := pattern.search(word.text))), None)

def identify_boreprofile(lines: list[TextLine], words: list[TextWord], matching_params: dict, language:str, page_rect: pymupdf.Rect) -> bool:
    """Identifies whether a page contains a boreprofile based on presence of  a valid material description in given language"""
    material_descriptions = detect_material_description(lines, words, matching_params.get(language, {}))

    if material_descriptions:
        for description in material_descriptions:

            if description.is_valid(page_rect):
                return True 
    
    return False

def identify_map(lines: list[TextLine], text_blocks: list[TextBlock],matching_params, language) -> bool:
    """Identifies whether a page contains a map based on structure and keyword patterns."""
    info_lines = [
        line for line in lines
        if is_description(line, matching_params.get(language, {})) or find_map_scales(line)
    ]

    small_blocks = [text_block for text_block in text_blocks if len(text_block.lines) <= 3]
    filtered_lines = [
        line for block in small_blocks
        for line in block.lines
        if len(line.words) < 4 and line not in info_lines
    ]

    if filtered_lines and (len(filtered_lines)/len(lines) ) > 0.5:

        clusters = cluster_text_elements(filtered_lines, key_fn= lambda line:line.rect.x0)
        potential_scales = [cluster for cluster in clusters if len(cluster) > 3] #scales or legends
        map_clusters = list(filter(lambda cluster: cluster not in potential_scales, clusters))

        if map_clusters:
            filtered_words = [word
                              for lines in map_clusters
                              for line in lines
                              for word in line.words]

            if len(filtered_words) < 7 and not info_lines:
                return False

            def _is_a_number(string: str)-> bool:
                try:
                    float(string)
                    return True
                except ValueError:
                    return False

            map_like_words = [word for word in filtered_words
                if (word.text.isalpha() and word.text.istitle()
                    or word.text.isupper()
                    or _is_a_number(word.text))]

            if map_like_words:
                ratio = (len(map_like_words)) / len(filtered_words)
                threshold = 0.6 if info_lines else 0.75

                if ratio > threshold:
                    return True

    return False

def classify_page(page, page_number, matching_params, language) -> dict:

    classification = {
        "Page": page_number,
        "Boreprofile": 0,
        "Maps": 0,
        "Text": 0,
        "Title_Page": 0,
        "Unknown": 0
    }

    words = extract_words(page, page_number)
    if not words:
        classification["Unknown"] = 1
        return classification

    # Compute word distances and line attributes
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

    page_text_rect = merge_bounding_boxes([line.rect for line in lines]) if lines else page.rect
    # Rule-based classification
    if block_area > 0 and word_area / block_area > 1 and mean_words_per_line > 3:
            classification["Text"] = 1

    elif identify_boreprofile(lines, words, matching_params["material_description"], language, page_text_rect): ## TODO: Ensure the boreprofile check is independent of what happens above (if and not elif?)
        classification["Boreprofile"] = 1                                                        # should require text sparsity as a necessary condition.

    elif identify_map(lines, text_blocks, matching_params["map_terms"], language):
        classification["Maps"] = 1

    elif sparse_title_page(lines):
        classification["Title_Page"] = 1
        logger.info(f" title page on page: {page_number}")

    else:
        classification["Unknown"] = 1
    logger.info(classification)
    return classification

def classify_pdf(file_path: Path, matching_params)-> dict:
    """Processes a pdf File, classifies each page"""

    if not file_path.is_file() or file_path.suffix.lower() != '.pdf':
        logging.error(f"Invalid file path: {file_path}. Must be a valid PDF file.")
        return {}

    classification = []

    with pymupdf.Document(file_path) as doc:
        for page_number, page in enumerate(doc, start = 1):
            
            language = detect_language_of_page(page)

            if language not in matching_params["material_description"]:
                logging.warning(f"Language '{language}' not supported. Using default german language.")
                language = "de"

            page_classification = classify_page(page,
                                                page_number,
                                                matching_params,
                                                language)   
            classification.append(page_classification)
  
    return {"filename": file_path.name,
            "classification": classification}