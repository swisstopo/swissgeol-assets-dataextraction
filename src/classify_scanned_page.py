import os
import pymupdf
import numpy as np
import regex
import logging
logger = logging.getLogger(__name__)

from .text import extract_words, create_text_lines, create_text_blocks, TextLine
from .utils import TextWord, closest_word_distances, cluster_text_elements, classify_text_density, classify_wordpos
from .title_page import title_page_type,  sparse_title_page
from .detect_language import detect_language_of_page
from .material_description import detect_material_description

pattern_maps = [
    regex.compile(r"1\s*:\s*[125](25|5)?000+"),
    regex.compile(r"1\s*:\s*[125]((0{1,2})?([',]000)+)")
]

def find_maps_pattern(words: list[TextWord]) -> regex.Match | None:
    return next((match 
                 for pattern in pattern_maps 
                 for word in words 
                 if (match := pattern.search(word.text))), None)


def identify_boreprofile(lines: list[TextLine], words: list[TextWord], matching_params: dict, language:str) -> bool:
    """Identifies whether a page conatins a boreprofile based on presence of  a valid material description in given language"""

    if language not in matching_params["material_description"]:
        logging.warning(f"Language '{language}' not supported. Using default german language.")
        language = "de"

    material_descriptions = detect_material_description(lines, words, matching_params["material_description"].get(language, {}))

    if material_descriptions:
        for description in material_descriptions:
            if description.noise < 1.75:
                logger.info(
                    "Detected boreprofile")
                return True 
    
    return False


def identify_map(lines: list[TextLine],words: list[TextWord], median_distance: float, page_size) -> bool: ## refine this!!
    """Identifies whether a page conatins a map based on scale pattern."""
    
    if find_maps_pattern(words):  # too unspecific
        logger.info("Map detected based on pattern")
        return True  
    
    # Classify text density and distribution
    text_metrics = classify_text_density(words, page_size)
    structure_metrics = classify_wordpos(words)

    # Text density: Low-density pages are more likely to contain maps.
    text_is_sparse = text_metrics["text_density"] < 0.00005  
    text_area_is_small = text_metrics["text_area"] < 0.1 

    # Word positioning: Check if text is scattered rather than structured
    irregular_text_structure = (
        structure_metrics["mean_y_spacing"] > page_size[1] * 0.03 and 
        structure_metrics["width_std"] > page_size[0] * 0.1 
    )

    clusters = cluster_text_elements(lines, key = "y0") 
    filtered_clusters = [cluster for cluster in clusters if len(cluster) > 1]
    longest_cluster = max(map(len, filtered_clusters), default=0)

    if (
        (median_distance is not None and median_distance >= 20 and longest_cluster <= 4)
        and (text_is_sparse and irregular_text_structure)
        or text_area_is_small
    ):
        logger.info("Map detected based on clustering and text structure.")
        return True
    return False  


def classify_page(page, page_number, matching_params, language) -> dict: ##inclusion based instead! 1. identify as borehole, 2. map 3. title page 4. rest text
    
    page_size = (page.rect.width, page.rect.height)
    text = page.get_text()
    words = extract_words(page, page_number)
    classification = {
        "Page": page_number,
        "Boreprofile": 0,
        "Maps": 0,
        "Text": 0,
        "Title_Page": 0,
        "Unknown": 0
    }

    if not words:
        classification["Unknown"] = 1
        return classification

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

    # Rule-based classification
    if block_area > 0 and word_area / block_area > 1 and mean_words_per_line > 3:
        if title_page_type(text):
            classification["Title_Page"] = 1
        else:
            classification["Text"] = 1

    elif identify_boreprofile(lines, words, matching_params, language):
        classification["Boreprofile"] = 1

    elif identify_map(lines, words, median_distance, page_size):
        classification["Maps"] = 1

    elif sparse_title_page(lines):
        classification["Title_Page"] = 1
        logger.info(f"possible title page {page_number}")

    else:
        classification["Unknown"] = 1

    return classification

def classify_pdf(file_path, matching_params)-> dict:
    """Processes a pdf File, classfies each page"""
    classification = []

    if not os.path.isfile(file_path) or not file_path.lower().endswith('.pdf'):
        logging.error(f"Invalid file path: {file_path}. Must be a valid PDF file.")
        return []
    
    filename = os.path.basename(file_path)
    with pymupdf.Document(file_path) as doc:
        for page_number, page in enumerate(doc, start = 1):
            
            language = detect_language_of_page(page)

            page_classification = classify_page(page,
                                                page_number,
                                                matching_params,
                                                language)   
            classification.append(page_classification)
  
    return {"filename": filename,
            "classification": classification}