import os
import pymupdf
import numpy as np
import pandas as pd
import regex
import logging
from collections import defaultdict
from tabulate import tabulate

from text import extract_words, create_text_lines, create_text_blocks
from utils import TextWord, closest_word_distances, y0_word_cluster
from keyword_finding import find_keywords_in_lines
from title_page import title_page_type

keywords_boreprofile = ["bohrung", "bohrprofil", "sondage"]

pattern_maps = [
    regex.compile(r"1\s*:\s*[125](25|5)?000+"),
    regex.compile(r"1\s*:\s*[125]((0{1,2})?([',]000)+)")
]

def find_maps_pattern(words: list[TextWord]) -> regex.Match | None:
    return next((match 
                 for pattern in pattern_maps 
                 for word in words 
                 if (match := pattern.search(word.text))), None)


def classify_on_keywords(lines: list[str], words: list[TextWord]) -> str | None:

    if find_keywords_in_lines(lines, keywords_boreprofile):
        return "Boreprofile"
    if find_maps_pattern(words):
        return "Map"  
    return None

def classify_page(page, page_number, filename) -> dict:
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
        classify_keywords = classify_on_keywords(lines, words)
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

    return {"Filename": filename, "Page Number": page_number, "Classification": classification}

def classify_pdf(pdf_path: str)-> pd.DataFrame:
    """Processes a pdf File, classfies each page"""
    classification_counts = defaultdict(int)
    classification_data = []
    total_pages = 0

    for filename in os.listdir(pdf_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(pdf_path, filename)

            with pymupdf.Document(file_path) as doc:
                for page_index, page in enumerate(doc):
                    total_pages += 1
                    page_number = page_index + 1

                    ##classify page
                    page_classification = classify_page(page, page_number, filename)

                    ##update classification count
                    classification_data.append(page_classification)
                    classification_counts[page_classification["Classification"]] += 1
                    

    df = pd.DataFrame(classification_data)

    # classification summary
    summary = pd.DataFrame.from_dict(classification_counts, orient='index', columns=['Count'])
    summary['Percentage'] = (summary['Count'] / total_pages * 100).round(2)

    logging.info("Classification Summary:")
    logging.info(tabulate(summary, headers="keys", tablefmt="grid"))

    return df