import pymupdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian
import os
import math
from collections import defaultdict
from text import TextWord


def is_digitally_born(page: pymupdf.Page) -> bool:
    bboxes = page.get_bboxlog()

    for boxType, rectangle in bboxes:
        if (boxType == "fill-text" or boxType == "stroke-text") and not pymupdf.Rect(rectangle).is_empty:
            return True
    return False


def classify_text_density(words, page_size):
    if not words:
        return {
            "classification": "No text",
            "text_density": 0,
            "text_area": 0,
            "avg_word_height": 0,
            "std_word_height": 0
        }

    page_area = page_size[0] * page_size[1]
    text_density = len(words) / page_area

    text_area = sum(word.rect.width * word.rect.height for word in words) / page_area

    word_heights = [word.rect.height for word in words]
    avg_word_height = float(np.mean(word_heights))
    std_word_height = float(np.std(word_heights))

    density_threshold = 0.0001
    height_threshold = page_size[1] * 0.02


    return {
        "text_density": text_density,
        "text_area": text_area,
        "avg_word_height": avg_word_height,
        "std_word_height": std_word_height
    }

def classify_wordpos(words: list[TextWord]):
    """Classifies text structure on page based on distribution."""
    
    if not words:
        print( "Unknown")
        return

    # Extract Y-axis positions and widths
    y_positions = np.array([word.rect.y0 for word in words])
    x_positions = np.array([word.rect.x0 for word in words])
    widths = np.array([word.rect.x1 - word.rect.x0 for word in words])
    heights = np.array([word.rect.y1 - word.rect.y0 for word in words])

    # plt.hist2d(x_positions, y_positions, bins=(20, 20), cmap='Blues')
    # plt.xlabel("X Position")
    # plt.ylabel("Y Position")
    # plt.gca().invert_yaxis()
    # plt.title("Text Bounding Box Distribution")
    # plt.colorbar(label="Frequency")
    # plt.show()

    # Compute pairwise Euclidean distances
    dist_matrix = squareform(pdist(y_positions.reshape(-1, 1))) #instead use boundingbox?
    threshold = np.percentile(dist_matrix, 20) 
    graph_matrix = (dist_matrix < threshold).astype(int)
    lap_matrix = laplacian(graph_matrix, normed=True)
   
     
    # Compute spacing bewtween word to next word
    y_spacing = np.diff(np.sort(y_positions)) 
    x_spacing = np.diff(np.sort(x_positions))
    
    mean_y_spacing = float(np.mean(y_spacing)) if len(y_spacing) > 0 else 0
    median_x_spacing = float(np.median(x_spacing)) if len(x_spacing) > 0 else 0
    width_std = np.std(widths)
    height_std = np.std(heights)
    

    return {
        "mean_y_spacing": mean_y_spacing,
        "median_x_spacing": median_x_spacing,
        "median width": float(np.median(widths)),
        "width_std": float(width_std),
        "height_std":float(height_std)
    }
 
def calculate_distance(word1, word2):
    """Calculate Euclidean distance between two TextWord objects based on x0 and y0"""
    x_dist = word1.rect.x0 - word2.rect.x0
    y_dist = word1.rect.y0 - word2.rect.y0
    return math.sqrt(x_dist**2 + y_dist**2)

def closest_word_distances(words):
    """Calculate distances between each word and its closest neighbor"""
    if not words or len(words) < 2:
        return []

    distances = []
    for i, word in enumerate(words):
        other_words = words[:i] + words[i+1:]  # Exclude current word
        closest_word = min(other_words, key=lambda w: calculate_distance(word, w))
        distances.append(calculate_distance(word, closest_word))

    return distances


def process_documents(input_path, function):
    """ Retrieves text from input file or folder and executes function (has to take doc as input and return a dictionary)"""
    
    results = {}
    
    if os.path.isfile(input_path):
        with pymupdf.Document(input_path) as doc:

            results[os.path.basename(input_path)] = function(doc)

    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):

            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(input_path, filename)
                with pymupdf.Document(file_path) as doc:
                    results[filename] = function(doc)
    else:
        print(f"Input path is invalid: {input_path}")
    
    return results

def y0_word_cluster(all_words, tolerance: int = 10):
   
    if not all_words:
        return []

    # Dictionary to hold clusters, keys are representative y0 values
    grouped_y0 = defaultdict(list)

    for word in all_words:
        y0 = word.rect.y0
        matched_y0 = None

        # Check if y0 is within tolerance of an existing cluster
        for key in grouped_y0:
            if abs(key - y0) <= tolerance:
                matched_y0 = key
                break

        # Add to an existing cluster or create a new one
        if matched_y0 is not None:
            grouped_y0[matched_y0].append(word)
        else:
            grouped_y0[y0].append(word)

    clusters = list(grouped_y0.values())

    return clusters