"""Very similar to code in swissgeol-boreholes-dataextraction repo https://github.com/swisstopo/swissgeol-boreholes-dataextraction
The same including and excluding matching parameters used."""
import pymupdf
import logging

from .text import TextLine, TextWord
from .utils import cluster_text_elements
logger = logging.getLogger(__name__)

class MaterialDescription:
    "Stores information about material description block."

    def __init__(self, text_lines: list[TextLine], all_words: list[TextWord]): ## maybe possible only use words or lines here...
        self.text_lines =text_lines
        self.rect = self.compute_bbox()
        self.noise = self.compute_noise(all_words)

    def __repr__(self):
        return f"MaterialDescription( lines = {[ line.line_text() for line in self.text_lines]}, rect = {self.rect}, , noise={self.noise} )"

    def compute_bbox(self):
        """Calculate the bounding box of the material description."""
        start_line = min(self.text_lines, key=lambda line: line.rect.y0)
        end_line = max(self.text_lines, key=lambda line: line.rect.y1)

        return pymupdf.Rect(start_line.rect.x0, start_line.rect.y0, end_line.rect.x1, end_line.rect.y1)
    

    def compute_noise(self,all_words:list[TextWord]):
        """For bounding box compute noise of words not containing to bbox entries"""
        description_words = [word for line in self.text_lines for word in line.words]
        noise_words= [word for word in all_words 
                     if self.rect.contains(word.rect) and word not in description_words]

        return len(noise_words)/len(description_words)


def is_description(line: TextLine, material_description: dict):
    """Check if the line is a material description."""
    line_text = line.line_text().lower()
    return any(
        line_text.find(word) > -1 for word in material_description["including_expressions"]
    ) and not any(line_text.find(word) > -1 for word in material_description["excluding_expressions"])



def detect_material_description(lines: list[TextLine], words:list[TextWord], material_description: dict) -> list[MaterialDescription]:
    """Detects material descriptions in Textlines and returns List of MaterialDescriptions."""
    material_lines = [
            line
            for line in lines
            if is_description(line, material_description)
        ]

    line_clusters = cluster_text_elements(material_lines, key_fn = lambda line: line.rect.x0) ##cluster based on x0
    
    filtered_cluster = [cluster for cluster in line_clusters if len(cluster) > 2] # valid description box should have at least 3 lines
    
    if not filtered_cluster:
        return None

    descriptions = [MaterialDescription(cluster, words) for cluster in filtered_cluster]
    
    return descriptions