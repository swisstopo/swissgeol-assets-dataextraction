import pymupdf
import logging

from .text import TextLine
from .utils import cluster_text_elements
logger = logging.getLogger(__name__)

class MaterialDescription:
    "Stores information about material description block:"

    def __init__(self, text_lines: list[TextLine]):
        self.text_lines =text_lines
        self.rect = self.compute_bbox()

    def __repr__(self):
        return f"MaterialDescription( lines = {self.text_lines.line_text()}, rect = {self.rect} )"

    def compute_bbox(self):
        """Calculate the bounding box of the material description."""
        start_line = min(self.text_lines, key=lambda line: line.rect.y0)
        end_line = max(self.text_lines, key=lambda line: line.rect.y1)

        return pymupdf.Rect(start_line.rect.x0, start_line.rect.y0, end_line.rect.x1, end_line.rect.y1)

def is_description(line: TextLine, material_description: dict):
    """Check if the line is a material description."""
    line_text = line.line_text().lower()
    return any(
        line_text.find(word) > -1 for word in material_description["including_expressions"]
    ) and not any(line_text.find(word) > -1 for word in material_description["excluding_expressions"])

def detect_material_description(lines: list[TextLine], material_description: dict) -> list[MaterialDescription]:
    """Detects  material descriptions in Textlines and returns List of MateriaDescriptions."""
    material_lines = [
            line
            for line in lines
            if is_description(line, material_description)
        ]

     ## TODO:  c) check if valid material description box
     ## TODO: not a lot of noise lines, should cover a certain area of the page...  maye not robust

    line_clusters = cluster_text_elements(material_lines, key = "x0") ##cluster based on x0
    
    filtered_cluster = [cluster for cluster in line_clusters if len(cluster) > 2] # valid description box should have at least 3 lines
    
    if not filtered_cluster:
        return None

    descriptions = [MaterialDescription(cluster) for cluster in filtered_cluster]
    
    
    return descriptions