"""This module contains functionalities to detect table like structures."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import pymupdf

from src.geometric_objects import Line
from src.text_objects import TextLine, cluster_connected_components
from src.utils import read_params

logger = logging.getLogger(__name__)

config = read_params("config/table_detection_params.yml")


@dataclass
class TableStructure:
    """Represents a detected table structure."""

    rect: pymupdf.Rect
    horizontal_lines: list[Line]
    vertical_lines: list[Line]
    confidence: float
    line_density: float

    def height_coverage(self, page_height: float) -> float:
        """Fraction of page height covered by structure tables bounding box."""
        return self.rect.height / page_height


def detect_table_structures(
    page_rect: pymupdf.Rect,
    structure_lines: list[StructureLine],
    text_lines: list[TextLine],
) -> list[TableStructure]:
    """Table detection."""
    min_confidence = config.get("min_confidence", 0.6)

    if not structure_lines:
        return []

    components = cluster_connected_components(structure_lines, intersects_or_near)
    candidates = (to_table(c, page_rect, text_lines) for c in components)
    filtered = [t for t in candidates if t and t.confidence >= min_confidence]

    return nms(filtered)


def to_table(comp: list[StructureLine], page_rect: pymupdf.Rect, text_lines: list) -> TableStructure | None:
    min_h, min_v = config.get("min_h_lines", 20), config.get("min_v_lines", 5)

    page_w, page_h = page_rect.width, page_rect.height
    line_weight = config.get("line_scoring", {}).get("line_weights", 0.5)
    max_lines_bonus = config.get("line_scoring", {}).get("max_n_lines_bonus", 30)
    area_weight = config.get("area_scoring", {}).get("area_weights", 0.4)
    min_area_ratio = config.get("area_scoring", {}).get("min_table_area_ratio", 0.3)
    text_weight = config.get("text_scoring", {}).get("text_weights", 0.4)
    per_text_bonus = config.get("text_scoring", {}).get("text_presence_weight", 0.01)

    h_lines = [s.line for s in comp if not s.is_vertical]
    v_lines = [s.line for s in comp if s.is_vertical]
    if len(h_lines) < min_h or len(v_lines) < min_v:
        return None

    all_lines = h_lines + v_lines
    # Calculate bounding box of this region
    xs = [p.x for L in all_lines for p in (L.start, L.end)]
    ys = [p.y for L in all_lines for p in (L.start, L.end)]
    rect = pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))

    area_ratio = max(0.0, (rect.width * rect.height) / (page_w * page_h + 1e-9))
    size_score = area_weight * min(1.0, area_ratio / min_area_ratio)

    total_lines = len(all_lines)
    line_score = line_weight * min(1.0, total_lines / max_lines_bonus)

    if text_lines:
        n_text = sum(1 for tl in text_lines if rect.intersects(tl.rect))
        text_score = text_weight * min(1.0, n_text * per_text_bonus)
    else:
        text_score = 0.0

    confidence = min(1.0, size_score + line_score + text_score)

    area = rect.width * rect.height
    line_density = total_lines / (area / 10_000.0) if area > 0 else 0.0

    return TableStructure(
        rect=rect,
        horizontal_lines=h_lines,
        vertical_lines=v_lines,
        confidence=confidence,
        line_density=line_density,
    )


def intersects_or_near(a: StructureLine, b: StructureLine) -> bool:
    connection_threshold = config.get("connection_threshold", 0.5)
    L1, L2 = a.line, b.line
    if L1.intersects_with(L2):
        return True
    for p in (L1.start, L1.end):
        if L2.point_near_segment(p, connection_threshold):
            return True
    return any(L1.point_near_segment(p, connection_threshold) for p in (L2.start, L2.end))


def detect_structure_lines(geometric_lines: list[Line]) -> list[StructureLine]:
    """Detect significant horizonal and vertical lines in a document."""
    filtered_lines = _filter_significant_lines(geometric_lines, config)
    return _separate_by_orientation(filtered_lines, config)


def _filter_significant_lines(lines: list[Line], config_file: dict) -> list[Line]:
    """Filter to keep only significantly long lines that could form table structures."""
    min_length = config_file.get("min_line_length")
    return [line for line in lines if line.length > min_length]


def _separate_by_orientation(lines: list[Line], config_file: dict) -> list[StructureLine]:
    """Separate lines into horizontal and vertical based on angle and tolerance."""
    angle_tolerance = config_file.get("angle_tolerance")
    structure_lines = []

    for line in lines:
        angle = abs(line.line_angle)

        # Horizontal lines (close to 0° or 180°)
        if angle <= angle_tolerance or angle >= (180 - angle_tolerance):
            structure_lines.append(
                StructureLine(
                    start=min(line.start.x, line.end.x),
                    end=max(line.start.x, line.end.x),
                    position=(line.start.y + line.end.y) / 2,
                    is_vertical=False,
                    line=line,
                )
            )
        # Vertical lines (close to 90°)
        elif angle - 90 <= angle_tolerance:
            structure_lines.append(
                StructureLine(
                    start=min(line.start.y, line.end.y),
                    end=max(line.start.y, line.end.y),
                    position=(line.start.x + line.end.x) / 2,
                    is_vertical=True,
                    line=line,
                )
            )

    return structure_lines


def nms(tables: list[TableStructure], iou_th: float = 0.1) -> list[TableStructure]:
    """Greedy non-maximum suppression by bbox, keep higher confidence."""

    def iou(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
        ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
        ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = a.get_area() + b.get_area() - inter
        return inter / max(union, 1e-9)

    out = []
    for t in sorted(tables, key=lambda x: x.confidence, reverse=True):
        if all(iou(t.rect, o.rect) <= iou_th for o in out):
            out.append(t)
    return out


@dataclasses.dataclass
class StructureLine:
    """Helper class for representing horizontal and vertical lines in a table structure."""

    start: float
    end: float
    position: float
    is_vertical: bool
    line: Line
