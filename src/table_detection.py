"""This module contains functionalities to detect table like structures."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import pymupdf
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textline import TextLine

from src.bounding_box import bbox_of_lines, rects_intersect
from src.utils import read_params

logger = logging.getLogger(__name__)

config = read_params("config/_params.yml")


@dataclass
class TableStructure:
    """Represents a detected table structure."""

    rect: pymupdf.Rect
    horizontal_lines: list[Line]
    vertical_lines: list[Line]
    confidence: float
    line_density: float


def area_ratio(rect: pymupdf.Rect, page_w: float, page_h: float) -> float:
    """Area of rect divided by area of page."""
    page_area = max(page_w * page_h, 1e-9)
    rect_area = max(rect.width * rect.height, 0.0)
    return rect_area / page_area


def score_area(rect: pymupdf.Rect, page_w: float, page_h: float) -> float:
    """Grow with area until min_table_area_ratio; cap at p.area_weight."""
    ratio = area_ratio(rect, page_w, page_h) if page_w and page_h else 0.0
    score_config = config.get("area_scoring")
    area_weights = score_config.get("area_weights")
    return min(area_weights, (ratio / score_config.get("min_table_area_ratio")) * area_weights)


def score_lines(total_lines: int) -> float:
    """Grow with number of lines until max_n_lines_bonus; cap at p.line_weight."""
    score_config = config.get("line_scoring")
    line_weights = score_config.get("line_weights")

    return min(line_weights, (total_lines / max(score_config.get("max_n_lines_bonus"), 1)) * line_weights)


def score_text(
    rect: pymupdf.Rect,
    text_lines: list[TextLine],
) -> float:
    """Score based on number of text lines intersecting the rectangle.

    Each text line contributes p.text_presence_weight, capped at p.text_weight.
    """
    if not text_lines:
        return 0.0
    n_text = sum(1 for tl in text_lines if rect.intersects(tl.rect))
    score_config = config.get("text_scoring")

    return min(score_config.get("text_weights"), n_text * score_config.get("text_presence_weight"))


def detect_table_structures(
    page_rect: pymupdf.Rect,
    structure_lines: list[StructureLine],
    text_lines: list[TextLine],
) -> list[TableStructure]:
    """Detect multiple non-overlapping table structures on a page.

    Logic:
    - Group connected lines.
    - For each group create a table if:
        - it has at least 2 horizontal and 1 vertical line
        - its confidence is above threshold.
    - Score tables based on confidence (size, line count, text presence).
    - Remove overlapping tables by confidence (keep highest).
    """
    regions = _find_table_regions(structure_lines)

    detected: list[TableStructure] = []
    for h_lines, v_lines in regions:
        if len(h_lines) < 2 or len(v_lines) < 2:
            continue

        if (
            table := _create_table_from_region(h_lines, v_lines, page_rect, text_lines)
        ) and table.confidence >= config.get("min_confidence"):
            detected.append(table)

    detected.sort(key=lambda table: table.confidence, reverse=True)

    final: list[TableStructure] = []

    for table in detected:
        if not _table_overlaps(table, final):
            final.append(table)

    return final


def _find_table_regions(lines: list[StructureLine]) -> list[tuple[list[Line], list[Line]]]:
    """Group connected lines; return horizontal and vertical lines for each region.

    Logic:
    - Group lines that are connected.
    - For each group, separate horizontal and vertical lines.
    - Return only groups that have horizontal and vertical line.
    """
    groups: list[list[StructureLine]] = []
    for line in lines:
        matches = [idx for idx, g in enumerate(groups) if _line_connects_to_group(line, g)]
        if not matches:
            groups.append([line])
        elif len(matches) == 1:
            groups[matches[0]].append(line)
        else:
            # merge groups + add line
            merged = [line]
            for idx in sorted(matches, reverse=True):
                merged.extend(groups.pop(idx))
            groups.append(merged)

    regions: list[tuple[list[Line], list[Line]]] = []
    for g in groups:
        horizontal = [line.line for line in g if not line.is_vertical]
        vertical = [line.line for line in g if line.is_vertical]
        if horizontal and vertical:
            regions.append((horizontal, vertical))
    return regions


def _line_connects_to_group(line: StructureLine, group: list[StructureLine]) -> bool:
    """Connection check: intersection, end-point proximity, or T-junction.

    Args:
        line: StructureLine to check.
        group: List of StructureLines representing a group.

    Returns:
        True if line connects to any line in the group, else False.
    """
    thr = config.get("connection_threshold")

    for g in group:
        if line.line.intersects_with(g.line):
            return True

        for p1 in (line.line.start, line.line.end):
            for p2 in (g.line.start, g.line.end):
                if p1.distance_to(p2) <= thr:
                    return True

        for point in (line.line.start, line.line.end):
            if g.line.point_near_segment(point, thr):
                return True
        for point in (g.line.start, g.line.end):
            if line.line.point_near_segment(point, thr):
                return True
    return False


def _table_overlaps(table: TableStructure, existing: list[TableStructure]) -> bool:
    """Check if a table overlaps with any existing tables."""
    return any(rects_intersect(table.rect, t.rect) for t in existing)


def _create_table_from_region(
    horizontal_lines: list[Line],
    vertical_lines: list[Line],
    page_rect: pymupdf.Rect,
    text_lines: list[TextLine],
) -> TableStructure | None:
    """Create a TableStructure from given horizontal and vertical lines, scoring its confidence."""
    page_w, page_h = page_rect.width, page_rect.height

    all_lines = horizontal_lines + vertical_lines
    rect = bbox_of_lines(all_lines)
    if rect is None:
        return None

    # Scores
    total_lines = len(all_lines)
    size_score = score_area(rect, page_w, page_h)
    line_score = score_lines(total_lines)
    text_score = score_text(rect, text_lines)

    confidence = min(1.0, size_score + line_score + text_score)

    area = max(rect.width * rect.height, 0.0)
    line_density = (total_lines / (area / 10_000.0)) if area > 0 else 0.0

    return TableStructure(
        rect=rect,
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        confidence=confidence,
        line_density=line_density,
    )


@dataclasses.dataclass
class StructureLine:
    """Helper class for representing horizontal and vertical lines in a table structure."""

    start: float
    end: float
    position: float
    is_vertical: bool
    line: Line


def detect_structure_lines(geometric_lines: list[Line]) -> list[StructureLine]:
    """Detects horizontal and vertical lines longer than a minimum length on a page."""
    min_length = config.get("min_line_length")

    filtered_lines = [line for line in geometric_lines if line.length > min_length]
    return _separate_by_orientation(filtered_lines, config)


def _separate_by_orientation(lines: list[Line], config_file: dict) -> list[StructureLine]:
    """Separate lines into horizontal and vertical based on angle and tolerance."""
    angle_tolerance = config_file.get("angle_tolerance")
    structure_lines = []

    for line in lines:
        angle = abs(line.angle)

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
