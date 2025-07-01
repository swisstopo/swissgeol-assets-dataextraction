import logging

import numpy as np
import pymupdf
from shapely.geometry import box
from shapely.ops import unary_union

from ..page_structure import PageContext

logger = logging.getLogger(__name__)


def get_union_areas(rects: list[pymupdf.Rect]) -> float:
    """Compute total non-overlapping area from list of bounding box rects."""
    shapes = [box(rect.x0, rect.y0, rect.x1, rect.y1) for rect in rects]
    return unary_union(shapes).area if shapes else 0


def identify_text(ctx: PageContext) -> bool:
    """Classifies Page as Text Page based on word density in TextBlocks and average words per TextLine"""
    if not ctx.lines:
        return False
    words_per_line = [len(line.words) for line in ctx.lines]
    mean_words_per_line = np.mean(words_per_line)

    block_union = get_union_areas([block.rect for block in ctx.text_blocks])

    if block_union == 0:
        logger.warning("Text block area is zero. Cannot compute word density.")
        return False

    word_union = get_union_areas(
        [word.rect for block in ctx.text_blocks for line in block.lines for word in line.words if len(line.words) > 1]
    )
    word_density = word_union / block_union
    return word_density > 0.5 and mean_words_per_line > 3
