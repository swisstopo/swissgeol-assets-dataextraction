from ..page_structure import PageContext
from ..material_description import detect_material_description
import logging
import re
logger = logging.getLogger(__name__)

def identify_boreprofile(ctx: PageContext, matching_params) -> bool:
    """Identifies whether a page contains a boreprofile based on presence of a valid material description in given language"""

    if ctx.is_digital and not (ctx.drawings or ctx.images):
        return False

    if ctx.is_digital and ctx.images:
        return True if keywords_in_figure_description(ctx, matching_params) else False

    material_descriptions = detect_material_description(ctx.lines, ctx.words, matching_params["material_description"].get(ctx.language, {}))

    if ctx.geometric_lines:
        long_geometric_lines = [line for line in ctx.geometric_lines if line.length > ctx.page_rect.height / 3]
    else:
        long_geometric_lines = []

    return any(description.is_valid(ctx.page_rect, long_geometric_lines) for description in material_descriptions)

def keywords_in_figure_description(ctx,matching_params):

    figure_patterns = [r"\b\d{1,2}(?:\.\d{1,2}){0,3}\b"]

    boreprofile_keywords = matching_params["boreprofile"].get(ctx.language, {})
    relevant_lines = []

    def is_close_to_image(line_rect, image_rect):
        image_y0, image_y1 = image_rect[1], image_rect[3]
        return (
                abs(line_rect.y1 - image_y0) < 20 or  # directly above
                abs(line_rect.y0 - image_y1) < 20  # directly below
        )
    for line in ctx.lines:
        for image in ctx.images:
            if is_close_to_image(line.rect, image["bbox"]):
                relevant_lines.append(line)

    figure_description_lines = []

    for line in ctx.lines:
        line_text = line.line_text()
        for pattern in figure_patterns:
            if re.search(pattern, line_text):
                logger.info(f"Matched figure pattern in line: {line_text}")
                figure_description_lines.append(line_text.lower())
                break

    boreprofile_lines = [
        line for line in figure_description_lines
        if any(keyword in line for keyword in boreprofile_keywords)
    ]

    logger.info(boreprofile_lines)
    return boreprofile_lines