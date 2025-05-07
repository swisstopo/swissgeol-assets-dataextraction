from ..page_structure import PageContext
from ..material_description import detect_material_description


def identify_boreprofile(ctx: PageContext, matching_params) -> bool:
    """Identifies whether a page contains a boreprofile based on presence of a valid material description in given language"""
    material_descriptions = detect_material_description(ctx.lines, ctx.words, matching_params["material_description"].get(ctx.language, {}))

    if ctx.geometric_lines:
        long_geometric_lines = [line for line in ctx.geometric_lines if line.length > ctx.page_rect.height / 3]
    else:
        long_geometric_lines = []

    return any(description.is_valid(ctx.page_rect, long_geometric_lines) for description in material_descriptions)
