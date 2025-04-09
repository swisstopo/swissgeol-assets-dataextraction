from ..page_structure import PageContext
from ..material_description import detect_material_description


def identify_boreprofile(ctx: PageContext, matching_params) -> bool:
    """Identifies whether a page contains a boreprofile based on presence of  a valid material description in given language"""
    material_descriptions = detect_material_description(ctx.lines, ctx.words, matching_params["material_description"].get(ctx.language, {}))

    return any(description.is_valid(ctx.page_rect) for description in material_descriptions)
