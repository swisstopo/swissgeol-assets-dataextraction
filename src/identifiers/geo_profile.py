from identifiers.map import map_lines_score
from language_detection.detect_language import DEFAULT_LANGUAGE
from page_structure import PageContext


def identify_geo_profile(ctx: PageContext, matching_params: dict) -> bool:
    """Determines whether a page should be classified as a geo profile page.

    Geo profile are identified by the presence of sidebars, certain keywords, a medium line score.

    Args:
        ctx (PageContext): The context of the page containing text and other information.
        matching_params (dict): A dictionary containing geo profile keywords for different languages.

    """
    key_words = matching_params["geo_profile"].get(ctx.language, DEFAULT_LANGUAGE)
    if any([any(kw in line.line_text().lower() for kw in key_words) for line in ctx.lines]):
        return True

    line_score = map_lines_score(ctx)  # tends to 0 if all lines are axis-alligned and have the same angles

    # line_score smaller than 0.15 is likely a diagram, above 0.25 likelly a map
    return 0.15 < line_score < 0.25 and len(ctx.geometric_lines) > 500
