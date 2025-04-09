import regex
from ..text import TextLine, TextWord
from ..utils import is_description, cluster_text_elements
from ..page_structure import PageContext

pattern_maps = [
    regex.compile(r"1\s*:\s*[125](25|5)?000+"),
    regex.compile(r"1\s*:\s*[125]((0{1,2})?([',]000)+)")
]

def find_map_scales(line: TextLine) -> regex.Match | None:
    return next((match
                 for pattern in pattern_maps
                 for word in line.words
                 if (match := pattern.search(word.text))), None)

def get_map_entry_lines(ctx: PageContext, keyword_lines: list[TextLine]) -> list[TextLine]:
    """
    Extracts candidate lines that are likely to be map entries (e.g., street names, place labels)
    These are:
        - Lines from small text blocks (<= 3 lines per block)
        - Short lines (< 4 words)
        - Not already identified as containing map keywords
    """

    map_entry_blocks = [block for block in ctx.text_blocks if len(block.lines) <= 3]

    map_entry_lines = [
        line for block in map_entry_blocks
        for line in block.lines
        if len(line.words) < 4 and line not in keyword_lines
    ]

    return map_entry_lines

def has_enough_map_entry_lines(map_entry_lines,lines) -> bool:
    """Checks whether a 50% of the page consists of potential map entry lines."""

    return map_entry_lines and (len(map_entry_lines) / len(lines)) > 0.5

def are_words_map_like(words: list[TextWord], keyword_lines: list[TextLine]) -> bool:
    """Evaluates whether the majority of words follow a typical map entry format:
        - All uppercase (e.g., "BASEL")
        - Title case (e.g., "Bern")
        - Contains numbers (e.g., "3", "A1")
    """
    ## Needs to have at least some words if no keyword lines are provided
    if len(words) < 7 and not keyword_lines:
        return False

    def _is_a_number(string: str) -> bool:
        try:
            float(string)
            return True
        except ValueError:
            return False

    map_like_words = [word for word in words
                      if ((word.text.isalpha() and word.text.istitle())
                          or word.text.isupper()
                          or _is_a_number(word.text))]

    if not map_like_words:
        return False

    ratio = (len(map_like_words)) / len(words)
    # reduce threshold if keyword lines were found on page
    threshold = 0.6 if keyword_lines else 0.75


    return ratio > threshold

def identify_map(ctx: PageContext, matching_params) -> bool:
    """"
    Determines whether a page contains a map based on layout patterns and keyword detection.
    Detection logic:
    - Searches for known map terms or scale patterns (legend lines)
    - Identifies short text blocks typical of map entries
    - Clusters entry lines by x-position and filters large clusters
    - Confirms detection if word formatting follows typical map label format

    Args:
        ctx: Lines, blocks, language and layout information of the page.
        matching_params: Dictionary with keyword patterns for identifying map content.

    Returns:
        bool: True if the page likely contains a map.
    """

    # Find lines containing map-related keywords or scale patterns
    keyword_lines = [
        line for line in ctx.lines
        if is_description(line, matching_params["map_terms"].get(ctx.language, {})) or find_map_scales(line)
    ]

    map_entry_lines = get_map_entry_lines(ctx, keyword_lines)

    # Only proceed if a substantial portion of the page is made up of map entry lines
    if not has_enough_map_entry_lines(map_entry_lines, ctx.lines):
        return False

    # Cluster lines based on horizontal alignment
    clusters = cluster_text_elements(map_entry_lines, key_fn= lambda line:line.rect.x0)
    map_clusters = [cluster for cluster in clusters if len(cluster) <= 3]

    if not map_clusters:
        return False

    words_in_map_clusters = [word
                      for lines in map_clusters
                      for line in lines
                      for word in line.words]

    return are_words_map_like(words_in_map_clusters, keyword_lines)
