import regex
from ..text import TextLine
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

def identify_map(ctx: PageContext, matching_params) -> bool:
    """Identifies whether a page contains a map based on structure and keyword patterns."""
    info_lines = [
        line for line in ctx.lines
        if is_description(line, matching_params["map_terms"].get(ctx.language, {})) or find_map_scales(line)
    ]

    small_blocks = [text_block for text_block in ctx.text_blocks if len(text_block.lines) <= 3]
    filtered_lines = [
        line for block in small_blocks
        for line in block.lines
        if len(line.words) < 4 and line not in info_lines
    ]

    if filtered_lines and (len(filtered_lines)/len(ctx.lines) ) > 0.5:

        clusters = cluster_text_elements(filtered_lines, key_fn= lambda line:line.rect.x0)
        potential_scales = [cluster for cluster in clusters if len(cluster) > 3] #scales or legends
        map_clusters = list(filter(lambda cluster: cluster not in potential_scales, clusters))

        if map_clusters:
            filtered_words = [word
                              for lines in map_clusters
                              for line in lines
                              for word in line.words]

            if len(filtered_words) < 7 and not info_lines:
                return False

            def _is_a_number(string: str)-> bool:
                try:
                    float(string)
                    return True
                except ValueError:
                    return False

            map_like_words = [word for word in filtered_words
                if (word.text.isalpha() and word.text.istitle()
                    or word.text.isupper()
                    or _is_a_number(word.text))]

            if map_like_words:
                ratio = (len(map_like_words)) / len(filtered_words)
                threshold = 0.6 if info_lines else 0.75

                return ratio > threshold

    return False