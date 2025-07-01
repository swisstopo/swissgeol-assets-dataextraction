from src.text_objects import TextLine


def sparse_title_page(lines: list[TextLine]) -> bool:
    if not lines or len(lines) > 30:
        return False

    font_sizes = [line.font_size for line in lines]

    multiple_sizes = len(set(font_sizes)) > 5
    large_font = max(font_sizes) > 20

    if multiple_sizes and large_font:
        return True

    return False
