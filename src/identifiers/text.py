import numpy as np
from ..page_structure import PageContext

def identify_text(ctx:PageContext):
    """ Classifies Page as Text Page based on word density in TextBlocks and average words per TextLine"""
    if not ctx.lines:
        return False
    words_per_line = [len(line.words) for line in ctx.lines]
    mean_words_per_line = np.mean(words_per_line)

    block_area = sum(block.rect.get_area() for block in ctx.text_blocks)

    if block_area == 0:
        logger.warning("Text block area is zero. Cannot compute word density.")
        return False

    word_area = sum(
        word.rect.get_area()
        for block in ctx.text_blocks
        for line in block.lines
        for word in line.words if len(line.words) > 1
    )

    word_density = word_area / block_area
    return word_density > 1 and mean_words_per_line > 3
