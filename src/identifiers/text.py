
def identify_text(ctx,features:dict):
    """ Classifies Page as Text Page based on word density in TextBlocks and average words per TextLine"""

    if ctx.is_digital and ctx.images:
        return False

    return (features["word_density"] > 1 and features["mean_words_per_line"] > 3)