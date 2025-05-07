
def identify_text(features:dict):
    """ Classifies Page as Text Page based on word density in TextBlocks and average words per TextLine"""
    return (features["word_density"] > 1 and features["mean_words_per_line"] > 3)