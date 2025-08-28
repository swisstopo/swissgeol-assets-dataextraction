import fitz

from src.text_objects import TextLine, TextWord


def test_textline_with_valid_words():
    word1 = TextWord(fitz.Rect(10, 20, 50, 30), "Hello", 1)
    word2 = TextWord(fitz.Rect(60, 20, 100, 30), "World", 1)
    words = [word1, word2]

    text_line = TextLine(words)

    # Assert
    assert text_line.line_text() == "Hello World"
    assert text_line.rect == fitz.Rect(10, 20, 100, 30)
    assert text_line.page_number == 1


def test_textline_with_empty_words():
    words = []

    try:
        TextLine(words)
    except ValueError as e:
        assert str(e) == "Cannot create an empty TextLine."
    else:
        raise AssertionError("Expected ValueError was not raised")
