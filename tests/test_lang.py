import pytest

from src.language_detection.detect_language import (
    predict_language,
    select_classification_language,
    summarize_language_metadata,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        # Supported languages
        ("Dies ist ein Test zur Überprüfung der Textsprache.", "de"),
        ("This is a test to check text language.", "en"),
        ("Ceci est un test pour vérifier la langue du texte.", "fr"),
        ("Questo è un test per verificare la lingua del testo.", "it"),
        # Non-supported languages
        ("Hoc est experimentum ad linguam textus probandam.", None),
        # Empty input
        (None, None),
    ],
)
def test_predict_language(text: str | None, expected: str | None) -> None:
    """Test language prediction package."""
    assert predict_language(text) == expected


@pytest.mark.parametrize(
    "predicted, word_count, expected",
    [
        # Supported languages with enough words
        ("de", 100, "de"),
        ("en", 100, "en"),
        ("fr", 100, "fr"),
        ("it", 100, "it"),
        # Non-supported languages -> defaults to DE
        ("la", 100, "de"),
        (None, 100, "de"),
        # Not enough words -> defaults to DE
        ("en", 0, "de"),
    ],
)
def test_select_classification_language(predicted: str | None, word_count: int, expected: str | None) -> None:
    """Test language selection for classificaiton."""
    assert select_classification_language(predicted, word_count) == expected


@pytest.mark.parametrize(
    "scores, long_counts, page_count, expected",
    [
        # Single lang detected
        ({"de": 2.5}, {"de": 1}, 1, {"languages": ["de"], "page_count": 1}),
        # Two lang detected, but second has only 1 page
        ({"de": 2.5, "fr": 0.5}, {"de": 1, "fr": 1}, 2, {"languages": ["de"], "page_count": 2}),
        # Two lang detected, higher score in DE
        ({"de": 2.5, "fr": 0.5}, {"de": 2, "fr": 2}, 4, {"languages": ["de", "fr"], "page_count": 4}),
        ({"de": 2.5, "fr": 0.5}, {"de": 2, "fr": 4}, 6, {"languages": ["de", "fr"], "page_count": 6}),
        # Two lang detected, higher score in FR
        ({"de": 0.5, "fr": 2.5}, {"de": 2, "fr": 2}, 4, {"languages": ["fr", "de"], "page_count": 4}),
        ({"de": 0.5, "fr": 2.5}, {"de": 4, "fr": 2}, 6, {"languages": ["fr", "de"], "page_count": 6}),
    ],
)
def test_summarize_language_metadata(
    scores: dict[str, float], long_counts: dict[str, int], page_count: int, expected: dict
) -> None:
    """Test language summarized for classificaiton."""
    assert summarize_language_metadata(scores, long_counts, page_count) == expected
