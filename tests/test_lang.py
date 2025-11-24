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
        ("Hoc est experimentum ad linguam textus probandam.", "de"),
        # Empty input
        (None, None),
    ],
    ids=["de-supported", "en-supported", "fr-supported", "it-supported", "unsupported-lang", "empty-input"],
)
def test_predict_language(text: str | None, expected: str | None) -> None:
    """Validate language prediction for supported/unsupported and empty inputs.

    Args:
        text (str | None): Input text to classify. May be None to represent empty input.
        expected (str | None): Language code for supported inputs, otherwise None.
    """
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
    ids=[
        "de-enough-words",
        "de-enough-words",
        "de-enough-words",
        "de-enough-words",
        "unsupported-default-de",
        "none-default-de",
        "not-enough-words-default-de",
    ],
)
def test_select_classification_language(predicted: str | None, word_count: int, expected: str | None) -> None:
    """Ensure classification language selection follows support and length rules.

    Args:
        predicted (str | None): Predicted language code from detection, or None if unknown.
        word_count (int): Number of words in the document; must exceed the minimum threshold.
        expected (str): Expected language used for downstream classification (defaults to 'de'
            when unsupported or too few words).
    """
    assert select_classification_language(predicted, word_count) == expected


@pytest.mark.parametrize(
    "scores, long_counts, page_count, expected",
    [
        # Single lang detected
        ({"de": 2.5}, {"de": 1}, 1, {"languages": ["de"], "page_count": 1}),
        # Two lang detected, but second has only 1 page
        ({"de": 2.5, "fr": 0.5}, {"de": 1, "fr": 1}, 2, {"languages": ["de"], "page_count": 2}),
        # Two lang detected, higher score in DE
        ({"de": 2.5, "fr": 0.5}, {"de": 3, "fr": 2}, 5, {"languages": ["de", "fr"], "page_count": 5}),
        ({"de": 2.5, "fr": 0.5}, {"de": 2, "fr": 3}, 5, {"languages": ["de", "fr"], "page_count": 5}),
        # Two lang detected, higher score in FR
        ({"de": 0.5, "fr": 2.5}, {"de": 3, "fr": 2}, 5, {"languages": ["fr", "de"], "page_count": 5}),
        ({"de": 0.5, "fr": 2.5}, {"de": 2, "fr": 3}, 5, {"languages": ["fr", "de"], "page_count": 5}),
    ],
    ids=[
        "single-language",
        "two-langs-fr-too-short",
        "two-langs-order-by-score-de",
        "two-langs-order-by-score-not-page-de",
        "two-langs-order-by-score-fr",
        "two-langs-order-by-score-not-page",
    ],
)
def test_summarize_language_metadata(
    scores: dict[str, float], long_counts: dict[str, int], page_count: int, expected: dict
) -> None:
    """Check language summary ordering and page aggregation logic.

    Args:
        scores (dict[str, float]): Aggregate confidence per language (higher is better).
        long_counts (dict[str, int]): Count of long/valid pages per language.
        page_count (int): Total number of processed pages in the document.
        expected (dict[str, Any]): Expected summary payload with keys.
    """
    assert summarize_language_metadata(scores, long_counts, page_count) == expected
