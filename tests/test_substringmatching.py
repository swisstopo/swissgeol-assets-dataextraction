from types import SimpleNamespace

import pytest

from src.language_detection.pages_to_ignore import max_title_page_keyword_score


def _make_lines(text: str) -> list:
    return [SimpleNamespace(text=line) for line in text.split("\n")]


@pytest.fixture
def patched_title_patterns(monkeypatch):
    patterns = {
        "belegblatt": [
            ["BELEGBLATT"],
            ["*bitte wenden!"],
            ["Standort"],
            ["Geograph. Lage"],
        ],
        "detailbild_der_doku": [
            ["DETAILBILD DER DOKU", "D E T A I L B I L D"],
            ["Z-SETZ"],
            ["SGM-DZ"],
            ["BERICHTE"],
            ["TABELLEN"],
            ["NOTIZEN"],
            ["DIAGRAMME"],
            ["SEITEN"],
            ["BEILAGEN"],
        ],
    }

    monkeypatch.setattr(
        "src.language_detection.pages_to_ignore.title_page_substrings",
        patterns,
    )


def test_detailbild_case(patched_title_patterns):
    # reduced version from 34506_1.pdf
    text = """SGM-D2
        DETAILBILD DER DOKU:E
        Z-SETZ BERICHTE 0
        NOTIZEN 0
        TABELLEN 0
        DIAGRAMME 0
        SEITEN 0
        FOTOS 0
        ZEILE: 1
        BEILAGEN 1
        ENDE - FIN - FINE
        ORIGINAL IM ARCHIV"""

    lines = _make_lines(text)

    score = max_title_page_keyword_score(lines)

    assert score == pytest.approx(8 / 9)


def test_belegblatt_case(patched_title_patterns):
    #  reduced version from 6491_1.pdf
    text = """EB
        Standort
        SGD
        Geograph. Lage
        591'51/200'12
        Bericht
        Kartenblatt
        BELEGBLATT
        *bitte wenden!"""

    lines = _make_lines(text)

    score = max_title_page_keyword_score(lines)

    assert score == pytest.approx(1.0)
