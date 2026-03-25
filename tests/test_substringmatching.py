from types import SimpleNamespace

import pytest

from src.language_detection.pages_to_ignore import max_title_page_keyword_score


def _make_lines(text: str) -> list:
    return [SimpleNamespace(text=line) for line in text.split("\n")]


@pytest.fixture
def patched_title_patterns(monkeypatch):
    patterns = {
        "auszug_aus_dem_titelverzeichnis_1": [
            ["Auszug aus dem Titelverzeichnis"],
            ["Sortierung nach:"],
            ["Unser Zeichen:"],
            ["InfoGeol-Nr."],
            ["Metatitel"],
            ["Dokumentenzusammensetzung"],
        ],
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
    # 34506_1.pdf
    text = """SGM-D2
        34506
        V+G -K
        D-DAT
        DETAILBILD DER DOKU:
        GESTEINSBESCHREIBUNG BEI DOS-D'âNE, ENE LA VALSAINTE
        52VAL
        17 03 1943
        AUTOR GDA FUR
        A.GEB GEOLOGISCHER DIENST DER ARMEE
        Z-SETZ BERICHTE 0
        NOTIZEN 0
        TABELLEN 0
        DIAGRAMME 0
        SEITEN 0
        FOTOS 0
        1 SITUATIONSSKIZZE CA. 1:5000 MIT GEOLOGISCHEN ANGABEN
        1 GESTEINSSKIZZE 1:5 / 1 ZEICHNUNG DÜNNSCHLIFF
        O-ZEI REG.NR. 11541
        SIEGFRIEDBLATT 361 (BERRA)
        ZEILE: 1
        GEOL. 230 57 400
        430 57 400
        BEILAGEN 1
        ENDE - FIN - FINE
        ORIGINAL IM ARCHIV"""

    lines = _make_lines(text)

    score = max_title_page_keyword_score(lines)

    assert score == pytest.approx(8 / 9)


def test_belegblatt_case(patched_title_patterns):
    # 6491_1.pdf
    text = """EB
        Standort
        SGD
        Geograph. Lage
        591'51/200'12
        setzung
        Dokument-
        Auftraggeber
        0
        Autor (en)
        37
        Datum
        S
        SGD
        30
        Verfügbarkeit 28
        Bericht
        Auszug
        Notiz
        Anzahl Seiten
        Graphika, als Beilage
        Kartenblatt
        24
        A
        Km-Netz
        18,21
        Geologie
        Records
        (RA)
        Anzahl
        SGD-Nr.
        (Koordinaten)
        1
        oder allein
        Tafeln, Abbildungen
        1
        Tabelle Laborresultate
        1
        Diagramme Kornvert.kurv.
        Fotografien
        Situation 1:1000
        1
        geol.Bohrprofil 1:100
        1
        Tiefe: 10,0m
        Rammprofil 1:50
        Titel
        SGD-Nr.
        Autobahnamt des Kantons Bern
        30
        0306975
        Autor (en)
        6491
        15
        13
        10
        -0
        1166
        1166
        591
        200
        591
        200
        8
        220
        67
        120
        11
        7
        6491
        300
        57
        220
        1
        Tiefe: 5,0m
        S. auch SGD 6479
        4708
        10
        37
        BZN
        41
        45
        49
        53
        57
        61
        12
        03
        02
        1166
        1166
        591
        200
        591 200
        10
        100
        64
        120
        13
        200
        (oder Gebiet)'
        61
        120
        14
        15
        Dünnschliffe
        Anschliffe
        Handstücke
        Proben
        Bohrkerne
        Bohrproben
        65
        16
        69
        73
        Verfalldatum:
        17
        18
        19
        01
        N1,
        SONDIERUNGEN
        F.UEBERFUEHRUNG
        BEI
        HUBEL,
        700M
        WSW
        FRAUENKAPPELEN
        RA
        BELEGBLATT
        *bitte wenden!
        16 676
        des Dokumentes
        -kennzeichen
        oder
        Originalnummer
        90
        Mut. 3
        75
        20
        4.
        3.
        2.
        1. 43FRA
        82
        33
        Mut.
        Selektion"""

    lines = _make_lines(text)

    score = max_title_page_keyword_score(lines)

    assert score == pytest.approx(1.0)
