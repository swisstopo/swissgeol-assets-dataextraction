from .text import TextLine
import logging
logger = logging.getLogger(__name__)

title_page_substrings = {
    "page_de_garde_1": [
        ["PAGE DE GARDE", "P A G E DE G A R D E"],
        ["No AGS"],
        ["numéro ou signe"],
        ["distinctif original"],
        ["Commettants"],
        ["Réseau km"]
    ],
    "page_de_garde_2": [
        ["PAGE DE GARDE", "P A G E DE G A R D E"],
        ["Commettants"],
        ["Renseignements"],
        ["(composition du"],
        ["Carte référence"],
        ["et carroyage"],
        ["des mots-clés"],
        ["Lieu de dépôt"],
        ["Disponibilité"],
        ["date limite"],
        ["Numéro AGS"]
    ],
    "page_de_garde_3": [
        ["PAGE DE GARDE", "P A G E DE G A R D E"],
        ["No AGS"],
        ["numéro ou autre"],
        ["signe distinctif"],
        ["original du"],
        ["Commettants"],
        ["Réseau km"]
    ],
    "belegblatt_1": [
        ["BELEGBLATT", "B E L E G B L A T T"],
        ["SGD-Nr"],
        ["Originalkennzeichen"],
        ["des Dokumentes"],
        ["Km-Netz"],
        ["Kartenblatt"],
        ["Auftraggeber"],
        ["oder Rechts"],
        ["nachfolger"]
    ],
    "belegblatt_2": [
        ["BELEGBLATT", "B E L E G B L A T T"],
        ["SGD-Nr"],
        ["Originalkenn"],
        ["zeichen des"],
        ["Dokumentes"],
        ["Km-Netz"],
        ["Kartenblatt"],
        ["Auftraggeber"]
    ],
    "belegblatt_3": [
        ["BELEGBLATT", "B E L E G B L A T T"],
        ["SGD-Nr"],
        ["Originalnummer"],
        ["kennzeichen"],
        ["des Dokumentes"],
        ["Km-Netz"],
        ["Kartenblatt"],
        ["Auftraggeber"]
    ],
    "auszug_aus_dem_titelverzeichnis_1": [
        ["Auszug aus dem Titelverzeichnis"],
        ["Sortierung nach:"],
        ["Unser Zeichen:"],
        ["InfoGeol-Nr."],
        ["Metatitel"],
        ["Dokumentenzusammensetzung"]
    ],
    "auszug_aus_dem_titelverzeichnis_2": [
        ["Auszug aus dem Titelverzeichnis"],
        ["Sortierung nach:"],
        ["Unser Zeichen:"],
        ["InfoGeol-Nr."],
        ["Fläche"],
        ["Chronostratigraphie"],
        ["Tektonische Einheit"],
        ["Lithologie"],
        ["Aufschlussart"],
        ["Hydrologie"],
        ["swisstopo, Geologische Informationsstelle"],
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
        ["BEILAGEN"]
    ],
    "sgs_dokumentnummer": [
        ["SGS-DOKUMENTNUMMER"],
        ["AUTOREN"],
        ["AUFTRAGGEBER oder HERAUSGEBER"],
        ["ZUSAMMENSETZUNG DES DOKUMENTES"],
        ["DATUM"],
        ["STANDORT"],
        ["VERFUEGBARKEIT"],
        ["NATIONALKARTENBLATT"],
        ["STICHWOERTER"]
    ]
}


def title_page_type(text: str) -> str | None:
    return next(
        (
            type_id
            for type_id, substrings in title_page_substrings.items()
            if is_title_page(text, substrings)
        ),
        None
    )

def is_title_page(text: str, substrings: list[list[str]]) -> bool:
    evaluations = [
        any(substring in text for substring in substring_list)
        for substring_list in substrings
    ]
    return len([evaluation for evaluation in evaluations if evaluation]) / len(evaluations) >= 0.7

def sparse_title_page(lines: list[TextLine]):
    # if len(lines) > 30: # too many lines -> prob no sparse title page
    #     return False
    
    # not_right_aligned_lines = [line for line in lines if line.rect.x0 > 50] 
    # if len(not_right_aligned_lines) < 2:
    #     return False
    
    font_sizes = [line.font_size for line in lines]

    multiple_sizes = len(set(font_sizes)) > 5
    large_font = max(font_sizes)>20

    if multiple_sizes > 5 and large_font > 20:
        logger.info((multiple_sizes,large_font, len(lines)))
        return True
    return False