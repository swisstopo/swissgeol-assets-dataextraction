title_page_substrings = {
    "page_de_garde_1": [
        ["PAGE DE GARDE", "P A G E DE G A R D E"],
        ["No AGS"],
        ["numéro ou signe"],
        ["distinctif original"],
        ["Commettants"],
        ["Réseau km"],
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
        ["Numéro AGS"],
    ],
    "page_de_garde_3": [
        ["PAGE DE GARDE", "P A G E DE G A R D E"],
        ["No AGS"],
        ["numéro ou autre"],
        ["signe distinctif"],
        ["original du"],
        ["Commettants"],
        ["Réseau km"],
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
        ["nachfolger"],
    ],
    "belegblatt_2": [
        ["BELEGBLATT", "B E L E G B L A T T"],
        ["SGD-Nr"],
        ["Originalkenn"],
        ["zeichen des"],
        ["Dokumentes"],
        ["Km-Netz"],
        ["Kartenblatt"],
        ["Auftraggeber"],
    ],
    "belegblatt_3": [
        ["BELEGBLATT", "B E L E G B L A T T"],
        ["SGD-Nr"],
        ["Originalnummer"],
        ["kennzeichen"],
        ["des Dokumentes"],
        ["Km-Netz"],
        ["Kartenblatt"],
        ["Auftraggeber"],
    ],
    "auszug_aus_dem_titelverzeichnis_1": [
        ["Auszug aus dem Titelverzeichnis"],
        ["Sortierung nach:"],
        ["Unser Zeichen:"],
        ["InfoGeol-Nr."],
        ["Metatitel"],
        ["Dokumentenzusammensetzung"],
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
        ["BEILAGEN"],
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
        ["STICHWOERTER"],
    ],
}


def is_belegblatt(text: str) -> str | None:
    """Check if the text matches any of the Belegblatt patterns."""
    return any(contains_substrings(text, substrings) for substrings in title_page_substrings.values())


def contains_substrings(text: str, substrings: list[list[str]]) -> bool:
    """Check if the text contains at least 70% of the substrings."""
    evaluations = [any(substring in text for substring in substring_list) for substring_list in substrings]
    return len([evaluation for evaluation in evaluations if evaluation]) / len(evaluations) >= 0.7
