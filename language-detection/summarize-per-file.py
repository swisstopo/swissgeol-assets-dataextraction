import csv
import math
import pandas as pd


def format_summary(value: str, start_page: int, end_page: int) -> str:
    if end_page:
        return "{} (p. {}-{})".format(value, start_page, end_page)
    else:
        return "{} (p. {})".format(value, start_page)


def language_summary(asset_pages) -> str:
    current_language = None
    start_page = None
    end_page = None
    entries = []

    for i, entry in enumerate(asset_pages):
        page_number = int(entry["page_number"])
        language_code = entry["language_code"]
        title_page_type = entry["title_page_type"]
        if title_page_type:
            language_code = "title_page"

        if language_code:
            if current_language is None:
                start_page = page_number
                current_language = language_code
            elif current_language != language_code:
                entries.append(format_summary(current_language, start_page, end_page))
                start_page = page_number
                end_page = None
                current_language = language_code
            else:  # current_language == language_code
                end_page = page_number

    if current_language:
        entries.append(format_summary(current_language, start_page, end_page))

    return ", ".join(entries)


def ocr_type(asset_pages) -> str:
    filtered_pages = [page for page in asset_pages if not page["title_page_type"] and page["ocr_type"]]
    if len(filtered_pages) == 0:
        return ""
    if all(page["ocr_type"] == "ocr" for page in filtered_pages):
        return "ocr"
    if all(page["ocr_type"] == "digital" for page in filtered_pages):
        return "digital"
    else:
        return "mixed"


def ocr_summary(asset_pages) -> str:
    current_value = None
    start_page = None
    end_page = None
    entries = []

    for i, entry in enumerate(asset_pages):
        page_number = int(entry["page_number"])
        ocr_type = entry["ocr_type"]
        title_page_type = entry["title_page_type"]
        if title_page_type:
            if current_value:
                entries.append(format_summary(current_value, start_page, end_page))
                start_page = None
                end_page = None
                current_value = None
        elif ocr_type:
            if current_value is None:
                start_page = page_number
                current_value = ocr_type
            elif current_value != ocr_type:
                entries.append(format_summary(current_value, start_page, end_page))
                start_page = page_number
                end_page = None
                current_value = ocr_type
            else:  # current_value == ocr_type
                end_page = page_number

    if current_value:
        entries.append(format_summary(current_value, start_page, end_page))

    return ", ".join(entries)


output = []
with open("data/pages.csv", "r", newline="") as pages_file:
    reader = csv.DictReader(pages_file)
    files = {}
    for row in reader:
        filename = row["filename"]
        if filename not in files:
            files[filename] = []
        files[filename].append(row)

    for filename, file_pages in files.items():
        scores = {}
        long_pages = {}
        for i, entry in enumerate(file_pages):
            page_number = int(entry["page_number"])
            language_code = entry["language_code"]
            character_count = int(entry["character_count"])
            word_count_not_short = int(entry["word_count_not_short"])
            title_page_type = entry["title_page_type"]

            if language_code and not title_page_type:
                if language_code not in scores:
                    scores[language_code] = 0
                if word_count_not_short > 0:
                    scores[language_code] += math.log(word_count_not_short) / page_number

                if word_count_not_short > 50:
                    if language_code not in long_pages:
                        long_pages[language_code] = 0
                    long_pages[language_code] += 1

        if len(scores) > 0:
            best_language = max(scores, key=scores.get)
        else:
            best_language = ""

        page_count = len(file_pages)
        other_significant_langauges = [
            language_code
            for language_code, value in long_pages.items()
            if value >= 2
            if language_code != best_language
        ]
        output.append(
            [
                filename,
                page_count,
                best_language,
                ",".join(other_significant_langauges),
                language_summary(file_pages),
                ocr_type(file_pages),
                ocr_summary(file_pages),
            ]
        )

df = pd.DataFrame(
    output,
    columns=[
        "filename",
        "page_count",
        "best_language",
        "other_significant_languages",
        "notes",
        "text_type",
        "text_type_notes",
    ],
)
df.to_csv("data/files.csv", index=False)

regular = df[~df["filename"].str.lower().str.endswith("_ldoc.pdf")]
ldoc = df[df["filename"].str.lower().str.endswith("_ldoc.pdf")]


def stats(df):
    other_significant_languages = (
        df["other_significant_languages"].str.split(",").explode("other_significant_languages")
    )
    print()
    print(df["best_language"].value_counts(dropna=False))
    with_additional_count = df["other_significant_languages"].map(lambda x: len(x) > 0).sum()
    print()
    print("{} files with additional significant languages.".format(with_additional_count))
    print(other_significant_languages.value_counts())
    print()
    print()


print("{} regular files".format(len(regular)))
stats(regular)

print("{} _LDoc.pdf files.".format(len(ldoc)))
stats(ldoc)
