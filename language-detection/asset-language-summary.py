import csv


def asset_rows(asset, entries) -> list[list[str]]:
    best_language = ""
    all_languages = set()
    for entry in entries:
        if entry["file_language"]:
            if best_language == "":
                best_language = entry["file_language"]
            all_languages.add(entry["file_language"])
        if entry["file_additional_languages"]:
            for other_language in entry["file_additional_languages"].split(","):
                all_languages.add(other_language)
    if best_language in all_languages:
        all_languages.remove(best_language)

    mismatch = ""
    if asset["language_item_code"] and asset["language_item_code"] != 'other' and asset["language_item_code"].lower() != best_language:
        mismatch = "!!"

    rows = []
    asset_data = [
        asset["asset_id"],
        asset["sgs_id"],
        asset["language_item_code"],
        mismatch,
        best_language,
        "",
        ",".join(all_languages),
        "",
        asset["title_original"],
        asset["title_public"]
    ]
    for index, entry in enumerate(entries):
        row = asset_data.copy()
        row.extend([
            entry["file_name"],
            "https://assets.swissgeol.ch/api/file/{}".format(entry["file_id"]),
            entry["file_language"],
            entry["file_additional_languages"],
            entry["file_language_details"]
        ])
        rows.append(row)

        asset_data = ["" for value in asset_data]

    return rows


with open("data/assets-language.csv", 'w', newline='') as summary_file:
    writer = csv.writer(summary_file, quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "asset_id",
        "sgs_id",
        "current_language",
        "mismatch",
        "identified_language",
        "corrected_language",
        "identified_additional_languages",
        "corrected_additional_languages",
        "original_title",
        "meta_title",
        "file_name",
        "link",
        "file_language",
        "file_additional_languages",
        "file_language_details"
    ])

    with open("data/files.csv", 'r', newline='') as asset_files_input:
        reader = csv.DictReader(asset_files_input)
        files = {}
        for row in reader:
            files[row["filename"]] = row

    with open("data/assets-db.csv", 'r', newline='') as assets_db_file:
        reader = csv.DictReader(assets_db_file)
        current_asset = None
        asset_entries = []

        for row in reader:
            if current_asset is not None:
                if current_asset["asset_id"] != row["asset_id"]:
                    writer.writerows(asset_rows(current_asset, asset_entries))
                    asset_entries = []

            filename = row["file_name"]
            if not filename.lower().endswith("_ldoc.pdf"):
                if filename in files:
                    new_data = files[row["file_name"]]
                    row["file_language"] = new_data["best_language"]
                    row["file_additional_languages"] = new_data["other_significant_languages"]
                    row["file_language_details"] = new_data["notes"]
                else:
                    row["file_language"] = ""
                    row["file_additional_languages"] = ""
                    row["file_language_details"] = ""
                current_asset = row
                asset_entries.append(row)

        writer.writerows(asset_rows(current_asset, asset_entries))
