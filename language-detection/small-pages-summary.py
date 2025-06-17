import csv
import pandas as pd

with open("data/assets-db.csv", 'r', newline='') as assets_db_file:
    reader = csv.DictReader(assets_db_file)
    assets = {}
    for row in reader:
        assets[row["file_name"]] = row

with open("data/pages.csv", 'r', newline='') as pages_file:
    rows = list(csv.DictReader(pages_file))

page_count = {}

output = []
total_page_count = len(rows)

for row in rows:
    filename = row["filename"]
    page_number = int(row["page_number"])
    if filename not in page_count:
        page_count[filename] = page_number
    else:
        page_count[filename] = max(page_count[filename], page_number)

for row in rows:
    width = float(row["width"])
    height = float(row["height"])
    area = width * height
    filename = row["filename"]
    page_number = int(row["page_number"])

    if filename in assets:
        db_data = assets[filename]
        asset_id = db_data["asset_id"]
        sgs_id = db_data["sgs_id"]
    else:
        asset_id = ""
        sgs_id = ""

    if area < 100000:
        output.append([
            asset_id,
            sgs_id,
            filename,
            page_count[filename],
            page_number,
            "{:.2f}".format(width),
            "{:.2f}".format(height),
            "{:.2f}".format(area)
        ])

df = pd.DataFrame(
    output,
    columns=[
        "asset_id",
        "sgs_id",
        "file_name",
        "file_page_count",
        "page_number",
        "width",
        "height",
        "area"
    ]
)

df.to_csv("data/assets-small-pages.csv", index=False)

grouped_df = df.groupby(['file_name', 'sgs_id', 'file_page_count'], as_index=False).agg({
    'page_number': lambda x: list(x)
})
grouped_df.insert(3, 'small_page_count', grouped_df['page_number'].apply(len))
grouped_df.sort_values(by=['sgs_id', 'file_name'])
grouped_df.to_csv("data/assets-with-small-pages.csv", index=False)


unique_file_count = df["file_name"].nunique()
small_page_count = len(df)

print("Small pages: {} out of {}.".format(small_page_count, total_page_count))
print("{} unique assets and {} unique files affected.".format(
    df["asset_id"].nunique(),
    unique_file_count
))

grouped = df.groupby("file_name")
grouped_max = grouped.max()
grouped_max["count"] = grouped.size()

full_files = grouped_max[grouped_max["file_page_count"] == grouped.size()]
print("* {} files with only small pages ({} pages).".format(
    len(full_files),
    full_files["count"].sum()
))

majority_files = grouped_max[grouped.size().between(
    0.5 * grouped_max["file_page_count"],
    grouped_max["file_page_count"],
    inclusive="left"
)]
print("* {} files with 50%-99.9% small pages ({} pages).".format(
    len(majority_files),
    majority_files["count"].sum()
))

minority_files = grouped_max[grouped.size() < 0.5 * grouped_max["file_page_count"]]
print("* {} files with 0.1%-49.9% small pages ({} pages).".format(
    len(minority_files),
    minority_files["count"].sum()
))
