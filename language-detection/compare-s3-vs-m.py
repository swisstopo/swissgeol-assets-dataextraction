import csv
import math
import pandas as pd
import re


data = {}
with open("data/pages_bu_20240207.csv", "r", newline="") as pages_file:
    reader = csv.DictReader(pages_file)
    files_s3 = {}
    for row in reader:
        filename = row["filename"]
        if not filename.startswith("a44376_"):
            filename = re.sub("^a[0-9]+_", "", filename)

        if not filename.endswith("_LDoc.pdf"):
            if filename not in files_s3:
                files_s3[filename] = []
            files_s3[filename].append(row)

    for filename, file_pages in files_s3.items():
        for i, entry in enumerate(file_pages):
            pass

        if filename not in data:
            data[filename] = {"page_count_local": 0, "page_count_s3": 0}

        data[filename]["page_count_s3"] = len(file_pages)

with open("data/assetsNatRel4Cloud-pages.csv", "r", newline="") as pages_file:
    reader = csv.DictReader(pages_file)
    files_local = {}
    for row in reader:
        filename = row["filename"]
        if filename not in files_local:
            files_local[filename] = []
        files_local[filename].append(row)

    for filename, file_pages in files_local.items():
        for i, entry in enumerate(file_pages):
            pass

        if filename not in data:
            data[filename] = {"page_count_local": 0, "page_count_s3": 0}

        data[filename]["page_count_local"] = len(file_pages)

output = []
for filename, entry in data.items():
    output.append([filename, entry["page_count_local"], entry["page_count_s3"]])

df = pd.DataFrame(output, columns=["filename", "page_count_local", "page_count_s3"])
df.to_csv("data/compare-s3-vs-local.csv", index=False)

print(df[df["page_count_local"] > df["page_count_s3"]])
print(df[df["page_count_local"] < df["page_count_s3"]])

equal_df = df[df["page_count_local"] == df["page_count_s3"]].copy()


def count(fn):
    return lambda filename: sum(
        fn(entry1, entry2) for entry1, entry2 in zip(files_local[filename], files_s3[filename])
    )


def equal(entry1, entry2):
    return math.isclose(float(entry1["width"]), float(entry2["width"]), rel_tol=0.01) and math.isclose(
        float(entry1["height"]), float(entry2["height"]), rel_tol=0.01
    )


def mediabox(entry1, entry2):
    return not equal(entry1, entry2) and (
        math.isclose(float(entry1["mediabox_width"]), float(entry2["width"]), rel_tol=0.01)
        and math.isclose(float(entry1["mediabox_height"]), float(entry2["height"]), rel_tol=0.01)
    )


def mediabox_rot(entry1, entry2):
    return (
        not equal(entry1, entry2)
        and not mediabox(entry1, entry2)
        and (
            math.isclose(float(entry1["mediabox_width"]), float(entry2["height"]), rel_tol=0.01)
            and math.isclose(float(entry1["mediabox_height"]), float(entry2["width"]), rel_tol=0.01)
        )
    )


def x20(entry1, entry2):
    return math.isclose(20 * float(entry1["width"]), float(entry2["width"]), rel_tol=0.01) and math.isclose(
        20 * float(entry1["height"]), float(entry2["height"]), rel_tol=0.01
    )


def x20_rot(entry1, entry2):
    return not x20(entry1, entry2) and (
        math.isclose(20 * float(entry1["width"]), float(entry2["height"]), rel_tol=0.01)
        and math.isclose(20 * float(entry1["height"]), float(entry2["width"]), rel_tol=0.01)
    )


def x20_media(entry1, entry2):
    return (
        not x20(entry1, entry2)
        and not x20_rot(entry1, entry2)
        and (
            math.isclose(20 * float(entry1["mediabox_width"]), float(entry2["width"]), rel_tol=0.01)
            and math.isclose(20 * float(entry1["mediabox_height"]), float(entry2["height"]), rel_tol=0.01)
        )
    )


def x20_media_rot(entry1, entry2):
    return (
        not x20(entry1, entry2)
        and not x20_rot(entry1, entry2)
        and not x20_media(entry1, entry2)
        and (
            math.isclose(20 * float(entry1["mediabox_width"]), float(entry2["height"]), rel_tol=0.01)
            and math.isclose(20 * float(entry1["mediabox_height"]), float(entry2["width"]), rel_tol=0.01)
        )
    )


def mismatch(entry1, entry2):
    return not (
        equal(entry1, entry2)
        or mediabox(entry1, entry2)
        or mediabox_rot(entry1, entry2)
        or x20(entry1, entry2)
        or x20_rot(entry1, entry2)
        or x20_media(entry1, entry2)
        or x20_media_rot(entry1, entry2)
    )


equal_df["equal_size"] = equal_df["filename"].apply(count(equal))
equal_df["mediabox"] = equal_df["filename"].apply(count(mediabox))
equal_df["mediabox_rot"] = equal_df["filename"].apply(count(mediabox_rot))
equal_df["x20_size"] = equal_df["filename"].apply(count(x20))
equal_df["x20_size_rot"] = equal_df["filename"].apply(count(x20_rot))
equal_df["x20_size_media"] = equal_df["filename"].apply(count(x20_media))
equal_df["x20_size_media_rot"] = equal_df["filename"].apply(count(x20_media_rot))
equal_df["mismatch"] = equal_df["filename"].apply(count(mismatch))

print(
    "{} total pages ({} assets)".format(equal_df["page_count_local"].sum(), (equal_df["page_count_local"] > 0).sum())
)
print("{} equal size pages (in {} assets)".format(equal_df["equal_size"].sum(), (equal_df["equal_size"] > 0).sum()))
print("{} mediabox resized pages (in {} assets)".format(equal_df["mediabox"].sum(), (equal_df["mediabox"] > 0).sum()))
print(
    "{} mediabox resized pages, rotated (in {} assets)".format(
        equal_df["mediabox_rot"].sum(), (equal_df["mediabox_rot"] > 0).sum()
    )
)
print("{} x20 size pages (in {} assets)".format(equal_df["x20_size"].sum(), (equal_df["x20_size"] > 0).sum()))
print(
    "{} x20 size pages, rotated (in {} assets)".format(
        equal_df["x20_size_rot"].sum(), (equal_df["x20_size_rot"] > 0).sum()
    )
)
print(
    "{} x20 size pages from mediabox (in {} assets)".format(
        equal_df["x20_size_media"].sum(), (equal_df["x20_size_media"] > 0).sum()
    )
)

print(
    "{} x20 size pages from mediabox, rotated (in {} assets)".format(
        equal_df["x20_size_media_rot"].sum(), (equal_df["x20_size_media_rot"] > 0).sum()
    )
)
print("{} mismatch pages (in {} assets)".format(equal_df["mismatch"].sum(), (equal_df["mismatch"] > 0).sum()))


print()

data = equal_df["x20_size"] + equal_df["x20_size_rot"] + equal_df["x20_size_media"] + equal_df["x20_size_media_rot"]
print("{} resized pages (in {} assets)".format(data.sum(), (data > 0).sum()))

data = equal_df["mediabox_rot"] + equal_df["mediabox"] + equal_df["x20_size_media"] + equal_df["x20_size_media_rot"]
print("{} with incorrect crop (in {} assets)".format(data.sum(), (data > 0).sum()))
