import csv
import pandas as pd
import boto3
import os
import fitz

s3_input = "asset/asset_files_new_ocr/"
s3_aws_profile = "s3-assets"
s3_session = boto3.Session(profile_name=s3_aws_profile)
s3 = s3_session.resource("s3")
bucket = s3.Bucket("swissgeol-assets-swisstopo")

objs = list(bucket.objects.filter(Prefix=s3_input))

with open("data/pages.csv", "r", newline="") as assets_db_file:
    reader = csv.DictReader(assets_db_file)
    asset_last_page = {}
    for row in reader:
        asset_last_page[row["filename"]] = row

output = []

for filename, last_page in asset_last_page.items():
    if int(last_page["page_number"]) == 1 and not filename.lower().endswith("_ldoc.pdf"):
        print(filename)

        tmp_file_path = os.path.join("tmp", filename)
        key = "{}{}".format(s3_input, filename)
        bucket.download_file(key, tmp_file_path)

        with fitz.Document(tmp_file_path) as doc:
            text = doc.get_page_text(0).replace("\n", " ")

        if len(text) < 200:
            filesize = os.path.getsize(tmp_file_path)

            output.append([filename, last_page["width"], last_page["height"], filesize, text])

            os.remove(tmp_file_path)


df = pd.DataFrame(output, columns=["filename", "width", "height", "filesize", "text"])

df.to_csv("data/assets-single-pages.csv", index=False)
