import boto3
import os
import fitz
from fasttext.FastText import _FastText
import csv
import re
import title_page
from ocrdetection import text_type

s3_input = "asset/asset_files_new_ocr/"
s3_aws_profile = "swissgeol-prod"
s3_session = boto3.Session(profile_name=s3_aws_profile)
s3 = s3_session.resource("s3")
bucket = s3.Bucket("swissgeol-assets-swisstopo")


def key_to_filename(key):
    return key.split("/")[-1]


objs = list(bucket.objects.filter(Prefix=s3_input))

detector = _FastText("models/lid.176.bin")  # cf. https://github.com/facebookresearch/fastText/issues/1056


with open("data/pages.csv", "w", newline="") as pages_file:
    pages_writer = csv.writer(pages_file, quoting=csv.QUOTE_MINIMAL)
    pages_writer.writerow(
        [
            "filename",
            "page_number",
            "language_code",
            "character_count",
            "word_count_not_short",
            "ocr_type",
            "title_page_type",
            "width",
            "height",
            "repeated_text",
        ]
    )
    for obj in objs:
        if obj.size:
            filename = key_to_filename(obj.key)
            # if filename != '42344.pdf':
            #    continue
            if filename.endswith(".pdf"):
                tmp_file_path = os.path.join("tmp", filename)
                bucket.download_file(obj.key, tmp_file_path)

                print()
                print(obj.key)

                with fitz.Document(tmp_file_path) as doc:
                    for page_index, page in enumerate(doc):
                        page_number = page_index + 1
                        # if page_number != 24:
                        #     continue
                        print("Page {}".format(page_number))
                        text = page.get_text()
                        character_count = len(text)
                        word_count_not_short = len(re.findall(r"[^\W\d_]{5,}", text))

                        repeated_text = ""
                        half_length = int(len(text) / 2)
                        if len(text) > 5 and text[:half_length] == text[-half_length:]:
                            repeated_text = "repeat"
                            print("repeat")

                        # Ignore lines with at most four letters, because they often cause false positives (e.g.
                        # many occurrences of "Ng" as a label on a graph, might cause the language identification to
                        # return Vietnamese).
                        # Also, Fasttext works on single lines of text only, so we convert newlines to a regular space.
                        text_for_detection = " ".join(
                            [line for line in text.split("\n") if len([char for char in line if char.isalpha()]) > 4]
                        )
                        # We ignore single-character words, because they cause too many false positives. For example,
                        # the OCR might misread zeroes as the letter O, and then language identification might pick
                        # Portuguese, because "o" is an article in the Portuguese language. In doing this, we might
                        # lose some valuable data as well (words such a "a"/"I" in English or "à"/"y" in French), but
                        # generally the benefits outweigh the disadvantages.
                        text_for_detection = " ".join(
                            [token for token in re.split(r"\s+", text_for_detection) if len(token) > 1]
                        )
                        # Remove stuff between whitespace that doesn't contain any regular letter. Removes numeric
                        # data, as this tends to confuse the Fasttext language identification model. This also
                        # removes "Unicode junk" that sometimes appears in digitally-born PDFs such as 44165.pdf.
                        text_for_detection = re.sub(r"(^|\s)[^a-zA-Zéàèöäüç]+(?=\s|$)", " ", text_for_detection)
                        # Ignore any token with digits.
                        text_for_detection = re.sub(r"(^|\s)\S*[0-9]\S*(?=\s|$)", " ", text_for_detection)
                        # The Fasttext language identification model does not work well with all-uppercase text (cf.
                        # e.g. https://github.com/UKPLab/EasyNMT/issues/2). If there are more uppercase characters than
                        # lowercase characters, then we convert everything to lowercase. In doing this, we might lose
                        # some valuable information (e.g. capitalisation of nouns in German), but generally, the
                        # benefits seem to outweigh the disadvantages.
                        uppercase_count = sum(1 for c in text_for_detection if c.isupper())
                        lowercase_count = sum(1 for c in text_for_detection if c.islower())
                        if uppercase_count > lowercase_count:
                            text_for_detection = text_for_detection.lower()

                        language_code = ""

                        # Fasttext tends to give random results on very short input (e.g. "kanal kanal" as Cebuano).
                        if word_count_not_short >= 4:
                            labels, scores = detector.predict(text_for_detection.lower(), k=5)

                            # print(text_for_detection)
                            # print(labels)
                            # print(scores)

                            if len(labels) and len(scores):
                                score = scores[0]

                                if score > 0.7:
                                    language_code = labels[0].replace("__label__", "")

                        title_page_type = title_page.title_page_type(text) or ""

                        pages_writer.writerow(
                            [
                                filename,
                                page_number,
                                language_code,
                                character_count,
                                word_count_not_short,
                                text_type(page),
                                title_page_type,
                                page.rect.width,
                                page.rect.height,
                                repeated_text,
                            ]
                        )

                os.remove(tmp_file_path)
