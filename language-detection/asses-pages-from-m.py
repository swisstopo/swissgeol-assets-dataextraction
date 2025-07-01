import os
import fitz
import csv
import re

directory = r"M:\Appl\DATA\GD\landesgeologie\lgAssets\assetsNatRel4Cloud"
csvPath = "assetsNatRel4Cloud-pages.csv"


def text_type(page: fitz.Page) -> str:
    bboxes = page.get_bboxlog()
    has_ignore_text = False

    for boxType, rectangle in bboxes:
        # Empty rectangle that should be ignored occurs sometimes, e.g. SwissGeol 44191 page 37.
        if (boxType == "fill-text" or boxType == "stroke-text") and not fitz.Rect(rectangle).is_empty:
            return "digital"
        if boxType == "ignore-text":
            has_ignore_text = True

    if has_ignore_text:
        return "ocr"

    else:
        return "none"


if os.path.exists(csvPath):
    with open(csvPath) as file:
        lines = list(csv.reader(file))
        last_filename = lines[-1][0]
        last_page_number = int(lines[-1][1])
        print("Last entry in csv file: {} page {}.".format(last_filename, last_page_number))
else:
    last_filename = None
    last_page_number = None

with open(csvPath, "a", newline="") as pages_file:
    pages_writer = csv.writer(pages_file, quoting=csv.QUOTE_MINIMAL)
    if last_filename is None:
        pages_writer.writerow(
            [
                "filename",
                "page_number",
                "character_count",
                "word_count_not_short",
                "ocr_type",
                "width",
                "height",
                "mediabox_width",
                "mediabox_height",
                "rotation",
            ]
        )

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".pdf"):
                # if last_filename is not None and last_filename != filename:
                #     continue
                #
                # print()
                # print(filename)

                if filename in ["32669.pdf", "32670.pdf", "32672.pdf", "32673.pdf"]:
                    print(filename)
                    with fitz.Document(os.path.join(root, filename)) as doc:
                        for page_index, page in enumerate(doc):
                            page_number = page_index + 1

                            if last_page_number is not None and page_number <= last_page_number:
                                continue

                            print("Page {}".format(page_number))
                            text = page.get_text()
                            character_count = len(text)
                            word_count_not_short = len(re.findall(r"[^\W\d_]{5,}", text))

                            pages_writer.writerow(
                                [
                                    filename,
                                    page_number,
                                    character_count,
                                    word_count_not_short,
                                    text_type(page),
                                    page.rect.width,
                                    page.rect.height,
                                    page.mediabox.width,
                                    page.mediabox.height,
                                    page.rotation,
                                ]
                            )
                            pages_file.flush()

                last_filename = None
                last_page_number = None
