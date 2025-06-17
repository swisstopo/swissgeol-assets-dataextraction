## Workflow for language identification

### Execute `language-per-page-from-s3.py`

Prerequisite: download the [lid.176.bin model from Fasttext](https://fasttext.cc/docs/en/language-identification.html) to the `models/` directory.

The script:
* Iterates over all PDF files in an S3 Bucket, detects the language of each page using Fasttext.
* Produces a file `data/pages.csv` where language and other data (e.g. width, height) per page is saved.

### Execute `summarize-per-file.py`

This script reads `data/pages.csv` and produces a summary per PDF document that is written to `data/files.csv`.

### Obtain data from Assets application database

```sql
SELECT 
  asset.asset_id,
  sgs_id,
  language_item_code,
  "file".file_id,
  "file".file_name,
  "file".file_size,
  title_public,
  title_original
FROM asset
LEFT JOIN asset_file ON asset_file.asset_id = asset.asset_id
LEFT JOIN "file" ON "file".file_id = asset_file.file_id
ORDER BY sgs_id ASC
```

Save results as a CSV file to `data/assets-db.csv`.

### Execute `asset-language-summary.py`

Joins information from `data/files.csv` and `data/assets-db.csv` into a single file `data/assets-language.csv` that can be used for quality control of language detection results.

The columns in this file are as follows:
* `asset_id`
* `sgs_id`
* `current_language`: as read from the database dump.
* `mismatch`: contains `!!` when the current language is defined (not `other`) and different from the automatically identified language.
* `identified_language`: main language of the asset based on the results of the automatic language identification for all linked files.
* `corrected_language`: column for entering any necessary corrections during a quality control process
* `identified_additional_languages`: additional significant languages (comma-separated ISO 639-2 language codes), if any, for the asset based on the results of the automatic language identification
* `corrected_additional_languages`: column for entering any necessary corrections during a quality control process
* `original_title`: as read from the database dump.
* `meta_title`: as read from the database dump.
* `file_name`: note that a single asset can be linked with multiple files, in that case they will each appear on a separate row
* `link`: direct hyperlink to the PDF document on assets.swissgeol.ch (you need to be logged in before accessing the link)
* `file_language`: main language of the file based on the results of the automatic language identification
* `file_additional_languages`: additional significant languages (comma-separated ISO 639-2 language codes), if any, for the file based on the results of the automatic language identification
* `file_language_details`: a page-by-page overview of the different languages that were identified in the document, as well as any _title pages_.

## Notes / limitations

* Title pages ("Auszug aus dem Titelverzeichnis", "Belegblatt", "Page de garde", etc.) are ignored (as long as they are correctly recognized) when selecting the significant languages for a file. I.e. a French document with a German title page should not have German as the main nor as an additional language (e.g. SGS id 1034).
* Below are some known sources or incorrectly identified languages for individual pages. Mostly however, this only applies to a small number of pages in a longer document, so that the significant languages that are selected for the entire file are usually not affected.
  * Lists with literature references. E.g. when a French document contains a list of references to mostly German publications, then those pages might also be identified as German, even though the document itself is still in German (e.g. SGS id 35498 page 6).
  * Maps with place names. E.g. if a page in a German document contains a map with many French- or Italian-sounding place names that are picked up by the OCR, then this page might also be identified as French or Italian, even though the document itself is still in German.
* It is not entirely clear what we should do with documents that don't contain much text, but only maps, tables, profiles, graphs, etc. Those pages might still contain some headings and/or labels in a certain language, but is that enough to assign that language to the corresponding asset or not? Similarly, if a document contains a lot of German text, and then a few graphs/tables with French labels/headings, is that enough to add French as an additional significant language or not?
  * The Assets application currently also contains a language category "numerical", described as _"asset with numerical structure, e.g. programme code, configurations"_. Do we have clearer guidelines/examples for when an asset should be assigned to this class? Will this class disappear once we enable assigning more than one language to a single asset, or will it still be relevant?

## Other scripts

### `small-pages-summary.py`

Reads from `data/pages.csv` and `data/assets-db.csv` in order to identify asset files that contain "stamp-sized" pages for the ticket https://jira.swisstopo.ch/browse/LGD-307. The output is written to the files `data/assets-small-pages.csv` (one entry for each small page) and `data/assets-with-small-pages.csv` (small pages grouped by file).

### `single-page-assets.py`

Reads from `data/pages.csv` to identify asset files that only contain a single page, then downloads those files from S3 and extracts text from that single page. The aim is to identify asset files that only contain a note such as "Kein Mikrofil vorhanden -> Bibliothek GLA" for the ticket https://jira.swisstopo.ch/browse/LGD-312. The output is written to the file `data/assets-single-pages.csv`.

### `compare_pages.py`

Reads from two different `pages.csv` files and compares the results, to check if there are pages that are missing from one file compared to another, pages that have a different size, pages where the text type of the language has changed, etc. This can be used to verify the results of a new OCR method, by comparing that pages data from before and after applying the OCR.
