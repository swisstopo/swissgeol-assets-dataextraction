The purpose of this project is to develop a method for classifying document pages as either text-based or image-based (e.g., borehole profiles, maps, graphs, tables). This classification aims to support the structural analysis of reports within Assets. By categorizing document pages, we aim to facilitate the identification of borehole profiles and maps in PDFs, ultimately linking extracted assets to boreholes effectively.

## Objectives

- Extract raw text content and determine its position on the page.
- Develop an initial classification pipeline for image-based pages using a stepwise approach:
    - Text Positioning Analysis: Identify potential image regions by analyzing text placement.
    - Rule-Based Methods: Use heuristic approaches, such as keyword detection, to distinguish different image types.

The classification process is handled in `classify_scanned_page.ipynb`.


### Data
A subfolder in local `data/inputs/` directory is needed as an input directory.

The dataset used for training stored in the S3 bucket `swisstopo-lillemor-haibach-workspaces-data` in the `input` folder, which contains the following subfolders:

- maps: Contains scanned map pages.
- boreprofile: Holds borehole profile pagestitle_page: Stores title pages from Infogeol reports.
- text: Includes continuous text pages.

### Classification Structure

The classification currently follows a structured categorization:

- Text Pages (Fliesstext): Pages containing continuous flowing text.
- Borehole Profiles: Pages containing structured borehole profile images.
- Maps: Pages featuring geographical or technical maps.
- Title Pages: Infogeol title pages, where classification has been pre-defined in the `language-detection` repository.