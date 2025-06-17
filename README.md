# Page Classification for Geological Documents in Assets

## **Purpose** 

This repository provides a classification pipeline to categorize PDF pages 
from geological reports into document classes, with the goal of supporting document
understanding and metadata extraction in the [Assets](https://assets.swissgeol.ch/) platform.

This classification helps map individual pages in a document to classes such as borehole profiles or maps,
which should facilitate the identification of borehole profiles and maps in PDFs, ultimately linking documents on [Assets](https://assets.swissgeol.ch/) with boreprofiles on [Boreholes](https://boreholes.swissgeol.ch/).
---
## Classes

Each page is categorized into one of the following:

1. **Text Pages** — Continuous flowing text.  
2. **Borehole Profiles** — Boreholes, longitudinal profiles, and dynamic probing logs.  
3. **Maps** — Geological or topographic maps.  
4. **Title Pages** — Report cover/title pages (currently not fully implemented).  
5. **Unknown** — Other content like tables, mixed pages, graphs, and summary sheets.

---

## 📁 Data

### 📦 Dataset Source

The dataset is stored in the S3 bucket `stijnvermeeren-assets-data`, under the `single_pages/` folder. It contains categorized subfolders:

| Folder         | Description                                                                                             |
|----------------|---------------------------------------------------------------------------------------------------------|
| `maps/`        | Pages with maps                                                                                         |
| `boreprofile/` | Borehole-related profiles (e.g., boreholes, longitudinal profiles, probing logs)                        |
| `title_page/`  | Cover/title pages from reports                                                                          |
| `text/`        | Continuous text pages                                                                                   |
| `unknown/`     | Other pages: tables, graphs, mixed layouts, summary sheets (e.g. Infogeol summary pages)                |

In addition, boreprofile data from the `zurich` and `geoquat/validation` folders used in the [swissgeol-boreholes-dataextraction](https://github.com/swisstopo/swissgeol-boreholes-dataextraction) repository and stored in the S3 bucket `stijnvermeeren-boreholes-data` can be classified and compared using existing ground truth.

### Ground Truth

- Classification output: `data/prediction.json`  
- Single-page ground truths: `data/gt_single_pages.json`  
- External evaluation sets:
  - Zurich: `data/gt_zurich.json`
  - GeoQuat: `data/gt_geoquat.json`

---

## Repository Structure

The structure of the repository is:
- data/
    - `single_pages/`: folder with subfolders containing one pagetype per folder.
    -  `prediction.json`: classification results
    - `gt_single_pages.json`: groundtruth for each folder if it exists, including groundtruth for `zurich`and and `geoquat/validation` of the repository `swisstopo/swissgeol-boreholes-dataextraction`
    - `test/`: output folder for jupyter notebooks where images, drawings etc. get saved to.
- src/
    - includes scripts, needed for the page classification.
- tests/
    - unittests
- evaluation/
    - evaluation metrics and per page comparisons get saved to here
- notebooks/
    - notebooks for exploration.
    
- `main.py`: executes classification process 
- `matching_params.yml`: contains list of keywords for TOC, material description, boreprofiles.
- `.gitigore`
- `README.md`
- `requirements.txt`
- setup.py

## How to run Classification
1. create virtual environment
2. Install dependencies:
```
pip install -r requirements.txt
```
3. For tracking metrics for classification set environment variable `MLFLOW_TRACKING=True` and `MLFLOW_TRACKING_URI="http://localhost:5000"`, f.e in `.env` file. Run cli command:
```
mlflow ui
```
4. Ensure the input directory exists: `data/single_pages/{maps, boreprofile, title_page, text, unknown}`
5. Specify the input directory/ file(--input_path, -i)  and optionally, the ground truth (--ground_truth_path, -g) file to run the classification of one pdf, folder or whole folder including subfolders:
```
python main.py --input_path replace/with/path --ground_truth_path replace/with/path.json
```
Example
```
python main.py -i data/single_pages/  -g data/gt_single_pages.json
```

## Further infos to the Notebooks

If you want to run the jupyter notebooks, you need to install:
```
pip install -e .
```

The Notebooks are mainly used for exploration.

### extract_images.ipynb

Extracts images and drawings from digitally born documents (and from scanned pdfs if drawings are detected via OCR). It visualizes the detected structures by drawing bounding boxes around:
- **Text Lines**
- **Text Blocks**
- **Drawings**

**How It Works**

1. Specify the input file name and path inside the notebook.
2. The notebook processes the document and extracts images and drawings
3. It clusters drawings and text and draws bounding boxes around detected elements to visualize behavior.
4. Saves results in `data/test/filename/`