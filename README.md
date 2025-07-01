# Page Classification for Geological Documents in Assets

## **Purpose** 

This repository provides a classification pipeline to categorize PDF pages 
from geological reports into document classes, with the goal of supporting document
understanding and metadata extraction in the [Assets](https://assets.swissgeol.ch/) platform.

This classification helps to map individual pages in a document,
which ultimately should facilitate the identification of borehole profiles and maps in PDFs to link between documents on [Assets](https://assets.swissgeol.ch/) and boreprofiles on [Boreholes](https://boreholes.swissgeol.ch/).

## Classes

Each page is categorized into one of the following:

1. **Text Pages** - Continuous flowing text.  
2. **Borehole Profiles** - Boreholes, longitudinal profiles, and dynamic probing logs.  
3. **Maps** - Geological or topographic maps.  
4. **Title Pages** - Report cover/title pages (currently not fully implemented).  
5. **Unknown** - Other content like tables, mixed pages, graphs, and summary sheets.

## Data

### Dataset Source

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

## Repository Structure

- `data/`
    - `single_pages/`: Input data split by class
    -  `prediction.json`: Output predictions
    - `gt_*.json`: Ground truth files
    - `test/`: Output visualization from notebooks
- `evaluation/`: Evaluation results and metrics
- `src/`: Utility scripts and core logic
- `tests/`: Unit tests
- `language-detection`: Language detection module

- `main.py`: Entry point for classification
- `matching_params.yml`: Keywords for classification/matching
- `requirements.txt`
- `setup.py`
- `README.md`

## How to run Classifier

1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Configure environment variables and start Mlflow logging (optional)
```bash
mlflow ui
```
4. Run the classification:
```bash
python main.py -i <input_path> -g <ground_truth_path>
```
**Example**
```bash
python main.py -i data/single_pages/ -g data/gt_single_pages.json
```

## Output Format
The script processes PDF pages and outputs predictions in `data/predictions.json`. 
The output is structured as a list of classification results per page per report.
#### Example Output
```json
[
  {
    "filename": "1799_3.pdf",
    "classification": [
      {
        "Page": 1,
        "Text": 0,
        "Boreprofile": 0,
        "Maps": 1,
        "Title_Page": 0,
        "Unknown": 0
      },
      {
        "Page": 2,
        "Text": 1,
        "Boreprofile": 0,
        "Maps": 0,
        "Title_Page": 0,
        "Unknown": 0
      }
    ]
  },
  {
    "filename": "1800_1.pdf",
    "classification": [
      {
        "Page": 1,
        "Text": 0,
        "Boreprofile": 1,
        "Maps": 0,
        "Title_Page": 0,
        "Unknown": 0
      }
    ]
  }
]

```
**Explanation**:
- filename: The name of the processed PDF file.
- classification: A list of dictionaries, each representing the classification of  a PDF page of the report.
  - Each dictionary contains:
  - Page: The page number (1-indexed). 
  - One key per possible class (e.g., Text, Boreprofile, Maps, Title_Page, Unknown) with binary values:
    - 1: class was assigned to the page. 
    - 0: class was not assigned.

**Further Notes:**
- The classifier supports batch input of multiple reports.
- Output is returned as a standard Python list of dictionaries and can be serialized directly as JSON.
- Input must be preprocessed: PDFs should already have OCR. 
- Classification is currently multi-class with a single label per page. Future updates may support multiple-labels.

## Language-Detection

This module performs language detection on input documents.
It is intended to be integrated into the classification pipline in a later stage.
For execution details, please refer to [language-detection/README.md](README.md).