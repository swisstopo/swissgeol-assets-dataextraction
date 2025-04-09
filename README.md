# Page Classification for Borehole and Maps in Assets

## **Purpose** 

The purpose of this project is to develop a method for classifying document pages as either **text-based** or **image-based** (e.g., borehole profiles, maps, text, title pages). This classification aims to support the structural analysis of reports within [Assets](assets.swissgeol.ch). By categorizing document pages, we aim to facilitate the identification of borehole profiles and maps in PDFs, ultimately linking extracted assets to boreholes effectively.

---

## **Objectives**

- **Extract raw text content** and determine its position on the page.
- **Develop an initial classification pipeline** for image-based pages using a stepwise approach:
    - **Text Positioning Analysis**: Identify potential image regions by analyzing text placement or density.
    - **Rule-Based Methods**: Use heuristic approaches, such as keyword detection, to distinguish different images/ drawings.

The classification process for scanned documents is executed in **`main.py`**.

---

### Data

### **Dataset Source**

The training dataset stored in the S3 bucket:`swisstopo-lillemor-haibach-workspaces-data` ->  `input/`

It contains the following subfolders:  

| Folder                      | Description                                                                                                                  |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `single_pages/maps/`        | Contains scanned **map pages**                                                                                               |
| `single_pages/boreprofile/` | Holds **borehole profile pages**                                                                                             |
| `single_pages/title_page/`  | Stores **title pages** from reports (no Summary sheet)                                                                       |
| `single_pages/text/`        | Includes **continuous text pages**                                                                                           |
| `single_pages/unknown/`     | Includes **pages not belonging to classes above**, f.e tables, mixed pages, graphs, Summary sheet (like Infogeol title pages |
| `reports/`                  | Short reports a lot of the single pages stem from. Ground truth available.                                                   |
| `reports_no_gt/`            | Longer reports, were no ground turth is avaibale, but interesting for exploration  longer reports)                           |

Additionally, boreprofile data from `zurich` and `geoquat/validation` in the repository **`swisstopo/swissgeol-boreholes-dataextraction`** can be used for classification.  

### **Ground Truth**  

A **ground truth dataset** was created for each category, including the boreprofile data from `zurich` and `geoquat/validation` in **`swisstopo/swissgeol-boreholes-dataextraction`**.  

- **Classification results** are stored in **`data/prediction.json`**.  
- **Ground truth files** are stored as **`data/gt_single_pages.json`** for all pdf files within subfoldes of `single_pages`. 
- For reports folder groundtruths are stored as **`data/get_reports.json`**. Not completely accurate!
- For reports_no_gt no groundtruths is available.

---

### Classification Structure

The classification currently follows a structured categorization:

1. **Text Pages (Fliesstext)** → Pages containing continuous flowing text.  
2. **Borehole Profiles** → Pages containing borehole profiles.  
3. **Maps** → Pages containing maps.  
4. **Title Pages** → Title Pages within report (not implemented). Summary sheets are not included
---

## Repository Structure

The structure of the repository is:
- data/
    - `input/`: folder with subfolders containing one type of pages (text, title_page, boreprofile, maps)
    -  `prediction.json`: classification results
    - `gt_{subfolder}.json`: groundtruth for each folder if it exists, including groundtruth for `zurich`and and `geoquat/validation` of the repository `swisstopo/swissgeol-boreholes-dataextraction`
    - `test/`: output folder for jupyter notebooks where images, drawings etc. get saved to.
- src/
    - includes python util scripts, needed for the page classification or other jupyter notebooks.
- tests/
    - unittests
- evaluation/
    - evaluation_metrics and per_page_comparisons get saved to here
- notebooks/
    - `create_testdata.ipynb`: notebook used to create single page input data for the classification. 
    - `classify_digital_page.ipynb`: Try out Notebook to classify digital pdf pages. Code might be reusable
    - `extract_images.ipynb`: Extracts images/ drawings from digitally born pdfs. Draws bounding boxes for text and drawings in pdfs. Reusable Code
    - `asset-notebook.ipynb`: Extracts Table of Content out odf pdfs. Might come in handy for detecting report structure..
    - `asset-notebook_S3.ipynb`: Retrieve and save all text for all files on the S3 bucket as .txt files
    - `layout_parser.ipynb`: Uses layout parser to identify layout of pages in pdf files.
    - `corner_detection.ipynb`: Followed tutorial for corner detection within image.
    - `pdf_type.ipynb`: naive approach to classifying pages into digitally or scanned pdfs.
- `main.py`: executes classification process 
- `matching_params.yml`: contains list of keywords for TOC, material description, boreprofiles.
- `.gitigore`
- `README.md`
- `requirements.txt`


## How to run classify_scanned_page.py
1. Install dependencies:
```
pip install -r requirements.txt
```
2. For tracking metrics for classification set environment variable `MLFLOW_TRACKING=True` and `MLFLOW_TRACKING_URI="http://localhost:5000"`, f.e in `.env` file. Run cli command ```mlflow ui```. 
3. Ensure the input directory exists: `data/input/single_pages/{maps, boreprofile, title_page, text}`
4. Specify the input directory/ file(--input_path, -i)  and optionally, the ground truth (--ground_truth_path, -g) file to run the classification of one pdf or whole directory including subdirectories:
```
python main.py --input_path replace/with/path --ground_truth_path replace/with/path.csv
```
Example
```
python main.py -i data/single_pages/  -g data/gt_single_pages.json
``` 


## Further infos on some of the Notebooks

### create_testdata.ipynb

This notebook is used to **create input pages** for `classify_scanned.py` from original reports.

**How it works**
1. **Place the original reports (PDFs) in** `data/input/`  
   - The folder should contain **PDF files** from which pages will be extracted.

2. **Specify the following parameters in the notebook**:  
   - `filename`: Name of the PDF file to extract pages from.  
   - `wanted_page`: The page number to extract.  
   - `out_dir`: The target directory where the extracted page will be stored.

3. **Set `out_dir` to one of the following categories**:  
   - `"boreprofile"`: For borehole profile pages.  
   - `"maps"`: For map pages.  
   - `"text"`: For continuous text pages.  
   - `"title_page"`: For title pages.  

4. **Extracted pages will be saved to** `data/input/{out_dir}/`  
   - Example: If `out_dir = "maps"`, the page will be saved in `data/input/maps/`.  

The extracted pages are used as **training data** for classification.

--- 
### classify_digital_page.ipynb
Classifies pages of digitally born PDFs by analyzing their content and categorizing each page into:
- **Text**: Pages containing only text.
- **Image**: Pages containing embedded images.
- **Drawing**: Pages containing technical drawings or diagrams.

**How It Works**
1. Specify the PDF file inside the notebook.
2. The file should be placed in the input folder `data/input/NAB/` (NAGRA reports)
3. The notebook analyzes each page and determines whether it contains text, images, or drawings.
4. The classification results are saved as `predictions.json` in the `data/` directory.

### extract_images.ipynb

Extracts images and drawings from digitally born documents (and from scanned pdfs if drawings are detected via OCR). It also visualizes the detected structures by drawing bounding boxes around:
- **Text Lines**
- **Text Blocks**
- **Drawings**

**How It Works**

1. Specify the input file name and path inside the notebook.
2. The notebook processes the document and extracts images and drawings
3. It clusters drawings and text and draws bounding boxes around detected elements to visualize behavior.
4. Saves results in `data/test/filename/`


TODO:
- asset-notebook.ipynb
- asset-notebook_S3.ipynb
- layout_parser.ipynb
- corner_detection.ipynb
- pdf_type.ipynb

