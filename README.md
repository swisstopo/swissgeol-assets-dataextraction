# Page Classification for Borehole and Maps in Assets

## **Purpose** 

The purpose of this project is to develop a method for classifying document pages as either **text-based** or **image-based** (e.g., borehole profiles, maps, text, title pages). This classification aims to support the structural analysis of reports within [Assets](assets.swissgeol.ch). By categorizing document pages, we aim to facilitate the identification of borehole profiles and maps in PDFs, ultimately linking extracted assets to boreholes effectively.

---

## **Objectives**

- **Extract raw text content** and determine its position on the page.
- **Develop an initial classification pipeline** for image-based pages using a stepwise approach:
    - **Text Positioning Analysis**: Identify potential image regions by analyzing text placement or density.
    - **Rule-Based Methods**: Use heuristic approaches, such as keyword detection, to distinguish different images/ drawings.

The classification process for scanned documents is executed in **`src/main.py`** and handled in .

---

### Data

### **Dataset Source**

The training dataset stored in the S3 bucket:`swisstopo-lillemor-haibach-workspaces-data` ->  `input/`

It contains the following subfolders:  

| Folder      | Description |
|-------------|-------------|
| `maps/`     | Contains scanned **map pages** |
| `boreprofile/` | Holds **borehole profile pages** |
| `title_page/` | Stores **title pages** from Infogeol reports |
| `text/`     | Includes **continuous text pages** |

Additionally, boreprofile data from `zurich` and `geoquat/validation` in the repository **`swisstopo/swissgeol-boreholes-dataextraction`** can be used for classification.  

### **Ground Truth**  

A **ground truth dataset** was created for each category, including the boreprofile data from `zurich` and `geoquat/validation` in **`swisstopo/swissgeol-boreholes-dataextraction`**.  

- **Classification results** are stored in **`data/classification_results.csv`**.  
- **Ground truth files** are stored as **`data/groundtruth_{subfolder}.csv`**.

---

### Classification Structure

The classification currently follows a structured categorization:


1. **Text Pages (Fliesstext)** → Pages containing continuous flowing text.  
2. **Borehole Profiles** → Pages containing borehole profiles.  
3. **Maps** → Pages containing maps.  
4. **Title Pages** → Infogeol title pages (pre-defined classification in the `language-detection` repository).  

---

## Repository Structure

The structure of the repository is:
- data/
    - `input/`: folder with subfolders containing one type of pages (text, title_page, boreprofile, maps)
    -  `classtification_results.csv`: classification results
    - `groundtruth_{subfolder}.csv`: groundtruth for each subfolder including groundtruth for `zurich`and and `geoquat/validation` of the repository `swisstopo/swissgeol-boreholes-dataextraction`
    - `NAB/` : folder containing NAGRA reports, digitally born
    - `test/`: output folder images, drawings etc get saved to.
- src/
    - includes python util scripts, needed for the execution of the classification or other jupyter notebooks
- notebooks/
    - `classify_scanned_page.ipynb`: classifies scanned pdfs into text, maps, boreprofile, infogeol title page
    - `create_testdata.ipynb`: notebook used to create input data for `classify_scanned_page.ipynb`. Takes 
    - `classify_digital_page.ipynb`:
    - `extract_images.ipynb`: Extracts images/ drawings from digitally boren pdfs. Draws boudning boxes for text, and drawings for pdfs.
    - `asset-notebook.ipynb`: Extracts Table of Content out odf pdfs.
    - `asset-notebook_S3.ipynb`: Retrieve and save all text for all files on the S3 bucket as .txt files
    - `layout_parser.ipynb`: Uses layout parser to identiy layout of pages in pdf files
    - `corner_detection.ipynb`: Followed tutorial for corner detection within image.
    - `pdf_type.ipynb`: naiv approach to classifying pages into digitally or scanned pdfs.
    - `find_documents.ipynb`: finds all files with certain keyword in all subfolders of base directory. It also extends legaldocs futher described in [Legal Docs](https://ltwiki.adr.admin.ch:8443/pages/viewpage.action?pageId=637241440&spaceKey=LG&title=Legal%2BDocs).

- `matching_params.yml`: contains list of keywords for TOC
- `.gitigore`
- `README.md`

## How to run classify_scanned_page.ipynb
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Ensure the input directory exists: `data/input/{maps, boreprofile, title_page, text}`
3. Specify the input directory, output file, and optionally, the ground truth file to run the classification:
```
python src/main.py --input_dir replace/with/path --output_file replace/with/path.csv --ground_truth_path replace/with/path.csv
```

Example:
```
python src/main.py --input_dir data/input/maps --output_dir data/ --ground_truth_path data/ground_truth_maps.csv
```

## Further infos on some of the Notebooks

### create_testdata.ipynb

This notebook is used to **create input pages** for `classify_scanned_page.ipynb` from original reports.

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
3. It clusters drawings and text and draws bounding boxes around detected elements to visualize  behavior.
4. Saves results in `data/test/filename/`


TODO:
- asset-notebook.ipynb
- asset-notebook_S3.ipynb
- layout_parser.ipynb
- corner_detection.ipynb
- pdf_type.ipynb
- find_documents.ipynb

