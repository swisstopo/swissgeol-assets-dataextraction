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

- `config/`: YAML configuration files for models and keyword matching
- `data/`
    - `single_pages/`: Input data split by class
    - `single_pages_split/: `Training/validation split created by `split_data.py`
    -  `prediction.json`: Output predictions
    - `gt_*.json`: Ground truth files
- `evaluation/`: Evaluation outputs, metrics and visualization
- `language-detection`: Language detection module
- `models/`: Trained model folders (e.g. LayoutLMv3, TreeBased)
- `prompts/`: Prompt templates for pixtral classification
- `src/`: Utility scripts and core logic 
- `tests/`: Unit tests
- `main.py`: Entry point for classification
- `pyproject.toml`
- `setup.py`
- `.env.template`: Template for .env file
- `README.md`

## How to run Classifier

### 1. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install .
```
For development, install optional tools with:
```bash
pip install '.[deep-learning,test,lint,experiment-tracking]'
```
Make sure you have `fasttext-predict` installed instead of `fasttext` (see 6. Setup FastText Language Detection).

### 3. Copy the provided environment variable template and specify your paths:
```bash
cp .env.template .env
```
For development and if you want to track your experiments, set `MLFLOW_TRACKING=True` in `.env` file.
### 4. (Optional) Start the MLflow UI

For development: Start MLflow UI (optional, for experiment tracking):
```
mlflow ui
```
### 5. (Optional) Use a pre-trained model:
- Option A: Download a pre-trained model from the [S3 bucket: stijnvermeeren-assets-data ](https://eu-central-1.console.aws.amazon.com/s3/buckets/stijnvermeeren-assets-data?region=eu-central-1&bucketType=general&tab=objects).
- Option B: Train your own model as described in [Train your Model](#train-your-model).

### 6. Setup FastText Language Detection

This project uses [fasttext-predict](https://github.com/searxng/fasttext-predict/),  a lightweight, dependency-free wrapper exposing only the predict method.
We use this. because [FastText](https://github.com/facebookresearch/fastText) is archived.
Download the FastText language identification model lid.176.bin form this [Website](https://fasttext.cc/docs/en/language-identification.html):
```
mkdir -p models/FastText
curl -o models/FastText/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```
In your `.env` file, set  the `FASTTEXT_MODEL_PATH` variable to your model path

8. Run the classification:
```bash
python main.py -i <input_path> -g <ground_truth_path> -c <classifier_name> 
```
If no classifier is specified, the baseline classifier is used by default.
If classifier is `layoutlmv3` or `treebased`, `--model_path` must be specified to locate the trained model.

| Classifier Name | Description                                                                   |
|------------------|-------------------------------------------------------------------------------|
| `baseline`       | Default. Rule-based classifier using layout, keyword matching, and heuristics |
| `pixtral`        | Uses the Pixtral Large via Amazon Bedrock to classify PDF pages               |
| `layoutlmv3`     | Transformer model (pretrained or fine-tuned LayoutLMv3) |
|`treebased` | Feature-based model (RandomForest or XGBoost)|

**Example**
```bash
python main.py -i data/single_pages/ -g data/gt_single_pages.json -c baseline
```

## Train your Model
To train your own model, first split your dataset into training and validation:
```bash
python scripts/split_data.py
```
This will create:
```bash
data/single_pages_split/train/
data/single_pages_split/val/
```
You can then train either:
- `layoutlmv3`: Fine tuning the layoulmv3 model
- `treebased`: Train a RandomForest or XGBoost model

Training logs and metrics will be tracked via MLflow.

### Train LayoutLMv3

To train a LayoutLMv3 model, run the training script directly:
```bash
python src.models.layoutlmv3.train.py
    --config_file_path config/layoutlmv3_config.yml
    --out_directory models/layoutlmv3_output 
    [--model_checkpoint models/layoutlmv3_pretrained_checkpoint]
```
**Arguments**:
- config_file_path: Path to the YAML configuration file with model parameters and dataset paths.
- out_directory: Directory where the trained model will be saved.
- model_checkpoint (optional): Path to a pre-trained model checkpoint. If not provided, the model will be initialized from the Hugging Face hub based on the config.

This training script supports freezing/unfreezing specific layers and uses the Hugging Face Trainer API under the hood.

### Train TreeBased Models (RandomForest or XGBoost)
To train a RandomForest or XGBoost classifier, use:
```bash
python src.models.treebased.train.py \
    --config_file_path config/xgboost_config.yml \
    --out_directory models/xgboost_model
```
- config_file_path: Path to the YAML config specifying hyperparameters and feature extraction settings.
- out_directory: Output path for the trained model.

If you're training an XGBoost model on macOS, you may encounter issues related to OpenMP. To resolve this, install the OpenMP library using Homebrew:
```bash
brew install libomp
```
## Pre-Commit
We use pre-commit hooks to format our code in a unified way.

Pre-commit comes in the venv environment (installed as described above). After activating the environment you have to install pre-commit  in your terminal by running:
```bash
pre-commit install
```
This needs to be done only once.

After installing pre-commit, it will trigger 'hooks' upon each `git commit -m ...` command. The hooks will be applied on all the files in the commit. A hook is nothing but a script specified in `.pre-commit-config.yaml`.

We use [ruffs](https://github.com/astral-sh/ruff) [pre-commit package](https://github.com/astral-sh/ruff-pre-commit) for linting and formatting. It will apply the same formating as the vscode Ruff extension would (v0.12.0).

If you want to skip the hooks, you can use `git commit -m "..." --no-verify`.

More information about pre-commit can be found [here](https://pre-commit.com).

## AWS Setup for pixtral Classifier

To run classification using the Pixtral Large Model, you must configure your AWS credentials:
1. Ensure you have access to Amazon Bedrock and the Pixtral model.
2. Set up your credentials:
   1. **AWS CLI**

     ```
     aws configure
     ```

   2. **Manually via config files**
   
     Create or edit the following files
     **~/.aws/config**
     ```
     [default]
     region=eu-central-1
     output=json
     ```
     **~/.aws/credentials**
     ```
     [default]
     aws_access_key_id=YOUR_ACCESS_KEY
     aws_secret_access_key=YOUR_SECRET_KEY
     ```


## Output Format
The script processes PDF pages and outputs predictions in `data/predictions.json`. 
The output is structured as a list of metadata and page predictions per file. 
Page predications "pages" is a list of classification results per page and metadata per page (language) .
#### Example Output
```json
[
    {
        "filename": "1858.pdf",
        "metadata": {
            "page_count": 3,
            "languages": ["de", "fr"]
        },
        "pages": [
            {
                "page": 1,
                "classification": {
                    "Text": 1,
                    "Boreprofile": 0,
                    "Maps": 0,
                    "Title_Page": 0,
                    "Unknown": 0
                },
                "metadata": {
                    "language": "de",
                    "is_frontpage": false
                }
            },
            {
                "page": 2,
                "classification": {
                    "Text": 0,
                    "Boreprofile": 1,
                    "Maps": 0,
                    "Title_Page": 0,
                    "Unknown": 0
                },
                "metadata": {
                    "language": "fr",
                    "is_frontpage": false
                }
            },
            {
                "page": 3,
                "classification": {
                    "Text": 1,
                    "Boreprofile": 0,
                    "Maps": 0,
                    "Title_Page": 0,
                    "Unknown": 0
                },
                "metadata": {
                    "language": null,
                    "is_frontpage": false
                }
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