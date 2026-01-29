# Page Classification for Geological Documents in Assets

## Purpose

This repository provides a classification pipeline to categorize PDF pages from geological reports into document classes, with the goal of supporting document
understanding and metadata extraction in the [Assets](https://assets.swissgeol.ch/) platform. The solution can be used as a standalone API.

This classification helps to map individual pages in a document, which ultimately should facilitate the identification of borehole profiles and maps in PDFs to link between documents on [Assets](https://assets.swissgeol.ch/) and boreprofiles on [Boreholes](https://boreholes.swissgeol.ch/).

## API endpoints
Current API supports two endpoint versions **V1** with the current implementation and **V2** with development features.


**Endpoints for V1:**
 - `/v1` - main document selection endpoint
 - `/v1/collect` - response collection

**Endpoints for V2:**
 - `/v2` - main document selection endpoint
 - `/v2/collect` - response collection

The request JSON body structure for all the endpoints follows the same pattern: `{"file": "filename.pdf"}`




## Classes

For each file a [response](#output-format) is compiled classifying the page into one of the defined page classes.

## Model Information

The API currently uses the treebased classifier as the default trained model.

* Model type: "treebased"
* Model path: models/stable/model.joblib

This model was trained on data from `data/single_pages_split_new/train` and saved as `model.joblib`. It uses 17 input features to predict class.


### Classes
Each page is categorized into one of the following:

1. `Text` - Continuous text page.
2. `Boreprofile` - Boreholes.
3. `Maps` - Geological or topographic maps.
4. `TitlePage` - Title pages of original reports.
5. `GeoProfile` - Geological cross-sections or longitudinal profiles.
6. `Table` -  Tabular numeric/textual data.
7. `Diagram` - Scientific 2D graphs or plots.
8. `Unknown` - Everything else.


## Output Format


#### Example Output (v1)
```jsonc
{
	"has_finished": true,
	"data": [
		{
			"filename": "input.pdf",					// Name of the file
			"metadata": {
				"page_count": 1,						// Number of pages
				"languages": [							// Detected languages
					"de"
				]
			},
			"pages": [									// List of dictionaries containing:
				{
					"predicted_class": "Boreprofile",	// Type of element (PascalPageClasses)
					"page_number": 1,					// The page number (1-indexed)
					"page_metadata": {
						"language": "de",				// Language of page
						"is_frontpage": false
					}
				}
			]
		}
	]
}
```


#### Example Output (v2)

```jsonc
{
	"has_finished": true,
	"data":{
		"filename": "input.pdf",					// Name of the file
		"page_count": 3,							// Number of pages
		"languages": [								// Detected languages
			"de"
		],
		"entities": [								// List of elements present in file
			{
				"classification": "boreprofile",	// Type of element (PageClasses)
				"page_start": 1,					// Starting page
				"page_end": 3,						// Ending page
				"language": "de"			    	// Detected language
			}
		]
	}
}
```


#### General Notes

- The classifier supports batch input of multiple reports.
- Input must be preprocessed: PDFs should already have OCR.
- Classification is multi-class with a single label per page. Future updates may support multiple-labels.


---
## Development quick start
Requirements: Python 3.10(recommended), OCR'ed PDFs.

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
### 2. Install dependencies
For basic runtime API install based dependencies:
```bash
pip install .
```
For development, install optional tools with:
```bash
pip install '.[all]'
```

### 3. Copy .env.template and specify your paths:
```bash
cp .env.template .env
```
For development:
- Set `MLFLOW_TRACKING=True` in `.env` file for experiment tracking.

### 4. (Optional) Use a pre-trained model:
- Option A: Download a pre-trained model from the [S3 bucket: stijnvermeeren-assets-data ](https://eu-central-1.console.aws.amazon.com/s3/buckets/stijnvermeeren-assets-data?region=eu-central-1&bucketType=general&tab=objects).
- Option B: Train your own model as described in [Train your Model](#train-your-model).

### 5. (Optional) Start the MLflow UI

For development: Start MLflow UI:
```
mlflow ui
```
### 6. Run the classification:
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
---
## Run the API locally

If you want to run the API on your own local documents instead of AWS S3, enable the local S3 mode and spin up MinIO (see below).

### MinIO setup (optional)

In your .env file, activate the local mode flag:

```bash
# Use local S3 (MinIO) instead of AWS
USE_LOCAL=True

# Bucket and prefix used by the API
S3_BUCKET="my-bucket"			# choose your own
S3_FOLDER="my-folder/"			# choose your own

# Local MinIO connection
LOCAL_S3_ENDPOINT="http://localhost:9000"
LOCAL_S3_ACCESS_KEY="admin"     # choose your own
LOCAL_S3_SECRET_KEY="admin123"  # choose your own
```

Replace `${LOCAL_S3_ACCESS_KEY}` / `${LOCAL_S3_ACCESS_KEY}` with the values set in .env.

```bash
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=${LOCAL_S3_ACCESS_KEY} \
  -e MINIO_ROOT_PASSWORD=${LOCAL_S3_SECRET_KEY} \
  -v "$(pwd)/minio/data:/data" \
  quay.io/minio/minio server /data --console-address ":9001"
```

Open the [MiniIO UI](http://localhost:9001) console and log in using the credentials defined in your `.env` file (`${LOCAL_S3_ACCESS_KEY}` / `${LOCAL_S3_SECRET_KEY}`). From the web interface, create a bucket named `${S3_BUCKET}`, then create a folder inside it called `${S3_FOLDER}`. Finally, upload your local PDF files to this folder. These files will then be available for the API when you run classification requests locally.


### Run

To test the API locally run the following command:

```bash
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

This will start the server on port 8000 of the localhost and enable automatic reloading whenever changes are made to the code.

---

## Build docker image

To test the docker image locally you can build the image using the following command:

```bash
docker build -t assets-api . -f Dockerfile
```

This command will build the Docker image with the tag `assets-api`.

Verify that the Docker image has been successfully built by running the following command:

```bash
docker images
```

To run the Docker container, use the following command, and remember to add your AWS credentials in the `.env` file:

```bash
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env.api:ro assets-api
```

You can now run a classification job on your own PDF (replace `${YOUR_OWN_FILENAME}.pdf` with your actual file name). Make sure this file exists in the configured S3 bucket and folder before starting the process.

```
# Run classification
curl -X POST http://127.0.0.1:8000/v1/ \
  -H "Content-Type: application/json" \
  -d '{"file": "YOUR_OWN_FILENAME.pdf"}' -i

# Collect results
curl -X POST http://127.0.0.1:8000/v1/collect \
  -H "Content-Type: application/json" \
  -d '{"file": "YOUR_OWN_FILENAME.pdf"}' -i
```


---

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

## Data
The dataset is stored in the S3 bucket `stijnvermeeren-assets-data`, under the `single_pages/` folder. 
It contains categorized subfolders per class.
In addition, boreprofile data from the `zurich` and `geoquat/validation` folders used in the [swissgeol-boreholes-dataextraction](https://github.com/swisstopo/swissgeol-boreholes-dataextraction) repository and stored in the S3 bucket `stijnvermeeren-boreholes-data` can be classified and compared using existing ground truth.

### Ground Truth
- Single-page ground truths: `data/gt_single_pages.json`  
- External evaluation sets:
  - Zurich: `data/gt_zurich.json`
  - GeoQuat: `data/gt_geoquat.json`
---

## Repository Structure

- `config/`: YAML configs (models, matching, prediction profiles)
- `data/` : input data,  predictions and ground truths
- `evaluation/`: Evaluation and metrics
- `models/`: Models (e.g. LayoutLMv3, TreeBased)
- `prompts/`: Pixtral prompts
- `src/`: Utility scripts and core logic 
- `tests/`: Unit tests
- `main.py`: CLI entry point
- `api/`: API
---

## Train your Model
### Split data
Split data into train and validation set.
```bash
python scripts/split_data.py
# creates:
# data/single_pages_split/train/
# data/single_pages_split/val/
```
### Train LayoutLMv3

To train a LayoutLMv3 model, run:
```bash
python -m src.models.layoutlmv3.train \
    --config-file-path config/layoutlmv3_config.yaml \
    --out-directory models/layoutlmv3_output \
    # Optional argument:
    --model-checkpoint models/layoutlmv3_pretrained_checkpoint
```
**Arguments**:
- `config_file_path`: Path to the YAML configuration file with model parameters and dataset paths.
- `out_directory`: Directory where the trained model will be saved.
- `model_checkpoint` (optional): Path to a pre-trained model checkpoint. If not provided, the model will be initialized from the Hugging Face hub based on the config.

The script supports freezing/unfreezing specific layers and uses the Hugging Face Trainer API under the hood.

### Train TreeBased (RandomForest or  XGBoost)
To train a RandomForest or XGBoost classifier, use:
```bash
python -m src.models.treebased.train \
    --config-file-path config/xgboost_config.yml \
    --out-directory models/xgboost_model
```
- `config_file_path`: Path to the YAML config specifying hyperparameters and feature extraction settings.
- `out_directory`: Output path for the trained model.

If you're training an XGBoost model on macOS, you may encounter issues related to OpenMP. To resolve this, install the OpenMP library using Homebrew:
```bash
brew install libomp
```

#### Model explainability (with treebased models)

See [this file](src/models/treebased/README.md) for details about model explainability and SHAP-based interpretation for tree-based models.

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

