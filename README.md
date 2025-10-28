# Page Classification for Geological Documents in Assets

## Purpose

This repository provides a classification pipeline to categorize PDF pages 
from geological reports into document classes, with the goal of supporting document
understanding and metadata extraction in the [Assets](https://assets.swissgeol.ch/) platform. The solution can be used as a standalone API.

This classification helps to map individual pages in a document, which ultimately should facilitate the identification 
of borehole profiles and maps in PDFs to link between documents on [Assets](https://assets.swissgeol.ch/) and 
boreprofiles on [Boreholes](https://boreholes.swissgeol.ch/).

## API endpoints
Current API supports two endpoint versions **V1** with the latest changes (e.g., extended classes and different [response schema](#output-format)) and **V0** for backwards compatability.

**Endpoints for V0:**
 - `/` - main document selection endpoint
 - `/collect` - response collection

 **Endpoints for V1:**
 - `/v1` - main document selection endpoint
 - `/v1/collect` - response collection

The request JSON body structure for all the endpoints follows the same pattern: `{"file": "filename.pdf"}`

## Classes

For each file a [response](#output-format) is compiled classifying the page into one of the defined page classes.

### V0 version
Each page is categorized into one of the following:

1. `Text` - Continuous text page.
2. `Boreprofile` - Boreholes.
3. `Maps` - Geological or topographic maps.
4. `Title_Page` - Title pages of original reports.
5. `Unknown` - Everything else.

Extended classes in available in V1 version are mapped to `unknown` when running the V0 API version.

### V1 version 
The V1 version containes extended classes from v0 and Each page is categorized into one of the following:

1. `Text` - Continuous text page.  
2. `Boreprofile` - Boreholes. 
3. `Maps` - Geological or topographic maps.  
4. `TitlePage` - Title pages of original reports.  
5. `GeoProfile` - Geological cross-sections or longitudinal profiles.
6. `Table` -  Tabular numeric/textual data.
7. `Diagram` - Scientific 2D graphs or plots.
8. `Unknown` - Everything else.


## Output Format
`data/prediction.json` (if `-w`/`--write_result`) or returned as a Python object.
#### Example Output (v0)
```json
{
	"has_finished": true,
	"data": [
		{
			"filename": "input.pdf",
			"metadata": {
				"page_count": 1,
				"languages": [
					"de"
				]
			},
			"pages": [
				{
					"page": 1,
					"classification": {
						"Text": 0,
						"Boreprofile": 1,
						"Maps": 0,
						"Title_Page": 0,
						"Unknown": 0
					},
					"metadata": {
						"language": "de",
						"is_frontpage": false
					}
				}
			]
		}
	]
}
```

**V0 Notes**:
- `filename`: The name of the processed PDF file.
- `metadata`: metadata about the file.
- `pages`: list of dictionaries containing:
  - `page`: The page number (1-indexed). 
  - `classification`: Classification of a current page:
    - 1: class was assigned to the page. 
    - 0: class was not assigned.
  - `metadata`: metadata about the current page.


#### Example Output (v1)
```json
{
	"has_finished": true,
	"data": [
		{
			"filename": "742_6.pdf",
			"metadata": {
				"page_count": 1,
				"languages": [
					"de"
				]
			},
			"pages": [
				{
					"predicted_class": "Boreprofile",
					"page_number": 1,
					"page_metadata": {
						"language": "de",
						"is_frontpage": false
					}
				}
			]
		}
	]
}
```
**V1 Notes**:
- `filename`: The name of the processed PDF file.
- `metadata`: metadata about the file.
- `pages`: list of dictionaries containing:
  - `predicted_class`: The class name of the predicted class (e.g. "Boreprofile"). All possible classes are listed above in the section "Classes".
  - `page_number`: The page number (1-indexed).
  - `page_metadata`: metadata about the current page.


**General Notes:**

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
Make sure you have `fasttext-predict` installed instead of `fasttext` (see 5. Setup FastText Language Detection).

### 3. Copy .env.template and specify your paths:
```bash
cp .env.template .env
```
For development:
- Set `MLFLOW_TRACKING=True` in `.env` file for experiment tracking.

### 4. (Optional) Use a pre-trained model:
- Option A: Download a pre-trained model from the [S3 bucket: stijnvermeeren-assets-data ](https://eu-central-1.console.aws.amazon.com/s3/buckets/stijnvermeeren-assets-data?region=eu-central-1&bucketType=general&tab=objects).
- Option B: Train your own model as described in [Train your Model](#train-your-model).

### 5. Setup FastText Language Detection

This project uses [fasttext-predict](https://github.com/searxng/fasttext-predict/), a lightweight, dependency-free wrapper exposing only the predict method.
We use this because [FastText](https://github.com/facebookresearch/fastText) is archived.
Download the FastText language identification model lid.176.bin form [this website](https://fasttext.cc/docs/en/language-identification.html):
```
mkdir -p models/FastText
curl -o models/FastText/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```
Set in `.env`:
```
 FASTTEXT_MODEL_PATH=models/FastText/lid.176.bin
```
### 6. (Optional) Start the MLflow UI

For development: Start MLflow UI:
```
mlflow ui
```
### 7. Run the classification:
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
## Start the FastAPI server

To test the API locally run the following commant: 

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
- `models/`: Models (e.g. FastText, LayoutLMv3, TreeBased)
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




## Model explainability (with treebased models)

Modern machine learning models like boosted trees can be very accurate but often behave like a black box.
To understand why a model makes a certain prediction for a document, we can use SHAP values, which provide a way to explain model outputs.

### Understanding SHAP values

SHAP (SHapley Additive exPlanations) values are based on a concept from game theory called **Shapley values**.
They quantify how much each *player* (here, each **feature**) contributes to the *outcome* (the model prediction).

In this analogy:
- **Players** → model features
- **Game outcome** → model prediction
- **Coalition** → a subset of features used by the model to make a prediction

Each model input (in our usecase, each page) receives its **own set of SHAP values** — one per feature. These values explain how much each feature contributed to the specific prediction made for that input. In multiclass models, there is one SHAP value **per feature and per class**.

The **Shapley value** of a feature is the *expected marginal contribution* of that feature, averaged over all possible coalitions of the other features.
In other words, it is the **expected change in the model output** when the feature is added to a random coalition of other features.

Formally, for a model $f$ and feature $i$, this expectation is computed as:

$$
\phi_i = 
\mathbb{E}_{S \subseteq F \setminus \{i\}} \big[ f(S \cup \{i\}) - f(S) \big] =
\sum_{S \subseteq F \setminus \{i\}} 
\frac{|S|! \, (|F| - |S| - 1)!}{|F|!} 
\left[ f(S \cup \{i\}) - f(S) \right]
$$

where:
- $\phi_i$ is the shapley value of the feature $i$ (specific to one input)
- $F$ is the full set of features,
- $|S|$ is the number of elements in the set $S$.
- $f(S)$ is the model output when only the features in $S$ are used.



### Simple example

Let’s consider a model with three features **A**, **B**, and **C**.
The SHAP value of feature **A** is the *average change* in the model output when we add **A** to every possible coalition of the other features.

$$
\phi_A =
\frac{1}{3}(f(A,B,C) - f(B,C)) +
\frac{1}{6}(f(A,B) - f(B)) +
\frac{1}{6}(f(A,C) - f(C)) +
\frac{1}{3}(f(A) - f(\varnothing))
$$

Each term measures how much the prediction changes when **A** joins a coalition,
and the weights ensure a fair average over all possible feature orderings.


### In practice

You can generate plots to interpret the model's decisions by enabling the `-x` flag.
This will e**x**plain the model's decisions for a single input. Note that saving the plots in high quality will considerably
slow down the pipeline. You can reduce the time it takes by lowering the `dpi` parameter of savefig calls.

```bash
python main.py -i data/single_pages/ -g data/gt_single_pages.json -c treebased -p models/stable/model.joblib -x
```

#### Stacked force plot (Local importance)
This plot is computed for a single input page and shows one subplot per class.
Each subplot is a force plot for that class, where the contribution of each feature to the class logit (log-odds) is shown.
The model predicts the class with the highest logit (shown with *(predicted)* on the plot). The features represented in blue negativelly contributed
the the output, and the ones in red positevelly influenced the decision.

#### Waterfall plot (Local importance)
This plot is computed for a single input page and for one class only (typically the predicted class). It is essentially a detailled version of the previous force plot for a specific class and shows the individual contribution of each feature to the class logit, along with the value of the feature for this input.
This allows to see which features drove the model to choose this class over others.


#### Absolute beeswarm plot (Global importance)
Plots will also be automatically generated during model training to see which features have the biggest impact on each class prediction.

For each class, the absolute beeswarm plot shows the magnitude of SHAP values for each feature across all samples.
It also shows an overall plot that takes the mean across all classes (not strictly correct according to Shapley theory, but gives a good idea of which features have the greatest impact on the model).

### Features currently used

Below is the list of all features currently used by the tree-based model, along with a short explanation of what they represent:

- **Words Per Line** – Average number of words per line on the page.
- **Text Zone Density** – Fraction of area occupied by text relative to the page area.
- **Mean Left** – Average horizontal position of the left edge of the text lines.
- **Text Width** – Average width of text lines.
- **Line Count** – Total number of lines on the page.
- **Indent Std Dev** – Standard deviation of line indentations, indicating alignment variability.
- **Capitalization Ratio** – Ratio of capitalized letters to total letters.
- **Has Sidebar** – Boolean indicating presence of a sidebar (column of numbers) on the page.
- **Has Borehole Keyword** – Boolean indicating if the text mentions a borehole-related keyword.
- **Num Valid Material Descriptions** – Count of lines that contain valid material descriptions.
- **Num Map Keyword Lines** – Number of lines containing map-related keywords.
- **Grid Line Length Sum** – Total length of detected grid lines on the page (lines that are horizontal or vertical).
- **Non Grid Line Length Sum** – Total length of lines not part of the grid.
- **Line Angle Entropy** – Entropy of line angles, measuring variation in line orientation.
- **Line Score** – A score combining line entropy and the number of non-grid lines.
- **Num Geo Profile Keywords** – Number of text lines containing geological profile keywords.
- **Num Unit Keyword** – Number of lines containing unit-related keywords (e.g., m, km).
- **Y Scale OK** – Boolean indicating a Y-axis scale was found on the page.
- **X Scale OK** – Boolean indicating a X-axis scale was found on the page.

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

