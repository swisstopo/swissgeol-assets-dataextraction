# Page Classification for Geological Documents

This repository provides a classification pipeline to categorise PDF pages from geological reports into document classes, with the goal of supporting document understanding and metadata extraction in the [Assets](https://assets.swissgeol.ch/) platform. The solution can be used as a standalone API.

The classification helps to map individual pages in a document, which facilitates the identification of borehole profiles and maps in PDFs to link between documents on [Assets](https://assets.swissgeol.ch/) and boreprofiles on [Boreholes](https://boreholes.swissgeol.ch/).

Features:

- Classifies individual PDF pages into 8 document classes: Text, Boreprofile, Maps, TitlePage, GeoProfile, Table, Diagram, Unknown.
- Two classifier backends: feature-based XGBoost (default) and Pixtral Large (via Amazon Bedrock).
- REST API with versioned endpoints (V1, V2) and batch processing support.
- SHAP-based model explainability for tree-based classifiers.
- MLflow experiment tracking (optional).

## Usage

### 1. Installation

Python >=3.11 is required. Example using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
To install base dependencies:

```bash
pip install .
```

For development, install all optional tools:

```bash
pip install '.[all]'
```

### 3. Configuration

Copy the environment template and configure your settings:

```bash
cp .env.template .env
```

### 4. Running as CLI

We base our running pipeline on the command:

```bash
python main.py -i <input_path> [-g <ground_truth_path>] [-c <classifier_name>] [-p <model_path>] [-w]
```

The input path `-i` is mandatory and can be either a single PDF file or a directory. In the latter case, all PDF files in that directory will be processed. To simply obtain predictions, use `-w` to write the results to `data/prediction.json`.

```bash
python main.py -i path/to/document.pdf -w
```

If no classifier (`-c`) is specified, the default `treebased` classifier is used. The model path (`-p` / `--model_path`) defaults to `models/stable/model.joblib`. The ground truth file (`-g`) is optional and only required to compute accuracy metrics:

```bash
python main.py -i data/single_pages/ -g data/gt_single_pages.json
```

See [Model Training](docs/training.md) for the ground truth file format.


| Classifier | Description |
|------------|-------------|
| `treebased` | Default. Feature-based XGBoost model |
| `pixtral` | Uses Pixtral Large via Amazon Bedrock |


### 5. Running as API

```bash
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

For detailed endpoint documentation, output formats, and local S3 setup, see the [API Usage Guide](docs/api-usage.md).

## Documentation

| Document | Description |
|----------|-------------|
| [API Architecture](api/README.md) | API versioning and OpenAPI spec |
| [API Usage Guide](docs/api-usage.md) | Endpoints, output formats, MinIO setup |
| [Docker Deployment](docs/docker-deployment.md) | Building and running Docker images |
| [Model Overview](models/stable/README.md) | Stable model features and usage |
| [Model Explainability](src/models/treebased/README.md) | SHAP interpretation for tree-based models |
| [Model Training](docs/training.md) | Data, XGBoost training, hyperparameter tuning |
| [Pixtral Setup](docs/pixtral-setup.md) | AWS Bedrock configuration |


## Repository Structure

- `api/`: FastAPI application
- `config/`: YAML configs (models, matching, prediction profiles)
- `data/`: Input data, predictions and ground truths
- `docs/`: Detailed documentation
- `evaluation/`: Evaluation and metrics
- `models/`: Trained models (TreeBased)
- `prompts/`: Pixtral prompts
- `src/`: Core logic and utility scripts
- `tests/`: Unit tests
- `main.py`: CLI entry point

## Contributing

We use [pre-commit](https://pre-commit.com) hooks with [Ruff](https://github.com/astral-sh/ruff) for code formatting. After installing dependencies, run:

```bash
pre-commit install
```

This needs to be done only once. After installing, hooks will run automatically on each `git commit`.

## Governance

This repository is managed by the Swiss Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/). The project lead and primary maintainer is Stijn Vermeeren ([@stijnvermeeren-swisstopo](https://www.github.com/stijnvermeeren-swisstopo)). Support has come from external contractors at [Visium](https://www.visium.ch/) and [EBP](https://www.ebp.global/). Individual contributors are listed on [GitHub's *Contributors* page](https://github.com/swisstopo/swissgeol-assets-dataextraction/graphs/contributors).

We welcome suggestions, bug reports and code contributions from third parties. However, the priority of any external request will have to be evaluated based on compatibility with our legal mandate as a government agency.

## Licence

This project is released as open-source software, under the principle of "*public money, public code*", in accordance with the 2023 federal law "[*EMBAG*](https://www.fedlex.admin.ch/eli/fga/2023/787/de)", and following the guidance of the [tools for OSS published by the Federal Chancellery](https://www.bk.admin.ch/bk/en/home/digitale-transformation-ikt-lenkung/bundesarchitektur/open_source_software/hilfsmittel_oss.html).

The source code is licensed under the [AGPL-3.0-only License](LICENSE). This is due to the licensing of certain dependencies, most notably [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright), which is only available under either the AGPL license or a commercial license. If this dependency is removed in the future, we will switch to a more permissive license for this project.
