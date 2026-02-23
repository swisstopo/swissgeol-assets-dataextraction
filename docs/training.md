# Training Classifiers

## Data

The dataset is stored in the S3 bucket `stijnvermeeren-assets-data`, under the `single_pages/` folder. It contains categorised subfolders per class.

In addition, boreprofile data from the `zurich` and `geoquat/validation` folders used in the [swissgeol-boreholes-dataextraction](https://github.com/swisstopo/swissgeol-boreholes-dataextraction) repository (stored in the S3 bucket `stijnvermeeren-boreholes-data`) can be classified and compared using existing ground truth.

### Ground Truth

- Single-page ground truths: `data/gt_single_pages.json`
- External evaluation sets:
  - Zurich: `data/gt_zurich.json`
  - GeoQuat: `data/gt_geoquat.json`

## Splitting Data

Split data into train and validation sets:

```bash
python scripts/split_data.py
# creates:
# data/single_pages_split/train/
# data/single_pages_split/val/
```

## Training XGBoost (TreeBased)

To train an XGBoost classifier:

```bash
python -m src.models.treebased.train \
    --config-file-path config/xgboost_config.yml \
    --out-directory models/xgboost_model
```

- `--config-file-path`: Path to the YAML config specifying hyperparameters and feature extraction settings.
- `--out-directory`: Output path for the trained model.

### Parallel Feature Extraction

For faster training, use `train_parallel.py` which parallelises the feature extraction step:

```bash
python -m src.models.treebased.train_parallel \
    --config-file-path config/xgboost_config.yml \
    --out-directory models/xgboost_model \
    --max-workers 8
```

Additional arguments:
- `--max-workers N`: Limit the number of parallel workers (defaults to CPU count)
- `--tuning`: Enables hyperparameter tuning for the XGBoost algorithm

### macOS OpenMP Note

If you're training an XGBoost model on macOS, you may encounter issues related to OpenMP. To resolve this, install the OpenMP library using Homebrew:

```bash
brew install libomp
```

## Model Explainability

For SHAP-based interpretation of tree-based models, see [Model Explainability](../src/models/treebased/README.md).

## MLflow Experiment Tracking

Set `MLFLOW_TRACKING=True` in your `.env` file, then start the MLflow UI:

```bash
mlflow ui
```
