# Training XGBoost Classifier

To train the classification, the development package needs to be installed and MLflow tracking activated.

The dataset used to train the provided model (`models/stable/model.joblib`) is internal and not publicly available. It is stored in a private S3 bucket (`stijnvermeeren-assets-data`) accessible only to the project team. The dataset is composed of 1011 labeled single-page PDF across 9 classes, with ground truth available under `data/gt_single_pages.json`. The distribution of the pages is listed below.

| Class           | Number | Percentage |
|-----------------|-------:|-----------:|
| boreprofile     |    115 |       13.4 |
| diagram         |    106 |       10.5 |
| geo_profile     |     74 |        7.3 |
| map             |    126 |       12.5 |
| section_header  |     93 |        9.2 |
| table           |     60 |        5.9 |
| text            |    202 |       20.0 |
| title_page      |    109 |       10.8 |
| unknown         |    126 |       12.5 |


The classification results on the validation set are reported below.

| Class           | Precision | Recall | F1-score |
|-----------------|----------:|-------:|---------:|
| boreprofile     |      96.7 |   87.9 |     92.1 |
| diagram         |      84.6 |   84.6 |     84.6 |
| geo_profile     |      55.6 |   71.4 |     62.5 |
| map             |      63.6 |   80.8 |     71.2 |
| section_header  |      64.7 |   73.3 |     68.8 |
| table           |      90.9 |   83.3 |     87.0 |
| text            |      84.4 |   88.4 |     86.4 |
| title_page      |      95.0 |   95.0 |     95.0 |
| unknown         |      57.9 |   39.3 |     46.8 |
| Overall (macro) |      77.0 |   78.2 |     77.1 |


The `section_header` class is used internally as section title pages and does not appear as a classified entity in the API output. It is merged into the following page class.

## Train with your own data

### 1. Prepare the folder structure

Organize your labeled single-page images with one subfolder per class:

```
data/single_pages/
├── boreprofile/
├── diagram/
├── geo_profile/
├── map/
├── section_header/
├── table/
├── text/
├── title_page/
└── unknown/
```

### 2. Prepare the ground truth

The ground truth file is a JSON list of labeled documents.

```jsonc
[
  {
    "filename": "24911_1.pdf",       // file name relative to train / validation folder
    "metadata": {
      "page_count": 1                // total number of pages in the document
    },
    "pages": [
      {
        "page": 1,                   // page number (1-indexed)
        "classification": {          // one-hot encoding of the page class
          "text": 0,
          "boreprofile": 0,
          "map": 0,
          "geo_profile": 0,
          "title_page": 1,
          "diagram": 0,
          "table": 0,
          "unknown": 0,
          "section_header": 0
        }
      }
    ]
  }
]
```

### 3. Split into train and validation sets

Split the dataset using an 80-20% ratio based on filename:

```bash
python src/scripts/split_data.py \
    -i data/single_pages \
    -o data/single_pages_splits \
    -rv 0.2 \
    -rt 0.0
```

### 4. Update the config

Edit `config/xgboost_config.yml` to point to your data:

```yaml
# Path to the training set
train_folder_path: "data/single_pages_splits/train"
# Path to the validation set
val_folder_path: "data/single_pages_splits/validation"
# Ground truth for model training and validation
ground_truth_file_path: "data/gt_single_pages.json"
```

### 5. Train the model

To train the classifier, use:

```bash
python -m src.models.treebased.train \
    --config-file-path config/xgboost_config.yml \
    --out-directory models/xgboost_model
```

For faster training, use `train_parallel.py` which parallelizes the feature extraction step:

```bash
python -m src.models.treebased.train_parallel \
    --config-file-path config/xgboost_config.yml \
    --out-directory models/xgboost_model \
	  --max-workers 8
```

The trained model will be saved under `models/xgboost_model`. For macOS users, if you encounter OpenMP issues, install the library via Homebrew first:
 ```bash
 brew install libomp
```
