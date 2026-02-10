# Classification

To run classification, development package need to be installed and MLFLow tracking activated.


## Data

The dataset is stored in the S3 bucket `stijnvermeeren-assets-data`, under the `single_pages/` folder. It contains categorized subfolders per class.
In addition, the bucket contains a split of the dataset in `single_pages_splits/` folder. The ground truth is available under `data/gt_single_pages_2026.json`. The whole datset (train + valid) is composed of 1011 single pages files split spread over 9 classes.


| Class			 | Number   | Percentage |
|----------------|---------:|-----------:|
| boreprofile    | 		115 | 		13.4 |
| diagram   	 | 		106 | 		10.5 |
| geo_profile    | 		 74 | 		 7.3 |
| map   		 | 		126	| 		12.5 |
| section_header | 		 95 | 		 9.4 |
| table   		 | 		 60	| 		 5.9 |
| text   		 | 		202 | 		20.0 |
| title_page   	 | 		107 | 		10.6 |
| unknown   	 | 		126	| 		12.5 |

```yaml
stijnvermeeren-assets-data
│
│   # Reference classes
├── single_pages
│   ├── boreprofile
│   │   └── ...
│   ├── diagram
│   │   └── ...
│   ├── geo_profile
│   │   └── ...
│   ├── map
│   │   └── ...
│   ├── section_header
│   │   └── ...
│   ├── table
│   │   └── ...
│   ├── title_page
│   │   └── ...
│   └── unknown
│       └── ...
│
│   # Single pages split into two sets
└── single_pages_splits
    ├── train
    │   └── ...
    └── validation
        └── ...
```


The files are splitted in a 80-20% ratio based on filename using script `src/scripts/split_data.py`.

```bash
# Splits dataset into train and validation sets
python src/scripts/split_data.py -i data/single_pages -o data/single_pages_splits -rv 0.2 -rt 0.0
```

## Train XGBoost
To train a RandomForest or XGBoost classifier, use:

```bash
python -m src.models.treebased.train \
    --config-file-path config/xgboost_config.yml \
    --out-directory models/xgboost_model
```

Where `config_file_path` is the path to the YAML config specifying hyperparameters and feature extraction settings and `out_directory` the output path for the trained model.

If you're training an XGBoost model on macOS, you may encounter issues related to OpenMP. To resolve this, install the OpenMP library using Homebrew:
```bash
brew install libomp
```


## Results XGBoost

The classification results on the validation set are reported below

| Class			  | Precision | Recall | F1-score |
|-----------------|----------:|-------:|---------:|
| boreprofile     |		 93.8 |   90.9 | 	 92.3 |
| diagram   	  |	    100.0 |   84.6 | 	 91.7 |
| geo_profile     |		 40.0 |   57.1 | 	 47.1 |
| map   		  |		 65.6 |   80.8 | 	 72.4 |
| section_header  |		 72.2 |   86.7 | 	 78.8 |
| table   		  |		 90.0 |   75.0 | 	 81.8 |
| text   		  |		 86.4 |   88.4 | 	 87.4 |
| title_page   	  |		 90.4 |   95.0 | 	 92.7 |
| unknown   	  |		 57.9 |   39.3 | 	 46.8 |
| Overall (macro) |		 77.4 |   77.5 |     76.8 |


# Reporte reuslts compared to old profil
