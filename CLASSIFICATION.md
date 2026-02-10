# Classification

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
| title_page   	 | 		107 | 		10.6 |
| text   		 | 		202 | 		20.0 |
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
    └── val
        └── ...
```


The files are splitted in a 80-20% ratio based on filename using script `src/scripts/split_data.py`.

```bash
# Splits dataset into train and validation sets
python src/scripts/split_data.py -i data/single_pages -o data/single_pages_splits -rv 0.2 -rt 0.0
```
