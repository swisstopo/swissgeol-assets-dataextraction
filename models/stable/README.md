# Model Overview

This model was trained on data from single_page_split_new/train and saved as model.joblib.
It uses 26 input features to predict one of 8 possible classes:

"text", "boreprofile", "map", "title_page", "unknown", "geo_profile", "diagram", "table".

## Features: 
  ### Text/word features (6)
  - Words Per Line
  - Text Zone Density
  - Mean Left
  - Text Width
  - Indent Std Dev
  - Capitalization Ratio
  ### Map features (5)
  - Num Map Keyword Lines
  - Grid Line Length Sum
  - Non Grid Line Length Sum
  - Line Angle Entropy
  - Line Score
  ### Geo profile and diagram features (4)
  - Num Geo Profile Keywords
  - Num Unit Keyword
  - Y Scale OK
  - X Scale OK
  ### Borehole features (8)
  - Num Valid Borehole Descriptions
  - Num Strip Logs
  - Num Tables
  - Num Boreholes
  - Num Good Sidebars
  - Best Sidebar Score
  - Num Long or Horizontal Lines
  - Text Line Count

# Version Notes

In the initial version (available on the v0 router with endpoint '/'), the classes "geo_profile", "diagram", and "table" are mapped to "unknown" after prediction.

# Example usage
```
import joblib

model = joblib.load("path_to_model.joblib")

y_pred = model.predict(X)  # X must be a list containing 17 features
```

