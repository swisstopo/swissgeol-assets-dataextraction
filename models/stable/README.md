# Model Overview

This model was trained on data from single_page_split_new/train and saved as model.joblib.
It uses 17 input features to predict one of 8 possible classes:

"text", "boreprofile", "map", "title_page", "unknown", "geo_profile", "diagram", "table".


# Example usage
```
import joblib

model = joblib.load("path_to_model.joblib")

y_pred = model.predict(X)  # X must be a list containing 17 features
```

