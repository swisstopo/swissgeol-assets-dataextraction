# Model Explainability for Tree-Based Models

Modern machine learning models like boosted trees can be very accurate but often behave like a black box. To understand why a model makes a certain prediction for a document, we can use SHAP values, which provide a way to explain model outputs.

## Understanding SHAP values

SHAP (SHapley Additive exPlanations) values are based on a concept from game theory called **Shapley values**. They quantify how much each *player* (here, each **feature**) contributes to the *outcome* (the model prediction).

In this analogy:
- **Players** → model features
- **Game outcome** → model prediction
- **Coalition** → a subset of features used by the model to make a prediction

Each model input (in our usecase, each page) receives its **own set of SHAP values** — one per feature. These values explain how much each feature contributed to the specific prediction made for that input. In multiclass models, there is one SHAP value **per feature and per class**.

The **Shapley value** of a feature is the *expected marginal contribution* of that feature, averaged over all possible coalitions of the other features. In other words, it is the **expected change in the model output** when the feature is added to a random coalition of other features.

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


## Simple example

Let’s consider a model with three features **A**, **B**, and **C**. The SHAP value of feature **A** is the *average change* in the model output when we add **A** to every possible coalition of the other features.

$$
\phi_A =
\frac{1}{3}(f(A,B,C) - f(B,C)) +
\frac{1}{6}(f(A,B) - f(B)) +
\frac{1}{6}(f(A,C) - f(C)) +
\frac{1}{3}(f(A) - f(\varnothing))
$$

Each term measures how much the prediction changes when **A** joins a coalition, and the weights ensure a fair average over all possible feature orderings.


## In practice

You can generate plots to interpret the model's decisions by enabling the `-x` flag. This will e**x**plain the model's decisions for a single input. Note that saving the plots in high quality will considerably slow down the pipeline. You can reduce the time it takes by lowering the `dpi` parameter of savefig calls.

```bash
python main.py -i data/single_pages/ -g data/gt_single_pages.json -c treebased -p models/stable/model.joblib -x
```

### Stacked force plot (Local importance)
This plot is computed for a single input page and shows one subplot per class. Each subplot is a force plot for that class, where the contribution of each feature to the class logit (log-odds) is shown. The model predicts the class with the highest logit (shown with *(predicted)* on the plot). The features represented in blue negatively contributed to the model output, and the ones in red positively influenced the decision.

### Waterfall plot (Local importance)
This plot is computed for a single input page and for one class only (typically the predicted class). It is essentially a detailed version of the previous force plot for a specific class and shows the individual contribution of each feature to the class logit, along with the value of the feature for this input. This allows to see which features drove the model to choose this class over others.

### Absolute beeswarm plot (Global importance)
Plots will also be automatically generated during model training to see which features have the biggest impact on each class prediction.

For each class, the absolute beeswarm plot shows the magnitude of SHAP values for each feature across all samples. It also shows an overall plot that takes the mean across all classes (not strictly correct according to Shapley theory, but gives a good idea of which features have the greatest impact on the model).

## Features currently used

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
