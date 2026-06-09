import io
import logging
import os
import tempfile

import numpy as np
from dotenv import load_dotenv
from xgboost import XGBClassifier

from src.utils.utility import read_params

xg_boost_config = read_params("config/xgboost_config.yml")

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"

logger = logging.getLogger(__name__)

EXPLANATION_DEPENDENCIES = False
try:
    import matplotlib.pyplot as plt
    import shap
    from PIL import Image

    if mlflow_tracking:
        import mlflow

    EXPLANATION_DEPENDENCIES = True
except ImportError:
    pass


def explain_prediction(
    model: XGBClassifier,
    input_features: list[float],
    pred_idx: int,
    page_name: str,
    id2label: dict[int, str],
):
    """Generates SHAP explanations for a single prediction and logs the plots to MLflow.

    Args:
        model (XGBClassifier): The trained model.
        input_features (list[float]): The input features for the prediction.
        pred_idx (int): The index of the predicted class.
        page_name (str): The name of the page being explained.
        id2label (dict[int, str]): Mapping from class IDs to class labels.
    """
    if not verify_dependencies_and_flags():
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        feature_names = xg_boost_config["feature_names"]
        explainer = shap.TreeExplainer(model, feature_names=feature_names)
        shap_values = explainer([input_features])
        single_explanation = shap.Explanation(
            values=shap_values.values[..., pred_idx],
            base_values=shap_values.base_values[..., pred_idx],
            data=shap_values.data,
            feature_names=feature_names,
        )
        shap.plots.waterfall(single_explanation[0], show=False)
        fig = plt.gcf()
        fig.suptitle(f"SHAP Waterfall for the predicted class: {id2label[pred_idx]} (index={pred_idx})")
        fig.tight_layout()
        log_plt_fig_to_mlflow(tmpdir, f"{page_name}.png", "waterfall")

        main_fig, axes = plt.subplots(4, 2, figsize=(30, 14))
        axes = axes.flatten()

        for class_id in range(shap_values.values.shape[-1]):
            label = id2label.get(class_id)
            fig = shap.plots.force(
                shap_values.base_values[0, class_id],
                shap_values.values[0, :, class_id],
                feature_names=feature_names,
                matplotlib=True,
                show=False,
            )
            for text in plt.gca().texts:
                # Detect feature name labels (bottom text)
                if text.get_position()[1] < 0:  # y-coordinate is negative for feature names
                    text.set_rotation(45)
                    text.set_va("top")

            # Trick to assign current ax to existing subplots
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            buf.seek(0)
            img = Image.open(buf)
            ax = axes[class_id]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{label}" + (" (predicted class)" if pred_idx == class_id else ""), fontsize=16)

            plt.close(fig)  # Close the SHAP-generated figure

        main_fig.suptitle("SHAP force Plots for all classes.", fontsize=20)
        main_fig.tight_layout()
        log_plt_fig_to_mlflow(tmpdir, f"{page_name}.png", "stacked_force", dpi=300)


def explain_model(model: XGBClassifier, X_train: list[list[float]], id2label: dict[int, str]):
    """Generates global SHAP explanations for the model and logs the plots to MLflow.

    Args:
        model (XGBClassifier): The trained Tree-based model.
        X_train (list[list[float]]): The training features data used for the model.
        id2label (dict[int, str]): Mapping from class IDs to class labels.
    """
    if not verify_dependencies_and_flags():
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        feature_names = xg_boost_config["feature_names"]
        explainer = shap.TreeExplainer(model, feature_names=feature_names)
        shap_values = explainer(X_train)
        artifact_path = "global_explanation_plots"

        overall_abs_explanation = shap.Explanation(
            values=np.mean(np.abs(shap_values.values), axis=2),
            base_values=np.mean(np.abs(shap_values.base_values), axis=1),
            data=shap_values.data,
            feature_names=feature_names,
        )
        ax = shap.plots.beeswarm(overall_abs_explanation, color="shap_red", max_display=None, show=False)
        ax.set_xlabel("mean(|SHAP values|) (impact on model output)")
        fig = ax.get_figure()
        fig.suptitle("Overall beeswarm feature importance")
        fig.tight_layout()
        log_plt_fig_to_mlflow(tmpdir, "abs_beeswarm_overall.png", artifact_path, pad_inches=1)

        for class_id in range(shap_values.values.shape[-1]):
            label = id2label.get(class_id)
            class_explanation = shap.Explanation(
                values=shap_values.values[..., class_id],
                base_values=shap_values.base_values[..., class_id],
                data=shap_values.data,
                feature_names=feature_names,
            )

            # absolute beeswarm
            ax = shap.plots.beeswarm(class_explanation.abs, color="shap_red", show=False)
            ax.set_xlabel("|SHAP values| (impact on model output)")
            fig = ax.get_figure()
            fig.suptitle(f"Beeswarm feature importance for the class {label}.")
            fig.tight_layout()
            log_plt_fig_to_mlflow(tmpdir, f"abs_beeswarm_{label}.png", artifact_path)


def log_plt_fig_to_mlflow(local_dir: str, name: str, artifact_path: str, dpi=200, pad_inches=0.1):
    """Saves the current matplotlib figure to a local directory and logs it to MLflow.

    Args:
        local_dir (str): Local directory to save the figure temporarily (e.g., a temp directory).
        name (str): Name of the figure file.
        artifact_path (str): MLflow artifact path to log the figure.
        dpi (int): Dots per inch for the saved figure. Default is 200.
        pad_inches (float): Padding around the figure when saving. Default is 0.1.
    """
    fig_path = os.path.join(local_dir, name)
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close()
    mlflow.log_artifact(fig_path, artifact_path)


def verify_dependencies_and_flags():
    """Verifies that the necessary dependencies are installed and MLflow tracking is enabled.

    Returns:
        bool: True if dependencies are met and MLflow tracking is enabled, False otherwise.
    """
    if not EXPLANATION_DEPENDENCIES:
        logger.warning("Model explanation requested but dependencies not available. Install with: uv sync --extra dev")
        return False
    if not mlflow_tracking:
        logger.warning(
            "Explanation plots are only saved to MLflow. To view them, enable MLflow by setting the environment"
            "variable MLFLOW_TRACKING=True."
        )
        return False
    return True
