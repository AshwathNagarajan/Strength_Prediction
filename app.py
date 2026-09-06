"""Streamlit GUI for prediction and explainability review demo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.performance_config import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FEATURES,
    METADATA_FILE,
    NUMERIC_FEATURES,
    PERFORMANCE_FIGURES_DIR,
    PERFORMANCE_MODEL_DIR,
    PERFORMANCE_SHAP_DIR,
    TARGETS,
)
from src.performance_data import clean_performance_data, load_performance_dataset
from src.performance_predictor import PerformancePredictor
from src.utils import load_json


st.set_page_config(page_title="AI Structural Performance Predictor", layout="wide")


def image_if_exists(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not found: {path.name}. Run `python train_performance.py`.")


def require_training() -> bool:
    expected = [METADATA_FILE, *[PERFORMANCE_MODEL_DIR / f"{key}_model_bundle.pkl" for key in TARGETS]]
    missing = [path for path in expected if not path.exists()]
    if missing:
        st.error("Trained performance models are missing. Please run `python train_performance.py` first.")
        st.caption("Missing: " + ", ".join(str(path) for path in missing))
        return False
    return True


@st.cache_resource(show_spinner=False)
def get_predictor() -> PerformancePredictor:
    return PerformancePredictor()


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return load_json(METADATA_FILE)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    data, _ = load_performance_dataset(DATASET_PATH)
    return clean_performance_data(data)


def best_metric(metadata: dict, target_key: str, metric: str) -> float:
    target = metadata["targets"][target_key]
    best = target["best_model"]
    return float(target["metrics"][best][metric])


def dashboard(metadata: dict, data: pd.DataFrame) -> None:
    st.title("AI-Driven Support System for Sustainable Steel-Concrete Composite Structures")
    st.subheader("Review Scope: Prediction and Explainable AI")
    st.info("This demo predicts ultimate load and deflection from available beam, material, and connection parameters. Optimization is intentionally not implemented for this review.")

    cols = st.columns(5)
    cols[0].metric("Dataset Samples", f"{len(data):,}")
    cols[1].metric("Input Features", str(len(FEATURES)))
    cols[2].metric("Load Best Model", metadata["targets"]["ultimate_load"]["best_model"])
    cols[3].metric("Load R²", f"{best_metric(metadata, 'ultimate_load', 'r2'):.3f}")
    cols[4].metric("Deflection R²", f"{best_metric(metadata, 'deflection', 'r2'):.3f}")

    st.markdown(
        """
        **Demonstration workflow**

        Beam + Material + Connector Inputs -> XGBoost and CatBoost -> Ultimate Load + Deflection -> Best Model -> SHAP Explanation
        """
    )

    st.write("### Prediction Targets")
    target_cols = st.columns(2)
    for col, target_key in zip(target_cols, TARGETS):
        result = metadata["targets"][target_key]
        best = result["best_model"]
        target = TARGETS[target_key]
        col.metric(
            f"{target['label']} Best Model",
            best,
            f"RMSE {result['metrics'][best]['rmse']:.2f} {target['unit']}",
        )


def prediction_page(metadata: dict, predictor: PerformancePredictor) -> None:
    st.header("Prediction")
    st.caption("Inputs are checked against dataset-derived ranges. Out-of-range predictions are allowed but flagged as extrapolations.")
    defaults = predictor.default_inputs()

    values = {}
    with st.expander("Material Parameters", expanded=True):
        cols = st.columns(3)
        for index, feature in enumerate(
            [
                "Compressive Strength fc' (MPa)",
                "Modulus Ec (MPa)",
                "w/b Ratio",
                "Fly Ash (%)",
                "GGBS (%)",
                "Recycled Aggregate (%)",
                "Yield Strength fy (MPa)",
            ]
        ):
            bounds = predictor.profile["numeric_ranges"][feature]
            values[feature] = cols[index % 3].number_input(
                feature,
                value=float(defaults[feature]),
                step=0.01 if "Ratio" in feature else 1.0,
                help=f"Training range: {bounds['min']:g} to {bounds['max']:g}",
            )

    with st.expander("Beam and Connector Parameters", expanded=True):
        cols = st.columns(3)
        for index, feature in enumerate(["Stud Dia. ds (mm)", "Span L (mm)", "Beam Depth h (mm)", "Slab Thickness hc (mm)"]):
            bounds = predictor.profile["numeric_ranges"][feature]
            values[feature] = cols[index % 3].number_input(
                feature,
                value=float(defaults[feature]),
                step=1.0,
                help=f"Training range: {bounds['min']:g} to {bounds['max']:g}",
            )

        cat_cols = st.columns(2)
        for index, feature in enumerate(CATEGORICAL_FEATURES):
            categories = predictor.profile["categorical_values"][feature]
            values[feature] = cat_cols[index].selectbox(feature, categories, index=0)

    if st.button("PREDICT PERFORMANCE", type="primary", width="stretch"):
        try:
            result = predictor.predict(values)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        for warning in result["domain_warnings"]:
            st.warning("Warning: " + warning + " The prediction is an extrapolation and may be less reliable.")

        for target_key, target_result in result["targets"].items():
            target_meta = metadata["targets"][target_key]
            unit = target_result["unit"]
            st.subheader(target_result["label"])
            xgb_col, cat_col, best_col = st.columns(3)
            xgb_col.metric(
                "XGBoost Prediction",
                f"{target_result['xgboost_prediction']:.2f} {unit}",
                f"R² {target_meta['metrics']['XGBoost']['r2']:.3f}",
            )
            cat_col.metric(
                "CatBoost Prediction",
                f"{target_result['catboost_prediction']:.2f} {unit}",
                f"R² {target_meta['metrics']['CatBoost']['r2']:.3f}",
            )
            best_col.metric(
                "BEST MODEL",
                target_result["best_model"],
                f"Recommended: {target_result['best_prediction']:.2f} {unit}",
            )
            st.write(f"Model agreement: absolute difference = {target_result['difference_between_models']:.2f} {unit}.")

            st.write("#### Why was this prediction obtained?")
            explanation = target_result["explanation"]
            contribution_df = pd.DataFrame(explanation["contributions"])
            contribution_df["Input"] = contribution_df["Input"].astype(str)
            contribution_df["SHAP Contribution"] = contribution_df["SHAP Contribution"].round(4)
            st.dataframe(contribution_df, hide_index=True, width="stretch")

            fig = px.bar(
                contribution_df.sort_values("SHAP Contribution"),
                x="SHAP Contribution",
                y="Feature",
                color="Effect",
                orientation="h",
                title=f"{target_result['label']} Local SHAP Contributions",
                color_discrete_map={"Increased": "#16a34a", "Reduced": "#dc2626"},
            )
            st.plotly_chart(fig, width="stretch")
            st.write(explanation["text"])


def explainability_page() -> None:
    st.header("Explainable AI")
    st.info("Feature importance describes the trained model's predictive behavior and does not by itself prove physical causation.")

    for target_key, target in TARGETS.items():
        st.subheader(target["label"])
        tabs = st.tabs(["XGBoost Explanation", "CatBoost Explanation"])
        with tabs[0]:
            image_if_exists(PERFORMANCE_SHAP_DIR / f"{target_key}_xgboost_shap_summary.png")
            image_if_exists(PERFORMANCE_SHAP_DIR / f"{target_key}_xgboost_shap_importance.png")
        with tabs[1]:
            image_if_exists(PERFORMANCE_SHAP_DIR / f"{target_key}_catboost_shap_summary.png")
            image_if_exists(PERFORMANCE_SHAP_DIR / f"{target_key}_catboost_shap_importance.png")


def model_results_page(metadata: dict) -> None:
    st.header("Model Results")
    for target_key, target in TARGETS.items():
        st.subheader(target["label"])
        target_meta = metadata["targets"][target_key]
        rows = [
            {"Model": model, **metrics}
            for model, metrics in target_meta["metrics"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.success(f"Selected Best Model: {target_meta['best_model']}")
        tabs = st.tabs(["Actual vs Predicted", "Residuals"])
        with tabs[0]:
            cols = st.columns(2)
            with cols[0]:
                image_if_exists(PERFORMANCE_FIGURES_DIR / f"{target_key}_xgboost_actual_vs_predicted.png")
            with cols[1]:
                image_if_exists(PERFORMANCE_FIGURES_DIR / f"{target_key}_catboost_actual_vs_predicted.png")
        with tabs[1]:
            cols = st.columns(2)
            with cols[0]:
                image_if_exists(PERFORMANCE_FIGURES_DIR / f"{target_key}_xgboost_residual_plot.png")
            with cols[1]:
                image_if_exists(PERFORMANCE_FIGURES_DIR / f"{target_key}_catboost_residual_plot.png")


def dataset_page(data: pd.DataFrame, predictor: PerformancePredictor) -> None:
    st.header("Dataset Analysis")
    st.write(f"Default dataset: `{DATASET_PATH}`")
    st.write(f"Records after validation: **{len(data):,}**")
    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(data.head(50), width="stretch")
    with st.expander("Feature Ranges"):
        st.dataframe(pd.DataFrame(predictor.profile["numeric_ranges"]).T, width="stretch")
    with st.expander("Categorical Domains"):
        st.json(predictor.profile["categorical_values"])
    with st.expander("Descriptive Statistics"):
        st.dataframe(data.describe(include="all").T, width="stretch")
    with st.expander("EDA Figures", expanded=True):
        image_if_exists(PERFORMANCE_FIGURES_DIR / "performance_correlation_heatmap.png")
        cols = st.columns(2)
        with cols[0]:
            image_if_exists(PERFORMANCE_FIGURES_DIR / "ultimate_load_distribution.png")
        with cols[1]:
            image_if_exists(PERFORMANCE_FIGURES_DIR / "deflection_distribution.png")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Page", ["Dashboard", "Prediction", "Explainability", "Model Results", "Dataset Analysis"])
    if not require_training():
        return

    metadata = get_metadata()
    predictor = get_predictor()
    data = get_dataset()

    if page == "Dashboard":
        dashboard(metadata, data)
    elif page == "Prediction":
        prediction_page(metadata, predictor)
    elif page == "Explainability":
        explainability_page()
    elif page == "Model Results":
        model_results_page(metadata)
    else:
        dataset_page(data, predictor)


if __name__ == "__main__":
    main()
