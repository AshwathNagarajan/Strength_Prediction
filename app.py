"""Streamlit GUI for Review 1 compressive-strength prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATASET_PATH, FEATURES, FIGURES_DIR, METADATA_FILE, MODEL_FILES, SHAP_DIR, TARGET
from src.data_loader import load_dataset
from src.explainability import create_waterfall_figure
from src.predictor import StrengthPredictor
from src.utils import format_model_reason, load_json


st.set_page_config(page_title="Compressive Strength AI", layout="wide")


def require_training() -> bool:
    missing = [path for path in [METADATA_FILE, *MODEL_FILES.values()] if not path.exists()]
    if missing:
        st.error("Trained model artifacts are missing. Please run `python train.py` before launching the GUI.")
        st.caption("Missing: " + ", ".join(str(path) for path in missing))
        return False
    return True


@st.cache_resource(show_spinner=False)
def get_predictor() -> StrengthPredictor:
    return StrengthPredictor()


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return load_json(METADATA_FILE)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    data, _ = load_dataset(DATASET_PATH)
    for column in FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=FEATURES + [TARGET])


def image_if_exists(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not found: {path.name}. Run `python train.py` to regenerate outputs.")


def metric_cards(metadata: dict, data: pd.DataFrame) -> None:
    best = metadata["best_model"]
    best_key = best.lower()
    cols = st.columns(6)
    cols[0].metric("Dataset Samples", f"{len(data):,}")
    cols[1].metric("Input Features", str(len(FEATURES)))
    cols[2].metric("Best Model", best)
    cols[3].metric("Best R²", f"{metadata[best_key]['r2']:.3f}")
    cols[4].metric("Best MAE", f"{metadata[best_key]['mae']:.2f} MPa")
    cols[5].metric("Best RMSE", f"{metadata[best_key]['rmse']:.2f} MPa")


def dashboard(metadata: dict, data: pd.DataFrame) -> None:
    st.title("Development of AI-Driven Support System for Performance Prediction and Material Optimization of Sustainable Steel-Concrete Composite Structures")
    st.subheader("Review 1 - Compressive Strength Prediction")
    metric_cards(metadata, data)
    st.markdown(
        """
        **Review 1 workflow**

        Material Parameters → XGBoost + CatBoost → Model Comparison → Best Model → Compressive Strength → SHAP Explanation
        """
    )
    st.info("Current scope: four sustainable concrete material inputs only. Structural capacity, shear, deflection, inverse design, and GWP optimization are intentionally excluded.")
    st.write("### XGBoost vs CatBoost")
    image_if_exists(FIGURES_DIR / "model_comparison_metrics.png")


def prediction_page(metadata: dict, predictor: StrengthPredictor) -> None:
    st.header("Strength Prediction")
    ranges = predictor.feature_ranges
    st.caption("Inputs use dataset-derived training ranges. Values outside the range are allowed but flagged as extrapolation.")

    cols = st.columns(4)
    defaults = {"w/b Ratio": 0.38, "Fly Ash (%)": 20.0, "GGBS (%)": 30.0, "Recycled Aggregate (%)": 40.0}
    inputs = {}
    for col, feature in zip(cols, FEATURES):
        min_value = float(ranges[feature]["min"])
        max_value = float(ranges[feature]["max"])
        value = min(max(defaults[feature], min_value), max_value)
        inputs[feature] = col.number_input(
            feature,
            value=float(value),
            min_value=None,
            max_value=None,
            step=0.01 if feature == "w/b Ratio" else 1.0,
            help=f"Training range: {min_value:g} to {max_value:g}",
        )

    if st.button("PREDICT STRENGTH", type="primary", width="stretch"):
        try:
            result = predictor.predict_strength(
                inputs["w/b Ratio"],
                inputs["Fly Ash (%)"],
                inputs["GGBS (%)"],
                inputs["Recycled Aggregate (%)"],
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        for warning in result["domain_warnings"]:
            st.warning("Warning: " + warning)

        xgb_col, cat_col = st.columns(2)
        xgb_col.metric("XGBoost Prediction", f"{result['xgboost_prediction']:.2f} MPa", f"R² {metadata['xgboost']['r2']:.3f} | RMSE {metadata['xgboost']['rmse']:.2f}")
        cat_col.metric("CatBoost Prediction", f"{result['catboost_prediction']:.2f} MPa", f"R² {metadata['catboost']['r2']:.3f} | RMSE {metadata['catboost']['rmse']:.2f}")

        st.success(f"BEST MODEL: {result['best_model']} | Recommended Prediction: {result['best_prediction']:.2f} MPa")
        st.write(
            f"Model Agreement: prediction difference = {result['difference_between_models']:.2f} MPa "
            f"({result['percentage_difference_between_models']:.2f}%)."
        )
        if result["agreement_warning"]:
            st.warning("The two models show a comparatively large prediction difference for this input. Interpret the result cautiously.")

        st.subheader("Why was this strength predicted?")
        contrib_df = pd.DataFrame(result["shap_values"])
        contrib_df["SHAP Contribution"] = contrib_df["SHAP Contribution"].map(lambda value: round(value, 4))
        st.dataframe(contrib_df, width="stretch", hide_index=True)

        positive = contrib_df[contrib_df["Effect"] == "Increased"]
        negative = contrib_df[contrib_df["Effect"] == "Reduced"]
        pos_col, neg_col = st.columns(2)
        pos_col.write("**Positive contributors**")
        pos_col.dataframe(positive[["Feature", "SHAP Contribution"]], hide_index=True, width="stretch")
        neg_col.write("**Negative contributors**")
        neg_col.dataframe(negative[["Feature", "SHAP Contribution"]], hide_index=True, width="stretch")

        bar_fig = px.bar(
            contrib_df,
            x="SHAP Contribution",
            y="Feature",
            color="Effect",
            orientation="h",
            title="Local SHAP Contributions",
            color_discrete_map={"Increased": "#16a34a", "Reduced": "#dc2626"},
        )
        st.plotly_chart(bar_fig, width="stretch")

        try:
            fig = create_waterfall_figure(
                {
                    "baseline": result["baseline_prediction"],
                    "contributions": result["shap_values"],
                }
            )
            st.pyplot(fig, clear_figure=True)
        except Exception as exc:
            st.info(f"Waterfall plot could not be rendered in this environment: {exc}")

        st.write(result["explanation"])


def model_comparison(metadata: dict) -> None:
    st.header("XGBoost vs CatBoost")
    metrics = pd.DataFrame(
        [
            {"Model": "XGBoost", **metadata["xgboost"]},
            {"Model": "CatBoost", **metadata["catboost"]},
        ]
    )
    st.dataframe(metrics, width="stretch", hide_index=True)
    st.success(f"Selected Best Model: {metadata['best_model']}")
    st.write(format_model_reason(metadata))

    st.subheader("Cross-Validation Metrics")
    st.dataframe(pd.DataFrame(metadata["cross_validation"]).T, width="stretch")

    st.subheader("Evaluation Graphs")
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Comparison Chart"])
    with tab1:
        cols = st.columns(2)
        with cols[0]:
            image_if_exists(FIGURES_DIR / "xgboost_actual_vs_predicted.png", "XGBoost")
        with cols[1]:
            image_if_exists(FIGURES_DIR / "catboost_actual_vs_predicted.png", "CatBoost")
    with tab2:
        cols = st.columns(2)
        with cols[0]:
            image_if_exists(FIGURES_DIR / "xgboost_residual_plot.png", "XGBoost")
        with cols[1]:
            image_if_exists(FIGURES_DIR / "catboost_residual_plot.png", "CatBoost")
    with tab3:
        image_if_exists(FIGURES_DIR / "model_comparison_metrics.png")


def explainable_ai() -> None:
    st.header("Explainable AI")
    st.info("Feature importance describes the trained model's predictive behavior and does not by itself prove physical causation.")
    tab1, tab2 = st.tabs(["XGBoost Explanation", "CatBoost Explanation"])
    with tab1:
        image_if_exists(SHAP_DIR / "xgboost_shap_summary.png", "XGBoost SHAP Summary")
        image_if_exists(SHAP_DIR / "xgboost_shap_feature_importance.png", "XGBoost SHAP Feature Importance")
    with tab2:
        image_if_exists(SHAP_DIR / "catboost_shap_summary.png", "CatBoost SHAP Summary")
        image_if_exists(SHAP_DIR / "catboost_shap_feature_importance.png", "CatBoost SHAP Feature Importance")


def dataset_analysis(data: pd.DataFrame, predictor: StrengthPredictor) -> None:
    st.header("Dataset Analysis")
    st.write(f"Records used after selected-column validation: **{len(data):,}**")
    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(data.head(50), width="stretch")
    with st.expander("Feature Ranges"):
        st.dataframe(pd.DataFrame(predictor.feature_ranges).T, width="stretch")
    with st.expander("Descriptive Statistics"):
        st.dataframe(data[FEATURES + [TARGET]].describe().T, width="stretch")
    with st.expander("Missing Values"):
        st.dataframe(data[FEATURES + [TARGET]].isna().sum().rename("missing_count"), width="stretch")
    with st.expander("EDA Figures", expanded=True):
        image_if_exists(FIGURES_DIR / "eda_strength_distribution.png")
        image_if_exists(FIGURES_DIR / "eda_correlation_heatmap.png")
        cols = st.columns(2)
        for index, filename in enumerate(
            [
                "eda_wb_ratio_vs_strength.png",
                "eda_fly_ash_vs_strength.png",
                "eda_ggbs_vs_strength.png",
                "eda_recycled_aggregate_vs_strength.png",
            ]
        ):
            with cols[index % 2]:
                image_if_exists(FIGURES_DIR / filename)


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Page",
        ["Dashboard", "Strength Prediction", "Model Comparison", "Explainable AI", "Dataset Analysis"],
    )
    if not require_training():
        return
    metadata = get_metadata()
    predictor = get_predictor()
    data = get_dataset()

    if page == "Dashboard":
        dashboard(metadata, data)
    elif page == "Strength Prediction":
        prediction_page(metadata, predictor)
    elif page == "Model Comparison":
        model_comparison(metadata)
    elif page == "Explainable AI":
        explainable_ai()
    else:
        dataset_analysis(data, predictor)


if __name__ == "__main__":
    main()
