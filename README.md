# Development of AI-Driven Support System for Performance Prediction and Material Optimization of Sustainable Steel-Concrete Composite Structures

## Review 1 Scope

This implementation covers only the **Compressive Strength Prediction Module**.

Implemented workflow:

Material Inputs -> XGBoost + CatBoost -> Compressive Strength Prediction -> Best Model Selection -> SHAP Explainability

Not included yet: material optimization, target-strength inverse design, structural capacity prediction, shear prediction, deflection prediction, genetic algorithms, GWP optimization, or full steel-concrete structural optimization.

## Objective

Predict **Compressive Strength fc' (MPa)** from sustainable concrete material parameters using two machine-learning regressors and explain the prediction with SHAP.

## Input Parameters

Only these four predictors are used:

- `w/b Ratio`
- `Fly Ash (%)`
- `GGBS (%)`
- `Recycled Aggregate (%)`

The dataset column `Recycled Agg. (%)` is safely mapped to the canonical project feature name `Recycled Aggregate (%)`.

## Target

- `Compressive Strength fc' (MPa)`

Structural variables such as span, beam geometry, steel grade, capacity, shear, and deflection are intentionally excluded to avoid target leakage and to preserve the Review 1 scope.

## ML Models

The user-facing comparison includes only:

- XGBoost Regressor
- CatBoost Regressor

Both models are tuned with reproducible 5-fold cross-validation and evaluated on an 80/20 holdout split using:

- R²: higher is better
- MAE: average absolute prediction error in MPa, lower is better
- RMSE: root mean squared error in MPa, lower is better

The best model is selected automatically by highest test R², then lower RMSE, then lower MAE.

## Explainable AI

SHAP is used for:

- Global feature importance
- SHAP summary plots
- Per-prediction feature contributions
- Deterministic human-readable explanation text

Feature importance describes the trained model's predictive behavior and does not by itself prove physical causation.

## Project Structure

```text
compressive_strength_ai/
|
├── data/
│   └── dataset_span_mm.csv
├── models/
│   ├── xgboost_model.pkl
│   ├── catboost_model.pkl
│   ├── best_model.pkl
│   ├── model_metadata.json
│   └── feature_ranges.json
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── shap/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── predictor.py
│   └── utils.py
├── notebooks/
│   └── optional_analysis.ipynb
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python train.py
```

This command loads and validates the dataset, trains/tunes XGBoost and CatBoost, evaluates both models, saves trained artifacts, generates EDA and evaluation plots, and creates global SHAP outputs.

## Run Streamlit GUI

```bash
streamlit run app.py
```

The GUI loads saved model files and does not retrain on interaction. If model files are missing, run `python train.py` first.

