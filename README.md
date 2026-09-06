# AI-Driven Support System for Sustainable Steel-Concrete Composite Structures

## Review Scope

This review implementation covers **prediction and explainable AI only**.

Implemented:

- Ultimate load prediction
- Deflection prediction
- XGBoost vs CatBoost model comparison
- Automatic best-model selection for each target
- SHAP global feature importance
- SHAP local explanation for each user prediction
- Streamlit GUI

Not implemented in this review:

- Material optimization
- Target-performance inverse design
- Genetic algorithms
- GWP optimization
- Shear prediction
- Failure-mode classification

## Default Dataset

The default dataset is:

```text
data/ScienceDirect_Optimized_SpanLength_With_ConcreteGrade.xlsx
```

The older `data/dataset_span_mm.csv` file was removed. The loader supports Excel and CSV files, but this Excel workbook is now the default.

## Input Parameters

The current prediction module uses available dataset-supported parameters:

- `Compressive Strength fc' (MPa)`
- `w/b Ratio`
- `Fly Ash (%)`
- `GGBS (%)`
- `Recycled Aggregate (%)`
- `Yield Strength fy (MPa)`
- `Stud Dia. ds (mm)`
- `Span L (mm)`
- `Beam Depth h (mm)`
- `Beam Width (mm)` in the GUI, backed by the available dataset column `Slab Thickness hc (mm)`
- `Steel Grade`
- `Shear Connector Type`

The workbook column `Recycled Agg. (%)` is safely mapped to `Recycled Aggregate (%)`.

## Prediction Targets

- `Target: Capacity Pu (kN)` as ultimate load
- `Target: Deflection δ (mm)` as deflection

The following columns are excluded from predictors to avoid leakage or future-scope behavior:

- `Target: Capacity Pu (kN)`
- `Target: Deflection δ (mm)`
- `Target: Shear Qu (kN)`
- `Target: Failure Mode`
- `Concrete GWP (kg CO2e/m3)`

## Models

For each target, the project trains:

- `xgboost.XGBRegressor`
- `catboost.CatBoostRegressor`

Models are tuned with 5-fold cross-validation and evaluated on an 80/20 holdout split using:

- R²
- MAE
- RMSE

Best model selection uses highest R², then lower RMSE, then lower MAE.

## Explainable AI

SHAP is used for:

- Global feature-importance charts
- SHAP summary plots
- Per-prediction contribution tables
- Deterministic explanation text generated from actual SHAP signs and magnitudes

Feature importance describes the trained model's predictive behavior and does not by itself prove physical causation.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_performance.py
```

This validates the dataset, trains XGBoost and CatBoost for ultimate load and deflection, saves model bundles, writes metrics, and generates EDA/evaluation/SHAP figures.

The earlier compressive-strength-only training script remains available as:

```bash
python train.py
```

## Run GUI

```bash
streamlit run app.py
```

The GUI loads saved model bundles and does not retrain on each interaction.
