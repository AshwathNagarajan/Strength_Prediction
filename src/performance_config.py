"""Configuration for structural performance prediction and explanation."""

from pathlib import Path

from .config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE

DATASET_PATH = DATA_DIR / "ScienceDirect_Optimized_SpanLength_With_ConcreteGrade.xlsx"

PERFORMANCE_MODEL_DIR = MODEL_DIR / "performance"
PERFORMANCE_OUTPUT_DIR = OUTPUT_DIR / "performance"
PERFORMANCE_FIGURES_DIR = PERFORMANCE_OUTPUT_DIR / "figures"
PERFORMANCE_METRICS_DIR = PERFORMANCE_OUTPUT_DIR / "metrics"
PERFORMANCE_SHAP_DIR = PERFORMANCE_OUTPUT_DIR / "shap"

NUMERIC_FEATURES = [
    "Compressive Strength fc' (MPa)",
    "w/b Ratio",
    "Fly Ash (%)",
    "GGBS (%)",
    "Recycled Aggregate (%)",
    "Yield Strength fy (MPa)",
    "Stud Dia. ds (mm)",
    "Span L (mm)",
    "Beam Depth h (mm)",
    "Slab Thickness hc (mm)",
]

CATEGORICAL_FEATURES = [
    "Steel Grade",
    "Shear Connector Type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = {
    "ultimate_load": {
        "column": "Target: Capacity Pu (kN)",
        "label": "Ultimate Load",
        "unit": "kN",
    },
    "deflection": {
        "column": "Target: Deflection δ (mm)",
        "label": "Deflection",
        "unit": "mm",
    },
}

LEAKAGE_COLUMNS = [
    "Target: Capacity Pu (kN)",
    "Target: Deflection δ (mm)",
    "Target: Shear Qu (kN)",
    "Target: Failure Mode",
    "Concrete GWP (kg CO2e/m3)",
]

METADATA_FILE = PERFORMANCE_MODEL_DIR / "performance_model_metadata.json"
FEATURE_PROFILE_FILE = PERFORMANCE_MODEL_DIR / "performance_feature_profile.json"
