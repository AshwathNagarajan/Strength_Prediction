"""Central project configuration."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
SHAP_DIR = OUTPUT_DIR / "shap"

DATASET_PATH = DATA_DIR / "ScienceDirect_Optimized_SpanLength_With_ConcreteGrade.xlsx"

FEATURES = [
    "w/b Ratio",
    "Fly Ash (%)",
    "GGBS (%)",
    "Recycled Aggregate (%)",
]
TARGET = "Compressive Strength fc' (MPa)"

MODEL_FILES = {
    "XGBoost": MODEL_DIR / "xgboost_model.pkl",
    "CatBoost": MODEL_DIR / "catboost_model.pkl",
}
BEST_MODEL_FILE = MODEL_DIR / "best_model.pkl"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
FEATURE_RANGES_FILE = MODEL_DIR / "feature_ranges.json"
