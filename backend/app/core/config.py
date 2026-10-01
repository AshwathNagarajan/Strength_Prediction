import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")
    app_name: str = "Sustainable Beam AI"
    debug: bool = False
    frontend_url: str = "http://localhost:5173"
    hf_provider: str = "disabled"
    hf_model_id: str = ""
    hf_token: str = ""
    model_dir: str = "artifacts/models"
    data_dir: str = "data"


settings = Settings()


def hf_settings() -> tuple[str, str, str]:
    """Return HF configuration while allowing process overrides in tests/deployments."""
    return (
        os.getenv("HF_PROVIDER", settings.hf_provider).lower(),
        os.getenv("HF_MODEL_ID", settings.hf_model_id).strip(),
        os.getenv("HF_TOKEN", settings.hf_token).strip(),
    )
DATA_DIR = BACKEND_DIR / settings.data_dir
RAW_DIR = DATA_DIR / "raw"
CONFIG_DIR = BACKEND_DIR / "config"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
MODELS_DIR = BACKEND_DIR / settings.model_dir
PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TRAINING_STATUS_FILE = CONFIG_DIR / "training_status.json"
PREDICTION_HISTORY_FILE = REPORTS_DIR / "prediction_history.json"
LATEST_PREDICTION_FILE = REPORTS_DIR / "latest_prediction.json"
LATEST_OPTIMIZATION_FILE = REPORTS_DIR / "latest_optimization.json"
GLOBAL_SHAP_FILE = REPORTS_DIR / "global_shap.json"

SCHEMA_FILE = CONFIG_DIR / "dataset_schema.json"
DATASET_STATE_FILE = CONFIG_DIR / "dataset_state.json"
MODEL_METADATA_FILE = CONFIG_DIR / "model_metadata.json"
DOMAIN_FILE = CONFIG_DIR / "feature_metadata.json"

for directory in (RAW_DIR, CONFIG_DIR, MODELS_DIR, PREPROCESSORS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
