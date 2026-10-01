from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache

import joblib

from app.core.config import MODEL_METADATA_FILE, MODELS_DIR, PREPROCESSORS_DIR
from app.utils.files import atomic_json, read_json


@lru_cache(maxsize=4)
def _load_artifact(path: str, modified_ns: int) -> dict:
    return joblib.load(path)


class ModelRegistry:
    def metadata(self) -> dict:
        return read_json(MODEL_METADATA_FILE, {})

    def save(self, bundle: dict, metadata: dict) -> Path:
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_dir = MODELS_DIR / version
        version_dir.mkdir(parents=True, exist_ok=False)
        models = bundle["models"]
        for name, pipeline in models.items():
            joblib.dump(pipeline.named_steps["model"], version_dir / f"{name.lower()}_model.joblib")
        active_pipeline = bundle["pipeline"]
        joblib.dump(active_pipeline.named_steps["preprocessor"], version_dir / "preprocessor.joblib")
        joblib.dump(bundle, version_dir / "best_model.joblib")
        metadata = {**metadata, "version": version, "artifact_dir": str(version_dir.resolve())}
        atomic_json(version_dir / "metadata.json", metadata)
        atomic_json(version_dir / "metrics.json", metadata["metrics"])
        joblib.dump(active_pipeline.named_steps["preprocessor"], PREPROCESSORS_DIR / "preprocessor.joblib")
        atomic_json(MODEL_METADATA_FILE, metadata)
        return version_dir

    def load(self) -> dict:
        metadata = self.metadata()
        path = Path(metadata.get("artifact_dir", "")) / "best_model.joblib"
        if not path.exists():
            raise FileNotFoundError("No trained model is available. Train models first.")
        return _load_artifact(str(path), path.stat().st_mtime_ns)

    def list_versions(self) -> list[dict]:
        versions = []
        for directory in sorted(MODELS_DIR.iterdir(), reverse=True):
            metadata_file = directory / "metadata.json"
            if directory.is_dir() and metadata_file.exists():
                versions.append(read_json(metadata_file, {}))
        return versions
