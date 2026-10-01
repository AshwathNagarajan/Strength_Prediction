from __future__ import annotations

import json
import os
import tempfile
import math
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)): return [json_safe(item) for item in value]
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float): return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None: return value
    return str(value)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, indent=2, ensure_ascii=True, allow_nan=False)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
