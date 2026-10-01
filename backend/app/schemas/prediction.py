from typing import Any

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: dict[str, Any]
    explain: bool = True
