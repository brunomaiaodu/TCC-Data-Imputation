from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


MissingMode = Literal["general", "selected", "original"]


class MissingConfig(BaseModel):
    mode: MissingMode = "general"
    ratio: float = Field(0.1, ge=0.0, le=0.95)
    block_length: int = Field(12, gt=0, le=240)
    selected_points: List[str] = Field(default_factory=list)


class ExperimentCreateRequest(BaseModel):
    dataset_id: str
    variable: str
    method_id: str
    missing_config: MissingConfig
    notes: Optional[str] = None
