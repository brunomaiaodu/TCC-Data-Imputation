from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class DatasetPoint(BaseModel):
    id: str
    label: str
    lat: float
    lon: float
    dataset_index: int
    region: str


class DatasetVariable(BaseModel):
    id: str
    label: str
    units: Optional[str]


class FailureMode(BaseModel):
    id: Literal["general", "selected", "original"]
    name: str
    description: str
    recommended_ratio: float
    block_range: List[int]


class Dataset(BaseModel):
    id: str
    name: str
    description: str
    variables: List[DatasetVariable]
    default_variable: str
    statistics: dict
    map_center: dict
    color: str
    available_missing_ratios: Optional[List[float]] = None
    failure_modes: List[FailureMode]
    points: List[DatasetPoint]
