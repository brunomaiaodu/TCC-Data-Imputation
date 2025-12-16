from __future__ import annotations

from typing import List
from io import BytesIO

from fastapi import APIRouter, HTTPException, Query, Response

from ..core.loaders import get_dataset, list_datasets
from ..schemas.dataset import Dataset
from ..services.map_render import render_dataset_map

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/", response_model=List[Dataset])
def read_datasets() -> List[Dataset]:
    return [Dataset(**dataset) for dataset in list_datasets()]


@router.get("/{dataset_id}", response_model=Dataset)
def read_dataset(dataset_id: str) -> Dataset:
    try:
        dataset = get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Dataset(**dataset)


@router.get("/{dataset_id}/map", response_class=Response)
def render_map(
    dataset_id: str,
    selected_points: str | None = Query(default=None, description="IDs separados por vírgula para destacar"),
    highlight_point: str | None = Query(default=None, description="ID único para destaque principal"),
) -> Response:
    try:
        dataset = get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    selected_set = set(filter(None, (selected_points or "").split(",")))
    image_bytes, media_type = render_dataset_map(dataset, selected_set, highlight_point)
    return Response(content=image_bytes.getvalue(), media_type=media_type)
