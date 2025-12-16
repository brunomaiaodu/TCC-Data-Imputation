from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..state import experiment_store

router = APIRouter(prefix="/experiments/{experiment_id}/points", tags=["points"])


@router.get("/")
def list_points(experiment_id: str) -> list[dict]:
    try:
        experiment = experiment_store.get_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
    return experiment["points_overview"]


@router.get("/{point_id}")
def read_point(experiment_id: str, point_id: str) -> dict:
    try:
        return experiment_store.get_point_series(experiment_id, point_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Point not found for experiment") from exc
