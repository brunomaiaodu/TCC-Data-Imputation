from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas.experiment import ExperimentCreateRequest
from ..state import experiment_store

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/")
def create_experiment(request: ExperimentCreateRequest) -> dict:
    try:
        return experiment_store.create_experiment(request.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str) -> dict:
    try:
        return experiment_store.get_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc
