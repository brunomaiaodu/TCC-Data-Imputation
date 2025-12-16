from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import datasets, experiments, methods, points

app = FastAPI(
    title="Spatiotemporal Imputation API",
    version="0.1.0",
    description="Backend service powering the Spatiotemporal Imputation Toolbox demo.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(methods.router)
app.include_router(experiments.router)
app.include_router(points.router)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}
