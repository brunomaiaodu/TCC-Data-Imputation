from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]


def _build_grid_points(
    base_lat: float,
    base_lon: float,
    rows: int,
    cols: int,
    lat_step: float,
    lon_step: float,
    prefix: str,
    region: str,
) -> List[Dict]:
    """Generate a regular rectangular grid of points to keep positions aligned on the map."""
    points: List[Dict] = []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            lat = base_lat + row * lat_step
            lon = base_lon + col * lon_step
            idx += 1
            points.append(
                {
                    "id": f"{prefix}_{idx:02d}",
                    "label": f"Grid {idx:02d}",
                    "lat": lat,
                    "lon": lon,
                    "dataset_index": idx - 1,
                    "region": region,
                },
            )
    return points


def _roms_points() -> List[Dict]:
    # Regular grid with consistent spacing (spread < 1º) to keep points agrupados no mapa.
    return _build_grid_points(
        base_lat=-23.0,
        base_lon=-43.0,
        rows=3,
        cols=4,
        lat_step=0.2,
        lon_step=0.2,
        prefix="roms",
        region="Área de teste",
    )


def _aqi_points() -> List[Dict]:
    # Grade regular compacta (spread < 1º) para facilitar visualização única.
    return _build_grid_points(
        base_lat=39.5,
        base_lon=116.0,
        rows=4,
        cols=9,
        lat_step=0.15,
        lon_step=0.15,
        prefix="aqi",
        region="Área urbana de teste",
    )


def _resolve_path(file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def _load_points_from_csv(path: Path, dataset: Dict) -> List[Dict]:
    """Carrega pontos de um CSV (id, latitude, longitude, node)."""
    points: List[Dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            raw_id = row.get("point_id") or row.get("id") or row.get("node") or idx + 1
            point_id = str(raw_id).strip()
            try:
                lat = float(row.get("latitude") or row.get("lat") or 0.0)
                lon = float(row.get("longitude") or row.get("lon") or 0.0)
            except ValueError:
                continue
            dataset_index = int(row.get("node") or idx)
            points.append(
                {
                    "id": point_id,
                    "label": f"Malha {int(point_id):02d}",
                    "lat": lat,
                    "lon": lon,
                    "dataset_index": dataset_index,
                    "region": dataset.get("region_label", "Grade"),
                },
            )
    return points


def _load_coords(dataset: Dict) -> List[Dict]:
    data_source = dataset.get("data_source", {})
    coords_file = data_source.get("coords_file")
    if not coords_file:
        return []

    path = _resolve_path(coords_file)
    if not path.exists():
        return []

    if path.suffix.lower() == ".csv":
        points = _load_points_from_csv(path, dataset)
        if points:
            return points

    coords = np.load(path)
    lat_grid = coords.get("lat") or coords.get("lats")
    lon_grid = coords.get("lon") or coords.get("lons")
    if lat_grid is None or lon_grid is None:
        return []

    points: List[Dict] = []
    lat_grid = np.array(lat_grid)
    lon_grid = np.array(lon_grid)
    rows, cols = lat_grid.shape[:2]
    for i in range(rows):
        for j in range(cols):
            points.append(
                {
                    "id": f"{dataset['id']}_{i:02d}_{j:02d}",
                    "label": f"Ponto {i:02d}-{j:02d}",
                    "lat": float(lat_grid[i, j]),
                    "lon": float(lon_grid[i, j]),
                    "dataset_index": i * cols + j,
                    "region": dataset.get("region_label", "Grade"),
                },
            )
    return points


def _populate_points(dataset: Dict) -> None:
    if dataset.get("_points_loaded"):
        return

    coords_points = _load_coords(dataset)
    if coords_points:
        dataset["points"] = coords_points
        dataset["statistics"]["points"] = len(coords_points)
        dataset["_points_loaded"] = True
        return

    # Fallback para manter o app funcional se o arquivo de coordenadas ainda não foi fornecido.
    generator = _roms_points if dataset["id"] == "roms" else _aqi_points
    dataset["points"] = generator()
    dataset["_points_loaded"] = True


DATASETS: Dict[str, Dict] = {
    "roms": {
        "id": "roms",
        "name": "ROMS · Altura da Superfície do Mar (SSH)",
        "description": "Altura da superfície do mar (SSH) gerada pelo ROMS na grade costeira.",
        "variables": [
            {"id": "ssh", "label": "SSH", "units": "m"},
        ],
        "default_variable": "ssh",
        "map_center": {"lat": -23.0, "lon": -43.0, "zoom": 6},
        "color": "#0f83ff",
        "region_label": "Malha ROMS",
        "available_missing_ratios": [0.25, 0.30, 0.50, 0.75],
        "data_source": {
            "coords_file": "backend/app/data/roms_latlon_enriched.csv",
            "processed_dir": "preprocessing/processed",
            "filename_template": "roms_{pct}pct_{method}_{point}.csv",
        },
        "statistics": {
            "points": 36,
            "variables": 1,
            "native_missing_ratio": 0.25,
            "avg_gap_hours": 2.0,
            "time_span": "Mar-Jul/2005",
            "resolution": "2 horas",
        },
        "failure_modes": [
            {
                "id": "general",
                "name": "Falha geral (percentual fixo)",
                "description": "Use os percentuais pré-processados de falha para toda a malha.",
                "recommended_ratio": 0.25,
                "block_range": [24, 72],
            },
        ],
        "points": [],
    },
    "aqi": {
        "id": "aqi",
        "name": "AQI-36 · PM2.5",
        "description": "Qualidade do ar (36 estações) focada em material particulado fino (PM2.5).",
        "variables": [
            {"id": "pm25", "label": "PM2.5", "units": "µg/m³"},
        ],
        "default_variable": "pm25",
        "map_center": {"lat": 39.9, "lon": 116.4, "zoom": 9},
        "color": "#f97316",
        "region_label": "Área urbana de teste",
        "available_missing_ratios": [0.23],
        "data_source": {
            "coords_file": "backend/app/data/aqi36_latlon_enriched.csv",
            "processed_dir": "preprocessing/processed",
            "filename_template": "aqi36_{method}_{point}.csv",
        },
        "statistics": {
            "points": 0,
            "variables": 1,
            "native_missing_ratio": 0.23,
            "avg_gap_hours": 1.0,
            "time_span": "2015 · Jan-Abr",
            "resolution": "1 hora",
        },
        "failure_modes": [
            {
                "id": "original",
                "name": "Falha original",
                "description": "Apenas o percentual original do conjunto (pré-mascarado).",
                "recommended_ratio": 0.23,
                "block_range": [3, 12],
            },
        ],
        "points": [],
    },
}


METHODS = [
    {
        "id": "knn",
        "name": "KNN Imputer",
        "category": "Estatístico clássico",
        "latency": "4.5s",
        "strengths": ["Bom para lacunas curtas", "Não exige treinamento pesado"],
        "limitations": ["Depende de normalização", "Perde qualidade em falhas extensas"],
    },
    {
        "id": "spin",
        "name": "SPIN",
        "category": "Deep Learning com GNNs e atenção",
        "latency": "34.7s",
        "strengths": ["Explora dependências espaço-temporais", "Mecanismo de atenção em grafos para priorizar nós relevantes"],
        "limitations": ["Requer topologia de malha bem definida", "Tempo de treino maior"],
    },
]


def list_datasets() -> List[Dict]:
    for dataset in DATASETS.values():
        _populate_points(dataset)
    return list(DATASETS.values())


def get_dataset(dataset_id: str) -> Dict:
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise KeyError(f"Dataset '{dataset_id}' not found.")
    _populate_points(dataset)
    return dataset


def list_methods() -> List[Dict]:
    return METHODS


def get_method(method_id: str) -> Dict:
    for method in METHODS:
        if method["id"] == method_id:
            return method
    raise KeyError(f"Method '{method_id}' not found.")
