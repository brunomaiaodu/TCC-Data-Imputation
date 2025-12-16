from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .metrics import compute_all

ROOT_DIR = Path(__file__).resolve().parents[3]


class NPZRun:
    def __init__(
        self,
        dataset_id: str,
        method_id: str,
        ratio: float,
        original: np.ndarray,
        masked: np.ndarray,
        imputed: np.ndarray,
        mask: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> None:
        self.dataset_id = dataset_id
        self.method_id = method_id
        self.ratio = ratio
        self.original = np.asarray(original)
        self.masked = np.asarray(masked)
        self.imputed = np.asarray(imputed)
        self.mask = np.asarray(mask).astype(bool)
        self.lats = np.asarray(lats)
        self.lons = np.asarray(lons)

        if self.original.shape != self.imputed.shape:
            raise ValueError("original e imputed possuem shapes diferentes.")
        if self.original.shape != self.masked.shape:
            raise ValueError("original e masked possuem shapes diferentes.")
        if self.original.shape != self.mask.shape:
            raise ValueError("mask deve ter o mesmo shape das séries.")

    @property
    def time_steps(self) -> int:
        return self.original.shape[0]

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        # Espera shape (T, lat, lon, 1)
        if len(self.original.shape) < 3:
            raise ValueError("Shape inesperado: séries precisam ter dimensões (tempo, lat, lon, canal).")
        return self.original.shape[1], self.original.shape[2]


def _resolve_path(file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _load_coords_from_file(coords_file: Optional[str], spatial_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if coords_file:
        path = _resolve_path(coords_file)
        if path.exists():
            coords = np.load(path)
            lat_grid = coords.get("lat") or coords.get("lats")
            lon_grid = coords.get("lon") or coords.get("lons")
            if lat_grid is not None and lon_grid is not None:
                return np.asarray(lat_grid), np.asarray(lon_grid)

    # Fallback: gera grade sintética centrada em torno da origem para manter a visualização ativa.
    lat_len, lon_len = spatial_shape
    lat_base = -0.5 * lat_len * 0.1
    lon_base = -0.5 * lon_len * 0.1
    lat_grid = np.zeros((lat_len, lon_len))
    lon_grid = np.zeros((lat_len, lon_len))
    for i in range(lat_len):
        for j in range(lon_len):
            lat_grid[i, j] = lat_base + i * 0.1
            lon_grid[i, j] = lon_base + j * 0.1
    return lat_grid, lon_grid


def _select_run_file(dataset: Dict, method_id: str, ratio: float) -> Tuple[str, float]:
    data_source = dataset.get("data_source") or {}
    runs = data_source.get("runs") or []

    if not runs:
        raise ValueError(f"Nenhum experimento pré-computado configurado para {dataset['id']}.")

    # Escolhe o run mais próximo do ratio solicitado.
    closest = min(runs, key=lambda run: abs(run.get("ratio", 0) - ratio))
    selected_ratio = float(closest.get("ratio", ratio))
    files = closest.get("files") or {}
    file_name = files.get(method_id)
    if not file_name:
        raise ValueError(f"Arquivo de resultados não configurado para método '{method_id}' em {dataset['id']}.")
    return file_name, selected_ratio


def load_npz_run(dataset: Dict, method_id: str, ratio: float) -> NPZRun:
    file_name, matched_ratio = _select_run_file(dataset, method_id, ratio)
    path = _resolve_path(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de resultados '{path}' não encontrado para {dataset['id']} ({method_id}).")

    npz = np.load(path)
    original = npz["original"]
    masked = npz.get("missing") or npz.get("masked") or original
    imputed = npz.get("imputed") or original
    mask = npz.get("mask")
    if mask is None:
        raise ValueError("Arquivo .npz precisa conter a chave 'mask'.")

    spatial_shape = (original.shape[1], original.shape[2])
    coords_file = (dataset.get("data_source") or {}).get("coords_file")
    lat_grid, lon_grid = _load_coords_from_file(coords_file, spatial_shape)

    return NPZRun(dataset["id"], method_id, matched_ratio, original, masked, imputed, mask, lat_grid, lon_grid)


def build_points_from_coords(dataset: Dict, lat_grid: np.ndarray, lon_grid: np.ndarray) -> List[Dict]:
    lat_grid = np.asarray(lat_grid)
    lon_grid = np.asarray(lon_grid)
    rows, cols = lat_grid.shape[:2]

    points: List[Dict] = []
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


def flatten_series(array: np.ndarray) -> List[List[float]]:
    """Transforma série (T, lat, lon, 1) em lista de pontos [ [t1..T], ... ]."""
    t, lat_len, lon_len = array.shape[:3]
    flat: List[List[float]] = []
    for i in range(lat_len):
        for j in range(lon_len):
            flat.append([float(v) for v in array[:, i, j, 0]])
    return flat


def flatten_mask(mask: np.ndarray) -> List[List[bool]]:
    t, lat_len, lon_len = mask.shape[:3]
    flat: List[List[bool]] = []
    bool_mask = mask.astype(bool)
    for i in range(lat_len):
        for j in range(lon_len):
            flat.append([bool(v) for v in bool_mask[:, i, j, 0]])
    return flat


def compute_point_metrics(original: List[float], imputed: List[float], mask: List[bool]) -> Dict[str, float]:
    return compute_all(original, imputed, mask)
