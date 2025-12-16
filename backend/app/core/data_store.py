from __future__ import annotations

import statistics
import uuid
from datetime import datetime, timedelta
from typing import Dict, List

from . import metrics
from .csv_loader import load_csv_run
from .loaders import get_dataset, get_method
from .npz_loader import build_points_from_coords, load_npz_run


class ExperimentStore:
    def __init__(self) -> None:
        self._experiments: Dict[str, Dict] = {}

    def create_experiment(self, payload: Dict) -> Dict:
        dataset = get_dataset(payload["dataset_id"])
        method = get_method(payload["method_id"])
        requested_ratio = payload["missing_config"]["ratio"]
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

        if dataset.get("data_source", {}).get("filename_template"):
            run = load_csv_run(dataset, method["id"], requested_ratio)
            points = dataset["points"]
            dataset["statistics"]["points"] = len(points)
            dataset["statistics"]["native_missing_ratio"] = run["ratio"]
            dataset["statistics"]["resolution"] = f'{run["resolution_hours"]:.0f} horas' if run.get("resolution_hours") else dataset["statistics"].get("resolution", "2 horas")
            dataset["statistics"]["time_span"] = run.get("time_span") or dataset["statistics"].get("time_span")
            dataset["statistics"]["avg_gap_hours"] = run.get("resolution_hours") or dataset["statistics"].get("avg_gap_hours", 0.0)

            flattened_original = [run["series_by_point"][point["id"]]["original"] for point in points]
            flattened_masked = [run["series_by_point"][point["id"]]["masked"] for point in points]
            flattened_imputed = [run["series_by_point"][point["id"]]["imputed"] for point in points]
            flattened_mask = [run["series_by_point"][point["id"]]["mask"] for point in points]
            time_axis = run["time_axis"]
        else:
            run = load_npz_run(dataset, method["id"], requested_ratio)
            points = build_points_from_coords(dataset, run.lats, run.lons)
            dataset["points"] = points  # garante alinhamento para visualizações e mapa
            dataset["statistics"]["points"] = len(points)
            dataset["statistics"]["native_missing_ratio"] = run.ratio

            flattened_original = _flatten_series(run.original)
            flattened_masked = _flatten_masked_series(run.masked, run.mask)
            flattened_imputed = _flatten_series(run.imputed)
            flattened_mask = _flatten_mask(run.mask)
            time_axis = _time_axis(run.time_steps)

        points_data, points_overview, overall_ratio, masked_points = _build_points_payload(
            points,
            flattened_original,
            flattened_masked,
            flattened_imputed,
            flattened_mask,
            time_axis,
        )

        summary_metrics = _summary_metrics(points_overview)
        timeline = _missing_timeline(flattened_mask, overall_ratio)
        ratio_value = run["ratio"] if isinstance(run, dict) else run.ratio

        experiment_payload = {
            "experiment_id": experiment_id,
            "dataset": {
                "id": dataset["id"],
                "name": dataset["name"],
                "variable": payload["variable"],
                "color": dataset["color"],
                "statistics": dataset["statistics"],
            },
            "method": method,
            "missing_config": {
                "mode": payload["missing_config"]["mode"],
                "ratio": ratio_value,
                "block_length": payload["missing_config"].get("block_length", 0),
                "selected_points": payload["missing_config"].get("selected_points", []),
            },
            "missing_stats": {
                "overall_missing_ratio": overall_ratio,
                "selected_missing_ratio": overall_ratio,
                "original_missing_ratio": dataset["statistics"]["native_missing_ratio"],
                "masked_points": masked_points,
                "total_points": len(points),
                "timeline": timeline,
            },
            "points_overview": points_overview,
            "metrics_overview": summary_metrics,
            "generated_insights": _build_insights(dataset, method, {"ratio": ratio_value}, summary_metrics),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        self._experiments[experiment_id] = {
            "meta": experiment_payload,
            "points": points_data,
        }
        return experiment_payload

    def get_experiment(self, experiment_id: str) -> Dict:
        return self._experiments[experiment_id]["meta"]

    def get_point_series(self, experiment_id: str, point_id: str) -> Dict:
        experiment = self._experiments[experiment_id]
        return experiment["points"][point_id]

    def list_points(self, experiment_id: str) -> List[Dict]:
        experiment = self._experiments[experiment_id]
        return list(experiment["points"].values())


def _flatten_series(series) -> List[List[float]]:
    t, lat_len, lon_len = series.shape[:3]
    flattened: List[List[float]] = []
    for i in range(lat_len):
        for j in range(lon_len):
            flattened.append([float(v) for v in series[:, i, j, 0]])
    return flattened


def _flatten_masked_series(series, mask) -> List[List[float | None]]:
    t, lat_len, lon_len = series.shape[:3]
    flattened: List[List[float | None]] = []
    bool_mask = mask.astype(bool)
    for i in range(lat_len):
        for j in range(lon_len):
            flattened.append(
                [None if bool_mask[k, i, j, 0] else float(series[k, i, j, 0]) for k in range(t)],
            )
    return flattened


def _flatten_mask(mask) -> List[List[bool]]:
    t, lat_len, lon_len = mask.shape[:3]
    flattened: List[List[bool]] = []
    bool_mask = mask.astype(bool)
    for i in range(lat_len):
        for j in range(lon_len):
            flattened.append([bool_mask[k, i, j, 0] for k in range(t)])
    return flattened


def _time_axis(length: int) -> List[str]:
    end = datetime.utcnow()
    start = end - timedelta(hours=length)
    return [(start + timedelta(hours=offset)).isoformat() + "Z" for offset in range(length)]


def _build_points_payload(
    points: List[Dict],
    originals: List[List[float]],
    masked_series: List[List[float | None]],
    imputed_series: List[List[float]],
    masks: List[List[bool]],
    time_axis: List[str],
) -> tuple[Dict[str, Dict], List[Dict], float, int]:
    points_data: Dict[str, Dict] = {}
    points_overview: List[Dict] = []
    total_masked = 0
    masked_points = 0

    for point, original, masked, imputed, mask in zip(points, originals, masked_series, imputed_series, masks):
        point_metrics = metrics.compute_all(original, imputed, mask)
        masked_count = sum(1 for flag in mask if flag)
        total_masked += masked_count
        if masked_count:
            masked_points += 1

        points_data[point["id"]] = {
            "point_id": point["id"],
            "label": point["label"],
            "time": time_axis,
            "original": original,
            "mask": mask,
            "masked": masked,
            "imputed": imputed,
            "metrics": point_metrics,
            "lat": point["lat"],
            "lon": point["lon"],
        }

        points_overview.append(
            {
                "point_id": point["id"],
                "label": point["label"],
                "lat": point["lat"],
                "lon": point["lon"],
                "region": point["region"],
                "metrics": point_metrics,
                "missing_ratio": masked_count / len(mask) if mask else 0.0,
            },
        )

    overall_ratio = total_masked / (len(points) * len(time_axis)) if points and time_axis else 0.0
    return points_data, points_overview, overall_ratio, masked_points


def _summary_metrics(points_overview: List[Dict]) -> Dict[str, float]:
    if not points_overview:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0}
    return {
        "mae": statistics.fmean(item["metrics"]["mae"] for item in points_overview),
        "mse": statistics.fmean(item["metrics"]["mse"] for item in points_overview),
        "rmse": statistics.fmean(item["metrics"]["rmse"] for item in points_overview),
    }


def _missing_timeline(mask_lists: List[List[bool]], overall_ratio: float) -> List[Dict[str, float]]:
    if not mask_lists:
        return []

    series_len = min(len(entry) for entry in mask_lists) if mask_lists else 0
    if series_len == 0:
        return []

    timeline = []
    window = max(1, series_len // 7)

    for idx in range(7):
        start = idx * window
        end = min(series_len, (idx + 1) * window)
        if start >= end:
            continue
        slice_missing = sum(sum(mask[start:end]) for mask in mask_lists)
        slice_total = len(mask_lists) * (end - start)
        slice_ratio = slice_missing / slice_total if slice_total else overall_ratio
        timeline.append(
            {
                "label": f"Janela {idx + 1}",
                "overall": slice_ratio,
                "selected": slice_ratio,
            },
        )
    return timeline


def _build_insights(dataset: Dict, method: Dict, config: Dict, metrics_summary: Dict) -> List[Dict]:
    mode_label = config.get("mode") or "pré-processado"
    block_display = config.get("block_length")
    block_msg = f" em blocos de {block_display}h" if block_display else ""
    ratio = int(round(config.get("ratio", 0) * 100))
    return [
        {
            "title": "Cobertura espacial",
            "details": f"{dataset['name']} com {dataset['statistics']['points']} nós e resolução de {dataset['statistics']['resolution']}.",
        },
        {
            "title": "Foco do método",
            "details": f"{method['name']} ({method['category']}) é forte em {method['strengths'][0].lower()} e exige atenção em {method['limitations'][0].lower()}.",
        },
        {
            "title": "Estratégia de falhas",
            "details": f"Modo '{mode_label}' com {ratio}% de faltas{block_msg}.",
        },
        {
            "title": "Erro atual",
            "details": f"RMSE médio {metrics_summary['rmse']:.3f} avaliado nos pontos mascarados.",
        },
    ]
