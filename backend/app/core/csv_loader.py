from __future__ import annotations

import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[3]


def _resolve_path(file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _parse_timestamp(value: str) -> Tuple[str, datetime]:
    text = (value or "").strip()
    dt: datetime
    try:
        dt = datetime.fromisoformat(text.replace("Z", ""))
    except ValueError:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S"), dt


def _clean_number(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "na", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_point_file(file_path: Path) -> Dict:
    timestamps: List[str] = []
    datetime_axis: List[datetime] = []
    original: List[float] = []
    missing: List[float | None] = []
    imputed: List[float] = []
    masks: List[bool] = []

    with file_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iso_ts, dt = _parse_timestamp(row.get("timestamp", ""))
            timestamps.append(iso_ts)
            datetime_axis.append(dt)

            orig_val = _clean_number(row.get("original"))
            imput_val = _clean_number(row.get("imputed"))
            miss_val = _clean_number(row.get("missing"))
            mask_flag = miss_val is None

            base_original = float(orig_val or 0.0)
            original.append(base_original)
            imputed.append(float(imput_val if imput_val is not None else base_original))
            missing.append(miss_val)
            masks.append(mask_flag)

    return {
        "time": timestamps,
        "datetime_axis": datetime_axis,
        "original": original,
        "missing": missing,
        "imputed": imputed,
        "mask": masks,
        "missing_ratio": sum(masks) / len(masks) if masks else 0.0,
    }


def _infer_resolution_hours(datetimes: List[datetime]) -> float:
    if len(datetimes) < 2:
        return 0.0
    deltas = [
        (t2 - t1).total_seconds() / 3600
        for t1, t2 in zip(datetimes, datetimes[1:])
        if t2 and t1 and t2 > t1
    ]
    if not deltas:
        return 0.0
    try:
        return round(statistics.mode(deltas), 2)
    except statistics.StatisticsError:
        return round(statistics.fmean(deltas), 2)


def _span_label(datetimes: List[datetime]) -> str:
    if not datetimes:
        return "Período não informado"
    start = min(datetimes)
    end = max(datetimes)
    return f"{start:%d/%m/%Y} a {end:%d/%m/%Y}"


def load_csv_run(dataset: Dict, method_id: str, ratio: float) -> Dict:
    data_source = dataset.get("data_source", {})
    processed_dir = _resolve_path(data_source.get("processed_dir", "preprocessing/processed"))
    template = data_source.get("filename_template") or "{method}_{point}.csv"

    available = dataset.get("available_missing_ratios") or []
    matched_ratio = min(available, key=lambda value: abs(value - ratio)) if available else ratio
    percent_label = int(round(matched_ratio * 100))

    series_by_point: Dict[str, Dict] = {}
    reference_time: List[str] = []
    reference_dt: List[datetime] = []

    for point in dataset.get("points", []):
        file_name = template.format(pct=percent_label, method=method_id, point=point["id"])
        file_path = processed_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo {file_path} não encontrado para {dataset['id']} ({method_id}).")
        point_series = _load_point_file(file_path)
        if not reference_time:
            reference_time = point_series["time"]
            reference_dt = point_series["datetime_axis"]
        series_by_point[point["id"]] = {
            "time": point_series["time"],
            "original": point_series["original"],
            "masked": point_series["missing"],
            "imputed": point_series["imputed"],
            "mask": point_series["mask"],
        }

    return {
        "ratio": matched_ratio,
        "time_axis": reference_time,
        "series_by_point": series_by_point,
        "time_span": _span_label(reference_dt),
        "resolution_hours": _infer_resolution_hours(reference_dt),
    }
