from __future__ import annotations

from typing import Dict, List, Optional

from . import arima, knn, pristi, spin

IMPUTERS = {
    "knn": knn.impute,
    "arima": arima.impute,
    "pristi": pristi.impute,
    "spin": spin.impute,
}


def run_imputation(method_id: str, y_masked: List[Optional[float]], mask: List[bool], context: dict | None = None) -> List[float]:
    try:
        fn = IMPUTERS[method_id]
    except KeyError as exc:
        raise ValueError(f"Unknown imputation method '{method_id}'") from exc
    return fn(y_masked, mask, context or {})
