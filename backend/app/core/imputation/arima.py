from __future__ import annotations

from typing import List, Optional


def impute(y_masked: List[Optional[float]], mask: List[bool], context: dict | None = None) -> List[float]:
    """Mimic an ARIMA-style interpolation by combining linear interpolation and seasonal mean."""
    filled: List[float] = [float(v) if v is not None else 0.0 for v in y_masked]
    season = (context or {}).get("season_period", 24)

    # forward fill
    last_value = filled[0]
    for idx, is_missing in enumerate(mask):
        if not is_missing and y_masked[idx] is not None:
            last_value = float(y_masked[idx])
        else:
            seasonal_idx = max(0, idx - season)
            seasonal_value = filled[seasonal_idx] if idx >= season else last_value
            filled[idx] = (last_value + seasonal_value) / 2

    # backward smoothing for trailing NaNs
    next_value = filled[-1]
    for idx in range(len(mask) - 1, -1, -1):
        if not mask[idx]:
            next_value = filled[idx]
        else:
            filled[idx] = (filled[idx] + next_value) / 2
    return filled
