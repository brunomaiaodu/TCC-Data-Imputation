from __future__ import annotations

from typing import List, Optional


def impute(y_masked: List[Optional[float]], mask: List[bool], context: dict | None = None) -> List[float]:
    """Graph-inspired fill that averages nearby temporal windows."""
    neighborhood = (context or {}).get("graph_window", 6)
    filled: List[float] = [float(v) if v is not None else 0.0 for v in y_masked]

    for idx, is_missing in enumerate(mask):
        if not is_missing:
            continue
        start = max(0, idx - neighborhood)
        end = min(len(filled), idx + neighborhood)
        window = [filled[pos] for pos in range(start, end) if not mask[pos] or pos == idx]
        if window:
            filled[idx] = sum(window) / len(window)
        elif idx > 0:
            filled[idx] = filled[idx - 1]
    return filled
