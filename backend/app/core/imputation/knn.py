from __future__ import annotations

from typing import List, Optional


def impute(y_masked: List[Optional[float]], mask: List[bool], context: dict | None = None) -> List[float]:
    """Fill gaps using a simple k-nearest neighbors interpolation based on closest valid samples."""
    window = (context or {}).get("window", 2)
    filled: List[float] = [float(v) if v is not None else 0.0 for v in y_masked]

    for idx, is_missing in enumerate(mask):
        if not is_missing:
            continue
        neighbors: List[float] = []
        # look backwards
        step = 1
        while len(neighbors) < window and idx - step >= 0:
            candidate = y_masked[idx - step]
            if candidate is not None:
                neighbors.append(float(candidate))
            step += 1
        # look forward
        step = 1
        while len(neighbors) < window * 2 and idx + step < len(y_masked):
            candidate = y_masked[idx + step]
            if candidate is not None:
                neighbors.append(float(candidate))
            step += 1
        if neighbors:
            filled[idx] = sum(neighbors) / len(neighbors)
        else:
            filled[idx] = filled[idx - 1] if idx > 0 else 0.0
    return filled
