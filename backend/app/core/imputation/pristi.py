from __future__ import annotations

import math
from typing import List, Optional


def impute(y_masked: List[Optional[float]], mask: List[bool], context: dict | None = None) -> List[float]:
    """Approximate a diffusion-based fill by applying a smooth exponential filter."""
    alpha = (context or {}).get("alpha", 0.6)
    smoothed: List[float] = []
    carry = float(y_masked[0] or 0.0)
    for value, is_missing in zip(y_masked, mask):
        target = carry if value is None or is_missing else float(value)
        carry = alpha * target + (1 - alpha) * carry
        smoothed.append(carry)

    # second pass using sinusoidal residuals
    result = smoothed[:]
    for idx, is_missing in enumerate(mask):
        if is_missing:
            phase = math.sin(idx / 8.0)
            result[idx] = smoothed[idx] + 0.1 * phase
    return result
