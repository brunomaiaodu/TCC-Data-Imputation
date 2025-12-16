from __future__ import annotations

import math
from typing import Iterable, List


def _masked_pairs(
    truth: Iterable[float],
    estimate: Iterable[float],
    mask: Iterable[bool],
) -> List[tuple[float, float]]:
    pairs: List[tuple[float, float]] = []
    for tgt, pred, is_missing in zip(truth, estimate, mask):
        if is_missing:
            pairs.append((float(tgt), float(pred)))
    return pairs


def mae(truth: Iterable[float], estimate: Iterable[float], mask: Iterable[bool]) -> float:
    points = _masked_pairs(truth, estimate, mask)
    if not points:
        return 0.0
    return sum(abs(t - p) for t, p in points) / len(points)


def mse(truth: Iterable[float], estimate: Iterable[float], mask: Iterable[bool]) -> float:
    points = _masked_pairs(truth, estimate, mask)
    if not points:
        return 0.0
    return sum((t - p) ** 2 for t, p in points) / len(points)


def rmse(truth: Iterable[float], estimate: Iterable[float], mask: Iterable[bool]) -> float:
    return math.sqrt(mse(truth, estimate, mask))


def compute_all(truth: List[float], estimate: List[float], mask: List[bool]) -> dict:
    return {"mae": mae(truth, estimate, mask), "mse": mse(truth, estimate, mask), "rmse": rmse(truth, estimate, mask)}
