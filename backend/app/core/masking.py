from __future__ import annotations

from typing import Iterable, List, Optional


def generate_block_mask(
    length: int,
    ratio: float,
    block_length: int,
    rng,
) -> List[bool]:
    """Return a boolean mask created by applying random contiguous blocks."""
    mask = [False] * length
    if length <= 0 or ratio <= 0:
        return mask

    target_missing = max(1, int(ratio * length))
    block = max(1, min(block_length, length))

    missing = 0
    attempts = 0
    while missing < target_missing and attempts < length * 4:
        start = rng.randint(0, max(0, length - 1))
        end = min(length, start + block)
        for idx in range(start, end):
            if not mask[idx]:
                mask[idx] = True
                missing += 1
        attempts += 1

    return mask


def apply_mask(values: Iterable[float], mask: Iterable[bool]) -> List[Optional[float]]:
    return [None if is_missing else float(value) for value, is_missing in zip(values, mask)]
