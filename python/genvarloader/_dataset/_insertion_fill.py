from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

REPEAT_5P = 0
REPEAT_5P_NORM = 1
CONSTANT = 2
FLANK_SAMPLE = 3
INTERPOLATE = 4

MAX_PARAMS = 1  # widest strategy uses 1 slot


class InsertionFill:
    """Base class for track insertion fill strategies. Do not instantiate directly."""

    def __init__(self):
        if type(self) is InsertionFill:
            raise TypeError("InsertionFill is abstract; instantiate a subclass.")


@dataclass(slots=True)
class Repeat5p(InsertionFill):
    """Repeat the value at the variant POS across the entire inserted region. Current default behavior."""


@dataclass(slots=True)
class Repeat5pNormalized(InsertionFill):
    """Repeat track[v_rel_pos] / (v_diff + 1) across the inserted region.

    Preserves the sum: when the full insertion stretch is written, the total
    written value equals track[v_rel_pos]. If the insertion is truncated at
    the output boundary, the sum is reduced proportionally.
    """


@dataclass(slots=True)
class Constant(InsertionFill):
    """Write a fixed value at every inserted position.

    Parameters
    ----------
    value
        Value to write. Defaults to NaN.
    """

    value: float = float("nan")


@dataclass(slots=True)
class FlankSample(InsertionFill):
    """Sample (with replacement) from the 2*flank_width+1 reference values
    centered at the variant POS. Each inserted position samples independently.
    Out-of-bounds neighbors are clamped to in-bounds values.

    Parameters
    ----------
    flank_width
        Half-width of the flanking pool. Must be >= 0.
    """

    flank_width: int = 5

    def __post_init__(self) -> None:
        if self.flank_width < 0:
            raise ValueError(f"flank_width must be >= 0, got {self.flank_width}")


@dataclass(slots=True)
class Interpolate(InsertionFill):
    """Polynomial interpolation across the inserted region.

    order=1: linear between track[v_rel_pos] and track[v_rel_pos + 1].
    order=2,3: Lagrange polynomial through ceil((order+1)/2) reference values
    on each side of the variant, clamped at boundaries.

    Parameters
    ----------
    order
        Polynomial order. Must be in {1, 2, 3}.
    """

    order: int = 1

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError(f"Interpolate order must be 1, 2, or 3 (got {self.order})")


def lower(
    strategies: Sequence[InsertionFill],
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Pack strategy instances into numba-friendly arrays.

    Returns
    -------
    strategy_ids
        Shape (n,), int8. One enum value per strategy.
    params
        Shape (n, MAX_PARAMS), float64. Per-strategy parameter slots:
        - Repeat5p / Repeat5pNormalized: unused (all zeros).
        - Constant: [value].
        - FlankSample: [flank_width].
        - Interpolate: [order].
    """
    n = len(strategies)
    ids = np.empty(n, dtype=np.int8)
    params = np.zeros((n, MAX_PARAMS), dtype=np.float64)
    for i, s in enumerate(strategies):
        if isinstance(s, Repeat5p):
            ids[i] = REPEAT_5P
        elif isinstance(s, Repeat5pNormalized):
            ids[i] = REPEAT_5P_NORM
        elif isinstance(s, Constant):
            ids[i] = CONSTANT
            params[i, 0] = s.value
        elif isinstance(s, FlankSample):
            ids[i] = FLANK_SAMPLE
            params[i, 0] = float(s.flank_width)
        elif isinstance(s, Interpolate):
            ids[i] = INTERPOLATE
            params[i, 0] = float(s.order)
        else:
            raise TypeError(f"Unknown InsertionFill subclass: {type(s).__name__}")
    return ids, params
