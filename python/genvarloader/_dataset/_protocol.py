"""Shared :class:`Reconstructor` Protocol.

Lives in its own module so the leaf reconstructor classes (``Ref``, ``Haps``,
``Tracks``) and the compound classes / factory in ``_reconstruct.py`` can all
depend on it without circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._splice import SplicePlan

T = TypeVar("T", covariant=True)


class Reconstructor(Protocol[T]):
    """Reconstructs data on-the-fly. e.g. personalized sequences, tracks, etc."""

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        flat: bool = False,
    ) -> T:
        """``flat`` only changes behavior for :class:`Haps` producing
        ``RaggedVariants`` (it returns a flat ``_FlatVariants`` instead); all
        other reconstructors are already flat-native and accept-and-ignore it."""
        ...
