from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = []

def count_intervals(
    paths: Sequence[str | Path],
    contig: str,
    starts: NDArray[np.int32],
    ends: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Count how many intervals from BigWig files overlap with each region.

    Parameters
    ----------
    paths : Sequence[str | Path]
        Paths to BigWig files.
    contig : str
        Contig name.
    starts : NDArray[int32]
        Start positions.
    ends : NDArray[int32]
        End positions.

    Returns
    -------
    n_per_query : NDArray[int32]
        Shape = (regions, samples) Number of intervals per query.
    """

def bigwig_intervals(
    paths: Sequence[str | Path],
    contig: str,
    starts: NDArray[np.int32],
    ends: NDArray[np.int32],
    offsets: NDArray[np.int64],
) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
    """Load intervals from BigWig files.

    Parameters
    ----------
    paths : Sequence[str | Path]
        Paths to BigWig files.
    contig : str
        Contig name.
    starts : NDArray[int32]
        Start positions.
    ends : NDArray[int32]
        End positions.
    offsets : NDArray[int64]
        Offsets corresponding to the returned interval data of shape (regions, samples). Can be
        computed from the number of intervals per query, e.g. with the count_intervals function.

    Returns
    -------
    coordinates : NDArray[uint32]
        Shape = (intervals, 2) Coordinates.
    values : NDArray[float32]
        Shape = (intervals) Values.
    """

def intervals_to_tracks(
    offset_idxs: NDArray[np.int64],
    starts: NDArray[np.int32],
    itv_starts: NDArray[np.int32],
    itv_ends: NDArray[np.int32],
    itv_values: NDArray[np.float32],
    itv_offsets: NDArray[np.int64],
    out: NDArray[np.float32],
    out_offsets: NDArray[np.int64],
) -> None:
    """Paint base-pair-resolution tracks from intervals, writing ``out`` in place.

    Rust backend for the dispatched ``intervals_to_tracks`` kernel (byte-identical
    to the numba reference in ``_dataset/_intervals.py``). Zeros ``out`` then, per
    query, copies each interval's value into its base-pair slice. Assumes intervals
    are sorted by start and non-overlapping; interval starts before the query start
    are clipped to the query window (per #242).
    """
