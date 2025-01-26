from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = []

def count_intervals(
    paths: Sequence[Union[str, Path]],
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
    ...

def intervals(
    paths: Sequence[Union[str, Path]],
    contig: str,
    starts: NDArray[np.int32],
    ends: NDArray[np.int32],
    offsets: NDArray[np.int64],
) -> Tuple[NDArray[np.uint32], NDArray[np.float32]]:
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
    ...
