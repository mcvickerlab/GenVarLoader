from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

def intervals(
    paths: Sequence[Union[str, Path]],
    contig: str,
    starts: NDArray[np.int32],
    ends: NDArray[np.int32],
) -> Tuple[NDArray[np.uint32], NDArray[np.float32], NDArray[np.int32]]:
    """Load intervals from BigWig files.

    Parameters
    ----------
    paths : List[str | Path]
        Paths to BigWig files.
    contig : str
        Contig name.
    starts : NDArray[int32]
        Start positions.
    ends : NDArray[int32]
        End positions.

    Returns
    -------
    coordinates : NDArray[uint32]
        Shape = (intervals) Coordinates.
    values : NDArray[float32]
        Shape = (intervals) Values.
    n_per_query : NDArray[int32]
        Shape = (samples, regions) Number of intervals per query.
    """
    ...
