from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray


class Reader(Protocol):
    name: str
    bytes_per_length: int

    def read(self, contig: str, start: int, end: int) -> NDArray:
        ...


class ToZarr(Protocol):
    def to_zarr(self, store: Path):
        ...


class Variants(Protocol):
    n_samples: int
    ploidy: int

    def read(
        self, contig: str, start: int, end: int
    ) -> Optional[Tuple[NDArray[np.uint32], NDArray[np.intp], NDArray[np.bytes_]]]:
        ...
