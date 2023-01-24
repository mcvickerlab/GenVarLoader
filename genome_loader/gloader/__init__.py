from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from genome_loader.utils import PathType


class GenomeLoader(Protocol):
    """Protocol defining behavior of genome loaders."""

    embedding: str
    ref_genome_path: PathType
    contigs: NDArray[np.str_]
    spec: Optional[NDArray[np.bytes_]] = None

    def sel(self) -> NDArray:
        ...

    def sel_from_bed(self) -> NDArray:
        ...
