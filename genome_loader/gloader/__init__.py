from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from genome_loader.utils import PathType


class GenomeLoader(Protocol):

    embedding: str
    ref_genome_path: PathType
    contigs: NDArray[np.str_]
    spec: Optional[NDArray[np.bytes_]] = None

    def __init__(self) -> None:
        pass

    def sel(self):
        pass

    def sel_from_bed(self):
        pass
