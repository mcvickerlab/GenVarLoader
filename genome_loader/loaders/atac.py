from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from genome_loader.utils import PathType


class Depth(ABC):
    COUNT_TYPE = Literal["Tn5_fragment", "reads"]
    ENCODING = Literal["per_base", "run_length"]

    path: Path
    count_type: COUNT_TYPE
    encoding: ENCODING

    @abstractmethod
    def sel(
        self,
        contigs: NDArray[np.str_],
        starts: NDArray[np.integer],
        ends: NDArray[np.integer],
    ):
        raise NotImplementedError


class H5Depth(Depth):
    COUNT_TYPE = Literal["Tn5_fragment", "reads"]
    ENCODING = Literal["per_base", "run_length"]

    path: Path
    count_type: COUNT_TYPE
    encoding: ENCODING

    def __init__(self, h5_path: PathType) -> None:
        self.path = Path(h5_path)

    def sel(
        self,
        contigs: NDArray[np.str_],
        starts: NDArray[np.integer],
        ends: NDArray[np.integer],
    ):
        return super().sel(contigs, starts, ends)
