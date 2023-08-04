from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class Reader(ABC):
    name: str
    dtype: np.dtype
    instance_shape: Tuple[int, ...]

    @abstractmethod
    def read(self, contig: str, start: int, end: int) -> NDArray:
        ...
