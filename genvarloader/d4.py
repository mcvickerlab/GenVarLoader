from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy import int64
from numpy.typing import NDArray
from pyd4 import D4File, D4Matrix

from .types import Reader


class D4(Reader):
    dtype: np.int32  # D4 always returns i32
    chunked = False

    def __init__(self, name: str, path: Union[str, Path]) -> None:
        self.name = name
        self.d4 = D4File(str(path))

        # self.sizes = ...
        # self.coords = ...

        # self.contig_starts_with_chr = ...

    def rev_strand_fn(self, x):
        return x[..., ::-1]

    def open_tracks(self, tracks: List[str]):
        return D4Matrix(
            [
                D4File(self.d4.get_track_specifier(track_label))
                for track_label in tracks
            ],
            track_names=tracks,
        )

    def read(
        self,
        contig: str,
        starts: NDArray[int64],
        ends: NDArray[int64],
        out: Optional[NDArray] = None,
        **kwargs,
    ) -> NDArray:
        regions = [(contig, start, end) for start, end in zip(starts, ends)]
        np.concatenate(self.d4.load_to_np(regions), axis=-1)

        raise NotImplementedError
