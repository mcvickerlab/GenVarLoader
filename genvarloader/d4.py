from pathlib import Path
from typing import List, Union, cast

import numpy as np
import pyd4
from numpy import int64
from numpy.typing import ArrayLike, NDArray

from .types import Reader
from .util import get_rel_starts


class D4(Reader):
    dtype: np.int32  # D4 always returns i32
    chunked = False

    def __init__(self, name: str, path: Union[str, Path]) -> None:
        self.name = name
        self.path = path
        matrix = pyd4.D4File(str(path)).open_all_tracks()
        self.sample_names = cast(List[str], matrix.track_names)
        self.tracks = None
        self.contigs = matrix.tracks[0].chrom_names()

        self.sizes = {"sample": len(self.sample_names)}
        self.coords = {"sample": np.asarray(self.sample_names)}

    def rev_strand_fn(self, x):
        return x[..., ::-1]

    def read(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        **kwargs,
    ) -> NDArray[np.int32]:
        contig = self.normalize_contig_name(contig, self.contigs)
        if contig is None:
            raise ValueError(f"Contig {contig} not found in D4 file.")

        samples = kwargs.get("sample", self.sample_names)
        if missing := set(samples).difference(self.sample_names):
            raise ValueError(f"Samples {missing} not found in D4 file.")

        starts = np.atleast_1d(np.asarray(starts, dtype=int64))
        ends = np.atleast_1d(np.asarray(ends, dtype=int64))

        if self.tracks is None:
            self.tracks = dict(
                zip(
                    self.sample_names,
                    pyd4.D4File(str(self.path)).open_all_tracks().tracks,
                )
            )

        values = np.empty((len(samples), (ends - starts).sum()), dtype=np.int32)
        rel_starts = get_rel_starts(starts, ends)
        rel_ends = rel_starts + (ends - starts)
        for s, e, r_s, r_e in zip(starts, ends, rel_starts, rel_ends):
            for i, sample in enumerate(samples):
                values[i, r_s:r_e] = self.tracks[sample][contig, s, e]

        return values
