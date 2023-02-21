import asyncio
from pathlib import Path
from typing import Dict, Union, cast

import numpy as np
import zarr
from numpy.typing import NDArray

from genvarloader.loaders.types import Queries, _TStore
from genvarloader.loaders.utils import ts_readonly_zarr
from genvarloader.types import PathType


class Coverage:
    path: Path

    def __init__(self, zarr_path: PathType) -> None:
        self.path = Path(zarr_path)
        root = zarr.open_group(self.path, mode="r")

        self.contig_lengths: Dict[str, int] = root.attrs["contig_lengths"]

        # Iterate over arrays under the group and grab them as TensorStores
        self.tstores: Dict[str, _TStore] = {}

        def add_array_to_tstores(p: str, val: Union[zarr.Group, zarr.Array]):
            if isinstance(val, zarr.Array):
                self.tstores[p] = ts_readonly_zarr(self.path / p).result()

        root.visit(add_array_to_tstores)

        # Expect shape to be (length) or (length alphabet) depending on whether
        # specific nucleotides are counted. We get nucleotide counts when using
        # `genvarloader coverage depth-only`
        # aka `pysam.bam.AlignmentFile::count_coverage`
        # which uses an alphabet of 'ACGT'.
        for name, arr in root.arrays():
            if isinstance(arr, zarr.Array) and len(arr.shape) == 2:
                self.acgt_counts = True
            else:
                self.acgt_counts = False
            break

    def sel(self, queries: Queries, length: int, **kwargs) -> NDArray[np.uint8]:
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(
        self, queries: Queries, length: int, **kwargs
    ) -> NDArray[np.uint8]:

        out_shape = [len(queries), length]

        queries["end"] = queries.start + length
        # map negative starts to 0
        queries["in_start"] = queries.start.clip(lower=0)
        # map ends > contig length to contig length
        queries["contig_length"] = queries.contig.replace(self.contig_lengths).astype(
            int
        )
        queries["in_end"] = np.minimum(queries.end, queries.contig_length)
        # get start, end index in output array
        queries["out_start"] = queries.in_start - queries.start
        queries["out_end"] = queries.in_end - queries.end

        def get_read(query):
            contig = query.contig
            return self.tstores[contig][query.in_start : query.in_end].read()

        # (q l [a])
        reads = await asyncio.gather(
            *[get_read(query) for query in queries.itertuples()]
        )

        # init array that will pad out-of-bound sequences
        out = cast(NDArray[np.uint8], np.zeros(out_shape, "u1"))
        for i, (read, query) in enumerate(zip(reads, queries.itertuples())):
            # (1 l [a]) = (l [a])
            out[i, query.out_start : query.out_end] = read

        # reverse complement negative stranded queries
        to_rev_comp = cast(NDArray[np.bool_], (queries["strand"] == "-").values)
        if self.acgt_counts:
            axes = (-2, -1)  # (l a)
        else:
            axes = -1  # (l)
        out[to_rev_comp] = np.flip(out[to_rev_comp], axis=axes)

        return out
