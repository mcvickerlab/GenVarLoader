from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .numba import gufunc_multi_slice
from .types import Reader
from .util import _set_uniform_length_around_center, read_bedlike


class GVL:
    def __init__(self, *readers: Reader) -> None:
        self.readers = readers
        pass

    def iter_batches(
        self,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_size: int,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        return_tuples: bool = False,
    ) -> Generator[Union[Dict[str, Any], Tuple[Any]], None, None]:
        rng = np.random.default_rng(seed)

        if isinstance(bed, (str, Path)):
            bed = read_bedlike(bed)

        bed = _set_uniform_length_around_center(bed, fixed_length)
        partitioned_bed = self.partition_bed(bed)

        starts_slice = slice(0, 0)
        batch_slice = slice(0, 0)

        batch = {
            r.name: np.empty(
                (batch_size, *r.instance_shape, fixed_length), dtype=r.dtype
            )
            for r in self.readers
        }

        for partition in partitioned_bed:
            n_regions = len(partition)
            contig: str
            start: int
            end: int
            contig, start, end = (
                partition.select(
                    pl.col("chrom").first(),
                    pl.col("chromStart").min(),
                    pl.col("chromEnd").max(),
                )
                .to_numpy()
                .squeeze()
            )

            # fill batch or take what's in buffer
            new_stop = min(batch_size, batch_slice.stop + n_regions)
            batch_slice = slice(batch_slice.stop, new_stop)
            len_batch_slice = batch_slice.stop - batch_slice.start
            starts_slice = slice(0, len_batch_slice)

            buffers = {r.name: r.read(contig, start, end) for r in self.readers}

            starts = (partition["chromStart"] - start).to_numpy()
            length = end - start
            if shuffle:
                starts = rng.permutation(starts)

            len_unused_buffer = n_regions

            while len_unused_buffer > 0:

                for name in batch:
                    batch[name][batch_slice] = gufunc_multi_slice(
                        buffers[name], starts[starts_slice], length
                    )[..., :length]

                len_unused_buffer = len_unused_buffer - starts_slice.stop

                # ready to yield batch
                if batch_slice.stop == batch_size:
                    # full batch or take what's left in the buffer
                    new_stop = min(batch_size, len_unused_buffer)
                    batch_slice = slice(0, new_stop)
                    if transform is not None:
                        batch = transform(batch)
                    if return_tuples:
                        yield tuple(batch.values())
                    else:
                        yield batch
                # not ready and more data in buffer
                else:
                    # fill batch or take what's left in the buffer
                    new_stop = min(batch_size, batch_slice.stop + len_unused_buffer)
                    batch_slice = slice(batch_slice.stop, new_stop)

                len_batch_slice = batch_slice.stop - batch_slice.start
                new_stop = min(n_regions, starts_slice.stop + len_batch_slice)
                starts_slice = slice(starts_slice.stop, new_stop)

    def partition_bed(self, bed: pl.DataFrame) -> List[pl.DataFrame]:
        # use polars.DataFrame.partition_by
        # intelligently partition regions to maximize size of in-memory buffer
        # but still must respect contig boundaries
        raise NotImplementedError
