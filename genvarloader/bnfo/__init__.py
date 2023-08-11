from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from natsort import natsorted
from numpy.typing import NDArray

from .bigwig import BigWig
from .fasta import Fasta
from .fasta_variants import FastaVariants
from .numba import gufunc_multi_slice, partition_regions
from .tiledb_vcf import TileDB_VCF
from .types import Reader
from .util import _set_uniform_length_around_center, read_bedlike

__all__ = ["BigWig", "Fasta", "TileDB_VCF", "FastaVariants"]


class GVL:
    """GenVarLoader

    Idea behind this implementation is to efficiently materialize sequences from long,
    overlapping ROIs. The algorithm is roughly:
    1. Partition the ROIs to maximize the size of the union of ROIs while respecting
    memory limits. Note any union of ROIs must be on the same contig. Buffer this
    union of ROIs in memory.
    2. Materialize batches of subsequences, i.e. the ROIs, by slicing the buffer. This
    keeps memory usage to a minimum since we only need enough for the buffer + a single
    batch. This should be fast because the buffer is the only part that uses file I/O
    whereas the batches are materialized from the buffer.
    """

    def __init__(self, *readers: Reader) -> None:
        self.readers = readers
        self.bytes_per_length = sum(r.bytes_per_length for r in self.readers)

    def iter_batches(
        self,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_size: int,
        max_memory_gb: float,
        buffer_transform: Optional[
            Callable[[Dict[str, NDArray]], Dict[str, NDArray]]
        ] = None,
        batch_transform: Optional[
            Callable[[Dict[str, NDArray]], Dict[str, NDArray]]
        ] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        return_tuples: bool = False,
    ) -> Generator[Union[Dict[str, Any], Tuple[Any]], None, None]:
        rng = np.random.default_rng(seed)

        if isinstance(bed, (str, Path)):
            bed = read_bedlike(bed)

        with pl.StringCache():
            pl.Series(natsorted(bed["chrom"].unique()), dtype=pl.Categorical)
            bed = bed.sort(pl.col("chrom").cast(pl.Categorical), "chromStart")
        bed = _set_uniform_length_around_center(bed, fixed_length)
        partitioned_bed = self.partition_bed(
            bed, fixed_length, batch_size, max_memory_gb
        )

        starts_slice = slice(0, 0)
        batch_slice = slice(0, 0)

        batch = None

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
            if buffer_transform is not None:
                buffers = buffer_transform(buffers)

            if batch is None:
                batch = {
                    name: np.empty_like(
                        buff, shape=(batch_size, *buff.shape[:-1], fixed_length)
                    )
                    for name, buff in buffers.items()
                }

            starts = (partition["chromStart"] - start).to_numpy()
            length = end - start
            if shuffle:
                starts = rng.permutation(starts)

            len_unused_buffer = n_regions
            length_placeholder = np.arange(length)

            while len_unused_buffer > 0:
                # Consider deferring multi slice implementation to the readers since the
                # length axis might depend on the modality? This may necessitate an
                # inefficient implementation to coax all arrays to have the length axis
                # in the right place so that broadcasting always works here. Or, having
                # to add a length_axis property for each reader, which seems
                # unnecessarily awkward too. Deferring would also allow a variant
                # applying Reader to do something that isn't multi-slicing and
                # potentially improve the speed/memory trade-off.
                for name in batch:
                    batch[name][batch_slice] = gufunc_multi_slice(
                        buffers[name], starts[starts_slice], length_placeholder
                    )

                len_unused_buffer = len_unused_buffer - starts_slice.stop

                # ready to yield batch
                if batch_slice.stop == batch_size:
                    # full batch or take what's left in the buffer
                    new_stop = min(batch_size, len_unused_buffer)
                    batch_slice = slice(0, new_stop)
                    if batch_transform is not None:
                        batch = batch_transform(batch)
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

    def partition_bed(
        self,
        bed: pl.DataFrame,
        fixed_length: int,
        batch_size: int,
        max_memory_gb: float,
    ) -> List[pl.DataFrame]:
        # use polars.DataFrame.partition_by
        # partition regions to maximize size of in-memory buffer
        # but still respect contig boundaries
        max_mem = max_memory_gb / 2 * 1e9
        batch_mem = self.bytes_per_length * batch_size * fixed_length / 1e9
        max_mem -= batch_mem * 2
        if max_mem < 0:
            min_mem = batch_mem * 4 / 1e9
            raise ValueError(
                f"Insufficient memory to process data. At least {min_mem:.3f} GB is needed."
            )
        max_length = int(max_mem / self.bytes_per_length)
        contig_partitions = bed.partition_by("chrom")
        partitions: List[pl.DataFrame] = []
        for c_part in contig_partitions:
            c_part = c_part.with_columns(
                partition=pl.lit(
                    partition_regions(
                        c_part["chromStart"].to_numpy(),
                        c_part["chromEnd"].to_numpy(),
                        max_length,
                    )
                )
            )
            partitions.extend(c_part.partition_by("partition", include_key=False))
        return partitions
