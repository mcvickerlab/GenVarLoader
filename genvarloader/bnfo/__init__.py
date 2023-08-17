import random
from collections import defaultdict
from copy import deepcopy
from itertools import accumulate, chain, repeat
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import polars as pl
import xarray as xr
from natsort import natsorted
from numpy.typing import NDArray

from .bigwig import BigWig
from .fasta import Fasta
from .fasta_variants import FastaVariants
from .numba import build_length_indices, partition_regions
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
        self.sizes: Dict[str, int] = {}
        self.itemsizes: Dict[str, int] = defaultdict(int)
        self.indexes: Dict[str, NDArray] = {}
        for r in self.readers:
            for dim, size in r.sizes.items():
                if dim not in self.sizes:
                    self.sizes[dim] = size
                elif self.sizes[dim] != size:
                    raise ValueError(
                        f"""Readers have inconsistent dimension sizes, at least for dimension {dim}
                        Sizes: {[r.sizes for r in self.readers]}
                        """
                    )

                if dim not in self.indexes:
                    self.indexes[dim] = r.indexes[dim]
                elif self.indexes[dim] != r.indexes[dim]:
                    raise ValueError(
                        f"""Readers have inconsistent indexes, at least for dimension {dim}
                        Indexes: {[r.sizes for r in self.readers]}
                        """
                    )

                self.itemsizes[dim] += r.dtype.itemsize

    @overload
    def iter_batches(
        self,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_size: int,
        max_memory_gb: float,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        return_tuples: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        ...

    @overload
    def iter_batches(
        self,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_size: int,
        max_memory_gb: float,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        return_tuples: bool = True,
    ) -> Generator[Tuple[Any], None, None]:
        ...

    def iter_batches(
        self,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_size: int,
        max_memory_gb: float,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        return_tuples: bool = False,
        return_index: bool = False,
    ) -> Generator[Union[Dict[str, Any], Tuple[Any]], None, None]:
        if isinstance(bed, (str, Path)):
            bed = read_bedlike(bed)

        if batch_dims is None:
            batch_dims = []
        elif missing_dims := (set(batch_dims) - set(self.sizes.keys())):
            raise ValueError(
                f"Got batch dimensions that are not available from the readers: {missing_dims}"
            )

        rng = np.random.default_rng(seed)

        max_length = cast(
            int,
            bed.groupby("chrom")
            .agg(
                (pl.col("chromEnd").max() - pl.col("chromStart")).min().alias("length")
            )["length"]
            .max(),
        )

        batch_mem = fixed_length * self.mem_per_length(
            {k: v for k, v in self.sizes.items() if k not in batch_dims}
        )
        FUDGE_FACTOR = 2
        max_mem = int((max_memory_gb * 1e9 - batch_mem) / FUDGE_FACTOR)

        buffer_sizes = self.set_buffer_sizes(max_mem, max_length, batch_dims)

        max_length = max_mem // self.mem_per_length(buffer_sizes)
        if max_length == 0:
            min_mem = (
                (self.mem_per_length(buffer_sizes) + batch_mem) * FUDGE_FACTOR / 1e9
            )
            raise ValueError(
                f"Not enough memory to process dataset. Minimum memory needed: {min_mem:.4f} GB."
            )

        with pl.StringCache():
            pl.Series(natsorted(bed["chrom"].unique()), dtype=pl.Categorical)
            bed = bed.sort(pl.col("chrom").cast(pl.Categorical), "chromStart")
        bed = _set_uniform_length_around_center(bed, fixed_length)
        partitioned_bed = self.partition_bed(bed, max_length)
        if shuffle:
            random.shuffle(partitioned_bed, rng.random)

        batch_slice = slice(0, 0)
        dim_slices = {d: slice(0, v) for d, v in buffer_sizes.items()}

        if shuffle:
            indexes = {d: rng.permutation(idx) for d, idx in self.indexes.items()}
        else:
            indexes = self.indexes

        partial_batches = []

        for partition in partitioned_bed:
            n_regions = len(partition)
            instances_in_partition = n_regions * np.prod(
                [self.sizes[d] for d in batch_dims], dtype=int
            )
            instances_yielded = 0

            contig: str
            start: int
            end: int
            contig, start, end = partition.select(
                pl.col("chrom").first(),
                pl.col("chromStart").min(),
                pl.col("chromEnd").max(),
            ).row(0)

            instances_in_buffer = 0
            buffer_idx_slice = slice(0, 0)
            len_unused_buffer = 0

            while instances_yielded < instances_in_partition:
                if len_unused_buffer == 0:
                    buffer = self.get_buffer(indexes, dim_slices, contig, start, end)
                    buffer_idx = self.get_buffer_idx(
                        partition, start, buffer, batch_dims
                    )
                    if shuffle:
                        buffer_idx = rng.permutation(buffer_idx)
                    dim_slices = {
                        d: slice(s.stop, s.stop + v)
                        for (d, s), v in zip(dim_slices.items(), buffer_sizes.values())
                    }
                    instances_in_buffer = len(buffer_idx)
                    new_stop = min(batch_size, batch_slice.stop + instances_in_buffer)
                    batch_slice = slice(batch_slice.stop, new_stop)
                    len_batch_slice = batch_slice.stop - batch_slice.start
                    buffer_idx_slice = slice(0, len_batch_slice)

                # guaranteed to init buffer_idx based on init of len_unused_buffer
                idx = buffer_idx[buffer_idx_slice]
                if len(batch_dims) > 0:
                    selector = {
                        d: xr.DataArray(col, dims="batch")
                        for d, col in zip(
                            batch_dims, np.hsplit(idx[:, 1:], len(batch_dims))
                        )
                    }
                else:
                    selector = {}
                selector["length"] = xr.DataArray(
                    build_length_indices(idx[:, 0], fixed_length),
                    dims=["batch", "length"],
                )

                batch: xr.Dataset
                # guaranteed to init buffer based on init of len_unused_buffer
                if batch_slice.start == 0 and batch_slice.stop == batch_size:
                    batch = buffer.isel(selector, missing_dims="ignore")
                else:
                    partial_batches.append(buffer.isel(selector, missing_dims="ignore"))
                    if batch_slice.stop == batch_size:
                        batch = xr.concat(partial_batches, dim="batch")

                len_unused_buffer = instances_in_buffer - buffer_idx_slice.stop

                # ready to yield batch
                if batch_slice.stop == batch_size:
                    batch = batch.transpose("batch", ...)
                    out = {
                        name: arr.to_numpy() for name, arr in batch.data_vars.items()
                    }
                    # TODO check this works
                    if return_index:
                        out["index"] = idx + np.array(
                            [s.start for s in dim_slices.values()]
                        )
                    if transform is not None:
                        out = transform(out)
                    if return_tuples:
                        yield tuple(out.values())
                    else:
                        yield out

                    instances_yielded += batch_slice.stop - batch_slice.start

                    # full batch or take what's left in the buffer
                    new_stop = min(batch_size, len_unused_buffer)
                    batch_slice = slice(0, new_stop)
                # not ready and more data in buffer
                else:
                    # fill batch or take what's left in the buffer
                    new_stop = min(batch_size, batch_slice.stop + len_unused_buffer)
                    batch_slice = slice(batch_slice.stop, new_stop)

                new_stop = min(buffer_idx_slice.stop + batch_size, instances_in_buffer)
                buffer_idx_slice = slice(buffer_idx_slice.stop, new_stop)

    def mem_per_length(self, sizes: Dict[str, int]):
        mpl = sum(sizes[dim] * self.itemsizes[dim] for dim in sizes)
        mpl = max(1, mpl)
        return mpl

    def set_buffer_sizes(self, max_mem: int, max_length: int, batch_dims: List[str]):
        buffer_sizes = deepcopy(self.sizes)
        if max_mem < max_length * self.mem_per_length(self.sizes):
            for dim in batch_dims:
                buffer_sizes.pop(dim)
                size = int(
                    (max_mem / max_length - self.mem_per_length(buffer_sizes))
                    / self.itemsizes[dim]
                )
                if size > 0:
                    buffer_sizes[dim] = size
                    break
                elif size == 0:
                    buffer_sizes[dim] = 1
        return buffer_sizes

    def partition_bed(
        self,
        bed: pl.DataFrame,
        max_length: int,
    ) -> List[pl.DataFrame]:
        """Partition regions of a BED file such that the overlap of each partition never
        exceeds the max length.

        Parameters
        ----------
        bed : pl.DataFrame
        max_length : int

        Returns
        -------
        List[pl.DataFrame]
            Partitions of the BED.
        """
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

    def get_buffer(
        self,
        indexes: Dict[str, NDArray],
        dim_slices: Dict[str, slice],
        contig: str,
        start: int,
        end: int,
    ):
        read_kwargs = {d: indexes[d][s] for d, s in dim_slices.items()}
        buffer = xr.Dataset(
            {r.name: r.read(contig, start, end, **read_kwargs) for r in self.readers}
        )
        return buffer

    def get_buffer_idx(self, partition, start, buffer, batch_dims):
        starts = cast(NDArray[np.int32], (partition["chromStart"] - start).to_numpy())
        buffer_indexes = [
            starts,
            *(
                np.arange(size, dtype=np.int32)
                for name, size in buffer.sizes.items()
                if name in batch_dims
            ),
        ]
        buffer_idx = _cartesian_product(buffer_indexes)
        return buffer_idx


def _cartesian_product(arrays: Sequence[NDArray]) -> NDArray:
    """Get the cartesian product of multiple arrays such that each entry corresponds to
    a unique combination of the input arrays' values.
    """
    # https://stackoverflow.com/a/49445693
    la = len(arrays)
    shape = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(shape, dtype=dtype)
    arrs = (*accumulate(chain((arr,), repeat(0, la - 1)), np.ndarray.__getitem__),)
    idx = slice(None), *repeat(None, la - 1)
    for i in range(la - 1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[: la - i]]
        arrs[i - 1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)
