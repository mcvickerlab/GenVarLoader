import random
from collections import defaultdict
from itertools import accumulate, chain, repeat
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
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
from .pgen import Pgen
from .rle_table import RLE_Table
from .tiledb_vcf import TileDB_VCF
from .types import Reader
from .util import _set_uniform_length_around_center, read_bedlike

__all__ = [
    "BigWig",
    "Fasta",
    "TileDB_VCF",
    "FastaVariants",
    "RLE_Table",
    "Pgen",
    "GVL",
    "view_virtual_data",
]


def view_virtual_data(readers: Union[Reader, Iterable[Reader]]):
    """View the virtual data corresponding from multiple readers. This is useful to
    inspect what non-length dimensions will be exist when constructing a GVL loader
    from them.

    Parameters
    ----------
    readers : Reader, Iterable[Reader]
        Readers to inspect.
    """
    if not isinstance(readers, Iterable):
        readers = [readers]
    return xr.merge([r.virtual_data for r in readers], join="exact")


# TODO test weighted upsampling
# TODO async reads
# have two buffers, one for reading data and for yielding batches
# note: this will half the memory budget for buffers
# use ray for concurrent work so it's aware of other concurrent readers
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

    def __init__(
        self,
        readers: Union[Reader, Iterable[Reader]],
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_size: int,
        max_memory_gb: float,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
        shuffle: bool = False,
        weights: Optional[Dict[str, NDArray]] = None,
        seed: Optional[int] = None,
        return_tuples: bool = False,
        return_index: bool = False,
        drop_last: bool = False,
    ) -> None:
        """GenVarLoader

        Parameters
        ----------
        readers : Reader, Iterable[Reader]
            Reader or iterable or readers. These all implement the read() method which
            returns data corresponding to a genomic range.
        bed : pl.DataFrame, str, Path
            BED3+ file.
        fixed_length : int
            Length of regions of interest to output. Will be centered at coordinates in
            bed file.
        batch_size : int
            Number of instances in the batch.
        max_memory_gb : float
            Maximum memory to use in GB.
        batch_dims : List[str], optional
            Dimensions that can be included in the batch dimension, by default None
        transform : (Dict[str, NDArray]) -> Dict[str, NDArray], optional
            Function to call on each batch before yielding, by default None. This
            function should accept a dictionary of NumPy arrays and return a dictionary
            of NumPy arrays.
        shuffle : bool, optional
            Whether to shuffle with respect to regions of interest and batch dimensions,
            by default False
        weights : Dict[str, NDArray], optional
            Dictionary mapping dimensions to weights. The "region" dimension corresponds
            to each region in the BED file and is always available, whereas others
            correspond to non-length dimensions seen in the virtual data.
        seed : int, optional
            Seed for shuffling, by default None
        return_tuples : bool, optional
            Whether to return a tuple instead of a dictionary, by default False
        return_index : bool, optional
            Whether to include an array of the indexes in the batch, by default False
        drop_last : bool, optional
            Whether to drop the last batch if the number of instances are not evenly
            divisible by the batch size.
        """

        if not isinstance(readers, Iterable):
            readers = [readers]
        self.readers = readers
        # TODO handle order of indices
        self.virtual_data = xr.merge(
            [r.virtual_data for r in self.readers], join="exact"
        )
        self.sizes = dict(self.virtual_data.sizes)
        self.sizes.pop("", None)
        self.itemsizes: Mapping[Hashable, int] = defaultdict(int)
        self.indexes = self.virtual_data.coords
        for arr in self.virtual_data.data_vars.values():
            for dim in arr.dims:
                if dim == "":
                    continue
                self.itemsizes[dim] += arr.dtype.itemsize

        if isinstance(bed, (str, Path)):
            bed = read_bedlike(bed)
        bed = bed.with_row_count("region_idx")

        self.fixed_length = fixed_length
        self.batch_size = batch_size

        if batch_dims is None:
            batch_dims = []
        elif missing_dims := (set(batch_dims) - set(self.virtual_data.dims)):
            raise ValueError(
                f"Got batch dimensions that are not available from the readers: {missing_dims}"
            )
        self.batch_dims = batch_dims

        if weights is not None:
            if extra_weights := set(weights.keys()) - set(self.virtual_data.dims):
                raise ValueError(
                    f"Got weights for dimensions that are not available from the readers: {extra_weights}"
                )
            if extra_weights := set(weights.keys()) - set(batch_dims):
                raise ValueError(
                    f"Got weights for dimensions that are not batch dimensions: {extra_weights}"
                )
        self.weights = weights
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
        max_mem = int(
            (max_memory_gb * 1e9 - batch_mem - bed.estimated_size()) / FUDGE_FACTOR
        )

        self.buffer_sizes = self.set_buffer_sizes(max_mem, max_length, batch_dims)

        max_length = max_mem // self.mem_per_length(self.buffer_sizes)
        if max_length == 0:
            min_mem = (
                (self.mem_per_length(self.buffer_sizes) + batch_mem)
                * FUDGE_FACTOR
                / 1e9
            )
            raise ValueError(
                f"Not enough memory to process dataset. Minimum memory needed: {min_mem:.4f} GB."
            )

        with pl.StringCache():
            pl.Series(natsorted(bed["chrom"].unique()), dtype=pl.Categorical)
            bed = bed.sort(pl.col("chrom").cast(pl.Categorical), "chromStart")
        self.bed = _set_uniform_length_around_center(bed, fixed_length)

        # TODO check if any regions are out-of-bounds and any readers have padding disabled
        # if so, raise an error. Otherwise, readers will catch the error downstream.

        self.partitioned_bed = self.partition_bed(self.bed, max_length)
        self.n_instances: int = self.bed.height * np.prod(
            [self.sizes[d] for d in self.batch_dims], dtype=int
        )

        self.transform = transform
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.return_tuples = return_tuples
        self.return_index = return_index
        self.drop_last = drop_last

    def __len__(self):
        if not self.drop_last:
            return -(-self.n_instances // self.batch_size)  # ceil
        else:
            return self.n_instances // self.batch_size

    def __iter__(self):
        return self.iter_batches()

    def iter_batches(self):
        if self.shuffle:
            random.shuffle(self.partitioned_bed, self.rng.random)

        self.batch_slice = slice(0, 0)

        if self.shuffle:
            self.indexes = {
                d: self.rng.permutation(idx) for d, idx in self.indexes.items()
            }

        self.partial_batches = []
        self.partial_indices = []
        self.total_yielded = 0

        for partition in self.partitioned_bed:
            # Better to use slices on batch dimensions in case one of the readers is a
            # chunked array format. This will reduce the amount of chunks hit on read() if
            # the chunk size is > 1. This is in contrast to having a range index that can be
            # randomly permuted. Because readers are not constrained to be chunked arrays we
            # also cannot randomly select chunks.
            # TODO allow randomization of dim_slices (i.e. random order of slices)
            # Order of batch dims should be handled on reader side and done in-memory
            # ! this could cause issues for chunked arrays formats if the requested
            # ! samples all live in different chunks
            dim_slices = {str(d): slice(0, 0) for d in self.buffer_sizes}

            n_regions = len(partition)
            instances_in_partition = n_regions * np.prod(
                [self.sizes[d] for d in self.batch_dims], dtype=int
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
                    dim_slices = {
                        d: slice(s.stop, s.stop + v)
                        for (d, s), v in zip(
                            dim_slices.items(), self.buffer_sizes.values()
                        )
                    }
                    buffer = self.get_buffer(
                        self.indexes, dim_slices, contig, start, end
                    )
                    # columns: starts, region_idx, dim1_idx, dim2_idx, ...
                    buffer_idx = self.get_buffer_idx(partition, start, buffer)
                    if self.weights is not None:
                        buffer_idx = self.resample_buffer_idx(buffer_idx)

                    if self.shuffle:
                        buffer_idx = self.rng.permutation(buffer_idx)

                    instances_in_buffer = len(buffer_idx)
                    new_stop = min(
                        self.batch_size, self.batch_slice.stop + instances_in_buffer
                    )
                    self.batch_slice = slice(self.batch_slice.stop, new_stop)
                    len_batch_slice = self.batch_slice.stop - self.batch_slice.start
                    buffer_idx_slice = slice(0, len_batch_slice)

                # guaranteed to init buffer_idx based on init of len_unused_buffer
                idx = buffer_idx[buffer_idx_slice]
                if len(self.batch_dims) > 0:
                    selector = {
                        d: xr.DataArray(col.squeeze(-1), dims="batch")
                        for d, col in zip(
                            self.batch_dims, np.hsplit(idx[:, 2:], len(self.batch_dims))
                        )
                    }
                else:
                    selector = {}
                selector["length"] = xr.DataArray(
                    build_length_indices(idx[:, 0], self.fixed_length),
                    dims=["batch", "length"],
                )

                batch: xr.Dataset
                if (
                    self.batch_slice.start == 0
                    and self.batch_slice.stop == self.batch_size
                ):
                    # guaranteed to init buffer based on init of len_unused_buffer
                    batch = buffer.isel(selector, missing_dims="ignore")
                    batch_idx = idx.copy()
                else:
                    # guaranteed to init buffer based on init of len_unused_buffer
                    self.partial_batches.append(
                        buffer.isel(selector, missing_dims="ignore")
                    )
                    self.partial_indices.append(idx)
                    if self.batch_slice.stop == self.batch_size:
                        batch = xr.concat(self.partial_batches, dim="batch")
                        batch_idx = np.concatenate(self.partial_indices)
                        self.partial_batches = []
                        self.partial_indices = []

                len_unused_buffer = instances_in_buffer - buffer_idx_slice.stop

                # full batch
                if self.batch_slice.stop == self.batch_size:
                    yield self.process_batch(batch, batch_idx, dim_slices)

                    instances_yielded += self.batch_size
                    self.total_yielded += self.batch_size

                    # full batch or take what's left in the buffer
                    new_stop = min(self.batch_size, len_unused_buffer)
                    self.batch_slice = slice(0, new_stop)
                # final batch incomplete
                elif self.total_yielded + self.batch_slice.stop == self.n_instances:
                    if self.drop_last:
                        return

                    # final incomplete batch is always a partial batch
                    batch = xr.concat(self.partial_batches, dim="batch")
                    batch_idx = np.concatenate(self.partial_indices, 0)

                    yield self.process_batch(batch, batch_idx, dim_slices)

                    instances_yielded += self.batch_slice.stop
                    self.total_yielded += self.batch_slice.stop
                # batch incomplete and more data in buffer
                else:
                    # fill batch or take what's left in the buffer
                    new_stop = min(
                        self.batch_size, self.batch_slice.stop + len_unused_buffer
                    )
                    self.batch_slice = slice(self.batch_slice.stop, new_stop)

                new_stop = min(
                    buffer_idx_slice.stop + self.batch_size, instances_in_buffer
                )
                buffer_idx_slice = slice(buffer_idx_slice.stop, new_stop)

    def mem_per_length(self, sizes: Mapping[Hashable, int]):
        mpl = sum(sizes[dim] * self.itemsizes[dim] for dim in sizes)
        mpl = max(1, mpl)
        return mpl

    def set_buffer_sizes(self, max_mem: int, max_length: int, batch_dims: List[str]):
        buffer_sizes = dict(self.sizes)
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
        indexes: Mapping[Hashable, NDArray],
        dim_slices: Mapping[str, slice],
        contig: str,
        start: int,
        end: int,
    ):
        read_kwargs = {d: indexes[d][s] for d, s in dim_slices.items()}
        buffer = xr.Dataset(
            {
                r.virtual_data.name: r.read(contig, start, end, **read_kwargs)
                for r in self.readers
            }
        )
        return buffer

    def get_buffer_idx(self, partition, start, buffer) -> NDArray[np.integer]:
        row_idx = partition.with_row_count()["row_nr"].to_numpy()
        buffer_indexes = [
            row_idx,
            *(
                np.arange(size, dtype=np.int32)
                for name, size in buffer.sizes.items()
                if name in self.batch_dims
            ),
        ]
        buffer_idx = _cartesian_product(buffer_indexes)
        # columns: starts, region_idx, dim1_idx, dim2_idx, ...
        starts = (partition["chromStart"].to_numpy() - start)[buffer_idx[:, 0]][:, None]
        region_idx = partition["region_idx"].to_numpy()[buffer_idx[:, 0]][:, None]
        return np.hstack([starts, region_idx, buffer_idx[:, 1:]])

    def resample_buffer_idx(self, buffer_idx: NDArray):
        idx_weights = np.ones(len(buffer_idx))
        # buffer_idx columns: starts, region_idx, dim1_idx, dim2_idx, ...
        for i, d in enumerate(self.batch_dims):
            w = self.weights.get(d, None)
            if w is not None:
                idx_weights *= w[buffer_idx[:, i + 2]]
        idx_weights = np.round(idx_weights).astype(int)
        return buffer_idx.repeat(idx_weights)

    def process_batch(
        self, batch: xr.Dataset, batch_idx: NDArray, dim_slices: Dict[str, slice]
    ):
        batch = batch.transpose("batch", ...)
        out = {name: arr.to_numpy() for name, arr in batch.data_vars.items()}

        if self.return_index:
            # cols: region_idx, dim1_idx, dim2_idx, ...
            out_idx = batch_idx[:, 1:]
            if len(self.batch_dims) > 0:
                out_idx[:, 1:] += np.array(
                    [dim_slices[d].start for d in self.batch_dims]
                )
            out["index"] = out_idx

        if self.transform is not None:
            out = self.transform(out)

        if self.return_tuples:
            out = tuple(out.values())

        return out


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
