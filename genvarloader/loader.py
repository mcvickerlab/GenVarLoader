from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generator,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import numba as nb
import numpy as np
import polars as pl
import ray
import xarray as xr
from loguru import logger
from more_itertools import chunked
from natsort import natsorted
from numpy.typing import NDArray

from .concurrent import Buffer, BufferMeta, DataVarsLike, ReaderActor
from .types import Reader
from .util import _cartesian_product, _set_fixed_length_around_center, read_bedlike

try:
    import torch  # noqa

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


BatchDict = Dict[Hashable, Tuple[List[Hashable], NDArray]]


def view_virtual_data(*readers: Reader):
    """View the virtual data corresponding from multiple readers. This is useful to
    inspect what non-length dimensions will be exist when constructing a GVL loader
    from them.

    Parameters
    ----------
    *readers : Reader
        Readers to inspect.
    """
    virtual_data = xr.merge([r.virtual_data for r in readers], join="inner")
    for dim, size in virtual_data.sizes.items():
        for reader in readers:
            if (
                dim in reader.virtual_data.dims
                and size != reader.virtual_data.sizes[dim]
            ):
                logger.warning(
                    f"Dimension {dim} has different sizes between readers. Values in this dimension that are not present in all readers will not be included."
                )
    return virtual_data


class GVL:
    # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
    BUFFER_IDX_START_COL = 0
    BUFFER_IDX_STRAND_COL = 1
    BUFFER_IDX_REGION_COL = 2
    BUFFER_IDX_MIN_DIM_COL = 3

    def __init__(
        self,
        readers: Union[Reader, Iterable[Reader]],
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_size: int,
        max_memory_gb: float,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[
            Callable[[Dict[Hashable, NDArray]], Dict[Hashable, NDArray]]
        ] = None,
        shuffle: bool = False,
        weights: Optional[Dict[str, NDArray]] = None,
        seed: Optional[int] = None,
        return_tuples: bool = False,
        return_index: bool = False,
        drop_last: bool = False,
        num_workers: int = 1,
        jitter_bed: Optional[int] = None,
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
            Whether to include an array of the indexes in the batch, by default False.
            Indices will be available from the "index" key or will be the last item in
            the tuple.
        drop_last : bool, optional
            Whether to drop the last batch if the number of instances are not evenly
            divisible by the batch size.
        num_workers : int, optional
            How many workers to use, default 1.
        jitter_bed : int, optional
            Jitter the regions in the BED file by up to this many nucleotides.
        """
        self.num_workers = num_workers
        self.fixed_length = fixed_length
        self.transform = transform
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.return_tuples = return_tuples
        self.return_index = return_index
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.jitter_bed = jitter_bed

        if not ray.is_initialized() and self.num_workers > 1:
            ray.init(num_cpus=self.num_workers - 1)

        if not isinstance(readers, Iterable):
            readers = [readers]
        self.readers = {r.virtual_data.name: r for r in readers}

        if self.num_workers >= 2:
            self.actors: List[ReaderActor] = [
                ReaderActor.remote(*self.readers.values(), actor_idx=i)
                for i in range(self.num_workers - 1)  # keep 1 cpu for main process
            ]

        self.virtual_data = view_virtual_data(*self.readers.values())
        self.sizes = dict(self.virtual_data.sizes)
        self.sizes.pop("", None)
        self.itemsizes: Mapping[Hashable, int] = defaultdict(int)
        self.indexes = {k: a.values for k, a in self.virtual_data.coords.items()}
        for arr in self.virtual_data.data_vars.values():
            for dim in arr.dims:
                if dim == "":
                    continue
                self.itemsizes[dim] += arr.dtype.itemsize

        if batch_dims is None:
            batch_dims = []
        self.batch_dims = batch_dims

        if missing_dims := (set(self.batch_dims) - set(self.virtual_data.dims)):  # type: ignore
            raise ValueError(
                f"Got batch dimensions that are not available in any reader: {missing_dims}"
            )

        self.non_batch_dims = [d for d in self.sizes.keys() if d not in self.batch_dims]
        self.non_batch_dim_shape: Dict[Hashable, List[int]] = {}
        self.buffer_idx_cols: Dict[Hashable, NDArray[np.integer]] = {}
        for name, a in self.virtual_data.data_vars.items():
            self.non_batch_dim_shape[name] = [a.sizes[d] for d in self.non_batch_dims]
            # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
            idx_cols = [
                self.batch_dims.index(d) + self.BUFFER_IDX_MIN_DIM_COL
                for d in a.dims
                if d in self.batch_dims
            ]
            self.buffer_idx_cols[name] = np.array(idx_cols, dtype=int)

        if isinstance(bed, (str, Path)):
            bed = read_bedlike(bed)

        if "strand" in bed:
            bed = bed.with_columns(
                pl.col("strand").map_dict({"-": -1, "+": 1}, return_dtype=pl.Int8)
            )
        else:
            bed = bed.with_columns(strand=pl.lit(1, dtype=pl.Int8))

        bed = bed.with_row_count("region_idx")
        with pl.StringCache():
            pl.Series(natsorted(bed["chrom"].unique()), dtype=pl.Categorical)
            bed = bed.sort(pl.col("chrom").cast(pl.Categorical), "chromStart")
        self.bed = _set_fixed_length_around_center(bed, self.fixed_length)
        # TODO check if any regions are out-of-bounds and any readers have padding
        # disabled. If so, raise an error. Otherwise, readers will catch the error
        # downstream.

        self.n_instances: int = self.bed.height * np.prod(
            [self.sizes[d] for d in self.batch_dims], dtype=int
        )

        self.max_length = self.get_max_length()
        self.partitioned_bed = self.partition_bed(self.bed, self.max_length)

        if weights is not None:
            if extra_weights := set(weights.keys()) - set(self.virtual_data.dims):  # type: ignore
                raise ValueError(
                    f"Got weights for dimensions that are not available from the readers: {extra_weights}"
                )
            if extra_weights := set(weights.keys()) - set(self.batch_dims):
                raise ValueError(
                    f"Got weights for dimensions that are not batch dimensions: {extra_weights}"
                )
        self.weights = weights

    def set(
        self,
        bed: Optional[Union[pl.DataFrame, str, Path]] = None,
        fixed_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_memory_gb: Optional[float] = None,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[
            Callable[[Dict[Hashable, NDArray]], Dict[Hashable, NDArray]]
        ] = None,
        shuffle: bool = False,
        weights: Optional[Dict[str, NDArray]] = None,
        seed: Optional[int] = None,
        return_tuples: bool = False,
        return_index: bool = False,
        drop_last: bool = False,
        jitter_bed: Optional[int] = None,
    ):
        """Update any parameters that don't require re-initializing Ray Actors. If you
        need to change readers or the number of workers, init a new GVL. Note: do NOT
        use this during iteration (i.e. during an epoch), or things will break. This is
        meant to be used in between iteration (i.e. between epochs).

        Parameters
        ----------
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
        jitter_bed : int, optional
            Jitter the regions in the BED file by up to this many nucleotides.
        """
        if fixed_length is not None:
            self.fixed_length = fixed_length
        if batch_size is not None:
            self.batch_size = batch_size
        if max_memory_gb is not None:
            self.max_memory_gb = max_memory_gb
        if transform is not None:
            self.transform = transform
        if shuffle is not None:
            self.shuffle = shuffle
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if return_tuples is not None:
            self.return_tuples = return_tuples
        if return_index is not None:
            self.return_index = return_index
        if drop_last is not None:
            self.drop_last = drop_last
        if jitter_bed is not None:
            self.jitter_bed = jitter_bed

        if bed is not None:
            if isinstance(bed, (str, Path)):
                bed = read_bedlike(bed)

            if "strand" in bed:
                bed = bed.with_columns(
                    pl.col("strand").map_dict({"-": -1, "+": 1}, return_dtype=pl.Int8)
                )
            else:
                bed = bed.with_columns(strand=pl.lit(1, dtype=pl.Int8))

            bed = bed.with_row_count("region_idx")

            with pl.StringCache():
                pl.Series(natsorted(bed["chrom"].unique()), dtype=pl.Categorical)
                bed = bed.sort(pl.col("chrom").cast(pl.Categorical), "chromStart")
            bed = _set_fixed_length_around_center(bed, self.fixed_length)
            self.bed = bed

        if batch_dims is not None:
            self.batch_dims = batch_dims
            if missing_dims := (set(self.batch_dims) - set(self.virtual_data.dims)):  # type: ignore
                raise ValueError(
                    f"Got batch dimensions that are not available from the readers: {missing_dims}"
                )
            self.non_batch_dims = [
                d for d in self.sizes.keys() if d not in self.batch_dims
            ]
            self.non_batch_dim_shape: Dict[Hashable, List[int]] = {}
            self.buffer_idx_cols: Dict[Hashable, NDArray[np.integer]] = {}
            for name, a in self.virtual_data.data_vars.items():
                self.non_batch_dim_shape[name] = [
                    a.sizes[d] for d in self.non_batch_dims
                ]
                # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
                idx_cols = [
                    i
                    for i, d in enumerate(self.batch_dims, self.BUFFER_IDX_MIN_DIM_COL)
                    if d in a.sizes
                ]
                self.buffer_idx_cols[name] = np.array(idx_cols, dtype=int)

        if None not in (fixed_length, bed, max_memory_gb, batch_dims):
            # don't need to check jitter_bed here because it's jittered at each call
            # to __iter__
            self.max_length = self.get_max_length()
            self.partitioned_bed = self.partition_bed(self.bed, self.max_length)
            self.n_instances: int = self.bed.height * np.prod(
                [self.sizes[d] for d in self.batch_dims], dtype=int
            )

        if weights is not None:
            if extra_weights := set(weights.keys()) - set(self.virtual_data.dims):  # type: ignore
                raise ValueError(
                    f"Got weights for dimensions that are not available from the readers: {extra_weights}"
                )
            if extra_weights := set(weights.keys()) - set(self.batch_dims):
                raise ValueError(
                    f"Got weights for dimensions that are not batch dimensions: {extra_weights}"
                )
            self.weights = weights

    def __len__(self):
        """Number of batches."""
        if not self.drop_last:
            return -(-self.n_instances // self.batch_size)  # ceil
        else:
            return self.n_instances // self.batch_size

    def mem_per_length(self, sizes: Mapping[Hashable, int]):
        mpl = sum(sizes[dim] * self.itemsizes[dim] for dim in sizes)
        mpl = max(1, mpl)
        return mpl

    def get_max_length(self):
        """Get the maximum length"""
        max_length = self.fixed_length
        batch_mem = self.fixed_length * self.mem_per_length(
            {k: v for k, v in self.sizes.items() if k not in self.batch_dims}
        )
        FUDGE_FACTOR = 6
        max_mem = int(
            (
                self.max_memory_gb * 1e9 / self.num_workers
                - batch_mem
                - self.bed.estimated_size()
            )
            / FUDGE_FACTOR
        )

        self.buffer_sizes = self.get_buffer_sizes(max_mem, max_length, self.batch_dims)

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
        return max_length

    def get_buffer_sizes(
        self, max_mem: int, max_length: int, batch_dims: List[str]
    ) -> Dict[Hashable, int]:
        """Get the size of batch dimensions such that the largest buffer (i.e. with max
        length) will fit into memory."""
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
                    # reducing this dim is enough to get a buffer that fits into memory
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

    def __iter__(self):
        return self.iter_batches()

    def iter_batches(self):
        if self.jitter_bed is not None:
            shifts = self.rng.integers(
                -self.jitter_bed, self.jitter_bed + 1, size=len(self.bed)
            )
            bed = self.bed.with_columns(
                pl.col("chromStart") + shifts,
                pl.col("chromEnd") + shifts,
            )
            self.partitioned_bed = self.partition_bed(bed, self.max_length)

        if self.shuffle:
            self.rng.shuffle(self.partitioned_bed)

        self.batch_slice = slice(0, 0)

        if self.shuffle:
            self.indexes = {
                d: self.rng.permutation(idx) for d, idx in self.indexes.items()
            }

        batch: BatchDict
        self.partial_batches: List[BatchDict] = []
        self.partial_indices: List[NDArray[np.integer]] = []
        self.total_yielded = 0

        if self.num_workers > 1:
            buffers = ConcurrentBuffers(self)
        else:
            buffers = SyncBuffers(self)

        for buffer in buffers:
            new_stop = min(self.batch_size, self.batch_slice.stop + len(buffer))
            self.batch_slice = slice(self.batch_slice.stop, new_stop)
            len_batch_slice = self.batch_slice.stop - self.batch_slice.start
            buffer.idx_slice = slice(0, len_batch_slice)

            while buffer.len_unused_buffer > 0:
                idx = buffer.buffer_idx[buffer.idx_slice]

                if (
                    self.batch_slice.start == 0
                    and self.batch_slice.stop == self.batch_size
                ):
                    batch = self.select_from_buffer(buffer.buffer, idx)
                    batch_idx = idx.copy()
                else:
                    self.partial_batches.append(
                        self.select_from_buffer(buffer.buffer, idx)
                    )
                    self.partial_indices.append(idx)
                    if self.batch_slice.stop == self.batch_size:
                        batch = self.concat_batches(self.partial_batches)
                        batch_idx = np.concatenate(self.partial_indices)
                        self.partial_batches = []
                        self.partial_indices = []

                # full batch
                if self.batch_slice.stop == self.batch_size:
                    yield self.process_batch(
                        batch,
                        batch_idx,
                        buffer.dim_idxs,  # pyright: ignore[reportUnboundVariable]
                    )

                    self.total_yielded += self.batch_size

                    # full batch or take what's left in the buffer
                    new_stop = min(self.batch_size, buffer.len_unused_buffer)
                    self.batch_slice = slice(0, new_stop)
                # final batch incomplete
                elif self.total_yielded + self.batch_slice.stop == self.n_instances:
                    if self.drop_last:
                        return

                    # final incomplete batch is always a partial batch
                    batch = self.concat_batches(self.partial_batches)
                    batch_idx = np.concatenate(self.partial_indices, 0)

                    yield self.process_batch(batch, batch_idx, buffer.dim_idxs)

                    self.total_yielded += self.batch_slice.stop
                # batch incomplete and more data in buffer
                else:
                    # fill batch or take what's left in the buffer
                    new_stop = min(
                        self.batch_size,
                        self.batch_slice.stop + buffer.len_unused_buffer,
                    )
                    self.batch_slice = slice(self.batch_slice.stop, new_stop)

                buffer.len_unused_buffer -= len(idx)
                new_stop = min(
                    buffer.idx_slice.stop + self.batch_size, buffer.instances_in_buffer
                )
                buffer.idx_slice = slice(buffer.idx_slice.stop, new_stop)

    def dim_idxs_generator(self) -> Generator[Dict[Hashable, List[int]], None, None]:
        """Yields dictionaries of dims->indices"""
        if self.shuffle:
            idxs = {
                d: self.rng.permutation(len(idx)) for d, idx in self.indexes.items()
            }
        else:
            idxs = {d: np.arange(len(idx)) for d, idx in self.indexes.items()}
        batcher = product(
            *(chunked(idxs[dim], size) for dim, size in self.buffer_sizes.items())
        )
        for buffer_dim_idxs in batcher:
            yield dict(zip(self.buffer_sizes.keys(), buffer_dim_idxs))

    def get_buffer_idx(
        self,
        partition: pl.DataFrame,
        merged_starts: NDArray[np.int64],
        read_kwargs: Dict[Hashable, NDArray[np.int64]],
    ) -> NDArray[np.integer]:
        row_idx = partition.with_row_count()["row_nr"].to_numpy()
        buffer_indexes = [row_idx]
        for dim in self.batch_dims:
            if dim in read_kwargs:
                size = len(read_kwargs[dim])
            else:
                size = self.sizes[dim]
            buffer_indexes.append(np.arange(size))
        buffer_idx = _cartesian_product(buffer_indexes)
        # buffer_idx columns: starts, region_idx, dim1_idx, dim2_idx, ...
        rel_starts = get_relative_starts(
            partition["chromStart"].to_numpy(), merged_starts, self.fixed_length
        )[buffer_idx[:, 0], None]
        strands = partition["strand"].to_numpy()[buffer_idx[:, 0], None]
        region_idx = partition["region_idx"].to_numpy()[buffer_idx[:, 0], None]
        # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
        return np.hstack([rel_starts, strands, region_idx, buffer_idx[:, 1:]])

    def resample_buffer_idx(self, buffer_idx: NDArray):
        idx_weights = np.ones(len(buffer_idx))
        # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
        for i, d in enumerate(self.batch_dims, self.BUFFER_IDX_MIN_DIM_COL):
            # caller responsible for weights existing
            w = self.weights.get(d, None)  # pyright: ignore[reportOptionalMemberAccess]
            if w is not None:
                idx_weights *= w[buffer_idx[:, i]]
        idx_weights = np.round(idx_weights).astype(int)
        return buffer_idx.repeat(idx_weights)

    def select_from_buffer(
        self,
        buffer: xr.Dataset,
        idx: NDArray[np.integer],
    ) -> BatchDict:
        """Despite this unvectorized implementation, it is in fact faster than both a
        numba implementation (perhaps due to small batch sizes) and using fancy indexing
        (with or without xarray.Dataset.isel). This may be faster without vectorization
        because it avoids fancy indexing in the length dimension.
        """
        out: BatchDict = {}
        for name, reader in self.readers.items():
            a = buffer[name]
            dims = ["batch", *self.non_batch_dims, "length"]
            out[name] = (
                dims,
                np.empty_like(
                    a.values,
                    shape=(
                        len(idx),
                        *self.non_batch_dim_shape[name],
                        self.fixed_length,
                    ),
                ),
            )
            self.select_from_buffer_array(
                a.values,
                idx,
                self.buffer_idx_cols[name],
                out[name][1],
                self.BUFFER_IDX_START_COL,
                self.BUFFER_IDX_STRAND_COL,
                reader.rev_strand_fn,
            )
        # TODO slowest part of this function is the creation of the Dataset
        return out

    def select_from_buffer_array(
        self,
        arr: NDArray,
        idx: NDArray[np.integer],
        idx_cols: NDArray[np.integer],
        out: NDArray,
        start_col: int,
        strand_col: int,
        rev_strand_fn: Callable[[NDArray], NDArray],
    ):
        # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
        for i in range(len(idx)):
            _idx: NDArray[np.integer] = idx[i]
            indexer = [slice(None)] * arr.ndim
            indexer[: len(idx_cols)] = _idx[idx_cols]
            indexer[-1] = slice(_idx[start_col], _idx[start_col] + self.fixed_length)
            subarr = arr[tuple(indexer)]
            if _idx[strand_col] == -1:
                subarr = rev_strand_fn(subarr)
            out[i] = subarr

    def concat_batches(self, batches: List[BatchDict]) -> BatchDict:
        out: BatchDict = {}
        for name, (dims, _) in batches[0].items():
            out[name] = dims, np.concatenate(
                [batch[name][1] for batch in batches], axis=0
            )

        return out

    def process_batch(
        self,
        batch: BatchDict,
        batch_idx: NDArray,
        dim_idxs: Dict[Hashable, List[int]],
    ):
        out = {name: arr for name, (dim, arr) in batch.items()}

        if self.return_index:
            # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
            out["index"] = np.array(
                [
                    dim_idxs[d][batch_idx[:, i]]
                    for i, d in enumerate(self.batch_dims, self.BUFFER_IDX_MIN_DIM_COL)
                ]
            )

        if self.transform is not None:
            out = self.transform(out)

        if self.return_tuples:
            out = tuple(out.values())

        return out

    def torch_dataloader(self):
        if not TORCH_AVAILABLE:
            raise ImportError(
                """
                PyTorch is not included as a genvarloader dependency and must be 
                manually installed to get a Pytorch DataLoader. PyTorch has special 
                installation instructions, see: https://pytorch.org/get-started/locally/
                """
            )
        from torch.utils.data import DataLoader

        dataset = GVLDataset(self)
        dataloader = DataLoader(dataset, batch_size=None)
        return dataloader


@nb.njit(nogil=True, cache=True)
def partition_regions(
    starts: NDArray[np.int64], ends: NDArray[np.int64], max_length: int
):
    """
    Partitions regions based on their lengths and distances between them.

    Parameters
    ----------
    starts : numpy.ndarray
        Start positions for each region.
    ends : numpy.ndarray
        End positions for each region.
    max_length : int
        Maximum length of each partition.

    Returns
    -------
    numpy.ndarray : Array of partition numbers for each region.
    """
    partitions = np.zeros_like(starts)
    partition = 0
    curr_length = ends[0] - starts[0]
    for i in range(1, len(partitions)):
        curr_length += min(ends[i] - ends[i - 1], ends[i] - starts[i])
        if curr_length > max_length:
            partition += 1
            curr_length = ends[i] - starts[i]
        partitions[i] = partition
    return partitions


@nb.njit(nogil=True, cache=True)
def merge_overlapping_regions(starts: NDArray[np.int64], ends: NDArray[np.int64]):
    merged_starts = np.empty_like(starts)
    merged_ends = np.empty_like(ends)
    region_idx = 0
    merged_starts[region_idx] = starts[0]
    merged_ends[region_idx] = ends[0]
    for i in range(1, len(starts)):
        if starts[i] <= merged_ends[region_idx]:
            merged_ends[region_idx] = ends[i]
        else:
            region_idx += 1
            merged_starts[region_idx] = starts[i]
            merged_ends[region_idx] = ends[i]
    return merged_starts[: region_idx + 1], merged_ends[: region_idx + 1]


@nb.njit(nogil=True, cache=True)
def get_relative_starts(
    starts: NDArray[np.int64], merged_starts: NDArray[np.int64], length: int
):
    rel_starts = np.empty(len(starts), dtype=np.int64)
    region_idx = 0
    region_rel_start = 0

    for i in range(len(starts)):
        start = starts[i]

        if (
            region_idx != len(merged_starts) - 1
            and start >= merged_starts[region_idx + 1]
        ):
            region_idx += 1
            rel_starts[i] = rel_starts[i - 1] + length
            region_rel_start = rel_starts[i]
        else:
            merged_start = merged_starts[region_idx]
            rel_starts[i] = start - merged_start + region_rel_start

    return rel_starts


class SyncBuffers:
    def __init__(self, gvl: GVL) -> None:
        self.gvl = gvl

    def __iter__(self):
        return self.generate_buffers()

    def generate_buffers(self):
        buffer: Dict[Hashable, Tuple[Tuple[Hashable, ...], NDArray]] = {}
        for name, reader in self.gvl.readers.items():
            shape: List[int] = []
            for dim in reader.virtual_data.dims:
                # Not all dims are batch dims, so some will be missing from buffer_sizes
                size = self.gvl.buffer_sizes.get(dim, None)
                if size is not None:
                    shape.append(size)
            dtype = reader.virtual_data.dtype
            shape.append(self.gvl.max_length)
            buffer[name] = (reader.virtual_data.dims, np.empty(shape, dtype=dtype))

        for partition in self.gvl.partitioned_bed:
            n_regions = len(partition)
            instances_in_partition = n_regions * np.prod(
                [self.gvl.sizes[d] for d in self.gvl.batch_dims], dtype=int
            )
            instances_in_partition_tasks = 0

            contig: str
            contig = partition.select("chrom").row(0)[0]
            starts, ends = [
                c.to_numpy()
                for c in partition.select("chromStart", "chromEnd").get_columns()
            ]
            merged_starts, merged_ends = merge_overlapping_regions(starts, ends)
            lengths = ends - starts
            total_length = lengths.sum()
            dim_idxs_gen = self.gvl.dim_idxs_generator()

            while instances_in_partition_tasks < instances_in_partition:
                dim_idxs = next(dim_idxs_gen)
                read_kwargs = {
                    dim: self.gvl.indexes[dim][idx] for dim, idx in dim_idxs.items()
                }
                buffer_len = n_regions * np.prod(
                    [len(a) for a in read_kwargs.values()], dtype=int
                )
                instances_in_partition_tasks += buffer_len

                buffer_idx = self.gvl.get_buffer_idx(
                    partition, merged_starts, read_kwargs
                )
                read_kwargs["target_length"] = self.gvl.fixed_length

                if self.gvl.weights is not None:
                    buffer_idx = self.gvl.resample_buffer_idx(buffer_idx)

                if self.gvl.shuffle:
                    buffer_idx = self.gvl.rng.permutation(buffer_idx)

                sliced_buffer: Dict[Hashable, NDArray] = {}
                for name, (dims, arr) in buffer.items():
                    slices: List[slice] = []
                    for dim in dims:
                        # No chance of KeyError here because dims are exclusively
                        # batch_dims, which are checked for compat at GVL init
                        slices.append(slice(None, len(read_kwargs[dim])))
                    slices.append(slice(None, total_length))
                    _slices = tuple(slices)
                    sliced_buffer[name] = arr[_slices]
                _buffer = xr.Dataset(
                    {
                        name: r.read(
                            contig,
                            merged_starts,
                            merged_ends,
                            out=sliced_buffer[name],
                            **read_kwargs,
                        )
                        for name, r in self.gvl.readers.items()
                    }
                )
                yield Buffer(_buffer, buffer_idx, dim_idxs, -1)


class ConcurrentBuffers:
    def __init__(self, gvl: GVL) -> None:
        self.gvl = gvl

    def __iter__(self):
        self.ready_actor_idxs = list(range(len(self.gvl.actors)))
        self.buffer_submitter = self.submit_buffer_tasks()
        self.buffer_futures, self.buffer_meta = next(self.buffer_submitter)
        self.buffers: List[Buffer] = []
        return self

    def __next__(self):
        if len(self.buffers) == 0:
            buffers, self.buffer_futures = ray.wait(self.buffer_futures)
            buffers = cast(List[Tuple[DataVarsLike, int]], ray.get(buffers))
            self.buffers = [
                self.buffer_meta[
                    actor_idx
                ].to_buffer(  # pyright: ignore[reportOptionalMemberAccess]; we always have at least 1 buffer
                    buffer
                )
                for buffer, actor_idx in buffers
            ]
            self.ready_actor_idxs.extend([actor_idx for _, actor_idx in buffers])
            buffer_futures, self.buffer_meta = next(self.buffer_submitter)
            self.buffer_futures.extend(buffer_futures)

        return self.buffers.pop()

    def submit_buffer_tasks(self):
        buffer_futures: List[ray.ObjectRef[Buffer]] = []
        buffer_meta: List[Optional[BufferMeta]] = [None] * len(self.gvl.actors)
        for partition in self.gvl.partitioned_bed:
            {str(d): slice(0, 0) for d in self.gvl.buffer_sizes}

            n_regions = len(partition)
            instances_in_partition = n_regions * np.prod(
                [self.gvl.sizes[d] for d in self.gvl.batch_dims], dtype=int
            )
            instances_in_partition_tasks = 0

            contig: str
            start: int
            end: int
            contig, start, end = partition.select(
                pl.col("chrom").first(),
                pl.col("chromStart").min(),
                pl.col("chromEnd").max(),
            ).row(0)
            merged_starts, merged_ends = merge_overlapping_regions(
                partition["chromStart"].to_numpy(), partition["chromEnd"].to_numpy()
            )
            dim_idxs_gen = self.gvl.dim_idxs_generator()

            while instances_in_partition_tasks < instances_in_partition:
                if len(self.ready_actor_idxs) == 0:
                    yield buffer_futures, buffer_meta
                    buffer_futures = []

                dim_idxs = next(dim_idxs_gen)
                read_kwargs = {
                    dim: self.gvl.indexes[dim][idx] for dim, idx in dim_idxs.items()
                }
                buffer_len = n_regions * np.prod(
                    [len(a) for a in read_kwargs.values()], dtype=int
                )
                instances_in_partition_tasks += buffer_len
                buffer_idx = self.gvl.get_buffer_idx(
                    partition, merged_starts, read_kwargs
                )
                if self.gvl.weights is not None:
                    buffer_idx = self.gvl.resample_buffer_idx(buffer_idx)
                if self.gvl.shuffle:
                    buffer_idx = self.gvl.rng.permutation(buffer_idx)
                ready_actor_idx = self.ready_actor_idxs.pop()
                read_kwargs["target_length"] = self.gvl.fixed_length
                buffer = self.gvl.actors[ready_actor_idx].read.remote(  # type: ignore
                    contig, merged_starts, merged_ends, **read_kwargs
                )
                buffer_meta[ready_actor_idx] = BufferMeta(
                    buffer_idx, dim_idxs, ready_actor_idx
                )
                buffer_futures.append(buffer)

        if len(buffer_futures) > 0:
            yield buffer_futures, buffer_meta


if TORCH_AVAILABLE:
    from torch.utils.data import IterableDataset

    class GVLDataset(IterableDataset):
        def __init__(self, gvl: GVL):
            self.gvl = gvl

        def __len__(self):
            return len(self.gvl)

        def __iter__(self):
            return iter(self.gvl)

        def set(
            self,
            bed: Optional[Union[pl.DataFrame, str, Path]] = None,
            fixed_length: Optional[int] = None,
            batch_size: Optional[int] = None,
            max_memory_gb: Optional[float] = None,
            batch_dims: Optional[List[str]] = None,
            transform: Optional[
                Callable[[Dict[Hashable, NDArray]], Dict[Hashable, NDArray]]
            ] = None,
            shuffle: bool = False,
            weights: Optional[Dict[str, NDArray]] = None,
            seed: Optional[int] = None,
            return_tuples: bool = False,
            return_index: bool = False,
            drop_last: bool = False,
            jitter_bed: Optional[int] = None,
        ):
            """Update any parameters that don't require re-initializing Ray Actors. If
            you need to change readers or the number of workers, init a new GVL.

            Parameters
            ----------
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
            jitter_bed : int, optional
                Jitter the regions in the BED file by up to this many nucleotides.
            """
            self.gvl.set(
                bed,
                fixed_length,
                batch_size,
                max_memory_gb,
                batch_dims,
                transform,
                shuffle,
                weights,
                seed,
                return_tuples,
                return_index,
                drop_last,
                jitter_bed,
            )
