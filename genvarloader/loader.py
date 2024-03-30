from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numba as nb
import numpy as np
import polars as pl
import xarray as xr
from attrs import define
from more_itertools import chunked, interleave_longest
from numpy.typing import NDArray

from .concurrent import Buffer
from .haplotypes import Haplotypes
from .types import Reader
from .utils import (
    _cartesian_product,
    construct_virtual_data,
    get_rel_starts,
    process_bed,
)

try:
    import torch  # noqa: F401
    import torch.distributed

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


BatchDict = Dict[str, Tuple[List[str], NDArray]]


class GVL:
    # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
    BUFFER_IDX_START_COL = 0
    BUFFER_IDX_STRAND_COL = 1
    BUFFER_IDX_REGION_COL = 2
    BUFFER_IDX_MIN_DIM_COL = 3
    FUDGE_FACTOR = 6
    LENGTH_AXIS = -1
    MIN_BATCH_DIM_SIZES = {"sample": 100, "ploid": 2}

    def __init__(
        self,
        readers: Union[Reader, Haplotypes, Sequence[Union[Reader, Haplotypes]]],
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_size: int,
        max_memory_gb: float,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
        shuffle: bool = False,
        weights: Optional[Dict[str, NDArray]] = None,
        seed: Optional[int] = None,
        return_tuples: Union[List[str], Literal[False]] = False,
        return_index: bool = False,
        drop_last: bool = False,
        jitter_bed: Optional[int] = None,
        min_batch_dim_sizes: Optional[Dict[str, int]] = None,
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
        return_tuples : list[str] or False
            Whether to return a tuple instead of a dictionary, by default False.
            Outputs will be in the order specified by what is passed.
        return_index : bool, optional
            Whether to include an array of the indexes in the batch, by default False.
            Indices will be available from the "index" key or will be the last item in
            the tuple.
        drop_last : bool, optional
            Whether to drop the last batch if the number of instances are not evenly
            divisible by the batch size.
        jitter_bed : int, optional
            Jitter the regions in the BED file by up to this many nucleotides.
        min_batch_dim_sizes : Dict[str, int], optional
            Minimum size of each batch dimension. If None and shuffle = False, batch
            sizes are set to be as large as possible to maximize performance, otherwise
            heuristic defaults are used. Limiting the minimum size of a batch dimension
            is important for training to reduce correlation across batch dimensions.
            This is due to the buffering strategy used by GVL that dramatically improves
            performance vs. naively accessing the disk for every batch of data. For
            example, suppose you are training on a dataset of 10,000 diploid
            individuals. With no constraints on batch dimension size and sufficient
            memory, GVL would buffer all 10,000 individuals for a potentially small
            number of regions. This means a model would only see a small amount of
            sequence diversity for 20,000 instances before seeing new regions of the
            genome. This can decrease final model performance and cause large changes in
            training loss when new buffers are loaded (i.e. new regions of the genome
            are seen). TL;DR for best dataloading performance, min_batch_dim_sizes needs
            to be as large as possible. But for best training performance,
            min_batch_dim_sizes need to be small enough to reduce correlation across
            batch dimensions. Choosing min_batch_dim_sizes is a tradeoff between these
            two goals and ultimately must be done empirically.
        """
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
        self.min_batch_dim_sizes = min_batch_dim_sizes

        if not isinstance(readers, Iterable):
            readers = [readers]
        self.readers = readers
        self.unnested_readers: Dict[str, Reader] = {}
        for r in readers:
            if isinstance(r, Haplotypes):
                self.unnested_readers.update({_r.name: _r for _r in r.readers})
            else:
                self.unnested_readers[r.name] = r
        self.any_chunked = any(r.chunked for r in self.readers)

        # TODO raise warning if readers have different contig prefixes
        # can check via Reader.contig_starts_with_chr

        # TODO check if any regions are out-of-bounds and any readers have padding
        # disabled. If so, raise an error. Otherwise, readers will catch the error
        # downstream.
        if self.jitter_bed is not None:
            self.bed = process_bed(bed, self.fixed_length + self.jitter_bed)
        else:
            self.bed = process_bed(bed, self.fixed_length)

        self.virtual_data = construct_virtual_data(
            *self.unnested_readers.values(),
            n_regions=self.bed.height,
            fixed_length=self.fixed_length,
        )

        if TORCH_AVAILABLE and torch.distributed.is_initialized():  # pyright: ignore[reportPossiblyUnboundVariable]
            n_subsets = torch.distributed.get_world_size()  # pyright: ignore[reportPossiblyUnboundVariable]
            i = torch.distributed.get_rank()  # pyright: ignore[reportPossiblyUnboundVariable]
            subset_len = round(self.bed.height / n_subsets)
            slice_start = i * subset_len

            if i + 1 < n_subsets:
                slice_stop = (i + 1) * subset_len
            else:
                slice_stop = self.bed.height
            region_slice = slice(slice_start, slice_stop)

            self.bed = self.bed[region_slice]
            self.virtual_data = self.virtual_data.isel(region=region_slice)

        # sizes does not include the length dimension
        self.sizes = cast(Dict[str, int], dict(self.virtual_data.sizes))
        del self.sizes["region"]
        del self.sizes["length"]
        # dimension -> sum of itemsizes across readers with that dimension
        self.itemsizes: Mapping[str, int] = defaultdict(int)
        # indexes does not include the length dimension
        self.indexes = {k: a.values for k, a in self.virtual_data.coords.items()}
        for arr in self.virtual_data.values():
            for dim in arr.dims:
                if dim == "length":
                    continue
                self.itemsizes[cast(str, dim)] += arr.dtype.itemsize

        if batch_dims is None:
            batch_dims = []
        self.batch_dims = batch_dims

        if missing_dims := (set(self.batch_dims) - set(self.sizes)):  # type: ignore
            raise ValueError(
                f"Got batch dimensions that are not available in any reader: {missing_dims}"
            )

        self.non_batch_dims = [d for d in self.sizes.keys() if d not in self.batch_dims]
        self.non_batch_dim_shape: Dict[str, List[int]] = {}
        # Mapping from array name to axis number in the buffer index column
        self.buffer_idx_cols: Dict[str, NDArray[np.integer]] = {}
        # Mapping from array name to axes in the buffer corresponding to each idx col
        self.buffer_idx_col_axes: Dict[str, NDArray[np.integer]] = {}
        # Mapping from array name to the axis number of the batch dimension after
        # vectorized indexing of the buffer to get a batch
        self.buffer_batch_axis: Dict[str, int] = {}
        for name, a in self.virtual_data.items():
            name = cast(str, name)
            self.non_batch_dim_shape[name] = [
                a.sizes[d] for d in self.non_batch_dims if d in a.dims
            ]
            # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
            idx_cols = [
                self.batch_dims.index(cast(str, d)) + self.BUFFER_IDX_MIN_DIM_COL
                for d in a.dims
                if d in self.batch_dims
            ]
            self.buffer_idx_cols[name] = np.array(idx_cols, dtype=int)

            idx_col_axes = [i for i, d in enumerate(a.dims) if d in self.batch_dims]
            self.buffer_idx_col_axes[name] = np.array(idx_col_axes, dtype=int)

            if len(idx_col_axes) == 0:
                # after vectorized indexing with no batch dims, the batch axis is the
                # length axis
                self.buffer_batch_axis[name] = a.get_axis_num(  # type: ignore[assignment]
                    "length"
                )
            else:
                # otherwise, it will be the smallest axis
                self.buffer_batch_axis[name] = (
                    min(idx_col_axes)
                    if np.all(np.diff(idx_col_axes + [a.get_axis_num("length")]) == 1)
                    else 0
                )

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
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
        shuffle: bool = False,
        weights: Optional[Dict[str, NDArray]] = None,
        seed: Optional[int] = None,
        return_tuples: Optional[Union[List[str], Literal[False]]] = None,
        return_index: bool = False,
        drop_last: bool = False,
        min_batch_dim_sizes: Optional[Dict[str, int]] = None,
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
        return_tuples : list[str] or False
            Whether to return a tuple instead of a dictionary, by default False.
            Outputs will be in the order specified by what is passed.
        return_index : bool, optional
            Whether to include an array of the indexes in the batch, by default False
        drop_last : bool, optional
            Whether to drop the last batch if the number of instances are not evenly
            divisible by the batch size.
        min_batch_dim_sizes : Dict[str, int], optional
            Minimum size of each batch dimension. If None and shuffle = False, batch
            sizes are set to be as large as possible to maximize performance, otherwise
            heuristic defaults are used. Limiting the minimum size of a batch dimension
            is important for training to reduce correlation across batch dimensions.
            This is due to the buffering strategy used by GVL that dramatically improves
            performance vs. naively accessing the disk for every batch of data. For
            example, suppose you are training on a dataset of 10,000 diploid
            individuals. With no constraints on batch dimension size and sufficient
            memory, GVL would buffer all 10,000 individuals for a potentially small
            number of regions. This means a model would only see a small amount of
            sequence diversity for 20,000 instances before seeing new regions of the
            genome. This can decrease final model performance and cause large changes in
            training loss when new buffers are loaded (i.e. new regions of the genome
            are seen). TL;DR for best dataloading performance, min_batch_dim_sizes needs
            to be as large as possible. But for best training performance,
            min_batch_dim_sizes need to be small enough to reduce correlation across
            batch dimensions. Choosing min_batch_dim_sizes is a tradeoff between these
            two goals and ultimately must be done empirically.
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
        if min_batch_dim_sizes is not None:
            self.min_batch_dim_sizes = min_batch_dim_sizes

        if bed is not None:
            if self.jitter_bed is not None:
                self.bed = process_bed(bed, self.fixed_length + self.jitter_bed)
            else:
                self.bed = process_bed(bed, self.fixed_length)

        if batch_dims is not None:
            self.batch_dims = batch_dims
            if missing_dims := (set(self.batch_dims) - set(self.virtual_data.dims)):  # type: ignore
                raise ValueError(
                    f"Got batch dimensions that are not available from the readers: {missing_dims}"
                )
            self.non_batch_dims = [
                d for d in self.sizes.keys() if d not in self.batch_dims
            ]
            self.non_batch_dim_shape = {}
            self.buffer_idx_cols = {}
            self.buffer_idx_col_axes = {}
            for name, a in self.virtual_data.items():
                name = cast(str, name)
                self.non_batch_dim_shape[name] = [
                    a.sizes[d] for d in self.non_batch_dims
                ]
                # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
                idx_cols = [
                    self.batch_dims.index(cast(str, d)) + self.BUFFER_IDX_MIN_DIM_COL
                    for d in a.dims
                    if d in self.batch_dims
                ]
                self.buffer_idx_cols[name] = np.array(idx_cols, dtype=int)

                idx_col_axes = [i for i, d in enumerate(a.dims) if d in self.batch_dims]
                self.buffer_idx_col_axes[name] = np.array(idx_col_axes, dtype=int)

                if len(idx_col_axes) == 0:
                    # after vectorized indexing with no batch dims, the batch axis is the
                    # length axis
                    self.buffer_batch_axis[name] = a.get_axis_num(  # type: ignore[assignment]
                        "length"
                    )
                else:
                    # otherwise, it will be the smallest axis
                    self.buffer_batch_axis[name] = (
                        min(idx_col_axes)
                        if np.all(
                            np.diff(idx_col_axes + [a.get_axis_num("length")]) == 1
                        )
                        else 0
                    )

        if None not in (
            fixed_length,
            bed,
            max_memory_gb,
            batch_dims,
            min_batch_dim_sizes,
        ):
            self.max_length = self.get_max_length()
            self.partitioned_bed = self.partition_bed(self.bed, self.max_length)
            self.n_instances = self.bed.height * np.prod(
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
        if self.drop_last:
            return self.n_instances // self.batch_size
        else:
            # ceil
            return -(-self.n_instances // self.batch_size)

    def mem_per_length(self, sizes: Mapping[str, int]):
        mpl = sum(sizes[dim] * self.itemsizes[dim] for dim in sizes)
        mpl = max(1, mpl)
        return mpl

    def get_max_length(self):
        """Get the maximum length"""
        batch_mem = self.fixed_length * self.mem_per_length(
            {k: v for k, v in self.sizes.items() if k not in self.batch_dims}
        )
        max_mem = int(
            (self.max_memory_gb * 1e9 - batch_mem - self.bed.estimated_size())
            / self.FUDGE_FACTOR
        )

        if self.shuffle:
            # assume training and need to ~uniformly sample across genomes
            # before sampling across batch dimensions
            # longest possible length, will minimize batch dim sizes
            offsets = np.empty(self.bed["chrom"].n_unique() + 1, dtype=np.uint32)
            offsets[0] = 0
            offsets[1:] = (
                self.bed.group_by("chrom", maintain_order=True)
                .count()["count"]
                .cum_sum()
                .to_numpy()
            )
            max_length = max_length_of_regions(
                self.bed["chromStart"].to_numpy(),
                self.bed["chromEnd"].to_numpy(),
                offsets,
            )

            # max_length = 0
            # for partition in self.bed.partition_by("chrom"):
            #     merged_starts, merged_ends = merge_overlapping_regions_one_contig(
            #         partition["chromStart"].to_numpy(), partition["chromEnd"].to_numpy()
            #     )
            #     max_length = max(max_length, (merged_ends - merged_starts).sum())
        else:
            # assume not training, so sampling does not matter, just maximize
            # throughput
            # shortest possible length, will maximize batch dim sizes
            max_length = self.fixed_length
        self.buffer_sizes = self.get_buffer_sizes(max_mem, max_length)

        # longest possible length that will fit into memory given buffer sizes
        max_length = max_mem // self.mem_per_length(self.buffer_sizes)
        if max_length == 0:
            min_mem = (
                (self.mem_per_length(self.buffer_sizes) + batch_mem)
                * self.FUDGE_FACTOR
                / 1e9
            )
            raise ValueError(
                f"Not enough memory to process dataset. Minimum memory needed: {min_mem:.4f} GB."
            )
        return max_length

    def get_buffer_sizes(self, max_mem: int, max_length: int) -> Dict[str, int]:
        """Get the size of batch dimensions such that the largest buffer (i.e. with max
        length) will fit into memory."""
        buffer_sizes = deepcopy(self.sizes)
        for dim in self.batch_dims:
            del buffer_sizes[dim]
            size = int(
                (max_mem / max_length - self.mem_per_length(buffer_sizes))
                / self.itemsizes[dim]
            )
            size = np.clip(size, 1, self.sizes[dim])
            if self.shuffle and self.min_batch_dim_sizes is None:
                size = max(size, self.MIN_BATCH_DIM_SIZES.get(dim, size))
            elif self.shuffle and self.min_batch_dim_sizes is not None:
                size = max(size, self.min_batch_dim_sizes.get(dim, size))
            buffer_sizes[dim] = size
        return buffer_sizes

    def partition_bed(
        self,
        bed: pl.DataFrame,
        max_length: int,
    ) -> List["PartitionOfRegions"]:
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
        dim_idxs = [dim_idx for dim_idx in self.dim_idxs_generator()]

        if self.shuffle:
            self.rng.shuffle(dim_idxs)  # type: ignore[arg-type]

        offsets: NDArray[np.uint32] = np.empty(
            bed["chrom"].n_unique() + 1, dtype=np.uint32
        )
        offsets[0] = 0
        offsets[1:] = (
            bed.group_by("chrom", maintain_order=True)
            .count()["count"]
            .cum_sum()
            .to_numpy()
        )
        partitions = partition_regions(
            bed["chromStart"].to_numpy(),
            bed["chromEnd"].to_numpy(),
            offsets,
            max_length,
        )
        partitions = bed.with_columns(partition=pl.lit(partitions)).partition_by(
            "chrom", "partition"
        )
        partitions = [PartitionOfRegions(part, dim_idxs) for part in partitions]

        return partitions

    def __iter__(self):
        return self.iter_batches()

    def iter_batches(
        self,
    ) -> Generator[Union[Tuple[NDArray, ...], Dict[str, NDArray]], None, None]:
        for partreg in self.partitioned_bed:
            partreg._reset_counter()

        if self.shuffle:
            self.rng.shuffle(self.partitioned_bed)  # type: ignore[arg-type]
            self.indexes = {
                d: self.rng.permutation(idx) for d, idx in self.indexes.items()
            }

        batch: BatchDict
        batch_slice = slice(0, 0)
        self.partial_batches: List[BatchDict] = []
        self.partial_indices: List[NDArray[np.integer]] = []
        self.total_yielded = 0

        buffers = SyncBuffers(self)

        for buffer in buffers:
            while buffer.len_unused_buffer > 0:
                new_stop = min(
                    self.batch_size, batch_slice.stop + buffer.len_unused_buffer
                )
                batch_slice = slice(batch_slice.stop, new_stop)

                len_buffer_to_batch = batch_slice.stop - batch_slice.start
                new_stop = min(len(buffer), buffer.idx_slice.stop + len_buffer_to_batch)
                buffer.idx_slice = slice(buffer.idx_slice.stop, new_stop)

                idx = buffer.buffer_idx[buffer.idx_slice]

                # copy data from buffer into batch
                # full batch
                if batch_slice.start == 0 and batch_slice.stop == self.batch_size:
                    batch = self.select_from_buffer(buffer.buffer, idx)
                    batch_idx = idx.copy()
                # partial batch
                else:
                    self.partial_batches.append(
                        self.select_from_buffer(buffer.buffer, idx)
                    )
                    self.partial_indices.append(idx)

                buffer.len_unused_buffer -= len_buffer_to_batch

                # full batch
                if batch_slice.stop == self.batch_size:
                    if len(self.partial_batches) > 0:
                        batch = self.concat_batches(self.partial_batches)
                        batch_idx = np.concatenate(self.partial_indices, axis=0)
                        self.partial_batches = []
                        self.partial_indices = []

                    yield self.process_batch(
                        batch,  # pyright: ignore[reportPossiblyUnboundVariable]
                        batch_idx,  # pyright: ignore[reportPossiblyUnboundVariable]
                        buffer.dim_idxs,
                    )

                    self.total_yielded += self.batch_size
                    batch_slice = slice(0, 0)

        # final batch incomplete
        if not self.drop_last and self.total_yielded < self.n_instances:
            # final incomplete batch is always a partial batch
            batch = self.concat_batches(self.partial_batches)
            batch_idx = np.concatenate(self.partial_indices, axis=0)

            yield self.process_batch(
                batch,
                batch_idx,
                buffer.dim_idxs,  # pyright: ignore[reportPossiblyUnboundVariable]
            )

            self.total_yielded += batch_slice.stop

    def dim_idxs_generator(self) -> Generator[Dict[str, List[int]], None, None]:
        """Yields dictionaries of dims->indices"""
        # Chunked formats require access patterns such that data locality is respected
        # i.e. data must be accessed that is close to each other in on disk.
        # Without precise knowledge of the chunk boundaries of underlying every reader,
        # the best we can do is to simply ensure every data access is from contiguous
        # data.
        if self.shuffle and not self.any_chunked:
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
        read_kwargs: Dict[str, NDArray[np.int64]],
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
        rel_starts = get_rel_starts(
            partition["chromStart"].to_numpy(), partition["chromEnd"].to_numpy()
        )[buffer_idx[:, 0], None]
        if self.jitter_bed is not None:
            rel_starts += self.rng.integers(
                2 * self.jitter_bed + 1, size=len(rel_starts)
            )
        strands = partition["strand"].to_numpy()[buffer_idx[:, 0], None]
        region_idx = partition["region_idx"].to_numpy()[buffer_idx[:, 0], None]
        # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
        return np.hstack([rel_starts, strands, region_idx, buffer_idx[:, 1:]])

    def resample_buffer_idx(self, buffer_idx: NDArray):
        idx_weights = np.ones(len(buffer_idx))
        # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
        for i, d in enumerate(self.batch_dims, self.BUFFER_IDX_MIN_DIM_COL):
            # caller responsible for weights existing
            w = self.weights.get(d, None)  # type: ignore[union-attr]
            if w is not None:
                idx_weights *= w[buffer_idx[:, i]]
        idx_weights = np.round(idx_weights).astype(int)
        return buffer_idx.repeat(idx_weights)  # type: ignore[arg-type]

    def select_from_buffer(
        self,
        buffer: xr.Dataset,
        idx: NDArray[np.integer],
    ) -> BatchDict:
        out: BatchDict = {}
        for name, reader in self.unnested_readers.items():
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
                self.buffer_idx_col_axes[name],
                self.buffer_batch_axis[name],
                out[name][1],
                reader.rev_strand_fn,
            )
        return out

    def select_from_buffer_array(
        self,
        arr: NDArray,
        idx: NDArray[np.integer],
        idx_cols: NDArray[np.integer],
        idx_col_axes: NDArray[np.integer],
        batch_axis: int,
        out: NDArray,
        rev_strand_fn: Callable[[NDArray], NDArray],
    ):
        _idx_cols: List[int] = idx_cols.tolist()
        # use vectorized indexing for large batch sizes
        if len(idx) > 32:
            # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
            indexer: List[Union[slice, NDArray]] = [slice(None)] * arr.ndim
            for axis, col in zip(idx_col_axes, _idx_cols):
                indexer[axis] = idx[:, col]
            indexer[self.LENGTH_AXIS] = idx[:, self.BUFFER_IDX_START_COL]
            view = np.lib.stride_tricks.sliding_window_view(
                arr, self.fixed_length, axis=self.LENGTH_AXIS
            )
            # shape: (..., batch, ..., fixed_length)
            batch = view[tuple(indexer)]
            # shape: (batch, ..., fixed_length)
            out[:] = np.moveaxis(batch, batch_axis, 0)
            rev_strand = idx[:, self.BUFFER_IDX_STRAND_COL] == -1
            if (rev_strand).any():
                out[rev_strand] = rev_strand_fn(out[rev_strand])
        else:
            for i in range(len(idx)):
                _idx: NDArray[np.integer] = idx[i]
                indexer = [slice(None)] * arr.ndim
                for axis, col in zip(idx_col_axes, _idx_cols):
                    indexer[axis] = _idx[col]
                indexer[self.LENGTH_AXIS] = slice(
                    _idx[self.BUFFER_IDX_START_COL],
                    _idx[self.BUFFER_IDX_START_COL] + self.fixed_length,
                )
                # shape: (..., fixed_length)
                subarr = arr[tuple(indexer)]
                if _idx[self.BUFFER_IDX_STRAND_COL] == -1:
                    subarr = rev_strand_fn(subarr)
                out[i] = subarr

    def concat_batches(self, batches: List[BatchDict]) -> BatchDict:
        out: BatchDict = {}
        for name, (dims, _) in batches[0].items():
            out[name] = (
                dims,
                np.concatenate([batch[name][1] for batch in batches], axis=0),
            )
        return out

    def process_batch(
        self,
        batch: BatchDict,
        batch_idx: NDArray,
        dim_idxs: Dict[str, List[int]],
    ):
        out = {name: arr for name, (dim, arr) in batch.items()}

        if self.return_index:
            # buffer_idx columns: starts, strands, region_idx, dim1_idx, dim2_idx, ...
            out["index"] = np.array(
                [batch_idx[:, self.BUFFER_IDX_REGION_COL]]
                + [
                    dim_idxs[d][batch_idx[:, i]]
                    for i, d in enumerate(self.batch_dims, self.BUFFER_IDX_MIN_DIM_COL)
                ]
            )

        if self.transform is not None:
            out = self.transform(out)

        if self.return_tuples:
            out = tuple(out[name] for name in self.return_tuples)  # type: ignore[assignment]

        return out

    def torch_dataset(self):
        if not TORCH_AVAILABLE:
            raise ImportError(
                """
                PyTorch is not included as a genvarloader dependency and must be 
                manually installed to get a Pytorch Dataset. PyTorch has special 
                installation instructions, see: https://pytorch.org/get-started/locally/
                """
            )
        return GVLDataset(self)

    def torch_dataloader(self):
        if not TORCH_AVAILABLE:
            raise ImportError(
                """
                PyTorch is not included as a genvarloader dependency and must be 
                manually installed to get a Pytorch DataLoader. PyTorch has special 
                installation instructions, see: https://pytorch.org/get-started/locally/
                """
            )
        return self.torch_dataset().torch_dataloader()


@nb.njit(parallel=True, nogil=True, cache=True)
def partition_regions(
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    offsets: NDArray[np.uint32],
    max_length: int,
):
    """
    Partitions regions based on their lengths and distances between them.

    Parameters
    ----------
    starts : numpy.ndarray
        Start positions for each region.
    ends : numpy.ndarray
        End positions for each region.
    offsets : numpy.ndarray
        Offsets for each region.
    max_length : int
        Maximum length of each partition.

    Returns
    -------
    numpy.ndarray : Array of partition numbers for each region.
    """
    partitions: NDArray[np.uint32] = np.zeros_like(starts, dtype=np.uint32)
    for i in nb.prange(len(offsets) - 1):
        s = offsets[i]
        e = offsets[i + 1]
        partition_regions_one_contig(
            starts[s:e], ends[s:e], max_length, out=partitions[s:e]
        )
    return partitions


@nb.njit(nogil=True, cache=True)
def partition_regions_one_contig(
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    max_length: int,
    out: NDArray[np.uint32],
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
    partition = 0
    curr_length = ends[0] - starts[0]
    for i in range(1, len(starts)):
        curr_length += min(ends[i] - ends[i - 1], ends[i] - starts[i])
        if curr_length > max_length:
            partition += 1
            curr_length = ends[i] - starts[i]
        out[i] = partition
    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def max_length_of_regions(starts, ends, offsets):
    """
    Merge overlapping regions, assuming they are sorted by start position.

    Parameters
    ----------
    starts : numpy.ndarray
        Start positions for each region.
    ends : numpy.ndarray
        End positions for each region.
    offsets : numpy.ndarray
        Offsets for each region.

    Returns
    -------
    max_length : int
    """
    max_lengths = np.zeros(len(offsets) - 1, dtype=np.int64)
    for i in nb.prange(len(offsets) - 1):
        s = offsets[i]
        e = offsets[i + 1]
        merged_starts, merged_ends = merge_overlapping_regions_one_contig(
            starts[s:e], ends[s:e]
        )
        max_lengths[i] = (merged_ends - merged_starts).sum()
    return max_lengths.max()


@nb.njit(nogil=True, cache=True)
def merge_overlapping_regions_one_contig(
    starts: NDArray[np.int64], ends: NDArray[np.int64]
):
    """Merge overlapping regions, assuming they are sorted by start position.

    Parameters
    ----------
    starts : NDArray[np.int64]
        Start positions for each region.
    ends : NDArray[np.int64]
        End positions for each region.

    Returns
    -------
    merged_starts : NDArray[np.int64]
        Start positions for each merged region.
    merged_ends : NDArray[np.int64]
        End positions for each merged region.
    """
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


@define
class PartitionOfRegions:
    partition: pl.DataFrame
    dim_idxs: List[Dict[str, List[int]]]
    counter: int = -1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dim_idxs)

    def __next__(self):
        self.counter += 1
        if self.counter == len(self):
            raise StopIteration
        return self.partition, self.dim_idxs[self.counter]

    def _reset_counter(self):
        self.counter = -1


class SyncBuffers:
    def __init__(self, gvl: GVL) -> None:
        self.gvl = gvl

    def __iter__(self):
        return self.generate_buffers()

    def generate_buffers(self) -> Generator[Buffer, None, None]:
        for partition, dim_idxs in interleave_longest(*self.gvl.partitioned_bed):
            # contig: str
            contig = partition.select("chrom").row(0)[0]
            starts, ends = [
                c.to_numpy()
                for c in partition.select("chromStart", "chromEnd").get_columns()
            ]
            merged_starts, merged_ends = merge_overlapping_regions_one_contig(
                starts, ends
            )

            read_kwargs = {
                dim: self.gvl.indexes[dim][idx] for dim, idx in dim_idxs.items()
            }

            buffer_idx = self.gvl.get_buffer_idx(partition, merged_starts, read_kwargs)

            if self.gvl.weights is not None:
                buffer_idx = self.gvl.resample_buffer_idx(buffer_idx)

            if self.gvl.shuffle:
                buffer_idx = self.gvl.rng.permutation(buffer_idx)

            _buffer_dict: Dict[str, Tuple[Tuple[str, ...], NDArray]] = {}
            for reader in self.gvl.readers:
                if isinstance(reader, Haplotypes):
                    data = reader.read(
                        contig,
                        merged_starts,
                        merged_ends,
                        **read_kwargs,
                    )
                    _buffer_dict.update(data)
                else:
                    data = reader.read(
                        contig,
                        merged_starts,
                        merged_ends,
                        **read_kwargs,
                    )
                    _buffer_dict[reader.name] = ((*reader.sizes.keys(), "length"), data)

            out_buffer = xr.Dataset(_buffer_dict)
            yield Buffer(out_buffer, buffer_idx, dim_idxs, -1)


if TORCH_AVAILABLE:
    from torch.utils.data import DataLoader, IterableDataset

    class GVLDataset(IterableDataset):
        def __init__(self, gvl: GVL):
            self.gvl = gvl

        def __len__(self):
            return len(self.gvl)

        def __iter__(self):
            return iter(self.gvl)

        def torch_dataloader(self):
            return DataLoader(self, batch_size=None)

        def set(
            self,
            bed: Optional[Union[pl.DataFrame, str, Path]] = None,
            fixed_length: Optional[int] = None,
            batch_size: Optional[int] = None,
            max_memory_gb: Optional[float] = None,
            batch_dims: Optional[List[str]] = None,
            transform: Optional[
                Callable[[Dict[str, NDArray]], Dict[str, NDArray]]
            ] = None,
            shuffle: bool = False,
            weights: Optional[Dict[str, NDArray]] = None,
            seed: Optional[int] = None,
            return_tuples: Union[List[str], Literal[False]] = False,
            return_index: bool = False,
            drop_last: bool = False,
            min_batch_dim_sizes: Optional[Dict[str, int]] = None,
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
            min_batch_dim_sizes : Dict[str, int], optional
                Minimum size of each batch dimension. If None and shuffle = False, batch
                sizes are set to be as large as possible to maximize performance, otherwise
                heuristic defaults are used. Limiting the minimum size of a batch dimension
                is important for training to reduce correlation across batch dimensions.
                This is due to the buffering strategy used by GVL that dramatically improves
                performance vs. naively accessing the disk for every batch of data. For
                example, suppose you are training on a dataset of 10,000 diploid
                individuals. With no constraints on batch dimension size and sufficient
                memory, GVL would buffer all 10,000 individuals for a potentially small
                number of regions. This means a model would only see a small amount of
                sequence diversity for 20,000 instances before seeing new regions of the
                genome. This can decrease final model performance and cause large changes in
                training loss when new buffers are loaded (i.e. new regions of the genome
                are seen). TL;DR for best dataloading performance, min_batch_dim_sizes needs
                to be as large as possible. But for best training performance,
                min_batch_dim_sizes need to be small enough to reduce correlation across
                batch dimensions. Choosing min_batch_dim_sizes is a tradeoff between these
                two goals and ultimately must be done empirically.
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
                min_batch_dim_sizes,
            )
