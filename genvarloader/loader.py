from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import (
    Callable,
    Dict,
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


class GVL:
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
        num_workers: int = 2,
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
        num_workers : int, optional
            How many workers to use for concurrent I/O, default 2. Recommended to set
            this to the number of processors available.
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

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_workers - 1)

        if not isinstance(readers, Iterable):
            readers = [readers]
        self.readers = {r.virtual_data.name: r for r in readers}

        self.actors: List[ReaderActor] = [
            ReaderActor.remote(*self.readers, actor_idx=i)
            for i in range(self.num_workers - 1)  # keep 1 cpu for main process
        ]

        self.virtual_data = xr.merge(
            [r.virtual_data for r in self.readers.values()], join="exact"
        )
        self.sizes = dict(self.virtual_data.sizes)
        self.sizes.pop("", None)
        self.itemsizes: Mapping[Hashable, int] = defaultdict(int)
        self.indexes = {k: a.values for k, a in self.virtual_data.coords.items()}
        for arr in self.virtual_data.data_vars.values():
            for dim in arr.dims:
                if dim == "":
                    continue
                self.itemsizes[dim] += arr.dtype.itemsize

        if isinstance(bed, (str, Path)):
            bed = read_bedlike(bed)
        bed = bed.with_row_count("region_idx")
        with pl.StringCache():
            pl.Series(natsorted(bed["chrom"].unique()), dtype=pl.Categorical)
            bed = bed.sort(pl.col("chrom").cast(pl.Categorical), "chromStart")
        self.bed = _set_fixed_length_around_center(bed, fixed_length)
        # TODO check if any regions are out-of-bounds and any readers have padding disabled
        # if so, raise an error. Otherwise, readers will catch the error downstream.

        if batch_dims is None:
            batch_dims = []
        self.batch_dims = batch_dims

        if missing_dims := (set(self.batch_dims) - set(self.virtual_data.dims)):  # type: ignore
            raise ValueError(
                f"Got batch dimensions that are not available from the readers: {missing_dims}"
            )
        self.non_batch_dims = [d for d in self.sizes.keys() if d not in self.batch_dims]
        self.non_batch_dim_shape: Dict[Hashable, List[int]] = {}
        self.buffer_idx_cols: Dict[Hashable, NDArray[np.integer]] = {}
        for k, a in self.virtual_data.data_vars.items():
            self.non_batch_dim_shape[k] = [a.sizes[d] for d in self.non_batch_dims]
            # buffer_idx columns: starts, region_idx, dim1_idx, dim2_idx, ...
            idx_cols = [i for i, d in enumerate(self.batch_dims, 2) if d in a.sizes]
            self.buffer_idx_cols[k] = np.array(idx_cols, dtype=int)

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
        return_tuples: bool = False,
        return_index: bool = False,
        drop_last: bool = False,
    ):
        """Update any parameters that don't require re-initializing Ray Actors. If you
        need to change readers or the number of workers, init a new GVL.

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
        """
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
        if fixed_length is not None:
            self.fixed_length = fixed_length
        if max_memory_gb is not None:
            self.max_memory_gb = max_memory_gb

        if bed is not None:
            if isinstance(bed, (str, Path)):
                bed = read_bedlike(bed)
            bed = bed.with_row_count("region_idx")

            with pl.StringCache():
                pl.Series(natsorted(bed["chrom"].unique()), dtype=pl.Categorical)
                bed = bed.sort(pl.col("chrom").cast(pl.Categorical), "chromStart")
            self.bed = _set_fixed_length_around_center(bed, self.fixed_length)

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
            for k, a in self.virtual_data.data_vars.items():
                self.non_batch_dim_shape[k] = [a.sizes[d] for d in self.non_batch_dims]
                # buffer_idx columns: starts, region_idx, dim1_idx, dim2_idx, ...
                idx_cols = [i for i, d in enumerate(self.batch_dims, 2) if d in a.sizes]
                self.buffer_idx_cols[k] = np.array(idx_cols, dtype=int)

        if None not in (fixed_length, bed, max_memory_gb, batch_dims):
            self.max_length = self.get_max_length()
            self.partitioned_bed = self.partition_bed(self.bed, self.max_length)
            self.n_instances: int = self.bed.height * np.prod(
                [self.sizes[d] for d in self.batch_dims], dtype=int
            )

        if batch_size is not None:
            self.batch_size = batch_size

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
        if self.shuffle:
            self.rng.shuffle(self.partitioned_bed)

        self.batch_slice = slice(0, 0)

        if self.shuffle:
            self.indexes = {
                d: self.rng.permutation(idx) for d, idx in self.indexes.items()
            }

        self.partial_batches = []
        self.partial_indices = []
        self.total_yielded = 0

        buffers = Buffers(self)

        for buffer in buffers:
            new_stop = min(self.batch_size, self.batch_slice.stop + len(buffer))
            self.batch_slice = slice(self.batch_slice.stop, new_stop)
            len_batch_slice = self.batch_slice.stop - self.batch_slice.start
            buffer.idx_slice = slice(0, len_batch_slice)

            while buffer.len_unused_buffer > 0:
                idx = buffer.buffer_idx[buffer.idx_slice]

                batch: xr.Dataset
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
                        batch = xr.concat(self.partial_batches, dim="batch")
                        batch_idx = np.concatenate(self.partial_indices)
                        self.partial_batches = []
                        self.partial_indices = []

                # full batch
                if self.batch_slice.stop == self.batch_size:
                    yield self.process_batch(batch, batch_idx, buffer.dim_slices)  # type: ignore

                    self.total_yielded += self.batch_size

                    # full batch or take what's left in the buffer
                    new_stop = min(self.batch_size, buffer.len_unused_buffer)
                    self.batch_slice = slice(0, new_stop)
                # final batch incomplete
                elif self.total_yielded + self.batch_slice.stop == self.n_instances:
                    if self.drop_last:
                        return

                    # final incomplete batch is always a partial batch
                    batch = xr.concat(self.partial_batches, dim="batch")
                    batch_idx = np.concatenate(self.partial_indices, 0)

                    yield self.process_batch(batch, batch_idx, buffer.dim_slices)

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

    def increment_dim_slices(self, dim_slices: Dict[str, slice]):
        dim_slices = {
            d: slice(s.stop, s.stop + size)
            for (d, s), size in zip(dim_slices.items(), self.buffer_sizes.values())
        }
        return dim_slices

    def get_buffer_idx(
        self,
        partition: pl.DataFrame,
        start: int,
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
        # columns: starts, region_idx, dim1_idx, dim2_idx, ...
        starts = (partition["chromStart"].to_numpy() - start)[buffer_idx[:, 0], None]
        region_idx = partition["region_idx"].to_numpy()[buffer_idx[:, 0], None]
        return np.hstack([starts, region_idx, buffer_idx[:, 1:]])

    def resample_buffer_idx(self, buffer_idx: NDArray):
        idx_weights = np.ones(len(buffer_idx))
        # buffer_idx columns: starts, region_idx, dim1_idx, dim2_idx, ...
        for i, d in enumerate(self.batch_dims):
            # caller responsible for weights existing
            w = self.weights.get(d, None)  # type: ignore[reportOptionalMemberAccess]
            if w is not None:
                idx_weights *= w[buffer_idx[:, i + 2]]
        idx_weights = np.round(idx_weights).astype(int)
        return buffer_idx.repeat(idx_weights)

    def process_batch(
        self, batch: xr.Dataset, batch_idx: NDArray, dim_slices: Dict[str, slice]
    ):
        batch = batch.transpose("batch", ...)
        out = {name: arr.values for name, arr in batch.data_vars.items()}

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

    def select_from_buffer(
        self,
        buffer: xr.Dataset,
        idx: NDArray[np.integer],
    ):
        """Despite this unvectorized implementation, it is in fact faster than both a
        numba implementation (perhaps due to small batch sizes) and using fancy indexing
        (with or without xarray.Dataset.isel). This may be faster without vectorization
        because it avoids fancy indexing in the length dimension.
        """
        out: Dict[Hashable, Tuple[List[str], NDArray]] = {}
        for k, a in buffer.data_vars.items():
            dims = ["batch", *self.non_batch_dims, "length"]
            out[k] = (
                dims,
                np.empty_like(
                    a.values,
                    shape=(len(idx), *self.non_batch_dim_shape[k], self.fixed_length),
                ),
            )
            self.select_from_buffer_array(
                a.values, idx, self.buffer_idx_cols[k], out[k][1]
            )
        return xr.Dataset(out)

    def select_from_buffer_array(
        self,
        arr: NDArray,
        idx: NDArray[np.integer],
        idx_cols: NDArray[np.integer],
        out: NDArray,
    ):
        # buffer_idx columns: starts, region_idx, dim1_idx, dim2_idx, ...
        for i in range(len(idx)):
            _idx: NDArray[np.integer] = idx[i]
            indexer = [slice(None)] * arr.ndim
            indexer[: len(idx_cols)] = _idx[idx_cols]
            indexer[-1] = slice(_idx[0], _idx[0] + self.fixed_length)
            out[i] = arr[tuple(indexer)]

    def torch_dataloader(self):
        if not TORCH_AVAILABLE:
            raise ImportError("Pytorch must be installed to get a Pytorch DataLoader.")
        from torch.utils.data import DataLoader

        dataset = GVLDataset(self)
        dataloader = DataLoader(dataset, batch_size=None)
        return dataloader


@nb.njit(nogil=True, cache=True)
def partition_regions(
    starts: NDArray[np.int64], ends: NDArray[np.int64], max_length: int
):
    partitions = np.zeros_like(starts)
    partition = 0
    curr_length = ends[0] - starts[0]
    for i in range(1, len(partitions)):
        curr_length += ends[i] - ends[i - 1]
        if curr_length > max_length:
            partition += 1
            curr_length = ends[i] - starts[i]
        partitions[i] = partition
    return partitions


class Buffers:
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
            dim_slices = {str(d): slice(0, 0) for d in self.gvl.buffer_sizes}

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

            while instances_in_partition_tasks < instances_in_partition:
                if len(self.ready_actor_idxs) == 0:
                    yield buffer_futures, buffer_meta
                    buffer_futures = []

                dim_slices = self.gvl.increment_dim_slices(dim_slices)
                read_kwargs = {d: self.gvl.indexes[d][s] for d, s in dim_slices.items()}
                buffer_len = n_regions * np.prod(
                    [len(a) for a in read_kwargs.values()], dtype=int
                )
                instances_in_partition_tasks += buffer_len
                buffer_idx = self.gvl.get_buffer_idx(partition, start, read_kwargs)
                if self.gvl.weights is not None:
                    buffer_idx = self.gvl.resample_buffer_idx(buffer_idx)
                if self.gvl.shuffle:
                    buffer_idx = self.gvl.rng.permutation(buffer_idx)
                ready_actor_idx = self.ready_actor_idxs.pop()
                buffer = self.gvl.actors[ready_actor_idx].read.remote(  # type: ignore
                    contig, start, end, **read_kwargs
                )
                buffer_meta[ready_actor_idx] = BufferMeta(
                    buffer_idx, dim_slices, ready_actor_idx
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
                Callable[[Dict[str, NDArray]], Dict[str, NDArray]]
            ] = None,
            shuffle: bool = False,
            weights: Optional[Dict[str, NDArray]] = None,
            seed: Optional[int] = None,
            return_tuples: bool = False,
            return_index: bool = False,
            drop_last: bool = False,
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
            )