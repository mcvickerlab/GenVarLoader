from __future__ import annotations

from pathlib import Path
from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)

import numba as nb
import numpy as np
import polars as pl
from attrs import define, evolve
from loguru import logger
from numpy.typing import NDArray
from seqpro._ragged import Ragged

from .._fasta import Fasta
from .._ragged import RaggedSeqs, to_padded
from .._torch import (
    TORCH_AVAILABLE,
    get_dataloader,
    no_torch_error,
    tensor_from_maybe_bytes,
)
from .._types import Idx
from .._utils import _lengths_to_offsets, _normalize_contig_name
from ._utils import bed_to_regions, padded_slice


@define
class Reference:
    """A reference genome kept in-memory. Typically this is only instantiated to be
    passed to :meth:`Dataset.open <genvarloader.Dataset.open>` and avoid data duplication.

    .. note::
        Do not instantiate this class directly. Use :meth:`Reference.from_path` instead.
    """

    reference: NDArray[np.uint8]
    contigs: List[str]
    offsets: NDArray[np.uint64]
    pad_char: int

    @classmethod
    def from_path(
        cls,
        fasta: Union[str, Path],
        contigs: List[str] | None = None,
        in_memory: bool = True,
    ):
        """Load a reference genome from a FASTA file.

        Parameters
        ----------
        fasta
            Path to the FASTA file.
        contigs
            List of contig names to load. If None, all contigs in the FASTA file are loaded.
            Can be either UCSC or Ensembl style (i.e. with or without the "chr" prefix) and
            will be handled appropriately to match the underlying FASTA.
        in_memory
            Whether to load the reference genome into memory. If True, the reference genome
            is loaded into memory. If False, the reference genome is read on-demand from a
            memory mapped array. This will still be much faster than reading from FASTA but
            slower than keeping it in memory. This is useful if you need to work with many
            reference genomes or have very limited RAM.
        """
        _fasta = Fasta("ref", fasta, "N")

        if not _fasta._valid_cache():
            logger.info("Memory-mapping FASTA file for faster access.")
            _fasta._write_to_cache()

        ref_mmap = np.memmap(_fasta.cache_path, np.uint8, "r")
        offsets = _lengths_to_offsets(
            np.array(list(_fasta.contigs.values())), np.uint64
        )
        pad_char = ord("N")

        if contigs is None or not in_memory:
            contigs = list(_fasta.contigs)

        if in_memory:
            _contigs = [_normalize_contig_name(c, _fasta.contigs) for c in contigs]
            if unmapped := [
                source for source, mapped in zip(contigs, _contigs) if mapped is None
            ]:
                raise ValueError(
                    f"Some of the given contig names are not present in reference file: {unmapped}"
                )
            contigs = cast(list[str], _contigs)

            reference = np.empty(sum(_fasta.contigs[c] for c in contigs), np.uint8)
            offset = 0
            for c in contigs:
                c_idx = list(_fasta.contigs).index(c)
                o_s, o_e = offsets[c_idx], offsets[c_idx + 1]
                reference[offset : offset + o_e - o_s] = ref_mmap[o_s:o_e]
                offset += o_e - o_s
            offsets = _lengths_to_offsets(
                np.array([_fasta.contigs[c] for c in contigs]), np.uint64
            )
        else:
            reference = ref_mmap

        return cls(reference, contigs, offsets, pad_char)

    def fetch(
        self, contig: str, start: int = 0, end: int | None = None
    ) -> NDArray[np.bytes_]:
        c = _normalize_contig_name(contig, self.contigs)
        if c is None:
            raise ValueError(f"Contig {contig} not found in reference.")
        c_idx = self.contigs.index(c)
        o_s, o_e = self.offsets[c_idx], self.offsets[c_idx + 1]

        if end is None:
            end = cast(int, self.offsets[c_idx + 1] - self.offsets[c_idx])

        _contig = self.reference[o_s:o_e]
        if start < 0 or end > len(_contig):
            seq = padded_slice(_contig, start, end, self.pad_char)
        else:
            seq = _contig[start:end]

        return seq.view("S1")


T = TypeVar("T", NDArray[np.bytes_], RaggedSeqs)


@define
class RefDataset(Generic[T]):
    """A reference dataset for pulling out sequences from a reference genome. Also a memory saver
    when the total length of the regions of interest are greater than the reference genome."""

    reference: Reference
    regions: pl.DataFrame
    max_jitter: int
    jitter: int
    output_length: Literal["ragged", "variable"] | int
    deterministic: bool
    _full_regions: NDArray[np.int32]
    _rng: np.random.Generator

    def __init__(
        self: RefDataset[RaggedSeqs],
        reference: Reference,
        regions: pl.DataFrame,
        jitter: int = 0,
        deterministic: bool = True,
        output_length: Literal["ragged", "variable"] | int = "ragged",
        seed: int | np.random.Generator | None = None,
    ):
        if regions.height == 0:
            raise ValueError("Table of regions has a height of zero.")

        if jitter < 0:
            raise ValueError(f"jitter ({jitter}) must be a non-negative integer.")
        elif jitter > (
            min_len := regions.select(
                (pl.col("chromEnd") - pl.col("chromStart")).min()
            ).item()
        ):
            raise ValueError(
                f"jitter ({jitter}) must be less than the minimum region length ({min_len})."
            )

        self.jitter = jitter
        self.deterministic = deterministic
        self.reference = reference
        self.regions = regions
        self.output_length = output_length
        self._full_regions = bed_to_regions(regions, reference.contigs)
        self._rng = np.random.default_rng(seed)

    @property
    def shape(self) -> tuple[int]:
        """Shape of the dataset."""
        return (self.regions.height,)

    def __len__(self) -> int:
        """Length of the dataset."""
        return self.regions.height

    @overload
    def with_len(self, output_length: Literal["ragged"]) -> RefDataset[RaggedSeqs]: ...
    @overload
    def with_len(
        self, output_length: Literal["variable"] | int
    ) -> RefDataset[NDArray[np.bytes_]]: ...
    def with_len(
        self, output_length: Literal["ragged", "variable"] | int
    ) -> RefDataset:
        if isinstance(output_length, int):
            if output_length < 1:
                raise ValueError(
                    f"Output length ({output_length}) must be a positive integer."
                )
            min_r_len: int = (self._full_regions[:, 2] - self._full_regions[:, 1]).min()
            max_output_length = min_r_len + 2 * self.max_jitter
            eff_length = output_length + 2 * self.jitter

            if eff_length > max_output_length:
                raise ValueError(
                    f"Jitter-expanded output length (out_len={self.output_length}) + 2 * ({self.jitter=}) = {eff_length} must be less"
                    f" than or equal to the maximum output length of the dataset ({max_output_length})."
                    f" The maximum output length is the minimum region length ({min_r_len}) + 2 * (max_jitter={self.max_jitter})."
                )

        return evolve(self, output_length=output_length)

    def __getitem__(self, idx: Idx) -> T:
        # (... 4)
        regions = self._full_regions[idx].copy()

        out_reshape = None
        squeeze = False
        if regions.ndim > 2:
            out_reshape = regions.shape[:-1]
        elif regions.ndim == 1:
            squeeze = True

        regions = regions.reshape(-1, 4)

        batch_size = len(regions)

        lengths = regions[:, 2] - regions[:, 1]

        if isinstance(self.output_length, int):
            # (b)
            out_lengths = np.full(batch_size, self.output_length, dtype=np.int32)
        else:
            out_lengths = lengths

        # (b)
        if self.deterministic:
            missing_len_shift = np.full(batch_size, 0)
        else:
            missing_len_shift = (lengths - out_lengths).clip(min=0)

        max_shift = missing_len_shift + 2 * self.jitter
        shifts = self._rng.integers(0, max_shift + 1, dtype=np.int32)
        regions[:, 1] += shifts - self.jitter
        regions[:, 2] = regions[:, 1] + out_lengths

        # (b+1)
        out_offsets = _lengths_to_offsets(out_lengths)

        # ragged (b ~l)
        ref = get_reference(
            regions=regions,
            out_offsets=out_offsets,
            reference=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
        ).view("S1")

        ref = cast(Ragged[np.bytes_], Ragged.from_offsets(ref, batch_size, out_offsets))

        if squeeze:
            ref = ref.squeeze()
        if out_reshape is not None:
            ref = ref.reshape(out_reshape)

        if self.output_length == "ragged":
            out = ref
        elif self.output_length == "variable":
            out = to_padded(ref, pad_value=self.reference.pad_char)
        else:
            out = ref.to_numpy()

        return cast(T, out)

    def to_torch_dataset(self) -> TorchDataset:
        """Convert the dataset to a PyTorch dataset.

        Returns
        -------
        TorchDataset
            A PyTorch dataset.
        """
        if self.output_length == "ragged":
            raise ValueError(
                "Cannot convert to PyTorch dataset with ragged output length."
            )
        self = cast(RefDataset[NDArray[np.bytes_]], self)
        return TorchDataset(self)

    def to_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: td.Sampler | Iterable | None = None,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Callable | None = None,
        generator: torch.Generator | None = None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> "td.DataLoader":
        """Convert the dataset to a PyTorch :class:`DataLoader <torch.utils.data.DataLoader>`. The parameters are the same as a
        :class:`DataLoader <torch.utils.data.DataLoader>` with a few omissions e.g. :code:`batch_sampler`.
        Requires PyTorch to be installed.

        Parameters
        ----------
        batch_size
            How many samples per batch to load.
        shuffle
            Set to True to have the data reshuffled at every epoch.
        sampler
            Defines the strategy to draw samples from the dataset. Can be any :py:class:`Iterable <typing.Iterable>` with :code:`__len__` implemented. If specified, shuffle must not be specified.

            .. important::
                Do not provide a :class:`BatchSampler <torch.utils.data.BatchSampler>` here. GVL Datasets use multithreading when indexed with batches of indices to avoid the overhead of multi-processing.
                To leverage this, GVL will automatically wrap the :code:`sampler` with a :class:`BatchSampler <torch.utils.data.BatchSampler>`
                so that lists of indices are given to the GVL Dataset instead of one index at a time. See `this post <https://discuss.pytorch.org/t/dataloader-sample-by-slices-from-dataset/113005>`_
                for more information.
        num_workers
            How many subprocesses to use for dataloading. :code:`0` means that the data will be loaded in the main process.

            .. tip::
                For GenVarLoader, it is generally best to set this to 0 or 1 since almost everything in
                GVL is multithreaded. However, if you are using a transform that is compute intensive and single threaded, there may
                be a benefit to setting this > 1.
        collate_fn
            Merges a list of samples to form a mini-batch of Tensor(s).
        pin_memory
            If :code:`True`, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your :code:`collate_fn` returns a batch that is a custom type, see the example below.
        drop_last
            Set to :code:`True` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If :code:`False` and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        timeout
            If positive, the timeout value for collecting a batch from workers. Should always be non-negative.
        worker_init_fn
            If not :code:`None`, this will be called on each worker subprocess with the worker id (an int in :code:`[0, num_workers - 1]`) as input, after seeding and before data loading.
        multiprocessing_context
            If :code:`None`, the default multiprocessing context of your operating system will be used.
        generator
            If not :code:`None`, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate :code:`base_seed` for workers.
        prefetch_factor
            Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers batches prefetched across all workers. (default value depends on the set value for num_workers. If value of num_workers=0 default is None. Otherwise, if value of num_workers > 0 default is 2).
        persistent_workers
            If :code:`True`, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive.
        pin_memory_device
            The device to :code:`pin_memory` to if :code:`pin_memory` is :code:`True`.
        """
        return get_dataloader(
            dataset=self.to_torch_dataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )


@nb.njit(parallel=True, nogil=True, cache=True)
def get_reference(
    regions: NDArray[np.int32],
    out_offsets: NDArray[np.int64],
    reference: NDArray[np.uint8],
    ref_offsets: NDArray[np.uint64],
    pad_char: int,
) -> NDArray[np.uint8]:
    out = np.empty(out_offsets[-1], np.uint8)
    for i in nb.prange(len(regions)):
        o_s, o_e = out_offsets[i], out_offsets[i + 1]
        c_idx, start, end = regions[i, :3]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        out[o_s:o_e] = padded_slice(reference[c_s:c_e], start, end, pad_char)
    return out


if TORCH_AVAILABLE:
    import torch
    import torch.utils.data as td

    class TorchDataset(td.Dataset):
        def __init__(self, dataset: RefDataset[NDArray[np.bytes_]]):
            self.dataset = dataset

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx: int | list[int]) -> torch.Tensor:
            batch = self.dataset[idx]
            batch = tensor_from_maybe_bytes(batch)
            return batch
else:
    TorchDataset = no_torch_error  # type: ignore
