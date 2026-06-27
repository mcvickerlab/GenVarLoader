from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast, overload

import numba as nb
import numpy as np
import polars as pl
from genoray._utils import ContigNormalizer
from hirola import HashTable
from numpy.typing import ArrayLike, NDArray
from seqpro.rag import Ragged, lengths_to_offsets
from typing_extensions import Self

from .._flat import _Flat
from .._fasta_cache import ensure_cache
from .._ragged import RaggedSeqs, reverse_complement_masked, to_padded
from .._torch import TORCH_AVAILABLE, get_dataloader, no_torch_error
from .._types import Idx, StrIdx
from .._utils import is_dtype
from ._indexing import is_str_arr, s2i
from ._splice import SpliceMap, SplicePlan, build_splice_plan
from ._utils import bed_to_regions, padded_slice
from .._threads import should_parallelize
from ..genvarloader import get_reference as _get_reference_rust_ffi

INT64_MAX = np.iinfo(np.int64).max


@dataclass(slots=True)
class Reference:
    """A reference genome kept in-memory. Typically this is only instantiated to be
    passed to :meth:`Dataset.open <genvarloader.Dataset.open>` and avoid data duplication.

    .. note::
        Do not instantiate this class directly. Use :meth:`Reference.from_path` instead.
    """

    path: Path
    """The path to the reference genome."""
    reference: NDArray[np.uint8]
    """The reference genome as a numpy array, with contigs concatenated."""
    offsets: NDArray[np.int64]
    """The offsets of the contigs in the reference genome. Shape: (n_contigs + 1)"""
    pad_char: int
    """The padding character used in the reference genome."""
    c_map: ContigNormalizer

    @classmethod
    def from_path(
        cls,
        fasta: str | Path,
        contigs: list[str] | None = None,
        in_memory: bool = True,
    ):
        """Load a reference genome from a FASTA file.

        Parameters
        ----------
        fasta
            Path to a ``.fa``/``.fa.bgz`` FASTA file or an existing ``.gvlfa``
            cache directory. When a FASTA path is given, a sibling ``.gvlfa``
            cache is built on first use and reused on subsequent calls; a legacy
            ``.fa.gvl`` flat cache is automatically migrated to the new format.
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
        path = Path(fasta)
        meta, data_path = ensure_cache(fasta)
        full_contigs = meta.contigs

        ref_mmap = np.memmap(data_path, np.uint8, "r")
        offsets = lengths_to_offsets(np.array(list(full_contigs.values())))
        pad_char = ord("N")

        c_map = ContigNormalizer(full_contigs)
        if contigs is None:
            contigs = c_map.contigs
        else:
            _contigs = c_map.norm(contigs)
            if unmapped := [
                source for source, mapped in zip(contigs, _contigs) if mapped is None
            ]:
                raise ValueError(
                    f"Some of the given contig names are not present in reference file: {unmapped}"
                )
            contigs = cast(list[str], _contigs)
            c_map = ContigNormalizer(contigs)

        if in_memory:
            reference = np.empty(sum(full_contigs[c] for c in contigs), np.uint8)
            offset = 0
            for c in contigs:
                c_idx = list(full_contigs).index(c)
                o_s, o_e = offsets[c_idx], offsets[c_idx + 1]
                reference[offset : offset + o_e - o_s] = ref_mmap[o_s:o_e]
                offset += o_e - o_s
            offsets = lengths_to_offsets(np.array([full_contigs[c] for c in contigs]))
        else:
            reference = ref_mmap

        return cls(path, reference, offsets, pad_char, c_map)

    @property
    def contigs(self) -> list[str]:
        return self.c_map.contigs

    def fetch(
        self, contigs: ArrayLike, starts: ArrayLike = 0, ends: ArrayLike = INT64_MAX
    ) -> Ragged[np.bytes_]:
        contigs = np.atleast_1d(contigs)
        starts = np.atleast_1d(starts)
        ends = np.atleast_1d(ends)

        if not is_dtype(contigs, np.integer):
            c_idxs = self.c_map.c_idxs(contigs)
            if (c_idxs == -1).any():
                raise ValueError("Some contigs not found in reference.")
        else:
            c_idxs = contigs

        lengths = ends - starts
        offsets = lengths_to_offsets(lengths)
        regions = np.stack(
            [
                np.asarray(c_idxs, np.int32),
                np.asarray(starts, np.int32),
                np.asarray(ends, np.int32),
            ],
            axis=1,
        )
        seqs = get_reference(
            regions, offsets, self.reference, self.offsets, int(self.pad_char)
        )
        seqs = Ragged.from_offsets(seqs.view("S1"), (len(contigs), None), offsets)
        return seqs


T = TypeVar("T", NDArray[np.bytes_], RaggedSeqs)


@dataclass(slots=True)
class RefDataset(Generic[T]):
    """A reference dataset for pulling out sequences from a reference genome.

    When ``splice_info`` is provided, the dataset returns per-transcript
    concatenated reference sequence, with one row per splice group instead of
    one row per BED region. Same semantics as
    :meth:`Dataset.open(splice_info=...) <genvarloader.Dataset.open>`.
    """

    reference: Reference
    """The reference genome."""
    full_bed: pl.DataFrame
    """A table of regions to extract from the reference genome. The table must have the following columns:
    - `chrom`: The name of the contig (e.g. "chr1", "chr2", etc.)
    - `chromStart`: The start position of the region (0-based).
    - `chromEnd`: The end position of the region (0-based).
    A `strand` column can also be included, in which case the regions will be reverse complemented if the strand is -1
    and the `rc_neg` parameter is set to True.
    """
    _subset_bed: pl.DataFrame = field(init=False)
    _subset_regions: NDArray[np.int32] = field(init=False)
    jitter: int = 0
    """The maximum length for randomly shifting start positions."""
    output_length: Literal["ragged", "variable"] | int = "ragged"
    """The output length of the dataset. Same meaning as :attr:`Dataset.output_length`."""
    deterministic: bool = True
    """If true, fixed length sequences will be right truncated from their full length to the output length.
    If false, fixed length sequences will be randomly shifted to be within the output length.
    """
    rc_neg: bool = True
    """Whether to reverse complement the regions that are on the negative strand."""
    seed: int | np.random.Generator | None = None
    _rng: np.random.Generator = field(init=False)
    """A random number generator."""
    region_names: str | None = None
    """The name of the column in the full_bed table to use as the region names."""
    _region_map: HashTable | None = field(init=False)
    splice_info: str | tuple[str, str] | None = None
    """If set, the dataset is spliced. Either the column name with rows already
    in splice order or a (group_col, sort_col) pair applied against ``full_bed``."""
    _splice_map: SpliceMap | None = field(init=False, default=None)
    _spliced_bed: pl.DataFrame | None = field(init=False, default=None)

    def __post_init__(self):
        if self.full_bed.height == 0:
            raise ValueError("Table of regions has a height of zero.")

        if self.jitter < 0:
            raise ValueError(f"jitter ({self.jitter}) must be a non-negative integer.")
        elif self.jitter > (
            min_len := self.full_bed.select(
                (pl.col("chromEnd") - pl.col("chromStart")).min()
            ).item()
        ):
            raise ValueError(
                f"jitter ({self.jitter}) must be less than the minimum region length ({min_len})."
            )
        self._subset_bed = self.full_bed
        self._subset_regions = bed_to_regions(self.full_bed, self.reference.c_map)
        self._rng = np.random.default_rng(self.seed)
        if self.region_names is not None:
            region_names = self.full_bed[self.region_names].to_numpy().astype(np.str_)
            self._region_map = HashTable(
                max=len(region_names) * 2,  # type: ignore[bad-argument-type]  # hirola HashTable.max typed as numpy.Number but accepts int
                dtype=region_names.dtype,
            )
            self._region_map.add(region_names)
        else:
            self._region_map = None

        if self.splice_info is not None:
            sm, sp_bed = SpliceMap.from_bed(self.splice_info, self.full_bed)
            self._splice_map = sm
            self._spliced_bed = sp_bed
            self._check_valid_state()
        else:
            self._splice_map = None
            self._spliced_bed = None

    def _check_valid_state(self):
        if self._splice_map is None:
            return
        if self.jitter > 0:
            raise RuntimeError(
                "Jitter is not supported with splicing. Please set jitter to 0."
            )
        if not self.deterministic:
            raise RuntimeError(
                "Non-deterministic algorithms are not supported with splicing."
                " Please set deterministic to True."
            )
        if isinstance(self.output_length, int):
            raise RuntimeError(
                "Splicing requires output_length='ragged' or 'variable',"
                " not a fixed integer length."
            )

    @property
    def regions(self) -> pl.DataFrame:
        return self._subset_bed

    @property
    def is_spliced(self) -> bool:
        """Whether the dataset is spliced."""
        return self._splice_map is not None

    @property
    def spliced_regions(self) -> pl.DataFrame:
        """The spliced BED, subset to the current row subset."""
        if self._spliced_bed is None or self._splice_map is None:
            raise ValueError("Dataset does not have splice information.")
        subset = self._splice_map.row_subset_idxs
        if subset is None:
            return self._spliced_bed
        return self._spliced_bed[subset]

    @property
    def shape(self) -> tuple[int]:
        """Shape of the dataset."""
        if self._splice_map is not None:
            return (self._splice_map.n_rows,)
        return (self.regions.height,)

    def __len__(self) -> int:
        """Length of the dataset."""
        if self._splice_map is not None:
            return self._splice_map.n_rows
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
            min_r_len: int = (
                self._subset_regions[:, 2] - self._subset_regions[:, 1]
            ).min()
            max_output_length = min_r_len
            eff_length = output_length + 2 * self.jitter

            if eff_length > max_output_length:
                raise ValueError(
                    f"Jitter-expanded output length (out_len={self.output_length}) + 2 * ({self.jitter=}) = {eff_length} must be less"
                    f" than or equal to the maximum output length of the dataset ({max_output_length})."
                    f" The maximum output length is the minimum region length ({min_r_len})."
                )

        out = replace(self, output_length=output_length)
        out._check_valid_state()
        return out

    def with_settings(
        self,
        jitter: int | None = None,
        deterministic: bool | None = None,
        rc_neg: bool | None = None,
        seed: int | np.random.Generator | None = None,
        splice_info: str | tuple[str, str] | Literal[False] | None = None,
    ) -> Self:
        to_evolve = {}

        if jitter is not None:
            if jitter < 0:
                raise ValueError(f"jitter ({jitter}) must be a non-negative integer.")
            elif (
                jitter
                > (
                    min_len := self._subset_regions[:, 2] - self._subset_regions[:, 1]
                ).min()
            ):
                raise ValueError(
                    f"jitter ({jitter}) must be less than the minimum region length ({min_len})."
                )
            to_evolve["jitter"] = jitter

        if deterministic is not None:
            to_evolve["deterministic"] = deterministic

        if rc_neg is not None:
            to_evolve["rc_neg"] = rc_neg

        if seed is not None:
            to_evolve["seed"] = np.random.default_rng(seed)

        new_sm = None
        new_bed = None
        if splice_info is not None:
            if splice_info is False:
                to_evolve["splice_info"] = None
            else:
                new_sm, new_bed = SpliceMap.from_bed(splice_info, self.full_bed)
                to_evolve["splice_info"] = splice_info

        out = replace(self, **to_evolve)

        if splice_info is not None:
            out._splice_map = new_sm
            out._spliced_bed = new_bed

        out._check_valid_state()
        return out

    def subset_to(self, regions: StrIdx):
        """Subset the dataset to a subset of regions (or transcripts, when spliced)."""
        if self._splice_map is not None:
            new_map = self._splice_map.subset_to(regions)
            flat = new_map.splice_map.to_packed().data
            self._splice_map = new_map
            self._subset_bed = self.full_bed[flat]
            self._subset_regions = bed_to_regions(
                self._subset_bed, self.reference.c_map
            )
            return self

        if self._region_map is not None:
            regions = s2i(regions, self._region_map)
        elif is_str_arr(regions):
            raise ValueError(
                "Cannot subset to regions by name because no region name was set."
            )

        if (
            isinstance(regions, (int, np.integer, slice))
            or is_dtype(regions, np.integer)
            or (isinstance(regions, Sequence) and isinstance(regions[0], int))
        ):
            self._subset_bed = self.full_bed[regions]  # type: ignore[bad-index]  # polars DataFrame.__getitem__ doesn't accept all our union members but runtime branch ensures valid kinds
        else:
            self._subset_bed = self.full_bed.filter(regions)  # type: ignore[bad-argument-type]  # polars filter accepts predicates / bool arrays; our union has equivalent shapes

        self._subset_regions = bed_to_regions(self._subset_bed, self.reference.c_map)
        return self

    def to_full_dataset(self) -> Self:
        """Reset the dataset to the full dataset."""
        if self._splice_map is not None:
            self._splice_map = self._splice_map.to_full()
        self._subset_bed = self.full_bed
        self._subset_regions = bed_to_regions(self._subset_bed, self.reference.c_map)
        return self

    def __getitem__(self, idx: Idx) -> T:
        if self._splice_map is not None:
            return self._getitem_spliced(idx)
        return self._getitem_unspliced(idx)

    def _getitem_spliced(self, idx: Idx) -> T:
        assert self._splice_map is not None
        assert not isinstance(self.output_length, int)

        flat_r_idx, offsets, out_reshape, squeeze = self._splice_map.parse_rows(idx)
        # flat_r_idx values are absolute indices into full_bed (not _subset_regions).
        # polars accepts a 1-D numpy integer array directly — no .tolist() needed.
        regions = bed_to_regions(self.full_bed[flat_r_idx], self.reference.c_map)
        lengths = (regions[:, 2] - regions[:, 1]).astype(np.int32, copy=False)

        n_rows = offsets.shape[0] - 1
        plan = build_splice_plan(
            lengths=lengths,
            splice_row_offsets=offsets,
            n_samples=1,
            n_rows=n_rows,
        )

        # Delegate kernel dispatch to the shared helper (eliminates duplication
        # with Ref.__call__'s splice branch). Returns a per-element _Flat (n_elements, None)
        # already in permuted write order.
        to_rc_perm: "NDArray[np.bool_] | None" = None
        if self.rc_neg:
            to_rc_unperm = regions[:, 3] == -1
            if to_rc_unperm.any():
                to_rc_perm = to_rc_unperm[plan.permutation]

        per_elem = _fetch_spliced_ref(
            regions=regions,
            plan=plan,
            reference=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
            to_rc=to_rc_perm,  # Rust: RC done in kernel; numba: handled below
        )

        if to_rc_perm is not None and os.environ.get("GVL_BACKEND", "rust") == "numba":
            from .._ragged import _COMP

            per_elem = per_elem.reverse_masked(to_rc_perm, comp=_COMP)

        # Rewrap with group_offsets at (n_rows, None) — skip the (n_rows, 1, None)
        # + squeeze(1) trick since RefDataset has no sample axis.
        ref = cast(
            Ragged[np.bytes_],
            _Flat.from_offsets(
                per_elem.data, (n_rows, None), plan.group_offsets
            ).to_ragged(),
        )

        if out_reshape is not None:
            ref = ref.reshape(out_reshape)

        if self.output_length == "ragged":
            out = ref
        elif self.output_length == "variable":
            out = to_padded(ref, pad_value=bytes([self.reference.pad_char]))
        else:
            raise AssertionError(
                "splice + fixed-length output should be blocked earlier"
            )

        if squeeze:
            out = out.squeeze(0)

        return cast(T, out)

    def _getitem_unspliced(self, idx: Idx) -> T:
        # (... 4)
        regions = self._subset_regions[idx].copy()

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
            extra_len = np.full(batch_size, 0)
        else:
            extra_len = (lengths - out_lengths).clip(min=0)

        max_shift = extra_len + 2 * self.jitter
        shifts = self._rng.integers(0, max_shift + 1, dtype=np.int32)
        regions[:, 1] += shifts - self.jitter
        regions[:, 2] = regions[:, 1] + out_lengths

        # (b+1)
        out_offsets = lengths_to_offsets(out_lengths)

        # ragged (b ~l)
        # On the Rust backend, RC is folded into the kernel via to_rc.
        # On the numba backend, get_reference ignores to_rc and the post-RC
        # below preserves the original behaviour.
        _to_rc_arr = regions[:, 3] == -1
        _to_rc: "NDArray[np.bool_] | None" = _to_rc_arr if _to_rc_arr.any() else None
        ref = get_reference(
            regions=regions,
            out_offsets=out_offsets,
            reference=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
            to_rc=_to_rc,
        ).view("S1")

        ref = cast(
            Ragged[np.bytes_], Ragged.from_offsets(ref, (batch_size, None), out_offsets)
        )

        if _to_rc is not None and os.environ.get("GVL_BACKEND", "rust") == "numba":
            ref = reverse_complement_masked(ref, _to_rc)

        if out_reshape is not None:
            ref = ref.reshape(out_reshape)

        if self.output_length == "ragged":
            out = ref
        elif self.output_length == "variable":
            out = to_padded(ref, pad_value=bytes([self.reference.pad_char]))
        else:
            out = ref.to_numpy(validate=False)

        if squeeze:
            out = out.squeeze(0)

        return cast(T, out)

    def to_torch_dataset(
        self, return_indices: bool = False, transform: Callable | None = None
    ) -> TorchDataset:
        """Convert the dataset to a PyTorch dataset.

        Parameters
        ----------
        return_indices
            If True, the dataset will return the indices of the regions in the reference genome.
        transform
            A function to transform the data. Should accept a numpy array of S1 with shape (batch_size, length).
            If return_indices is true, the function should accept a tuple of (sequences, indices).
        """
        if self.output_length == "ragged":
            raise ValueError(
                "Cannot convert to PyTorch dataset with ragged output length."
            )
        self = cast(RefDataset[NDArray[np.bytes_]], self)
        return TorchDataset(self, include_indices=return_indices, transform=transform)

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
        return_indices: bool = False,
        transform: Callable | None = None,
    ) -> td.DataLoader:
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
        return_indices
            If True, the dataset will return the indices of the regions in the reference genome.
        transform
            A function to transform the data. Should accept a numpy array of S1 with shape (batch_size, length).
            If return_indices is true, the function should accept a tuple of (sequences, indices).
        """
        return get_dataloader(
            dataset=self.to_torch_dataset(return_indices, transform),
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


@nb.njit(nogil=True, cache=True, inline="always")
def _get_reference_row(i, regions, out_offsets, reference, ref_offsets, pad_char, out):
    o_s, o_e = out_offsets[i], out_offsets[i + 1]
    c_idx, start, end = regions[i, 0], regions[i, 1], regions[i, 2]
    c_s = ref_offsets[c_idx]
    c_e = ref_offsets[c_idx + 1]
    padded_slice(reference[c_s:c_e], start, end, pad_char, out[o_s:o_e])


@nb.njit(parallel=True, nogil=True, cache=True)
def _get_reference_par(regions, out_offsets, reference, ref_offsets, pad_char, out):
    for i in nb.prange(len(regions)):
        _get_reference_row(
            i, regions, out_offsets, reference, ref_offsets, pad_char, out
        )
    return out


@nb.njit(nogil=True, cache=True)
def _get_reference_ser(regions, out_offsets, reference, ref_offsets, pad_char, out):
    for i in range(len(regions)):
        _get_reference_row(
            i, regions, out_offsets, reference, ref_offsets, pad_char, out
        )
    return out


def _get_reference_numba(
    regions, out_offsets, reference, ref_offsets, pad_char, parallel
):
    out = np.empty(out_offsets[-1], np.uint8)
    kernel = _get_reference_par if parallel else _get_reference_ser
    return kernel(regions, out_offsets, reference, ref_offsets, pad_char, out)


def _get_reference_rust(
    regions, out_offsets, reference, ref_offsets, pad_char, parallel, to_rc=None
):
    return _get_reference_rust_ffi(
        np.ascontiguousarray(regions, np.int32),
        np.ascontiguousarray(out_offsets, np.int64),
        np.ascontiguousarray(reference, np.uint8),
        np.ascontiguousarray(ref_offsets, np.int64),
        int(pad_char),
        bool(parallel),
        to_rc,
    )


def get_reference(
    regions: NDArray[np.integer],
    out_offsets: NDArray[np.integer],
    reference: NDArray[np.integer],
    ref_offsets: NDArray[np.integer],
    pad_char: int,
    to_rc: "NDArray[np.bool_] | None" = None,
) -> NDArray[np.uint8]:
    """Fetch reference-genome bytes for a batch of regions.

    ``to_rc`` is a per-query boolean mask (True = reverse-complement that query).
    The mask is consumed in-kernel by the Rust backend.
    """
    parallel = should_parallelize(int(out_offsets[-1]))
    _to_rc = None if to_rc is None else np.ascontiguousarray(to_rc, np.bool_)
    return _get_reference_rust(
        regions, out_offsets, reference, ref_offsets, pad_char, parallel, _to_rc
    )


def _fetch_spliced_ref(
    regions: NDArray[np.integer],
    plan: SplicePlan,
    reference: NDArray[np.uint8],
    ref_offsets: NDArray[np.int64],
    pad_char: int,
    to_rc: "NDArray[np.bool_] | None" = None,
) -> "_Flat[np.bytes_]":
    """Fetch reference bytes in splice-permuted order, returning a per-element
    flat ragged of shape ``(n_elements, None)``.

    This is the kernel-dispatch core shared by :class:`Ref.__call__`'s splice
    branch and :meth:`RefDataset._getitem_spliced`.

    ``to_rc`` is the permuted per-element boolean mask (True = RC that element).
    On the Rust backend it is passed into the ``get_reference`` kernel directly;
    on numba the caller's post-pass handles it.
    """
    permuted_regions = regions[plan.permutation]
    raw = get_reference(
        regions=permuted_regions,
        out_offsets=plan.permuted_out_offsets,
        reference=reference,
        ref_offsets=ref_offsets,
        pad_char=pad_char,
        to_rc=to_rc,
    )  # uint8 flat buffer
    n_elements = plan.permuted_lengths.shape[0]
    return cast(
        "_Flat[np.bytes_]",
        _Flat.from_offsets(raw, (n_elements, None), plan.permuted_out_offsets).view(
            "S1"
        ),
    )


if TORCH_AVAILABLE:
    import torch
    import torch.utils.data as td

    class TorchDataset(td.Dataset):
        dataset: RefDataset[NDArray[np.bytes_]]
        include_indices: bool
        transform: Callable | None

        def __init__(
            self,
            dataset: RefDataset[NDArray[np.bytes_]],
            include_indices: bool,
            transform: Callable | None,
        ):
            self.dataset = dataset
            self.include_indices = include_indices
            self.transform = transform

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx: list[int]):
            batch = (self.dataset[idx],)

            if self.include_indices:
                _idx = np.atleast_1d(idx)
                batch = (*batch, _idx)

            if self.transform is not None:
                batch = self.transform(*batch)

            if len(batch) == 1:
                batch = batch[0]

            return batch

else:
    TorchDataset = no_torch_error
