from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import polars as pl
import seqpro as sp
from genoray._contigs import ContigNormalizer
from numpy.typing import NDArray
from seqpro.rag import Ragged

from .._torch import requires_torch
from .._variants._utils import path_is_pgen, path_is_vcf
from ._utils import bed_to_regions

if TYPE_CHECKING:
    import torch.utils.data as td


@dataclass(frozen=True, slots=True)
class StreamingDataset:
    """Write-free, iterable-only dataset. Region-major iteration; no random access.

    Two ways to construct:

    - Public API: ``StreamingDataset(regions, reference=..., variants=<path>)``.
      ``variants`` is classified by path suffix, mirroring :func:`gvl.write`'s
      classification (``_write.py``): a ``.svar`` directory (a genoray
      ``SparseVar``/SVAR1 store) is supported in this plan; VCF, PGEN, and
      ``.svar2`` (SVAR2) inputs raise :class:`NotImplementedError` (later
      plans). Only ``jitter=0`` (the default) is supported in this plan.
    - Internal/test-oriented: ``StreamingDataset(regions, contigs=..., n_samples=...,
      ploidy=..., _reconstruct_window=...)`` injects a reconstruction callback
      directly, bypassing variant-source classification. Used by
      ``test_streaming_scheduler.py`` and ``test_svar1_window.py``.
    """

    _bed: pl.DataFrame
    # (n_regions, 4) sorted: (contig_idx, start, end, strand). Only cols 0-2 are
    # read here; `bed_to_regions` returns the 4th (strand) column.
    _regions: NDArray[np.int32]
    _sort_order: NDArray[np.intp]  # maps sorted position -> original bed row
    contigs: list[str]
    n_samples: int
    ploidy: int
    _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
    _batch_size: int = 1
    # Regions per read window. Window >> batch is the whole point: one Rust call per
    # window amortizes the search + page faults across many batches. 64 is a
    # placeholder default -- Task 4 measures and replaces it.
    _window_regions: int = 64

    def __init__(
        self,
        regions,
        reference: str | Path | None = None,
        variants: str | Path | None = None,
        *,
        jitter: int = 0,
        contigs: list[str] | None = None,
        n_samples: int | None = None,
        ploidy: int | None = None,
        _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
        | None = None,
    ):
        if _reconstruct_window is not None:
            # Internal/test-oriented path: caller injects the reconstruction
            # callback directly and must supply everything it would otherwise
            # be derived from.
            if contigs is None or n_samples is None or ploidy is None:
                raise ValueError(
                    "StreamingDataset(_reconstruct_window=...) requires "
                    "`contigs`, `n_samples`, and `ploidy` to be supplied "
                    "explicitly."
                )
        elif variants is not None:
            # Public API path: classify `variants` and build the backend.
            if reference is None:
                raise ValueError(
                    "StreamingDataset(...) requires `reference` to reconstruct "
                    "haplotypes."
                )
            if jitter != 0:
                raise NotImplementedError(
                    "StreamingDataset read-time jitter is not implemented yet; "
                    "only jitter=0 (the default) is supported in this plan."
                )

            p = Path(variants)
            if p.is_dir() and p.suffix == ".svar":
                from genoray import SparseVar

                contigs = SparseVar(str(p)).contigs
                backend = _Svar1Backend(p, reference, contigs, regions)
                n_samples = backend.n_samples
                ploidy = backend.ploidy
                _reconstruct_window = backend.reconstruct_window
            elif p.is_dir() and p.suffix == ".svar2":
                raise NotImplementedError(
                    f"StreamingDataset does not support SVAR2 stores yet ({p}); "
                    "this is a later plan. Use a SparseVar (.svar) store for now."
                )
            elif path_is_pgen(p):
                raise NotImplementedError(
                    f"StreamingDataset does not support PGEN input yet ({p}); "
                    "this is a later plan. Use a SparseVar (.svar) store for now."
                )
            elif path_is_vcf(p):
                raise NotImplementedError(
                    f"StreamingDataset does not support VCF input yet ({p}); "
                    "this is a later plan. Use a SparseVar (.svar) store for now."
                )
            else:
                raise ValueError(
                    f"variants={p} has an unrecognized file type; expected a "
                    "VCF, PGEN, or SparseVar (.svar) store."
                )
        else:
            raise ValueError(
                "StreamingDataset(...) requires either `variants` (a path to a "
                "VCF, PGEN, or SparseVar/.svar store, public API) or "
                "`_reconstruct_window` (injected-callback, internal/test API)."
            )

        bed = regions if isinstance(regions, pl.DataFrame) else sp.bed.read(regions)
        # record original-row order so emitted indices refer to the user's input order.
        # Positional (row-index carried through the sort), not value-based: a join on
        # BED columns would fan out on duplicate rows and corrupt `_sort_order`.
        sorted_bed = sp.bed.sort(bed.with_row_index("_r"))
        order = sorted_bed["_r"].to_numpy().astype(np.intp)
        regs = bed_to_regions(sorted_bed.drop("_r"), ContigNormalizer(contigs))
        object.__setattr__(self, "_bed", bed)
        object.__setattr__(self, "_regions", regs)
        object.__setattr__(self, "_sort_order", order)
        object.__setattr__(self, "contigs", list(contigs))
        object.__setattr__(self, "n_samples", int(n_samples))
        object.__setattr__(self, "ploidy", int(ploidy))
        object.__setattr__(self, "_reconstruct_window", _reconstruct_window)
        object.__setattr__(self, "_batch_size", 1)
        object.__setattr__(self, "_window_regions", 64)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._regions), self.n_samples)

    def __len__(self) -> int:
        return len(self._regions) * self.n_samples

    def _with_batch_size(self, batch_size: int) -> "StreamingDataset":
        # dataclasses.replace() would re-invoke __init__ (which doesn't accept
        # every field as a kwarg), so shallow-copy and mutate the frozen instance.
        new = copy.copy(self)
        object.__setattr__(new, "_batch_size", int(batch_size))
        return new

    def _plan(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield one WINDOW per step: (region_idxs, sample_idxs), cartesian.

        Region-major, single-contig per window (`self._regions` is sorted by
        (contig_idx, start), so each contig's regions are one contiguous run). Every
        sample is read per window -- variant stores return all samples per range read
        essentially for free, so the effective item order is region-major,
        sample-inner. NOT pairwise: `StreamingDataset` has no `__getitem__`, so the
        traversal is a fixed cartesian sweep and the window is the read granularity.
        """
        n_regions, n_samples = self.shape
        if n_regions == 0:
            return
        all_samples = np.arange(n_samples, dtype=np.intp)
        contig_idxs = self._regions[:, 0]
        run_bounds = np.flatnonzero(np.diff(contig_idxs)) + 1
        run_starts = np.concatenate(([0], run_bounds))
        run_ends = np.concatenate((run_bounds, [n_regions]))
        for r_lo, r_hi in zip(run_starts, run_ends):
            for w_lo in range(int(r_lo), int(r_hi), self._window_regions):
                w_hi = min(w_lo + self._window_regions, int(r_hi))
                yield np.arange(w_lo, w_hi, dtype=np.intp), all_samples

    def _iter_batches(self, batch_size: int) -> Iterator[tuple]:
        """Drive the plan and slice each reconstructed window into batches.

        The window is the READ granularity; a batch is a slice of it. Batches may span
        window boundaries only in the sense that a window's trailing partial batch is
        emitted as-is -- windows never split a (region, sample) cell.
        """
        for r_idx, s_idx in self._plan():
            data = self._reconstruct_window(r_idx, s_idx)
            # Window rows are C-order (region, sample): row bi = ri*n_samples + si.
            n_s = len(s_idx)
            flat_r = np.repeat(self._sort_order[r_idx], n_s)
            flat_s = np.tile(s_idx, len(r_idx))
            n_rows = len(flat_r)
            for lo in range(0, n_rows, batch_size):
                hi = min(lo + batch_size, n_rows)
                yield data[lo:hi], flat_r[lo:hi], flat_s[lo:hi]

    def __iter__(self) -> Iterator[tuple]:
        yield from self._iter_batches(self._batch_size)

    def with_seqs(self, kind: Literal["haplotypes"]) -> "StreamingDataset":
        """Select the sequence output kind. Only ``"haplotypes"`` is supported
        in this plan; reference, annotated, and variants output are later
        plans."""
        if kind != "haplotypes":
            raise NotImplementedError(
                f"StreamingDataset.with_seqs({kind!r}) is not implemented yet; "
                'only "haplotypes" is supported in this plan. Reference, '
                "annotated, and variants output are later plans."
            )
        return copy.copy(self)

    def __getitem__(self, idx) -> None:
        raise TypeError(
            "StreamingDataset is iterable-only; use to_dataloader() instead of "
            "map-style indexing."
        )

    def to_torch_dataset(self, *args, **kwargs) -> None:
        raise TypeError(
            "StreamingDataset is iterable-only; use to_dataloader() instead of "
            "to_torch_dataset() (there is no random-access torch Dataset for a "
            "streaming source)."
        )

    @requires_torch
    def to_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        return_indices: bool = True,
        *,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Callable | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> "td.DataLoader":
        """Wrap this ``StreamingDataset`` in a PyTorch
        :class:`DataLoader <torch.utils.data.DataLoader>`. Requires PyTorch to
        be installed.

        Parameters
        ----------
        batch_size
            Number of ``(region, sample)`` cells per yielded batch.
        num_workers
            Must be 0. ``StreamingDataset`` iteration is itself the
            concurrency strategy (mirrors :meth:`Dataset.to_dataloader`'s
            ``mode="buffered"``/``"double_buffered"`` restriction); worker-process
            sharding of the plan is a later plan.
        return_indices
            If ``True`` (the default), yield ``(data, region_idxs, sample_idxs)``
            tuples; if ``False``, yield ``data`` alone.
        """
        if num_workers > 0:
            raise ValueError(
                "StreamingDataset.to_dataloader: num_workers>0 is not "
                "implemented yet; the loader IS the concurrency strategy for "
                "StreamingDataset (mirrors gvl.Dataset.to_dataloader's "
                "buffered/double_buffered modes, which impose the same "
                "restriction). Use num_workers=0."
            )

        import torch.utils.data as td

        inner = _make_streaming_torch_dataset(self, batch_size, return_indices)
        return td.DataLoader(
            inner,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )


def _make_streaming_torch_dataset(
    dataset: StreamingDataset, batch_size: int, return_indices: bool
) -> "td.IterableDataset":
    """IterableDataset wrapper for `StreamingDataset`, mirroring the
    `make_buffered_dataset` pattern in `_buffered_loader.py`: `__iter__`
    drives `dataset`'s own iteration (batched), and `return_indices` toggles
    whether index arrays ride along -- kept out of `StreamingDataset.__iter__`
    itself so `list(sds)` (used by Tasks 2/4's tests) keeps yielding 3-tuples
    regardless of how a DataLoader wraps it.
    """
    import torch.utils.data as td

    batched = dataset._with_batch_size(batch_size)

    class _StreamingTorchDataset(td.IterableDataset):
        def __iter__(self):
            for data, r_idx, s_idx in batched:
                if return_indices:
                    yield data, r_idx, s_idx
                else:
                    yield data

        def __len__(self) -> int:
            # NOT `ceil(len(dataset) / batch_size)`: `_plan` yields one WINDOW per
            # step (region_idxs, all samples), and `_iter_batches` slices each
            # window's cells into batches independently, so every window's last
            # batch may be partial. The true count is
            # `sum(ceil(window_cells / batch_size))`. Compute this from `_plan()`
            # directly (it only materializes small index arrays -- no
            # reconstruction) rather than draining `_iter_batches`, so `len(dl)` --
            # which DataLoader forwards here -- matches the batches actually
            # yielded without redoing every read just to count them.
            total = 0
            for r_idx, s_idx in batched._plan():
                n_rows = len(r_idx) * len(s_idx)
                total += -(-n_rows // batch_size)  # ceil division
            return total

    return _StreamingTorchDataset()


class _Svar1Backend:
    """Streaming SVAR1 read backend: reconstructs haplotypes for a batch of
    ``(r_idx, s_idx)`` directly from a live ``.svar`` store, with no on-disk
    gvl dataset. Wraps `Svar1Store`/`reconstruct_haplotypes_svar1` (Rust) --
    a `.reconstruct_window` bound method is meant to be passed as
    `StreamingDataset(_reconstruct_window=...)`.

    The static variant table (positions/ILEN/ALT alleles, GLOBAL across
    contigs) is read once at construction from ``SparseVar(path).index``; only
    per-region live genotype reads hit the store during iteration.
    """

    def __init__(
        self,
        svar_path: str | Path,
        reference_path: str | Path,
        contigs: list[str],
        bed: pl.DataFrame | str | Path,
    ) -> None:
        from genoray import SparseVar

        from ..genvarloader import Svar1Store
        from ._haps import _canonicalize_variant_table, _variant_arrays_from_table
        from ._write import _reject_unsupported_variants
        from ._reference import Reference

        self._contigs = list(contigs)

        sv = SparseVar(str(svar_path))
        self.n_samples = len(sv.available_samples)
        self.ploidy = sv.ploidy

        self._ref = Reference.from_path(reference_path, self._contigs)

        # Same region-bounds derivation StreamingDataset itself uses (D1): a
        # batch's `r_idx` indexes into this same sorted regions table, so the
        # two stay aligned when both are built from the same `bed`.
        bed_df = bed if isinstance(bed, pl.DataFrame) else sp.bed.read(bed)
        self._regions = bed_to_regions(
            sp.bed.sort(bed_df), ContigNormalizer(self._contigs)
        )

        idx = sv.index.sort("index")
        # Same "valid inputs only" contract `gvl.write` enforces (validated, not
        # fixed up). This is load-bearing here, not just parity-cosmetic: `ilens`
        # (used both to derive `v_ends` for the range search and by the
        # reconstruct kernel itself) and `alt` are only meaningful for
        # bi-allelic, non-symbolic records -- a `<DEL>` ALT would corrupt both the
        # window's overlap bound and the reconstructed sequence. Must run BEFORE
        # canonicalization, which collapses the list-typed ALT this check inspects.
        _reject_unsupported_variants(idx, "SparseVar (.svar)")
        idx = _canonicalize_variant_table(idx)
        v_starts, ilens, ref, alt = _variant_arrays_from_table(idx, one_based=True)
        if ref is None:
            raise ValueError(f"SVAR1 store at {svar_path} has no REF allele column.")
        self._v_starts = np.ascontiguousarray(v_starts, np.int32)
        self._ilens = np.ascontiguousarray(ilens, np.int32)
        self._alt_alleles = np.ascontiguousarray(alt.data.view(np.uint8), np.uint8)
        self._alt_offsets = np.ascontiguousarray(alt.offsets, np.int64)

        self._store = Svar1Store(str(svar_path), self.n_samples, self.ploidy)

        # Per contig: register three scalars and cache the contig-local u32 arrays the
        # range search borrows. The arrays stay HERE (numpy) and cross per call as
        # zero-copy PyReadonlyArray1 -- nothing variant-scale is duplicated into Rust.
        # (The old skeleton pushed the whole POS/REF/ALT table across as Python lists
        # via .tolist() -- ~10M int objects for a human chr1 -- purely to feed
        # Svar1RecordSource's constructor. No record source, no table.)
        chrom = idx["CHROM"].cast(pl.Utf8).to_numpy()
        # v_end = POS_1based - min(ILEN, 0); genoray's `_var_end_expr()` convention
        # (genoray/_var_ranges.py) -- and what `_write.py`'s `v_ends` uses too, via
        # the SAME raw `idx["POS"]` column (1-based; `_canonicalize_variant_table`
        # never touches POS, so pre/post-canonicalization values are identical).
        # MUST be the raw 1-based POS, NOT `v_starts` (already `-1`'d to 0-based by
        # `_variant_arrays_from_table(one_based=True)`) -- subtracting the already-
        # decremented start silently produces a 0-length exclusive end for every SNP
        # (`v_end == v_start` instead of `v_start + 1`), which drops the variant
        # whenever a query's lower bound lands exactly on it. NOT the kernel's
        # `v_start - min(ilen,0) + 1` either -- that `+1` lives inside
        # get_diffs_sparse and is a different convention.
        v_ends_all = (idx["POS"].to_numpy() - np.minimum(ilens, 0)).astype(np.uint32)
        self._contig_arrays: dict[
            str, tuple[NDArray[np.uint32], NDArray[np.uint32]]
        ] = {}

        for c in self._contigs:
            mask = chrom == c
            n_local = int(mask.sum())
            if n_local == 0:
                self._store.set_contig_meta(c, 0, 0, 0)
                self._contig_arrays[c] = (
                    np.empty(0, np.uint32),
                    np.empty(0, np.uint32),
                )
                continue

            first = int(np.argmax(mask))
            # The per-contig slices below assume this contig's rows are one CONTIGUOUS
            # block starting at `first`. True for a SparseVar built from a
            # position-sorted VCF; if violated the failure mode is a silently WRONG
            # per-contig table -- parity breaks with no error. Fail fast instead.
            if not mask[first : first + n_local].all():
                raise ValueError(
                    f"SVAR index rows for contig {c!r} are not contiguous; "
                    "the streaming SVAR1 backend requires a position-sorted store."
                )

            vs_c = np.ascontiguousarray(v_starts[first : first + n_local], np.uint32)
            ve_c = np.ascontiguousarray(v_ends_all[first : first + n_local], np.uint32)
            # genoray's `var_ranges` binary-searches a `SearchTree` built over `vs_c`
            # and documents its input as ascending -- but enforces nothing beyond a
            # length `debug_assert`. A non-ascending POS within this contig (e.g. a
            # VCF sorted by contig but not by position) passes the contiguity check
            # above and then yields silently WRONG variant ranges with no error --
            # truncated haplotypes, no exception. Fail fast instead, same as above.
            if n_local > 1 and not (np.diff(vs_c.astype(np.int64)) >= 0).all():
                raise ValueError(
                    f"SVAR index POS for contig {c!r} is not ascending; "
                    "the streaming SVAR1 backend requires a position-sorted store."
                )
            # Python's var_ranges convention: max(v_ends - v_starts). Exactly 1 larger
            # than search::overlap_range's `>=` bound -- an OVER-estimate, which only
            # widens the candidate window and is provably overshoot-safe. Do not
            # subtract 1; UNDER-estimating would be a correctness bug.
            max_v_len = int((ve_c.astype(np.int64) - vs_c.astype(np.int64)).max())
            contig_start = int(idx["index"][first])

            self._store.set_contig_meta(c, contig_start, n_local, max_v_len)
            self._contig_arrays[c] = (vs_c, ve_c)

    def reconstruct_window(
        self, r_idx: NDArray[np.intp], s_idx: NDArray[np.intp]
    ) -> Ragged:
        """Reconstruct one CARTESIAN window: every region in `r_idx` x every sample in
        `s_idx`, single-contig. Returns a `Ragged[np.bytes_]` (S1) of shape
        `(len(r_idx) * len(s_idx), ploidy, ~length)`, C-order (region, sample).
        """
        from ..genvarloader import reconstruct_haplotypes_svar1

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)

        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError(
                "_Svar1Backend.reconstruct_window: window spans multiple contigs; "
                "every Rust call must be single-contig (the scheduler groups by contig)."
            )
        contig_name = self._contigs[contig_idx]
        vs_c, ve_c = self._contig_arrays[contig_name]

        region_bounds = np.ascontiguousarray(self._regions[r_idx, 1:3], np.int32)

        ref_c_idx = self._ref.c_map.contigs.index(contig_name)
        c_s = int(self._ref.offsets[ref_c_idx])
        c_e = int(self._ref.offsets[ref_c_idx + 1])
        ref_bytes = np.ascontiguousarray(self._ref.reference[c_s:c_e], np.uint8)
        ref_offsets = np.array([0, c_e - c_s], dtype=np.int64)

        data, offsets = reconstruct_haplotypes_svar1(
            self._store,
            contig_name,
            vs_c,
            ve_c,
            region_bounds,
            np.ascontiguousarray(s_idx, np.int64),
            self._v_starts,
            self._ilens,
            self._alt_alleles,
            self._alt_offsets,
            ref_bytes,
            ref_offsets,
            self._ref.pad_char,
            True,
        )
        batch = len(r_idx) * len(s_idx)
        return Ragged.from_offsets(
            data.view("S1"), (batch, self.ploidy, None), np.asarray(offsets, np.int64)
        )
