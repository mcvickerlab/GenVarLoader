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


def _parse_max_mem(max_mem: str | int) -> int:
    """Bytes from an int or a size string like '512MB' / '1g' / '2GiB'."""
    if isinstance(max_mem, int):
        return int(max_mem)
    s = str(max_mem).strip().lower().replace("ib", "b")
    units = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
    }
    # Multi-char suffixes first: "1tb" must match "tb", not fall through to bare "t"
    # (checking "t" first would strip only the trailing "b" and leave "1t" behind).
    for suffix in ("tb", "gb", "mb", "kb", "t", "g", "m", "k", "b"):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * units[suffix])
    return int(float(s))  # bare number = bytes


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
      ``test_streaming_scheduler.py`` and ``test_svar1_window.py``. ``samples`` may
      be supplied too; when omitted, placeholder names ``"0", "1", ...`` are used
      (this path never touches a real sample list).

    ``sample_idx`` means "index into :attr:`samples`" (lexicographically-sorted
    sample names, matching :func:`gvl.write`'s convention) -- NOT a variant store's
    native column order. See :attr:`samples`.

    Parameters
    ----------
    max_mem
        Approximate byte budget for the read-window's offsets buffer, i.e. the
        ``o_start``/``o_stop`` CSR-index pair read per ``(region, sample, ploid)``
        cell (``window_regions * window_samples * ploidy * 16`` bytes). Accepts an
        ``int`` (bytes) or a size string like ``"512MB"``, ``"1g"``, ``"2GiB"``
        (default ``"512MB"``). This budget only bounds the READ window -- the
        separate GENERATION granularity is ``batch_size`` (a :meth:`to_iter`
        argument), which bounds per-batch haplotype OUTPUT independently. Neither
        term scales with cohort size, so peak memory is bounded by
        ``max_mem`` (offsets) + ``batch_size`` (output), independent of the number
        of samples in the dataset.
    """

    # (n_regions, 4) sorted: (contig_idx, start, end, strand). Only cols 0-2 are
    # read here; `bed_to_regions` returns the 4th (strand) column.
    _regions: NDArray[np.int32]
    _sort_order: NDArray[np.intp]  # maps sorted position -> original bed row
    contigs: list[str]
    n_samples: int
    ploidy: int
    _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
    # Sample names in `sample_idx` order (lexicographically sorted -- see `samples`).
    _samples: list[str]
    # Read-window sizing, DERIVED from `max_mem` in __init__ (not user-set directly).
    # The window (regions x sample-chunk x ploidy) is the READ granularity; its offsets
    # buffer is what `max_mem` bounds. Per-batch generation (Task 3) bounds OUTPUT
    # separately by batch_size, so neither term scales with cohort size.
    #
    # 64 is a pragmatic REGION_TARGET, NOT a measured knee: a sweep (window_regions in
    # {1, 4, 16, 64, 256, 1024}) showed wall-clock improving monotonically with fewer
    # windows and flattening past ~64, with everything beyond that inside this shared
    # node's run-to-run noise. entries_touched was exactly flat across every setting,
    # confirming I/O is windowing-invariant, as designed. See
    # docs/roadmaps/streaming-dataset.md (Plan 2 Task 4) for the full sweep narrative.
    _window_regions: int = 64
    _window_samples: int = 1
    _max_mem_bytes: int = 512 * 1024 * 1024
    # The split read/generate backend (real SVAR1 path). When set, _iter_batches
    # generates per batch (output bounded by batch_size). The injected
    # `_reconstruct_window` remains a whole-window TEST seam used when `_backend` is None.
    _backend: _Svar1Backend | _VcfBackend | None = None
    # INTERNAL/EXPERIMENTAL (issue #283) -- not a public `__init__` kwarg, set only via
    # `object.__setattr__`. Selects which prefetch drive `_iter_batches` uses when
    # `_backend is not None`:
    #   - "engine" (default): the landed producer-thread `Svar1StreamEngine` (Design A).
    #   - "readahead": a single-thread read-ahead-one-window drive (Design C) that reuses
    #     the SAME `_Svar1Backend.read_window`/`generate_batch` calls the engine's
    #     consumer makes, just prefetching the next window's pages inline before
    #     generating the current one (no background thread). Output-identical to
    #     "engine" -- prefetch only warms pages, never changes what is generated.
    # This toggle exists ONLY for the cold-cache A-vs-C measurement
    # (`benchmarking/streaming/cold_cache_overlap.py`); it will be removed once a winner
    # is chosen (see docs/roadmaps/streaming-dataset.md).
    _prefetch_strategy: str = "engine"

    def __init__(
        self,
        regions,
        reference: str | Path | None = None,
        variants: str | Path | None = None,
        *,
        jitter: int = 0,
        max_mem: str | int = "512MB",
        contigs: list[str] | None = None,
        n_samples: int | None = None,
        ploidy: int | None = None,
        samples: list[str] | None = None,
        _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
        | None = None,
    ):
        # Every construction path must define this: the injected-callback (test) path
        # leaves it None; the real `.svar` branch below sets it to the `_Svar1Backend`
        # instance.
        _backend_obj = None
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
            # `samples` is optional here: this path is for scheduling/window-plan
            # tests that don't exercise real sample identity. Placeholder names
            # keep `.samples` well-defined without forcing every such test to
            # supply a real sample list.
            if samples is None:
                samples = [str(i) for i in range(n_samples)]
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
                samples = backend._sample_names
                _reconstruct_window = None
                _backend_obj = backend
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
                backend = _VcfBackend(p, reference, contigs, regions)
                n_samples = backend.n_samples
                ploidy = backend.ploidy
                samples = backend._sample_names
                contigs = backend._contigs
                _reconstruct_window = None
                _backend_obj = backend
                # No `_prefetch_strategy` override needed here: `__init__`
                # unconditionally sets it to "engine" below (the "readahead"
                # value is only ever flipped in post-construction by the
                # cold-cache A-vs-C harness, never chosen per-branch here).
                # VCF supports only "engine" -- `_VcfBackend` deliberately has
                # no `read_window`/`generate_batch` split (SVAR1-only seam).
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
        object.__setattr__(self, "_regions", regs)
        object.__setattr__(self, "_sort_order", order)
        object.__setattr__(self, "contigs", list(contigs))
        object.__setattr__(self, "n_samples", int(n_samples))
        object.__setattr__(self, "ploidy", int(ploidy))
        object.__setattr__(self, "_reconstruct_window", _reconstruct_window)
        object.__setattr__(self, "_samples", list(samples))
        object.__setattr__(self, "_backend", _backend_obj)
        # See the field's comment: internal/experimental, flipped only by the
        # cold-cache A-vs-C harness via `object.__setattr__`.
        object.__setattr__(self, "_prefetch_strategy", "engine")
        # Derive the read-window sizing from `max_mem`, NOT from the field defaults
        # above (those are just fallback literals for `__dataclass_fields__`; `slots=True`
        # means there's no class attribute to fall back on at runtime, and this class
        # defines its own `__init__` so the dataclass-generated one never runs). The
        # offsets buffer is `window_regions * window_samples * ploidy * 16 B`
        # (o_start + o_stop, i64 each); bound it by `max_mem` so peak memory stays
        # independent of cohort size, keeping whole sample sets when they fit and
        # holding regions at the measured amortization knee (REGION_TARGET=64).
        max_mem_bytes = _parse_max_mem(max_mem)
        # The engine (#283) double-buffers windows -- the producer reads the NEXT
        # window's offsets while the consumer generates batches from the CURRENT one --
        # so two windows' offsets can be resident at once; size the budget for that.
        n_slots = 2
        cell_bytes = (
            int(ploidy) * 16
        )  # o_start + o_stop, i64 each, per (region,sample,ploid)
        max_cells = max(1, max_mem_bytes // (cell_bytes * n_slots))
        window_samples = max(1, min(int(n_samples), max_cells))
        region_target = 64  # measured read-amortization knee; see roadmap Plan 2.
        window_regions = max(1, min(region_target, max_cells // window_samples))
        object.__setattr__(self, "_max_mem_bytes", max_mem_bytes)
        object.__setattr__(self, "_window_samples", int(window_samples))
        object.__setattr__(self, "_window_regions", int(window_regions))

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._regions), self.n_samples)

    @property
    def samples(self) -> list[str]:
        """The samples in the dataset, in ``sample_idx`` order.

        Lexicographically sorted (matching :func:`gvl.write`'s convention and
        :attr:`Dataset.samples <genvarloader.Dataset.samples>`) -- **not** the
        variant store's native (e.g. VCF column) order. ``to_iter``'s
        ``sample_idxs`` index into this list: ``samples[i]`` is the sample whose
        data arrives at ``sample_idx == i``.
        """
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._regions) * self.n_samples

    def _plan(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield one WINDOW per step: (region_idxs, sample_chunk), cartesian,
        single-contig. Both the region axis (`_window_regions`) and the sample axis
        (`_window_samples`) are chunked so the offsets buffer stays within `max_mem`
        regardless of cohort size. NOT pairwise: the traversal is a fixed cartesian
        sweep and the window is the read granularity.
        """
        n_regions, n_samples = self.shape
        if n_regions == 0:
            return
        contig_idxs = self._regions[:, 0]
        run_bounds = np.flatnonzero(np.diff(contig_idxs)) + 1
        run_starts = np.concatenate(([0], run_bounds))
        run_ends = np.concatenate((run_bounds, [n_regions]))
        for r_lo, r_hi in zip(run_starts, run_ends):
            for w_lo in range(int(r_lo), int(r_hi), self._window_regions):
                w_hi = min(w_lo + self._window_regions, int(r_hi))
                r_idx = np.arange(w_lo, w_hi, dtype=np.intp)
                for s_lo in range(0, n_samples, self._window_samples):
                    s_hi = min(s_lo + self._window_samples, n_samples)
                    yield r_idx, np.arange(s_lo, s_hi, dtype=np.intp)

    def _iter_batches(self, batch_size: int) -> Iterator[tuple]:
        """Drive the plan; generate each window PER BATCH so output is batch-bounded.

        The window is the READ granularity; a batch is the GENERATION granularity. For
        the real SVAR1 backend this drives a `Svar1StreamEngine` (#283) that overlaps
        producer I/O (reading the next window) with consumer generation (reconstructing
        the current one) -- output is still batch_size-bounded (issue #284). The
        injected `_reconstruct_window` path (tests) reconstructs the whole window and
        slices -- memory-unbounded, but only ever used with tiny fixtures.
        """
        if self._backend is not None:
            if self._prefetch_strategy == "engine":
                # Build a COMPACT, region-scale plan ONCE and drive off THAT (never a
                # second `list(self._plan())`). `_plan` always yields a CONTIGUOUS sample
                # chunk `arange(s_lo, s_hi)`, so each window is captured losslessly by
                # `(contig_idx, r_idx, s_lo, s_hi)` -- r_idx is `window_regions`-scale and
                # (s_lo, s_hi) is two ints. Total residency is O(n_windows x window_regions)
                # (region-scale), NEVER O(n_windows x n_samples): the engine holds the
                # public->physical map ONCE and its producer slices it per window (issue
                # #284 / final-review Finding 1).
                plan_jobs: list[tuple[int, NDArray[np.intp], int, int]] = []
                for r_idx, s_idx in self._plan():
                    contig_idx = int(self._regions[r_idx[0], 0])
                    plan_jobs.append(
                        (contig_idx, r_idx, int(s_idx[0]), int(s_idx[-1]) + 1)
                    )
                # Region bounds (u32) per window for the engine constructor; transient
                # (dropped after `build_engine`), also region-scale.
                engine_jobs = [
                    (
                        contig_idx,
                        np.ascontiguousarray(self._regions[r_idx, 1], np.uint32),
                        np.ascontiguousarray(self._regions[r_idx, 2], np.uint32),
                        s_lo,
                        s_hi,
                    )
                    for (contig_idx, r_idx, s_lo, s_hi) in plan_jobs
                ]
                engine = self._backend.build_engine(engine_jobs, batch_size)
                del engine_jobs
                for _contig_idx, r_idx, s_lo, s_hi in plan_jobs:
                    n_s = s_hi - s_lo
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(np.arange(s_lo, s_hi, dtype=np.intp), len(r_idx))
                    n_rows = len(flat_r)
                    for lo in range(0, n_rows, batch_size):
                        hi = min(lo + batch_size, n_rows)
                        nxt = engine.next_batch()
                        if nxt is None:
                            raise RuntimeError(
                                "Svar1StreamEngine exhausted before the plan did"
                            )
                        data, offsets = nxt
                        yield (
                            Ragged.from_offsets(
                                np.asarray(data).view("S1"),
                                (hi - lo, self._backend.ploidy, None),
                                np.asarray(offsets, np.int64),
                            ),
                            flat_r[lo:hi],
                            flat_s[lo:hi],
                        )
                # `-O`-safe (Minor 3): a bare `assert` is stripped under `python -O`,
                # silently dropping this end-of-plan invariant.
                if engine.next_batch() is not None:
                    raise RuntimeError(
                        "Svar1StreamEngine had extra batches beyond the plan"
                    )
            elif self._prefetch_strategy == "readahead":
                # Design C (issue #283): single-thread read-ahead-one-window drive.
                # Reuses the SAME `_Svar1Backend.read_window`/`generate_batch` calls
                # the engine's consumer makes (Task 3, parity-green) -- prefetch is a
                # pure page-warming no-op on output, so this is byte-identical to the
                # "engine" branch above. See `test_streaming_matches_written_all_cells`'s
                # "readahead" parity variant.
                from ..genvarloader import svar1_prefetch_runs

                # Compact, region-scale plan (same treatment as the engine branch): hold
                # only `(r_idx, s_lo, s_hi)` per window, NOT the per-window `s_idx` arrays.
                # `_plan` yields contiguous `arange(s_lo, s_hi)` chunks, so `s_idx` is
                # reconstructed per window on demand (only ~2 windows live at once under
                # the 1-ahead readahead) -- never O(n_windows x n_samples) resident.
                ra_jobs: list[tuple[NDArray[np.intp], int, int]] = [
                    (r_idx, int(s_idx[0]), int(s_idx[-1]) + 1)
                    for r_idx, s_idx in self._plan()
                ]
                if not ra_jobs:
                    return

                def _read(job: tuple[NDArray[np.intp], int, int]):
                    r_idx, s_lo, s_hi = job
                    return self._backend.read_window(
                        r_idx, np.arange(s_lo, s_hi, dtype=np.intp)
                    )

                # Read window 0's offsets up front; then for each window, prefetch the
                # NEXT window's runs before generating the CURRENT one so kernel
                # readahead of the next window's pages overlaps this window's
                # generation.
                cur = _read(ra_jobs[0])
                for i, (r_idx, s_lo, s_hi) in enumerate(ra_jobs):
                    if i + 1 < len(ra_jobs):
                        nxt = _read(ra_jobs[i + 1])
                        svar1_prefetch_runs(self._backend._store, nxt[0], nxt[1])
                    else:
                        nxt = None
                    s_idx = np.arange(s_lo, s_hi, dtype=np.intp)
                    n_s = len(s_idx)
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(s_idx, len(r_idx))
                    n_rows = len(flat_r)
                    for lo in range(0, n_rows, batch_size):
                        hi = min(lo + batch_size, n_rows)
                        data = self._backend.generate_batch(
                            r_idx, s_idx, cur[0], cur[1], lo, hi
                        )
                        yield data, flat_r[lo:hi], flat_s[lo:hi]
                    cur = nxt
            else:
                raise ValueError(
                    f"StreamingDataset._prefetch_strategy={self._prefetch_strategy!r} "
                    'is not recognized; expected "engine" or "readahead".'
                )
        else:
            for r_idx, s_idx in self._plan():
                n_s = len(s_idx)
                flat_r = np.repeat(self._sort_order[r_idx], n_s)
                flat_s = np.tile(s_idx, len(r_idx))
                n_rows = len(flat_r)
                data = self._reconstruct_window(r_idx, s_idx)
                for lo in range(0, n_rows, batch_size):
                    hi = min(lo + batch_size, n_rows)
                    yield data[lo:hi], flat_r[lo:hi], flat_s[lo:hi]

    def to_iter(
        self, batch_size: int = 1, return_indices: bool = True
    ) -> Iterator[tuple]:
        """Iterate haplotype batches. **This is the one iteration entry point** --
        :meth:`to_torch_dataset` and :meth:`to_dataloader` are thin wrappers over it,
        and there is no ``__iter__`` (one and only one obvious way).

        Iteration is a fixed cartesian sweep of BED regions x samples in a
        data-layout-optimal order (region-major for variants). There is no random
        access and no ad-hoc query: ``sds[r, s]`` raises :class:`TypeError`.

        Parameters
        ----------
        batch_size
            Number of ``(region, sample)`` cells per yielded batch. Batches are slices
            of a much larger read *window*; ``batch_size`` does not affect I/O
            granularity.
        return_indices
            If ``True`` (the default), yield ``(data, region_idxs, sample_idxs)``;
            if ``False``, yield ``data`` alone. Indices are in the caller's **original
            BED-row order** (not sorted-storage order), matching ``gvl.Dataset[r, s]``.
        """
        for data, r_idx, s_idx in self._iter_batches(batch_size):
            if return_indices:
                yield data, r_idx, s_idx
            else:
                yield data

    def n_batches(self, batch_size: int) -> int:
        """Number of batches :meth:`to_iter` will yield at ``batch_size``.

        NOT ``ceil(len(self) / batch_size)``: the plan batches *within* each window,
        so every window's last batch may be partial. Counting the plan is cheap (it
        only materializes small index arrays).
        """
        return sum(1 for _ in self._iter_batch_spans(batch_size))

    def _iter_batch_spans(self, batch_size: int) -> Iterator[int]:
        """Batch sizes the plan will yield, without reconstructing anything."""
        for r_idx, s_idx in self._plan():
            n_rows = len(r_idx) * len(s_idx)
            for lo in range(0, n_rows, batch_size):
                yield min(lo + batch_size, n_rows) - lo

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
            "StreamingDataset is iterable-only; use to_iter() instead of map-style "
            "indexing. Iteration order is fixed by the data layout, so there is no "
            "random access."
        )

    @requires_torch
    def to_torch_dataset(
        self, batch_size: int = 1, return_indices: bool = True
    ) -> "td.IterableDataset":
        """Wrap :meth:`to_iter` in a torch :class:`IterableDataset`. Thin wrapper --
        all the work is in ``to_iter``. Named to match
        :meth:`Dataset.to_torch_dataset` (same concept, same name)."""
        import torch.utils.data as td

        sds = self

        class _StreamingTorchDataset(td.IterableDataset):
            def __iter__(self):
                return sds.to_iter(batch_size, return_indices)

            def __len__(self) -> int:
                return sds.n_batches(batch_size)

        return _StreamingTorchDataset()

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
        """Wrap :meth:`to_torch_dataset` in a torch
        :class:`DataLoader <torch.utils.data.DataLoader>`. Thin wrapper.

        Parameters
        ----------
        num_workers
            Must be 0. ``StreamingDataset``'s own engine IS the concurrency strategy
            (mirrors :meth:`Dataset.to_dataloader`'s ``buffered``/``double_buffered``
            restriction); worker-process sharding of the window plan is a later plan.
        """
        if num_workers > 0:
            raise ValueError(
                "StreamingDataset.to_dataloader: num_workers>0 is not implemented "
                "yet; the streaming engine IS the concurrency strategy for "
                "StreamingDataset (mirrors gvl.Dataset.to_dataloader's "
                "buffered/double_buffered modes, which impose the same restriction). "
                "Use num_workers=0."
            )

        import torch.utils.data as td

        return td.DataLoader(
            self.to_torch_dataset(batch_size, return_indices),
            batch_size=None,  # the dataset yields pre-assembled batches
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


class _Svar1Backend:
    """Streaming SVAR1 read backend: reconstructs haplotypes for a batch of
    ``(r_idx, s_idx)`` directly from a live ``.svar`` store, with no on-disk
    gvl dataset. Wraps `Svar1Store`/`svar1_read_window`/`svar1_generate_batch`
    (Rust) -- an instance is assigned to `StreamingDataset._backend`
    internally by the `.svar` construction branch (not a public `__init__`
    kwarg) so `_iter_batches` can read a window's offsets once
    (`read_window`) and generate each batch slice separately
    (`generate_batch`), bounding peak output by `batch_size` (issue #284).

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
        # `gvl.write()` always lexicographically sorts sample names
        # (`_write.py`'s unconditional `samples.sort()`), so `gvl.Dataset`'s
        # sample index `s` means "the s-th name in sorted order" -- NOT the
        # store's native (VCF column) order. `sample_idx` must mean the same
        # thing here for parity with `gvl.Dataset.open(...)[r, s]` to hold
        # (see `to_iter`'s docstring). Toy fixtures with <=3 single-digit
        # sample names never exposed this because sort order and native
        # order coincide there; a 20-sample "S0".."S19" fixture does not
        # (lexicographically "S10" < "S2"). `_phys_sample_idx[i]` is the
        # store's native column for the i-th name in sorted order; every
        # sample index that crosses into Rust must go through it first.
        native_samples = sv.available_samples
        self._sample_names = sorted(native_samples)
        _name_to_phys = {name: i for i, name in enumerate(native_samples)}
        self._phys_sample_idx = np.array(
            [_name_to_phys[s] for s in self._sample_names], dtype=np.int64
        )
        self.n_samples = len(self._sample_names)
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

        self._svar_path = str(svar_path)
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
        # Parallel cache of the three scalars also passed to `set_contig_meta`, so
        # `build_engine` (Step 2) can assemble the engine's per-contig arrays later
        # without re-deriving them from the index table.
        self._contig_meta: dict[str, tuple[int, int, int]] = {}

        for c in self._contigs:
            mask = chrom == c
            n_local = int(mask.sum())
            if n_local == 0:
                self._store.set_contig_meta(c, 0, 0, 0)
                self._contig_arrays[c] = (
                    np.empty(0, np.uint32),
                    np.empty(0, np.uint32),
                )
                self._contig_meta[c] = (0, 0, 0)
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
            self._contig_meta[c] = (contig_start, n_local, max_v_len)

    def build_engine(
        self,
        jobs: list[tuple[int, NDArray[np.uint32], NDArray[np.uint32], int, int]],
        batch_size: int,
    ) -> object:
        """Construct a `Svar1StreamEngine` (Rust producer/consumer engine, #283) that
        overlaps window I/O with batch generation. `jobs` is one entry per WINDOW,
        `(contig_idx, region_starts, region_ends, s_lo, s_hi)`, in the SAME order
        `_iter_batches` will drive `.next_batch()`.

        Cohort-independent job residency (issue #284 / final-review Finding 1): the
        full public->physical sample map `self._phys_sample_idx` crosses ONCE (length
        `n_samples`); each job carries only its contiguous physical-sample sub-range
        `[s_lo, s_hi)` (two ints), NOT a per-window copy of that window's physical
        samples. `_plan` always yields a contiguous `arange(s_lo, s_hi)` sample chunk,
        so the engine's producer reconstructs the window's physical samples on the fly
        as `phys_sample_idx[s_lo..s_hi]`. Total job metadata is region-scale
        (`O(n_windows * window_regions)`), never `O(n_windows * n_samples)`.
        """
        from ..genvarloader import Svar1StreamEngine

        contig_names = list(self._contigs)
        contig_starts: list[int] = []
        n_locals: list[int] = []
        max_v_lens: list[int] = []
        v_starts_c: list[NDArray[np.uint32]] = []
        v_ends_c: list[NDArray[np.uint32]] = []
        contig_ref_bytes: list[NDArray[np.uint8]] = []
        for i, c in enumerate(contig_names):
            cs, nl, mv = self._contig_meta[c]
            vs_c, ve_c = self._contig_arrays[c]
            contig_starts.append(cs)
            n_locals.append(nl)
            max_v_lens.append(mv)
            v_starts_c.append(vs_c)
            v_ends_c.append(ve_c)
            ref_bytes_i, _ref_off = self._ref._contig_slice(i)
            contig_ref_bytes.append(ref_bytes_i)

        job_contig_idx = [int(j[0]) for j in jobs]
        job_region_starts = [np.ascontiguousarray(j[1], np.uint32) for j in jobs]
        job_region_ends = [np.ascontiguousarray(j[2], np.uint32) for j in jobs]
        job_s_lo = [int(j[3]) for j in jobs]
        job_s_hi = [int(j[4]) for j in jobs]
        # The full public->physical sample map, crossed ONCE (n_samples-scale). Each
        # job's (s_lo, s_hi) slices into this on the producer thread -- no per-window copy.
        phys_sample_idx = self._phys_sample_idx.astype(np.int64, copy=False).tolist()

        return Svar1StreamEngine(
            self._svar_path,
            self.n_samples,
            self.ploidy,
            contig_names,
            contig_starts,
            n_locals,
            max_v_lens,
            v_starts_c,
            v_ends_c,
            contig_ref_bytes,
            phys_sample_idx,
            job_contig_idx,
            job_region_starts,
            job_region_ends,
            job_s_lo,
            job_s_hi,
            self._v_starts,
            self._ilens,
            self._alt_alleles,
            self._alt_offsets,
            self._ref.pad_char,
            True,
            batch_size,
        )

    def read_window(
        self, r_idx: NDArray[np.intp], s_idx: NDArray[np.intp]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Read one window's CSR offsets: every region in `r_idx` x every sample in
        `s_idx`, single-contig. Returns (o_starts, o_stops), each
        `len(r_idx) * len(s_idx) * ploidy`, C-order (region, sample, ploid) -- absolute
        indices into the store's variant_idxs mmap. No haplotypes are generated here.
        """
        from ..genvarloader import svar1_read_window

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)

        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError(
                "_Svar1Backend.read_window: window spans multiple contigs; "
                "every Rust call must be single-contig."
            )
        contig_name = self._contigs[contig_idx]
        vs_c, ve_c = self._contig_arrays[contig_name]
        region_bounds = np.ascontiguousarray(self._regions[r_idx, 1:3], np.int32)

        # `s_idx` is a PUBLIC index (sorted-name order, matching `gvl.Dataset`);
        # the store's genotype CSR is laid out in native (VCF column) order, so
        # translate before crossing into Rust. See `__init__`'s comment on
        # `_phys_sample_idx`. Output row order is unaffected -- only which
        # physical column each row reads from changes.
        phys_s_idx = self._phys_sample_idx[s_idx]

        o_starts, o_stops = svar1_read_window(
            self._store,
            contig_name,
            vs_c,
            ve_c,
            region_bounds,
            np.ascontiguousarray(phys_s_idx, np.int64),
        )
        return np.asarray(o_starts, np.int64), np.asarray(o_stops, np.int64)

    def generate_batch(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        o_starts: NDArray[np.int64],
        o_stops: NDArray[np.int64],
        lo: int,
        hi: int,
    ) -> Ragged:
        """Generate haplotypes for window rows [lo:hi] (C-order (region, sample)).
        Output is (hi-lo)-bounded -- NEVER the whole window (issue #284). `o_starts`/
        `o_stops` are the whole window's offsets (from `read_window`); this slices the
        CSR rows [lo*ploidy : hi*ploidy] and the matching per-row region bounds.
        """
        from ..genvarloader import svar1_generate_batch

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)
        n_s = len(s_idx)
        # `contig_idx` already indexes `self._contigs`, and `Reference.from_path`
        # (called with `self._contigs` in `__init__`) builds `offsets` in that same
        # order -- so `contig_idx` indexes `self._ref` directly, no name lookup
        # needed. Previously this did `self._ref.c_map.contigs.index(contig_name)`,
        # which is both redundant (same answer as `contig_idx`) AND a bug:
        # `Reference.from_path` normalizes contig names to the FASTA's naming style
        # (UCSC "chr1" vs Ensembl "1"), so a store using one style paired with a
        # FASTA in the other style made `contig_name` absent from
        # `c_map.contigs`, raising `ValueError`. See `Reference._contig_slice`'s
        # docstring and `Svar2Haps._ref_for_contig` for the shared convention.
        contig_idx = int(self._regions[r_idx[0], 0])
        ref_bytes, ref_offsets = self._ref._contig_slice(contig_idx)

        # Per (region, sample) row bounds for rows [lo:hi], C-order (region, sample):
        # window row bi = ri*n_s + si -> region r_idx[bi // n_s].
        rows = np.arange(lo, hi)
        region_bounds_b = np.ascontiguousarray(
            self._regions[r_idx[rows // n_s], 1:3], np.int32
        )
        o_lo, o_hi = lo * self.ploidy, hi * self.ploidy

        data, offsets = svar1_generate_batch(
            self._store,
            np.ascontiguousarray(o_starts[o_lo:o_hi], np.int64),
            np.ascontiguousarray(o_stops[o_lo:o_hi], np.int64),
            region_bounds_b,
            self._v_starts,
            self._ilens,
            self._alt_alleles,
            self._alt_offsets,
            ref_bytes,
            ref_offsets,
            self._ref.pad_char,
            True,
        )
        n_rows = hi - lo
        return Ragged.from_offsets(
            data.view("S1"), (n_rows, self.ploidy, None), np.asarray(offsets, np.int64)
        )


class _VcfBackend:
    """Streaming VCF read backend: drives a `RecordStreamEngine` (issue #276
    tasks 3b/5) directly over a live VCF/BCF, with no on-disk `.svar` store and
    no on-disk gvl dataset. Unlike `_Svar1Backend` there is no split
    read/generate seam (`read_window`/`generate_batch`) -- a VCF/BCF has no
    equivalent of SVAR1's precomputed CSR offsets to read ahead of generation,
    so this backend supports ONLY the "engine" prefetch strategy
    (`StreamingDataset._iter_batches`'s `"engine"` branch calls nothing but
    `build_engine` on the backend).

    Header metadata (sample names, ploidy, contigs) is read once at
    construction from `genoray.VCF(path)`; per-region variant records are
    decoded window-by-window by the Rust `VcfWindowFiller` inside the engine,
    not read/cached here.
    """

    def __init__(
        self,
        vcf_path: str | Path,
        reference_path: str | Path,
        contigs: list[str] | None,
        bed: pl.DataFrame | str | Path,
    ) -> None:
        from genoray import VCF

        from ._reference import Reference

        self._vcf_path = str(vcf_path)

        vcf = VCF(self._vcf_path)
        # `gvl.write()` always lexicographically sorts sample names
        # (`_write.py`'s unconditional `samples.sort()`), so `gvl.Dataset`'s
        # sample index `s` means "the s-th name in sorted order" -- the same
        # convention `_Svar1Backend` follows (see its `__init__` comment). A
        # VCF/BCF has no separate "native order" concern for the streaming
        # engine the way SVAR1's on-disk genotype CSR does -- `RecordStreamEngine`
        # takes `sample_names` directly and looks samples up by name -- but the
        # PUBLIC sample_idx contract must still be sorted-name order to match
        # `gvl.Dataset[r, s]`.
        self._sample_names = sorted(vcf.available_samples)
        self.n_samples = len(self._sample_names)
        self.ploidy = vcf.ploidy

        # `contigs` is `None` unless the caller passed an explicit `contigs=`
        # to `StreamingDataset` -- unlike the `.svar` branch, which always
        # derives `contigs` from the store before constructing its backend,
        # the VCF branch defers to the VCF header (naturally sorted, via
        # `genoray.VCF.contigs`) when the caller didn't supply one.
        self._contigs = list(contigs) if contigs is not None else list(vcf.contigs)

        self._ref = Reference.from_path(reference_path, self._contigs)

        # `bed` is accepted for interface symmetry with `_Svar1Backend.__init__`
        # (the public ladder branch constructs both the same way) but unused
        # here: `build_engine`'s `jobs` already carry each window's
        # (contig_idx, region_starts, region_ends) directly from
        # `StreamingDataset._plan`/`_regions`, so this backend never needs its
        # own region table the way `_Svar1Backend` does for its readahead path.
        del bed

    def build_engine(
        self,
        jobs: list[tuple[int, NDArray[np.uint32], NDArray[np.uint32], int, int]],
        batch_size: int,
    ) -> object:
        """Construct a `RecordStreamEngine("vcf", ...)` (Rust producer/consumer
        engine, issue #276 tasks 3b/5) that decodes each window's variant
        records straight from the VCF/BCF. `jobs` is one entry per WINDOW,
        `(contig_idx, region_starts, region_ends, s_lo, s_hi)`, in the SAME
        order `_iter_batches` will drive `.next_batch()` -- mirrors
        `_Svar1Backend.build_engine`'s job-array unpacking exactly, minus the
        SVAR1-only store/physical-sample-map arguments (a VCF job's
        `[s_lo, s_hi)` indexes straight into `sample_names`, no public->
        physical indirection).
        """
        from ..genvarloader import RecordStreamEngine

        contig_names = list(self._contigs)
        contig_ref_bytes = [
            self._ref._contig_slice(i)[0] for i in range(len(contig_names))
        ]

        job_contig_idx = [int(j[0]) for j in jobs]
        job_region_starts = [np.ascontiguousarray(j[1], np.uint32) for j in jobs]
        job_region_ends = [np.ascontiguousarray(j[2], np.uint32) for j in jobs]
        job_s_lo = [int(j[3]) for j in jobs]
        job_s_hi = [int(j[4]) for j in jobs]

        return RecordStreamEngine(
            "vcf",
            self._vcf_path,
            self._sample_names,
            self.ploidy,
            contig_names,
            contig_ref_bytes,
            job_contig_idx,
            job_region_starts,
            job_region_ends,
            job_s_lo,
            job_s_hi,
            # `fasta_path=None` -- PARITY-CRITICAL, not a placeholder. Task 4
            # established that `gvl.write` does NO read-time reference/left-
            # alignment for VCF input, so the streaming decoder must not
            # either, to stay byte-identical (see `src/record_stream/vcf.rs`'s
            # module doc on the `fasta_path: None` parity default). `self._ref`
            # above is used ONLY to derive `contig_ref_bytes` for haplotype
            # reconstruction padding -- it is NOT the decode-time FASTA, and
            # passing its path here would enable left-alignment in the Rust
            # decoder and silently diverge from the write path.
            None,
            self._ref.pad_char,
            True,
            batch_size,
        )
