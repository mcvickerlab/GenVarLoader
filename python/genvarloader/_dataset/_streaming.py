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
    _regions: NDArray[np.int32]  # (n_regions, 3) sorted (contig_idx, start, end)
    _sort_order: NDArray[np.intp]  # maps sorted position -> original bed row
    contigs: list[str]
    n_samples: int
    ploidy: int
    _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
    _batch_size: int = 1

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
        # region-major flat index over (n_regions, n_samples): sample varies
        # fastest, WITHIN a contig -- batches never cross a contig boundary
        # (`_Svar1Backend.reconstruct_window` requires a single-contig batch).
        # `self._regions` is sorted by (contig_idx, start), so each contig's
        # regions form one contiguous run of sorted-region indices.
        n_regions, n_samples = self.shape
        if n_regions == 0:
            return
        contig_idxs = self._regions[:, 0]
        run_bounds = np.flatnonzero(np.diff(contig_idxs)) + 1
        run_starts = np.concatenate(([0], run_bounds))
        run_ends = np.concatenate((run_bounds, [n_regions]))
        for r_lo, r_hi in zip(run_starts, run_ends):
            n_run_regions = int(r_hi - r_lo)
            flat = np.arange(n_run_regions * n_samples, dtype=np.intp)
            for start in range(0, flat.size, self._batch_size):
                chunk = flat[start : start + self._batch_size]
                r_local, s_idx = np.unravel_index(chunk, (n_run_regions, n_samples))
                r_idx = (r_local + r_lo).astype(np.intp)
                yield r_idx, s_idx.astype(np.intp)

    def __iter__(self) -> Iterator[tuple]:
        for r_idx, s_idx in self._plan():
            data = self._reconstruct_window(r_idx, s_idx)
            # map sorted region positions back to the user's original bed rows
            yield (data, self._sort_order[r_idx], s_idx)

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

    class _StreamingTorchDataset(td.IterableDataset):
        def __iter__(self):
            for data, r_idx, s_idx in dataset._with_batch_size(batch_size):
                if return_indices:
                    yield data, r_idx, s_idx
                else:
                    yield data

        def __len__(self) -> int:
            n = len(dataset)
            return -(-n // batch_size)  # ceil division

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
        idx = _canonicalize_variant_table(idx)
        v_starts, ilens, ref, alt = _variant_arrays_from_table(idx, one_based=True)
        if ref is None:
            raise ValueError(f"SVAR1 store at {svar_path} has no REF allele column.")
        self._v_starts = np.ascontiguousarray(v_starts, np.int32)
        self._ilens = np.ascontiguousarray(ilens, np.int32)
        self._alt_alleles = np.ascontiguousarray(alt.data.view(np.uint8), np.uint8)
        self._alt_offsets = np.ascontiguousarray(alt.offsets, np.int64)

        self._store = Svar1Store(
            str(svar_path), self._contigs, self.n_samples, self.ploidy
        )

        chrom = idx["CHROM"].cast(pl.Utf8).to_numpy()
        for c in self._contigs:
            mask = chrom == c
            n_local = int(mask.sum())
            if n_local == 0:
                self._store.set_contig_table(c, 0, 0, [], [], [0], [], [0])
                continue

            first = int(np.argmax(mask))
            contig_start = int(idx["index"][first])
            pos_c = v_starts[first : first + n_local].astype(np.uint32).tolist()

            r_s, r_e = int(ref.offsets[first]), int(ref.offsets[first + n_local])
            ref_bytes_c = ref.data.view(np.uint8)[r_s:r_e].tolist()
            ref_offsets_c = (
                (ref.offsets[first : first + n_local + 1] - r_s)
                .astype(np.int64)
                .tolist()
            )

            a_s, a_e = int(alt.offsets[first]), int(alt.offsets[first + n_local])
            alt_bytes_c = alt.data.view(np.uint8)[a_s:a_e].tolist()
            alt_offsets_c = (
                (alt.offsets[first : first + n_local + 1] - a_s)
                .astype(np.int64)
                .tolist()
            )

            self._store.set_contig_table(
                c,
                contig_start,
                n_local,
                pos_c,
                ref_bytes_c,
                ref_offsets_c,
                alt_bytes_c,
                alt_offsets_c,
            )

    def reconstruct_window(
        self, r_idx: NDArray[np.intp], s_idx: NDArray[np.intp]
    ) -> Ragged:
        """Reconstruct haplotypes for one batch of `(region, sample)` pairs.

        This plan (Task 4 / walking-skeleton scope) requires every row in a
        batch to share one region -- i.e. one contig; `StreamingDataset`
        batches a single region's samples per call, which satisfies this.
        Returns a `Ragged[np.bytes_]` (S1) of shape `(batch, ploidy, ~length)`.
        """
        from ..genvarloader import reconstruct_haplotypes_svar1

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)

        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError(
                "_Svar1Backend.reconstruct_window: batch spans multiple "
                "contigs; this plan only supports single-contig batches "
                "(one region's samples per call)."
            )
        contig_name = self._contigs[contig_idx]

        region_bounds = np.ascontiguousarray(self._regions[r_idx, 1:3], np.int32)

        ref_c_idx = self._ref.c_map.contigs.index(contig_name)
        c_s = int(self._ref.offsets[ref_c_idx])
        c_e = int(self._ref.offsets[ref_c_idx + 1])
        ref_bytes = np.ascontiguousarray(self._ref.reference[c_s:c_e], np.uint8)
        ref_offsets = np.array([0, c_e - c_s], dtype=np.int64)

        data, offsets = reconstruct_haplotypes_svar1(
            self._store,
            contig_name,
            region_bounds,
            np.ascontiguousarray(s_idx, np.int64),
            self._v_starts,
            self._ilens,
            self._alt_alleles,
            self._alt_offsets,
            ref_bytes,
            ref_offsets,
            ord("N"),
            True,
        )
        batch = len(r_idx)
        return Ragged.from_offsets(
            data.view("S1"), (batch, self.ploidy, None), np.asarray(offsets, np.int64)
        )
