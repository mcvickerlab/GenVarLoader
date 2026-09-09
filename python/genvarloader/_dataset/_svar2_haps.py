"""SVAR2-backed haplotype/variant reconstructor (dataset read dispatch).

``Svar2Haps`` is the read-bound sibling of :class:`Svar1Haps`; the two share
only the :class:`Haps` role (see ``_haps.py``), which declares what a
haplotype reconstructor must answer and says nothing about how genotypes are
stored. There is no per-region sparse genotype array and no in-memory variant
table here -- everything is decoded from the ``.svar2`` store at read time.

For a query block of ``n_q`` rows, each row ``q = (region r_q, sample slot si_q)``
with post-jitter bounds ``[start_q, end_q)``, the cache (written by
``_write._write_from_svar2`` under ``genotypes/svar2_ranges/``) is sliced by
fancy-indexing — NO per-read interval search, NO dense-union rebuild — and fed to
the read-bound Rust kernels (``reconstruct_haplotypes_from_svar2_readbound`` /
``decode_variants_from_svar2_readbound`` / ``hap_diffs_from_svar2_readbound``).

The FFI-input shaping + output wrapping mirror
``tests/_oracles/svar2_readbound_inputs.build_readbound_*`` (test oracle) exactly; the only difference is the source
of the per-query ranges (this module slices the on-disk cache for the specific
``(r_q, si_q)`` block, whereas the helpers call ``SparseVar2._find_ranges`` over
the full cohort).

Out of scope (guarded with ``NotImplementedError``): ``min_af``/``max_af``,
annotated haps, and exonic filtering for non-haplotype outputs.
(``unphased_union`` and ``variant-windows`` ARE supported.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
from genoray._contigs import ContigNormalizer
from genoray._types import POS_TYPE
from numpy.typing import NDArray
from seqpro.rag import Ragged

from .._flat import _Flat
from .._ragged import RaggedAnnotatedHaps, RaggedIntervals, _COMP
from .._threads import should_parallelize
from .._utils import lengths_to_offsets
from ..genvarloader import (
    Svar2Store,
    decode_variants_from_svar2_readbound,
    Svar2ReadboundGather,
    gather_svar2_readbound,
    hap_diffs_from_svar2_readbound,
    reconstruct_haplotypes_from_svar2_readbound,
    reconstruct_haplotypes_from_svar2_readbound_into,
    shift_and_realign_tracks_from_svar2_readbound,
)
from ._flat_variants import (
    _FlatVariantWindows,
    _FlatWindow,
    _assemble_variant_buffers_rust,
)
from ._intervals import intervals_to_tracks
from ._haps import _H, Haps
from ._protocol import TrackRealigner
from ._rag_variants import RaggedVariants
from ._reference import Reference
from ._svar2_link import Svar2Link, _resolve_svar2, _verify_svar2_fingerprint

if TYPE_CHECKING:
    from genoray._svar2_fields import StoredField

    from ._splice import SplicePlan


_GatherInputs: TypeAlias = tuple[
    NDArray[np.uint32],  # region_starts
    NDArray[np.int64],  # orig_samples
    NDArray[np.int64],  # vk_snp_range
    NDArray[np.int64],  # vk_indel_range
    NDArray[np.int64],  # dense_snp_range
    NDArray[np.int64],  # dense_indel_range
    NDArray[np.int32],  # region_bounds
]
"""The read-bound FFI input arrays produced by ``Svar2Haps._gather_inputs``."""

_GatheredGroup: TypeAlias = tuple[int, NDArray[np.intp], _GatherInputs]
"""``(dataset contig index, its query positions, that group's gathered inputs)``."""


def _range_work_bytes(n_queries: int, P: int, gi: _GatherInputs) -> int:
    """Rough byte-cost of the read-bound work a gathered group implies.

    Feeds ``should_parallelize`` only -- it is a scheduling heuristic and never
    affects output bytes. The floor keeps a wide-but-shallow block (many rows,
    few variants each) from reading as no work at all.
    """
    span = sum(int(np.maximum(0, r[:, 1] - r[:, 0]).sum()) for r in gi[2:])
    return max(n_queries * P * 64, span * 16)


_BUILTIN_VAR_FIELDS: frozenset[str] = frozenset(
    {"alt", "ilen", "start", "ref", "dosage"}
)
"""Variant-field keys the reconstructors handle natively (never store fields)."""


def _field_spec(sf: "StoredField") -> tuple[str, str, str]:
    """(category, name, dtype_str) as the Rust FFI expects it."""
    from genoray._svar2_fields import _META_DTYPE

    return (sf.category, sf.name, _META_DTYPE[sf.dtype])


@dataclass(slots=True)
class _Svar2Cache:
    """The six memmapped ``svar2_ranges/`` arrays (all int64), sliced per query.

    ``vk_*_range`` are ``(R, S, P, 2)`` (per region/sample/ploid byte windows into
    the store's var_key tables); ``dense_*_range`` are ``(R, 2)`` (per-region,
    sample-independent); ``sample_cols`` is ``(S,)`` (selected slot -> original
    store sample index). Per-query starts are recomputed post-jitter at read time,
    so they are not cached here.
    """

    vk_snp_range: NDArray[np.int64]
    vk_indel_range: NDArray[np.int64]
    dense_snp_range: NDArray[np.int64]
    dense_indel_range: NDArray[np.int64]
    sample_cols: NDArray[np.int64]


def _ragged_arange_src(
    offsets: NDArray[np.integer], perm: NDArray[np.integer]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Source-row index + new offsets for a 1-level ragged reorder by ``perm``.

    ``new_data == data[src]``; ``src`` and ``new_off`` depend only on
    ``(offsets, perm)`` — so callers reordering several parallel data arrays by
    the same key compute this ONCE and index each array.
    """
    offsets = np.asarray(offsets, np.int64)
    lens = np.diff(offsets)
    new_lens = lens[perm]
    new_off = lengths_to_offsets(new_lens, np.int64)
    n = int(new_off[-1])
    if n == 0:
        return np.zeros(0, np.int64), new_off
    within = np.arange(n, dtype=np.int64) - np.repeat(new_off[:-1], new_lens)
    src = np.repeat(offsets[perm], new_lens) + within
    return src, new_off


def _ragged_arange_gather(
    data: NDArray, offsets: NDArray[np.integer], perm: NDArray[np.integer]
) -> tuple[NDArray, NDArray[np.int64]]:
    """Reorder the rows of a 1-level ragged array ``(data, offsets)`` by ``perm``.

    Copies each permuted row's byte span into the output with a single
    ``np.concatenate`` over per-row slices. This is ~50x faster on the spliced
    haplotype path than the byte-level fancy-index form (``data[src]`` with
    ``src = arange(n_bytes) - repeat(...)``): the Python-level work scales with
    the number of rows (``len(perm)``), not the number of output bytes, while
    the byte movement itself stays a single C-level copy. ``_ragged_arange_src``
    keeps the index-based form for the two-source variants path, which needs the
    explicit ``src`` to co-index several parallel arrays.
    """
    offsets = np.asarray(offsets, np.int64)
    perm = np.asarray(perm, np.intp)
    new_off = lengths_to_offsets(np.diff(offsets)[perm], np.int64)
    if int(new_off[-1]) == 0:
        return data[:0].copy(), new_off
    starts = offsets[perm].tolist()
    ends = offsets[perm + 1].tolist()
    return np.concatenate([data[s:e] for s, e in zip(starts, ends)]), new_off


def _ragged_arange_gather_2level(
    data: NDArray,
    var_off: NDArray[np.integer],
    str_off: NDArray[np.integer],
    perm: NDArray[np.integer],
) -> tuple[NDArray, NDArray[np.int64], NDArray[np.int64]]:
    """Reorder the rows of a 2-level ragged (opaque-string) array by ``perm``.

    ``var_off`` (len ``n_rows + 1``) bounds variants per row; ``str_off`` (len
    ``n_variants + 1``) bounds bytes per variant. Returns
    ``(new_data, new_var_off, new_str_off)``. Fully vectorized.
    """
    var_off = np.asarray(var_off, np.int64)
    str_off = np.asarray(str_off, np.int64)
    var_lens = np.diff(var_off)
    new_var_lens = var_lens[perm]
    new_var_off = lengths_to_offsets(new_var_lens, np.int64)
    total_vars = int(new_var_off[-1])
    if total_vars == 0:
        return data[:0].copy(), new_var_off, np.zeros(1, np.int64)
    within_var = np.arange(total_vars, dtype=np.int64) - np.repeat(
        new_var_off[:-1], new_var_lens
    )
    old_var_idx = np.repeat(var_off[perm], new_var_lens) + within_var
    var_byte_len = np.diff(str_off)
    new_byte_len = var_byte_len[old_var_idx]
    new_str_off = lengths_to_offsets(new_byte_len, np.int64)
    nbytes = int(new_str_off[-1])
    if nbytes == 0:
        return data[:0].copy(), new_var_off, new_str_off
    within_b = np.arange(nbytes, dtype=np.int64) - np.repeat(
        new_str_off[:-1], new_byte_len
    )
    src = np.repeat(str_off[old_var_idx], new_byte_len) + within_b
    return data[src], new_var_off, new_str_off


@dataclass(kw_only=True, slots=True)
class Svar2Haps(Haps[_H]):
    """Read-bound SVAR2 reconstructor. See module docstring."""

    store: "Svar2Store"
    """The query-only handle on the ``.svar2`` store."""
    cache: "_Svar2Cache"
    """The per-query variant ranges written at ``genotypes/svar2_ranges/``."""
    store_contigs: list[str]
    """The .svar2 store's contig names (used to open the store's ContigReaders)."""
    ds_contigs: list[str]
    """The dataset's contig names (``regions[:, 0]`` indexes into this)."""
    n_regions: int
    """The dataset's region count, from the range cache's ``dense_snp_range`` shape."""
    n_samples: int
    """The dataset's sample count, from the range cache's ``vk_snp_range`` shape.

    Together with :attr:`n_regions` this is the ``(R, S)`` grid a flat dataset
    index unravels into. SVAR1 reads the same two numbers off its ``genotypes``
    array's leading shape; there is no such array here.
    """
    ploidy: int
    """The store's ploidy, from the svar2 range cache's ``svar2_meta.json``.

    Backs :attr:`stored_ploidy`.
    """
    store_fields: dict[str, "StoredField"]
    """The .svar2 store's INFO/FORMAT field manifest, keyed by field key.

    Populated from ``SparseVar2.available_fields``. These keys are additionally
    advertised in ``available_var_fields`` so users can request them via ``var_fields``.
    """
    max_jitter: int = 0
    """The dataset's write-time max_jitter. When > 0 the cache's per-query ranges
    were computed over a max_jitter-padded window, which over-includes variants past
    the (unpadded) read window in variants mode (the decode kernel has no right-clip);
    guarded below."""
    _gather_memo: (
        tuple[int, NDArray[np.intp], NDArray[np.int32], list[_GatheredGroup]] | None
    ) = field(init=False, repr=False)
    """One-batch memo behind :meth:`_gathered_groups`. See that method."""
    _rust_gathers: dict[int, Svar2ReadboundGather] = field(init=False, repr=False)
    """Per-contig Rust gather handles for the batch in ``_gather_memo``.

    Filled lazily by :meth:`_readbound_gather` and cleared whenever that memo misses,
    so a handle can never outlive the batch it was built for.
    """
    _ds_to_store: list[str | None] = field(init=False)
    """``ds_contigs`` mapped to the store's spelling, parallel to ``ds_contigs``.

    The dataset and its .svar2 store may name contigs differently (e.g. a UCSC-named
    ``chr1`` reference/GTF over an Ensembl-named ``1`` store) — the write path
    reconciles the two, so the read path must too. ``None`` marks a dataset contig
    with no equivalent in the store; that is only an error if such a contig is
    actually queried, so it is raised lazily by :meth:`_store_contig`.
    """

    def __post_init__(self):
        # Deliberately does NOT call any SVAR1 setup (there is no variants table
        # or AF cache to read). Set only the role's init=False fields.
        #
        # n_variants is all zeros, and wrong: the read-bound decode never counts
        # a query's variants without also decoding them, so there is nothing to
        # fill this with at open time. Tracked as #363 -- Dataset.n_variants()
        # reports zeros on SVAR2. Kept zero-valued rather than absent because the
        # shape (R, S, P) is what callers read it for.
        self.n_variants = np.zeros(
            (self.n_regions, self.n_samples, self.ploidy), np.int32
        )
        self.available_var_fields = ["alt", "ilen", "start"] + [
            k for k in self.store_fields if k not in _BUILTIN_VAR_FIELDS
        ]
        self._gather_memo = None
        self._rust_gathers = {}
        # Resolve once per contig at open, not once per query.
        self._ds_to_store = ContigNormalizer(self.store_contigs).norm(self.ds_contigs)

    def _store_contig(self, ci: int) -> str:
        """Return the .svar2 store's spelling of dataset contig index ``ci``.

        The read-bound Rust kernels look contigs up in the store's own keyspace, so
        they must be handed the store's spelling rather than the dataset's.
        """
        contig = self._ds_to_store[ci]
        if contig is None:
            raise ValueError(
                f"Dataset contig {self.ds_contigs[ci]!r} has no equivalent in the"
                f" .svar2 store, whose contigs are {self.store_contigs}."
            )
        return contig

    # ---- backend-agnostic query surface (see Haps) ----

    @property
    def stored_ploidy(self) -> int:
        """Ploidy as laid out in the .svar2 store."""
        return self.ploidy

    @property
    def has_ref_alleles(self) -> bool:
        """SVAR2 never carries REF allele bytes (the decode is ALT-only)."""
        return False

    @property
    def has_dosages(self) -> bool:
        """SVAR2 never emits dosages.

        ``dosage`` is a builtin name the read-bound decode kernel does not
        produce, so it is filtered out of both :attr:`available_var_fields` and
        :meth:`_requested_store_fields` and can never be requested.
        """
        return False

    def check_track_realign_support(
        self,
        output_length: Literal["ragged", "variable"] | int,
        splice_plan: "SplicePlan | None",
        to_rc: NDArray[np.bool_] | None,
        ragged_tracks: bool,
    ) -> None:
        """Reject the request shapes the read-bound track path cannot serve.

        Args:
            output_length: The requested output length.
            splice_plan: The requested splice plan, if any.
            to_rc: Per-query reverse-complement mask, if any.
            ragged_tracks: Whether the tracks are being realigned at all.

        Raises:
            NotImplementedError: For splicing, annotated haps, a fixed output
                length, or in-kernel reverse-complement.
        """
        if splice_plan is not None:
            raise NotImplementedError(
                "Splicing of haplotypes + tracks is not supported for svar2 "
                "datasets yet."
            )
        # Annotated haps are out of scope for svar2 (matches Svar2Haps.__call__).
        # HapsTracks checks support BEFORE reaching Svar2Haps's own annotated
        # guard, and get_haps_and_shifts returns plain Ragged[S1] regardless of
        # kind, so without this an annotated view would silently yield
        # non-annotated haps.
        if issubclass(self.kind, RaggedAnnotatedHaps):
            raise NotImplementedError(
                "svar2 datasets do not support with_seqs('annotated') with tracks yet."
            )
        # The realign kernel has no in-kernel reverse-complement.
        if to_rc is not None and bool(np.asarray(to_rc).any()):
            raise NotImplementedError(
                "In-kernel reverse-complement is not supported for svar2 "
                "haplotype-realigned tracks."
            )
        # The readbound track kernel always sizes each hap to ref_len + diff
        # (no output_length override), so a fixed-length request cannot be
        # honored byte-identically. Guard rather than silently mis-size.
        if ragged_tracks and isinstance(output_length, int):
            raise NotImplementedError(
                "Fixed-length (int output_length) haplotype-realigned tracks "
                "are not supported for svar2 datasets yet; use ragged/variable "
                "output."
            )

    def var_field_dtype(self, field: str) -> np.dtype:
        """The dtype of a scalar variant field, from the store manifest.

        There is no in-memory variant table here: the .svar2 store's INFO/FORMAT
        dtypes live in ``store_fields``, and ``start``/``ilen`` are fixed by the
        read-bound decode kernel's output.
        """
        if field in ("alt", "ref"):
            raise KeyError(
                f"{field!r} is a variable-length allele field with no scalar dtype;"
                " size it from its byte payload instead."
            )
        if field == "start":
            return np.dtype(POS_TYPE)
        if field == "ilen":
            return np.dtype(np.int32)
        if field in self.store_fields:
            return self.store_fields[field].dtype
        raise KeyError(f"unknown variant field {field!r}")

    # ---- construction ----

    @classmethod
    def from_path(  # type: ignore[override]  # separate svar2 signature; base returns Haps[RaggedVariants]
        cls,
        path: Path,
        reference: Reference | None,
        samples: list[str],
        ploidy: int,
        svar2_link: Svar2Link | None,
        svar2_override: Path | str | None,
        contigs: list[str],
        kind: type = RaggedVariants,
        min_af: float | None = None,
        max_af: float | None = None,
        max_jitter: int = 0,
        var_fields: list[str] | None = None,
    ) -> "Svar2Haps":
        # Default var_fields for loading. var_fields=None means "use the default
        # set" — mirrors Haps.from_path's resolution.
        if var_fields is None:
            var_fields = ["alt", "ilen", "start"]

        ranges_dir = path / "genotypes" / "svar2_ranges"
        with open(ranges_dir / "svar2_meta.json") as f:
            meta = json.load(f)

        def _mm(name: str, shape: list[int]) -> NDArray[np.int64]:
            return np.memmap(
                ranges_dir / name, dtype=np.int64, mode="r", shape=tuple(shape)
            )

        R = int(meta["dense_snp_range"]["shape"][0])
        S = int(meta["vk_snp_range"]["shape"][1])
        P = int(meta["ploidy"])
        if P != ploidy:
            raise ValueError(f"svar2 cache ploidy ({P}) != dataset ploidy ({ploidy}).")

        cache = _Svar2Cache(
            vk_snp_range=_mm("vk_snp_range.npy", meta["vk_snp_range"]["shape"]),
            vk_indel_range=_mm("vk_indel_range.npy", meta["vk_indel_range"]["shape"]),
            dense_snp_range=_mm(
                "dense_snp_range.npy", meta["dense_snp_range"]["shape"]
            ),
            dense_indel_range=_mm(
                "dense_indel_range.npy", meta["dense_indel_range"]["shape"]
            ),
            sample_cols=np.load(ranges_dir / "sample_cols.npy"),
        )

        svar2_path = _resolve_svar2(path, svar2_link, svar2_override)
        _verify_svar2_fingerprint(svar2_path, svar2_link)

        # Open the query-only store. n_samples must be the store's FULL sample
        # count (orig_samples / sample_cols index into it), not len(samples).
        from genoray import SparseVar2

        sv = SparseVar2(str(svar2_path))
        store = Svar2Store(str(svar2_path), sv.contigs, sv.n_samples, sv.ploidy)
        store_fields = dict(sv.available_fields)

        allowed = {"alt", "ilen", "start"} | set(store_fields)
        if missing := [f for f in var_fields if f not in allowed]:
            raise ValueError(f"Missing variant fields: {missing}")

        return cls(
            path=path,
            reference=reference,
            kind=cast("type[_H]", kind),
            filter=None,
            min_af=min_af,
            max_af=max_af,
            store=store,
            cache=cache,
            store_contigs=list(sv.contigs),
            ds_contigs=list(contigs),
            n_regions=R,
            n_samples=S,
            ploidy=P,
            max_jitter=max_jitter,
            store_fields=store_fields,
            var_fields=var_fields,
        )

    # ---- reconstructor entry ----

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: "SplicePlan | None" = None,
        flat: bool = False,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> _H:
        self._guard_unsupported(splice_plan)

        if issubclass(self.kind, (RaggedVariants, _FlatVariantWindows)):
            if splice_plan is not None:
                raise NotImplementedError(
                    "Spliced output is not supported for the 'variants' or "
                    "'variant-windows' sequence types."
                )
            if self.filter == "exonic" and issubclass(self.kind, _FlatVariantWindows):
                raise NotImplementedError(
                    "var_filter='exonic' is not supported for SVAR2 "
                    "variant-window output."
                )
            # variants AND variant-windows decode variants; the read-bound decode
            # has NO right-clip, so max_jitter>0 / jitter>0 would over-include
            # variants past the (unpadded) read window. Guard both modes.
            if self.max_jitter > 0 or jitter > 0:
                raise NotImplementedError(
                    "variants/variant-windows output for svar2 datasets written with"
                    f" max_jitter>0 (here max_jitter={self.max_jitter}) or read with"
                    f" jitter>0 (here jitter={jitter}) is not yet supported: the"
                    " read-bound decode does not right-clip to the post-jitter window."
                )
            if issubclass(self.kind, _FlatVariantWindows):
                return cast(_H, self._reconstruct_variant_windows(idx, regions))
            # RaggedVariants: RC is applied by the caller (_getitem_unspliced),
            # so to_rc is intentionally ignored here (mirrors SVAR1 Haps).
            return cast(_H, self._reconstruct_variants(idx, regions))

        if issubclass(self.kind, RaggedAnnotatedHaps):
            raise NotImplementedError(
                "svar2 datasets do not support with_seqs('annotated') yet."
            )

        if splice_plan is not None:
            return cast(
                _H,
                self._reconstruct_spliced(
                    idx=idx,
                    regions=np.asarray(regions, np.int32),
                    splice_plan=splice_plan,
                    to_rc=to_rc,
                ),
            )

        haps, *_ = self.get_haps_and_shifts(
            idx=idx,
            regions=regions,
            output_length=output_length,
            rng=rng,
            deterministic=deterministic,
            splice_plan=None,
            to_rc=None,
            need_hap_lengths=False,
        )

        if to_rc is None or not bool(np.asarray(to_rc).any()):
            return cast(_H, haps)

        flat = _Flat.from_offsets(
            np.asarray(haps.data), haps.shape, np.asarray(haps.offsets, np.int64)
        ).view("S1")
        flat = flat.reverse_masked(np.asarray(to_rc, np.bool_), comp=_COMP)
        return cast(_H, flat)

    def _reconstruct_spliced(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        splice_plan: "SplicePlan",
        to_rc: "NDArray[np.bool_] | None",
    ) -> _Flat[np.bytes_]:
        """Reconstruct spliced haplotypes directly into spliced layout (no re-order).

        The splice plan already knows every element's final address, so instead of
        reconstructing in region order and permuting the OUTPUT BYTES afterwards, we
        permute the per-row METADATA (O(rows)) and let each contig group's kernel call
        scatter straight into the shared buffer — the same trick SVAR1's fused spliced
        entry uses (``reconstruct_haplotypes_spliced_fused``).

        The plan's k-index (``k = query * E + e`` with ``E = ploidy`` for haplotypes,
        see ``_splice.build_splice_plan``) is exactly the kernel's row index
        ``k = q * P + p``, so ``plan.permutation`` indexes hap rows with no translation.

        Callers reach this only via ``_getitem_spliced``, which asserts ``jitter == 0``
        and ``deterministic`` — hence zero shifts.
        """
        regions = np.asarray(regions, np.int32)
        P = self.stored_ploidy
        b = len(idx)

        perm = np.asarray(splice_plan.permutation, np.intp)
        off = np.asarray(splice_plan.permuted_out_offsets, np.int64)
        n_work = b * P
        if len(perm) != n_work:
            raise AssertionError(
                f"splice permutation length {len(perm)} != n_queries*ploidy {n_work}"
            )

        # dest_rank[k] = position of kernel row k within the permuted (spliced) layout.
        dest_rank = np.empty(n_work, np.intp)
        dest_rank[perm] = np.arange(n_work, dtype=np.intp)
        bounds_all = np.empty((n_work, 2), np.int64)
        bounds_all[:, 0] = off[dest_rank]
        bounds_all[:, 1] = off[dest_rank + 1]

        # to_rc arrives in permuted order (_getitem_spliced builds it as
        # to_rc_flat[plan.permutation]); the kernel wants it per row.
        rc_all: NDArray[np.bool_] | None = None
        if to_rc is not None and bool(np.asarray(to_rc).any()):
            rc_all = np.empty(n_work, np.bool_)
            rc_all[perm] = np.asarray(to_rc, np.bool_)

        out = np.empty(int(off[-1]), np.uint8)
        shifts_all = np.zeros((b, P), np.int32)
        p_range = np.arange(P, dtype=np.intp)

        for ci, qsel, gi in self._gathered_groups(idx, regions, P):
            ref_, ref_offsets = self._ref_for_contig(ci)
            rows = (qsel[:, None] * P + p_range).ravel()
            g_bounds = np.ascontiguousarray(bounds_all[rows], np.int64)
            g_rc = (
                None if rc_all is None else np.ascontiguousarray(rc_all[rows], np.bool_)
            )
            g_total = int((g_bounds[:, 1] - g_bounds[:, 0]).sum())
            reconstruct_haplotypes_from_svar2_readbound_into(
                out,
                g_bounds,
                self.store,
                self._store_contig(ci),
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                gi[6],
                np.ascontiguousarray(shifts_all[qsel], np.int32),
                ref_,
                ref_offsets,
                np.uint8(self.reference.pad_char),  # type: ignore[union-attr]  # reference guaranteed for haplotypes
                g_rc,
                should_parallelize(g_total),
                self.filter == "exonic",
                self._readbound_gather(ci, gi, P),
            )

        return _Flat.from_offsets(out, (len(perm), None), off).view("S1")

    def haplotype_lengths_for_plan(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """Compute per-query SVAR2 haplotype lengths for a splice plan."""
        regions = np.asarray(regions, np.int32)
        lengths = regions[:, 2] - regions[:, 1]
        return (lengths[:, None] + self._haplotype_diffs(idx, regions)).astype(
            np.int32, copy=False
        )

    def _haplotype_diffs(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """Return ``(query, ploidy)`` SVAR2 length deltas."""
        regions = np.asarray(regions, np.int32)
        ploidy = self.stored_ploidy
        diffs = np.empty((len(idx), ploidy), np.int32)
        for ci, qsel, gi in self._gathered_groups(idx, regions, ploidy):
            d = hap_diffs_from_svar2_readbound(
                self.store,
                self._store_contig(ci),
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                gi[6],
                ploidy,
                self.filter == "exonic",
                should_parallelize(_range_work_bytes(len(qsel), ploidy, gi)),
                self._readbound_gather(ci, gi, ploidy),
            )
            diffs[qsel] = np.asarray(d, np.int32).reshape(len(qsel), ploidy)
        return diffs

    def _haplotype_ilens(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        deterministic: bool,
        keep: NDArray[np.bool_] | None = None,
        keep_offsets: NDArray[np.integer] | None = None,
    ) -> NDArray[np.int32]:
        """SVAR2 per-haplotype length deltas. Backs ``Dataset.haplotype_lengths``.

        The deltas come from the read-bound kernel, not from a stored genotype
        array. Before the role split, ``Svar2Haps`` inherited the SVAR1 body and
        it read empty placeholders, silently yielding all-zero deltas -- i.e.
        ``haplotype_lengths()`` reporting the unadjusted reference span, and
        ``_output_bytes_per_instance`` under-sizing the
        ``"haplotypes"``/``"annotated"`` slots (the sibling of the
        ``"variants"``/``"variant-windows"`` defect in #315). Fixed in #364.

        ``keep``/``keep_offsets`` carry the SVAR1 exonic *pre*-filter's output.
        The read-bound kernel applies the exonic filter itself (it is handed
        ``self.filter == "exonic"``), so a caller-supplied mask has no meaning
        here; reject it rather than ignore it.
        """
        if keep is not None or keep_offsets is not None:
            raise NotImplementedError(
                "Svar2Haps._haplotype_ilens does not accept a caller-supplied keep"
                " mask: the read-bound kernel applies the exonic filter itself."
            )
        return self._haplotype_diffs(idx, np.asarray(regions, np.int32))

    def get_haps_and_shifts(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        output_length: Literal["ragged", "variable"] | int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: "SplicePlan | None" = None,
        to_rc: "NDArray[np.bool_] | None" = None,
        need_hap_lengths: bool = True,
    ) -> tuple[
        _Flat[np.bytes_],
        NDArray[np.intp],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.bool_] | None,
        NDArray[np.int64] | None,
    ]:
        """Reconstruct haplotypes + return the SVAR1-shaped 7-tuple.

        Track re-alignment reuses this for the shared shifts/diffs/
        hap_lengths; ``geno_offset_idx`` is a placeholder for svar2 (the cache is
        re-sliced from ``idx`` there), and ``keep``/``keep_offsets`` are None.
        """
        self._guard_unsupported(splice_plan)
        if splice_plan is not None:
            raise NotImplementedError(
                "Svar2Haps.__call__ owns splice permutation; "
                "get_haps_and_shifts expects splice_plan=None."
            )
        regions = np.asarray(regions, np.int32)
        P = self.stored_ploidy
        b = len(idx)
        lengths = (regions[:, 2] - regions[:, 1]).astype(np.int64)

        # diffs are needed pre-reconstruct ONLY to (a) bound randomized jitter
        # shifts, or (b) return hap_lengths/diffs to a caller that uses them
        # (the tracks path). A deterministic/ragged haplotypes read needs
        # neither: reconstruct sizes itself internally. Avoid the redundant
        # gather+split+diffs in that (common warm-read) case.
        randomized = not (deterministic or isinstance(output_length, str))
        need_diffs = randomized or need_hap_lengths

        # --- diffs (per contig group, stitched back to (b, P) query order) ---
        if need_diffs:
            diffs = self._haplotype_diffs(idx, regions)

            hap_lengths = (lengths[:, None] + diffs).astype(np.int32)
        else:
            diffs = np.zeros(
                (b, P), np.int32
            )  # diffs discarded by __call__ (only caller reaching this branch)
            # hap_lengths below still feeds g_total -> should_parallelize (a perf
            # heuristic only; never affects output bytes), so the ref-length
            # placeholder is byte-identical-safe.
            hap_lengths = np.broadcast_to(
                lengths[:, None].astype(np.int32), (b, P)
            ).copy()

        # --- shifts (single rng draw; mirrors Haps._prepare_request) ---
        if randomized:
            max_shift = diffs.clip(min=0)
            max_shift = max_shift + (lengths - output_length).clip(min=0)[:, None]
            shifts = rng.integers(0, max_shift + 1, dtype=np.int32)
        else:
            shifts = np.zeros((b, P), np.int32)

        ffi_out_len = (
            np.int64(-1) if isinstance(output_length, str) else np.int64(output_length)
        )

        # --- reconstruct (per contig group), collect in grouped query order ---
        cat_data: list[NDArray[np.uint8]] = []
        cat_hap_lens: list[NDArray[np.int64]] = []
        cat_query_order: list[NDArray[np.intp]] = []
        for ci, qsel, gi in self._gathered_groups(idx, regions, P):
            ref_, ref_offsets = self._ref_for_contig(ci)
            g_shifts = np.ascontiguousarray(shifts[qsel], np.int32)
            if isinstance(output_length, int):
                g_total = int(output_length) * len(qsel) * P
            else:
                g_total = int(hap_lengths[qsel].sum())
            g_data, g_off = reconstruct_haplotypes_from_svar2_readbound(
                self.store,
                self._store_contig(ci),
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                gi[6],
                g_shifts,
                ref_,
                ref_offsets,
                np.uint8(self.reference.pad_char),  # type: ignore[union-attr]  # reference guaranteed for haplotypes
                ffi_out_len,
                should_parallelize(g_total),
                self.filter == "exonic",
            )
            cat_data.append(np.asarray(g_data, np.uint8))
            cat_hap_lens.append(np.diff(np.asarray(g_off, np.int64)))
            cat_query_order.append(qsel)

        out = self._assemble_haps(cat_data, cat_hap_lens, cat_query_order, b, P)

        geno_offset_idx = np.repeat(
            np.asarray(idx, np.intp)[:, None], P, axis=1
        )  # svar2 placeholder; realign_track_block re-slices the cache from idx.
        return out, geno_offset_idx, shifts, diffs, hap_lengths, None, None

    # ---- tracks ----

    def track_realigner(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        shifts: NDArray[np.int32],
        geno_idx: NDArray[np.integer],
        track_lengths: NDArray[np.integer],
        out_offsets: NDArray[np.integer],
        keep: NDArray[np.bool_] | None,
        keep_offsets: NDArray[np.integer] | None,
        to_rc: NDArray[np.bool_] | None,
        base_seed: int,
    ) -> TrackRealigner:
        """Prepare the per-batch state for the read-bound track path.

        ``geno_idx``/``keep``/``keep_offsets``/``to_rc`` describe the SVAR1
        sparse-genotype layout and the exonic filter, neither of which the
        read-bound decode uses: the kernel re-derives its own per-contig ranges
        from the store, and :meth:`check_track_realign_support` has already
        rejected the reverse-complement and exonic cases.

        Args:
            idx: Flat ``(region, sample)`` dataset indices for the batch.
            regions: ``(b, 3)`` contig/start/end of each query.
            shifts: ``(b, p)`` per-haplotype jitter shifts.
            geno_idx: Unused; see above.
            track_lengths: ``(b,)`` reference span each track block is read over.
            out_offsets: Unused; the kernel sizes each hap natively.
            keep: Unused; see above.
            keep_offsets: Unused; see above.
            to_rc: Unused; see above.
            base_seed: Seed for seed-dependent insertion fills.

        Returns:
            A realigner whose ``fill`` writes one track at a time.
        """
        del geno_idx, out_offsets, keep, keep_offsets, to_rc
        return _Svar2TrackRealigner(
            haps=self,
            idx=idx,
            regions=regions,
            shifts=shifts,
            track_lengths=track_lengths,
            base_seed=base_seed,
        )

    def realign_track_block(
        self,
        idx: NDArray[np.integer],
        o_idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        shifts: NDArray[np.int32],
        track_lengths: NDArray[np.integer],
        intervals,
        params: NDArray[np.float64],
        strategy_id: int,
        base_seed: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Haplotype-realign ONE track for a query block, returning a flat f32 buffer + offsets in global ``(b, P)`` C-order (row = ``q * P + p``).

        The two-step SVAR2 track path (there is no fused interval→realign kernel):

        1. **Materialize** the per-query reference-space track window from
           ``intervals`` via the standalone :func:`intervals_to_tracks` FFI. Each
           window starts at ``regions[q, 1]`` and spans ``track_lengths[q]`` ref
           bases (= region length + max deletion across ploidy), exactly the
           ``tracks``/``track_offsets`` input the realign kernel expects (one
           window per query; all P haps of a query share it).
        2. **Realign** to haplotype coordinates via
           :func:`shift_and_realign_tracks_from_svar2_readbound`, cache-sliced per
           contig group EXACTLY like :meth:`get_haps_and_shifts` slices for haps.

        ``o_idx`` selects the interval row per query (``idx`` for SAMPLE tracks,
        ``r_idx`` otherwise). Per-hap output length is ``ref_len + diff`` (the
        kernel's native sizing), so the stitched offsets equal the haps'
        ``hap_lengths`` in the same order.
        """
        if self.filter == "exonic":
            raise NotImplementedError(
                "SVAR2 exonic filtering with haplotype-realigned tracks is not "
                "supported yet."
            )
        regions = np.asarray(regions, np.int32)
        P = self.stored_ploidy
        b = len(idx)

        params_c = np.ascontiguousarray(params, np.float64)
        o_idx = np.asarray(o_idx)
        track_lengths = np.asarray(track_lengths, np.int64)

        cat_data: list[NDArray[np.float32]] = []
        cat_lens: list[NDArray[np.int64]] = []
        cat_query_order: list[NDArray[np.intp]] = []
        for ci, qsel, gi in self._gathered_groups(idx, regions, P):
            # (1) materialize ref-space track windows for this group's queries.
            tl_g = track_lengths[qsel]
            track_ofsts_g = lengths_to_offsets(tl_g, np.int64)
            tracks_buf = np.empty(int(track_ofsts_g[-1]), np.float32)
            intervals_to_tracks(
                offset_idxs=o_idx[qsel],
                starts=regions[qsel, 1],
                itv_starts=intervals.starts.data,
                itv_ends=intervals.ends.data,
                itv_values=intervals.values.data,
                itv_offsets=intervals.starts.offsets,
                out=tracks_buf,
                out_offsets=track_ofsts_g,
            )

            # (2) realign to haplotype coordinates (cache-sliced, per contig).
            g_shifts = np.ascontiguousarray(shifts[qsel], np.int32)
            g_total = int(track_ofsts_g[-1])
            out_data, out_off = shift_and_realign_tracks_from_svar2_readbound(
                self.store,
                self._store_contig(ci),
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                gi[6],
                g_shifts,
                np.ascontiguousarray(tracks_buf, np.float32),
                track_ofsts_g,
                params_c,
                np.int64(strategy_id),
                np.uint64(base_seed),
                # GLOBAL batch row per local query: this group's queries land at
                # positions `qsel` in the full (b, P) batch. The kernel seeds the
                # FlankSample fill with this (not the contig-local `k / ploidy`),
                # so the per-contig-group split matches the single fused SVAR1
                # call byte-for-byte (issue #267).
                np.ascontiguousarray(qsel, np.int64),
                should_parallelize(g_total * 4),
            )
            cat_data.append(np.asarray(out_data, np.float32))
            cat_lens.append(np.diff(np.asarray(out_off, np.int64)))
            cat_query_order.append(qsel)

        # Single contig group: grouped order already equals global (b, P) order,
        # so the reorder is the identity — return the sole group's buffer directly.
        if len(cat_data) == 1:
            return cat_data[0], lengths_to_offsets(cat_lens[0], np.int64)
        data = np.concatenate(cat_data) if cat_data else np.zeros(0, np.float32)
        lens = np.concatenate(cat_lens) if cat_lens else np.zeros(0, np.int64)
        grouped_off = lengths_to_offsets(lens, np.int64)
        perm = self._inverse_row_perm(cat_query_order, b, P)
        return _ragged_arange_gather(data, grouped_off, perm)

    # ---- variants ----

    def _requested_store_fields(
        self,
    ) -> tuple[list[str], list[tuple[str, str, str]], list[np.dtype]]:
        """The store INFO/FORMAT fields requested via ``var_fields``.

        Returns ``(keys, specs, dtypes)`` where ``specs`` is what the decode kernel
        expects and ``keys``/``dtypes`` are positionally parallel to the field
        buffers it returns. Builtin names (alt/start/ilen/ref/dosage) always mean the
        builtin, even if the store happens to carry a field of the same name.
        """
        keys = [
            f
            for f in self.var_fields
            if f not in _BUILTIN_VAR_FIELDS and f in self.store_fields
        ]
        specs = [_field_spec(self.store_fields[k]) for k in keys]
        dtypes = [self.store_fields[k].dtype for k in keys]
        return keys, specs, dtypes

    def _type_field_bufs(
        self,
        bufs: list[NDArray],
        isizes: list[int],
        keys: list[str],
        dtypes: list[np.dtype],
    ) -> list[NDArray]:
        """View each raw ``uint8`` field buffer the decode kernel returned as its store dtype.

        Guards that the kernel's reported itemsize agrees with the store
        manifest -- a store/kernel disagreement, not an internal invariant, so
        this raises ``ValueError`` (not an assertion).
        """
        typed = []
        for j, dt in enumerate(dtypes):
            if isizes[j] != dt.itemsize:
                raise ValueError(
                    f"field {keys[j]!r}: kernel itemsize {isizes[j]} != "
                    f"store dtype {dt} itemsize {dt.itemsize}"
                )
            typed.append(np.asarray(bufs[j], np.uint8).view(dt))
        return typed

    def measure_variant_payload(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Per-instance variant count, ref-window span, and ALT-allele byte sum.

        Runs the SAME per-instance decode (``decode_variants_from_svar2_readbound``)
        that :meth:`_reconstruct_variants`/:meth:`_reconstruct_variant_windows` use, so
        callers that only need cheap per-instance *measures* -- not the full
        reconstructed output -- can share this single counting entry point instead of
        re-deriving the ``p_eff``/fold logic. Used by
        ``Dataset._output_bytes_per_instance`` for the ``"variants"``/``"variant-windows"``
        estimate branches.

        Args:
            idx: Flat ``(region, sample)`` query indices for the block.
            regions: ``(n_regions, 3)`` array of ``(contig_id, start, end)``.

        Returns:
            ``(n_vars_total, ref_span_sum, alt_bytes_sum)``, each shape ``(len(idx),)``
            int64:

            - ``n_vars_total``: total variant count per instance, folded across ploidy
              per the ``unphased_union`` union-count contract (naive sum of
              per-haplotype counts, no dedup) -- the same semantics
              ``Dataset.n_variants()`` documents for the SVAR1 path.
            - ``ref_span_sum``: sum of ``1 + max(-ilen, 0)`` (the ``ref="window"``
              per-variant span) over the instance's variants.
            - ``alt_bytes_sum``: sum of bare ALT-allele byte lengths over the
              instance's variants.
        """
        regions = np.asarray(regions, np.int32)
        P = self.stored_ploidy
        b = len(idx)

        n_vars_total = np.zeros(b, np.int64)
        ref_span_sum = np.zeros(b, np.int64)
        alt_bytes_sum = np.zeros(b, np.int64)

        for ci, qsel, gi in self._gathered_groups(idx, regions, P):
            work_bytes = _range_work_bytes(len(qsel), P, gi)
            pos, ilen, alt_bytes, str_off, var_off, field_bufs, field_isizes = (
                decode_variants_from_svar2_readbound(
                    self.store,
                    self._store_contig(ci),
                    gi[0],
                    gi[1],
                    gi[2],
                    gi[3],
                    gi[4],
                    gi[5],
                    P,
                    [],  # no extra fields needed -- only counts/spans/bytes
                    np.ascontiguousarray(regions[qsel, 2], np.uint32),
                    self.filter == "exonic",
                    should_parallelize(work_bytes),
                )
            )
            ilen = np.asarray(ilen, np.int32)
            str_off = np.asarray(str_off, np.int64)
            var_off = np.asarray(var_off, np.int64)

            # Fold ploidy per the SAME p_eff/union rule the reconstructors apply:
            # var_off[::P] takes each query's FIRST haplotype-row start offset,
            # which (haplotype rows are contiguous per query) equals the start of
            # the whole P-haplotype block -- i.e. the un-deduped union count.
            row_off = (
                np.ascontiguousarray(var_off[::P]) if self.unphased_union else var_off
            )
            row_lens = np.diff(row_off)

            span = (1 + np.maximum(-ilen, 0)).astype(np.int64)
            span_csum = np.concatenate([[np.int64(0)], np.cumsum(span, dtype=np.int64)])
            row_span = span_csum[row_off[1:]] - span_csum[row_off[:-1]]

            bytelen = np.diff(str_off)
            byte_csum = np.concatenate(
                [[np.int64(0)], np.cumsum(bytelen, dtype=np.int64)]
            )
            row_bytes = byte_csum[row_off[1:]] - byte_csum[row_off[:-1]]

            if self.unphased_union:
                n_vars_total[qsel] = row_lens
                ref_span_sum[qsel] = row_span
                alt_bytes_sum[qsel] = row_bytes
            else:
                n_vars_total[qsel] = row_lens.reshape(len(qsel), P).sum(-1)
                ref_span_sum[qsel] = row_span.reshape(len(qsel), P).sum(-1)
                alt_bytes_sum[qsel] = row_bytes.reshape(len(qsel), P).sum(-1)

        return n_vars_total, ref_span_sum, alt_bytes_sum

    def ref_allele_bytes(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> NDArray[np.int64]:
        """Unsupported: the ``.svar2`` store carries no REF allele bytes.

        The read-bound decode is ALT-only, so there is nothing to measure.
        :attr:`has_ref_alleles` is ``False`` and callers must gate on it;
        reaching here means a caller did not.

        Args:
            idx: Unused.
            regions: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "the .svar2 store carries no REF allele bytes (has_ref_alleles is False)"
        )

    def prepare_var_fields(self, var_fields: list[str]) -> "Svar2Haps":
        """Record ``var_fields``; SVAR2 has no storage to pre-load.

        Field values are read on demand by the decode kernel
        (``decode_variants_from_svar2_readbound``) straight from the ``.svar2``
        store, so unlike SVAR1 there are no INFO/dosage/custom-FORMAT columns
        to memmap ahead of time.

        Args:
            var_fields: The variant fields the dataset should emit.

        Returns:
            A new reconstructor with ``var_fields`` set.
        """
        return replace(self, var_fields=var_fields)

    def _reconstruct_variants(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> RaggedVariants:
        """Decode the per-query variant records (SoA) for a query block.

        Args:
            idx: Flat ``(region, sample)`` query indices for the block.
            regions: ``(n_regions, 3)`` array of ``(contig_id, start, end)``.

        Returns:
            RaggedVariants: Per-query ragged variant records (position, indel length, ALT
                bytes, and any requested INFO/FORMAT fields).
        """
        regions = np.asarray(regions, np.int32)
        P = self.stored_ploidy
        b = len(idx)
        p_eff = 1 if self.unphased_union else P

        req_keys, field_specs, field_dtypes = self._requested_store_fields()

        cat_var_lens: list[NDArray[np.int64]] = []
        cat_pos: list[NDArray[np.int32]] = []
        cat_ilen: list[NDArray[np.int32]] = []
        cat_alt: list[NDArray[np.uint8]] = []
        cat_var_bytelen: list[NDArray[np.int64]] = []
        cat_query_order: list[NDArray[np.intp]] = []
        cat_fields: list[list[NDArray]] = []
        for ci, qsel, gi in self._gathered_groups(idx, regions, P):
            work_bytes = _range_work_bytes(len(qsel), P, gi)
            pos, ilen, alt_bytes, str_off, var_off, field_bufs, field_isizes = (
                decode_variants_from_svar2_readbound(
                    self.store,
                    self._store_contig(ci),
                    gi[0],
                    gi[1],
                    gi[2],
                    gi[3],
                    gi[4],
                    gi[5],
                    P,
                    field_specs,
                    np.ascontiguousarray(regions[qsel, 2], np.uint32),
                    self.filter == "exonic",
                    should_parallelize(work_bytes),
                )
            )
            var_off = np.asarray(var_off, np.int64)
            if self.unphased_union:
                var_off = np.ascontiguousarray(var_off[::P])
            str_off = np.asarray(str_off, np.int64)
            cat_var_lens.append(np.diff(var_off))
            cat_pos.append(np.asarray(pos, np.int32))
            cat_ilen.append(np.asarray(ilen, np.int32))
            cat_alt.append(np.asarray(alt_bytes, np.uint8))
            cat_var_bytelen.append(np.diff(str_off))
            cat_query_order.append(qsel)

            cat_fields.append(
                self._type_field_bufs(field_bufs, field_isizes, req_keys, field_dtypes)
            )

        # Single contig group: grouped order already equals global (b, P) order,
        # so the reorder is the identity and every concatenate is a 1-element no-op.
        # Skip both (the numpy reorder otherwise dominates single-contig reads).
        if len(cat_pos) == 1:
            shape = (b, p_eff, None)
            var_off_g = lengths_to_offsets(cat_var_lens[0], np.int64)
            str_off_g = lengths_to_offsets(cat_var_bytelen[0], np.int64)
            extra = {
                k: Ragged.from_offsets(cat_fields[0][j], shape, var_off_g)
                for j, k in enumerate(req_keys)
            }
            return RaggedVariants(
                alt=Ragged.from_offsets(
                    cat_alt[0].view("S1"), shape, var_off_g, str_offsets=str_off_g
                ),
                start=Ragged.from_offsets(cat_pos[0], shape, var_off_g),
                ilen=Ragged.from_offsets(cat_ilen[0], shape, var_off_g),
                **extra,
            )

        # Concatenate grouped outputs, then permute hap-rows back to global order.
        var_lens = (
            np.concatenate(cat_var_lens) if cat_var_lens else np.zeros(0, np.int64)
        )
        grouped_var_off = lengths_to_offsets(var_lens, np.int64)
        pos_c = np.concatenate(cat_pos) if cat_pos else np.zeros(0, np.int32)
        ilen_c = np.concatenate(cat_ilen) if cat_ilen else np.zeros(0, np.int32)
        alt_c = np.concatenate(cat_alt) if cat_alt else np.zeros(0, np.uint8)
        var_bytelen = (
            np.concatenate(cat_var_bytelen)
            if cat_var_bytelen
            else np.zeros(0, np.int64)
        )
        grouped_str_off = lengths_to_offsets(var_bytelen, np.int64)

        perm = self._inverse_row_perm(cat_query_order, b, p_eff)

        src, var_off_g = _ragged_arange_src(grouped_var_off, perm)
        if src.size == 0:
            pos_g = pos_c[:0].copy()
            ilen_g = ilen_c[:0].copy()
        else:
            pos_g = pos_c[src]
            ilen_g = ilen_c[src]
        alt_g, alt_var_off_g, alt_str_off_g = _ragged_arange_gather_2level(
            alt_c, grouped_var_off, grouped_str_off, perm
        )

        shape = (b, p_eff, None)
        pos_r = Ragged.from_offsets(pos_g, shape, var_off_g)
        ilen_r = Ragged.from_offsets(ilen_g, shape, var_off_g)
        alt_r = Ragged.from_offsets(
            alt_g.view("S1"), shape, alt_var_off_g, str_offsets=alt_str_off_g
        )
        extra = {}
        for j, k in enumerate(req_keys):
            fc = (
                np.concatenate([g[j] for g in cat_fields])
                if cat_fields
                else np.zeros(0, field_dtypes[j])
            )
            fg = fc[:0].copy() if src.size == 0 else fc[src]
            extra[k] = Ragged.from_offsets(fg, shape, var_off_g)
        return RaggedVariants(alt=alt_r, start=pos_r, ilen=ilen_r, **extra)

    def _reconstruct_variant_windows(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> _FlatVariantWindows:
        """Variant-windows for svar2: decode variants per contig group, then run the shared ``assemble_variant_buffers`` window kernel over the decoded arrays via an identity gather.

        ``ref="allele"`` is blocked upstream, so ref is always a
        reference-read window; ``alt`` follows ``window_opt.alt``.
        """
        assert self.window_opt is not None and self.token_lut is not None
        assert self.reference is not None

        opt = self.window_opt
        L = opt.flank_length
        ref_mode = 1  # ref="window" (ref="allele" rejected in with_seqs)
        alt_mode = 1 if opt.alt == "window" else 2
        include_ilen = "ilen" in self.var_fields

        regions = np.asarray(regions, np.int32)
        P = self.stored_ploidy
        b = len(idx)

        p_eff = 1 if self.unphased_union else P

        req_keys, field_specs, field_dtypes = self._requested_store_fields()

        cat_row_off: list[NDArray[np.int64]] = []  # per-group var boundaries
        cat_pos: list[NDArray[np.int32]] = []
        cat_ilen: list[NDArray[np.int32]] = []
        cat_query_order: list[NDArray[np.intp]] = []
        cat_fields: list[list[NDArray]] = []
        # name -> per-group (token_data, per-variant seq offsets)
        win_data: dict[str, list[NDArray]] = {}
        win_seq_off: dict[str, list[NDArray[np.int64]]] = {}

        for ci, qsel, gi in self._gathered_groups(idx, regions, P):
            work_bytes = _range_work_bytes(len(qsel), P, gi)
            pos, ilen, alt_bytes, str_off, var_off, field_bufs, field_isizes = (
                decode_variants_from_svar2_readbound(
                    self.store,
                    self._store_contig(ci),
                    gi[0],
                    gi[1],
                    gi[2],
                    gi[3],
                    gi[4],
                    gi[5],
                    P,
                    field_specs,
                    np.ascontiguousarray(regions[qsel, 2], np.uint32),
                    False,
                    should_parallelize(work_bytes),
                )
            )
            pos = np.asarray(pos, np.int32)
            ilen = np.asarray(ilen, np.int32)
            alt_bytes = np.asarray(alt_bytes, np.uint8)
            str_off = np.asarray(str_off, np.int64)
            var_off = np.asarray(var_off, np.int64)

            row_off = (
                np.ascontiguousarray(var_off[::P]) if self.unphased_union else var_off
            )
            n_var = int(len(pos))
            ref_, ref_offsets = self._ref_for_contig(ci)
            bufs = _assemble_variant_buffers_rust(
                1,  # windows mode
                np.arange(n_var, dtype=np.int32),  # identity v_idxs (data pre-gathered)
                row_off,
                alt_bytes,  # alt_global
                str_off,  # alt_off_global
                None,  # ref_global (ref="window")
                None,  # ref_off_global
                False,  # want_ref_bytes
                False,  # want_flank
                ref_mode,
                alt_mode,
                L,
                self.token_lut,
                np.zeros(n_var, np.int32),  # v_contigs (single-contig ref slice)
                pos,  # v_starts
                ilen,  # ilens
                ref_,
                ref_offsets,
                self.reference.pad_char,
            )

            cat_row_off.append(row_off)
            cat_pos.append(pos)
            cat_ilen.append(ilen)
            cat_query_order.append(qsel)
            for name, (data, seq_off) in bufs.items():
                win_data.setdefault(name, []).append(np.asarray(data))
                win_seq_off.setdefault(name, []).append(np.asarray(seq_off, np.int64))

            cat_fields.append(
                self._type_field_bufs(field_bufs, field_isizes, req_keys, field_dtypes)
            )

        shape: tuple[int | None, ...] = (b, p_eff, None)
        wshape: tuple[int | None, ...] = (b, p_eff, None, None)

        # Single contig group: grouped order already equals global (b, p_eff) order.
        if len(cat_pos) == 1:
            row_off = cat_row_off[0]
            fields: dict[str, Any] = {
                "start": _Flat.from_offsets(cat_pos[0], shape, row_off)
            }
            if include_ilen:
                fields["ilen"] = _Flat.from_offsets(cat_ilen[0], shape, row_off)
            for j, k in enumerate(req_keys):
                fields[k] = _Flat.from_offsets(cat_fields[0][j], shape, row_off)
            win = _FlatVariantWindows(fields)
            for name in win_data:
                setattr(
                    win,
                    name,
                    _FlatWindow(
                        win_data[name][0], win_seq_off[name][0], row_off, wshape
                    ),
                )
        else:
            perm = self._inverse_row_perm(cat_query_order, b, p_eff)
            grouped_row_off = lengths_to_offsets(
                np.concatenate([np.diff(r) for r in cat_row_off]), np.int64
            )
            pos_c = np.concatenate(cat_pos)
            ilen_c = np.concatenate(cat_ilen)
            src, row_off_g = _ragged_arange_src(grouped_row_off, perm)
            if src.size == 0:
                pos_g = pos_c[:0].copy()
                ilen_g = ilen_c[:0].copy()
            else:
                pos_g = pos_c[src]
                ilen_g = ilen_c[src]
            fields = {"start": _Flat.from_offsets(pos_g, shape, row_off_g)}
            if include_ilen:
                fields["ilen"] = _Flat.from_offsets(ilen_g, shape, row_off_g)
            for j, k in enumerate(req_keys):
                fc = (
                    np.concatenate([g[j] for g in cat_fields])
                    if cat_fields
                    else np.zeros(0, field_dtypes[j])
                )
                fg = fc[:0].copy() if src.size == 0 else fc[src]
                fields[k] = _Flat.from_offsets(fg, shape, row_off_g)
            win = _FlatVariantWindows(fields)
            for name in win_data:
                data_c = np.concatenate(win_data[name])
                grouped_seq_off = lengths_to_offsets(
                    np.concatenate([np.diff(s) for s in win_seq_off[name]]), np.int64
                )
                nd, nvar, nseq = _ragged_arange_gather_2level(
                    data_c, grouped_row_off, grouped_seq_off, perm
                )
                setattr(win, name, _FlatWindow(nd, nseq, nvar, wshape))

        if self.dummy_variant is not None:
            win = win.fill_empty_groups(
                self.dummy_variant, unk=self.unknown_token, flank_length=L
            )
        return win

    # ---- helpers ----

    def _guard_unsupported(self, splice_plan: "SplicePlan | None") -> None:
        if self.min_af is not None or self.max_af is not None:
            raise NotImplementedError(
                "min_af/max_af filtering is not supported for svar2 datasets yet."
            )
        # No unphased_union guard: variants/variant-windows honor it via the
        # ploidy-1 fold in their reconstructors; haplotypes/annotated + union is
        # blocked upstream in _impl.py, and the haps/track spine ignores union
        # (same as SVAR1), so no path reaches here needing a union guard.

    def _contig_groups(
        self, contig_ids: NDArray[np.int64]
    ) -> list[tuple[int, NDArray[np.intp]]]:
        """Group query positions by contig id (store readers are per-contig).

        Preserves original order within each contig group.
        """
        groups: list[tuple[int, NDArray[np.intp]]] = []
        for ci in np.unique(contig_ids):
            qsel = np.nonzero(contig_ids == ci)[0].astype(np.intp)
            groups.append((int(ci), qsel))
        return groups

    def _gathered_groups(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        P: int,
    ) -> list[_GatheredGroup]:
        """Contig groups paired with their gathered FFI inputs, memoized per batch.

        Every SVAR2 read gathers the same inputs **twice** for one batch: once to
        size the output and once to fill it. On the spliced path that is
        ``build_recon_splice_plan`` -> :meth:`haplotype_lengths_for_plan` ->
        :meth:`_haplotype_diffs`, then :meth:`_reconstruct_spliced`; on the
        unspliced path it is the ``need_diffs`` branch of
        :meth:`get_haps_and_shifts` followed by that method's own reconstruct
        loop. The two calls carry identical ``(idx, regions, P)``, so the second
        gather is pure waste -- ~7% of wall on a production spliced batch
        (issue #349).

        The memo holds exactly one batch. It is keyed by the **contents** of a
        private copy of ``idx``/``regions``, not by their identity: comparing
        ~200 KB costs ~20 us against a gather that costs orders of magnitude
        more, and an identity key would silently serve stale rows to a caller
        that refilled a reused buffer.
        """
        idx = np.asarray(idx)
        regions = np.asarray(regions, np.int32)
        memo = self._gather_memo
        if (
            memo is not None
            and memo[0] == P
            and np.array_equal(memo[1], idx)
            and np.array_equal(memo[2], regions)
        ):
            return memo[3]

        R_all, S_all = self.n_regions, self.n_samples
        r_q, si_q = np.unravel_index(idx, (R_all, S_all))
        groups: list[_GatheredGroup] = [
            (ci, qsel, self._gather_inputs(r_q[qsel], si_q[qsel], regions[qsel], P))
            for ci, qsel in self._contig_groups(regions[:, 0].astype(np.int64))
        ]
        self._gather_memo = (P, idx.copy(), regions.copy(), groups)
        self._rust_gathers = {}
        return groups

    def _readbound_gather(
        self, ci: int, gi: _GatherInputs, P: int
    ) -> Svar2ReadboundGather:
        """The Rust-side read-bound gather for one contig group, memoized per batch.

        ``gather_haps_readbound`` + ``split_to_flat`` + the store's decode LUT is the
        single most expensive thing a read-bound SVAR2 batch does, and the spliced path
        used to run it **twice** over identical ranges: once inside
        ``hap_diffs_from_svar2_readbound`` to size the output, then again inside
        ``reconstruct_haplotypes_from_svar2_readbound_into`` to fill it.

        On an 8192-cell (9.4 MiB) spliced batch that walk cost 12.0 ms per pass against
        5.0 ms for the ``hap_diffs_svar2`` compute it feeds -- the sizing pass is mostly
        gather, not arithmetic. Dropping the duplicate took the batch from 201 ms to
        171 ms single-threaded and 103 ms to 88 ms on 8 threads (~1.18x either way; the
        gather is serial, so the win does not wash out as threads go up), with
        byte-identical output (issue #349).

        The cost is holding one batch's flat channels until the next batch displaces
        them -- memory the old code allocated anyway, just twice and transiently.

        Lifetime is tied to :meth:`_gathered_groups`' memo: that method clears this
        cache on a miss, so a handle can never be served to a different batch. The
        Rust side re-checks ``(n_queries, ploidy)`` and raises rather than
        mis-reconstructing if it ever is.
        """
        g = self._rust_gathers.get(ci)
        if g is None:
            g = self._rust_gathers[ci] = gather_svar2_readbound(
                self.store,
                self._store_contig(ci),
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                P,
                len(gi[6]),
            )
        return g

    def _gather_inputs(
        self,
        r_q: NDArray[np.integer],
        si_q: NDArray[np.integer],
        regions_grp: NDArray[np.int32],
        P: int,
    ) -> _GatherInputs:
        """Cache-slice a per-contig query block into the read-bound FFI inputs.

        Fancy-indexes the memmapped cache (sub-linear; no per-read search). The
        vk_* rows come out ``(n, P, 2)`` -> reshaped ``(n*P, 2)`` in row = q*P+p
        order, which is exactly what the kernel expects.
        """
        c = self.cache
        region_starts = np.ascontiguousarray(regions_grp[:, 1], np.uint32)
        orig_samples = np.ascontiguousarray(c.sample_cols[si_q], np.int64)
        vk_snp = np.ascontiguousarray(
            np.asarray(c.vk_snp_range[r_q, si_q]).reshape(-1, 2), np.int64
        )
        vk_indel = np.ascontiguousarray(
            np.asarray(c.vk_indel_range[r_q, si_q]).reshape(-1, 2), np.int64
        )
        dense_snp = np.ascontiguousarray(np.asarray(c.dense_snp_range[r_q]), np.int64)
        dense_indel = np.ascontiguousarray(
            np.asarray(c.dense_indel_range[r_q]), np.int64
        )
        region_bounds = np.ascontiguousarray(regions_grp[:, 1:3], np.int32)
        return (
            region_starts,
            orig_samples,
            vk_snp,
            vk_indel,
            dense_snp,
            dense_indel,
            region_bounds,
        )

    def _ref_for_contig(
        self, contig_idx: int
    ) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
        """The single-contig reference slice + ``[0, len]`` offsets the kernel wants.

        ``reference.offsets`` is built in ``ds_contigs`` order (Reference.from_path
        was called with the dataset's contigs), so ``contig_idx`` indexes it
        directly.
        """
        ref = self.reference
        assert ref is not None
        o_s = int(ref.offsets[contig_idx])
        o_e = int(ref.offsets[contig_idx + 1])
        ref_ = np.ascontiguousarray(ref.reference[o_s:o_e], np.uint8)
        ref_offsets = np.array([0, o_e - o_s], np.int64)
        return ref_, ref_offsets

    @staticmethod
    def _inverse_row_perm(
        cat_query_order: list[NDArray[np.intp]], b: int, P: int
    ) -> NDArray[np.intp]:
        """Permutation mapping grouped hap-row order back to global (b, P) order.

        ``final_row[k] == grouped_row[perm[k]]``.
        """
        if cat_query_order:
            grouped_queries = np.concatenate(cat_query_order)
        else:
            grouped_queries = np.zeros(0, np.intp)
        grouped_rows = (
            grouped_queries[:, None] * P + np.arange(P, dtype=np.intp)
        ).ravel()
        perm = np.empty(b * P, np.intp)
        perm[grouped_rows] = np.arange(b * P, dtype=np.intp)
        return perm

    def _assemble_haps(
        self,
        cat_data: list[NDArray[np.uint8]],
        cat_hap_lens: list[NDArray[np.int64]],
        cat_query_order: list[NDArray[np.intp]],
        b: int,
        P: int,
    ) -> Ragged[np.bytes_]:
        # Single contig group: grouped hap-row order already equals global (b, P)
        # order (the sole group's qsel is [0..b-1]), so the reorder is the identity
        # and the concatenate is a 1-element no-op. Skip both — this is the common
        # single-contig read, where the O(total_bytes) numpy reorder otherwise
        # dominates (~96% of the call). Byte-identical to the general path.
        if len(cat_data) == 1:
            out_data = cat_data[0]
            out_off = lengths_to_offsets(cat_hap_lens[0], np.int64)
            return cast(
                "Ragged[np.bytes_]",
                _Flat.from_offsets(out_data, (b, P, None), out_off).view("S1"),
            )
        data = np.concatenate(cat_data) if cat_data else np.zeros(0, np.uint8)
        hap_lens = (
            np.concatenate(cat_hap_lens) if cat_hap_lens else np.zeros(0, np.int64)
        )
        grouped_off = lengths_to_offsets(hap_lens, np.int64)
        perm = self._inverse_row_perm(cat_query_order, b, P)
        out_data, out_off = _ragged_arange_gather(data, grouped_off, perm)
        return cast(
            "Ragged[np.bytes_]",
            _Flat.from_offsets(out_data, (b, P, None), out_off).view("S1"),
        )


@dataclass(slots=True)
class _Svar2TrackRealigner:
    """SVAR2 :class:`TrackRealigner`: two standalone kernels per track.

    Thin per-batch holder around :meth:`Svar2Haps.realign_track_block`, which
    does its own per-contig cache slicing and therefore has no batch-scoped
    arrays worth precomputing here.
    """

    haps: "Svar2Haps"
    idx: NDArray[np.integer]
    regions: NDArray[np.int32]
    shifts: NDArray[np.int32]
    track_lengths: NDArray[np.integer]
    base_seed: int

    def fill(
        self,
        out: NDArray[np.float32],
        o_idx: NDArray[np.integer],
        intervals: RaggedIntervals,
        params: NDArray[np.float64],
        strategy_id: int,
    ) -> None:
        """Fill one track's block from the read-bound realign kernels."""
        block_data, _block_off = self.haps.realign_track_block(
            idx=self.idx,
            o_idx=o_idx,
            regions=self.regions,
            shifts=self.shifts,
            track_lengths=self.track_lengths,
            intervals=intervals,
            params=np.ascontiguousarray(params, np.float64),
            strategy_id=int(strategy_id),
            base_seed=int(self.base_seed),
        )
        # block_data is (b, P) C-ordered with per-hap lengths == hap_lengths, so
        # its offsets equal the caller's per-track offsets; copy into the slice.
        out[:] = block_data
