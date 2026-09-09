"""Haplotype reconstructor role + the SVAR1 implementation of it.

Houses:

- :class:`Haps` — the haplotype-reconstructor *role*: what every backend must
  answer, and nothing about how any of them stores genotypes.
- :class:`Svar1Haps` — the SVAR1 implementation: reconstructs haplotype bytes
  (and optionally per-nucleotide annotations or :class:`RaggedVariants`) from
  sparse per-region genotypes plus an in-memory variant table. The read-bound
  sibling is :class:`Svar2Haps` in ``_svar2_haps.py``.
- :class:`_Variants` — internal variant-storage struct, SVAR1-only.
- :class:`ReconstructionRequest` — per-batch prep state passed to the kernel-
  facing reconstruction methods on :class:`Svar1Haps`.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, cast

if TYPE_CHECKING:
    from ._flat_variants import DummyVariant

import numpy as np
import polars as pl
from genoray._types import DOSAGE_TYPE, POS_TYPE, V_IDX_TYPE
from genoray.exprs import ILEN
from loguru import logger
from numpy.typing import NDArray
from pydantic_extra_types.semantic_version import SemanticVersion
from seqpro.rag import OFFSET_TYPE, Ragged
from typing_extensions import assert_never

from .._flat import _Flat, _FlatAnnotatedHaps
from .._ragged import RaggedAnnotatedHaps, RaggedIntervals, RaggedSeqs
from ._flat_variants import _FlatVariantWindows, VarWindowOpt
from .._utils import lengths_to_offsets
from .._variants._records import RaggedAlleles

# Fused tracks entry: intervals -> scratch -> realign, one FFI crossing.
# Imported at module level so the spy in test_fused_tracks_parity can monkeypatch it.
from ..genvarloader import (
    intervals_and_realign_track_fused as intervals_and_realign_track_fused,
    reconstruct_annotated_haplotypes_fused as reconstruct_annotated_haplotypes_fused,
    reconstruct_annotated_haplotypes_spliced_fused as reconstruct_annotated_haplotypes_spliced_fused,
    reconstruct_haplotypes_fused as reconstruct_haplotypes_fused,
    reconstruct_haplotypes_spliced_fused as reconstruct_haplotypes_spliced_fused,
)
from ._genotypes import (
    _as_starts_stops,
    choose_exonic_variants,
    get_diffs_sparse,
)
from .._threads import should_parallelize
from ._utils import _ffi_array
from ._protocol import Reconstructor, TrackRealigner
from ._rag_variants import RaggedVariants
from ._reference import Reference
from ._splice import SplicePlan
from ._svar_link import SvarLink, _resolve_svar, _verify_fingerprint


@dataclass(frozen=True, slots=True)
class ReconstructionRequest:
    """Per-batch prep state for haplotype reconstruction.

    Describes *what* to reconstruct: which variants apply for each
    ``(region, sample, ploid)`` triple, what shifts to apply, where to write,
    and (optionally) how to splice the output. Produced by
    :meth:`Svar1Haps._prepare_request`; consumed by
    :meth:`Svar1Haps._reconstruct_haplotypes` and
    :meth:`Svar1Haps._reconstruct_annotated_haplotypes`.

    Decoupled from region-major iteration: a caller (e.g. a future
    variant-major reconstructor) can build a :class:`ReconstructionRequest`
    directly and invoke the kernel-facing methods without going through
    :meth:`Svar1Haps.get_haps_and_shifts`.
    """

    geno_offset_idx: NDArray[np.intp]
    """Shape ``(batch, ploidy)``. Indices into the sparse-genotype offsets."""
    regions: NDArray[np.int32]
    """Shape ``(batch, 3)``. Regions ``(contig_idx, start, end)``."""
    shifts: NDArray[np.int32]
    """Shape ``(batch, ploidy)``. Per-haplotype shifts."""
    out_offsets: NDArray[np.integer]
    """Shape ``(batch*ploidy + 1)``. Offsets into the kernel's output buffer.
    For spliced requests this is ``splice_plan.permuted_out_offsets``."""
    diffs: NDArray[np.int32]
    """Shape ``(batch, ploidy)``. Per-haplotype length deltas vs reference."""
    hap_lengths: NDArray[np.int32]
    """Shape ``(batch, ploidy)``. Per-haplotype output lengths."""
    keep: NDArray[np.bool_] | None
    """Optional keep mask (e.g. exonic filter), packed across batch*ploidy."""
    keep_offsets: NDArray[np.integer] | None
    """Offsets matching ``keep``, when ``keep`` is not None."""
    splice_plan: SplicePlan | None
    """If set, reconstruct into a spliced layout."""


@dataclass(slots=True)
class _Variants:
    path: Path
    start: NDArray[POS_TYPE]
    ilen: NDArray[np.int32]
    ref: RaggedAlleles | None
    alt: RaggedAlleles
    info: dict[str, NDArray[np.number]]

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        one_based: bool = True,
        info_fields: set[str] | None = None,
    ):
        """Loads variant info from a table. Must always have POS, ILEN, and ALT.

        Args:
            path (str | Path): The path to the variants table.
            one_based (bool, optional): Whether the variants are one-based, by default False.
            info_fields: Optional whitelist of numeric column names to load as info.
                If ``None`` (default), load every numeric column except POS/ILEN.
        """
        path = Path(path).resolve()
        variants = pl.read_ipc(path, memory_map=False)

        if variants.schema["ALT"] == pl.List(pl.Utf8):
            ilen = ILEN
        else:
            ilen = pl.col("ALT").str.len_bytes().cast(pl.Int32) - pl.col(
                "REF"
            ).str.len_bytes().cast(pl.Int32)

        if "ILEN" not in variants:
            variants = variants.with_columns(ILEN=ilen)

        is_list_type = [
            col for col in ("ALT", "ILEN") if variants[col].dtype == pl.List
        ]
        variants = variants.with_columns(pl.col(is_list_type).list.first())

        info = {
            k: variants[k].to_numpy()
            for k, v in variants.schema.items()
            if v.is_numeric()
            and k not in {"POS", "ILEN"}
            and (info_fields is None or k in info_fields)
        }

        ref = (
            RaggedAlleles.from_polars(variants["REF"])
            if "REF" in variants.schema
            else None
        )

        return cls(
            path,
            variants["POS"].to_numpy() - int(one_based),
            variants["ILEN"].to_numpy(),
            ref,
            RaggedAlleles.from_polars(variants["ALT"]),
            info,
        )

    def __len__(self) -> int:
        return len(self.start)

    @staticmethod
    def available_info_fields(path: str | Path) -> list[str]:
        """Return numeric column names that would be loaded as info, without materializing any data.

        ``POS`` and ``ILEN`` are excluded — they're positional, not info.
        """
        schema = pl.scan_ipc(path).collect_schema()
        return [
            k for k, v in schema.items() if v.is_numeric() and k not in {"POS", "ILEN"}
        ]

    def load_info(self, fields) -> None:
        """Lazily load additional numeric info columns from ``self.path``.

        Fields already present in ``self.info`` are skipped. Unknown numeric
        columns silently no-op (the caller should validate against
        :meth:`available_info_fields` first).
        """
        missing = [f for f in fields if f not in self.info]
        if not missing:
            return
        df = pl.read_ipc(self.path, columns=missing, memory_map=False)
        for f in missing:
            self.info[f] = df[f].to_numpy()


_H = TypeVar("_H", RaggedSeqs, RaggedAnnotatedHaps, RaggedVariants)
_NewH = TypeVar("_NewH", RaggedSeqs, RaggedAnnotatedHaps, RaggedVariants)


def _build_allele_layout(
    data: NDArray[np.uint8],
    allele_offsets: NDArray[np.integer],
    group_offsets: NDArray[np.integer],
    ploidy: int,
) -> Ragged:
    """Wrap flat allele bytes + two offset levels into a (b, p, ~v, ~l) S1 Ragged.

    ``data`` is the contiguous allele byte buffer (uint8). ``allele_offsets`` are the
    per-variant byte boundaries (len n_alleles + 1). ``group_offsets`` are the
    per-(b*p)-row variant boundaries (len b*p + 1). Both offset arrays must be
    zero-based. ``ploidy`` groups the b*p rows into the outer regular axis.
    """
    # rc_ mutates this leaf in place (reverse_complement_masked), so it must be
    # writable; callers may pass a read-only buffer (e.g. np.frombuffer on bytes).
    buf = np.ascontiguousarray(data)
    if not buf.flags.writeable:
        buf = buf.copy()
    n_groups = group_offsets.size - 1
    b = n_groups // ploidy
    return Ragged.from_offsets(
        buf.view("S1"),
        (b, ploidy, None, None),
        [np.asarray(group_offsets, np.int64), np.asarray(allele_offsets, np.int64)],
    )


def _svar_format_fields(svar_dir: Path) -> dict[str, np.dtype]:
    """Genoray custom per-call FORMAT fields: name -> dtype, from <svar>/metadata.json.

    Returns {} when the metadata file is absent (non-SVAR / synthetic datasets).
    """
    meta = svar_dir / "metadata.json"
    if not meta.is_file():
        return {}
    fields = json.loads(meta.read_text()).get("fields", {})
    return {name: np.dtype(dt) for name, dt in fields.items()}


@dataclass(slots=True)
class _HapsFfiStatic:
    """FFI-ready, contiguous, correctly-typed sub-linear arrays consumed by the fused kernels.

    Grows only with the variant/reference count (sub-linear in
    samples), so it is cached for the lifetime of the Svar1Haps reconstructor.
    """

    v_starts: NDArray[np.int32]
    ilens: NDArray[np.int32]
    alt_alleles: NDArray[np.uint8]
    alt_offsets: NDArray[np.int64]
    ref: "NDArray[np.uint8] | None"
    ref_offsets: "NDArray[np.int64] | None"


@dataclass(kw_only=True, slots=True)
class Haps(Reconstructor[_H], ABC):
    """The haplotype-reconstructor *role*: what every backend must answer.

    Holds only what is true of any haplotype reconstructor -- where the dataset
    lives, what it is being asked to produce, and which settings shape the
    output. How the genotypes are stored, and how a haplotype is decoded from
    them, belongs to the implementations: :class:`Svar1Haps` reads a sparse
    per-region genotype array plus an in-memory variant table, while
    :class:`Svar2Haps` decodes read-bound from a ``.svar2`` store and has
    neither.

    Splitting the two is what keeps a caller from reading storage internals off
    whatever reconstructor it was handed and silently getting the wrong answer
    on the other backend (see
    docs/superpowers/specs/2026-09-08-haps-role-split-design.md). Everything a
    caller outside this module needs is declared abstract below; anything it
    cannot get from this class it is not entitled to.

    Fields are keyword-only so implementations can add required fields of their
    own without having to give them defaults just to follow the role's
    defaulted ones.
    """

    path: Path
    """The path to the GVL dataset."""
    reference: Reference | None
    """The reference genome. This is kept in memory."""
    kind: type[_H]
    """What to reconstruct: sequences, annotated haplotypes, or variants."""
    filter: Literal["exonic"] | None
    """Restrict applied variants to those overlapping the query's exons."""
    min_af: float | None
    """The minimum allele frequency to keep."""
    max_af: float | None
    """The maximum allele frequency to keep."""
    n_variants: NDArray[np.int32] = field(init=False)
    """Shape: (regions, samples, ploidy). The number of variants in the dataset."""
    available_var_fields: list[str] = field(init=False)
    """Every variant field this dataset could serve, whether or not it is loaded."""
    var_fields: list[str] = field(default_factory=lambda: ["alt", "ilen", "start"])
    """The variant fields to emit for ``kind=RaggedVariants``."""
    dummy_variant: "DummyVariant | None" = None
    flank_length: int | None = None
    """Number of reference flank bases on each side for flank/window tokenization. ``0``/``None`` disables."""
    token_lut: NDArray | None = None
    """256-entry byte->token lookup table (see ``build_token_lut``). Set together with ``token_dtype``."""
    token_dtype: np.dtype | None = None
    """Output dtype of tokens produced via ``token_lut``."""
    unknown_token: int | None = None
    """Token id for bytes outside ``token_alphabet`` (set with ``token_lut``)."""
    token_alphabet: bytes | None = None
    """The normalized alphabet ``token_lut`` was built from (see
    ``_normalize_token_alphabet``). Set together with ``token_lut``/``token_dtype``/
    ``unknown_token`` so the original alphabet survives alongside the derived LUT
    (needed to serialize a ``with_settings(token_alphabet=...)`` config for
    ``mode='double_buffered'`` without lossily inverting the LUT)."""
    window_opt: VarWindowOpt | None = None
    """Options for variant-windows output mode. Set via ``with_seqs('variant-windows', opt)``."""
    unphased_union: bool = False
    """When True, fold the stored ``ploidy`` haplotypes onto a single haploid sequence
    (union of called ALTs per region/sample) for variant/variant-windows output. Phase is
    discarded; suited to unphased somatic calls. Set via ``with_settings(unphased_union=True)``.
    See issue #222."""

    # ---- backend-agnostic query surface ----
    #
    # These describe the dataset, not the storage layout, so every caller in
    # ``_impl.py`` / ``_reconstruct.py`` can ask a ``Haps`` about it without
    # reaching into one backend's fields. Each member exists because some
    # caller was already doing exactly that.

    @property
    @abstractmethod
    def stored_ploidy(self) -> int:
        """Ploidy as laid out on disk, ignoring any ``unphased_union`` folding.

        Distinct from :attr:`Dataset.ploidy`, which reports ``1`` under
        ``unphased_union``. Grouping that is keyed on the stored layout must use
        this.
        """
        ...

    @property
    @abstractmethod
    def has_ref_alleles(self) -> bool:
        """Whether REF allele bytes are available (needed by ``ref='allele'``)."""
        ...

    @property
    @abstractmethod
    def has_dosages(self) -> bool:
        """Whether per-call dosages can be emitted for ``var_fields=['dosage']``."""
        ...

    @abstractmethod
    def var_field_dtype(self, field: str) -> np.dtype:
        """The numpy dtype of per-variant *scalar* field ``field``.

        Args:
            field: A scalar variant field: ``"start"``, ``"ilen"``, ``"dosage"``,
                or the name of a numeric INFO / per-call FORMAT field.

        Returns:
            The field's numpy dtype.

        Raises:
            KeyError: If ``field`` is unknown, or is one of the variable-length
                allele fields ``"alt"``/``"ref"``, which have no scalar dtype --
                callers size those from their actual byte payload instead.
        """
        ...

    @abstractmethod
    def measure_variant_payload(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Per-instance variant count, ref-window span, and ALT-allele byte sum.

        The single counting entry point behind
        ``Dataset._output_bytes_per_instance``'s ``"variants"`` and
        ``"variant-windows"`` branches, so those branches never have to know
        which backend they are sizing.

        Args:
            idx: Flat ``(region, sample)`` query indices for the block.
            regions: ``(len(idx), 3)`` array of ``(contig_id, start, end)``.

        Returns:
            ``(n_vars_total, ref_span_sum, alt_bytes_sum)``, each shape
            ``(len(idx),)`` int64 and summed over ploidy.
        """
        ...

    @abstractmethod
    def ref_allele_bytes(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> NDArray[np.int64]:
        """Per-instance sum of bare REF allele byte lengths.

        Distinct from :meth:`measure_variant_payload`'s ``ref_span_sum``, which
        measures the ``ref="window"`` reference-genome span. Only callable when
        :attr:`has_ref_alleles` is true; callers must gate on it.

        Args:
            idx: Flat ``(region, sample)`` query indices for the block.
            regions: ``(len(idx), 3)`` array of ``(contig_id, start, end)``.

        Returns:
            Shape ``(len(idx),)`` int64, summed over ploidy.
        """
        ...

    @abstractmethod
    def prepare_var_fields(self, var_fields: list[str]) -> "Haps[_H]":
        """Record ``var_fields``, loading whatever storage they need.

        Callers validate membership in ``available_var_fields`` first.

        Args:
            var_fields: The variant fields the dataset should emit.

        Returns:
            A new reconstructor with ``var_fields`` set and its backing storage
            loaded.
        """
        ...

    @abstractmethod
    def _haplotype_ilens(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        deterministic: bool,
        keep: NDArray[np.bool_] | None = None,
        keep_offsets: NDArray[np.integer] | None = None,
    ) -> NDArray[np.int32]:
        """``(B, P)`` per-haplotype length deltas vs the reference. ``idx`` must be 1D."""
        ...

    @abstractmethod
    def haplotype_lengths_for_plan(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """``(B, P)`` per-query haplotype lengths, without running the full reconstruction.

        Used by the spliced path to size buffers and build a ``SplicePlan``
        before the kernel is invoked.
        """
        ...

    @abstractmethod
    def get_haps_and_shifts(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        output_length: Literal["ragged", "variable"] | int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> tuple[
        _H,
        NDArray[np.intp],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.bool_] | None,
        NDArray[np.int64] | None,
    ]:
        """Reconstruct the batch and return it with the per-batch state tracks need.

        Returns:
            ``(out, geno_offset_idx, shifts, diffs, hap_lengths, keep,
            keep_offsets)``.
        """
        ...

    # ---- haplotype-realigned tracks ----
    #
    # ``HapsTracks.__call__`` runs one body for every backend and reaches the
    # backend only through these two members.

    def check_track_realign_support(
        self,
        output_length: Literal["ragged", "variable"] | int,
        splice_plan: SplicePlan | None,
        to_rc: NDArray[np.bool_] | None,
        ragged_tracks: bool,
    ) -> None:
        """Reject request shapes this backend cannot realign tracks for.

        The base implementation rejects only what no backend can serve;
        override to add further limits.

        Args:
            output_length: The requested output length.
            splice_plan: The requested splice plan, if any.
            to_rc: Per-query reverse-complement mask, if any.
            ragged_tracks: Whether the tracks are being realigned at all
                (False means the caller returns stored intervals untouched, so
                realign-only limits do not apply).

        Raises:
            NotImplementedError: If this backend cannot serve the request.
        """
        del output_length, to_rc, ragged_tracks
        if splice_plan is not None:
            raise NotImplementedError(
                "Splicing of haplotypes + tracks (shape (b, t, p, ~l)) is not "
                "supported."
            )

    @abstractmethod
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
        """Prepare the per-batch state for filling realigned track blocks.

        Args:
            idx: Flat ``(region, sample)`` dataset indices for the batch.
            regions: ``(b, 3)`` contig/start/end of each query.
            shifts: ``(b, p)`` per-haplotype jitter shifts.
            geno_idx: ``(b, p)`` indices into the sparse genotype offsets.
            track_lengths: ``(b,)`` reference span each track block is read over.
            out_offsets: ``(b*p+1,)`` per-haplotype offsets into one track's block.
            keep: Optional per-variant keep mask (``filter='exonic'``).
            keep_offsets: Offsets into ``keep``.
            to_rc: Per-query reverse-complement mask, if any.
            base_seed: Seed for seed-dependent insertion fills.

        Returns:
            A realigner whose ``fill`` writes one track at a time.
        """
        ...

    def to_kind(self, kind: type[_NewH]) -> Haps[_NewH]:
        """A copy of this reconstructor producing ``kind`` instead."""
        if kind != RaggedVariants and self.reference is None:
            raise ValueError(
                f"Cannot return {kind.__name__}: no reference genome was provided."
            )
        return cast(Haps[_NewH], replace(self, kind=kind))


@dataclass(kw_only=True, slots=True)
class Svar1Haps(Haps[_H]):
    """The SVAR1 haplotype reconstructor: sparse per-region genotypes + a variant table.

    Genotypes are stored once per ``(region, sample, ploid)`` as a ragged array
    of variant indices into an in-memory :class:`_Variants` table, so every
    query is a gather over arrays that are already grouped the way the query
    asks for them. :class:`Svar2Haps` is the read-bound sibling; the role they
    share is :class:`Haps`.
    """

    variants: _Variants
    """The variant sites in the dataset. This is kept in memory."""
    genotypes: Ragged[V_IDX_TYPE]
    """Shape: (regions, samples, ploidy). The genotypes in the dataset. This is memory mapped."""
    dosages: Ragged[DOSAGE_TYPE] | None
    var_field_data: dict[str, Ragged] = field(default_factory=dict)
    """Custom per-call (Number=G) FORMAT fields requested via ``var_fields``,
    memmapped on the genotype offsets. Parallel to ``dosages``. See issue #231."""
    _ffi_static: "_HapsFfiStatic | None" = field(default=None, init=False)

    def __post_init__(self):
        self.n_variants = self.genotypes.lengths

        # Discover available info fields from the on-disk schema, not from the
        # (possibly-filtered) loaded info dict. This way the user can see every
        # field they could request, even if only a subset was loaded. Fall back
        # to whatever was loaded if the variants path isn't a readable file
        # (e.g. synthetic in-memory _Variants used by the dummy dataset).
        if self.variants.path.is_file():
            schema_info_fields = _Variants.available_info_fields(self.variants.path)
        else:
            schema_info_fields = list(self.variants.info.keys())
        has_dosage_file = self._has_dosage_file_on_disk()

        custom_fmt = _svar_format_fields(self.variants.path.parent)
        base = (
            ["alt", "ilen", "start"]
            + schema_info_fields
            + (["ref"] if self.variants.ref is not None else [])
            + (["dosage"] if has_dosage_file else [])
        )
        # Per-call FORMAT fields win over a same-named INFO column; list each once.
        self.available_var_fields = base + [f for f in custom_fmt if f not in base]

        if (
            self.min_af is not None or self.max_af is not None
        ) and "AF" not in schema_info_fields:
            raise RuntimeError(
                "Either this dataset is not backed by an SVAR file, or the SVAR file has not had AFs cached yet."
                + "Doing this automatically is not yet supported."
            )

    # ---- backend-agnostic query surface (see Haps) ----

    @property
    def stored_ploidy(self) -> int:
        """Ploidy as laid out on disk, ignoring any ``unphased_union`` folding.

        Distinct from :attr:`Dataset.ploidy`, which reports ``1`` under
        ``unphased_union``. Grouping that is keyed on the stored layout (e.g.
        :meth:`_allele_bytes_sum`'s result shape) must use this.
        """
        return int(self.genotypes.shape[-2])

    @property
    def has_ref_alleles(self) -> bool:
        """Whether REF allele bytes are available (needed by ``ref='allele'``)."""
        return self.variants.ref is not None

    @property
    def has_dosages(self) -> bool:
        """Whether per-call dosages can be emitted for ``var_fields=['dosage']``.

        SVAR1 memmaps ``dosages`` only when the dataset was written with them, so
        this is a storage question the memory estimate has to ask before charging
        for a dosage column.
        """
        return self.dosages is not None

    def var_field_dtype(self, field: str) -> np.dtype:
        """The numpy dtype of per-variant *scalar* field ``field``.

        Args:
            field: A scalar variant field: ``"start"``, ``"ilen"``, ``"dosage"``,
                or the name of a numeric INFO / per-call FORMAT field.

        Returns:
            The field's numpy dtype.

        Raises:
            KeyError: If ``field`` is unknown, or is one of the variable-length
                allele fields ``"alt"``/``"ref"``, which have no scalar dtype --
                callers size those from their actual byte payload instead.
        """
        if field in ("alt", "ref"):
            raise KeyError(
                f"{field!r} is a variable-length allele field with no scalar dtype;"
                " size it from its byte payload instead."
            )
        if field == "start":
            return self.variants.start.dtype
        if field == "ilen":
            return self.variants.ilen.dtype
        if field == "dosage":
            if self.dosages is None:
                raise KeyError("this dataset has no dosages")
            return self.dosages.data.dtype
        try:
            return self.variants.info[field].dtype
        except KeyError:
            raise KeyError(f"unknown variant field {field!r}") from None

    @property
    def ffi_static(self) -> _HapsFfiStatic:
        """Lazily-computed, cached FFI-ready sub-linear arrays (see _HapsFfiStatic)."""
        if self._ffi_static is None:
            ref = self.reference
            self._ffi_static = _HapsFfiStatic(
                v_starts=np.ascontiguousarray(self.variants.start, np.int32),
                ilens=np.ascontiguousarray(self.variants.ilen, np.int32),
                alt_alleles=np.ascontiguousarray(
                    self.variants.alt.data.view(np.uint8), np.uint8
                ),
                alt_offsets=np.ascontiguousarray(self.variants.alt.offsets, np.int64),
                ref=None
                if ref is None
                else np.ascontiguousarray(ref.reference, np.uint8),
                ref_offsets=None
                if ref is None
                else np.ascontiguousarray(ref.offsets, np.int64),
            )
        return self._ffi_static

    def _has_dosage_file_on_disk(self) -> bool:
        """True iff the linked SVAR contains a dosages.npy.

        Returns False for non-SVAR datasets (no dosage path).
        """
        # If we already loaded dosages, we definitely had the file.
        if self.dosages is not None:
            return True
        # Otherwise inspect the SVAR directory next to the variants table.
        # _Variants.path is set to <svar_dir>/index.arrow for SVAR datasets,
        # or <gvl>/genotypes/variants.arrow for legacy. We treat "next-to
        # variants table" as "is dosage possible here".
        candidate = self.variants.path.parent / "dosages.npy"
        return candidate.exists()

    @classmethod
    def from_path(
        cls: type[Svar1Haps[RaggedVariants]],
        path: Path,
        reference: Reference | None,
        regions: NDArray[np.int32],
        samples: list[str],
        ploidy: int,
        version: SemanticVersion | None,
        svar_link: SvarLink | None = None,
        svar_override: Path | str | None = None,
        min_af: float | None = None,
        max_af: float | None = None,
        filter: Literal["exonic"] | None = None,
        var_fields: list[str] | None = None,
    ) -> Svar1Haps[RaggedVariants]:
        # Default var_fields for loading. var_fields=None means "use the default
        # set" — we resolve it here so we know exactly which info columns to load.
        if var_fields is None:
            var_fields = ["alt", "ilen", "start"]
        # Which numeric info columns to eagerly load: those in var_fields that
        # aren't built-ins. (alt/ilen/start/ref/dosage are handled separately.)
        builtin = {"alt", "ilen", "start", "ref", "dosage"}
        info_fields = {f for f in var_fields if f not in builtin}

        svar_meta_path = path / "genotypes" / "svar_meta.json"
        dosages = None
        var_field_data: dict[str, Ragged] = {}

        if svar_meta_path.exists():
            with open(svar_meta_path) as f:
                metadata = json.load(f)
            # (2 r s p)
            shape = cast(tuple[int, ...], tuple(metadata["shape"]))
            dtype = np.dtype(metadata["dtype"])

            offset_path = path / "genotypes" / "offsets.npy"

            if svar_link is not None:
                svar_path = _resolve_svar(path, svar_link, svar_override)
                _verify_fingerprint(svar_path, svar_link)
            else:
                legacy_link = path / "genotypes" / "link.svar"
                if svar_override is not None:
                    svar_path = Path(svar_override)
                    if not svar_path.is_dir():
                        raise FileNotFoundError(
                            f"svar override does not exist: {svar_path}"
                        )
                elif legacy_link.exists():
                    warnings.warn(
                        f"GVL dataset at {path} uses the legacy link.svar "
                        f"symlink. Run "
                        f"`genvarloader.migrate_svar_link({str(path)!r})` "
                        f"to upgrade.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    svar_path = legacy_link.resolve()
                else:
                    raise FileNotFoundError(
                        f"Legacy GVL dataset at {path} is missing its link.svar "
                        f"symlink and has no svar_link metadata. "
                        f"Pass `svar=` to Dataset.open(...) to recover, or "
                        f"re-run `gvl.write`."
                    )

            geno_path = svar_path / "variant_idxs.npy"
            dosage_path = svar_path / "dosages.npy"

            offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
            v_idxs = np.memmap(geno_path, dtype=V_IDX_TYPE, mode="r")
            rag_shape = (*shape[1:], None)
            genotypes = Ragged.from_offsets(v_idxs, rag_shape, offsets.reshape(2, -1))

            if "dosage" in var_fields and dosage_path.exists():
                dosages_mm = np.memmap(dosage_path, dtype=DOSAGE_TYPE, mode="r")
                dosages = Ragged.from_offsets(
                    dosages_mm, rag_shape, offsets.reshape(2, -1)
                )

            custom_fmt = _svar_format_fields(svar_path)
            info_fields = info_fields - set(custom_fmt)
            for name in var_fields:
                if name in custom_fmt:
                    field_mm = np.memmap(
                        svar_path / f"{name}.npy", dtype=custom_fmt[name], mode="r"
                    )
                    var_field_data[name] = Ragged.from_offsets(
                        field_mm, rag_shape, offsets.reshape(2, -1)
                    )

            logger.info("Loading variant data.")
            variants = _Variants.from_table(
                svar_path / "index.arrow", info_fields=info_fields
            )
        else:
            logger.info("Loading variant data.")
            variants = _Variants.from_table(
                path / "genotypes" / "variants.arrow",
                one_based=version is not None
                and version >= SemanticVersion.parse("0.18.0"),
                info_fields=info_fields,
            )
            v_idxs = np.memmap(
                path / "genotypes" / "variant_idxs.npy",
                dtype=V_IDX_TYPE,
                mode="r",
            )
            offsets = np.memmap(
                path / "genotypes" / "offsets.npy", dtype=np.int64, mode="r"
            )
            shape = (len(regions), len(samples), ploidy, None)
            genotypes = Ragged.from_offsets(v_idxs, shape, offsets)

        return cls(
            path=path,
            reference=reference,
            variants=variants,
            genotypes=genotypes,
            dosages=dosages,
            var_field_data=var_field_data,
            kind=RaggedVariants,
            filter=filter,
            min_af=min_af,
            max_af=max_af,
            var_fields=var_fields,
        )

    def _haplotype_ilens(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        deterministic: bool,
        keep: NDArray[np.bool_] | None = None,
        keep_offsets: NDArray[np.integer] | None = None,
    ) -> NDArray[np.int32]:
        """`idx` must be 1D."""
        # (b p)
        geno_offset_idx = self._get_geno_offset_idx(idx, self.genotypes)

        if self.filter == "exonic":
            keep, keep_offsets = choose_exonic_variants(
                starts=regions[:, 1],
                ends=regions[:, 2],
                geno_offset_idx=geno_offset_idx,
                geno_v_idxs=self.genotypes.data,
                geno_offsets=self.genotypes.offsets,
                v_starts=self.variants.start,
                ilens=self.variants.ilen,
            )
        else:
            keep, keep_offsets = None, None

        # (r s p)
        hap_ilens = get_diffs_sparse(
            geno_offset_idx=geno_offset_idx,
            geno_v_idxs=self.genotypes.data,
            geno_offsets=self.genotypes.offsets,
            ilens=self.variants.ilen,
            q_starts=regions[:, 1],
            q_ends=regions[:, 2],
            v_starts=self.variants.start,
            keep=keep,
            keep_offsets=keep_offsets,
        )

        # genotypes are (r, s, p, ~v)
        ploidy = self.stored_ploidy
        return hap_ilens.reshape(-1, ploidy)

    def haplotype_lengths_for_plan(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """Compute ``(B, P)`` per-query haplotype lengths without running the full reconstruction.

        Used by the spliced path to size buffers and
        build a ``SplicePlan`` before the kernel is invoked.

        The body mirrors the length-calculation prefix of
        ``get_haps_and_shifts``: optional exonic filter, then
        ``_haplotype_ilens``, then ``region_length[:, None] + diffs``.
        """
        lengths = regions[:, 2] - regions[:, 1]
        geno_offset_idx = self._get_geno_offset_idx(idx, self.genotypes)
        if self.filter == "exonic":
            keep, keep_offsets = choose_exonic_variants(
                starts=regions[:, 1],
                ends=regions[:, 2],
                geno_offset_idx=geno_offset_idx,
                geno_v_idxs=self.genotypes.data,
                geno_offsets=self.genotypes.offsets,
                v_starts=self.variants.start,
                ilens=self.variants.ilen,
            )
        else:
            keep = None
            keep_offsets = None
        diffs = self._haplotype_ilens(
            idx, regions, deterministic=True, keep=keep, keep_offsets=keep_offsets
        )
        hap_lengths = lengths[:, None] + diffs
        return hap_lengths.astype(np.int32, copy=False)

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        flat: bool = False,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> _H:
        if issubclass(self.kind, (RaggedVariants, _FlatVariantWindows)):
            if splice_plan is not None:
                raise NotImplementedError(
                    "Spliced output is not supported for the 'variants' or"
                    " 'variant-windows' sequence types."
                )
            if issubclass(self.kind, _FlatVariantWindows) and not flat:
                raise ValueError(
                    "with_seqs('variant-windows') requires the flat output format;"
                    " call with_output_format('flat')."
                )
            from ._flat_variants import get_variants_flat

            # `flat` is not checked here: variants always decode flat (the query
            # boundary converts to RaggedVariants when ragged output is requested);
            # the param is retained for protocol/signature stability. `regions` is
            # threaded for flank/window computation. (variant-windows required flat
            # output above.)
            return cast(_H, get_variants_flat(self, idx, regions))
        else:
            haps, *_ = self.get_haps_and_shifts(
                idx=idx,
                regions=regions,
                output_length=output_length,
                rng=rng,
                deterministic=deterministic,
                splice_plan=splice_plan,
                to_rc=to_rc,
            )
            return haps

    def get_haps_and_shifts(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        output_length: Literal["ragged", "variable"] | int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> tuple[
        _H,
        NDArray[np.intp],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.bool_] | None,
        NDArray[np.int64] | None,
    ]:
        req = self._prepare_request(
            idx=idx,
            regions=regions,
            output_length=output_length,
            rng=rng,
            deterministic=deterministic,
            splice_plan=splice_plan,
        )

        # (b p l), (b p l), (b p l)
        if issubclass(self.kind, RaggedSeqs):
            out = self._reconstruct_haplotypes(req, to_rc=to_rc)
        elif issubclass(self.kind, RaggedAnnotatedHaps):
            haps, annot_v_idx, annot_pos = self._reconstruct_annotated_haplotypes(
                req, to_rc=to_rc
            )
            out = _FlatAnnotatedHaps(haps, annot_v_idx, annot_pos)
        elif issubclass(self.kind, RaggedVariants):
            if splice_plan is not None:
                raise NotImplementedError(
                    "Spliced output is not supported for RaggedVariants."
                )
            from ._flat_variants import get_variants_flat

            out = get_variants_flat(self, idx)
        else:
            assert_never(self.kind)

        return (
            out,
            req.geno_offset_idx,
            req.shifts,
            req.diffs,
            req.hap_lengths,
            req.keep,
            req.keep_offsets,
        )

    def _prepare_request(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        output_length: Literal["ragged", "variable"] | int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
    ) -> ReconstructionRequest:
        """Compute the per-batch prep state for haplotype reconstruction."""
        ploidy = self.stored_ploidy
        batch_size = len(idx)
        # (b)
        lengths = regions[:, 2] - regions[:, 1]

        geno_offset_idx = self._get_geno_offset_idx(idx, self.genotypes)

        if self.min_af is not None or self.max_af is not None:
            raise NotImplementedError(
                "Filtering by AF is not supported for haplotype output yet."
            )

        if self.filter == "exonic":
            keep, keep_offsets = choose_exonic_variants(
                starts=regions[:, 1],
                ends=regions[:, 2],
                geno_offset_idx=geno_offset_idx,
                geno_v_idxs=self.genotypes.data,
                geno_offsets=self.genotypes.offsets,
                v_starts=self.variants.start,
                ilens=self.variants.ilen,
            )
        else:
            keep = None
            keep_offsets = None

        # (b p)
        diffs = self._haplotype_ilens(
            idx, regions, deterministic, keep=keep, keep_offsets=keep_offsets
        )
        hap_lengths = lengths[:, None] + diffs

        if deterministic or isinstance(output_length, str):
            # (b p)
            shifts = np.zeros((batch_size, ploidy), dtype=np.int32)
        else:
            # if the haplotype is longer than the region, shift it randomly
            # by up to:
            # the difference in length between the haplotype and the region
            # PLUS the difference in length between the region and the output_length
            max_shift = diffs.clip(min=0)
            max_shift += (lengths - output_length).clip(min=0)[:, None]
            shifts = rng.integers(0, max_shift + 1, dtype=np.int32)

        if not isinstance(output_length, int):
            out_lengths = hap_lengths
        else:
            out_lengths = np.full((batch_size, ploidy), output_length, dtype=np.int32)

        if splice_plan is None:
            # (b*p+1)
            out_offsets = lengths_to_offsets(out_lengths, OFFSET_TYPE)
        else:
            # Plan owns the (permuted) per-element offsets the kernel will use.
            out_offsets = splice_plan.permuted_out_offsets

        return ReconstructionRequest(
            geno_offset_idx=geno_offset_idx,
            regions=regions.astype(np.int32, copy=False),
            shifts=shifts,
            out_offsets=out_offsets,
            diffs=diffs,
            hap_lengths=hap_lengths,
            keep=keep,
            keep_offsets=keep_offsets,
            splice_plan=splice_plan,
        )

    @staticmethod
    def _get_geno_offset_idx(
        idx: NDArray[np.integer],
        genotypes: Ragged[V_IDX_TYPE],
    ) -> NDArray[np.intp]:
        r_idx, s_idx = np.unravel_index(idx, genotypes.shape[:2])  # type: ignore[no-matching-overload]  # Ragged.shape is tuple[int | None, ...]; numpy overload expects all-int
        ploid_idx = np.arange(genotypes.shape[-2], dtype=np.intp)
        # (region, sample, ploid) index tuple for ravel_multi_index.
        region_sample_ploid_idx = (r_idx[:, None], s_idx[:, None], ploid_idx)
        geno_offset_idx = np.ravel_multi_index(
            region_sample_ploid_idx, genotypes.shape[:-1]
        )  # type: ignore[no-matching-overload]  # Ragged.shape is tuple[int | None, ...]; numpy overload expects all-int
        return geno_offset_idx

    def _allele_bytes_sum(
        self, idx: NDArray[np.integer], kind: Literal["alt", "ref"]
    ) -> NDArray[np.int64]:
        """Exact total bytes of the selected variants' `kind` allele payload, per instance flattened over ploidy.

        Returns shape (len(idx) * ploidy,) of int64. O(|selected variants|);
        does not touch allele payload bytes — only the RaggedAlleles offsets.
        """
        v_idxs, group_offsets = self._selected_groups(idx)
        offsets = getattr(self.variants, kind).offsets  # int-typed, length n_variants+1
        v_lens = (offsets[v_idxs + 1] - offsets[v_idxs]).astype(np.int64)
        return self._segment_sum(v_lens, group_offsets)

    def _selected_groups(
        self, idx: NDArray[np.integer]
    ) -> tuple[NDArray[np.integer], NDArray[np.int64]]:
        """The AF-filtered variant indices and per-group offsets for ``idx``.

        Args:
            idx: Flat ``(region, sample)`` query indices for the block.

        Returns:
            ``(v_idxs, group_offsets)``: the selected variant indices, packed
            group-major, and their ``b * ploidy + 1`` group offsets.
        """
        r, s = np.unravel_index(idx, self.genotypes.shape[:2])  # type: ignore[no-matching-overload]
        genos = cast(Ragged[V_IDX_TYPE], self.genotypes[r, s]).to_packed()
        v_idxs = genos.data
        group_offsets = np.asarray(genos.offsets, np.int64)

        if self.min_af is not None or self.max_af is not None:
            geno_afs = self.variants.info["AF"][v_idxs]
            keep = np.full(len(v_idxs), True, np.bool_)
            if self.min_af is not None:
                keep &= geno_afs >= self.min_af
            if self.max_af is not None:
                keep &= geno_afs <= self.max_af
            keep_csum = np.concatenate(
                [[np.int64(0)], np.cumsum(keep.astype(np.int64), dtype=np.int64)]
            )
            group_offsets = keep_csum[group_offsets]
            v_idxs = v_idxs[keep]

        return v_idxs, group_offsets

    @staticmethod
    def _segment_sum(
        per_variant: NDArray[np.int64], group_offsets: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Sum ``per_variant`` within each group delimited by ``group_offsets``.

        Uses cumsum-indexing rather than :func:`np.add.reduceat`, which
        mishandles zero-length groups and indexes out of bounds when a group
        starts at ``len(per_variant)``.

        Args:
            per_variant: One value per selected variant.
            group_offsets: Group boundaries, length ``n_groups + 1``.

        Returns:
            One sum per group, shape ``(n_groups,)`` int64.
        """
        csum = np.concatenate([[np.int64(0)], np.cumsum(per_variant, dtype=np.int64)])
        return csum[group_offsets[1:]] - csum[group_offsets[:-1]]

    def measure_variant_payload(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Per-instance variant count, ref-window span, and ALT-allele byte sum.

        The single counting entry point behind
        ``Dataset._output_bytes_per_instance``'s ``"variants"`` and
        ``"variant-windows"`` branches, so those branches never have to know
        which backend they are sizing. See :class:`Svar2Haps` for the read-bound
        sibling implementation.

        ``ref_span_sum`` and ``alt_bytes_sum`` are taken *after* the
        ``min_af``/``max_af`` filter; ``n_vars_total`` is the *raw* on-disk
        count, matching :meth:`Dataset.n_variants`. That asymmetry is
        deliberate -- it reproduces the accounting
        ``_output_bytes_per_instance`` has always used. Under AF filtering the
        raw count over-charges the scalar fields, and that over-charge is
        currently the only thing covering a constant per-offsets-array deficit
        elsewhere in the estimate (see #362); tightening it here turns
        ``tests/unit/dataset/test_output_bytes_dummy_variant.py`` red.

        Args:
            idx: Flat ``(region, sample)`` query indices for the block.
            regions: ``(len(idx), 3)`` array of ``(contig_id, start, end)``.
                Unused here -- SVAR1 genotypes are already stored per region --
                and accepted only to match the role's signature.

        Returns:
            ``(n_vars_total, ref_span_sum, alt_bytes_sum)``, each shape
            ``(len(idx),)`` int64:

            - ``n_vars_total``: raw on-disk variant count per instance, summed
              over ploidy. Under ``unphased_union`` that naive sum *is* the
              union count contract (no dedup), so no separate fold is needed.
            - ``ref_span_sum``: sum of ``1 + max(-ilen, 0)`` (the ``ref="window"``
              per-variant span) over the instance's variants.
            - ``alt_bytes_sum``: sum of bare ALT-allele byte lengths over the
              instance's variants.
        """
        del regions  # SVAR1 genotypes are already grouped by region
        ploidy = self.stored_ploidy
        b = len(idx)
        v_idxs, group_offsets = self._selected_groups(idx)

        r, s_ = np.unravel_index(idx, self.genotypes.shape[:2])  # type: ignore[no-matching-overload]
        n_vars_total = self.n_variants[r, s_].astype(np.int64).sum(-1)

        ilen_sel = np.asarray(self.variants.ilen)[v_idxs].astype(np.int64)
        ref_span_sum = (
            self._segment_sum(1 + np.maximum(-ilen_sel, 0), group_offsets)
            .reshape(b, ploidy)
            .sum(-1)
        )

        alt_offsets = self.variants.alt.offsets
        alt_lens = (alt_offsets[v_idxs + 1] - alt_offsets[v_idxs]).astype(np.int64)
        alt_bytes_sum = (
            self._segment_sum(alt_lens, group_offsets).reshape(b, ploidy).sum(-1)
        )

        return n_vars_total, ref_span_sum, alt_bytes_sum

    def ref_allele_bytes(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> NDArray[np.int64]:
        """Per-instance sum of bare REF allele byte lengths.

        Distinct from :meth:`measure_variant_payload`'s ``ref_span_sum``, which
        measures the ``ref="window"`` reference-genome span. Only callable when
        :attr:`has_ref_alleles` is true; callers must gate on it.

        Args:
            idx: Flat ``(region, sample)`` query indices for the block.
            regions: ``(len(idx), 3)`` array of ``(contig_id, start, end)``.
                Unused here; accepted to match the role's signature.

        Returns:
            Shape ``(len(idx),)`` int64, summed over ploidy.
        """
        del regions  # SVAR1 genotypes are already grouped by region
        return (
            self._allele_bytes_sum(idx, "ref").reshape(-1, self.stored_ploidy).sum(-1)
        )

    def prepare_var_fields(self, var_fields: list[str]) -> "Svar1Haps[_H]":
        """Record ``var_fields``, loading whatever storage they need.

        SVAR1 keeps its INFO columns, dosages and custom FORMAT fields in
        separate on-disk arrays that are memmapped on demand, so a field
        requested after open time has to be loaded before it can be read.
        Callers validate membership in ``available_var_fields`` first.

        Args:
            var_fields: The variant fields the dataset should emit.

        Returns:
            A new reconstructor with ``var_fields`` set and its backing
            storage loaded.
        """
        # Discover custom FORMAT fields so we don't try to load them as INFO.
        custom_fmt = _svar_format_fields(self.variants.path.parent)
        # Lazily load any newly-requested info columns into the existing
        # _Variants struct (mutates self.variants.info in place).
        builtin = {"alt", "ilen", "start", "ref", "dosage"}
        new_info_fields = [
            f
            for f in var_fields
            if f not in builtin and f not in self.variants.info and f not in custom_fmt
        ]
        if new_info_fields:
            self.variants.load_info(new_info_fields)

        haps = self
        # Lazily memmap dosages if newly requested.
        if "dosage" in var_fields and haps.dosages is None:
            haps = _lazy_load_dosages(haps)
        # Lazily memmap custom FORMAT fields if newly requested.
        new_custom_fields = {
            f: custom_fmt[f]
            for f in var_fields
            if f in custom_fmt and f not in haps.var_field_data
        }
        if new_custom_fields:
            haps = _lazy_load_custom_fields(haps, new_custom_fields)
        return replace(haps, var_fields=var_fields)

    # ---- haplotype-realigned tracks ----

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
        """Prepare the per-batch state for filling realigned track blocks.

        Args:
            idx: Flat ``(region, sample)`` dataset indices for the batch.
            regions: ``(b, 3)`` contig/start/end of each query.
            shifts: ``(b, p)`` per-haplotype jitter shifts.
            geno_idx: ``(b, p)`` indices into the sparse genotype offsets.
            track_lengths: ``(b,)`` reference span each track block is read over.
            out_offsets: ``(b*p+1,)`` per-haplotype offsets into one track's block.
            keep: Optional per-variant keep mask (``filter='exonic'``).
            keep_offsets: Offsets into ``keep``.
            to_rc: Per-query reverse-complement mask, if any.
            base_seed: Seed for seed-dependent insertion fills.

        Returns:
            A realigner whose ``fill`` writes one track at a time.
        """
        del idx  # SVAR1 addresses each track by ``o_idx`` alone
        return _Svar1TrackRealigner(
            haps=self,
            regions=np.ascontiguousarray(regions, np.int32),
            shifts=np.ascontiguousarray(shifts, np.int32),
            geno_offset_idx=np.ascontiguousarray(geno_idx, np.int64),
            # Materialized once per batch rather than once per track: it is
            # (2, regions*samples*ploidy). Dataset-scoped in principle, but
            # caching it on the reconstructor would pin a per-sample-scale
            # array for the dataset's lifetime.
            geno_offsets=_as_starts_stops(self.genotypes.offsets),
            out_offsets=np.ascontiguousarray(out_offsets, np.int64),
            track_offsets=np.ascontiguousarray(
                lengths_to_offsets(track_lengths), np.int64
            ),
            keep=None if keep is None else np.ascontiguousarray(keep, np.bool_),
            keep_offsets=None
            if keep_offsets is None
            else np.ascontiguousarray(keep_offsets, np.int64),
            # Expand per-query to_rc to per-(query, hap) for the track kernel.
            to_rc=None
            if to_rc is None
            else np.ascontiguousarray(np.repeat(to_rc, geno_idx.shape[-1]), np.bool_),
            base_seed=base_seed,
        )

    def _reconstruct_haplotypes(
        self,
        req: ReconstructionRequest,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> Ragged[np.bytes_]:
        """Reconstruct haplotype byte sequences from sparse genotypes."""
        assert self.reference is not None

        if req.splice_plan is None:
            shape = (*req.shifts.shape, None)
            # --- fused path (Rust): one FFI crossing, no Python-side np.empty ---
            # Detect ragged vs fixed-length output from req.out_offsets.
            # Ragged: out_lengths == hap_lengths (per-hap variable length).
            # Fixed:  out_lengths is all the same constant value.
            _out_per = (req.out_offsets[1:] - req.out_offsets[:-1]).reshape(
                req.shifts.shape
            )
            if np.array_equal(
                _out_per.astype(np.int64), req.hap_lengths.astype(np.int64)
            ):
                _fused_output_length = np.int64(-1)  # ragged mode
            else:
                _fused_output_length = np.int64(
                    int(req.out_offsets[1] - req.out_offsets[0])
                )
            # Expand per-query to_rc → per-(query, hap) for the fused kernel.
            # req.shifts.shape == (b, ploidy); np.repeat broadcasts (b,) → (b*p,).
            _ploidy = req.shifts.shape[1] if req.shifts.ndim > 1 else 1
            _to_rc_hap = (
                None
                if to_rc is None
                else np.ascontiguousarray(np.repeat(to_rc, _ploidy), np.bool_)
            )
            out_data, out_offsets = reconstruct_haplotypes_fused(
                regions=np.ascontiguousarray(req.regions, np.int32),
                shifts=np.ascontiguousarray(req.shifts, np.int32),
                geno_offset_idx=np.ascontiguousarray(req.geno_offset_idx, np.int64),
                geno_offsets=_as_starts_stops(self.genotypes.offsets),
                geno_v_idxs=_ffi_array(self.genotypes.data, np.int32, "geno_v_idxs"),
                v_starts=self.ffi_static.v_starts,
                ilens=self.ffi_static.ilens,
                alt_alleles=self.ffi_static.alt_alleles,
                alt_offsets=self.ffi_static.alt_offsets,
                ref_=self.ffi_static.ref,
                ref_offsets=self.ffi_static.ref_offsets,
                pad_char=np.uint8(self.reference.pad_char),
                output_length=_fused_output_length,
                keep=None
                if req.keep is None
                else np.ascontiguousarray(req.keep, np.bool_),
                keep_offsets=None
                if req.keep_offsets is None
                else np.ascontiguousarray(req.keep_offsets, np.int64),
                to_rc=_to_rc_hap,
                parallel=should_parallelize(int(req.out_offsets[-1])),
            )
            return cast(
                "Ragged[np.bytes_]",
                _Flat.from_offsets(out_data, shape, out_offsets).view("S1"),
            )

        # ---- splice plan path ----
        flat_geno_idx, flat_shifts, permuted_regions, keep_perm, keep_offsets_perm = (
            self._permute_request_for_splice(req)
        )
        splice_plan = req.splice_plan

        per_elem_shape = (splice_plan.permuted_lengths.shape[0], None)

        # Fused path (Rust): one FFI crossing, Python already holds out_offsets.
        # to_rc is already in permuted per-element order (passed from
        # _getitem_spliced as to_rc_per_elem = to_rc_flat[plan.permutation]).
        _to_rc_spliced = (
            None if to_rc is None else np.ascontiguousarray(to_rc, np.bool_)
        )
        out_buf = reconstruct_haplotypes_spliced_fused(
            permuted_regions=np.ascontiguousarray(permuted_regions, np.int32),
            flat_shifts=np.ascontiguousarray(flat_shifts.reshape(-1, 1), np.int32),
            flat_geno_offset_idx=np.ascontiguousarray(
                flat_geno_idx.reshape(-1, 1), np.int64
            ),
            out_offsets=np.ascontiguousarray(
                splice_plan.permuted_out_offsets, np.int64
            ),
            geno_offsets=_as_starts_stops(self.genotypes.offsets),
            geno_v_idxs=_ffi_array(self.genotypes.data, np.int32, "geno_v_idxs"),
            v_starts=self.ffi_static.v_starts,
            ilens=self.ffi_static.ilens,
            alt_alleles=self.ffi_static.alt_alleles,
            alt_offsets=self.ffi_static.alt_offsets,
            ref_=self.ffi_static.ref,
            ref_offsets=self.ffi_static.ref_offsets,
            pad_char=np.uint8(self.reference.pad_char),
            keep=None
            if keep_perm is None
            else np.ascontiguousarray(keep_perm, np.bool_),
            keep_offsets=None
            if keep_offsets_perm is None
            else np.ascontiguousarray(keep_offsets_perm, np.int64),
            to_rc=_to_rc_spliced,
            parallel=should_parallelize(int(splice_plan.permuted_out_offsets[-1])),
        )

        return cast(
            "Ragged[np.bytes_]",
            _Flat.from_offsets(
                out_buf, per_elem_shape, splice_plan.permuted_out_offsets
            ).view("S1"),
        )

    def _reconstruct_annotated_haplotypes(
        self,
        req: ReconstructionRequest,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> tuple[Ragged[np.bytes_], Ragged[V_IDX_TYPE], Ragged[np.int32]]:
        """Reconstruct haplotypes plus per-nucleotide annotations.

        Returns the haplotype bytes, the variant index at each position
        (or -1 for reference), and the reference coordinate at each position
        (or -1 for padded bases).
        """
        assert self.reference is not None

        if req.splice_plan is None:
            shape = (*req.shifts.shape, None)
            # --- fused path (Rust): one FFI crossing, no Python-side np.empty ---
            # Detect ragged vs fixed-length output from req.out_offsets.
            # Ragged: out_lengths == hap_lengths (per-hap variable length).
            # Fixed:  out_lengths is all the same constant value.
            _out_per = (req.out_offsets[1:] - req.out_offsets[:-1]).reshape(
                req.shifts.shape
            )
            if np.array_equal(
                _out_per.astype(np.int64), req.hap_lengths.astype(np.int64)
            ):
                _fused_output_length = np.int64(-1)  # ragged mode
            else:
                _fused_output_length = np.int64(
                    int(req.out_offsets[1] - req.out_offsets[0])
                )
            # Expand per-query to_rc → per-(query, hap) for the fused kernel.
            _ploidy = req.shifts.shape[1] if req.shifts.ndim > 1 else 1
            _to_rc_hap = (
                None
                if to_rc is None
                else np.ascontiguousarray(np.repeat(to_rc, _ploidy), np.bool_)
            )
            out_data, annot_v_data, annot_pos_data, out_offsets = (
                reconstruct_annotated_haplotypes_fused(
                    regions=np.ascontiguousarray(req.regions, np.int32),
                    shifts=np.ascontiguousarray(req.shifts, np.int32),
                    geno_offset_idx=np.ascontiguousarray(req.geno_offset_idx, np.int64),
                    geno_offsets=_as_starts_stops(self.genotypes.offsets),
                    geno_v_idxs=_ffi_array(
                        self.genotypes.data, np.int32, "geno_v_idxs"
                    ),
                    v_starts=self.ffi_static.v_starts,
                    ilens=self.ffi_static.ilens,
                    alt_alleles=self.ffi_static.alt_alleles,
                    alt_offsets=self.ffi_static.alt_offsets,
                    ref_=self.ffi_static.ref,
                    ref_offsets=self.ffi_static.ref_offsets,
                    pad_char=np.uint8(self.reference.pad_char),
                    output_length=_fused_output_length,
                    keep=None
                    if req.keep is None
                    else np.ascontiguousarray(req.keep, np.bool_),
                    keep_offsets=None
                    if req.keep_offsets is None
                    else np.ascontiguousarray(req.keep_offsets, np.int64),
                    to_rc=_to_rc_hap,
                    parallel=should_parallelize(int(req.out_offsets[-1])),
                )
            )
            return (
                cast(
                    "Ragged[np.bytes_]",
                    _Flat.from_offsets(out_data, shape, out_offsets).view("S1"),
                ),
                cast(
                    "Ragged[V_IDX_TYPE]",
                    _Flat.from_offsets(annot_v_data, shape, out_offsets),
                ),
                cast(
                    "Ragged[np.int32]",
                    _Flat.from_offsets(annot_pos_data, shape, out_offsets),
                ),
            )

        # ---- splice plan path ----
        flat_geno_idx, flat_shifts, permuted_regions, keep_perm, keep_offsets_perm = (
            self._permute_request_for_splice(req)
        )
        splice_plan = req.splice_plan
        per_elem_shape = (splice_plan.permuted_lengths.shape[0], None)
        off = splice_plan.permuted_out_offsets

        # Fused path (Rust): one FFI crossing. RC is folded in-kernel (sequence bytes
        # reverse-complemented, annotation rows reversed), so there is NO Python
        # reverse_masked post-pass. to_rc is already in permuted per-element order
        # (from _getitem_spliced), and _getitem_spliced treats the rust output as
        # already-RC'd (its post-pass is numba-only).
        _to_rc_spliced = (
            None if to_rc is None else np.ascontiguousarray(to_rc, np.bool_)
        )
        out_buf, annot_v_buf, annot_pos_buf = (
            reconstruct_annotated_haplotypes_spliced_fused(
                permuted_regions=np.ascontiguousarray(permuted_regions, np.int32),
                flat_shifts=np.ascontiguousarray(flat_shifts.reshape(-1, 1), np.int32),
                flat_geno_offset_idx=np.ascontiguousarray(
                    flat_geno_idx.reshape(-1, 1), np.int64
                ),
                out_offsets=np.ascontiguousarray(off, np.int64),
                geno_offsets=_as_starts_stops(self.genotypes.offsets),
                geno_v_idxs=_ffi_array(self.genotypes.data, np.int32, "geno_v_idxs"),
                v_starts=self.ffi_static.v_starts,
                ilens=self.ffi_static.ilens,
                alt_alleles=self.ffi_static.alt_alleles,
                alt_offsets=self.ffi_static.alt_offsets,
                ref_=self.ffi_static.ref,
                ref_offsets=self.ffi_static.ref_offsets,
                pad_char=np.uint8(self.reference.pad_char),
                keep=None
                if keep_perm is None
                else np.ascontiguousarray(keep_perm, np.bool_),
                keep_offsets=None
                if keep_offsets_perm is None
                else np.ascontiguousarray(keep_offsets_perm, np.int64),
                to_rc=_to_rc_spliced,
                parallel=should_parallelize(int(off[-1])),
            )
        )

        haps_rag = cast(
            "Ragged[np.bytes_]",
            _Flat.from_offsets(out_buf, per_elem_shape, off).view("S1"),
        )
        annot_v_rag = cast(
            "Ragged[V_IDX_TYPE]",
            _Flat.from_offsets(annot_v_buf, per_elem_shape, off),
        )
        annot_pos_rag = cast(
            "Ragged[np.int32]",
            _Flat.from_offsets(annot_pos_buf, per_elem_shape, off),
        )
        return haps_rag, annot_v_rag, annot_pos_rag

    def _permute_request_for_splice(
        self, req: ReconstructionRequest
    ) -> tuple[
        NDArray[np.intp],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.bool_] | None,
        NDArray[np.integer] | None,
    ]:
        """Permute the per-element arrays in ``req`` according to ``splice_plan.permutation``.

        ``geno_offset_idx`` and ``shifts`` have shape ``(B, P)``; flatten to
        ``(B*P,)`` in (query, ploidy) C-order, then permute. The kernel then
        runs with ploidy=1 over the ``B*P`` flattened queries.
        """
        assert req.splice_plan is not None
        splice_plan = req.splice_plan
        ploidy = req.shifts.shape[1] if req.shifts.ndim > 1 else 1
        permutation = splice_plan.permutation

        flat_geno_idx = req.geno_offset_idx.reshape(-1)[permutation].astype(
            np.intp, copy=False
        )
        flat_shifts = req.shifts.reshape(-1)[permutation].astype(np.int32, copy=False)
        # regions has shape (B, 3). For (B*P, 3), each query repeats P times
        # consecutively, then we apply the same permutation.
        regions_flat = np.repeat(req.regions, ploidy, axis=0)
        permuted_regions = regions_flat[permutation]

        # keep / keep_offsets: per-k granularity (length B*P + 1).
        if req.keep is not None and req.keep_offsets is not None:
            keep_lens = np.diff(req.keep_offsets)
            keep_lens_perm = keep_lens[permutation]
            keep_offsets_perm = lengths_to_offsets(
                keep_lens_perm.astype(np.int64), dtype=np.int64
            )
            keep_perm = np.empty(int(keep_lens_perm.sum()), dtype=np.bool_)
            write_cursor = 0
            for k_old in permutation:
                s = int(req.keep_offsets[k_old])
                e = int(req.keep_offsets[k_old + 1])
                width = e - s
                keep_perm[write_cursor : write_cursor + width] = req.keep[s:e]
                write_cursor += width
        else:
            keep_perm = None
            keep_offsets_perm = None

        return (
            flat_geno_idx,
            flat_shifts,
            permuted_regions,
            keep_perm,
            keep_offsets_perm,
        )


@dataclass(slots=True)
class _Svar1TrackRealigner:
    """SVAR1 :class:`TrackRealigner`: one fused FFI crossing per track.

    Every array here is FFI-ready, so the per-track :meth:`fill` does no
    conversion work that does not depend on the track. ``geno_offsets`` in
    particular is ``(2, regions*samples*ploidy)``; materializing it once per
    track instead of once per batch was measurably wasteful.
    """

    haps: "Svar1Haps"
    regions: NDArray[np.int32]
    shifts: NDArray[np.int32]
    geno_offset_idx: NDArray[np.int64]
    geno_offsets: NDArray[np.int64]
    out_offsets: NDArray[np.int64]
    track_offsets: NDArray[np.int64]
    keep: NDArray[np.bool_] | None
    keep_offsets: NDArray[np.int64] | None
    to_rc: NDArray[np.bool_] | None
    base_seed: int

    def fill(
        self,
        out: NDArray[np.float32],
        o_idx: NDArray[np.integer],
        intervals: RaggedIntervals,
        params: NDArray[np.float64],
        strategy_id: int,
    ) -> None:
        """Fill one track's block via the fused intervals->realign kernel.

        Replaces the unfused ``np.empty`` scratch + ``intervals_to_tracks`` +
        ``shift_and_realign_tracks_sparse`` sequence with a single FFI crossing
        that writes in place into ``out`` (a contiguous slice of the caller's
        pre-allocated buffer, so no ``ascontiguousarray`` is needed for it).
        """
        stat = self.haps.ffi_static
        intervals_and_realign_track_fused(
            out=out,
            out_offsets=self.out_offsets,
            regions=self.regions,
            shifts=self.shifts,
            geno_offset_idx=self.geno_offset_idx,
            geno_v_idxs=_ffi_array(self.haps.genotypes.data, np.int32, "geno_v_idxs"),
            geno_offsets=self.geno_offsets,
            v_starts=stat.v_starts,
            ilens=stat.ilens,
            offset_idxs=np.ascontiguousarray(o_idx, np.int64),
            itv_starts=_ffi_array(intervals.starts.data, np.int32, "itv_starts"),
            itv_ends=_ffi_array(intervals.ends.data, np.int32, "itv_ends"),
            itv_values=_ffi_array(intervals.values.data, np.float32, "itv_values"),
            itv_offsets=_ffi_array(intervals.starts.offsets, np.int64, "itv_offsets"),
            track_offsets=self.track_offsets,
            params=np.ascontiguousarray(params, np.float64),
            strategy_id=int(strategy_id),
            base_seed=int(self.base_seed),
            keep=self.keep,
            keep_offsets=self.keep_offsets,
            to_rc=self.to_rc,
            parallel=should_parallelize(int(self.out_offsets[-1]) * 4),
        )


def _lazy_load_dosages(haps: Svar1Haps) -> Svar1Haps:
    """Open the dosages memmap for a Svar1Haps that didn't request them at open time.

    Reuses the same path-resolution logic that ``Svar1Haps.from_path`` used. Returns
    a new ``Svar1Haps`` with ``dosages`` populated (does NOT mutate the input).
    """
    import json as _json

    from genoray._types import DOSAGE_TYPE

    from ._svar_link import _resolve_svar
    from ._write import Metadata

    path = haps.path
    svar_meta_path = path / "genotypes" / "svar_meta.json"
    if not svar_meta_path.exists():
        raise ValueError(
            "Dosage requested but this dataset is not SVAR-backed; no dosages.npy possible."
        )

    with open(svar_meta_path) as f:
        svar_meta = _json.load(f)
    shape = tuple(svar_meta["shape"])
    dtype = np.dtype(svar_meta["dtype"])

    offset_path = path / "genotypes" / "offsets.npy"

    # Resolve the SVAR directory the same way Svar1Haps.from_path did. Dataset does
    # not retain Metadata, so re-read metadata.json from disk.
    meta = Metadata.model_validate_json((path / "metadata.json").read_text())
    svar_link = meta.svar_link
    if svar_link is not None:
        svar_path = _resolve_svar(path, svar_link, None)
    else:
        legacy_link = path / "genotypes" / "link.svar"
        svar_path = legacy_link.resolve()

    dosage_path = svar_path / "dosages.npy"
    if not dosage_path.exists():
        raise ValueError(
            f"Dosage requested but {dosage_path} does not exist. "
            f"Check the SVAR was built with dosages."
        )

    offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
    dosages_mm = np.memmap(dosage_path, dtype=DOSAGE_TYPE, mode="r")
    rag_shape = (*shape[1:], None)
    dosages = Ragged.from_offsets(dosages_mm, rag_shape, offsets.reshape(2, -1))
    return replace(haps, dosages=dosages)


def _lazy_load_custom_fields(
    haps: Svar1Haps,
    new_fields: dict[str, np.dtype],
) -> Svar1Haps:
    """Memmap custom FORMAT fields (Number=G, stored as <name>.npy) into ``haps.var_field_data`` for fields that were not loaded at open time.

    ``new_fields`` maps field name → numpy dtype (already confirmed present in
    the SVAR metadata). Returns a new ``Svar1Haps`` with updated ``var_field_data``.
    """
    import json as _json

    path = haps.path
    svar_meta_path = path / "genotypes" / "svar_meta.json"
    if not svar_meta_path.exists():
        raise ValueError(
            "Custom FORMAT fields requested but this dataset is not SVAR-backed."
        )

    with open(svar_meta_path) as f:
        svar_meta = _json.load(f)
    shape = tuple(svar_meta["shape"])
    dtype = np.dtype(svar_meta["dtype"])

    offset_path = path / "genotypes" / "offsets.npy"

    # The resolved SVAR directory is already embedded in haps.variants.path
    # (which was set to <svar_path>/index.arrow by Svar1Haps.from_path, respecting any
    # svar_override). Using .parent avoids re-resolving from metadata and correctly
    # handles the svar_override case that the legacy link.svar branch would miss.
    svar_path = haps.variants.path.parent

    offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
    rag_shape = (*shape[1:], None)

    updated_var_field_data = dict(haps.var_field_data)
    for name, ftype in new_fields.items():
        field_path = svar_path / f"{name}.npy"
        if not field_path.exists():
            raise ValueError(
                f"Custom FORMAT field '{name}' registered in SVAR metadata but "
                f"{field_path} does not exist."
            )
        field_mm = np.memmap(field_path, dtype=ftype, mode="r")
        updated_var_field_data[name] = Ragged.from_offsets(
            field_mm, rag_shape, offsets.reshape(2, -1)
        )
    return replace(haps, var_field_data=updated_var_field_data)
