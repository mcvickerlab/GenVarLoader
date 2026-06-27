"""Haplotype reconstructor + supporting value objects.

Houses:

- :class:`_Variants` — internal variant-storage struct.
- :class:`ReconstructionRequest` — per-batch prep state passed to the kernel-
  facing reconstruction methods on :class:`Haps`.
- :class:`Haps` — reconstructs haplotype bytes (and optionally per-nucleotide
  annotations or :class:`RaggedVariants`) from sparse genotypes.
"""

from __future__ import annotations

import json
import warnings
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
from .._ragged import RaggedAnnotatedHaps, RaggedSeqs
from ._flat_variants import _FlatVariantWindows, VarWindowOpt
from .._utils import lengths_to_offsets
from .._variants._records import RaggedAlleles
from ..genvarloader import (
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
from ._utils import _ffi_array
from ._protocol import Reconstructor
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
    :meth:`Haps._prepare_request`; consumed by
    :meth:`Haps._reconstruct_haplotypes` and
    :meth:`Haps._reconstruct_annotated_haplotypes`.

    Decoupled from region-major iteration: a caller (e.g. a future
    variant-major reconstructor) can build a :class:`ReconstructionRequest`
    directly and invoke the kernel-facing methods without going through
    :meth:`Haps.get_haps_and_shifts`.
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
        """
        Loads variant info from a table. Must always have POS, ILEN, and ALT.

        Parameters
        ----------
        path : str | Path
            The path to the variants table.
        one_based : bool, optional
            Whether the variants are one-based, by default False.
        info_fields
            Optional whitelist of numeric column names to load as info.
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
        """Return numeric column names that would be loaded as info, without
        materializing any data.

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
    """genoray custom per-call FORMAT fields: name -> dtype, from <svar>/metadata.json.

    Returns {} when the metadata file is absent (non-SVAR / synthetic datasets).
    """
    meta = svar_dir / "metadata.json"
    if not meta.is_file():
        return {}
    fields = json.loads(meta.read_text()).get("fields", {})
    return {name: np.dtype(dt) for name, dt in fields.items()}


@dataclass(slots=True)
class _HapsFfiStatic:
    """FFI-ready, contiguous, correctly-typed sub-linear arrays consumed by the
    fused kernels. Grows only with the variant/reference count (sub-linear in
    samples), so it is cached for the lifetime of the Haps reconstructor."""

    v_starts: NDArray[np.int32]
    ilens: NDArray[np.int32]
    alt_alleles: NDArray[np.uint8]
    alt_offsets: NDArray[np.int64]
    ref: "NDArray[np.uint8] | None"
    ref_offsets: "NDArray[np.int64] | None"


@dataclass(slots=True)
class Haps(Reconstructor[_H]):
    path: Path
    """The path to the GVL dataset."""
    reference: Reference | None
    """The reference genome. This is kept in memory."""
    variants: _Variants
    """The variant sites in the dataset. This is kept in memory."""
    genotypes: Ragged[V_IDX_TYPE]
    """Shape: (regions, samples, ploidy). The genotypes in the dataset. This is memory mapped."""
    dosages: Ragged[DOSAGE_TYPE] | None
    kind: type[_H]
    filter: Literal["exonic"] | None
    n_variants: NDArray[np.int32] = field(init=False)
    """Shape: (regions, samples, ploidy). The number of variants in the dataset."""
    min_af: float | None
    """The minimum allele frequency to keep."""
    max_af: float | None
    """The maximum allele frequency to keep."""
    var_fields: list[str] = field(default_factory=lambda: ["alt", "ilen", "start"])
    var_field_data: dict[str, Ragged] = field(default_factory=dict)
    """Custom per-call (Number=G) FORMAT fields requested via ``var_fields``,
    memmapped on the genotype offsets. Parallel to ``dosages``. See issue #231."""
    dummy_variant: "DummyVariant | None" = None
    available_var_fields: list[str] = field(init=False)
    _ffi_static: "_HapsFfiStatic | None" = field(default=None, init=False)
    flank_length: int | None = None
    """Number of reference flank bases on each side for flank/window tokenization. ``0``/``None`` disables."""
    token_lut: NDArray | None = None
    """256-entry byte->token lookup table (see ``build_token_lut``). Set together with ``token_dtype``."""
    token_dtype: np.dtype | None = None
    """Output dtype of tokens produced via ``token_lut``."""
    unknown_token: int | None = None
    """Token id for bytes outside ``token_alphabet`` (set with ``token_lut``)."""
    window_opt: VarWindowOpt | None = None
    """Options for variant-windows output mode. Set via ``with_seqs('variant-windows', opt)``."""
    unphased_union: bool = False
    """When True, fold the stored ``ploidy`` haplotypes onto a single haploid sequence
    (union of called ALTs per region/sample) for variant/variant-windows output. Phase is
    discarded; suited to unphased somatic calls. Set via ``with_settings(unphased_union=True)``.
    See issue #222."""

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
        cls: type[Haps[RaggedVariants]],
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
    ) -> Haps[RaggedVariants]:
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
        ploidy = cast(int, self.genotypes.shape[-2])
        return hap_ilens.reshape(-1, ploidy)

    def haplotype_lengths_for_plan(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """Compute ``(B, P)`` per-query haplotype lengths without running the
        full reconstruction. Used by the spliced path to size buffers and
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

    def to_kind(self, kind: type[_NewH]) -> Haps[_NewH]:
        if kind != RaggedVariants and self.reference is None:
            raise ValueError(
                f"Cannot return {kind.__name__}: no reference genome was provided."
            )
        return cast(Haps[_NewH], replace(self, kind=kind))

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
        ploidy = cast(int, self.genotypes.shape[-2])
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
        """Exact total bytes of the selected variants' `kind` allele payload, per
        instance flattened over ploidy.

        Returns shape (len(idx) * ploidy,) of int64. O(|selected variants|);
        does not touch allele payload bytes — only the RaggedAlleles offsets.
        """
        r, s = np.unravel_index(idx, self.genotypes.shape[:2])  # type: ignore[no-matching-overload]
        genos = cast(Ragged[V_IDX_TYPE], self.genotypes[r, s]).to_packed()
        v_idxs = genos.data

        if self.min_af is not None or self.max_af is not None:
            geno_afs = self.variants.info["AF"][v_idxs]
            keep = np.full(len(v_idxs), True, np.bool_)
            if self.min_af is not None:
                keep &= geno_afs >= self.min_af
            if self.max_af is not None:
                keep &= geno_afs <= self.max_af
            # Filter variants per group using the flat boolean mask.
            # Build new offsets via cumsum-indexing (handles empty groups correctly).
            filtered_data = genos.data[keep]
            keep_int = keep.astype(np.int64)
            csum = np.concatenate([[np.int64(0)], np.cumsum(keep_int, dtype=np.int64)])
            new_offsets = csum[np.asarray(genos.offsets, np.int64)]
            genos = Ragged.from_offsets(
                filtered_data, genos.shape, new_offsets
            ).to_packed()
            v_idxs = genos.data

        offsets = getattr(self.variants, kind).offsets  # int-typed, length n_variants+1
        v_lens = (offsets[v_idxs + 1] - offsets[v_idxs]).astype(np.int64)
        # genos.offsets has length b*p + 1 (one offset per (instance, ploid)
        # group). Segment-sum v_lens per group via a cumulative sum: this
        # handles empty groups correctly (np.add.reduceat would index
        # out-of-bounds / mishandle zero-length groups when a group's start
        # offset equals len(v_lens)).
        group_offsets = np.asarray(genos.offsets, dtype=np.int64)
        csum = np.concatenate([[0], np.cumsum(v_lens, dtype=np.int64)])
        return csum[group_offsets[1:]] - csum[group_offsets[:-1]]

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
                    geno_offset_idx=np.ascontiguousarray(
                        req.geno_offset_idx, np.int64
                    ),
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
                flat_shifts=np.ascontiguousarray(
                    flat_shifts.reshape(-1, 1), np.int32
                ),
                flat_geno_offset_idx=np.ascontiguousarray(
                    flat_geno_idx.reshape(-1, 1), np.int64
                ),
                out_offsets=np.ascontiguousarray(off, np.int64),
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
                keep=None
                if keep_perm is None
                else np.ascontiguousarray(keep_perm, np.bool_),
                keep_offsets=None
                if keep_offsets_perm is None
                else np.ascontiguousarray(keep_offsets_perm, np.int64),
                to_rc=_to_rc_spliced,
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
