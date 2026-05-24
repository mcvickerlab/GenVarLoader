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
from typing import Literal, TypeVar, cast

import awkward as ak
import numpy as np
import polars as pl
from awkward.contents import ListOffsetArray, NumpyArray, RegularArray
from awkward.index import Index
from genoray._types import DOSAGE_TYPE, POS_TYPE, V_IDX_TYPE
from genoray.exprs import ILEN
from loguru import logger
from numpy.typing import NDArray
from pydantic_extra_types.semantic_version import SemanticVersion
from seqpro.rag import OFFSET_TYPE, Ragged
from typing_extensions import assert_never

from .._ragged import RaggedAnnotatedHaps, RaggedSeqs
from .._utils import lengths_to_offsets
from .._variants._records import RaggedAlleles
from ._genotypes import (
    choose_exonic_variants,
    get_diffs_sparse,
    reconstruct_haplotypes_from_sparse,
)
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
    def from_table(cls, path: str | Path, one_based: bool = True):
        """
        Loads variant info from a table. Must always have POS, ILEN, and ALT.
        Any numeric columns will be loaded as info.

        Parameters
        ----------
        path : str | Path
            The path to the variants table or a polars DataFrame.
        one_based : bool, optional
            Whether the variants are one-based, by default False.
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
            if v.is_numeric() and k not in {"POS", "ILEN"}
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


_H = TypeVar("_H", RaggedSeqs, RaggedAnnotatedHaps, RaggedVariants)
_NewH = TypeVar("_NewH", RaggedSeqs, RaggedAnnotatedHaps, RaggedVariants)


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
    available_var_fields: list[str] = field(init=False)

    def __post_init__(self):
        self.n_variants = ak.num(self.genotypes, -1).to_numpy()
        self.available_var_fields = (
            ["alt", "ilen", "start"]
            + list(self.variants.info.keys())
            + (["ref"] if self.variants.ref is not None else [])
        )

        if (
            self.min_af is not None or self.max_af is not None
        ) and "AF" not in self.variants.info:
            raise RuntimeError(
                "Either this dataset is not backed by an SVAR file, or the SVAR file has not had AFs cached yet."
                + "Doing this automatically is not yet supported."
            )

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
    ) -> Haps[RaggedVariants]:
        svar_meta_path = path / "genotypes" / "svar_meta.json"
        dosages = None

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

            if dosage_path.exists():
                dosages = np.memmap(dosage_path, dtype=DOSAGE_TYPE, mode="r")
                dosages = Ragged.from_offsets(
                    dosages, rag_shape, offsets.reshape(2, -1)
                )

            logger.info("Loading variant data.")
            variants = _Variants.from_table(svar_path / "index.arrow")
        else:
            logger.info("Loading variant data.")
            variants = _Variants.from_table(
                path / "genotypes" / "variants.arrow",
                one_based=version is not None
                and version >= SemanticVersion.parse("0.18.0"),
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
            kind=RaggedVariants,
            filter=filter,
            min_af=min_af,
            max_af=max_af,
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
        geno_offset_idxs = self._get_geno_offset_idx(idx, self.genotypes)

        if self.filter == "exonic":
            keep, keep_offsets = choose_exonic_variants(
                starts=regions[:, 1],
                ends=regions[:, 2],
                geno_offset_idxs=geno_offset_idxs,
                geno_v_idxs=self.genotypes.data,
                geno_offsets=self.genotypes.offsets,
                v_starts=self.variants.start,
                ilens=self.variants.ilen,
            )
        else:
            keep, keep_offsets = None, None

        # (r s p)
        hap_ilens = get_diffs_sparse(
            geno_offset_idxs=geno_offset_idxs,
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
                geno_offset_idxs=geno_offset_idx,
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
    ) -> _H:
        if issubclass(self.kind, RaggedVariants):
            if splice_plan is not None:
                raise NotImplementedError(
                    "Spliced output is not supported for RaggedVariants."
                )
            ragv = self._get_variants(idx=idx, regions=None, shifts=None)
            ragv = cast(_H, ragv)
            return ragv
        else:
            haps, *_ = self.get_haps_and_shifts(
                idx=idx,
                regions=regions,
                output_length=output_length,
                rng=rng,
                deterministic=deterministic,
                splice_plan=splice_plan,
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
            out = self._reconstruct_haplotypes(req)
        elif issubclass(self.kind, RaggedAnnotatedHaps):
            haps, annot_v_idx, annot_pos = self._reconstruct_annotated_haplotypes(req)
            out = RaggedAnnotatedHaps(haps, annot_v_idx, annot_pos)
        elif issubclass(self.kind, RaggedVariants):
            if splice_plan is not None:
                raise NotImplementedError(
                    "Spliced output is not supported for RaggedVariants."
                )
            out = self._get_variants(
                idx=idx,
                regions=req.regions,
                shifts=req.shifts,
                keep=req.keep,
                keep_offsets=req.keep_offsets,
            )
        else:
            assert_never(self.kind)

        return (
            out,  # type: ignore | pylance doesn't like this but it's correct behavior for the signature
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
                geno_offset_idxs=geno_offset_idx,
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
        r_idx, s_idx = np.unravel_index(idx, genotypes.shape[:2])  # type: ignore
        ploid_idx = np.arange(genotypes.shape[-2], dtype=np.intp)
        rsp_idx = (r_idx[:, None], s_idx[:, None], ploid_idx)
        geno_offset_idx = np.ravel_multi_index(rsp_idx, genotypes.shape[:-1])  # type: ignore
        return geno_offset_idx

    def _get_variants(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer] | None = None,
        shifts: NDArray[np.integer] | None = None,
        keep: NDArray[np.bool_] | None = None,
        keep_offsets: NDArray[np.integer] | None = None,
    ) -> RaggedVariants:
        # TODO: maybe filter variants for region, shifts?
        r, s = np.unravel_index(idx, self.genotypes.shape[:2])  # type: ignore
        # (b p ~v)
        genos = cast(Ragged[V_IDX_TYPE], self.genotypes[r, s])

        genos = ak.to_packed(genos)
        v_idxs = genos.data

        if self.min_af is not None or self.max_af is not None:
            geno_afs = self.variants.info["AF"][v_idxs]
            keep = np.full(len(v_idxs), True, np.bool_)
            if self.min_af is not None:
                keep &= geno_afs >= self.min_af
            if self.max_af is not None:
                keep &= geno_afs <= self.max_af
            _keep = Ragged.from_offsets(keep, genos.shape, genos.offsets)
            genos = Ragged(ak.to_packed(ak.to_regular(genos[_keep], 1)))
            v_idxs = genos.data
        else:
            _keep = None

        fields = {}

        fields["alt"] = self._get_alleles(genos, "alt")
        fields["start"] = Ragged.from_offsets(
            self.variants.start[v_idxs], genos.shape, genos.offsets
        )

        if "ref" in self.var_fields:
            fields["ref"] = self._get_alleles(genos, "ref")
        if "ilen" in self.var_fields:
            fields["ilen"] = Ragged.from_offsets(
                self.variants.ilen[v_idxs], genos.shape, genos.offsets
            )

        if self.dosages is not None:
            # guaranteed to have same shape as genotypes but need to make it contiguous/copy the data
            dosages = self.dosages[r, s]
            if _keep is not None:
                dosages = ak.to_regular(dosages[_keep], 1)  # type: ignore
            fields["dosage"] = Ragged(ak.to_packed(dosages))

        fields.update({
            k: self._get_info(genos, k)
            for k in self.var_fields
            if k not in {"alt", "start", "ref", "ilen", "dosage"}
        })

        variants = RaggedVariants(**fields)

        return variants

    def _get_alleles(
        self, genos: Ragged[V_IDX_TYPE], kind: Literal["alt", "ref"]
    ) -> ak.Array:
        v_idxs = genos.data

        # (b*p*v ~l)
        alleles = ak.to_packed(
            cast(RaggedAlleles, getattr(self.variants, kind)[v_idxs])
        )
        # reshape to (b, p, ~v, ~l)
        node = alleles.layout
        while not isinstance(node, NumpyArray):
            node = node.content
        l_content = ListOffsetArray(
            Index(alleles.offsets), node, parameters={"__array__": "bytestring"}
        )
        vl_content = ListOffsetArray(Index(genos.offsets), l_content)
        pvl_content = RegularArray(vl_content, genos.shape[-2])
        alleles = ak.Array(pvl_content)

        return alleles

    def _get_info(self, genos: Ragged[V_IDX_TYPE], attr: str) -> Ragged[np.number]:
        data = self.variants.info[attr][genos.data]
        return Ragged.from_offsets(data, genos.shape, genos.offsets)

    def _reconstruct_haplotypes(self, req: ReconstructionRequest) -> Ragged[np.bytes_]:
        """Reconstruct haplotype byte sequences from sparse genotypes."""
        assert self.reference is not None

        if req.splice_plan is None:
            haps = Ragged.from_offsets(
                np.empty(req.out_offsets[-1], np.uint8),
                (*req.shifts.shape, None),
                req.out_offsets,
            )
            reconstruct_haplotypes_from_sparse(
                geno_offset_idxs=req.geno_offset_idx,
                out=haps.data,
                out_offsets=haps.offsets,
                regions=req.regions,
                shifts=req.shifts,
                geno_offsets=self.genotypes.offsets,
                geno_v_idxs=self.genotypes.data,
                v_starts=self.variants.start,
                ilens=self.variants.ilen,
                alt_alleles=self.variants.alt.data.view(np.uint8),
                alt_offsets=self.variants.alt.offsets,
                ref=self.reference.reference,
                ref_offsets=self.reference.offsets,
                pad_char=self.reference.pad_char,
                keep=req.keep,
                keep_offsets=req.keep_offsets,
                annot_v_idxs=None,
                annot_ref_pos=None,
            )
            return cast(Ragged[np.bytes_], haps.view("S1"))

        # ---- splice plan path ----
        flat_geno_idx, flat_shifts, permuted_regions, keep_perm, keep_offsets_perm = (
            self._permute_request_for_splice(req)
        )
        splice_plan = req.splice_plan

        total = int(splice_plan.permuted_out_offsets[-1])
        out_buf = np.empty(total, np.uint8)

        reconstruct_haplotypes_from_sparse(
            geno_offset_idxs=flat_geno_idx.reshape(-1, 1),
            out=out_buf,
            out_offsets=splice_plan.permuted_out_offsets,
            regions=permuted_regions,
            shifts=flat_shifts.reshape(-1, 1),
            geno_offsets=self.genotypes.offsets,
            geno_v_idxs=self.genotypes.data,
            v_starts=self.variants.start,
            ilens=self.variants.ilen,
            alt_alleles=self.variants.alt.data.view(np.uint8),
            alt_offsets=self.variants.alt.offsets,
            ref=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
            keep=keep_perm,
            keep_offsets=keep_offsets_perm,
            annot_v_idxs=None,
            annot_ref_pos=None,
        )

        per_elem_shape = (splice_plan.permuted_lengths.shape[0], None)
        return cast(
            Ragged[np.bytes_],
            Ragged.from_offsets(
                out_buf.view("S1"),
                per_elem_shape,
                splice_plan.permuted_out_offsets,
            ),
        )

    def _reconstruct_annotated_haplotypes(
        self, req: ReconstructionRequest
    ) -> tuple[Ragged[np.bytes_], Ragged[V_IDX_TYPE], Ragged[np.int32]]:
        """Reconstruct haplotypes plus per-nucleotide annotations.

        Returns the haplotype bytes, the variant index at each position
        (or -1 for reference), and the reference coordinate at each position
        (or -1 for padded bases).
        """
        assert self.reference is not None

        if req.splice_plan is None:
            haps = Ragged.from_offsets(
                np.empty(req.out_offsets[-1], np.uint8),
                (*req.shifts.shape, None),
                req.out_offsets,
            )
            annot_v_idxs = Ragged.from_offsets(
                np.empty(req.out_offsets[-1], V_IDX_TYPE),
                (*req.shifts.shape, None),
                req.out_offsets,
            )
            annot_positions = Ragged.from_offsets(
                np.empty(req.out_offsets[-1], np.int32),
                (*req.shifts.shape, None),
                req.out_offsets,
            )

            # annot offsets match haps offsets, so we share them.
            reconstruct_haplotypes_from_sparse(
                geno_offset_idxs=req.geno_offset_idx,
                out=haps.data,
                out_offsets=haps.offsets,
                regions=req.regions,
                shifts=req.shifts,
                geno_offsets=self.genotypes.offsets,
                geno_v_idxs=self.genotypes.data,
                v_starts=self.variants.start,
                ilens=self.variants.ilen,
                alt_alleles=self.variants.alt.data.view(np.uint8),
                alt_offsets=self.variants.alt.offsets,
                ref=self.reference.reference,
                ref_offsets=self.reference.offsets,
                pad_char=self.reference.pad_char,
                keep=req.keep,
                keep_offsets=req.keep_offsets,
                annot_v_idxs=annot_v_idxs.data,
                annot_ref_pos=annot_positions.data,
            )
            return (
                cast(Ragged[np.bytes_], haps.view("S1")),
                annot_v_idxs,
                annot_positions,
            )

        # ---- splice plan path ----
        flat_geno_idx, flat_shifts, permuted_regions, keep_perm, keep_offsets_perm = (
            self._permute_request_for_splice(req)
        )
        splice_plan = req.splice_plan

        total = int(splice_plan.permuted_out_offsets[-1])
        out_buf = np.empty(total, np.uint8)
        annot_v_buf = np.empty(total, V_IDX_TYPE)
        annot_pos_buf = np.empty(total, np.int32)

        reconstruct_haplotypes_from_sparse(
            geno_offset_idxs=flat_geno_idx.reshape(-1, 1),
            out=out_buf,
            out_offsets=splice_plan.permuted_out_offsets,
            regions=permuted_regions,
            shifts=flat_shifts.reshape(-1, 1),
            geno_offsets=self.genotypes.offsets,
            geno_v_idxs=self.genotypes.data,
            v_starts=self.variants.start,
            ilens=self.variants.ilen,
            alt_alleles=self.variants.alt.data.view(np.uint8),
            alt_offsets=self.variants.alt.offsets,
            ref=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
            keep=keep_perm,
            keep_offsets=keep_offsets_perm,
            annot_v_idxs=annot_v_buf,
            annot_ref_pos=annot_pos_buf,
        )

        per_elem_shape = (splice_plan.permuted_lengths.shape[0], None)
        haps_rag = cast(
            Ragged[np.bytes_],
            Ragged.from_offsets(
                out_buf.view("S1"),
                per_elem_shape,
                splice_plan.permuted_out_offsets,
            ),
        )
        annot_v_rag = Ragged.from_offsets(
            annot_v_buf, per_elem_shape, splice_plan.permuted_out_offsets
        )
        annot_pos_rag = Ragged.from_offsets(
            annot_pos_buf, per_elem_shape, splice_plan.permuted_out_offsets
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
        """Permute the per-element arrays in ``req`` according to ``splice_plan.perm``.

        ``geno_offset_idx`` and ``shifts`` have shape ``(B, P)``; flatten to
        ``(B*P,)`` in (query, ploidy) C-order, then permute. The kernel then
        runs with ploidy=1 over the ``B*P`` flattened queries.
        """
        assert req.splice_plan is not None
        splice_plan = req.splice_plan
        ploidy = req.shifts.shape[1] if req.shifts.ndim > 1 else 1
        perm = splice_plan.perm

        flat_geno_idx = req.geno_offset_idx.reshape(-1)[perm].astype(
            np.intp, copy=False
        )
        flat_shifts = req.shifts.reshape(-1)[perm].astype(np.int32, copy=False)
        # regions has shape (B, 3). For (B*P, 3), each query repeats P times
        # consecutively, then we apply the same perm.
        regions_flat = np.repeat(req.regions, ploidy, axis=0)
        permuted_regions = regions_flat[perm]

        # keep / keep_offsets: per-k granularity (length B*P + 1).
        if req.keep is not None and req.keep_offsets is not None:
            keep_lens = np.diff(req.keep_offsets)
            keep_lens_perm = keep_lens[perm]
            keep_offsets_perm = lengths_to_offsets(
                keep_lens_perm.astype(np.int64), dtype=np.int64
            )
            keep_perm = np.empty(int(keep_lens_perm.sum()), dtype=np.bool_)
            write_cursor = 0
            for k_old in perm:
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
