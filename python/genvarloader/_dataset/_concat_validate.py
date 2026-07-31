"""Preconditions for :func:`genvarloader.concat`.

All checks run before any bytes move, so a rejected merge costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .._fasta_cache import Fingerprint, fingerprint
from ._svar2_link import Svar2Fingerprint, _resolve_svar2
from ._svar_link import SvarFingerprint, _resolve_svar
from ._write import Metadata

__all__ = ["ConcatInput", "load_inputs", "validate_concat", "variants_fingerprint"]

VariantFingerprint = Fingerprint | SvarFingerprint | Svar2Fingerprint


@dataclass(frozen=True)
class ConcatInput:
    """One resolved input dataset plus the facts validation needs."""

    path: Path | None
    meta: Metadata
    bed: pl.DataFrame
    n_regions: int
    n_samples: int
    backend: str
    tracks: list[str]
    annot_tracks: list[str]
    has_dosages: bool
    fingerprint: VariantFingerprint | None = None


def _backend_of(path: Path, meta: Metadata) -> str:
    if meta.svar2_link is not None:
        return "svar2"
    if meta.svar_link is not None:
        return "svar"
    if (path / "genotypes").is_dir():
        return "pgen_vcf"
    return "tracks_only"


def _resolved_svar_dir(path: Path, meta: Metadata) -> Path | None:
    """Resolve the externally-linked svar/svar2 store, if this dataset has one.

    Mirrors the resolution order used by :class:`Haps` (``_haps.py``): svar2
    link first, then svar link, then ``None`` for pgen/vcf and tracks-only
    datasets, which have no external store.
    """
    if meta.svar2_link is not None:
        return _resolve_svar2(path, meta.svar2_link, None)
    if meta.svar_link is not None:
        return _resolve_svar(path, meta.svar_link, None)
    return None


def _has_dosages(path: Path, meta: Metadata) -> bool:
    """True iff the resolved variant store contains ``dosages.npy``.

    Dosages only ever live inside an externally-linked svar/svar2 store (see
    ``_haps.py:_has_dosage_file_on_disk``); pgen/vcf and tracks-only datasets
    never have dosages.
    """
    svar_dir = _resolved_svar_dir(path, meta)
    if svar_dir is None:
        return False
    return (svar_dir / "dosages.npy").exists()


def _variant_source_fingerprint(
    path: Path, meta: Metadata
) -> VariantFingerprint | None:
    """Identity of the variant table/store backing this dataset, if any.

    svar/svar2-backed datasets already carry a fingerprint of their linked
    store on ``meta``; reuse it instead of re-hashing an external store.
    pgen/vcf datasets fall back to :func:`variants_fingerprint` over their own
    ``genotypes/variants.arrow``. Tracks-only datasets have no variant source.
    """
    if meta.svar2_link is not None:
        return meta.svar2_link.fingerprint
    if meta.svar_link is not None:
        return meta.svar_link.fingerprint
    if (path / "genotypes").is_dir():
        return variants_fingerprint(path)
    return None


def load_inputs(paths: list[Path]) -> list[ConcatInput]:
    """Read each dataset's metadata, bed, and store inventory.

    Args:
        paths: Dataset directories.

    Returns:
        One :class:`ConcatInput` per path, in the given order.
    """
    out = []
    for p in paths:
        meta = Metadata.model_validate_json((p / "metadata.json").read_text())
        bed = pl.read_ipc(p / "input_regions.arrow")
        tracks = sorted(d.name for d in (p / "intervals").glob("*") if d.is_dir())
        annot = sorted(d.name for d in (p / "annot_intervals").glob("*") if d.is_dir())
        out.append(
            ConcatInput(
                path=p,
                meta=meta,
                bed=bed,
                n_regions=meta.n_regions,
                n_samples=len(meta.samples),
                backend=_backend_of(p, meta),
                tracks=tracks,
                annot_tracks=annot,
                has_dosages=_has_dosages(p, meta),
                fingerprint=_variant_source_fingerprint(p, meta),
            )
        )
    return out


def variants_fingerprint(path: Path) -> Fingerprint:
    """Bounded content fingerprint of a dataset's ``genotypes/variants.arrow``.

    Args:
        path: Dataset directory.

    Returns:
        A blake2b-over-1-MiB-plus-size fingerprint, cheap on a multi-GB index.
    """
    return fingerprint(path / "genotypes" / "variants.arrow")


def _region_names(inp: ConcatInput) -> list[str] | None:
    if "name" not in inp.bed.columns:
        return None
    return inp.bed["name"].to_list()


def validate_concat(inputs: list[ConcatInput], axis: str) -> None:
    """Check every precondition for a merge.

    Args:
        inputs: Resolved input datasets, in merge order.
        axis: Either ``"regions"`` or ``"samples"``.

    Raises:
        ValueError: If any precondition fails. The message names the offending
            input and the expected value.
    """
    if axis not in ("regions", "samples"):
        raise ValueError(f'axis must be "regions" or "samples", got {axis!r}')
    if len(inputs) < 2:
        raise ValueError(f"concat needs at least two datasets, got {len(inputs)}")

    for i, inp in enumerate(inputs):
        coord_cols = ["chrom", "chromStart", "chromEnd"]
        dupes = (
            inp.bed.select(coord_cols)
            .filter(pl.struct(coord_cols).is_duplicated())
            .unique()
        )
        if dupes.height > 0:
            chrom, start, end = dupes.row(0)
            raise ValueError(
                f"input #{i} has duplicate regions at {chrom}:{start}-{end} "
                "(identical chrom/chromStart/chromEnd within a single input's "
                "bed); this makes the merged sort order ambiguous for that "
                "input's rows, which can silently swap which stored region a "
                "read returns. Deduplicate the input before concatenating. "
                "(Identical coordinates across different inputs are fine.)"
            )

    ref = inputs[0]
    for i, inp in enumerate(inputs[1:], start=1):
        if inp.backend != ref.backend:
            raise ValueError(
                f"input #{i} uses variant source {inp.backend!r} but input #0 uses "
                f"{ref.backend!r}; all inputs must share the same variant source"
            )
        if (
            ref.fingerprint is not None
            and inp.fingerprint is not None
            and inp.fingerprint != ref.fingerprint
        ):
            raise ValueError(
                f"input #{i} has a different variant source than input #0 "
                "(variants fingerprint mismatch); all inputs must share one "
                "variant source"
            )
        if inp.meta.ploidy != ref.meta.ploidy:
            raise ValueError(
                f"input #{i} has ploidy {inp.meta.ploidy}, expected {ref.meta.ploidy}"
            )
        if inp.meta.max_jitter != ref.meta.max_jitter:
            raise ValueError(
                f"input #{i} has max_jitter {inp.meta.max_jitter}, "
                f"expected {ref.meta.max_jitter}"
            )
        if inp.meta.contigs != ref.meta.contigs:
            raise ValueError(f"input #{i} has different contigs than input #0")
        if inp.tracks != ref.tracks:
            raise ValueError(
                f"input #{i} has tracks {inp.tracks}, expected {ref.tracks}"
            )
        if inp.annot_tracks != ref.annot_tracks:
            raise ValueError(
                f"input #{i} has annot tracks {inp.annot_tracks}, "
                f"expected {ref.annot_tracks}"
            )
        if inp.has_dosages != ref.has_dosages:
            raise ValueError(
                f"input #{i} {'has' if inp.has_dosages else 'lacks'} dosages, "
                "which does not match input #0"
            )

    if axis == "samples":
        for i, inp in enumerate(inputs[1:], start=1):
            if inp.n_regions != ref.n_regions or not inp.bed.equals(ref.bed):
                raise ValueError(
                    f"axis='samples' requires identical regions across inputs; "
                    f"input #{i} differs from input #0"
                )
        seen: dict[str, int] = {}
        for i, inp in enumerate(inputs):
            for s in inp.meta.samples:
                if s in seen:
                    raise ValueError(
                        f"axis='samples' requires non-overlapping samples across "
                        f"inputs, but sample {s!r} appears in inputs #{seen[s]} "
                        f"and #{i}"
                    )
                seen[s] = i
    else:
        for i, inp in enumerate(inputs[1:], start=1):
            if inp.meta.samples != ref.meta.samples:
                raise ValueError(
                    f"axis='regions' requires identical samples in identical order; "
                    f"input #{i} differs from input #0"
                )
        ref_cols = set(ref.bed.columns) - {"r_idx_map"}
        for i, inp in enumerate(inputs[1:], start=1):
            inp_cols = set(inp.bed.columns) - {"r_idx_map"}
            if inp_cols != ref_cols:
                raise ValueError(
                    f"input #{i}'s bed has columns {sorted(inp_cols)}, expected "
                    f"{sorted(ref_cols)} (from input #0); all inputs must have the "
                    f"same bed columns to concatenate. Differing columns: "
                    f"{sorted(inp_cols ^ ref_cols)}"
                )
        names_seen: dict[str, int] = {}
        for i, inp in enumerate(inputs):
            names = _region_names(inp)
            if names is None:
                continue
            for nm in names:
                if nm is None:
                    # A BED "name" field of "." (no name) reads back as null.
                    # Absent names are not identity: only collisions between two
                    # actual names are ambiguous after merging.
                    continue
                if nm in names_seen:
                    raise ValueError(
                        f"duplicate region name {nm!r} in inputs #{names_seen[nm]} "
                        f"and #{i}; region names must be unique after merging"
                    )
                names_seen[nm] = i
