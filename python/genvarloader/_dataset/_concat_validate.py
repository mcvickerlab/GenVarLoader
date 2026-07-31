"""Preconditions for :func:`genvarloader.concat`.

All checks run before any bytes move, so a rejected merge costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .._fasta_cache import Fingerprint, fingerprint
from ._write import Metadata

__all__ = ["ConcatInput", "load_inputs", "validate_concat", "variants_fingerprint"]


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


def _backend_of(path: Path, meta: Metadata) -> str:
    if meta.svar2_link is not None:
        return "svar2"
    if meta.svar_link is not None:
        return "svar"
    if (path / "genotypes").is_dir():
        return "pgen_vcf"
    return "tracks_only"


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
                has_dosages=(p / "genotypes" / "dosages.npy").exists(),
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

    ref = inputs[0]
    for i, inp in enumerate(inputs[1:], start=1):
        if inp.backend != ref.backend:
            raise ValueError(
                f"input #{i} uses variant source {inp.backend!r} but input #0 uses "
                f"{ref.backend!r}; all inputs must share the same variant source"
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
        names_seen: dict[str, int] = {}
        for i, inp in enumerate(inputs):
            names = _region_names(inp)
            if names is None:
                continue
            for nm in names:
                if nm in names_seen:
                    raise ValueError(
                        f"duplicate region name {nm!r} in inputs #{names_seen[nm]} "
                        f"and #{i}; region names must be unique after merging"
                    )
                names_seen[nm] = i
