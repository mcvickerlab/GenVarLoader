"""Merge on-disk GVL datasets along the region or sample axis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .._atomic import atomic_dir
from .._fasta_cache import fingerprint as _bounded_fingerprint
from ._concat_io import copy_runs, gather_fixed, link_or_copy_buffered
from ._concat_plan import Run, coalesce, provenance
from ._concat_validate import (
    ConcatInput,
    load_inputs,
    validate_concat,
    variants_fingerprint,
)
from ._write import DATASET_FORMAT_VERSION, Metadata, _prep_bed

if TYPE_CHECKING:
    from ._impl import Dataset

__all__ = ["concat"]


def _merged_bed(
    inputs: list[ConcatInput], axis: str
) -> tuple[pl.DataFrame, pl.DataFrame, "NDArray[np.int64] | None"]:
    """Return (input_bed_with_r_idx_map, sorted_gvl_bed, dataset_of_each_sorted_row).

    ``dataset_of_each_sorted_row`` is ``None`` for ``axis="samples"``: regions are
    shared and unmerged there, so there is nothing to attribute per row. For
    ``axis="regions"`` the returned ``sorted_gvl_bed`` is unjittered/unextended
    (built with ``max_jitter=None``) and is used only to establish sort order and
    contigs; the actual merged ``regions.npy`` is gathered from each input's own
    stored regions (see ``concat``), not recomputed from this frame.
    """
    if axis == "samples":
        bed = inputs[0].bed.drop("r_idx_map", strict=False)
        gvl_bed, _contigs, r_map = _prep_bed(bed, None)
        return bed.with_columns(r_idx_map=pl.Series(r_map)), gvl_bed, None

    parts = []
    for d, inp in enumerate(inputs):
        b = inp.bed.drop("r_idx_map", strict=False)
        parts.append(b.with_columns(_ds=pl.lit(d, pl.Int64)))
    merged_in = pl.concat(parts, how="vertical_relaxed")
    gvl_bed, _contigs, r_map = _prep_bed(merged_in.drop("_ds"), None)
    # r_map maps input row order -> sorted order; invert to get sorted -> input row.
    sorted_to_input = np.argsort(r_map)
    ds_of_sorted = merged_in["_ds"].to_numpy()[sorted_to_input].astype(np.int64)
    out_bed = merged_in.drop("_ds").with_columns(r_idx_map=pl.Series(r_map))
    return out_bed, gvl_bed, ds_of_sorted


def _region_order(ds_of_sorted: "NDArray[np.int64]", n_ds: int) -> "NDArray[np.int64]":
    """Build the ``(dataset_idx, within_dataset_idx)`` order for merged regions.

    Restricting a consistent sort of the union to one dataset's rows yields that
    dataset's own sorted order (every on-disk store is written in sorted region
    order), so the within-dataset index is just the running count of prior
    occurrences of that dataset in ``ds_of_sorted``.
    """
    order = np.empty((len(ds_of_sorted), 2), np.int64)
    order[:, 0] = ds_of_sorted
    within = np.empty(len(ds_of_sorted), np.int64)
    for d in range(n_ds):
        idx = np.flatnonzero(ds_of_sorted == d)
        within[idx] = np.arange(len(idx), dtype=np.int64)
    order[:, 1] = within
    return order


def _sample_order(inputs: list[ConcatInput], samples: list[str]) -> "NDArray[np.int64]":
    """Build the ``(dataset_idx, within_dataset_idx)`` order for merged samples.

    Merged samples are ``sorted(union)``; each merged sample's ``within_dataset_idx``
    is its position in the owning dataset's own (already-sorted, since ``write``
    sorts samples unconditionally) ``meta.samples`` list.
    """
    owner: dict[str, tuple[int, int]] = {}
    for d, inp in enumerate(inputs):
        for w, s in enumerate(inp.meta.samples):
            owner[s] = (d, w)
    return np.array([owner[s] for s in samples], dtype=np.int64)


def _gather_regions(
    paths: list[Path], ds_of_sorted: "NDArray[np.int64]"
) -> "NDArray[np.int32]":
    """Gather each input's stored ``regions.npy`` rows into merged sorted order.

    Recomputing merged regions from the raw bed (as a single-shot ``write`` would)
    would drop per-dataset adjustments already baked into each source's
    ``regions.npy`` (``max_jitter`` expansion, ``extend_to_length`` end-extension).
    Region rows are independent across datasets on ``axis="regions"``, so gathering
    the already-computed rows reproduces exactly what a single-shot write would
    have stored.
    """
    regions_per_ds = [np.load(p / "regions.npy") for p in paths]
    merged = np.empty(
        (len(ds_of_sorted), regions_per_ds[0].shape[1]), regions_per_ds[0].dtype
    )
    for d, regions in enumerate(regions_per_ds):
        idx = np.flatnonzero(ds_of_sorted == d)
        merged[idx] = regions
    return merged


def _merge_regions_npy_samples_axis(paths: list[Path]) -> "NDArray[np.int32]":
    """Merge ``regions.npy`` for ``axis="samples"``: elementwise max of chromEnd.

    Regions are shared and unmerged across inputs on this axis, but each
    input's own ``extend_to_length`` can have extended ``chromEnd`` (column 2,
    per ``bed_to_regions``'s ``[chrom, chromStart, chromEnd, strand]`` layout)
    to a different length depending on which samples were present at that
    input's write time. The merged dataset must cover the union: taking the
    per-row max of the stored ``chromEnd`` reproduces what a single-shot write
    over the full sample set would have stored, without dropping ``max_jitter``
    (recomputing from the raw bed, as a plain ``_write_regions`` call would, does
    drop it).

    The other columns (chrom, chromStart, strand) are asserted equal across
    inputs rather than assumed: ``validate_concat`` already requires identical
    beds and a matching ``max_jitter`` for ``axis="samples"``, so a mismatch
    here would mean the inputs' regions were not actually identical despite
    passing validation, which is a bug worth failing loudly on rather than
    silently taking input #0's columns.
    """
    stored = [np.load(p / "regions.npy") for p in paths]
    merged = stored[0].copy()
    for i, s in enumerate(stored[1:], start=1):
        assert np.array_equal(s[:, [0, 1, 3]], stored[0][:, [0, 1, 3]]), (
            f"input #{i}'s regions.npy has different chrom/chromStart/strand than "
            "input #0's; axis='samples' requires identical regions across inputs "
            "(validate_concat should have caught this)"
        )
        merged[:, 2] = np.maximum(merged[:, 2], s[:, 2])
    return merged


def _assert_annot_track_matches(name: str, src_dirs: list[Path]) -> None:
    """Verify an annot track's on-disk data agrees across inputs before copying it.

    Copied verbatim from input #0 (the design spec's rule for
    ``axis="samples"``: "sample-independent; fingerprint-compare across
    inputs, then copy input[0]").

    Annotation tracks are sample-*independent* but not dataset-independent:
    each input's own ``_write_annot_track`` call ran against that input's own
    stored ``regions.npy`` (``_write.py:354-356``), and ``extend_to_length``
    can extend ``chromEnd`` by a different amount per input depending on which
    samples were present at that input's write time -- precisely the case
    ``_merge_regions_npy_samples_axis``'s elementwise-max exists to handle. If
    chromEnd diverged, input #0's annot data was computed over a *narrower*
    window than the merged (max) region, and linking it verbatim would
    silently under-cover the merged dataset's annotation reads. This compare
    turns that into a raised error instead of a silent truncation.

    Reuses the existing bounded fingerprint idiom (``_fasta_cache.fingerprint``,
    blake2b over the first 1 MiB + total size -- the same one
    ``variants_fingerprint`` uses for ``genotypes/variants.arrow``) rather than
    hashing whole files, which keeps this bounded even if a track's payload is
    large.
    """
    ref_dir = src_dirs[0]
    ref_fps = {
        fname: _bounded_fingerprint(ref_dir / f"{fname}.npy")
        for fname in ("starts", "ends", "values", "offsets")
    }
    for i, d in enumerate(src_dirs[1:], start=1):
        for fname, ref_fp in ref_fps.items():
            fp = _bounded_fingerprint(d / f"{fname}.npy")
            if fp != ref_fp:
                raise ValueError(
                    f"annot track {name!r}: input #{i}'s {fname}.npy differs from "
                    f"input #0's ({fp} vs {ref_fp}). axis='samples' requires "
                    "identical annotation data across inputs -- this usually means "
                    "extend_to_length produced a different chromEnd per input (see "
                    "_merge_regions_npy_samples_axis), so input #0's annot_intervals "
                    "no longer covers the merged (elementwise-max) region window."
                )


def _gather_svar_offsets(
    paths: list[Path],
    out_dir: Path,
    runs: list[Run],
    shapes: list[tuple[int, int]],
    ploidy: int,
) -> None:
    """Gather a .svar dataset's (2, R, S, P) absolute start/stop offsets.

    Stored as two leading planes (starts, then stops), each a flat ``(R*S*P,)``
    int64 array over the same ``(region, sample, ploid)`` C-order flat-slot
    space ``runs`` was coalesced over (see ``_concat_plan``). ``gather_fixed``
    streams fixed-stride *records* through a single byte-addressed file and
    can't express this layout: the slot axis is nested *inside* the two
    leading planes, so a given slot's start and stop live ``R*S*P`` elements
    apart, not adjacent in a 16-byte record. That makes this the one gather in
    this module that can't be routed through ``gather_fixed`` -- materializing
    each plane (each is ``n_slots`` int64s, the same order of magnitude as the
    offsets arrays ``copy_runs`` already builds in memory for the pgen/vcf
    backend) is the correct call here, not a shortcut around the "don't
    materialize a full array" constraint.
    """
    n_src_slots = [r * s * ploidy for r, s in shapes]
    planes = []
    for plane in (0, 1):
        srcs = []
        for p, n in zip(paths, n_src_slots):
            arr = np.fromfile(p / "genotypes" / "offsets.npy", dtype=np.int64)
            srcs.append(arr.reshape(2, -1)[plane])
        out = np.empty(sum(r.src_stop - r.src_start for r in runs), np.int64)
        for r in runs:
            n = r.src_stop - r.src_start
            out[r.dst_start : r.dst_start + n] = srcs[r.src][r.src_start : r.src_stop]
        planes.append(out)
    with open(out_dir / "offsets.npy", "wb") as f:
        f.write(np.stack(planes).tobytes())


def _concat_svar2_ranges(
    paths: list[Path],
    out_dir: Path,
    axis: str,
    shapes: list[tuple[int, int]],
    ploidy: int,
    n_regions: int,
    n_samples: int,
    order: "NDArray[np.int64]",
) -> None:
    """Merge a .svar2 dataset's cached range arrays.

    ``vk_snp_range``/``vk_indel_range`` are per-``(region, sample, ploid)``
    fixed 16-byte (2 x int64) records in the same C-order flat-slot space as
    genotype offsets, so they gather through ``gather_fixed`` using the same
    ``runs`` the svar/svar2 offsets and per-sample tracks use (built from
    ``provenance(axis, shapes, ploidy, order=order)``).

    ``dense_snp_range``/``dense_indel_range`` are per-region only (sample- and
    ploidy-independent): on ``axis="samples"`` they're identical across inputs
    and just linked from input #0; on ``axis="regions"`` they need the region
    ordering (not a block concatenation), so they gather through a *separate*
    ``gather_fixed`` call using region-only runs (``n_samples=1, ploidy=1``)
    built from the same merged region ``order``.

    ``sample_cols`` maps merged sample slot -> index into the linked svar2
    store's ``available_samples``; each input's own ``sample_cols.npy`` is
    already indexed by that input's own (sorted) sample list, so the merged
    array is a direct per-merged-sample lookup through ``order``.
    """
    prov = provenance(axis, shapes, ploidy, order=order)
    runs = coalesce(prov)

    for name in ("vk_snp_range", "vk_indel_range"):
        gather_fixed(
            [p / "genotypes" / "svar2_ranges" / f"{name}.npy" for p in paths],
            out_dir / f"{name}.npy",
            runs,
            record_bytes=16,
        )

    if axis == "samples":
        for name in ("dense_snp_range", "dense_indel_range"):
            link_or_copy_buffered(
                paths[0] / "genotypes" / "svar2_ranges" / f"{name}.npy",
                out_dir / f"{name}.npy",
            )
    else:
        region_shapes = [(r, 1) for r, _ in shapes]
        region_runs = coalesce(provenance("regions", region_shapes, 1, order=order))
        for name in ("dense_snp_range", "dense_indel_range"):
            gather_fixed(
                [p / "genotypes" / "svar2_ranges" / f"{name}.npy" for p in paths],
                out_dir / f"{name}.npy",
                region_runs,
                record_bytes=16,
            )

    cols = [
        np.load(p / "genotypes" / "svar2_ranges" / "sample_cols.npy") for p in paths
    ]
    if axis == "samples":
        merged_cols = np.array([cols[d][w] for d, w in order], np.int64)
    else:
        merged_cols = cols[0]
    np.save(out_dir / "sample_cols.npy", merged_cols)

    src_meta = json.loads(
        (paths[0] / "genotypes" / "svar2_ranges" / "svar2_meta.json").read_text()
    )
    R, S, P = n_regions, n_samples, ploidy
    src_meta["vk_snp_range"]["shape"] = [R, S, P, 2]
    src_meta["vk_indel_range"]["shape"] = [R, S, P, 2]
    src_meta["dense_snp_range"]["shape"] = [R, 2]
    src_meta["dense_indel_range"]["shape"] = [R, 2]
    src_meta["sample_cols"]["shape"] = [S]
    (out_dir / "svar2_meta.json").write_text(json.dumps(src_meta))


def concat(
    path: str | Path,
    datasets: "Sequence[str | Path | Dataset]",
    axis: Literal["regions", "samples"],
    *,
    overwrite: bool = False,
    max_mem: int | str = "4g",
) -> None:
    """Merge GVL datasets along the region or sample axis.

    All inputs must share one variant source: the same PGEN/VCF variant table, or
    the same ``.svar``/``.svar2`` store. Merging datasets built from *different*
    variant sources is not supported and raises.

    On ``axis="regions"`` the inputs must have identical samples in identical order
    and their regions are concatenated. On ``axis="samples"`` the inputs must have
    identical regions and disjoint sample sets, and their samples are merged into
    sorted order.

    This moves roughly the full size of the merged dataset at sequential-IO speed.
    It is worth doing only against the alternative of re-extracting genotypes.

    Args:
        path: Destination dataset directory.
        datasets: Two or more dataset directories or opened :class:`Dataset` objects.
        axis: ``"regions"`` to concatenate regions, ``"samples"`` to concatenate samples.
        overwrite: Replace ``path`` if it already exists.
        max_mem: Advisory memory budget. Accepted for symmetry with :func:`write`.

    Raises:
        ValueError: If any precondition fails; the message names the offending input.
        FileExistsError: If ``path`` exists and ``overwrite`` is False.
    """
    from ._impl import Dataset as _Dataset

    paths = [Path(d.path if isinstance(d, _Dataset) else d) for d in datasets]
    dest = Path(path)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} exists; pass overwrite=True to replace it")

    inputs = load_inputs(paths)
    validate_concat(inputs, axis)

    ref = inputs[0]
    ploidy = ref.meta.ploidy or 1
    shapes = [(i.n_regions, i.n_samples) for i in inputs]

    if axis == "regions":
        n_regions = sum(i.n_regions for i in inputs)
        samples = list(ref.meta.samples)
    else:
        n_regions = ref.n_regions
        samples = sorted(s for i in inputs for s in i.meta.samples)

    input_bed, _gvl_bed, ds_of_sorted = _merged_bed(inputs, axis)

    if axis == "regions":
        assert ds_of_sorted is not None
        order = _region_order(ds_of_sorted, len(inputs))
    else:
        order = _sample_order(inputs, samples)

    with atomic_dir(dest, overwrite=overwrite) as tmp:
        input_bed.write_ipc(tmp / "input_regions.arrow")
        if axis == "regions":
            assert ds_of_sorted is not None
            merged_regions = _gather_regions(paths, ds_of_sorted)
            np.save(tmp / "regions.npy", merged_regions)
        else:
            # axis="samples": regions are validated identical across inputs, but
            # per-input extend_to_length outcomes can still differ, so the merged
            # chromEnd is the elementwise max of each input's own stored value
            # (see _merge_regions_npy_samples_axis).
            merged_regions = _merge_regions_npy_samples_axis(paths)
            np.save(tmp / "regions.npy", merged_regions)

        meta: dict = {
            "samples": samples,
            "contigs": ref.meta.contigs,
            "n_regions": n_regions,
            "ploidy": ref.meta.ploidy,
            "max_jitter": ref.meta.max_jitter,
            "version": ref.meta.version,
            "format_version": DATASET_FORMAT_VERSION,
        }

        if ref.backend == "pgen_vcf":
            geno = tmp / "genotypes"
            geno.mkdir(parents=True, exist_ok=True)
            link_or_copy_buffered(
                paths[0] / "genotypes" / "variants.arrow",
                geno / "variants.arrow",
            )
            meta["variants_fingerprint"] = variants_fingerprint(paths[0])

            prov = provenance(axis, shapes, ploidy, order=order)
            runs = coalesce(prov)
            src_offsets = [
                np.fromfile(p / "genotypes" / "offsets.npy", dtype=np.int64)
                for p in paths
            ]
            from genoray._types import V_IDX_TYPE

            merged = copy_runs(
                [p / "genotypes" / "variant_idxs.npy" for p in paths],
                geno / "variant_idxs.npy",
                runs,
                src_offsets,
                itemsize=np.dtype(V_IDX_TYPE).itemsize,
            )
            with open(geno / "offsets.npy", "wb") as f:
                f.write(merged.tobytes())
        elif ref.backend == "svar":
            geno = tmp / "genotypes"
            geno.mkdir(parents=True, exist_ok=True)

            svar_meta = json.loads(
                (paths[0] / "genotypes" / "svar_meta.json").read_text()
            )
            # offsets are (2, R, S, P) absolute start/stop pairs into the svar's
            # global array. Gather the R/S axes in the same merged `order` as
            # every other store; values copy verbatim (they're absolute indices
            # into the shared external store, not slot-relative).
            prov = provenance(axis, shapes, ploidy, order=order)
            runs = coalesce(prov)
            _gather_svar_offsets(paths, geno, runs, shapes, ploidy)
            shape = [2, n_regions, len(samples), ploidy]
            (geno / "svar_meta.json").write_text(
                json.dumps({"shape": shape, "dtype": svar_meta["dtype"]})
            )
            meta["svar_link"] = ref.meta.svar_link.model_dump()

        elif ref.backend == "svar2":
            out_dir = tmp / "genotypes" / "svar2_ranges"
            out_dir.mkdir(parents=True, exist_ok=True)
            _concat_svar2_ranges(
                paths,
                out_dir,
                axis,
                shapes,
                ploidy,
                n_regions,
                len(samples),
                order,
            )
            meta["svar2_link"] = ref.meta.svar2_link.model_dump()

        # per-sample tracks: offsets over (R, S), no ploidy axis. Same merged
        # `order` as everything else -- a track store is indexed by the same
        # (region, sample) grid as the genotypes/offsets stores above.
        if ref.tracks:
            t_prov = provenance(axis, shapes, 1, order=order)
            t_runs = coalesce(t_prov)
            for name in ref.tracks:
                src_dirs = [p / "intervals" / name for p in paths]
                out_t = tmp / "intervals" / name
                out_t.mkdir(parents=True, exist_ok=True)
                t_offsets = [
                    np.fromfile(d / "offsets.npy", dtype=np.int64) for d in src_dirs
                ]
                merged_t = None
                for fname, dt in (
                    ("starts", np.int32),
                    ("ends", np.int32),
                    ("values", np.float32),
                ):
                    merged_t = copy_runs(
                        [d / f"{fname}.npy" for d in src_dirs],
                        out_t / f"{fname}.npy",
                        t_runs,
                        t_offsets,
                        itemsize=np.dtype(dt).itemsize,
                    )
                assert merged_t is not None
                with open(out_t / "offsets.npy", "wb") as f:
                    f.write(merged_t.tobytes())

        # annot tracks: sample-independent, offsets over R only. On the sample
        # axis every input SHOULD have identical track data (same regions, no
        # sample dependence) -- but "should" is not "does": extend_to_length
        # can diverge chromEnd per input, so this is fingerprint-verified
        # (per the design spec) before linking from input #0, not assumed. On
        # the region axis it needs the same region ordering as everything
        # else -- derived from the same `order` the region axis already
        # computed above, NOT a block concatenation.
        if ref.annot_tracks:
            for name in ref.annot_tracks:
                out_a = tmp / "annot_intervals" / name
                out_a.mkdir(parents=True, exist_ok=True)
                src_dirs = [p / "annot_intervals" / name for p in paths]
                if axis == "samples":
                    _assert_annot_track_matches(name, src_dirs)
                    for fname in ("starts", "ends", "values", "offsets"):
                        link_or_copy_buffered(
                            src_dirs[0] / f"{fname}.npy", out_a / f"{fname}.npy"
                        )
                else:
                    a_prov = provenance(
                        "regions", [(r, 1) for r, _ in shapes], 1, order=order
                    )
                    a_runs = coalesce(a_prov)
                    a_offsets = [
                        np.fromfile(d / "offsets.npy", dtype=np.int64) for d in src_dirs
                    ]
                    merged_a = None
                    for fname, dt in (
                        ("starts", np.int32),
                        ("ends", np.int32),
                        ("values", np.float32),
                    ):
                        merged_a = copy_runs(
                            [d / f"{fname}.npy" for d in src_dirs],
                            out_a / f"{fname}.npy",
                            a_runs,
                            a_offsets,
                            itemsize=np.dtype(dt).itemsize,
                        )
                    assert merged_a is not None
                    with open(out_a / "offsets.npy", "wb") as f:
                        f.write(merged_a.tobytes())

        with open(tmp / "metadata.json", "w") as f:
            f.write(Metadata(**meta).model_dump_json())
