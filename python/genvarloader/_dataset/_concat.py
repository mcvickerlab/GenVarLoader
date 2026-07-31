"""Merge on-disk GVL datasets along the region or sample axis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .._atomic import atomic_dir
from ._concat_io import copy_runs, link_or_copy_buffered
from ._concat_plan import coalesce, provenance
from ._concat_validate import (
    ConcatInput,
    load_inputs,
    validate_concat,
    variants_fingerprint,
)
from ._write import DATASET_FORMAT_VERSION, Metadata, _prep_bed, _write_regions

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

    input_bed, gvl_bed, ds_of_sorted = _merged_bed(inputs, axis)

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
            # per-input extend_to_length outcomes can still differ; merging that
            # correctly (an elementwise max of the end column) is deferred to a
            # follow-up task, so this recomputes an un-extended regions.npy from
            # the shared raw bed, matching input #0's regions structurally.
            _write_regions(tmp, gvl_bed, ref.meta.contigs)

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
        elif ref.backend in ("svar", "svar2"):
            raise NotImplementedError(
                f"concat for {ref.backend!r}-backed datasets lands in a follow-up task"
            )

        with open(tmp / "metadata.json", "w") as f:
            f.write(Metadata(**meta).model_dump_json())
