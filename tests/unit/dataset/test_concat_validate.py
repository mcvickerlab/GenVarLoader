"""Unit tests for gvl.concat preconditions."""

from dataclasses import replace

import polars as pl
import pytest

from genvarloader._dataset._concat_validate import ConcatInput, validate_concat
from genvarloader._dataset._write import Metadata


def _mk(samples, n_regions, *, backend="pgen_vcf", ploidy=2, tracks=(), chroms=None):
    n = n_regions
    bed = pl.DataFrame(
        {
            "chrom": chroms or ["chr1"] * n,
            "chromStart": list(range(0, 100 * n, 100)),
            "chromEnd": list(range(50, 100 * n + 50, 100)),
        }
    )
    meta = Metadata(
        samples=list(samples),
        contigs=["chr1", "chr2"],
        n_regions=n,
        ploidy=ploidy,
    )
    return ConcatInput(
        path=None,
        meta=meta,
        bed=bed,
        n_regions=n,
        n_samples=len(samples),
        backend=backend,
        tracks=list(tracks),
        annot_tracks=[],
        has_dosages=False,
    )


def test_accepts_disjoint_samples_on_sample_axis():
    validate_concat([_mk(["a"], 2), _mk(["b"], 2)], "samples")


def test_rejects_overlapping_samples_on_sample_axis():
    with pytest.raises(ValueError, match="overlapping samples"):
        validate_concat([_mk(["a", "b"], 2), _mk(["b"], 2)], "samples")


def test_rejects_mismatched_regions_on_sample_axis():
    with pytest.raises(ValueError, match="identical regions"):
        validate_concat([_mk(["a"], 2), _mk(["b"], 3)], "samples")


def test_rejects_mismatched_samples_on_region_axis():
    with pytest.raises(ValueError, match="identical samples"):
        validate_concat([_mk(["a"], 2), _mk(["b"], 2)], "regions")


def test_rejects_mixed_backends():
    with pytest.raises(ValueError, match="same variant source"):
        validate_concat(
            [_mk(["a"], 2, backend="pgen_vcf"), _mk(["a"], 2, backend="svar2")],
            "regions",
        )


def test_rejects_mismatched_ploidy():
    with pytest.raises(ValueError, match="ploidy"):
        validate_concat([_mk(["a"], 2, ploidy=2), _mk(["a"], 2, ploidy=1)], "regions")


def test_rejects_mismatched_track_sets():
    with pytest.raises(ValueError, match="track"):
        validate_concat(
            [_mk(["a"], 2, tracks=("atac",)), _mk(["a"], 2, tracks=("dnase",))],
            "regions",
        )


def test_rejects_single_input():
    with pytest.raises(ValueError, match="at least two"):
        validate_concat([_mk(["a"], 2)], "regions")


def test_rejects_duplicate_region_names_on_region_axis():
    a = _mk(["a"], 2)
    b = _mk(["a"], 2)  # identical coordinates -> identical derived names
    a = replace(a, bed=a.bed.with_columns(name=pl.Series(["r0", "r1"])))
    b = replace(b, bed=b.bed.with_columns(name=pl.Series(["r1", "r2"])))
    with pytest.raises(ValueError, match="duplicate region name"):
        validate_concat([a, b], "regions")


def test_identical_coordinates_without_names_are_allowed():
    """Coordinate collisions are fine; only *name* collisions raise."""
    validate_concat([_mk(["a"], 2), _mk(["a"], 2)], "regions")
