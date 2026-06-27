"""Tests for atomic gvl.write and format_version stamping."""

import json
from pathlib import Path

import polars as pl
import pytest

import genvarloader as gvl
from genvarloader._dataset._write import DATASET_FORMAT_VERSION, Metadata


def test_metadata_has_format_version_field():
    m = Metadata(samples=["s0"], contigs=["chr1"], n_regions=1)
    # default is None for back-compat; write() stamps the current version
    assert m.format_version is None


def test_dataset_format_version_is_2_0_0():
    assert str(DATASET_FORMAT_VERSION) == "2.0.0"


def test_write_stamps_format_version():
    raw = Metadata(
        samples=["s0"],
        contigs=["chr1"],
        n_regions=1,
        format_version=DATASET_FORMAT_VERSION,
    ).model_dump_json()
    back = Metadata.model_validate_json(raw)
    assert str(back.format_version) == "2.0.0"


def test_write_is_atomic_no_temp_left(phased_vcf_gvl):
    # The phased_vcf_gvl fixture already exercised gvl.write; assert no temp dirs
    # were left beside it.
    parent = phased_vcf_gvl.parent
    assert list(parent.glob(f"{phased_vcf_gvl.name}.tmp.*")) == []
    assert list(parent.glob(f"{phased_vcf_gvl.name}.old.*")) == []


def test_overwrite_false_existing_raises(synthetic_case, tmp_path):
    """Write a minimal dataset once, then assert a second write raises FileExistsError."""
    dest = tmp_path / "test_overwrite.gvl"

    # Use the VCF from the synthetic case as a variant source
    vcf_path = synthetic_case.vcf_path
    bed = synthetic_case.regions.select(
        chrom=pl.col("chrom"),
        chromStart=pl.col("start"),
        chromEnd=pl.col("end"),
    ).head(2)

    # First write should succeed
    gvl.write(
        path=dest,
        bed=bed,
        variants=str(vcf_path),
        overwrite=False,
    )
    assert dest.exists()

    # Second write to the same path with overwrite=False must raise FileExistsError
    with pytest.raises(FileExistsError):
        gvl.write(
            path=dest,
            bed=bed,
            variants=str(vcf_path),
            overwrite=False,
        )


def test_format_version_stamped_on_disk(synthetic_case, tmp_path):
    """gvl.write stamps format_version in metadata.json on a real write."""
    dest = tmp_path / "test_format_version.gvl"
    bed = synthetic_case.regions.select(
        chrom=pl.col("chrom"),
        chromStart=pl.col("start"),
        chromEnd=pl.col("end"),
    ).head(2)

    gvl.write(
        path=dest,
        bed=bed,
        variants=str(synthetic_case.vcf_path),
        overwrite=False,
    )

    meta = json.loads((dest / "metadata.json").read_text())
    assert meta["format_version"] == "2.0.0"


def test_failure_leaves_no_partial_artifacts(synthetic_case, tmp_path):
    """A mid-write failure cleans up: no dest dir and no .tmp.* sibling left."""
    dest = tmp_path / "test_failure_atomic.gvl"
    bed = synthetic_case.regions.select(
        chrom=pl.col("chrom"),
        chromStart=pl.col("start"),
        chromEnd=pl.col("end"),
    ).head(2)

    # Pass a sample name that doesn't exist in the VCF — raises ValueError after
    # atomic_dir has created the temp directory and written input_regions.arrow.
    with pytest.raises(ValueError, match="not found in variants or tracks"):
        gvl.write(
            path=dest,
            bed=bed,
            variants=str(synthetic_case.vcf_path),
            samples=["NOT_A_REAL_SAMPLE"],
            overwrite=False,
        )

    assert not dest.exists()
    assert list(dest.parent.glob(f"{dest.name}.tmp.*")) == []


def test_empty_sample_intersection_raises_clear_error(
    synthetic_case, bigwig_dir: Path, tmp_path
):
    """gvl.write raises a clear ValueError when variants and tracks share no samples.

    Regression test for https://github.com/mcvickerlab/GenVarLoader/issues/225.
    Without the guard, the crash was an opaque array-geometry error deep inside
    _write_from_svar / Ragged.from_offsets with no mention of 'samples'.
    """
    # BigWig samples are "sample_0" / "sample_1"; VCF samples are different IDs.
    bw = gvl.BigWigs(
        "signal",
        {
            "sample_0": str(bigwig_dir / "sample_0.bw"),
            "sample_1": str(bigwig_dir / "sample_1.bw"),
        },
    )
    bed = synthetic_case.regions.select(
        chrom=pl.col("chrom"),
        chromStart=pl.col("start"),
        chromEnd=pl.col("end"),
    ).head(1)

    with pytest.raises(ValueError, match="No samples remain after intersecting"):
        gvl.write(
            path=tmp_path / "empty_intersection.gvl",
            bed=bed,
            variants=str(synthetic_case.vcf_path),
            tracks=[bw],
        )
