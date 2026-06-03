"""Tests for atomic gvl.write and format_version stamping."""

import polars as pl
import pytest

import genvarloader as gvl
from genvarloader._dataset._write import DATASET_FORMAT_VERSION, Metadata


def test_metadata_has_format_version_field():
    m = Metadata(samples=["s0"], contigs=["chr1"], n_regions=1)
    # default is None for back-compat; write() stamps the current version
    assert m.format_version is None


def test_dataset_format_version_is_1_0_0():
    assert str(DATASET_FORMAT_VERSION) == "1.0.0"


def test_write_stamps_format_version():
    raw = Metadata(
        samples=["s0"],
        contigs=["chr1"],
        n_regions=1,
        format_version=DATASET_FORMAT_VERSION,
    ).model_dump_json()
    back = Metadata.model_validate_json(raw)
    assert str(back.format_version) == "1.0.0"


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
