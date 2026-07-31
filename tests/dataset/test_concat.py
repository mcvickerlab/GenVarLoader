"""Integration tests for gvl.concat.

Oracle: shard a dataset, concat the shards, and compare against a single-shot
gvl.write of the whole thing.
"""

from pathlib import Path

import pytest

import genvarloader as gvl


@pytest.fixture(scope="session")
def concat_case(synthetic_case):
    """Reuse the shared synthetic case (VCF + PGEN + SVAR + BED + samples)."""
    return synthetic_case


@pytest.fixture(scope="session")
def region_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]:
    """Split the BED in half by row; write one dataset per half plus the whole."""
    d = tmp_path_factory.mktemp("concat_region")
    bed = gvl.read_bedlike(concat_case.bed_path)
    half = bed.height // 2
    assert half >= 1, "need >=2 regions to shard"
    parts = [bed[:half], bed[half:]]

    shards = []
    for i, part in enumerate(parts):
        p = d / f"shard{i}.gvl"
        gvl.write(p, part, concat_case.pgen_path, samples=concat_case.samples)
        shards.append(p)

    whole = d / "whole.gvl"
    gvl.write(whole, bed, concat_case.pgen_path, samples=concat_case.samples)
    return shards, whole


@pytest.fixture(scope="session")
def sample_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]:
    """Split samples in half; write one dataset per half plus the whole."""
    d = tmp_path_factory.mktemp("concat_sample")
    bed = gvl.read_bedlike(concat_case.bed_path)
    samples = sorted(concat_case.samples)
    half = len(samples) // 2
    assert half >= 1, "need >=2 samples to shard"
    groups = [samples[:half], samples[half:]]

    shards = []
    for i, grp in enumerate(groups):
        p = d / f"shard{i}.gvl"
        gvl.write(p, bed, concat_case.pgen_path, samples=grp)
        shards.append(p)

    whole = d / "whole.gvl"
    gvl.write(whole, bed, concat_case.pgen_path, samples=samples)
    return shards, whole


def test_region_shards_fixture_builds(region_shards):
    shards, whole = region_shards
    assert len(shards) == 2
    assert all((p / "metadata.json").exists() for p in shards)
    assert (whole / "metadata.json").exists()


def test_sample_shards_fixture_builds(sample_shards):
    shards, whole = sample_shards
    assert len(shards) == 2
    assert all((p / "metadata.json").exists() for p in shards)
    assert (whole / "metadata.json").exists()
