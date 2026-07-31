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


def _read_offsets(p: Path):
    import numpy as np

    return np.fromfile(p / "genotypes" / "offsets.npy", dtype=np.int64)


def _read_v_idxs(p: Path):
    import numpy as np
    from genoray._types import V_IDX_TYPE

    return np.fromfile(p / "genotypes" / "variant_idxs.npy", dtype=V_IDX_TYPE)


def test_concat_regions_matches_single_shot_bytes(tmp_path, region_shards):
    """Region concat must be byte-identical: regions are independent and the
    sample set is unchanged."""
    shards, whole = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    assert (_read_offsets(out) == _read_offsets(whole)).all()
    assert (_read_v_idxs(out) == _read_v_idxs(whole)).all()


def test_concat_regions_metadata_matches(tmp_path, region_shards):
    shards, whole = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    import json

    got = json.loads((out / "metadata.json").read_text())
    exp = json.loads((whole / "metadata.json").read_text())
    assert got["samples"] == exp["samples"]
    assert got["n_regions"] == exp["n_regions"]
    assert got["ploidy"] == exp["ploidy"]
    assert got["contigs"] == exp["contigs"]


def test_concat_samples_merges_sample_list(tmp_path, sample_shards):
    shards, whole = sample_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples")

    import json

    got = json.loads((out / "metadata.json").read_text())
    exp = json.loads((whole / "metadata.json").read_text())
    assert got["samples"] == exp["samples"]
    assert got["n_regions"] == exp["n_regions"]


def test_concat_rejects_wrong_axis(tmp_path, region_shards):
    shards, _ = region_shards
    with pytest.raises(ValueError, match="axis must be"):
        gvl.concat(tmp_path / "x.gvl", shards, axis="contigs")


def test_concat_refuses_overwrite_by_default(tmp_path, region_shards):
    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")
    with pytest.raises(FileExistsError):
        gvl.concat(out, shards, axis="regions")


def test_concat_records_variants_fingerprint(tmp_path, region_shards):
    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    import json

    meta = json.loads((out / "metadata.json").read_text())
    fp = meta["variants_fingerprint"]
    assert fp["algorithm"] == "blake2b"
    assert fp["size_bytes"] > 0
    assert len(fp["digest"]) > 0
