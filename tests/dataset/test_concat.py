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


@pytest.fixture(scope="session")
def interleaved_sample_shards(tmp_path_factory, concat_case) -> list[Path]:
    """Sample groups whose merged sorted order interleaves the two datasets.

    ``sample_shards`` above splits alphabetically (``[s0]`` + ``[s1, s2]``), so
    after sorting the merge reduces to a de-facto block layout -- the same
    layout block-concatenation would also produce. That never exercises the
    glue that DERIVES ``order`` from real datasets (``_sample_order``,
    ``_merged_bed``) on a case old block-concatenation code would get wrong.

    Here shard 0 gets ``s0`` and ``s2``, shard 1 gets ``s1`` (which sorts
    between them), so the true merged order is shard0, shard1, shard0 --
    genuinely interleaved, not reducible to a block layout.
    """
    d = tmp_path_factory.mktemp("concat_sample_interleaved")
    bed = gvl.read_bedlike(concat_case.bed_path)
    samples = sorted(concat_case.samples)
    assert samples == ["s0", "s1", "s2"], (
        f"fixture assumes samples s0/s1/s2 to interleave; got {samples}"
    )
    groups = [["s0", "s2"], ["s1"]]

    shards = []
    for i, grp in enumerate(groups):
        p = d / f"shard{i}.gvl"
        gvl.write(p, bed, concat_case.pgen_path, samples=grp)
        shards.append(p)
    return shards


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


def test_concat_regions_matches_single_shot_regions_npy(tmp_path, region_shards):
    """Correction 3: axis="regions" gathers each shard's own stored regions.npy
    (already reflecting that shard's max_jitter/extend_to_length) into merged
    sorted order, rather than recomputing from the raw bed. That must reproduce
    exactly what a single-shot write of the whole bed would have stored."""
    shards, whole = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    import numpy as np

    got = np.load(out / "regions.npy")
    exp = np.load(whole / "regions.npy")
    assert got.dtype == exp.dtype
    assert got.shape == exp.shape
    assert (got == exp).all()


def _load_geno(p: Path):
    return _read_offsets(p), _read_v_idxs(p)


def _sample_meta(p: Path) -> tuple[list[str], int, int]:
    """Return (samples, n_regions, ploidy) from a dataset's metadata.json."""
    import json

    meta = json.loads((p / "metadata.json").read_text())
    return meta["samples"], meta["n_regions"], meta["ploidy"]


def _assert_sample_axis_self_consistent(shards: list[Path], merged: Path) -> None:
    """For every (region, sample, ploid) slot contributed by a shard, the
    merged dataset's variant_idxs slice for that (region, merged-sample-index,
    ploid) slot must equal the shard's own slice for (region,
    shard-sample-index, ploid).

    This is self-consistency against the SOURCE shards, not byte-identity
    against a single-shot `whole` write: extend_to_length sizes each region's
    window to the max haplotype length over the cohort present at write time,
    so a shard's stored variant set can legitimately differ from the full
    write's. Comparing flat slots computed with each dataset's own sample
    count (`S_d`) is what actually exercises the provenance mapping -- a wrong
    `S_d` (e.g. the merged sample count, or dataset 0's count for every
    dataset) misaligns these slices.
    """
    import numpy as np

    merged_offsets, merged_v_idxs = _load_geno(merged)
    merged_samples, n_regions, ploidy = _sample_meta(merged)
    s_merged = len(merged_samples)

    for shard in shards:
        shard_offsets, shard_v_idxs = _load_geno(shard)
        shard_samples, shard_n_regions, shard_ploidy = _sample_meta(shard)
        assert shard_n_regions == n_regions
        assert shard_ploidy == ploidy
        s_shard = len(shard_samples)

        for w, sample in enumerate(shard_samples):
            j = merged_samples.index(sample)
            for r in range(n_regions):
                for p in range(ploidy):
                    shard_slot = (r * s_shard + w) * ploidy + p
                    merged_slot = (r * s_merged + j) * ploidy + p

                    shard_slice = shard_v_idxs[
                        shard_offsets[shard_slot] : shard_offsets[shard_slot + 1]
                    ]
                    merged_slice = merged_v_idxs[
                        merged_offsets[merged_slot] : merged_offsets[merged_slot + 1]
                    ]
                    assert np.array_equal(shard_slice, merged_slice), (
                        f"sample {sample!r} region {r} ploid {p}: shard "
                        f"{shard} slot {shard_slot} != merged slot {merged_slot}"
                    )


def test_concat_samples_genotypes_self_consistent(tmp_path, sample_shards):
    """The dangerous S_d failure mode (using the wrong per-dataset sample
    count when computing a source flat slot) is otherwise only unit-tested.
    `sample_shards` splits 3 samples unevenly (1 vs 2 -- `half = 3 // 2 = 1`),
    which is exactly the S_d asymmetry a wrong per-dataset sample count would
    misalign."""
    shards, _whole = sample_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples")
    _assert_sample_axis_self_consistent(shards, out)


def test_concat_samples_interleaved_genotypes_self_consistent(
    tmp_path, interleaved_sample_shards
):
    """Integration-level proof that Correction 1's order-derivation glue
    (`_sample_order`, `_merged_bed`) is exercised on a genuinely interleaved
    merge -- not one that reduces to block-concatenation after sorting, which
    is all `sample_shards` above covers."""
    out = tmp_path / "merged.gvl"
    gvl.concat(out, interleaved_sample_shards, axis="samples")
    _assert_sample_axis_self_consistent(interleaved_sample_shards, out)
