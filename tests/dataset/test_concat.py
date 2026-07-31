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


# --- Task 6: svar backend, tracks, annot tracks -----------------------------


@pytest.fixture(scope="session")
def svar_region_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]:
    assert concat_case.svar_path is not None, (
        "concat_case has no svar_path; svar concat coverage is blocked -- see "
        "task-6-report.md"
    )
    d = tmp_path_factory.mktemp("concat_svar")
    bed = gvl.read_bedlike(concat_case.bed_path)
    half = bed.height // 2
    shards = []
    for i, part in enumerate([bed[:half], bed[half:]]):
        p = d / f"shard{i}.gvl"
        gvl.write(p, part, concat_case.svar_path, samples=concat_case.samples)
        shards.append(p)
    whole = d / "whole.gvl"
    gvl.write(whole, bed, concat_case.svar_path, samples=concat_case.samples)
    return shards, whole


def test_concat_svar_regions_matches_single_shot(tmp_path, svar_region_shards):
    import numpy as np

    shards, whole = svar_region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    got = np.fromfile(out / "genotypes" / "offsets.npy", dtype=np.int64)
    exp = np.fromfile(whole / "genotypes" / "offsets.npy", dtype=np.int64)
    assert (got == exp).all()


def test_concat_svar_preserves_link(tmp_path, svar_region_shards):
    import json

    shards, whole = svar_region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    got = json.loads((out / "metadata.json").read_text())
    exp = json.loads((whole / "metadata.json").read_text())
    assert got["svar_link"]["fingerprint"] == exp["svar_link"]["fingerprint"]


def test_concat_regions_reads_equal_to_single_shot(tmp_path, region_shards, reference):
    """The real acceptance check: every merged cell reads identically."""
    import numpy as np

    shards, whole = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    a = gvl.Dataset.open(out, reference).with_seqs("haplotypes")
    b = gvl.Dataset.open(whole, reference).with_seqs("haplotypes")
    assert a.shape == b.shape
    for r in range(a.shape[0]):
        for s in range(a.shape[1]):
            ha, hb = a[r, s], b[r, s]
            assert np.array_equal(ha.to_padded(b"N"), hb.to_padded(b"N")), (r, s)


def test_concat_samples_reads_equal_to_single_shot(tmp_path, sample_shards, reference):
    """Sample axis asserts read equality, NOT byte identity.

    extend_to_length sizes each region's window to the max over the cohort present
    at write time, so a shard's stored variant set can legitimately differ from the
    full write's. Byte identity is not expected here. Per the task brief: if this
    fails, DO NOT weaken the assertion -- a read-level divergence here is a real
    finding, not test flakiness.
    """
    import numpy as np

    shards, whole = sample_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples")

    a = gvl.Dataset.open(out, reference).with_seqs("haplotypes")
    b = gvl.Dataset.open(whole, reference).with_seqs("haplotypes")
    assert a.shape == b.shape
    for r in range(a.shape[0]):
        for s in range(a.shape[1]):
            ha, hb = a[r, s], b[r, s]
            assert np.array_equal(ha.to_padded(b"N"), hb.to_padded(b"N")), (r, s)


# --- Task 6 Correction 1: interleaved-sample coverage for tracks and svar --
#
# `interleaved_sample_shards` (above) only covers the pgen/vcf genotypes path.
# Correction 1's bug (omitting `order=` reintroduces block-concatenation) is
# specific to each store that has its own gather call, so the per-sample track
# gather and the svar offsets gather each need their own interleaved-sample
# case -- a block-layout shard (`sample_shards`) cannot distinguish "order
# handled correctly" from "order silently dropped".


@pytest.fixture(scope="session")
def interleaved_sample_track_shards(tmp_path_factory, concat_case) -> list[Path]:
    """Track-bearing analogue of `interleaved_sample_shards`.

    Same interleaving as `interleaved_sample_shards` (shard 0 = s0+s2, shard 1
    = s1, so the merged sorted order is shard0, shard1, shard0), but with a
    per-sample BigWig track attached so the track gather's `order` handling
    (not just the genotypes gather's) is exercised.
    """
    import pyBigWig

    d = tmp_path_factory.mktemp("concat_sample_track_interleaved")
    bed = gvl.read_bedlike(concat_case.bed_path)
    samples = sorted(concat_case.samples)
    assert samples == ["s0", "s1", "s2"], (
        f"fixture assumes samples s0/s1/s2 to interleave; got {samples}"
    )
    groups = [["s0", "s2"], ["s1"]]

    contig_sizes = [("chr1", 1_300_000), ("chr2", 1_300_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(concat_case.samples):
        bw_path = d / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            # cover the whole contig so every region overlaps regardless of
            # its exact coordinates; value differs per sample so a slot mixup
            # (wrong `order`) would read another sample's value.
            value = float(i + 1)
            bw.addEntries(
                ["chr1", "chr2"],
                [0, 0],
                ends=[1_300_000, 1_300_000],
                values=[value, value],
            )
        bw_paths[sample] = str(bw_path)

    shards = []
    for i, grp in enumerate(groups):
        p = d / f"shard{i}.gvl"
        track = gvl.BigWigs("sig", {s: bw_paths[s] for s in grp})
        gvl.write(p, bed, concat_case.pgen_path, tracks=track, samples=grp)
        shards.append(p)
    return shards


def _read_track(p: Path, name: str):
    import numpy as np

    d = p / "intervals" / name
    return (
        np.fromfile(d / "offsets.npy", dtype=np.int64),
        np.fromfile(d / "starts.npy", dtype=np.int32),
        np.fromfile(d / "ends.npy", dtype=np.int32),
        np.fromfile(d / "values.npy", dtype=np.float32),
    )


def test_concat_samples_track_interleaved_self_consistent(
    tmp_path, interleaved_sample_track_shards
):
    """Per-sample track gather must respect the interleaved merged sample
    order, not a block layout. For every (region, shard-local-sample) slot, the
    merged track's slice at the corresponding merged-sample slot must equal
    the shard's own slice."""
    import numpy as np

    shards = interleaved_sample_track_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples")

    merged_offsets, merged_starts, merged_ends, merged_values = _read_track(out, "sig")
    merged_samples, n_regions, _ploidy = _sample_meta(out)
    s_merged = len(merged_samples)

    for shard in shards:
        shard_offsets, shard_starts, shard_ends, shard_values = _read_track(
            shard, "sig"
        )
        shard_samples, shard_n_regions, _ = _sample_meta(shard)
        assert shard_n_regions == n_regions
        s_shard = len(shard_samples)

        for w, sample in enumerate(shard_samples):
            j = merged_samples.index(sample)
            for r in range(n_regions):
                shard_slot = r * s_shard + w
                merged_slot = r * s_merged + j

                shard_sl = slice(
                    shard_offsets[shard_slot], shard_offsets[shard_slot + 1]
                )
                merged_sl = slice(
                    merged_offsets[merged_slot], merged_offsets[merged_slot + 1]
                )
                assert np.array_equal(
                    shard_starts[shard_sl], merged_starts[merged_sl]
                ), (sample, r, "starts")
                assert np.array_equal(shard_ends[shard_sl], merged_ends[merged_sl]), (
                    sample,
                    r,
                    "ends",
                )
                assert np.array_equal(
                    shard_values[shard_sl], merged_values[merged_sl]
                ), (sample, r, "values")


@pytest.fixture(scope="session")
def interleaved_sample_svar_shards(tmp_path_factory, concat_case) -> list[Path]:
    """svar-backend analogue of `interleaved_sample_shards`: same s0+s2 /
    s1 interleaving, but built from `concat_case.svar_path` so the svar
    offsets gather's `order` handling is exercised (distinct code path from
    the pgen/vcf genotypes gather and the track gather)."""
    assert concat_case.svar_path is not None, (
        "concat_case has no svar_path; svar concat coverage is blocked -- see "
        "task-6-report.md"
    )
    d = tmp_path_factory.mktemp("concat_sample_svar_interleaved")
    bed = gvl.read_bedlike(concat_case.bed_path)
    samples = sorted(concat_case.samples)
    assert samples == ["s0", "s1", "s2"], (
        f"fixture assumes samples s0/s1/s2 to interleave; got {samples}"
    )
    groups = [["s0", "s2"], ["s1"]]

    shards = []
    for i, grp in enumerate(groups):
        p = d / f"shard{i}.gvl"
        gvl.write(p, bed, concat_case.svar_path, samples=grp)
        shards.append(p)
    return shards


def test_concat_samples_svar_interleaved_self_consistent(
    tmp_path, interleaved_sample_svar_shards
):
    """svar's `offsets.npy` holds absolute indices into the shared external
    store, so a correctly-ordered merge is byte-identical per slot (not just
    read-equivalent): for every (region, shard-local-sample, ploid) slot, the
    merged (2, R, S, P) offsets at the corresponding merged-sample slot must
    equal the shard's own slot exactly."""
    import json

    import numpy as np

    shards = interleaved_sample_svar_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples")

    merged_meta = json.loads((out / "genotypes" / "svar_meta.json").read_text())
    _two, R, s_merged, P = merged_meta["shape"]
    merged_offsets = np.fromfile(
        out / "genotypes" / "offsets.npy", dtype=np.int64
    ).reshape(2, R, s_merged, P)
    merged_samples, n_regions, _ = _sample_meta(out)
    assert n_regions == R

    for shard in shards:
        shard_meta = json.loads((shard / "genotypes" / "svar_meta.json").read_text())
        _two, shard_r, s_shard, shard_p = shard_meta["shape"]
        assert shard_r == R
        assert shard_p == P
        shard_offsets = np.fromfile(
            shard / "genotypes" / "offsets.npy", dtype=np.int64
        ).reshape(2, R, s_shard, P)
        shard_samples, _, _ = _sample_meta(shard)

        for w, sample in enumerate(shard_samples):
            j = merged_samples.index(sample)
            assert np.array_equal(
                shard_offsets[:, :, w, :], merged_offsets[:, :, j, :]
            ), f"sample {sample!r}: shard {shard} != merged"
