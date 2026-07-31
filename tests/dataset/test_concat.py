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


# --- Task 6 fix round 1 -----------------------------------------------------
#
# Finding 1: annot tracks on axis="samples" were linked from input #0 with no
# verification that the inputs' annot data actually agrees, even though
# Correction 3 (regions.npy = elementwise max of chromEnd) exists precisely
# because per-input extend_to_length CAN diverge -- silently copying input
# #0's (potentially narrower) annot data would under-cover the merged
# dataset's reads. Fixed by `_assert_annot_track_matches` (fingerprint-compare
# before copy, per the design spec's `annot_intervals/<n>/` rule).
#
# Finding 2: annot tracks had zero test coverage, and region-axis per-sample
# tracks were untested (only the sample axis was covered). Both addressed
# below with fixtures constructed to be genuinely discriminating, not
# block-layout shards that happen to pass either way.


def _make_annot_df(bed, offset: float = 0.0):
    """A bed-like annotation frame with one 1bp interval per region, anchored
    at that region's own chromStart so it's guaranteed to overlap (any
    interval `[chromStart, chromStart+1)` is inside `[chromStart, chromEnd)`
    for every region here, since every region spans more than 1bp). `score`
    encodes the region's chromStart (offset by `offset`), which makes a
    region mixup detectable rather than incidentally passing on equal scores.
    """
    import polars as pl

    return bed.select(
        "chrom",
        "chromStart",
        (pl.col("chromStart") + 1).alias("chromEnd"),
        (pl.col("chromStart").cast(pl.Float64) + offset).alias("score"),
    )


def _read_annot(p: Path, name: str):
    import numpy as np

    d = p / "annot_intervals" / name
    return (
        np.fromfile(d / "offsets.npy", dtype=np.int64),
        np.fromfile(d / "starts.npy", dtype=np.int32),
        np.fromfile(d / "ends.npy", dtype=np.int32),
        np.fromfile(d / "values.npy", dtype=np.float32),
    )


def _ds_of_sorted_from_output(out: Path, part_lens: list[int]) -> list[int]:
    """Recover, for an axis="regions" merge, which input each on-disk
    (sorted) region row came from.

    Inverts `input_regions.arrow`'s `r_idx_map` (input-row-order ->
    sorted/on-disk-order), the same information `_region_order` derives
    internally, rather than hand-computing or assuming a sort order. Used to
    PROVE a fixture is genuinely interleaved instead of trusting a comment.
    """
    import numpy as np
    import polars as pl

    ib = pl.read_ipc(out / "input_regions.arrow")
    r_map = ib["r_idx_map"].to_numpy()
    ds_id = np.concatenate([np.full(n, i) for i, n in enumerate(part_lens)])
    ds_of_sorted = np.empty(len(r_map), dtype=np.int64)
    ds_of_sorted[r_map] = ds_id
    return ds_of_sorted.tolist()


def _n_runs(seq: list) -> int:
    """Count maximal contiguous same-value runs. A block-concatenation layout
    has exactly `n_datasets` runs; more than that means genuine interleaving."""
    if not seq:
        return 0
    return 1 + sum(a != b for a, b in zip(seq, seq[1:]))


def _region_keys(path: Path) -> list[tuple[int, int]]:
    """A dataset's own on-disk region order, as `(chrom_idx, chromStart)` keys
    (columns 0 and 1 of `regions.npy`, per `bed_to_regions`). Region identity
    by coordinate, not row position: `gvl.write` always stores regions sorted
    on disk, a permutation of whatever row order the caller's bed was in."""
    import numpy as np

    regions = np.load(path / "regions.npy")
    return [(int(c), int(s)) for c, s in zip(regions[:, 0], regions[:, 1])]


def _region_key_to_row(path: Path) -> dict[tuple[int, int], int]:
    return {key: i for i, key in enumerate(_region_keys(path))}


# --- Finding 2b: region-axis per-sample tracks, currently unverified -------


@pytest.fixture(scope="session")
def region_track_shards(tmp_path_factory, concat_case) -> tuple[list[Path], list[int]]:
    """Region-axis analogue of `interleaved_sample_track_shards`.

    `concat_case`'s bed is NOT already in genomic sort order (verified via
    `_ds_of_sorted_from_output` in the test below, not assumed here) -- so
    unlike the sample axis, no special row-selection trick is needed to
    produce a genuinely interleaved merge: a plain contiguous split (as
    `region_shards` already uses) interleaves once `_prep_bed` re-sorts it
    into on-disk order. A per-sample BigWig track is attached so the track
    gather's `order` handling is exercised on the region axis, which
    (pre-fix-round-1) only had sample-axis coverage.
    """
    import pyBigWig

    d = tmp_path_factory.mktemp("concat_region_track")
    bed = gvl.read_bedlike(concat_case.bed_path)
    half = bed.height // 2
    assert half >= 2, "need >=4 regions for a meaningful interleave check"
    parts = [bed[:half], bed[half:]]

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
    track = gvl.BigWigs("sig", bw_paths)

    shards = []
    for i, part in enumerate(parts):
        p = d / f"shard{i}.gvl"
        gvl.write(
            p, part, concat_case.pgen_path, tracks=track, samples=concat_case.samples
        )
        shards.append(p)
    return shards, [part.height for part in parts]


def test_concat_regions_track_interleaved_self_consistent(
    tmp_path, region_track_shards
):
    """Per-sample track gather on the REGION axis. First proves (does not
    assume) the fixture is genuinely interleaved -- more runs of
    dataset-of-origin than there are shards -- then checks every shard's own
    (region, sample) track slice against the merged dataset's slice at the
    corresponding on-disk region row.

    Region rows are located by their own ``(chrom_idx, chromStart)`` coordinate
    in each dataset's stored ``regions.npy``, not by raw input-row position:
    ``gvl.write`` always stores regions in its own sorted on-disk order, which
    is a permutation of the raw bed row order fed into that write call, so a
    shard's own on-disk row index is generally NOT the same as its position in
    the raw bed part used to build the fixture.
    """
    import numpy as np

    shards, part_lens = region_track_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    ds_of_sorted = _ds_of_sorted_from_output(out, part_lens)
    assert _n_runs(ds_of_sorted) > len(shards), (
        "fixture reduced to a block layout; not a discriminating interleave test"
    )

    merged_offsets, merged_starts, merged_ends, merged_values = _read_track(out, "sig")
    merged_samples, _n_regions, _ = _sample_meta(out)
    s = len(merged_samples)
    merged_key_to_row = _region_key_to_row(out)

    for shard, n in zip(shards, part_lens):
        shard_offsets, shard_starts, shard_ends, shard_values = _read_track(
            shard, "sig"
        )
        shard_samples, shard_n_regions, _ = _sample_meta(shard)
        assert shard_samples == merged_samples
        assert shard_n_regions == n

        for shard_r, key in enumerate(_region_keys(shard)):
            merged_r = merged_key_to_row[key]
            for w in range(s):
                shard_slot = shard_r * s + w
                merged_slot = merged_r * s + w
                shard_sl = slice(
                    shard_offsets[shard_slot], shard_offsets[shard_slot + 1]
                )
                merged_sl = slice(
                    merged_offsets[merged_slot], merged_offsets[merged_slot + 1]
                )
                assert np.array_equal(
                    shard_starts[shard_sl], merged_starts[merged_sl]
                ), (shard, shard_r, w, "starts")
                assert np.array_equal(shard_ends[shard_sl], merged_ends[merged_sl]), (
                    shard,
                    shard_r,
                    w,
                    "ends",
                )
                assert np.array_equal(
                    shard_values[shard_sl], merged_values[merged_sl]
                ), (shard, shard_r, w, "values")


# --- Finding 2a: annot tracks, previously zero coverage ---------------------


@pytest.fixture(scope="session")
def region_annot_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]:
    """Region-axis annot-track fixture: same contiguous split as
    `region_track_shards` (and, per that fixture, genuinely interleaved after
    sorting -- checked again in the test rather than assumed), with a
    `annot_tracks={"ann": ...}` source shared by both shards and the
    single-shot oracle."""
    d = tmp_path_factory.mktemp("concat_region_annot")
    bed = gvl.read_bedlike(concat_case.bed_path)
    half = bed.height // 2
    assert half >= 2, "need >=4 regions for a meaningful interleave check"
    parts = [bed[:half], bed[half:]]
    annot_df = _make_annot_df(bed)

    shards = []
    for i, part in enumerate(parts):
        p = d / f"shard{i}.gvl"
        gvl.write(
            p,
            part,
            concat_case.pgen_path,
            annot_tracks={"ann": annot_df},
            samples=concat_case.samples,
        )
        shards.append(p)

    whole = d / "whole.gvl"
    gvl.write(
        whole,
        bed,
        concat_case.pgen_path,
        annot_tracks={"ann": annot_df},
        samples=concat_case.samples,
    )
    return shards, whole


def test_concat_regions_annot_tracks_matches_single_shot(tmp_path, region_annot_shards):
    """Region-axis annot gather must reproduce a single-shot write exactly:
    annot tracks are sample-independent and the region axis is
    disjoint-and-gathered (kind (a) over R per the design spec), so byte
    identity -- not just read-equality -- is the right bar here."""
    import numpy as np

    shards, whole = region_annot_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    got = _read_annot(out, "ann")
    exp = _read_annot(whole, "ann")
    for g, e, label in zip(got, exp, ("offsets", "starts", "ends", "values")):
        assert np.array_equal(g, e), label


@pytest.fixture(scope="session")
def sample_annot_matching_shards(
    tmp_path_factory, concat_case
) -> tuple[list[Path], Path]:
    """Sample-axis annot-track fixture where both shards legitimately carry
    the SAME annot source (the expected/supported case): the fingerprint
    compare added for Finding 1 must NOT raise here, and the merged data must
    equal a single-shot write's."""
    d = tmp_path_factory.mktemp("concat_sample_annot_matching")
    bed = gvl.read_bedlike(concat_case.bed_path)
    samples = sorted(concat_case.samples)
    half = len(samples) // 2
    assert half >= 1, "need >=2 samples to shard"
    groups = [samples[:half], samples[half:]]
    annot_df = _make_annot_df(bed)

    shards = []
    for i, grp in enumerate(groups):
        p = d / f"shard{i}.gvl"
        gvl.write(
            p, bed, concat_case.pgen_path, annot_tracks={"ann": annot_df}, samples=grp
        )
        shards.append(p)

    whole = d / "whole.gvl"
    gvl.write(
        whole,
        bed,
        concat_case.pgen_path,
        annot_tracks={"ann": annot_df},
        samples=samples,
    )
    return shards, whole


def test_concat_samples_annot_tracks_matches_single_shot(
    tmp_path, sample_annot_matching_shards
):
    """The fingerprint compare added for Finding 1 must not raise a false
    positive when the inputs' annot data genuinely agrees, and the linked
    result must match a single-shot write."""
    import numpy as np

    shards, whole = sample_annot_matching_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples")  # must not raise

    got = _read_annot(out, "ann")
    exp = _read_annot(whole, "ann")
    for g, e, label in zip(got, exp, ("offsets", "starts", "ends", "values")):
        assert np.array_equal(g, e), label


@pytest.fixture(scope="session")
def sample_annot_mismatched_shards(tmp_path_factory, concat_case) -> list[Path]:
    """Sample-axis annot-track fixture where the two shards are deliberately
    given DIFFERENT annot source content for the same track name `"ann"`.

    This is a direct, deterministic trigger for Finding 1's raise path: it
    does not depend on `extend_to_length` organically diverging per-input
    chromEnd (which this session's small fixture does not reliably do -- see
    `test_concat_samples_reads_equal_to_single_shot`'s pass), so the compare
    itself is exercised regardless of whether that specific root cause fires
    for this dataset.
    """
    d = tmp_path_factory.mktemp("concat_sample_annot_mismatched")
    bed = gvl.read_bedlike(concat_case.bed_path)
    samples = sorted(concat_case.samples)
    half = len(samples) // 2
    assert half >= 1, "need >=2 samples to shard"
    groups = [samples[:half], samples[half:]]
    annot_dfs = [_make_annot_df(bed, offset=0.0), _make_annot_df(bed, offset=100.0)]

    shards = []
    for i, grp in enumerate(groups):
        p = d / f"shard{i}.gvl"
        gvl.write(
            p,
            bed,
            concat_case.pgen_path,
            annot_tracks={"ann": annot_dfs[i]},
            samples=grp,
        )
        shards.append(p)
    return shards


def test_concat_samples_annot_tracks_mismatch_raises(
    tmp_path, sample_annot_mismatched_shards
):
    """Finding 1: divergent annot data across inputs must raise, not be
    silently linked from input #0 (which would silently serve input #0's
    stale/mismatched data for every sample, including those from input #1)."""
    shards = sample_annot_mismatched_shards
    out = tmp_path / "merged.gvl"
    with pytest.raises(ValueError, match=r"annot track 'ann'") as excinfo:
        gvl.concat(out, shards, axis="samples")
    assert "input #1" in str(excinfo.value)
    assert not out.exists()


def test_open_rejects_mutated_variants_arrow(tmp_path, region_shards, reference):
    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    # Break the hardlink, then corrupt the copy so the fingerprint mismatches.
    va = out / "genotypes" / "variants.arrow"
    data = bytearray(va.read_bytes())
    va.unlink()
    data[:16] = b"\x00" * 16
    va.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="variants.arrow"):
        gvl.Dataset.open(out, reference)


def test_open_accepts_absent_fingerprint(tmp_path, region_shards, reference):
    """Datasets written before the field exists must still open."""
    import json

    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    meta = json.loads((out / "metadata.json").read_text())
    meta["variants_fingerprint"] = None
    (out / "metadata.json").write_text(json.dumps(meta))

    gvl.Dataset.open(out, reference)  # must not raise


def test_open_rejects_missing_variants_arrow_with_recorded_fingerprint(
    tmp_path, region_shards, reference
):
    """A dataset that records a `variants_fingerprint` requires
    genotypes/variants.arrow to be present -- a deleted hardlink target must
    raise loudly, not silently skip verification."""
    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    va = out / "genotypes" / "variants.arrow"
    va.unlink()

    with pytest.raises(ValueError, match="variants.arrow is missing"):
        gvl.Dataset.open(out, reference)
