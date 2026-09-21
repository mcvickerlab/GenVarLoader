"""Rust seam for VCF/PGEN mixed variants+tracks (issue #375, Track B).

`window_realign_inputs` is the pymethod the Python mixed path uses to size a
window's deletion-extended track query before pulling the window's first
batch. Task 6 consumes it; this file pins its contract.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pyBigWig
import pytest

import genvarloader as gvl
from genvarloader.genvarloader import RecordStreamEngine

from tests.dataset._mixed_assertions import _assert_cell_equal, _assert_haps_cell_equal

BACKENDS = ("vcf", "pgen")


def test_window_realign_inputs_matches_decode_and_csr_is_well_formed(
    vcf_snp_ins_del_multi,
):
    f = vcf_snp_ins_del_multi
    contig_len = int(f.regions["chromEnd"][0])
    ref_seq = "".join(f.fasta.read_text().splitlines()[1:])

    eng = RecordStreamEngine(
        "vcf",
        str(f.vcf),
        f.sample_names,
        f.ploidy,
        [f.contig],
        [ref_seq.encode()],
        [],
        [],
        [],
        [],
        [],
        None,  # fasta_path=None -- matches gvl.write's VCF parity (no read-time left-align)
        ord("N"),
        False,
        32,
        -1,  # output_length: ragged (unused by window_realign_inputs/debug_decode_window)
    )

    v_starts, ilens, geno_v_idxs, geno_offsets = eng.window_realign_inputs(
        0, [0], [contig_len], 0, f.n_samples
    )
    d_v_starts, d_ilens, _alt, _alt_off = eng.debug_decode_window(
        0, [0], [contig_len], 0, f.n_samples
    )

    # Same decode path -- the variant table must agree exactly.
    np.testing.assert_array_equal(v_starts, np.asarray(d_v_starts, np.int32))
    np.testing.assert_array_equal(ilens, np.asarray(d_ilens, np.int32))

    # Dtypes are part of the contract Task 6 relies on.
    assert v_starts.dtype == np.int32
    assert ilens.dtype == np.int32
    assert geno_v_idxs.dtype == np.int32
    assert geno_offsets.dtype == np.int64

    # CSR is per-hap over the sample sub-range, not per (region, sample) row.
    assert geno_offsets.shape == (f.n_samples * f.ploidy + 1,)
    assert geno_offsets[0] == 0
    assert geno_offsets[-1] == geno_v_idxs.size
    assert np.all(np.diff(geno_offsets) >= 0), "CSR offsets must be non-decreasing"

    # Every CSR value indexes a real column of this window's variant table.
    if geno_v_idxs.size:
        assert geno_v_idxs.min() >= 0
        assert geno_v_idxs.max() < v_starts.size

    # The fixture carries indels, so the window must actually contain variants --
    # otherwise every assertion above is vacuously true.
    assert v_starts.size > 0
    assert geno_v_idxs.size > 0
    assert np.any(ilens != 0), "fixture must contain indels for this seam to matter"

    # M1 (#375 Track B fix round 1): a SINGLE-region window can't distinguish a
    # correct per-hap CSR from a wrong per-(region, sample) CSR -- both produce
    # the exact same `(n_samples * ploidy + 1,)` shape when there is only one
    # region. Query the SAME sample sub-range but split across TWO disjoint
    # regions of the same contig instead: under a (wrong) per-(region, sample)
    # CSR the offsets array would have shape `2 * n_samples * ploidy + 1`
    # (one CSR segment per region), so this is the one case that actually
    # discriminates the two implementations.
    split = contig_len // 2
    v_starts_2r, ilens_2r, geno_v_idxs_2r, geno_offsets_2r = eng.window_realign_inputs(
        0, [0, split], [split, contig_len], 0, f.n_samples
    )
    assert geno_offsets_2r.shape == (f.n_samples * f.ploidy + 1,), (
        "geno_offsets must stay per-hap over the sample sub-range regardless of "
        "region count -- a per-(region, sample) CSR would produce "
        f"{2 * f.n_samples * f.ploidy + 1} here instead"
    )
    assert geno_offsets_2r[0] == 0
    assert geno_offsets_2r[-1] == geno_v_idxs_2r.size
    assert np.all(np.diff(geno_offsets_2r) >= 0)
    if geno_v_idxs_2r.size:
        assert geno_v_idxs_2r.min() >= 0
        assert geno_v_idxs_2r.max() < v_starts_2r.size
    # The two-region window still covers the whole contig, so its variant table
    # must match the single-region call's table exactly.
    np.testing.assert_array_equal(v_starts_2r, v_starts)
    np.testing.assert_array_equal(ilens_2r, ilens)


def test_window_realign_inputs_matches_before_and_during_producer(
    pgen_snp_ins_del_multi,
):
    """M2 (#375 Track B fix round 1): pins the H1 fix -- `window_realign_inputs`
    must return the SAME result whether called before the producer thread has
    started or while it is concurrently decoding other jobs. `PgenWindowFiller`
    mutates a single shared pgenlib reader (`apply_sample_subset`) and releases
    the GIL before reading, so an unserialized concurrent decode from this
    method (the consumer thread) racing the producer's own decode could
    silently swap in the wrong sample columns, or -- if the GIL were held
    across the call -- deadlock against the producer entirely (`reader_lock`
    needs the GIL too). Several identical jobs are registered so the producer
    keeps actively decoding (prefetching into its free-slot pool) right after
    the first `next_batch()` call returns, maximizing the chance this test
    actually overlaps the two decodes rather than merely running twice
    sequentially.
    """
    f = pgen_snp_ins_del_multi
    contig_len = int(f.regions["chromEnd"][0])
    ref_seq = "".join(f.fasta.read_text().splitlines()[1:])

    n_jobs = 6
    eng = RecordStreamEngine(
        "pgen",
        str(f.pgen),
        f.sample_names,
        f.ploidy,
        [f.contig],
        [ref_seq.encode()],
        [0] * n_jobs,  # job_contig_idx
        [[0]] * n_jobs,  # job_region_starts
        [[contig_len]] * n_jobs,  # job_region_ends
        [0] * n_jobs,  # job_s_lo
        [f.n_samples] * n_jobs,  # job_s_hi
        None,  # fasta_path=None -- matches gvl.write's PGEN parity (no read-time left-align)
        ord("N"),
        False,
        1,  # batch_size=1 -- many small batches, so the producer stays busy decoding
        -1,  # output_length: ragged
    )

    args = (0, [0], [contig_len], 0, f.n_samples)

    # 1. Baseline: call before the producer thread exists at all.
    before = eng.window_realign_inputs(*args)

    # 2. Start consuming -- this spawns the producer thread (`ensure_started`),
    # which immediately begins prefetching windows for the queued jobs ahead of
    # what we've consumed.
    batch = eng.next_batch()
    assert batch is not None, "fixture must yield at least one batch"

    # 3. Call again while the producer is (most likely) still actively decoding
    # one of the remaining queued jobs on its own thread.
    during = eng.window_realign_inputs(*args)

    # 4. Must be identical -- deterministic and race-free.
    for name, b, d in zip(
        ("v_starts", "ilens", "geno_v_idxs", "geno_offsets"), before, during
    ):
        np.testing.assert_array_equal(
            b, d, err_msg=f"{name} mismatch pre/during producer"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_tracks_fixture_writes_a_usable_oracle(
    streaming_record_tracks_fixture, backend
):
    """The Task 6/8 fixture actually produces a written mixed oracle.

    Task 6 consumes this fixture for its CSR-replication tests and Task 8 for
    byte-identical parity; this smoke test is what makes the fixture itself
    reviewable now rather than dead weight until then.
    """
    f = streaming_record_tracks_fixture(backend)

    ds = gvl.Dataset.open(f.dataset_path, f.reference_path)

    # tracks= was passed [zeta, alpha]; the written axis sorts. A fixture that
    # silently preserved argument order would defeat every downstream test
    # that indexes the track axis by position.
    assert ds.available_tracks == ["alpha", "zeta"]

    # The dataset's public sample order is what the bigwigs and table were
    # keyed by -- if these disagree, every track value in Tasks 6 and 8 is off
    # by a sample permutation.
    assert list(ds.samples) == f.samples

    # The real risk the deleted assertions were reaching for: track files keyed
    # by names that do not line up with the dataset's public sample order. That
    # shows up as samples sharing track values, so compare two samples' tracks
    # in the same region. The fixture gives every (sample, contig, bin) a
    # distinct value on purpose, so equal rows here mean a mis-keyed fixture --
    # which would silently corrupt every parity test in Tasks 6 and 8.
    assert len(f.samples) >= 2, "need two samples to detect a sample-keying bug"
    # `ds` has both a reference and tracks, so indexing returns the same
    # `(haps, tracks)` 2-tuple the parity tests in `test_streaming_tracks.py`
    # unpack (`exp_haps, exp_tracks = written[r, s]`) -- only the second half
    # is the track data this assertion cares about. The per-cell track is
    # itself Ragged (per-ploidy realigned length can differ within one
    # sample when its two haplotype copies carry different indels), so
    # `np.asarray()` cannot densify it -- compare via `.to_packed()`'s flat
    # `.lengths`/`.data` instead, which is well-defined regardless.
    #
    # Region 2 specifically (not region 0): region 0 has indels that already
    # make samples 0 and 1 differ in realigned LENGTH regardless of track
    # values, so an identical-length check there would pass on the length
    # difference alone and never reach the values -- non-falsifiable against
    # a mis-keyed fixture (proven by mutation, see H2 in the fix-round-1
    # report). Region 2 is the fixture's variant-free region, where every
    # sample's per-hap length is equal, so only the track VALUES can make
    # this assertion pass or fail.
    _, tracks0 = ds[2, 0]
    _, tracks1 = ds[2, 1]
    packed0, packed1 = tracks0.to_packed(), tracks1.to_packed()
    identical = np.array_equal(packed0.lengths, packed1.lengths) and np.array_equal(
        packed0.data, packed1.data
    )
    assert not identical, (
        "samples 0 and 1 have identical track values -- the fixture's bigwigs "
        "or table are keyed by the wrong sample names"
    )

    assert ds.n_regions == f.bed.height


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_window_csr_replicates_across_regions(
    streaming_record_tracks_fixture, backend
):
    """The engine's CSR is per (sample, ploid) for the whole window; the mixed
    path must replicate it across the window's regions in C-order (region,
    sample). Checked against an INDEPENDENT source -- the engine's own CSR,
    fetched directly via `window_realign_inputs` -- rather than against a
    restatement of the mixed path's own arithmetic: a transposed `hap_of_row`
    (region-major <-> sample-major) silently reads another sample's genotypes
    and must fail this test, not just an un-replicated (wrong-length) one.

    Uses the fixture's full 3-region window with all 3 samples (n_reg == n_s
    == 3): a transposition is only a genuine permutation of hap indices when
    the two axes have equal extent, so a smaller r_idx slice would let a
    transposed implementation pass by coincidence.
    """
    f = streaming_record_tracks_fixture(backend)
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")
    b = sds._backend
    P = sds.ploidy
    n_reg, n_s = len(sds._regions), sds.n_samples
    assert n_reg == n_s == 3, (
        "test assumes the fixture's full 3-region/3-sample window so a "
        "transposed hap_of_row is a genuine permutation -- fixture changed?"
    )
    r_idx = np.arange(n_reg, dtype=np.intp)
    s_idx = np.arange(n_s, dtype=np.intp)
    t_starts = np.ascontiguousarray(sds._regions[r_idx, 1], np.int32)
    t_ends = np.ascontiguousarray(sds._regions[r_idx, 2], np.int32)
    row_starts = np.repeat(t_starts, n_s).astype(np.int32)
    row_ends = np.repeat(t_ends, n_s).astype(np.int32)
    state, t_ends_ext = b.mixed_realign_window(
        r_idx, s_idx, t_starts, t_ends, row_starts, row_ends
    )
    assert state.geno_offsets.shape == (2, n_reg * n_s * P)
    assert state.diffs.shape == (n_reg * n_s, P)
    assert t_ends_ext.shape == (n_reg,)
    assert (t_ends_ext >= t_ends).all()

    # Independent source of truth: query the engine directly for this exact
    # (contig, t_starts/t_ends, sample sub-range) rather than trusting the
    # mixed path's own replication of it.
    contig_idx = int(b._regions[r_idx[0], 0])
    engine = b._mixed_engine()
    _, _, _, csr = engine.window_realign_inputs(
        contig_idx,
        np.ascontiguousarray(t_starts, np.uint32).tolist(),
        np.ascontiguousarray(t_ends, np.uint32).tolist(),
        int(s_idx[0]),
        int(s_idx[-1]) + 1,
    )
    csr = np.ascontiguousarray(csr, np.int64)
    assert csr.shape == (n_s * P + 1,)

    # Every (region, sample, ploid) row's genotype slice must equal the
    # engine's own slice for hap `si * ploidy + p` -- i.e. every region block
    # is an exact replica of the engine's per-hap CSR, sample-major within
    # the block. `bi = r * n_s + si` is the drive's documented row order
    # (region-major, sample-minor; see `_streaming.py`'s
    # `_record_mixed_realign_window` docstring and `row_starts`/`row_ends`'s
    # construction above via `np.repeat(t_starts, n_s)`), computed here from
    # scratch rather than reused from the code under test.
    for r in range(n_reg):
        for si in range(n_s):
            bi = r * n_s + si
            for p in range(P):
                col = state.geno_offset_idx[bi, p]
                expected = (int(csr[si * P + p]), int(csr[si * P + p + 1]))
                actual = (
                    int(state.geno_offsets[0, col]),
                    int(state.geno_offsets[1, col]),
                )
                assert actual == expected, (
                    f"region {r} sample {si} ploid {p} (row {bi}): expected "
                    f"hap {si * P + p}'s CSR slice {expected} from the "
                    f"engine, got {actual} from the mixed path"
                )


@pytest.mark.parametrize("backend", BACKENDS)
def test_current_window_realign_inputs_reads_the_producers_window(
    streaming_record_tracks_fixture, backend
):
    """Issue #400: the window the producer has ALREADY filled must be readable from
    the drive's own engine -- that is what removes the second decode the track side
    used to do. Checked against the independent `window_realign_inputs` decode of the
    same window, which runs in the caller's thread (`debug_fill`) and is now
    test-only (its `_mixed_engine()` production caller was deleted).

    Also pins the identity guard: a request that does not describe the engine's
    current window must raise rather than silently pair one window's tracks with
    another's variants.
    """
    f = streaming_record_tracks_fixture(backend)
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")
    b = sds._backend
    n_s = sds.n_samples
    r_idx = np.arange(len(sds._regions), dtype=np.intp)
    contig_idx = int(b._regions[r_idx[0], 0])
    t_starts = np.ascontiguousarray(b._regions[r_idx, 1], np.uint32)
    t_ends = np.ascontiguousarray(b._regions[r_idx, 2], np.uint32)

    engine = b.build_engine([(contig_idx, t_starts, t_ends, 0, n_s)], 4, -1)

    # A window that is NOT the engine's current one must fail loudly.
    with pytest.raises(ValueError, match="does not match"):
        engine.current_window_realign_inputs(
            contig_idx, t_starts.tolist(), (t_ends + 1).tolist(), 0, n_s
        )

    got = engine.current_window_realign_inputs(
        contig_idx, t_starts.tolist(), t_ends.tolist(), 0, n_s
    )
    assert got is not None, "the first window must be available after the peek"

    # Independent source of truth: a plan-less engine decodes this exact window in the
    # caller's thread, with no producer and no slot involved.
    oracle = b.build_engine([], 1, 1)
    expected = oracle.window_realign_inputs(
        contig_idx, t_starts.tolist(), t_ends.tolist(), 0, n_s
    )
    for name, a, e in zip(
        ("v_starts", "ilens", "geno_v_idxs", "geno_offsets"), got, expected
    ):
        e = np.ascontiguousarray(e)
        assert a.dtype == e.dtype, f"{name} dtype drifted"
        np.testing.assert_array_equal(
            a, e, err_msg=f"{name} drifted from the window's own decode"
        )


# --- Issue #375 Track B, Task 8: VCF/PGEN mixed parity, edge cases, docs ---


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_record_mixed_parity_with_indels(
    streaming_record_tracks_fixture, backend, batch_size
):
    """Both halves of every mixed cell against `Dataset[r, s]`.

    `batch_size=1` is included because it is the ONE case where the written
    path's own track-major layout bug (#371) is invisible, so it is also
    where a streaming reorder bug would hide.
    """
    f = streaming_record_tracks_fixture(backend)
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=batch_size, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 2, f"track axis {tracks.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"{backend} cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_cell_equal(tracks[i], exp_tracks, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_mixed_parity_under_forced_windowing(
    streaming_record_tracks_fixture, backend
):
    """Same parity as `test_record_mixed_parity_with_indels`, forced to
    actually plan multiple windows over the fixture's 3x3 region/sample grid.

    `max_mem="4k"` is tuned (task-8-brief.md), not guessed: at the default
    `TRACK_BYTES_PER_CELL` sizing, `max_mem="1m"` still fits the WHOLE 3x3
    grid in one window (`window_samples=3, window_regions=64`), which would
    make this test a silent duplicate of `test_record_mixed_parity_with_indels`
    exercising nothing about multi-window planning. The assertion just below
    -- not just the `"4k"` constant -- is what actually gates that: it fails
    loudly if `TRACK_BYTES_PER_CELL`'s placeholder estimate ever changes
    enough to make `"4k"` stop forcing multiple windows.
    """
    f = streaming_record_tracks_fixture(backend)
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
        max_mem="4k",
    ).with_seqs("haplotypes")

    assert sds._window_samples < len(f.samples) or sds._window_regions < f.bed.height, (
        "max_mem did not actually force a multi-window plan; this test is vacuous"
    )

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=2, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 2, f"track axis {tracks.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"{backend} cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_cell_equal(tracks[i], exp_tracks, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_mixed_fixed_length_parity(streaming_record_tracks_fixture, backend):
    """`with_len(10)` mixed-path parity, both halves.

    Unlike SVAR2 (which has no fused kernel for a fixed output length), the
    record backends' fused kernel honors an explicit output length directly:
    the drive takes `out_lengths` from the batch's own haplotype offsets,
    which already bake in `L`. So this must MATCH the written oracle rather
    than raise `NotImplementedError`.
    """
    f = streaming_record_tracks_fixture(backend)
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs("haplotypes")
        .with_len(10)
    )
    sds = (
        gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.variants_path,
            tracks=[f.table, f.bigwigs],
        )
        .with_seqs("haplotypes")
        .with_len(10)
    )

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 2, f"track axis {tracks.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"{backend} cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_cell_equal(tracks[i], exp_tracks, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_realign_false_matches_written(streaming_record_tracks_fixture, backend):
    """`realign_tracks=False` keeps tracks in reference coordinates.

    Dispatches through the backend-independent `_tracks_from_intervals` path
    instead of the fused realign kernel -- the track axis loses its ploidy
    dimension and no CSR is read at all -- but the HAPLOTYPE half is
    untouched by `realign_tracks` and must still match the written oracle.
    """
    f = streaming_record_tracks_fixture(backend)
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = (
        gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.variants_path,
            tracks=[f.table, f.bigwigs],
        )
        .with_seqs("haplotypes")
        .with_settings(realign_tracks=False)
    )

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 2, f"track axis {tracks.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, _exp_tracks = written[r, s]
            ctx = f"{backend} cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("max_mem", [None, "4k"])
def test_record_mixed_multi_contig_parity(
    vcf_multi_contig, pgen_multi_contig, tmp_path_factory, backend, max_mem
):
    """Regression coverage for the record path's contig resolution (final review, M3).

    Every other mixed fixture in this file is a single 250bp `chr1`, so
    `contig_idx = int(backend._regions[r_idx[0], 0])`
    (`_streaming.py::_record_mixed_realign_window`) is always 0 there -- the
    new `backend._regions` table, its agreement with
    `StreamingDataset._regions`, and the multi-contig `ValueError` guard were
    otherwise entirely untested (contig resolution is the one input to
    `mixed_realign_window` genuinely new to Track B). This drives 2 contigs x
    2 regions x 3 samples through both halves of the mixed cell, at the
    default window plan and forced to multiple windows (`max_mem="4k"`).
    """
    f = vcf_multi_contig if backend == "vcf" else pgen_multi_contig
    variants_path = str(f.vcf) if backend == "vcf" else str(f.pgen)
    bed = f.regions
    assert bed["chrom"].n_unique() == 2, "fixture regressed to single-contig coverage"

    tmp_dir = tmp_path_factory.mktemp(f"streaming_{backend}_mc_tracks")

    # Write once WITHOUT tracks purely to learn the dataset's public sample
    # order -- same two-step pattern `streaming_record_tracks_fixture` uses,
    # for the same reason (the track files must be keyed by those names).
    probe = tmp_dir / "probe.gvl"
    gvl.write(probe, bed, variants=variants_path, overwrite=True)
    samples = list(gvl.Dataset.open(probe).samples)

    fai = pl.read_csv(
        str(f.fasta) + ".fai",
        separator="\t",
        has_header=False,
        new_columns=["chrom", "length", "offset", "linebases", "linewidth"],
    )
    contig_sizes = [
        (r["chrom"], int(r["length"]))
        for r in fai.iter_rows(named=True)
        if r["chrom"] in set(bed["chrom"].to_list())
    ]
    assert len(contig_sizes) == 2, "fixture regressed to single-contig coverage"

    BIN = 10
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(samples):
        p = tmp_dir / f"{sample}.alpha.bw"
        with pyBigWig.open(str(p), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            chroms, starts, ends, values = [], [], [], []
            for contig, size in contig_sizes:
                for b, lo in enumerate(range(0, size, BIN)):
                    hi = min(lo + BIN, size)
                    chroms.append(contig)
                    starts.append(lo)
                    ends.append(hi)
                    # Distinct per (sample, contig, bin), and the
                    # contig-dependent term now actually fires with 2
                    # contigs -- unlike `streaming_record_tracks_fixture`'s
                    # single-contig bed, where it was dead code.
                    values.append(
                        float(10 * (i + 1) + b)
                        + (0.5 if contig != contig_sizes[0][0] else 0.0)
                    )
            bw.addEntries(chroms, starts, ends=ends, values=values)
        bw_paths[sample] = str(p)
    alpha = gvl.BigWigs("alpha", bw_paths)

    rows = []
    for i, sample in enumerate(samples):
        for c_idx, (contig, size) in enumerate(contig_sizes):
            for b, lo in enumerate(range(0, size, BIN)):
                rows.append(
                    {
                        "sample_id": sample,
                        "chrom": contig,
                        "start": lo,
                        "end": min(lo + BIN, size),
                        "value": float(1000 * c_idx + 100 * (i + 1) + b),
                    }
                )
    zeta = gvl.Table("zeta", pl.DataFrame(rows))

    out = tmp_dir / f"{backend}_mc_tracks.gvl"
    gvl.write(
        path=out,
        bed=bed,
        variants=variants_path,
        tracks=[zeta, alpha],
        extend_to_length=False,
        max_jitter=None,
        overwrite=True,
    )

    written = gvl.Dataset.open(out, reference=f.fasta).with_seqs("haplotypes")
    extra_kwargs = {} if max_mem is None else {"max_mem": max_mem}
    sds = gvl.StreamingDataset(
        bed,
        reference=f.fasta,
        variants=variants_path,
        tracks=[zeta, alpha],
        **extra_kwargs,
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=2, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 2, f"track axis {tracks.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"{backend} multi-contig cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_cell_equal(tracks[i], exp_tracks, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }
