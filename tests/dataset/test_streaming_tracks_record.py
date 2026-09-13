"""Rust seam for VCF/PGEN mixed variants+tracks (issue #375, Track B).

`window_realign_inputs` is the pymethod the Python mixed path uses to size a
window's deletion-extended track query before pulling the window's first
batch. Task 6 consumes it; this file pins its contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import genvarloader as gvl
from genvarloader.genvarloader import RecordStreamEngine


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


@pytest.mark.parametrize("backend", ["vcf", "pgen"])
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
    assert set(f.bigwigs.samples) == set(f.samples)

    assert f.bigwigs.name == "alpha"
    assert f.table.name == "zeta"
    assert Path(f.variants_path).exists()
    assert ds.n_regions == f.bed.height
