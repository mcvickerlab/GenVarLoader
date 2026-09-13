"""SVAR2 mixed variants + tracks parity (issue #375, Track A).

The written oracle is `HapsTracks._call_svar2` (`_reconstruct.py:309`), which
splits the interval->realign step into `intervals_to_tracks` +
`shift_and_realign_tracks_from_svar2_readbound` because SVAR2 has no fused
kernel. Track A wires `_Svar2Realign`/`_Svar2Backend.mixed_realign_window`
into the "sync" read drive, so `StreamingDataset(..., tracks=...)` over a
`.svar2` variant source now yields `(haplotypes, tracks)` matching
`Dataset[r, s]` byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl


def test_svar2_backend_declares_mixed_support():
    from genvarloader._dataset._streaming import _Svar2Backend

    assert _Svar2Backend.supports_mixed_tracks is True
    assert hasattr(_Svar2Backend, "mixed_realign_window")


def _assert_haps_cell_equal(streamed, expected, ploidy: int, ctx="") -> None:
    """Assert the HAPLOTYPE half of one mixed cell matches ``Dataset[r, s][0]``."""
    for h in range(ploidy):
        got = np.asarray(streamed[h])
        exp = np.asarray(expected[h])
        assert got.shape == exp.shape, (
            f"{ctx}hap {h}: shape {got.shape} != oracle {exp.shape}"
        )
        np.testing.assert_array_equal(
            got, exp, err_msg=f"{ctx}hap {h}: haplotype bytes differ"
        )


def _assert_tracks_cell_equal(streamed, expected, ctx="") -> None:
    """Assert the TRACK half of one mixed cell matches ``Dataset[r, s][1]``.

    Re-aligned track output is ragged per (track, hap) -- post-indel lengths
    differ across haplotypes even within one cell -- so a bare
    ``np.asarray()`` on both sides raises ``ValueError: cannot convert a
    jagged Ragged to a dense array`` rather than a useful diff. Compare
    through the Ragged's own ``shape``/``lengths``/packed values instead,
    mirroring ``test_streaming_tracks.py``'s ``_assert_cell_equal``.
    """
    if not hasattr(streamed, "lengths") or not hasattr(expected, "lengths"):
        # `with_len(L)` yields dense arrays on both sides.
        got_t, exp_t = np.asarray(streamed), np.asarray(expected)
        assert got_t.shape == exp_t.shape, (
            f"{ctx}track shape {got_t.shape} != oracle {exp_t.shape}"
        )
        np.testing.assert_allclose(
            got_t, exp_t, rtol=0, atol=0, err_msg=f"{ctx}track values differ"
        )
        return

    s_shape, e_shape = streamed.shape[:-1], expected.shape[:-1]
    assert s_shape == e_shape, f"{ctx}track shape {s_shape} != oracle {e_shape}"
    np.testing.assert_array_equal(
        np.asarray(streamed.lengths),
        np.asarray(expected.lengths),
        err_msg=f"{ctx}per-(track, hap) track lengths differ",
    )
    np.testing.assert_allclose(
        np.asarray(streamed.to_packed().data),
        np.asarray(expected.to_packed().data),
        rtol=0,
        atol=0,
        err_msg=f"{ctx}track values differ",
    )


@pytest.mark.parametrize("super_batch_rows", [4096, 5])
def test_svar2_mixed_parity_with_indels(
    streaming_svar2_tracks_fixture, super_batch_rows
):
    """Byte-identical parity for BOTH halves of the (haplotypes, tracks) pair.

    Parametrized over `super_batch_rows` (issue #375 Track A fix round 1,
    M1): the drive mixes two index spaces on adjacent lines --
    `_drain(buf, lo - sb_lo, hi - sb_lo)` is super-batch-LOCAL while
    `realign_batch(lo, hi, ...)`/`track_w.row_starts[lo:hi]`/
    `track_w.row_lengths[lo:hi]`/`flat_r[lo:hi]` are all window-GLOBAL. The
    default `SUPERBATCH_TARGET_ROWS` (4096) against this fixture's 18 rows
    per window means `sb_lo` is always 0, so `lo == lo - sb_lo` for every
    batch and a regression swapping either index would still pass -- the
    `5` case forces `sb_lo` to a nonzero value more than once per window (18
    rows / 5 = 4 super-batches), with `batch_size=4` straddling super-batch
    boundaries, so a swap actually mis-pairs tracks with rows and fails.
    Mirrors `test_streaming_parity_svar2.py`'s
    `object.__setattr__(sds._backend, "_super_batch_rows", 5)` seam.
    """
    f = streaming_svar2_tracks_fixture
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")
    object.__setattr__(sds._backend, "_super_batch_rows", super_batch_rows)

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 2, f"track axis {tracks.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_tracks_cell_equal(tracks[i], exp_tracks, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_svar2_engine_tracks_raises(streaming_svar2_tracks_fixture):
    """The test-only "svar2_engine" strategy has no track path (fix round 1, L3).

    Mirrors `test_streaming_parity_svar2.py`'s `_with_strategy` seam: force
    `_prefetch_strategy="svar2_engine"` on a clone and confirm `tracks=`
    still raises there, since only the default "sync" strategy is wired.
    """
    import copy

    f = streaming_svar2_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.table, f.bigwigs],
    )
    sds = copy.copy(sds)
    object.__setattr__(sds, "_prefetch_strategy", "svar2_engine")
    with pytest.raises(NotImplementedError, match="svar2_engine"):
        next(iter(sds.to_iter(batch_size=1)))


def test_svar2_mixed_with_len_rejected(streaming_svar2_tracks_fixture):
    """The written SVAR2 mixed path refuses fixed-length realigned tracks
    (`_reconstruct.py:_call_svar2`: the readbound kernel always sizes each hap
    to ref_len + diff, so an int output_length cannot be honored
    byte-identically). Streaming must refuse identically rather than silently
    mis-size."""
    f = streaming_svar2_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")
    with pytest.raises(NotImplementedError, match="fixed-length|with_len"):
        next(iter(sds.with_len(10).to_iter(batch_size=4)))


def test_svar2_realign_false_matches_written(streaming_svar2_tracks_fixture):
    """`realign_tracks=False` leaves tracks in reference coordinates -- the
    un-realigned path is backend-independent (`_tracks_from_intervals`), so it
    must work for SVAR2 too, and the track axis loses ploidy."""
    f = streaming_svar2_tracks_fixture
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs("haplotypes")
        .with_tracks(["alpha", "zeta"])
    )
    sds = (
        gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar2_path,
            tracks=[f.table, f.bigwigs],
        )
        .with_seqs("haplotypes")
        .with_settings(realign_tracks=False)
    )

    n = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        # (batch, n_tracks, ~length) -- no ploidy axis when un-realigned.
        assert tracks.shape[1] == 2
        for i in range(len(r_idx)):
            _assert_haps_cell_equal(
                haps[i], written[int(r_idx[i]), int(s_idx[i])][0], sds.ploidy
            )
            n += 1
    assert n == written.shape[0] * written.shape[1]


def test_svar2_realign_false_drops_ploidy_axis(streaming_svar2_tracks_fixture):
    """Mirrors SVAR1's `test_realign_false_drops_ploidy_axis`: with
    `realign_tracks=False`, the TRACK half must match the written oracle
    byte-for-byte, not just in shape. `test_svar2_realign_false_matches_written`
    above only checks the haplotype half and `tracks.shape[1]`; this pins the
    actual track values too, narrowed to a single track (`alpha`) so the
    written oracle can be indexed the same way SVAR1's analog does.
    """
    f = streaming_svar2_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=f.bigwigs,
    ).with_settings(realign_tracks=False)
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs("haplotypes")
        .with_tracks("alpha")
        .with_settings(realign_tracks=False)
    )
    (haps, tracks), r_idx, s_idx = next(
        iter(sds.to_iter(batch_size=1, return_indices=True))
    )
    r, s = int(r_idx[0]), int(s_idx[0])
    exp_haps, exp_tracks = written[r, s]
    ctx = f"cell (r={r}, s={s}): "
    # No ploidy axis on the track side when un-realigned.
    assert tracks[0].shape[:-1] == exp_tracks.shape[:-1]
    _assert_haps_cell_equal(haps[0], exp_haps, sds.ploidy, ctx=ctx)
    _assert_tracks_cell_equal(tracks[0], exp_tracks, ctx=ctx)


def test_svar2_mixed_single_track(streaming_svar2_tracks_fixture):
    """One track, not two: exercises the n_tracks == 1 reorder path (the
    track-major -> (b, t, p) permutation is a no-op there, so a bug in the
    permutation shows up only by contrast with the two-track case)."""
    f = streaming_svar2_tracks_fixture
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs("haplotypes")
        .with_tracks("alpha")
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.bigwigs],
    ).with_seqs("haplotypes")

    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 1
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_tracks_cell_equal(tracks[i], exp_tracks, ctx=ctx)


def test_svar2_mixed_batch_size_one(streaming_svar2_tracks_fixture):
    """batch_size=1 is the ONE case where the written path's own track-major
    layout bug (#371) is invisible, so it is also the case where a streaming
    reorder bug hides. Pin it explicitly."""
    f = streaming_svar2_tracks_fixture
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")

    for data, r_idx, s_idx in sds.to_iter(batch_size=1, return_indices=True):
        haps, tracks = data
        r, s = int(r_idx[0]), int(s_idx[0])
        exp_haps, exp_tracks = written[r, s]
        ctx = f"cell (r={r}, s={s}): "
        _assert_haps_cell_equal(haps[0], exp_haps, sds.ploidy, ctx=ctx)
        _assert_tracks_cell_equal(tracks[0], exp_tracks, ctx=ctx)
