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


def test_svar2_mixed_parity_with_indels(streaming_svar2_tracks_fixture):
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
