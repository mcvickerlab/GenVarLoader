"""Dataset read-path parity backstops for track kernels.

Covers two cases:

1. ``intervals_to_tracks`` only (track-only dataset, no variants):
   Proves that flipping GVL_BACKEND produces byte-identical tracks through
   the real Dataset.__getitem__ path.

2. ``shift_and_realign_tracks_sparse`` (haplotypes+tracks dataset with indels):
   Proves that the dispatch wiring for the realignment kernel is correct
   end-to-end, across every insertion-fill strategy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.parity._fixtures import build_haps_tracks_dataset, build_track_dataset

pytestmark = pytest.mark.parity


def _read_track_array(
    ds, r_idx: np.ndarray, s_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (data, offsets) from the RaggedTracks produced by ds[r_idx, s_idx].

    Dataset.open with no reference and no variants + with_tracks("signal") returns
    a RaggedTracks directly from __getitem__.  RaggedTracks is a Ragged[np.float32]
    so it carries .data (flat float32 buffer) and .offsets (int64).
    """
    result = ds[r_idx, s_idx]
    # result is RaggedTracks (a seqpro Ragged[np.float32]) when no seqs are configured
    data = np.asarray(result.data, dtype=np.float32)
    offsets = np.asarray(result.offsets, dtype=np.int64)
    return data, offsets


def test_track_getitem_identical_across_backends(tmp_path, monkeypatch):
    ds_dir = build_track_dataset(tmp_path)

    import genvarloader as gvl
    import genvarloader._dataset._reconstruct as _recon_mod
    import genvarloader._dataset._tracks as _tracks_mod

    ds = gvl.Dataset.open(ds_dir)
    # tracks-only dataset: with_tracks enables the signal track explicitly
    ds = ds.with_tracks("signal")

    # Use slice(None) for both dims so Dataset uses "basic" indexing (cross-product)
    # which returns shape (n_regions, n_samples, n_tracks, ~length).
    r_idx = slice(None)
    s_idx = slice(None)

    # --- spy: assert intervals_to_tracks is actually called on the live read path ---
    calls: dict[str, int] = {"n": 0}

    def _make_spy(orig):
        def spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        return spy

    # Patch BOTH call-site modules; the track-only path uses _tracks_mod
    monkeypatch.setattr(
        _tracks_mod, "intervals_to_tracks", _make_spy(_tracks_mod.intervals_to_tracks)
    )
    monkeypatch.setattr(
        _recon_mod, "intervals_to_tracks", _make_spy(_recon_mod.intervals_to_tracks)
    )

    # --- numba read ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    data_n, off_n = _read_track_array(ds, r_idx, s_idx)

    # Backstop guard: kernel must have been called at least once
    assert calls["n"] > 0, (
        f"intervals_to_tracks was NEVER called during the numba read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the read path and confirm the track reconstructor is active."
    )

    # --- rust read ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    data_r, off_r = _read_track_array(ds, r_idx, s_idx)

    # --- byte-identical comparison ---
    np.testing.assert_array_equal(
        off_n, off_r, err_msg="offsets differ across backends"
    )
    assert data_n.dtype == data_r.dtype == np.float32, (
        f"dtype mismatch: numba={data_n.dtype}, rust={data_r.dtype}"
    )
    np.testing.assert_array_equal(
        data_n, data_r, err_msg="track data differs across backends"
    )

    # Sanity: the read painted real non-zero signal (not an all-zero vacuous match)
    assert np.any(data_n != 0.0), (
        "Track data is all-zero — regions may not overlap synthetic intervals. "
        "Non-zero signal is required to prove the comparison is meaningful."
    )


# ---------------------------------------------------------------------------
# Haplotypes+tracks realignment backstop
# ---------------------------------------------------------------------------


def test_tracks_realign_getitem_identical_across_backends(
    synthetic_case, tmp_path, monkeypatch
):
    """Spy-guarded backstop for tracks realignment dispatch wiring (Task 11/14).

    Proves that materialising a haplotypes+tracks dataset (with indel-bearing
    genotypes) via ``ds[:, :]`` produces byte-identical track output across
    GVL_BACKEND=rust and GVL_BACKEND=numba, for every insertion-fill strategy.

    After Task 14, the Rust path calls the fused entry
    ``intervals_and_realign_track_fused`` (one FFI crossing per track) instead
    of the composed ``shift_and_realign_tracks_sparse`` dispatch.  The spy
    targets ``intervals_and_realign_track_fused`` on the Rust path.

    The numba path continues to use the composed path (intervals_to_tracks
    → shift_and_realign_tracks_sparse via dispatch); the parity check
    (byte-identical output) remains the gate.

    Fixture geometry:
    - A fresh GVL dataset is built in tmp_path via gvl.write with both the
      session SparseVar variants (which contain indels on chr1/chr2) and a
      synthetic BigWig ``signal`` track for samples s0/s1/s2.
    - max_jitter=0 is used to avoid the pre-existing intervals_to_tracks
      landmine: with max_jitter>0, gvl.write clips BigWig intervals to the
      jitter-expanded region boundaries (chromStart - max_jitter), but
      Dataset.open derives _full_regions from the original chromStart.  The
      gap of max_jitter bp causes stored interval starts to precede the
      query start, violating the Rust kernel contract and triggering a
      PanicException.  With max_jitter=0 the boundaries match exactly.

    Fill strategies covered: all 5 (Repeat5p, Repeat5pNormalized, Constant,
    FlankSample, Interpolate).  Each is set via with_insertion_fill and the
    byte-identical comparison is re-run.
    """
    import genvarloader as gvl
    import genvarloader._dataset._reconstruct as _recon_mod
    from genvarloader._dataset._insertion_fill import (
        Constant,
        FlankSample,
        Interpolate,
        Repeat5p,
        Repeat5pNormalized,
    )

    # --- build fixture: fresh variants+tracks dataset with max_jitter=0 ---
    ds_dir = build_haps_tracks_dataset(tmp_path, synthetic_case.svar_path)

    # Open with the session reference so haplotype reconstruction runs.
    # Use synthetic_case.ref_path to get the same reference used to build
    # the variants, not the pre-committed tests/data/fasta reference.
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds_base = gvl.Dataset.open(ds_dir, reference=ref)
    ds_base = ds_base.with_seqs("haplotypes").with_tracks("signal")

    # --- install spy on the fused Rust entry ---
    # After Task 14 the Rust path calls intervals_and_realign_track_fused
    # directly (not via _dispatch), so we monkeypatch _recon_mod.
    orig_fused = getattr(_recon_mod, "intervals_and_realign_track_fused", None)
    assert orig_fused is not None, (
        "intervals_and_realign_track_fused not found on _recon_mod — "
        "ensure it is imported at module level in _reconstruct.py"
    )

    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    # All 5 insertion-fill strategies to cover.
    fill_strategies = [
        Repeat5p(),
        Repeat5pNormalized(),
        Constant(0.0),
        FlankSample(flank_width=5),
        Interpolate(order=1),
    ]

    for strategy in fill_strategies:
        strategy_name = type(strategy).__name__
        ds = ds_base.with_insertion_fill(strategy)

        monkeypatch.setattr(_recon_mod, "intervals_and_realign_track_fused", _spy_fused)
        calls["n"] = 0  # reset per-strategy counter

        # --- rust read (fused path, spy active) ---
        monkeypatch.setenv("GVL_BACKEND", "rust")
        out_rust = ds[:, :]

        rust_call_count = calls["n"]

        # --- numba read (composed path — spy must NOT fire) ---
        monkeypatch.setenv("GVL_BACKEND", "numba")
        out_numba = ds[:, :]

        # Wiring guard: numba must NOT fire the fused spy.
        assert calls["n"] == rust_call_count, (
            f"[{strategy_name}] intervals_and_realign_track_fused spy fired during "
            f"the numba read (count went from {rust_call_count} to {calls['n']}) "
            "— spy is wired to the numba path, which is a bug."
        )

        # Anti-vacuous guard: fused entry must have been invoked.
        assert rust_call_count > 0, (
            f"[{strategy_name}] intervals_and_realign_track_fused was NEVER "
            f"invoked during the rust read (calls={rust_call_count}) — "
            "the backstop is vacuous. Inspect HapsTracks.__call__ to "
            "confirm intervals_and_realign_track_fused is called on the Rust path."
        )

        # --- extract track arrays from the (haps, tracks) tuple ---
        # out_rust and out_numba are (RaggedSeqs, RaggedTracks) tuples.
        _, tracks_rust = out_rust
        _, tracks_numba = out_numba
        data_r = np.asarray(tracks_rust.data, dtype=np.float32)
        off_r = np.asarray(tracks_rust.offsets, dtype=np.int64)
        data_n = np.asarray(tracks_numba.data, dtype=np.float32)
        off_n = np.asarray(tracks_numba.offsets, dtype=np.int64)

        # --- byte-identical comparison ---
        np.testing.assert_array_equal(
            off_n,
            off_r,
            err_msg=f"[{strategy_name}] track offsets differ across backends",
        )
        assert data_n.dtype == data_r.dtype == np.float32, (
            f"[{strategy_name}] dtype mismatch: numba={data_n.dtype}, "
            f"rust={data_r.dtype}"
        )
        np.testing.assert_array_equal(
            data_n,
            data_r,
            err_msg=f"[{strategy_name}] track data differs across backends",
        )

        # Non-triviality: at least some non-zero track values (not all-zero
        # vacuous match).  Signal values are drawn from N(0,1) so near-zero
        # is extremely unlikely but possible; we check the overall tensor.
        assert data_r.size > 0, (
            f"[{strategy_name}] Track output is empty — "
            "regions may not overlap stored intervals."
        )
        # At least one realigned haplotype must differ from the input track
        # values OR be non-zero — any non-zero value proves the track was
        # painted from the BigWig intervals.
        assert np.any(data_r != 0.0), (
            f"[{strategy_name}] All realigned track values are 0 — "
            "the BigWig intervals may not overlap the stored regions, "
            "making this comparison vacuous."
        )

        # Restore original between strategies.
        monkeypatch.setattr(_recon_mod, "intervals_and_realign_track_fused", orig_fused)
