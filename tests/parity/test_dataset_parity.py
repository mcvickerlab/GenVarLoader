"""Dataset read-path parity backstop for intervals_to_tracks.

Proves that flipping GVL_BACKEND (numba vs rust) produces byte-identical
track output through the real Dataset.__getitem__ path.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.parity._fixtures import build_track_dataset

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
    np.testing.assert_array_equal(off_n, off_r, err_msg="offsets differ across backends")
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
