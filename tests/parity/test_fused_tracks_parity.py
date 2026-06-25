"""Dataset-level parity backstop for the fused tracks __getitem__ kernel (Task 14).

Proves that the fused Rust entry ``intervals_and_realign_track_fused``
produces byte-identical track output to the composed numba pipeline
(intervals_to_tracks → shift_and_realign_tracks_sparse), which is the oracle.

The test asserts:
  1. The fused entry is actually invoked on the Rust path (non-vacuity spy guard).
  2. The fused Rust output is byte-identical to the composed numba output,
     across all 5 insertion-fill strategies.
  3. The output is non-trivial (contains non-zero values).

Scope:
  - Only the HapsTracks path is tested (track realignment requires variants).
  - Uses the ``max_jitter=0`` ``build_haps_tracks_dataset`` fixture (Task 11),
    which satisfies the ``intervals_to_tracks`` Rust contract
    (``itv_start >= query_start``).

Spy mechanism:
  - The fused entry is called directly (not via _dispatch) from
    ``HapsTracks.__call__`` in ``_reconstruct.py`` on the Rust path.
  - We monkeypatch ``_reconstruct_mod.intervals_and_realign_track_fused``
    to count calls. The spy must fire at least once during the rust read
    and must NOT fire during the numba read.
  - The numba read uses ``GVL_BACKEND=numba``, which forces the composed path
    (intervals_to_tracks numba → shift_and_realign_tracks_sparse numba).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.parity


def test_fused_tracks_dataset_parity(synthetic_case, tmp_path, monkeypatch):
    """Fused intervals_and_realign_track_fused is byte-identical to composed numba oracle.

    Covers all 5 insertion-fill strategies. The fused per-track entry (called
    directly from HapsTracks.__call__ on the non-numba path) must produce the
    same float32 bytes as the composed numba pipeline for every (region, sample,
    hap, track) combination.

    Spy guard: we monkeypatch ``_reconstruct_mod.intervals_and_realign_track_fused``
    to count calls. The spy must fire at least once during the rust read and
    must NOT fire during the numba read.
    """
    import genvarloader as gvl
    import genvarloader._dataset._reconstruct as _reconstruct_mod
    from genvarloader._dataset._insertion_fill import (
        Constant,
        FlankSample,
        Interpolate,
        Repeat5p,
        Repeat5pNormalized,
    )
    from tests.parity._fixtures import build_haps_tracks_dataset

    # --- build fixture: fresh variants+tracks dataset with max_jitter=0 ---
    ds_dir = build_haps_tracks_dataset(tmp_path, synthetic_case.svar_path)

    # Open with the session reference so haplotype reconstruction runs.
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds_base = gvl.Dataset.open(ds_dir, reference=ref)
    ds_base = ds_base.with_seqs("haplotypes").with_tracks("signal")

    # --- verify the fused entry is importable ---
    orig_fused = getattr(_reconstruct_mod, "intervals_and_realign_track_fused", None)
    assert orig_fused is not None, (
        "intervals_and_realign_track_fused not found on _reconstruct_mod — "
        "ensure it is imported at module level in _reconstruct.py"
    )

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

        # --- install spy on intervals_and_realign_track_fused ---
        calls: dict[str, int] = {"n": 0}

        def _make_spy(orig, c=calls):
            def spy(*a, **k):
                c["n"] += 1
                return orig(*a, **k)

            return spy

        spy_fn = _make_spy(orig_fused)
        monkeypatch.setattr(
            _reconstruct_mod, "intervals_and_realign_track_fused", spy_fn
        )

        calls["n"] = 0  # reset per-strategy

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
            f"the numba read (count went from {rust_call_count} to {calls['n']}) — "
            "the fused entry is being called on the numba path, which is a bug."
        )

        # Anti-vacuous guard: fused entry must have been invoked.
        assert rust_call_count > 0, (
            f"[{strategy_name}] intervals_and_realign_track_fused was NEVER invoked "
            f"during the rust read (calls={rust_call_count}) — the backstop is "
            "vacuous. Ensure HapsTracks.__call__ calls intervals_and_realign_track_fused "
            "on the Rust path."
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

        # Non-triviality: at least some non-zero track values.
        assert data_r.size > 0, (
            f"[{strategy_name}] Track output is empty — "
            "regions may not overlap stored intervals."
        )
        assert np.any(data_r != 0.0), (
            f"[{strategy_name}] All realigned track values are 0 — "
            "the BigWig intervals may not overlap the stored regions, "
            "making this comparison vacuous."
        )

        # Restore original (monkeypatch.setattr is undone at end of each iteration
        # via undo stack, but we re-patch each loop so explicitly restore too).
        monkeypatch.setattr(
            _reconstruct_mod, "intervals_and_realign_track_fused", orig_fused
        )
