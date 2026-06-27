"""Dataset-level parity backstop for the fused tracks __getitem__ kernel (Task 14).

Proves that the fused Rust entry ``intervals_and_realign_track_fused``
produces byte-identical track output to the frozen golden (generated from
the rust implementation, oracle-verified against the composed numba pipeline).

The test asserts:
  1. The fused entry is actually invoked on the Rust path (non-vacuity spy guard).
  2. The Rust output is byte-identical to the frozen golden,
     across all 5 insertion-fill strategies.
  3. The output is non-trivial (contains non-zero values).

Scope:
  - Only the HapsTracks path is tested (track realignment requires variants).
  - Uses the ``max_jitter=0`` ``build_haps_tracks_dataset`` fixture (Task 11).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.parity import _golden

pytestmark = pytest.mark.parity


def test_fused_tracks_dataset_parity(synthetic_case, tmp_path, monkeypatch):
    """Fused intervals_and_realign_track_fused output matches the frozen golden.

    Covers all 5 insertion-fill strategies. The fused per-track entry (called
    directly from HapsTracks.__call__ on the rust path) must produce the same
    float32 bytes as the frozen golden.

    Spy guard: we monkeypatch ``_reconstruct_mod.intervals_and_realign_track_fused``
    to count calls. The spy must fire at least once during the read.
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

    ds_dir = build_haps_tracks_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds_base = gvl.Dataset.open(ds_dir, reference=ref)
    ds_base = ds_base.with_seqs("haplotypes").with_tracks("signal")

    orig_fused = getattr(_reconstruct_mod, "intervals_and_realign_track_fused", None)
    assert orig_fused is not None, (
        "intervals_and_realign_track_fused not found on _reconstruct_mod — "
        "ensure it is imported at module level in _reconstruct.py"
    )

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

        # --- read (default rust backend, spy active) ---
        out = ds[:, :]

        # Anti-vacuous guard
        assert calls["n"] > 0, (
            f"[{strategy_name}] intervals_and_realign_track_fused was NEVER invoked "
            f"during the read (calls={calls['n']}) — the backstop is "
            "vacuous. Ensure HapsTracks.__call__ calls intervals_and_realign_track_fused "
            "on the Rust path."
        )

        # --- extract track arrays for non-triviality check ---
        _, tracks_out = out
        data_r = np.asarray(tracks_out.data, dtype=np.float32)

        # Non-triviality
        assert data_r.size > 0, (
            f"[{strategy_name}] Track output is empty — "
            "regions may not overlap stored intervals."
        )
        assert np.any(data_r != 0.0), (
            f"[{strategy_name}] All realigned track values are 0 — "
            "the BigWig intervals may not overlap the stored regions, "
            "making this comparison vacuous."
        )

        # --- replay against frozen golden ---
        golden_name = f"ds_haps_tracks_{strategy_name}"
        _golden.assert_output_matches_golden(out, _golden.load_flat_golden(golden_name))

        # Restore original between strategies.
        monkeypatch.setattr(
            _reconstruct_mod, "intervals_and_realign_track_fused", orig_fused
        )
