"""SVAR2 mixed variants + tracks parity (issue #375, Track A).

The written oracle is `HapsTracks._call_svar2` (`_reconstruct.py:309`), which
splits the interval->realign step into `intervals_to_tracks` +
`shift_and_realign_tracks_from_svar2_readbound` because SVAR2 has no fused
kernel. Track A (this task) only lands the split-kernel state/seam
(`_Svar2Realign`, `_Svar2Backend.mixed_realign_window`) -- it is not wired
into the read drive yet (see the bridge guard in `_iter_batches`), so there
is no `Dataset[r, s]`-parity case to run here today. This file currently
holds only the capability assertion below; Task 4 (issue #380) adds the
cell-by-cell parity cases for BOTH halves of the `(haps, tracks)` pair once
the drive wiring lands.
"""

from __future__ import annotations


def test_svar2_backend_declares_mixed_support():
    from genvarloader._dataset._streaming import _Svar2Backend

    assert _Svar2Backend.supports_mixed_tracks is True
    assert hasattr(_Svar2Backend, "mixed_realign_window")
