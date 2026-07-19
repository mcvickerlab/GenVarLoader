"""PR 2: Svar2ReconBuf super-batch fill + drain — byte-identical to the per-batch
read-bound FFI, and self-consistent across drain boundaries."""

from __future__ import annotations

import numpy as np


def _plan_windows(sds):
    """(r_idx, s_idx) windows exactly as the sync drive sees them."""
    return list(sds._plan())


def test_super_batch_fill_drain_matches_per_batch_ffi(
    svar2_multicontig_fixture,
) -> None:
    """Reconstructing a whole super-batch then draining batch_size slices is
    byte-identical to reconstructing each batch_size slice on its own via the
    Phase-1/PR-1 per-batch FFI (parallel=False)."""
    import genvarloader as gvl
    from genvarloader.genvarloader import Svar2ReconBuf, svar2_reconstruct_super_batch

    assert svar2_reconstruct_super_batch is not None

    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert backend is not None
    P = backend.ploidy

    for r_idx, s_idx in _plan_windows(sds):
        window = backend.read_window(r_idx, s_idx)
        n_rows = len(r_idx) * len(s_idx)

        # (a) super-batch: one fill over ALL window rows, drain per batch_size.
        buf = Svar2ReconBuf(P)
        backend._fill_super_batch(r_idx, s_idx, window, 0, n_rows, buf, parallel=False)
        assert buf.n_rows == n_rows
        drained = []
        for lo in range(0, n_rows, 4):
            hi = min(lo + 4, n_rows)
            rag = backend._drain(buf, lo, hi)
            drained.append(rag.data.view("S1").copy())
        got = np.concatenate(drained) if drained else np.empty(0, "S1")

        # (b) reference: per-batch FFI, parallel=False (the PR-1 path).
        ref_parts = []
        for lo in range(0, n_rows, 4):
            hi = min(lo + 4, n_rows)
            rag = backend._reconstruct_batch_reference(r_idx, s_idx, window, lo, hi)
            ref_parts.append(np.asarray(rag.data).view("S1").copy())
        ref = np.concatenate(ref_parts) if ref_parts else np.empty(0, "S1")

        np.testing.assert_array_equal(got, ref)
