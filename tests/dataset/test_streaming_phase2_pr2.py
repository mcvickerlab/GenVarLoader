"""PR 2: Svar2ReconBuf super-batch fill + drain — byte-identical to the per-batch
read-bound FFI, and self-consistent across drain boundaries."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from seqpro.rag import Ragged


def _plan_windows(sds):
    """(r_idx, s_idx) windows exactly as the sync drive sees them."""
    return list(sds._plan())


def _per_batch_reference(
    backend,
    r_idx: NDArray[np.intp],
    s_idx: NDArray[np.intp],
    window: dict[str, object],
    lo: int,
    hi: int,
) -> Ragged:
    """Parity reference (Phase-2 PR 2): the *old* `_Svar2Backend.generate_batch`/
    `_reconstruct_batch_reference` body, moved here once production switched over to
    the super-batch `_fill_super_batch`/`_drain` path (Task 3) -- so production has
    ONE obvious reconstruct path, and this stays as the independent oracle the
    super-batch fill+drain path is checked byte-identical against.
    """
    from genvarloader.genvarloader import reconstruct_haplotypes_from_svar2_readbound

    P = backend.ploidy
    (
        region_starts,
        orig_samples,
        vk_snp,
        vk_indel,
        dense_snp,
        dense_indel,
        region_bounds,
        shifts,
        ref_,
        ref_offsets,
    ) = backend._gather_rows(r_idx, s_idx, window, lo, hi)
    contig_idx = cast(int, window["contig_idx"])
    contig = backend._contigs[contig_idx]
    m = hi - lo

    data, offsets = reconstruct_haplotypes_from_svar2_readbound(
        backend._store,
        contig,
        region_starts,
        orig_samples,
        vk_snp,
        vk_indel,
        dense_snp,
        dense_indel,
        region_bounds,
        shifts,
        ref_,
        ref_offsets,
        np.uint8(backend._ref.pad_char),
        np.int64(-1),  # ragged output (no fixed output_length)
        False,  # parallel: per-batch reconstruct is tiny (~batch_size*ploidy
        #        haplotypes); the 96-thread rayon fork/join costs more than it saves
        #        here (measured 1.2-1.8x faster serial).
        False,  # filter_exonic (splicing out of scope)
    )
    return Ragged.from_offsets(
        np.asarray(data).view("S1"), (m, P, None), np.asarray(offsets, np.int64)
    )


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
            rag = _per_batch_reference(backend, r_idx, s_idx, window, lo, hi)
            ref_parts.append(np.asarray(rag.data).view("S1").copy())
        ref = np.concatenate(ref_parts) if ref_parts else np.empty(0, "S1")

        np.testing.assert_array_equal(got, ref)
