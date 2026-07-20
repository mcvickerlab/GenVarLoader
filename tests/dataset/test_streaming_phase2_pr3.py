"""PR 3 Task 1: GIL-free `svar2_fill_super_batch` core (find_ranges -> gather ->
reconstruct) -- byte-identical to the existing "sync" fill+drain path
(`_Svar2Backend._fill_super_batch` + `_drain`), per plan window."""

from __future__ import annotations

import numpy as np

import genvarloader as gvl
from genvarloader._dataset._streaming import _Svar2Backend


def _sync_reference(backend: _Svar2Backend, r_idx, s_idx, window, lo, hi):
    """Existing sync path: gather rows [lo,hi) + reconstruct into a fresh buffer, drain."""
    from genvarloader.genvarloader import Svar2ReconBuf

    buf = Svar2ReconBuf(backend.ploidy)
    backend._fill_super_batch(r_idx, s_idx, window, lo, hi, buf, parallel=False)
    return backend._drain(buf, 0, hi - lo)


def test_fill_super_batch_matches_sync_path(svar2_multicontig_fixture):
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert isinstance(backend, _Svar2Backend)
    from genvarloader.genvarloader import svar2_fill_super_batch

    for r_idx, s_idx in sds._plan():
        window = backend.read_window(r_idx, s_idx)
        n_rows = len(r_idx) * len(s_idx)
        # one-call Rust chain over the whole window
        contig_idx, contig = backend._contig_of(r_idx)
        rb = backend._regions[r_idx, 1:3]
        starts = np.ascontiguousarray(rb[:, 0], np.uint32)
        ends = np.ascontiguousarray(rb[:, 1], np.uint32)
        phys = np.ascontiguousarray(backend._phys_sample_idx[s_idx], np.int64)
        ref_, ref_offsets = backend._ref._contig_slice(contig_idx)
        data, offsets = svar2_fill_super_batch(
            backend._store,
            contig,
            starts,
            ends,
            phys,
            np.ascontiguousarray(rb, np.int32),
            ref_,
            ref_offsets,
            np.uint8(backend._ref.pad_char),
            0,
            n_rows,
            False,
        )
        # sync reference over the same rows
        ref_rag = _sync_reference(backend, r_idx, s_idx, window, 0, n_rows)
        np.testing.assert_array_equal(np.asarray(data).view("S1"), ref_rag.data)
        np.testing.assert_array_equal(np.asarray(offsets, np.int64), ref_rag.offsets)
