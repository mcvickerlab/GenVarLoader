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


def _with_strategy(sds, strategy):
    import copy

    clone = copy.copy(sds)
    object.__setattr__(clone, "_prefetch_strategy", strategy)
    return clone


def _collect(sds, batch_size):
    """All (r, s) -> per-ploid haplotype rows, keyed by cell, as a dict for
    order-independent compare. Stored per-ploid (not `np.asarray(data[i])` as a
    whole row) because ploids are independently variable-length under indels --
    e.g. this fixture has rows where ploid 0 and ploid 1 differ in length -- so a
    row is a genuinely jagged `Ragged`, not a rectangular array."""
    ploidy = sds._backend.ploidy
    out = {}
    for data, r_idx, s_idx in sds.to_iter(batch_size=batch_size, return_indices=True):
        for i in range(len(r_idx)):
            out[(int(r_idx[i]), int(s_idx[i]))] = [
                np.asarray(data[i][p]).copy() for p in range(ploidy)
            ]
    return out


def test_svar2_engine_matches_sync_bytewise(svar2_multicontig_fixture):
    fx = svar2_multicontig_fixture
    base = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    sync = _collect(base, batch_size=4)  # default "sync"
    eng_sds = _with_strategy(base, "svar2_engine")
    eng = _collect(eng_sds, batch_size=4)
    assert set(sync) == set(eng)
    for cell in sync:
        for p in range(base._backend.ploidy):
            np.testing.assert_array_equal(sync[cell][p], eng[cell][p])


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


def test_svar2_engine_nested_super_batch_matches_written(svar2_multicontig_fixture):
    """Whole-branch review Critical: `"svar2_engine"`'s drive loop used to step
    `batch_size` CONTINUOUSLY over a window's full `n_rows`, but the Rust
    `Svar2StreamEngine` resets its batch boundaries at every super-batch (it
    splits each window into `ceil(n_rows / super_batch_rows)` jobs and drains
    each independently in `batch_size` chunks from row 0 -- see
    `src/ffi/svar2_stream_engine.rs`'s `slice_current`/`CurrentWindow`). When a
    window spans >1 super-batch AND `super_batch_rows` doesn't divide
    `batch_size`, each super-batch emits a short tail batch mid-window and the
    Python continuous loop desyncs against the engine's actual batch grain.

    No existing parity test caught this because `svar2_multicontig_fixture`'s
    windows all fit in ONE super-batch at the real default
    (`SUPERBATCH_TARGET_ROWS=4096`). Force a tiny `_super_batch_rows=5` (5 does
    not divide `batch_size=3`, and the fixture's 6-region-per-contig x
    3-sample windows give `n_rows=18 > 5`, so a window spans multiple
    super-batches) to reproduce the uncovered combination, then assert
    byte-identical parity against the WRITTEN dataset for every cell -- same
    strength as `test_svar2_engine_matches_written` in
    `test_streaming_parity_svar2.py`."""
    fx = svar2_multicontig_fixture
    written = gvl.Dataset.open(fx.dataset_path, reference=fx.reference_path).with_seqs(
        "haplotypes"
    )
    base = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    sds = _with_strategy(base, "svar2_engine")
    assert isinstance(sds._backend, _Svar2Backend)
    # Set BEFORE iterating: `build_engine` (engine construction) and the drive
    # loop both read `backend._super_batch_rows` off the same attribute, so a
    # single override before `to_iter` keeps the engine and the Python nesting
    # in agreement.
    object.__setattr__(sds._backend, "_super_batch_rows", 5)

    seen = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=3, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp = written[r, s]
            for p in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data[i][p]),
                    np.asarray(exp[p]),
                    err_msg=f"mismatch at region={r} sample={s} ploid={p}",
                )
            seen += 1
    assert seen == fx.bed.height * sds.n_samples
