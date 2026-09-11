"""Byte-identical parity: StreamingDataset over a .svar2 store vs a written gvl.Dataset."""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl


def test_streaming_svar2_matches_written_all_cells(svar2_multicontig_fixture) -> None:
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    written = gvl.Dataset.open(fx.dataset_path, reference=fx.reference_path).with_seqs(
        "haplotypes"
    )

    seen = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
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


def test_streaming_svar2_covers_every_cell_once(svar2_multicontig_fixture) -> None:
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    cells = set()
    for _, r_idx, s_idx in sds.to_iter(batch_size=3, return_indices=True):
        for i in range(len(r_idx)):
            cells.add((int(r_idx[i]), int(s_idx[i])))
    assert cells == {(r, s) for r in range(fx.bed.height) for s in range(sds.n_samples)}


def _with_strategy(sds, strategy):
    """Test-only seam (PR-3 Task 3): force a specific `_prefetch_strategy` on a
    clone of an already-constructed `StreamingDataset` (frozen dataclass), so the
    same fixture-built dataset can be driven under both "sync" (the default) and
    "svar2_engine" (not yet the default -- Task 4 decides any flip)."""
    import copy

    clone = copy.copy(sds)
    object.__setattr__(clone, "_prefetch_strategy", strategy)
    return clone


def test_svar2_engine_matches_written(svar2_multicontig_fixture) -> None:
    """Strongest guarantee for the new `"svar2_engine"` strategy: byte-identical
    to the WRITTEN dataset (not just to "sync"), position-by-position for every
    (region, sample, ploid) cell -- same pattern as
    `test_streaming_svar2_matches_written_all_cells` above, driven through the
    `Svar2StreamEngine` (PR-3 Task 2) instead of the super-batch sync path."""
    fx = svar2_multicontig_fixture
    written = gvl.Dataset.open(fx.dataset_path, reference=fx.reference_path).with_seqs(
        "haplotypes"
    )
    base = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    sds = _with_strategy(base, "svar2_engine")
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


@pytest.mark.parametrize("strategy", ["sync", "svar2_engine"])
def test_n_batches_matches_super_batched_drive(
    svar2_multicontig_fixture, strategy
) -> None:
    """`n_batches` must count the SAME nesting the drive walks (issue #379).

    The SVAR2 drives nest `batch_size` inside `super_batch_rows`, so every
    super-batch's last batch may be partial. Counting a flat
    `range(0, n_rows, batch_size)` instead under-reports whenever
    `sb_rows % batch_size != 0`, and `len(dl)` then lies to progress bars and
    fixed-step training loops.
    """
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    sds = _with_strategy(sds, strategy)
    # Shrink the super-batch so the nesting actually bites. Overriding the
    # attribute is the sanctioned seam -- `_Svar2Backend.__init__` documents that
    # the drive READS `_super_batch_rows` rather than recomputing it, precisely so
    # test/sweep overrides stick.
    object.__setattr__(sds._backend, "_super_batch_rows", 5)
    batch_size = 3

    actual = sum(1 for _ in sds.to_iter(batch_size=batch_size))

    # Guard: these params must genuinely discriminate. If a flat count happened to
    # equal the nested one, the test would pass vacuously even with the bug present.
    flat = sum(
        -(-(len(r_idx) * len(s_idx)) // batch_size) for r_idx, s_idx in sds._plan()
    )
    assert flat != actual, (
        "fixture/params no longer exercise super-batch nesting; pick sb_rows and "
        "batch_size such that sb_rows % batch_size != 0 and n_rows % sb_rows != 0"
    )

    assert sds.n_batches(batch_size) == actual
