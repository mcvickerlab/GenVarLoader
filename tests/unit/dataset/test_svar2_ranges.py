"""Unit tests for the svar2 range-cache layouts (#357).

Pure numpy: no genoray store, no dataset on disk. The layouts are the one place
where "an absent cell is an empty range" becomes load-bearing, so parity against
a dense reference is asserted exhaustively and as a Hypothesis property.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from genvarloader._dataset._svar2_ranges import (
    ENTRY_DTYPE,
    _SparseRanges,
)


def _dense_reference(
    dense: np.ndarray, r_q: np.ndarray, si_q: np.ndarray, P: int
) -> tuple[np.ndarray, np.ndarray]:
    """What `_gather_inputs` does today: fancy-index (R, S, P, 2) memmaps.

    `dense` is (2, R, S, P, 2) -- channel 0 snp, channel 1 indel.
    """
    snp = np.ascontiguousarray(dense[0][r_q, si_q].reshape(-1, 2), np.int64)
    indel = np.ascontiguousarray(dense[1][r_q, si_q].reshape(-1, 2), np.int64)
    return snp, indel


def _sparse_from_dense(dense: np.ndarray, R: int, S: int, P: int) -> _SparseRanges:
    """Build the CSR table the writer would emit for this dense array."""
    snp, indel = dense[0], dense[1]
    ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
    ri, sj, pj = np.nonzero(ne)  # C-order => already key-ascending
    cell_id = (sj.astype(np.int64) * P + pj).astype(np.int32)
    ent = np.empty(len(ri), ENTRY_DTYPE)
    ent["snp_start"] = snp[ri, sj, pj, 0]
    ent["snp_len"] = snp[ri, sj, pj, 1] - snp[ri, sj, pj, 0]
    ent["indel_start"] = indel[ri, sj, pj, 0]
    ent["indel_len"] = indel[ri, sj, pj, 1] - indel[ri, sj, pj, 0]
    region_ptr = np.concatenate([[0], np.bincount(ri, minlength=R).cumsum()]).astype(
        np.int64
    )
    return _SparseRanges(region_ptr, cell_id, ent, R, S, P)


def _random_dense(rng: np.random.Generator, R: int, S: int, P: int, fill: float):
    """A dense (2, R, S, P, 2) cache with `fill` of its cells non-empty.

    Empty cells get a plausible non-zero insertion point, exactly like genoray:
    `(x, x)` with x > 0. This is what makes the parity tests meaningful -- a
    sparse lookup returns (0, 0) there, so they are NOT bit-equal, only
    equal-where-it-matters. See the spec's "empty cells carry no information".
    """
    dense = np.zeros((2, R, S, P, 2), np.int64)
    for ch in (0, 1):
        starts = rng.integers(0, 1000, size=(R, S, P))
        widths = np.where(
            rng.random((R, S, P)) < fill, rng.integers(1, 5, (R, S, P)), 0
        )
        dense[ch, ..., 0] = starts
        dense[ch, ..., 1] = starts + widths
    return dense


def _assert_parity(dense, sparse, r_q, si_q, P):
    """Sparse == dense on PRESENT cells; sparse == (0, 0) on absent ones.

    Presence is a property of the CELL, not of one channel. A cell is stored when
    the SNP range OR the indel range is non-empty, so a present cell whose SNP
    range happens to be empty carries the real `snp_start` with `snp_len == 0` and
    `lookup` returns the true insertion point `(x, x)` -- NOT `(0, 0)`. Only an
    absent cell returns zeros.

    Keying this off `d[:, 1] == d[:, 0]` per channel is wrong and fails on any
    table with mixed per-channel emptiness: 400/400 randomized grids. It is
    invisible at fill 0.0 and fill 1.0, which is exactly why a full-fill test
    would pass and mask it.
    """
    d_snp, d_indel = _dense_reference(dense, r_q, si_q, P)
    s_snp, s_indel = sparse.lookup(r_q, si_q, P)

    assert s_snp.dtype == np.int64 and s_snp.flags.c_contiguous
    assert s_snp.shape == d_snp.shape

    present = (d_snp[:, 1] > d_snp[:, 0]) | (d_indel[:, 1] > d_indel[:, 0])
    for d, s in ((d_snp, s_snp), (d_indel, s_indel)):
        np.testing.assert_array_equal(s[present], d[present])
        # An absent cell must be exactly (0, 0): unconditionally in bounds for
        # the Rust slicing in gather_haps_readbound_impl.
        np.testing.assert_array_equal(s[~present], 0)


def test_lookup_parity_partial_fill():
    rng = np.random.default_rng(0)
    R, S, P = 7, 5, 2
    dense = _random_dense(rng, R, S, P, fill=0.3)
    sparse = _sparse_from_dense(dense, R, S, P)
    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    _assert_parity(dense, sparse, r_q, si_q, P)


def test_lookup_parity_full_fill_uses_contiguous_path(monkeypatch):
    """100% fill: every region block is complete, so no binary search runs.

    Asserted, not assumed: this is the whole high-fill regime (sequence-model
    windows run at ~100% fill), measured at 0.35 ms against 0.89 ms, and a fast
    path that silently stopped firing would cost 2.5x with every test still
    green.
    """
    rng = np.random.default_rng(1)
    R, S, P = 4, 6, 2
    dense = _random_dense(rng, R, S, P, fill=1.0)
    sparse = _sparse_from_dense(dense, R, S, P)
    assert len(sparse.cell_id) == R * S * P, "fill=1.0 did not produce a full table"

    def boom(*a, **k):
        raise AssertionError("the full-block fast path did not fire")

    monkeypatch.setattr(_SparseRanges, "_lower_bound", boom)
    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    _assert_parity(dense, sparse, r_q, si_q, P)


def test_lookup_parity_mixed_full_and_partial_regions():
    """One partial region must not break the full ones sharing the query.

    The fast path is all-or-nothing per CALL -- `(hi - lo) == span` has to hold
    for every queried region -- so a mixed table sends full regions through the
    search too. They must come out identical either way; a search that assumed
    "not full" would be off by the block's own width.
    """
    rng = np.random.default_rng(12)
    R, S, P = 5, 4, 2
    dense = _random_dense(rng, R, S, P, fill=1.0)
    # Empty out one cell in region 2 only: regions 0, 1, 3, 4 stay complete.
    dense[0][2, 1, 0] = (0, 0)
    dense[1][2, 1, 0] = (0, 0)
    sparse = _sparse_from_dense(dense, R, S, P)
    assert len(sparse.cell_id) == R * S * P - 1
    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    _assert_parity(dense, sparse, r_q, si_q, P)
    # And the full regions alone must still take the fast path.
    keep = r_q != 2
    _assert_parity(dense, sparse, r_q[keep], si_q[keep], P)


def test_lookup_empty_table():
    """N == 0 must not memmap, not raise, and not probe."""
    R, S, P = 3, 4, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    snp, indel = sparse.lookup(np.array([0, 2]), np.array([1, 3]), P)
    assert snp.shape == (2 * P, 2)
    np.testing.assert_array_equal(snp, 0)
    np.testing.assert_array_equal(indel, 0)


def test_lookup_zero_queries():
    R, S, P = 3, 4, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    snp, indel = sparse.lookup(np.array([], np.int64), np.array([], np.int64), P)
    assert snp.shape == (0, 2) and indel.shape == (0, 2)


def test_lookup_duplicate_and_unsorted_queries():
    """Order and repetition must not change the answer, row for row."""
    rng = np.random.default_rng(2)
    R, S, P = 5, 4, 2
    dense = _random_dense(rng, R, S, P, fill=0.4)
    sparse = _sparse_from_dense(dense, R, S, P)
    r_q = np.array([4, 0, 4, 2, 0, 0])
    si_q = np.array([3, 1, 3, 0, 1, 2])
    _assert_parity(dense, sparse, r_q, si_q, P)


@pytest.mark.parametrize("bad", ["region", "sample"])
def test_lookup_out_of_bounds_raises(bad: str):
    """A miss and an empty cell are indistinguishable, so bounds must be checked.

    Today `vk_snp_range[r_q, si_q]` raises IndexError. Without this guard a bad
    index would silently return (0, 0) -- a reference-only haplotype, no error.
    """
    R, S, P = 3, 4, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    r_q = np.array([R if bad == "region" else 0])
    si_q = np.array([0 if bad == "region" else S])
    with pytest.raises(IndexError, match=bad):
        sparse.lookup(r_q, si_q, P)


def test_lookup_int32_indices_do_not_wrap():
    """int32 r_q * S wraps silently and the later promotion hides it."""
    R, S, P = 3, 4, 2
    rng = np.random.default_rng(3)
    dense = _random_dense(rng, R, S, P, fill=0.5)
    sparse = _sparse_from_dense(dense, R, S, P)
    r_q = np.arange(R * S, dtype=np.int32) // S
    si_q = np.arange(R * S, dtype=np.int32) % S
    _assert_parity(dense, sparse, r_q, si_q, P)


def test_entry_dtype_is_28_bytes_per_entry():
    """24-byte record + 4-byte cell_id = 28 B, against dense's 32 B per cell.

    This is what removes #357's fill threshold: sparse is strictly smaller at
    EVERY fill level, so there is no crossover and no dense-fallback writer.
    """
    assert ENTRY_DTYPE.itemsize == 24
    assert ENTRY_DTYPE.itemsize + np.dtype(np.int32).itemsize == 28
    assert ENTRY_DTYPE.names == ("snp_start", "indel_start", "snp_len", "indel_len")


@settings(deadline=None, max_examples=75)
@given(
    R=st.integers(1, 6),
    S=st.integers(1, 6),
    P=st.integers(1, 3),
    fill=st.floats(0.0, 1.0),
    seed=st.integers(0, 2**32 - 1),
    n_q=st.integers(0, 20),
)
def test_lookup_parity_property(
    R: int, S: int, P: int, fill: float, seed: int, n_q: int
):
    """Sparse and dense must agree for every grid, fill and query set."""
    rng = np.random.default_rng(seed)
    dense = _random_dense(rng, R, S, P, fill)
    sparse = _sparse_from_dense(dense, R, S, P)
    r_q = rng.integers(0, R, n_q)
    si_q = rng.integers(0, S, n_q)
    _assert_parity(dense, sparse, r_q, si_q, P)


def test_iter_entries_is_sorted_and_complete():
    """Concat's merge requires ascending, gap-free, complete key streams."""
    rng = np.random.default_rng(4)
    R, S, P = 9, 7, 2
    dense = _random_dense(rng, R, S, P, fill=0.35)
    sparse = _sparse_from_dense(dense, R, S, P)

    keys = np.concatenate(
        [k for k, _ in sparse.iter_entries()] or [np.empty(0, np.int64)]
    )
    ents = np.concatenate(
        [e for _, e in sparse.iter_entries()] or [np.empty(0, ENTRY_DTYPE)]
    )
    assert len(keys) == len(sparse.cell_id)
    assert np.all(np.diff(keys) > 0), "keys must be strictly ascending"

    snp, indel = dense[0], dense[1]
    ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
    ri, sj, pj = np.nonzero(ne)
    np.testing.assert_array_equal(keys, ri * (S * P) + sj * P + pj)
    np.testing.assert_array_equal(ents["snp_start"], snp[ri, sj, pj, 0])


def test_iter_entries_empty_table():
    R, S, P = 4, 3, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    assert list(sparse.iter_entries()) == []
