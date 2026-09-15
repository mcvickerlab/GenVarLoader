"""Unit tests for the svar2 range-cache layouts (#357).

Pure numpy: no genoray store, no dataset on disk. The layouts are the one place
where "an absent cell is an empty range" becomes load-bearing, so parity against
a dense reference is asserted exhaustively and as a Hypothesis property.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from genvarloader._dataset._svar2_ranges import (
    ENTRY_DTYPE,
    _DenseRanges,
    _RangeLookup,
    _ranges_reader,
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


def test_lookup_int32_indices_are_accepted():
    """int32 r_q/si_q are accepted and agree with int64 on the same query.

    This does NOT exercise wrapping: under the spec's own ``S * P < 2**31``
    invariant, ``si_q * P`` cannot overflow int32 at these (or any legal)
    sizes, and the subsequent ``+ arange(P)`` promotes to int64 regardless of
    whether ``si_q * P`` wrapped. The ``dtype=np.int64`` on that multiply is
    defence in depth against a hypothetical caller that violates the
    invariant, not something this test can force to matter.
    """
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


def test_sparse_ranges_satisfies_range_lookup_protocol() -> None:
    """Static, not runtime: pyrefly checks this assignment against `_RangeLookup`.

    The assignment itself has no runtime effect -- annotating a local doesn't
    check anything at import time or under pytest. It exists so a `pyrefly
    check` run fails the moment `_SparseRanges`'s public methods drift from the
    Protocol Tasks 4 and 5 must also match; nothing else in this module binds
    the two together.
    """
    lookup_iface: _RangeLookup = _SparseRanges(
        np.zeros(4, np.int64), np.empty(0, np.int32), np.empty(0, ENTRY_DTYPE), 3, 4, 2
    )
    assert lookup_iface.n_regions == 3


def test_post_init_rejects_region_ptr_length_mismatch():
    R, S, P = 3, 4, 2
    with pytest.raises(ValueError, match="n_regions"):
        _SparseRanges(
            np.zeros(R, np.int64),  # one short of n_regions + 1
            np.empty(0, np.int32),
            np.empty(0, ENTRY_DTYPE),
            R,
            S,
            P,
        )


def test_post_init_rejects_cell_id_cell_vk_length_mismatch():
    R, S, P = 3, 4, 2
    with pytest.raises(ValueError, match="parallel"):
        _SparseRanges(
            np.array([0, 0, 0, 1], np.int64),
            np.zeros(1, np.int32),
            np.empty(0, ENTRY_DTYPE),
            R,
            S,
            P,
        )


def test_post_init_rejects_non_monotonic_region_ptr():
    """A writer bug that emits a decreasing region_ptr must raise construction,
    not silently mis-route `lookup` to the wrong region's block.

    Before this fix, `region_ptr=[0, 2, 1, 3]` constructed without error and
    `lookup` silently returned `(0, 0)` for affected queries -- a
    reference-only haplotype with no signal anything was wrong.
    """
    R, S, P = 3, 4, 2
    with pytest.raises(ValueError, match="non-decreasing"):
        _SparseRanges(
            np.array([0, 2, 1, 3], np.int64),
            np.zeros(3, np.int32),
            np.zeros(3, ENTRY_DTYPE),
            R,
            S,
            P,
        )


def test_post_init_rejects_truncated_region_ptr():
    """`region_ptr[-1]` must equal `len(cell_id)`.

    Before this fix, `region_ptr=[0, 1, 2]` against a 3-entry `cell_id`
    constructed without error, permanently stranding the third entry: no
    region's block could ever reach it, and `lookup` would return
    wrong-but-plausible ranges for the regions that do validate.
    """
    S, P = 4, 2
    cell_id = np.zeros(3, np.int32)
    ent = np.zeros(3, ENTRY_DTYPE)
    region_ptr = np.array([0, 1, 2], np.int64)  # claims 2 entries, cell_id has 3
    with pytest.raises(ValueError, match=r"region_ptr\[-1\]"):
        _SparseRanges(region_ptr, cell_id, ent, 2, S, P)


def test_dense_post_init_rejects_wrong_shape():
    """`_DenseRanges` must validate its own shape invariant, in the same spirit
    as `_SparseRanges.__post_init__` -- a wrong-shaped array is a construction
    error, not something `lookup` should discover later via a numpy fancy-index
    error (or, worse, silently succeed on a shape that broadcasts wrong).
    """
    R, S, P = 3, 4, 2
    wrong = np.zeros((R, S, P + 1, 2), np.int64)  # ploidy off by one
    right = np.zeros((R, S, P, 2), np.int64)
    with pytest.raises(ValueError, match="vk_snp_range"):
        _DenseRanges(wrong, right, R, S, P)
    with pytest.raises(ValueError, match="vk_indel_range"):
        _DenseRanges(right, wrong, R, S, P)


def _empty_ranges(impl: type, R: int, S: int, P: int):
    """An empty (every cell absent) range table of the given layout.

    Shared by the per-layout bounds/error tests below, so both `_SparseRanges`
    and `_DenseRanges` are exercised by one test body instead of duplicating
    each check per class.
    """
    if impl is _SparseRanges:
        return _SparseRanges(
            np.zeros(R + 1, np.int64),
            np.empty(0, np.int32),
            np.empty(0, ENTRY_DTYPE),
            R,
            S,
            P,
        )
    assert impl is _DenseRanges
    return _DenseRanges(
        np.zeros((R, S, P, 2), np.int64), np.zeros((R, S, P, 2), np.int64), R, S, P
    )


@pytest.mark.parametrize("impl", [_SparseRanges, _DenseRanges])
def test_lookup_rejects_ploidy_mismatch(impl: type):
    R, S, P = 3, 4, 2
    ranges = _empty_ranges(impl, R, S, P)
    with pytest.raises(ValueError, match="ploidy"):
        ranges.lookup(np.array([0]), np.array([0]), P + 1)


@pytest.mark.parametrize("impl", [_SparseRanges, _DenseRanges])
@pytest.mark.parametrize("r0,r1", [(2, 0), (-1, 2), (0, 10)])
def test_entries_for_regions_rejects_invalid_range(impl: type, r0: int, r1: int):
    """Inverted, negative, or past-n_regions bounds must raise -- not silently
    return empty, which is indistinguishable from "the range held no cells".

    Before the sparse fix, `entries_for_regions(2, 0)` and
    `entries_for_regions(-1, 2)` both returned empty with no error (the
    inverted range's `b <= a` guard absorbed the first; negative indices simply
    wrapped into `region_ptr` for the second), while `r1 > n_regions` already
    raised a numpy IndexError -- an asymmetric guard on the one primitive
    `concat`'s merge is built on. `_DenseRanges` has no `region_ptr` to wrap
    into, so it needs its own explicit `0 <= r0 <= r1 <= n_regions` check to
    match -- this test parametrizes over both layouts to pin that.
    """
    R, S, P = 3, 4, 2
    ranges = _empty_ranges(impl, R, S, P)
    with pytest.raises(IndexError):
        ranges.entries_for_regions(r0, r1)


def test_lookup_needs_both_lo_and_hi_bound_checks():
    """The two-sided `lo <= pos < hi` hit test has two independently necessary
    halves, each guarding a distinct miss shape that the `cid[clamped] ==
    target` check alone does NOT catch.

    `_lower_bound`'s docstring says the `np.minimum` clamp exists only for a
    trailing empty region block; none of the other nine deterministic tests
    constructs one, and `test_lookup_parity_partial_fill` (R=7, S=5, fill=0.3)
    has roughly a 0.1% chance of doing so by chance. This table is built so the
    *coincidentally adjacent* `cid` value equals each query's target -- which
    is what makes each half of the hit test load-bearing rather than redundant
    with the `cid` equality check:

    - r=0, slot 2 is absent from region 0's block (only slots 0, 1 are
      present), but region 1's single entry has `cid == 2` right after region
      0's block ends. The search lands at `pos == hi`; only `pos < hi` rejects
      the bleed into region 1's entry.
    - r=2 (the LAST region) is empty, so `pos` clamps below `lo` to region 1's
      last entry, whose `cid` also happens to equal 2. Only `lo <= pos` rejects
      matching that unrelated, out-of-block entry.
    """
    R, S, P = 3, 3, 1
    cell_id = np.array([0, 1, 2], np.int32)  # region 0: slots 0, 1; region 1: slot 2
    ent = np.zeros(3, ENTRY_DTYPE)
    ent["snp_start"] = [10, 20, 30]
    ent["snp_len"] = [1, 1, 1]
    region_ptr = np.array([0, 2, 3, 3], np.int64)  # region 2 (LAST) is empty
    sparse = _SparseRanges(region_ptr, cell_id, ent, R, S, P)

    snp, indel = sparse.lookup(np.array([0, 2]), np.array([2, 2]), P)
    np.testing.assert_array_equal(snp, 0)
    np.testing.assert_array_equal(indel, 0)


def _write_dense_layout(d, dense: np.ndarray, R: int, S: int, P: int):
    """Emit the legacy dense layout at `d`, exactly as gvl <= 0.42.1 did."""
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in (("vk_snp_range", dense[0]), ("vk_indel_range", dense[1])):
        np.asarray(arr, np.int64).tofile(d / f"{name}.npy")
    for name in ("dense_snp_range", "dense_indel_range"):
        np.zeros((R, 2), np.int64).tofile(d / f"{name}.npy")
    np.save(d / "sample_cols.npy", np.arange(S, dtype=np.int64))
    (d / "svar2_meta.json").write_text(
        json.dumps(
            {
                "vk_snp_range": {"shape": [R, S, P, 2], "dtype": "<i8"},
                "vk_indel_range": {"shape": [R, S, P, 2], "dtype": "<i8"},
                "dense_snp_range": {"shape": [R, 2], "dtype": "<i8"},
                "dense_indel_range": {"shape": [R, 2], "dtype": "<i8"},
                "sample_cols": {"shape": [S], "dtype": "<i8"},
                "ploidy": P,
            }
        )
    )


def test_dense_ranges_matches_fancy_indexing(tmp_path):
    rng = np.random.default_rng(5)
    R, S, P = 5, 4, 2
    dense = _random_dense(rng, R, S, P, fill=0.5)
    _write_dense_layout(tmp_path, dense, R, S, P)

    reader = _ranges_reader(tmp_path)
    assert (reader.n_regions, reader.n_samples, reader.ploidy) == (R, S, P)

    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    got_snp, got_indel = reader.lookup(r_q, si_q, P)
    exp_snp, exp_indel = _dense_reference(dense, r_q, si_q, P)
    # The dense reader is the status quo: bit-equal, insertion points included.
    np.testing.assert_array_equal(got_snp, exp_snp)
    np.testing.assert_array_equal(got_indel, exp_indel)


def test_dense_and_sparse_iter_entries_agree(tmp_path):
    """Dense -> sparse concat feeds off _DenseRanges.iter_entries; it must match."""
    rng = np.random.default_rng(6)
    R, S, P = 6, 5, 2
    dense = _random_dense(rng, R, S, P, fill=0.4)
    _write_dense_layout(tmp_path, dense, R, S, P)

    d_keys = np.concatenate(
        [k for k, _ in _ranges_reader(tmp_path).iter_entries()]
        or [np.empty(0, np.int64)]
    )
    sparse = _sparse_from_dense(dense, R, S, P)
    s_keys = np.concatenate(
        [k for k, _ in sparse.iter_entries()] or [np.empty(0, np.int64)]
    )
    np.testing.assert_array_equal(d_keys, s_keys)


def test_entries_for_regions_agree_on_subranges(tmp_path):
    """concat asks for arbitrary [r0, r1); both layouts must answer identically.

    iter_entries only ever exercises the block boundaries the reader picks for
    itself. concat picks its own, so a dense/sparse disagreement on a partial
    region range -- an off-by-one in the region offset, say -- would slip past the
    test above and corrupt only merged datasets.
    """
    rng = np.random.default_rng(8)
    R, S, P = 7, 3, 2
    dense = _random_dense(rng, R, S, P, fill=0.4)
    _write_dense_layout(tmp_path, dense, R, S, P)
    d_reader = _ranges_reader(tmp_path)
    sparse = _sparse_from_dense(dense, R, S, P)

    for r0, r1 in ((0, 0), (0, 1), (2, 5), (3, 3), (0, R), (R, R)):
        d_key, d_ent = d_reader.entries_for_regions(r0, r1)
        s_key, s_ent = sparse.entries_for_regions(r0, r1)
        assert d_key.dtype == np.int64 and s_key.dtype == np.int64
        assert d_ent.dtype == ENTRY_DTYPE and s_ent.dtype == ENTRY_DTYPE
        np.testing.assert_array_equal(d_key, s_key)
        np.testing.assert_array_equal(d_ent, s_ent)
        # Keys must ascend; concat's merge and _SparseWriter both assume it.
        assert np.all(np.diff(d_key) > 0)


def test_dense_reader_bounds_check(tmp_path):
    """Both layouts must raise on a bad index, so callers behave identically."""
    rng = np.random.default_rng(7)
    R, S, P = 3, 3, 2
    _write_dense_layout(tmp_path, _random_dense(rng, R, S, P, 0.5), R, S, P)
    with pytest.raises(IndexError, match="region"):
        _ranges_reader(tmp_path).lookup(np.array([R]), np.array([0]), P)


def test_ranges_reader_rejects_mismatched_vk_shape(tmp_path):
    """A stale/wrong recorded vk_*_range shape must raise, not silently truncate.

    `vk_snp_range.npy`/`vk_indel_range.npy` are raw headerless `tofile` dumps,
    so `meta["vk_snp_range"]["shape"]` is the ONLY record of how to interpret
    the bytes. `np.memmap` only raises when the file is too SHORT for the
    requested shape, so before this fix a recorded shape claiming FEWER
    regions than the file actually holds was absorbed silently: the reader
    read a truncated prefix of the real grid with no error at all.

    Reproduces the reviewer's fixture: the on-disk vk arrays hold R=4 regions
    (and `meta["vk_snp_range"]["shape"]` correctly says so), but
    `dense_snp_range`'s recorded shape claims only R=2 regions.
    """
    R_real, R_claimed, S, P = 4, 2, 3, 2
    d = tmp_path
    d.mkdir(parents=True, exist_ok=True)
    dense = _random_dense(np.random.default_rng(9), R_real, S, P, fill=0.5)
    for name, arr in (("vk_snp_range", dense[0]), ("vk_indel_range", dense[1])):
        np.asarray(arr, np.int64).tofile(d / f"{name}.npy")
    # dense_*_range claims only R_claimed regions -- inconsistent with the
    # R_real-region vk_*_range files AND with vk_*_range's own recorded shape.
    for name in ("dense_snp_range", "dense_indel_range"):
        np.zeros((R_claimed, 2), np.int64).tofile(d / f"{name}.npy")
    np.save(d / "sample_cols.npy", np.arange(S, dtype=np.int64))
    (d / "svar2_meta.json").write_text(
        json.dumps(
            {
                "vk_snp_range": {"shape": [R_real, S, P, 2], "dtype": "<i8"},
                "vk_indel_range": {"shape": [R_real, S, P, 2], "dtype": "<i8"},
                "dense_snp_range": {"shape": [R_claimed, 2], "dtype": "<i8"},
                "dense_indel_range": {"shape": [R_claimed, 2], "dtype": "<i8"},
                "sample_cols": {"shape": [S], "dtype": "<i8"},
                "ploidy": P,
            }
        )
    )
    with pytest.raises(ValueError, match="vk_snp_range"):
        _ranges_reader(d)


def test_ranges_reader_rejects_unknown_layout(tmp_path):
    """An unrecognized `layout` value must raise, not fall through to dense.

    This is the forward-compat guard that matters most: a dataset written by a
    NEWER GenVarLoader with a layout this version doesn't know must fail
    loudly rather than being silently (mis)read as the legacy dense grid.
    """
    R, S, P = 3, 4, 2
    _write_dense_layout(
        tmp_path, _random_dense(np.random.default_rng(10), R, S, P, 0.3), R, S, P
    )
    meta = json.loads((tmp_path / "svar2_meta.json").read_text())
    meta["layout"] = "quadtree"
    (tmp_path / "svar2_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="quadtree"):
        _ranges_reader(tmp_path)


def test_ranges_reader_rejects_sample_cols_mismatch(tmp_path):
    """`sample_cols.npy`'s length must agree with the meta's sample count."""
    R, S, P = 3, 4, 2
    _write_dense_layout(
        tmp_path, _random_dense(np.random.default_rng(11), R, S, P, 0.3), R, S, P
    )
    np.save(tmp_path / "sample_cols.npy", np.arange(S - 1, dtype=np.int64))
    with pytest.raises(ValueError, match="sample_cols"):
        _ranges_reader(tmp_path)


def test_ranges_reader_accepts_explicit_dense_layout_key(tmp_path):
    """An explicit `"layout": "dense"` must dispatch identically to no key at all."""
    R, S, P = 3, 4, 2
    dense = _random_dense(np.random.default_rng(13), R, S, P, fill=0.4)
    _write_dense_layout(tmp_path, dense, R, S, P)
    meta = json.loads((tmp_path / "svar2_meta.json").read_text())
    meta["layout"] = "dense"
    (tmp_path / "svar2_meta.json").write_text(json.dumps(meta))

    reader = _ranges_reader(tmp_path)
    assert isinstance(reader, _DenseRanges)
    assert (reader.n_regions, reader.n_samples, reader.ploidy) == (R, S, P)


@pytest.mark.parametrize("impl", [_SparseRanges, _DenseRanges])
def test_impl_signature_matches_protocol(impl: type) -> None:
    """A pytest failure gates CI; a pyrefly ``bad-assignment`` warning does not.

    ``pyproject.toml`` sets ``bad-assignment = "warn"``, so a Protocol/impl
    signature mismatch there surfaces as one warning among hundreds already
    suppressed in a normal ``typecheck`` run. This test is the real gate: it
    fails the build the moment an implementation's parameter names, order,
    per-parameter annotation, or return annotation drift from ``_RangeLookup``.
    """
    for name in ("lookup", "entries_for_regions", "iter_entries"):
        proto_sig = inspect.signature(getattr(_RangeLookup, name), eval_str=True)
        impl_sig = inspect.signature(getattr(impl, name), eval_str=True)

        proto_params = list(proto_sig.parameters)
        impl_params = list(impl_sig.parameters)
        assert proto_params == impl_params, (
            f"{impl.__name__}.{name} parameter names/order {impl_params} diverge"
            f" from _RangeLookup.{name} {proto_params}"
        )
        proto_annots = [p.annotation for p in proto_sig.parameters.values()]
        impl_annots = [p.annotation for p in impl_sig.parameters.values()]
        assert proto_annots == impl_annots, (
            f"{impl.__name__}.{name} parameter annotations {impl_annots} diverge"
            f" from _RangeLookup.{name} {proto_annots}"
        )
        assert proto_sig.return_annotation == impl_sig.return_annotation, (
            f"{impl.__name__}.{name} return annotation"
            f" {impl_sig.return_annotation!r} diverges from _RangeLookup.{name}'s"
            f" {proto_sig.return_annotation!r}"
        )
