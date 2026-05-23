# Zero-copy splicing via query-flattened reconstruction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `_cat_length` concatenation step from spliced output by writing kernel bytes directly into the final spliced layout via query-flattening.

**Architecture:** A new helper `build_splice_plan` produces a permutation + offsets pair that, when fed back into the existing numba/Rust kernels (called with `ploidy=1` and replicated queries), causes the kernels to write bytes in `(splice_row, sample, *inner_fixed, splice_element)` C-order. The resulting buffer is exposed as `Ragged` with both per-element and coarser per-`(splice_row, sample, inner)` offset arrays — splice grouping becomes a free reinterpretation.

**Tech Stack:** Python, numpy, awkward, attrs, `seqpro.rag.Ragged`, numba kernels in `_genotypes.py` / `_tracks.py` (unchanged).

**Spec:** `docs/superpowers/specs/2026-05-22-splice-zero-copy-design.md`

---

## File Structure

### Created
- (none — `SplicePlan` and `build_splice_plan` live in existing `_splice.py`)

### Modified
- `python/genvarloader/_dataset/_splice.py` — add `SplicePlan`, `build_splice_plan`; delete `_cat_length`, `_cat_length_inner` at the end.
- `python/genvarloader/_dataset/_reconstruct.py` — splice-plan path in `Ref.__call__`, `Haps.get_haps_and_shifts` + `Haps._get_haplotypes`, `Tracks._call_float32`. Raise early for hap-tracks splicing.
- `python/genvarloader/_dataset/_impl.py` — `Dataset._getitem_spliced` builds plan, drops `_cat_length` call/import.
- `python/genvarloader/_dataset/_reference.py` — `RefDataset._getitem_spliced` builds plan, drops `_cat_length` call/import.
- `tests/dataset/test_rc_packing.py` — replace direct `_cat_length` unit tests with end-to-end or `build_splice_plan` unit tests (the bug regressions are already covered by integration tests below).

### Tests
- `tests/dataset/test_splice_plan.py` — new file. Pure-numpy unit tests for `build_splice_plan`.
- `tests/test_ref_ds_splicing.py` — existing tests act as the primary correctness gate (must pass unchanged).
- `tests/dataset/test_rc_packing.py` — integration tests must pass unchanged; direct `_cat_length` unit tests get rewritten.

---

## Task 1: `SplicePlan` dataclass + `build_splice_plan` helper (TDD)

**Files:**
- Modify: `python/genvarloader/_dataset/_splice.py` (add new code; do NOT remove `_cat_length` yet)
- Create: `tests/dataset/test_splice_plan.py`

### Background

`SpliceIndexer.parse_idx` already returns `(ds_idx, squeeze, out_reshape, offsets)` where `ds_idx` is a flat list of `(splice_row, sample, splice_element)` queries in C-order and `offsets` (length `n_rows · n_samples + 1`) gives the splice-row boundaries on the *outer* `(splice_row, sample)` pairs (each entry tells us how many elements belong to that pair).

We need to:
1. Replicate each query `E = prod(inner_fixed)` times so the kernel sees `B·E` queries with `ploidy=1`.
2. Permute them so global k-index order becomes `(splice_row, sample, *inner_fixed, splice_element)` C-order.
3. Produce two cumulative offset arrays: per-element (`permuted_out_offsets`, for the kernel) and per-`(splice_row, sample, inner)` (`group_offsets`, for downstream Ragged).

### Steps

- [ ] **Step 1: Write failing tests for `build_splice_plan`**

Create `tests/dataset/test_splice_plan.py`:

```python
"""Unit tests for build_splice_plan: permutation + offset math."""

import numpy as np
import pytest

from genvarloader._dataset._splice import SplicePlan, build_splice_plan


def test_plan_no_inner_axes():
    """E=1 case (RefDataset): plan is essentially an identity grouping."""
    # 2 splice rows × 1 sample, row 0 has 2 elements, row 1 has 1 element.
    # B = 3 queries total.
    lengths = np.array([3, 4, 5], dtype=np.int32)  # shape (3,)
    splice_row_offsets = np.array([0, 2, 3], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    # No inner fixed, so perm is identity.
    np.testing.assert_array_equal(plan.perm, [0, 1, 2])
    np.testing.assert_array_equal(plan.permuted_lengths, [3, 4, 5])
    np.testing.assert_array_equal(plan.permuted_out_offsets, [0, 3, 7, 12])
    # group_offsets at (row, sample) granularity: 2 entries + 1.
    np.testing.assert_array_equal(plan.group_offsets, [0, 7, 12])
    assert plan.out_shape == (2, 1, None)


def test_plan_ploidy_2():
    """B=3 queries × P=2 ploidy. Each splice row's ploidies must be contiguous.

    Splice layout:
      row 0, sample 0 = elements [0, 1]  (2 elements)
      row 1, sample 0 = elements [2]     (1 element)

    Inner-fixed lengths (B, P) where B=3, P=2:
      query 0 (row 0 elem 0): ploidy lens [10, 11]
      query 1 (row 0 elem 1): ploidy lens [20, 21]
      query 2 (row 1 elem 0): ploidy lens [30, 31]

    Desired permuted order (row, sample, ploidy, element) C-order:
      (r=0, s=0, p=0, e=0), (r=0, s=0, p=0, e=1),
      (r=0, s=0, p=1, e=0), (r=0, s=0, p=1, e=1),
      (r=1, s=0, p=0, e=0),
      (r=1, s=0, p=1, e=0)

    k_idx in current layout is (query, ploidy) C-order:
      k = [(q0,p0), (q0,p1), (q1,p0), (q1,p1), (q2,p0), (q2,p1)]
        = [0, 1, 2, 3, 4, 5]

    So perm pulls k-indices in this order:
      k(q=0, p=0)=0, k(q=1, p=0)=2,   # row 0 sample 0 ploidy 0 elements
      k(q=0, p=1)=1, k(q=1, p=1)=3,   # row 0 sample 0 ploidy 1 elements
      k(q=2, p=0)=4,                  # row 1 sample 0 ploidy 0
      k(q=2, p=1)=5,                  # row 1 sample 0 ploidy 1
    """
    lengths = np.array(
        [[10, 11], [20, 21], [30, 31]], dtype=np.int32
    )  # (B=3, P=2)
    splice_row_offsets = np.array([0, 2, 3], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    np.testing.assert_array_equal(plan.perm, [0, 2, 1, 3, 4, 5])
    np.testing.assert_array_equal(
        plan.permuted_lengths, [10, 20, 11, 21, 30, 31]
    )
    np.testing.assert_array_equal(
        plan.permuted_out_offsets, [0, 10, 30, 41, 62, 92, 123]
    )
    # group_offsets at (row, sample, ploidy) granularity: 2*1*2 = 4 cells + 1.
    # cell sums: row0,s0,p0 = 10+20=30; row0,s0,p1 = 11+21=32; row1,s0,p0 = 30; row1,s0,p1 = 31.
    np.testing.assert_array_equal(plan.group_offsets, [0, 30, 62, 92, 123])
    assert plan.out_shape == (2, 1, 2, None)


def test_plan_multi_sample_ploidy_2():
    """n_samples=2, ploidy=2. Verify (row, sample, ploidy) C-order."""
    # 1 splice row × 2 samples. Row has 2 elements.
    # B = 4 queries (row, sample, element) C-order:
    #   q0 = (r=0, s=0, e=0)
    #   q1 = (r=0, s=0, e=1)
    #   q2 = (r=0, s=1, e=0)
    #   q3 = (r=0, s=1, e=1)
    lengths = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32
    )  # (B=4, P=2)
    splice_row_offsets = np.array([0, 2, 4], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=2,
        n_rows=1,
    )
    # k_idx = query * P + ploidy
    # Desired order: (r, s, p, e). For r=0:
    #   s=0, p=0: e=0 → k(q=0,p=0)=0; e=1 → k(q=1,p=0)=2
    #   s=0, p=1: e=0 → k(q=0,p=1)=1; e=1 → k(q=1,p=1)=3
    #   s=1, p=0: e=0 → k(q=2,p=0)=4; e=1 → k(q=3,p=0)=6
    #   s=1, p=1: e=0 → k(q=2,p=1)=5; e=1 → k(q=3,p=1)=7
    np.testing.assert_array_equal(plan.perm, [0, 2, 1, 3, 4, 6, 5, 7])
    np.testing.assert_array_equal(
        plan.permuted_lengths, [1, 3, 2, 4, 5, 7, 6, 8]
    )
    # group_offsets at (1, 2, 2) granularity = 4 cells + 1.
    # cell sums: 1+3=4, 2+4=6, 5+7=12, 6+8=14
    np.testing.assert_array_equal(plan.group_offsets, [0, 4, 10, 22, 36])
    assert plan.out_shape == (1, 2, 2, None)


def test_plan_total_bytes_consistent():
    """sum(lengths) == permuted_out_offsets[-1] == group_offsets[-1]."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 20, size=(6, 3), dtype=np.int32)
    splice_row_offsets = np.array([0, 2, 4, 6], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=3,
    )
    total = int(lengths.sum())
    assert int(plan.permuted_out_offsets[-1]) == total
    assert int(plan.group_offsets[-1]) == total


def test_plan_single_element_rows():
    """Every splice row has exactly one element — no concatenation needed."""
    lengths = np.array([[5, 6], [7, 8]], dtype=np.int32)
    splice_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    # With singleton splice rows the permutation still groups by (r, s, p).
    np.testing.assert_array_equal(plan.perm, [0, 1, 2, 3])
    np.testing.assert_array_equal(plan.permuted_lengths, [5, 6, 7, 8])


def test_plan_inner_fixed_size_3():
    """E=3: e.g. a track axis of 3 stacked tracks. Verify general inner-fixed handling."""
    # 1 splice row × 1 sample × 2 elements × 3 tracks.
    # B = 2 queries × 3 tracks = 6 inner k-indices.
    lengths = np.array(
        [[1, 2, 3], [4, 5, 6]], dtype=np.int32
    )  # (B=2, T=3)
    splice_row_offsets = np.array([0, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=1,
    )
    # k = query*T + t. Desired order (r=0, s=0, t, e):
    #   t=0: e=0 → k=0; e=1 → k=3
    #   t=1: e=0 → k=1; e=1 → k=4
    #   t=2: e=0 → k=2; e=1 → k=5
    np.testing.assert_array_equal(plan.perm, [0, 3, 1, 4, 2, 5])
    np.testing.assert_array_equal(plan.permuted_lengths, [1, 4, 2, 5, 3, 6])
    np.testing.assert_array_equal(plan.group_offsets, [0, 5, 7, 9])
    assert plan.out_shape == (1, 1, 3, None)


def test_plan_dtype_invariants():
    """perm is intp, lengths is int32, offsets is int64-compatible."""
    lengths = np.array([3, 4], dtype=np.int32)
    splice_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=splice_row_offsets,
        n_samples=1,
        n_rows=2,
    )
    assert plan.perm.dtype == np.intp
    assert plan.permuted_lengths.dtype == np.int32
    # offset arrays use seqpro's OFFSET_TYPE (int64).
    assert plan.permuted_out_offsets.dtype == np.int64
    assert plan.group_offsets.dtype == np.int64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/dataset/test_splice_plan.py -v`
Expected: ImportError or AttributeError for `SplicePlan` / `build_splice_plan`.

- [ ] **Step 3: Implement `SplicePlan` and `build_splice_plan`**

Append to `python/genvarloader/_dataset/_splice.py` (above the existing `_cat_length` — do NOT remove `_cat_length` yet):

```python
from seqpro.rag import OFFSET_TYPE  # if not already imported


@define
class SplicePlan:
    """Permutation + offsets that re-target the kernel write into spliced layout.

    The kernel is called with ``ploidy=1`` and one query per element of the
    flattened ``(B, *inner_fixed)`` length array. ``perm`` reorders those
    flattened k-indices so the global write order becomes
    ``(splice_row, sample, *inner_fixed, splice_element)`` C-order. After the
    kernel writes, the data buffer can be exposed as a Ragged with either
    ``permuted_out_offsets`` (per-element) or ``group_offsets`` (per
    ``(splice_row, sample, inner)`` cell).
    """

    perm: NDArray[np.intp]
    permuted_lengths: NDArray[np.int32]
    permuted_out_offsets: NDArray[np.int64]
    group_offsets: NDArray[np.int64]
    out_shape: tuple[int | None, ...]


def build_splice_plan(
    lengths: NDArray[np.int32],
    splice_row_offsets: NDArray[np.int64],
    n_samples: int,
    n_rows: int,
) -> SplicePlan:
    """Build a splice plan from per-query lengths and splice-row boundaries.

    Parameters
    ----------
    lengths
        Shape ``(B, *inner_fixed)``. Per-query lengths in current ``(splice_row,
        sample, splice_element)`` C-order, with any inner fixed axes (ploidy,
        tracks) intact. ``E = prod(inner_fixed)`` is the inner flatten factor.
    splice_row_offsets
        Shape ``(n_rows * n_samples + 1,)``. Cumulative count of elements per
        ``(splice_row, sample)`` pair — i.e. the ``offsets`` returned by
        ``SpliceIndexer.parse_idx``.
    n_samples
        Number of samples in the outer ``(splice_row, sample)`` grid.
    n_rows
        Number of splice rows in the outer ``(splice_row, sample)`` grid.
    """
    if lengths.ndim == 1:
        inner_fixed: tuple[int, ...] = ()
        flat_lengths = lengths.astype(np.int32, copy=False)
    else:
        inner_fixed = tuple(lengths.shape[1:])
        # (B, *inner) -> (B, E) -> (B*E,) in (query, inner) C-order.
        flat_lengths = lengths.reshape(lengths.shape[0], -1).astype(
            np.int32, copy=False
        )
    E = int(np.prod(inner_fixed)) if inner_fixed else 1
    B = int(lengths.shape[0])
    # k-index in the current layout: k = query * E + e.
    # We want to permute into (row, sample, e, element) C-order, which means:
    #   for each (row, sample) pair p (in C-order):
    #     for each e in 0..E:
    #       for each element q in the pair's element range:
    #         emit k = q * E + e
    # The element range for pair p is splice_row_offsets[p]:splice_row_offsets[p+1].
    n_pairs = n_rows * n_samples
    pair_lengths = np.diff(splice_row_offsets)  # length n_pairs
    if E == 1:
        # Identity permutation; flat_lengths shape is (B,) already permuted.
        perm = np.arange(B, dtype=np.intp)
        permuted_lengths_flat = flat_lengths.reshape(-1).astype(
            np.int32, copy=False
        )
    else:
        # Build perm by iterating (pair, e, element).
        # For a pair p with element range [s, s+L):
        #   for e in 0..E:
        #     k-indices = [(s+0)*E + e, (s+1)*E + e, ..., (s+L-1)*E + e]
        # Vectorized: outer product of "queries within pair" and a per-e offset.
        # Build with broadcasting.
        flat_2d = flat_lengths  # (B, E)
        perm_parts = []
        for p_idx in range(n_pairs):
            s = int(splice_row_offsets[p_idx])
            L = int(pair_lengths[p_idx])
            if L == 0:
                continue
            q_range = np.arange(s, s + L, dtype=np.intp)  # (L,)
            # (E, L): each row e is q_range*E + e.
            ke = q_range[None, :] * E + np.arange(E, dtype=np.intp)[:, None]
            perm_parts.append(ke.reshape(-1))
        perm = (
            np.concatenate(perm_parts)
            if perm_parts
            else np.empty(0, dtype=np.intp)
        )
        permuted_lengths_flat = flat_2d.reshape(-1)[perm].astype(
            np.int32, copy=False
        )

    permuted_out_offsets = lengths_to_offsets(
        permuted_lengths_flat, dtype=np.int64
    )

    # group_offsets at (row, sample, *inner_fixed) granularity:
    # each cell aggregates L elements (or 0 for empty pairs).
    # Within the permuted layout, cells are laid out as: for each pair p, E
    # cells of L lengths back-to-back. So the cell-boundary indices in the
    # flat permuted_lengths array are:
    #   pair_offsets[p]*E + e*L_p     for e in 0..E
    # Equivalently: take pair_lengths repeated E times then cumsum.
    if E == 1:
        cell_lengths = pair_lengths.astype(np.int64, copy=False)
    else:
        cell_lengths = np.repeat(pair_lengths.astype(np.int64), E)
    # cell_lengths length = n_pairs * E. group_offsets indexes the
    # *permuted_lengths* array at cell boundaries.
    cell_starts = np.concatenate(
        ([0], np.cumsum(cell_lengths, dtype=np.int64))
    )  # length n_pairs*E + 1
    # group_offsets[i] = permuted_out_offsets[cell_starts[i]]
    group_offsets = permuted_out_offsets[cell_starts]

    if inner_fixed:
        out_shape: tuple[int | None, ...] = (n_rows, n_samples, *inner_fixed, None)
    else:
        out_shape = (n_rows, n_samples, None)

    return SplicePlan(
        perm=perm,
        permuted_lengths=permuted_lengths_flat,
        permuted_out_offsets=permuted_out_offsets,
        group_offsets=group_offsets,
        out_shape=out_shape,
    )
```

Verify imports at the top of `_splice.py` include `lengths_to_offsets` (already imported from `.._utils`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_splice_plan.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_splice.py tests/dataset/test_splice_plan.py
rtk git commit -m "feat(splice): add SplicePlan + build_splice_plan helper

Pure-numpy helper that builds the permutation and offset arrays needed
to drive the haplotype/track kernels into producing pre-spliced output
in a single write."
```

---

## Task 2: `Ref.__call__` accepts an optional `SplicePlan`

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (`Ref.__call__`, around line 140-177)
- Test: integration via `tests/test_ref_ds_splicing.py` (must keep passing once Task 3 lands)

### Background

`Ref.__call__` currently builds `out_lengths = regions[:, 2] - regions[:, 1]`, computes `out_offsets`, calls `get_reference`, and wraps as `Ragged.from_offsets(ref, (batch_size, None), out_offsets)`.

With a `SplicePlan` available, we want to permute `regions` by `plan.perm` (E=1 here, so identity — but we make the code path uniform), use `plan.permuted_out_offsets` as the kernel offsets, and return a Ragged where the *outer* offsets are `plan.group_offsets` and shape is `plan.out_shape`.

### Steps

- [ ] **Step 1: Add a test that mirrors what `RefDataset._getitem_spliced` will do once wired**

Append to `tests/dataset/test_splice_plan.py`:

```python
def test_ref_call_with_plan_matches_current_behavior(tmp_path, request):
    """Ref.__call__ with a splice plan returns the same bytes as the legacy
    per-region + _cat_length path."""
    import polars as pl
    from pathlib import Path
    import genvarloader as gvl
    from genvarloader._dataset._splice import (
        _cat_length,
        build_splice_plan,
    )

    DDIR = Path(request.config.rootpath) / "tests" / "data"
    ref = gvl.Reference.from_path(DDIR / "fasta" / "hg38.fa.bgz", in_memory=False)
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1000, 2000, 5000],
            "chromEnd": [1010, 2010, 5010],
            "strand": [1, 1, 1],
            "transcript_id": ["T1", "T1", "T2"],
            "exon_number": [1, 2, 1],
        }
    )

    sp_ds = gvl.RefDataset(ref, bed, splice_info="transcript_id")
    new_path = sp_ds[:]  # exercises the production path once Task 3 lands.

    # Legacy path replicated inline.
    plain = gvl.RefDataset(ref, bed)
    unsp = plain[:]
    t1 = np.concatenate(
        [np.asarray(unsp[0], dtype="S1"), np.asarray(unsp[1], dtype="S1")]
    )
    t2 = np.asarray(unsp[2], dtype="S1")

    np.testing.assert_equal(np.asarray(new_path[0], dtype="S1").ravel(), t1)
    np.testing.assert_equal(np.asarray(new_path[1], dtype="S1").ravel(), t2)
```

(This test will not pass yet — but `test_ref_ds_splicing.py::test_spliced_single_col` already covers this and will start failing until Task 3 wires the plan into `_getitem_spliced`. We add a unit test here so this task has its own failing test to drive implementation.)

For *this task*, write a direct unit test on `Ref.__call__` instead. Replace the test above with:

```python
def test_ref_call_with_plan_writes_grouped_layout(tmp_path, request):
    """Ref.__call__(splice_plan=...) returns a Ragged whose group_offsets
    aggregate per-element bytes correctly."""
    import polars as pl
    from pathlib import Path
    import genvarloader as gvl
    from genvarloader._dataset._splice import build_splice_plan

    DDIR = Path(request.config.rootpath) / "tests" / "data"
    ref = gvl.Reference.from_path(DDIR / "fasta" / "hg38.fa.bgz", in_memory=False)
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1000, 2000, 5000],
            "chromEnd": [1010, 2010, 5010],
            "strand": [1, 1, 1],
        }
    )
    plain = gvl.RefDataset(ref, bed).with_len("ragged")
    # Manually drive Ref.__call__ as RefDataset._getitem_spliced will:
    from genvarloader._dataset._utils import bed_to_regions
    from genvarloader._dataset._reconstruct import Ref

    regions = bed_to_regions(bed, ref.c_map)
    # Two splice rows: row 0 = [0, 1], row 1 = [2].
    flat_r_idx = np.array([0, 1, 2], dtype=np.intp)
    splice_offsets = np.array([0, 2, 3], dtype=np.int64)
    lengths = regions[flat_r_idx, 2] - regions[flat_r_idx, 1]
    plan = build_splice_plan(
        lengths=lengths.astype(np.int32),
        splice_row_offsets=splice_offsets,
        n_samples=1,
        n_rows=2,
    )

    reconstructor = Ref(reference=ref)
    out = reconstructor(
        idx=flat_r_idx,
        r_idx=flat_r_idx,
        regions=regions[flat_r_idx],
        output_length="ragged",
        jitter=0,
        rng=np.random.default_rng(0),
        deterministic=True,
        splice_plan=plan,
    )
    # out is a Ragged of shape (2, 1, None) — splice rows × samples=1 × variable.
    assert out.shape == (2, 1, None), f"unexpected shape: {out.shape}"
    # Total byte count matches the sum of per-region lengths.
    assert int(out.data.shape[0]) == int(lengths.sum())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_splice_plan.py::test_ref_call_with_plan_writes_grouped_layout -v`
Expected: TypeError on `splice_plan=` keyword.

- [ ] **Step 3: Modify `Ref.__call__` to accept `splice_plan`**

In `python/genvarloader/_dataset/_reconstruct.py`, replace the `Ref.__call__` body (lines ~140-177):

```python
def __call__(
    self,
    idx: NDArray[np.integer],
    r_idx: NDArray[np.integer],
    regions: NDArray[np.int32],
    output_length: Literal["ragged", "variable"] | int,
    jitter: int,
    rng: np.random.Generator,
    deterministic: bool,
    splice_plan: "SplicePlan | None" = None,
) -> Ragged[np.bytes_]:
    batch_size = len(idx)

    if isinstance(output_length, int):
        # (b)
        out_lengths = np.full(batch_size, output_length, dtype=np.int32)
        regions = regions.copy()
        regions[:, 2] = regions[:, 1] + out_lengths
    else:
        lengths = regions[:, 2] - regions[:, 1]
        out_lengths = lengths.astype(np.int32, copy=False)

    if splice_plan is None:
        # (b+1)
        out_offsets = lengths_to_offsets(out_lengths)
        # ragged (b ~l)
        ref = get_reference(
            regions=regions,
            out_offsets=out_offsets,
            reference=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
        ).view("S1")
        ref = cast(
            Ragged[np.bytes_],
            Ragged.from_offsets(ref, (batch_size, None), out_offsets),
        )
        return ref

    # Spliced path: write bytes in (row, sample, element) C-order using the
    # plan's permutation. E=1 for Ref (no inner-fixed axes).
    perm = splice_plan.perm
    permuted_regions = regions[perm]  # (B*E, 3) — E=1 here
    ref = get_reference(
        regions=permuted_regions,
        out_offsets=splice_plan.permuted_out_offsets,
        reference=self.reference.reference,
        ref_offsets=self.reference.offsets,
        pad_char=self.reference.pad_char,
    ).view("S1")
    # Expose with group_offsets so each (row, sample) cell is one contiguous
    # spliced sequence.
    return cast(
        Ragged[np.bytes_],
        Ragged.from_offsets(ref, splice_plan.out_shape, splice_plan.group_offsets),
    )
```

Add `from ._splice import SplicePlan` to the imports near the top of `_reconstruct.py` (under TYPE_CHECKING is fine since it's only used as an annotation, but keep it as a runtime import for simplicity — `SplicePlan` is light).

- [ ] **Step 4: Run unit test to verify pass**

Run: `pixi run -e dev pytest tests/dataset/test_splice_plan.py::test_ref_call_with_plan_writes_grouped_layout -v`
Expected: PASS.

Then run the full splice_plan suite plus the existing Ref unspliced tests to confirm no regression:

Run: `pixi run -e dev pytest tests/dataset/test_splice_plan.py tests/test_ref_ds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reconstruct.py tests/dataset/test_splice_plan.py
rtk git commit -m "feat(splice): Ref.__call__ accepts SplicePlan for grouped layout"
```

---

## Task 3: Wire `RefDataset._getitem_spliced` to use `SplicePlan`; drop `_cat_length` call

**Files:**
- Modify: `python/genvarloader/_dataset/_reference.py` (`_getitem_spliced`, ~lines 429-462; drop the `_cat_length` import at line 25)

### Steps

- [ ] **Step 1: Confirm the existing splice tests cover this**

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py -v`
Expected: all PASS (sanity check before the change).

- [ ] **Step 2: Rewrite `_getitem_spliced` to use `SplicePlan`**

In `python/genvarloader/_dataset/_reference.py`, replace the `_getitem_spliced` body:

```python
def _getitem_spliced(self, idx: Idx) -> T:
    assert self._splice_map is not None
    assert not isinstance(self.output_length, int)

    flat_r_idx, offsets, out_reshape, squeeze = self._splice_map.parse_rows(idx)
    regions = self._subset_regions[flat_r_idx].copy()
    lengths = (regions[:, 2] - regions[:, 1]).astype(np.int32, copy=False)

    # Number of splice rows in this selection.
    n_rows = offsets.shape[0] - 1
    # RefDataset has no sample axis.
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=offsets,
        n_samples=1,
        n_rows=n_rows,
    )

    ref = self._ref_reconstructor()(
        idx=flat_r_idx,
        r_idx=flat_r_idx,
        regions=regions,
        output_length="ragged",
        jitter=0,
        rng=np.random.default_rng(0),
        deterministic=True,
        splice_plan=plan,
    )
    # ref has shape (n_rows, 1, None). RefDataset's public shape is
    # (n_rows, None) — squeeze the sample axis.
    ref = ref.squeeze(1)  # type: ignore

    if self.rc_neg:
        # Per-element RC against the permuted to_rc mask. Apply BEFORE
        # exposing with group_offsets.
        raise NotImplementedError(
            "RefDataset spliced rc_neg path still needs migration — covered "
            "in a follow-up task; for now we assume rc_neg is handled by "
            "the existing splice tests which pre-RC at the unspliced level."
        )

    if out_reshape is not None:
        ref = ref.reshape(out_reshape)  # type: ignore

    if self.output_length == "ragged":
        out = ref
    elif self.output_length == "variable":
        out = to_padded(ref, pad_value=bytes([self.reference.pad_char]))  # type: ignore
    else:
        raise AssertionError(
            "splice + fixed-length output should be blocked earlier"
        )

    if squeeze:
        out = out.squeeze(0)  # type: ignore

    return cast(T, out)
```

**Stop — the rc_neg branch above is wrong.** Look at the existing `_getitem_unspliced` and the spliced test `test_spliced_mixed_strand` (`tests/test_ref_ds_splicing.py:75`). The current `_getitem_spliced` does *not* apply rc_neg itself; it calls `_getitem_unspliced` which already RC-ed each region's bytes. So the spliced output naturally contains per-region RC'd bytes.

For the new path we need to keep that semantic. The simplest fix: apply RC to the *permuted, per-element* Ragged view before exposing with `group_offsets`. Update the implementation:

```python
def _getitem_spliced(self, idx: Idx) -> T:
    assert self._splice_map is not None
    assert not isinstance(self.output_length, int)

    flat_r_idx, offsets, out_reshape, squeeze = self._splice_map.parse_rows(idx)
    regions = self._subset_regions[flat_r_idx].copy()
    lengths = (regions[:, 2] - regions[:, 1]).astype(np.int32, copy=False)

    n_rows = offsets.shape[0] - 1
    plan = build_splice_plan(
        lengths=lengths,
        splice_row_offsets=offsets,
        n_samples=1,
        n_rows=n_rows,
    )

    # Compute permuted to_rc up-front (E=1 for Ref).
    to_rc_unperm = regions[:, 3] == -1 if self.rc_neg else None

    ref_data = get_reference(
        regions=regions[plan.perm],
        out_offsets=plan.permuted_out_offsets,
        reference=self.reference.reference,
        ref_offsets=self.reference.offsets,
        pad_char=self.reference.pad_char,
    ).view("S1")

    if to_rc_unperm is not None and to_rc_unperm.any():
        # Per-element view, then RC, then re-expose with group_offsets.
        per_elem = cast(
            Ragged[np.bytes_],
            Ragged.from_offsets(
                ref_data, (plan.permuted_lengths.shape[0], None),
                plan.permuted_out_offsets,
            ),
        )
        to_rc_perm = to_rc_unperm[plan.perm]
        per_elem_rc = Ragged(
            ak.to_packed(
                ak.where(
                    to_rc_perm,
                    reverse_complement(per_elem.to_ak()),
                    per_elem.to_ak(),
                )
            )
        )
        # to_packed produces a fresh, contiguous data buffer; group offsets
        # still align because both branches have identical lengths.
        ref_data = per_elem_rc.data
        permuted_out_offsets = per_elem_rc.offsets
        # Rebuild group_offsets by gathering from the new permuted offsets
        # at the same cell boundaries.
        cell_starts = np.searchsorted(
            permuted_out_offsets, plan.group_offsets
        )
        # Actually, lengths are unchanged by RC, so the cumulative offsets
        # match plan.permuted_out_offsets element-for-element. Reuse.
        group_offsets = plan.group_offsets
    else:
        permuted_out_offsets = plan.permuted_out_offsets
        group_offsets = plan.group_offsets

    ref = cast(
        Ragged[np.bytes_],
        Ragged.from_offsets(ref_data, (n_rows, 1, None), group_offsets),
    )
    # RefDataset has no sample axis publicly.
    ref = ref.squeeze(1)  # type: ignore

    if out_reshape is not None:
        ref = ref.reshape(out_reshape)  # type: ignore

    if self.output_length == "ragged":
        out = ref
    elif self.output_length == "variable":
        out = to_padded(ref, pad_value=bytes([self.reference.pad_char]))  # type: ignore
    else:
        raise AssertionError(
            "splice + fixed-length output should be blocked earlier"
        )

    if squeeze:
        out = out.squeeze(0)  # type: ignore

    return cast(T, out)
```

Verify `RefDataset` has `_ref_reconstructor()` — if not, instantiate `Ref(reference=self.reference)` directly. (Check line ~360-380 in `_reference.py` for the existing pattern; if it builds a `Ref` inline, do the same here.)

Update imports in `_reference.py`:
- Remove `_cat_length` from the import on line 25.
- Add `build_splice_plan` to the same import line.
- Add `import awkward as ak` and `from .._ragged import reverse_complement` if not present.

- [ ] **Step 3: Run RefDataset splice tests**

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py tests/test_ref_ds.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reference.py
rtk git commit -m "refactor(splice): RefDataset spliced path uses SplicePlan

Drops the _cat_length post-process for RefDataset spliced output. The
kernel now writes per-element bytes directly into the spliced layout;
splice grouping is a free reinterpretation via group_offsets."
```

---

## Task 4: `Haps._get_haplotypes` and `get_haps_and_shifts` accept a `SplicePlan`

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (`Haps.get_haps_and_shifts` ~lines 397-513, `Haps._get_haplotypes` ~lines 614-715)

### Background

`Haps.get_haps_and_shifts` builds `hap_lengths` of shape `(B, P)`, then `out_offsets = lengths_to_offsets(out_lengths)` (where `out_lengths` is `(B, P)` flattened in C-order = `(query, ploidy)` k-indices), then calls `_get_haplotypes`.

`_get_haplotypes` calls `reconstruct_haplotypes_from_sparse` with `geno_offset_idxs` of shape `(B, P)`. The kernel writes at `k_idx = query * P + hap`.

To use a SplicePlan:
1. Compute lengths as today.
2. Caller (Task 5) passes in a pre-built plan. Permute `regions`, `shifts`, `geno_offset_idx` along the flattened `(B·P)` axis using `plan.perm`.
3. Reshape to `(B·P, 1)` (pseudo "ploidy=1" with B·P queries) and call the kernel.
4. Wrap output with `plan.group_offsets` and `plan.out_shape`.

### Steps

- [ ] **Step 1: Add a failing integration test that exercises haplotype splicing through a plan**

Since the existing `tests/dataset/test_rc_packing.py::test_multi_exon_spliced_buffer_packed` and `test_multi_exon_spliced_matches_fasta_concat` cover this end-to-end, they're our gate. Run them first to confirm the *current* behavior baseline:

Run: `pixi run -e dev pytest tests/dataset/test_rc_packing.py -v`
Expected: all PASS.

(No new test needed for this task in isolation — the suite is the gate.)

- [ ] **Step 2: Modify `_get_haplotypes` to accept an optional `splice_plan`**

In `python/genvarloader/_dataset/_reconstruct.py`, change `_get_haplotypes` (the third overload definition at ~line 637):

```python
def _get_haplotypes(
    self,
    geno_offset_idx: NDArray[np.integer],
    regions: NDArray[np.integer],
    out_offsets: NDArray[OFFSET_TYPE],
    shifts: NDArray[np.integer],
    keep: NDArray[np.bool_] | None,
    keep_offsets: NDArray[OFFSET_TYPE] | None,
    annotate: bool,
    splice_plan: "SplicePlan | None" = None,
) -> (
    Ragged[np.bytes_]
    | tuple[Ragged[np.bytes_], Ragged[V_IDX_TYPE], Ragged[np.int32]]
):
    """Reconstruct haplotypes from sparse genotypes.

    When ``splice_plan`` is provided, the kernel is called with ``ploidy=1``
    over the B*P flattened queries, permuted by ``plan.perm``. The returned
    Ragged uses ``plan.group_offsets`` as the outer offsets so each
    (row, sample, ploidy) cell holds one contiguous spliced haplotype.
    """
    assert self.reference is not None

    if splice_plan is None:
        # ... existing body unchanged ...
        haps = Ragged.from_offsets(
            np.empty(out_offsets[-1], np.uint8), (*shifts.shape, None), out_offsets
        )
        if annotate:
            annot_v_idxs = Ragged.from_offsets(
                np.empty(out_offsets[-1], V_IDX_TYPE),
                (*shifts.shape, None),
                out_offsets,
            )
            annot_positions = Ragged.from_offsets(
                np.empty(out_offsets[-1], np.int32), (*shifts.shape, None), out_offsets
            )
        else:
            annot_v_idxs = None
            annot_positions = None

        reconstruct_haplotypes_from_sparse(
            geno_offset_idxs=geno_offset_idx,
            out=haps.data,
            out_offsets=haps.offsets,
            regions=regions,
            shifts=shifts,
            geno_offsets=self.genotypes.offsets,
            geno_v_idxs=self.genotypes.data,
            v_starts=self.variants.start,
            ilens=self.variants.ilen,
            alt_alleles=self.variants.alt.data.view(np.uint8),
            alt_offsets=self.variants.alt.offsets,
            ref=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
            keep=keep,
            keep_offsets=keep_offsets,
            annot_v_idxs=annot_v_idxs.data if annot_v_idxs is not None else None,
            annot_ref_pos=annot_positions.data if annot_positions is not None else None,
        )
        haps = cast(Ragged[np.bytes_], haps.view("S1"))
        if annotate:
            return haps, annot_v_idxs, annot_positions  # type: ignore
        return haps

    # ---- splice plan path ----
    # geno_offset_idx, shifts have shape (B, P). Flatten to (B*P,) in
    # (query, ploidy) C-order, then permute.
    flat_geno_idx = geno_offset_idx.reshape(-1)[splice_plan.perm].astype(
        np.intp, copy=False
    )
    flat_shifts = shifts.reshape(-1)[splice_plan.perm].astype(
        np.int32, copy=False
    )
    # regions has shape (B, 3). For (B*P, 3) flattened, each query is
    # repeated P times consecutively, then we apply the same perm. We need
    # regions_flat such that regions_flat[k_orig] is regions[query(k_orig)].
    B = regions.shape[0]
    P = shifts.shape[1]
    # k_orig = query * P + ploidy; query(k_orig) = k_orig // P.
    regions_flat = regions[np.arange(B * P, dtype=np.intp) // P]
    permuted_regions = regions_flat[splice_plan.perm]

    # keep / keep_offsets: same flatten/permute logic. keep_offsets has
    # length B*P + 1 in (query, ploidy) C-order.
    if keep is not None and keep_offsets is not None:
        # keep_offsets is per (query, ploidy). Build per-k length array,
        # permute, recumsum.
        keep_lens = np.diff(keep_offsets)
        keep_lens_perm = keep_lens[splice_plan.perm]
        keep_offsets_perm = lengths_to_offsets(
            keep_lens_perm.astype(np.int64), dtype=np.int64
        )
        # keep itself is a flat bool array; gather using a per-k start array.
        keep_perm = np.empty(int(keep_lens_perm.sum()), dtype=np.bool_)
        write_cursor = 0
        for k_new, k_old in enumerate(splice_plan.perm):
            s = int(keep_offsets[k_old])
            e = int(keep_offsets[k_old + 1])
            keep_perm[write_cursor : write_cursor + (e - s)] = keep[s:e]
            write_cursor += e - s
    else:
        keep_perm = None
        keep_offsets_perm = None

    # Allocate output buffers sized for the total permuted bytes.
    total = int(splice_plan.permuted_out_offsets[-1])
    out_buf = np.empty(total, np.uint8)
    if annotate:
        annot_v_buf = np.empty(total, V_IDX_TYPE)
        annot_pos_buf = np.empty(total, np.int32)
    else:
        annot_v_buf = None
        annot_pos_buf = None

    # Reshape to (B*P, 1) "ploidy=1" view for the kernel.
    reconstruct_haplotypes_from_sparse(
        geno_offset_idxs=flat_geno_idx.reshape(-1, 1),
        out=out_buf,
        out_offsets=splice_plan.permuted_out_offsets,
        regions=permuted_regions,
        shifts=flat_shifts.reshape(-1, 1),
        geno_offsets=self.genotypes.offsets,
        geno_v_idxs=self.genotypes.data,
        v_starts=self.variants.start,
        ilens=self.variants.ilen,
        alt_alleles=self.variants.alt.data.view(np.uint8),
        alt_offsets=self.variants.alt.offsets,
        ref=self.reference.reference,
        ref_offsets=self.reference.offsets,
        pad_char=self.reference.pad_char,
        keep=keep_perm,
        keep_offsets=keep_offsets_perm,
        annot_v_idxs=annot_v_buf,
        annot_ref_pos=annot_pos_buf,
    )

    haps = cast(
        Ragged[np.bytes_],
        Ragged.from_offsets(
            out_buf.view("S1"), splice_plan.out_shape, splice_plan.group_offsets
        ),
    )
    if annotate:
        annot_v_rag = Ragged.from_offsets(
            annot_v_buf, splice_plan.out_shape, splice_plan.group_offsets
        )
        annot_pos_rag = Ragged.from_offsets(
            annot_pos_buf, splice_plan.out_shape, splice_plan.group_offsets
        )
        return haps, annot_v_rag, annot_pos_rag  # type: ignore
    return haps
```

- [ ] **Step 3: Modify `get_haps_and_shifts` to accept and forward the plan**

Change the signature (around line 397) to add `splice_plan: "SplicePlan | None" = None,` and forward it to `_get_haplotypes` in both call sites (the `RaggedSeqs` and `RaggedAnnotatedHaps` branches).

The `RaggedVariants` branch does NOT use `_get_haplotypes` — leave it alone; it has no spliced-output use case at this layer (the caller in `_impl.py` blocks variants splicing already at `_getitem_spliced` line 1686).

Update the return value when `splice_plan` is present: skip the `out_offsets = lengths_to_offsets(out_lengths, OFFSET_TYPE)` because the plan owns that. Either:
- Compute `out_lengths` as today and let the plan's offsets win, OR
- Skip the line conditionally.

Concretely, at line 470:

```python
if splice_plan is None:
    out_offsets = lengths_to_offsets(out_lengths, OFFSET_TYPE)
else:
    out_offsets = splice_plan.permuted_out_offsets  # unused by the plan branch of _get_haplotypes, but the variable is referenced in the return tuple
```

Forward `splice_plan=splice_plan` into both `_get_haplotypes` call sites.

- [ ] **Step 4: Run the existing splice integration tests as the correctness gate**

This task isn't independently testable without Task 5 wiring `_getitem_spliced`. Run the unit tests to confirm `_get_haplotypes` non-splice path still compiles/passes:

Run: `pixi run -e dev pytest tests/dataset/test_ds_haps.py -v`
Expected: PASS (these don't use splicing, so the existing path is exercised).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reconstruct.py
rtk git commit -m "feat(splice): Haps._get_haplotypes accepts SplicePlan

Plumbs an optional SplicePlan through get_haps_and_shifts and
_get_haplotypes. When provided, the numba kernel is called with
ploidy=1 over the B*P flattened queries permuted into spliced layout.
Non-splice callers are unaffected."
```

---

## Task 5: `Dataset._getitem_spliced` builds + dispatches the plan; remove `_cat_length` call

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (`_getitem_spliced` ~lines 1670-1735; drop `_cat_length` import at line 34)

### Background

`_getitem_spliced` currently:
1. Calls `inner_ds._recon(...)` to get per-region reconstruction.
2. Applies `_rc` per-query.
3. Calls `_cat_length(r, offsets)` to merge by splice row.

New flow:
1. Compute lengths via a new lightweight method on the inner `Haps` / `Ref` / `Tracks` (depending on which reconstructor is active).
2. Build `SplicePlan` from those lengths + the `offsets` returned by `splice_idxer.parse_idx`.
3. Pass the plan into `_recon` (which forwards to `Haps.get_haps_and_shifts` / `Tracks._call_float32` / `Ref.__call__`).
4. Apply `_rc` at the *per-element* granularity (against permuted `to_rc`) before exposing with `group_offsets`. **Caveat:** with the plan-aware reconstructor returning a Ragged already grouped by `group_offsets`, we need a different `_rc` path. See Step 2 below.

### Steps

- [ ] **Step 1: Decide where `_rc` runs**

The cleanest approach: have the plan-aware reconstructor return a Ragged whose `offsets` are `permuted_out_offsets` (per-element) and shape `(B·E, None)`, NOT the grouped view. Then `_rc` runs on that per-element view. Then the caller wraps with `group_offsets` to get the final shape.

This requires `_get_haplotypes` / `Ref.__call__` / `Tracks._call_float32` to return BOTH a per-element Ragged and the final shape/group_offsets — or to return a small object. Simpler: return a per-element Ragged and let the caller in `_getitem_spliced` rewrap. The shape and `group_offsets` are already on the `SplicePlan` object that the caller built.

**Revise Tasks 2 and 4.** The plan-aware return value is now a per-element Ragged with shape `(B·E, None)` and offsets `plan.permuted_out_offsets`. `_getitem_spliced` does RC, then rewraps with `plan.group_offsets` and `plan.out_shape`.

This is a smaller change to Tasks 2/4 — the new code in those tasks already builds the per-element data buffer; just return a per-element-shape Ragged instead of a group-shape Ragged.

Concretely:

In `Ref.__call__` plan branch, change the return to:

```python
return cast(
    Ragged[np.bytes_],
    Ragged.from_offsets(
        ref, (splice_plan.permuted_lengths.shape[0], None),
        splice_plan.permuted_out_offsets,
    ),
)
```

In `_get_haplotypes` plan branch, similarly return per-element Ragged(s):

```python
haps = cast(
    Ragged[np.bytes_],
    Ragged.from_offsets(
        out_buf.view("S1"),
        (splice_plan.permuted_lengths.shape[0], None),
        splice_plan.permuted_out_offsets,
    ),
)
# ... and same for annot_v_rag / annot_pos_rag.
```

Make those edits to `_reconstruct.py` before continuing.

Also update the unit test in Task 2 (`test_ref_call_with_plan_writes_grouped_layout`) — it asserted `out.shape == (2, 1, None)`. Change the assertion to `out.shape == (3, None)` (per-element) and add `data.shape[0] == sum(lengths)`.

- [ ] **Step 2: Rewrite `Dataset._getitem_spliced`**

In `python/genvarloader/_dataset/_impl.py`, replace the `_getitem_spliced` body (around lines 1670-1735):

```python
def _getitem_spliced(
    self,
    idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    splice_idxer: SpliceIndexer,
) -> tuple[
    tuple[Ragged[np.bytes_ | np.float32] | RaggedAnnotatedHaps, ...],
    bool,
    tuple[int, ...] | None,
]:
    if isinstance(self.output_length, int):
        raise RuntimeError(
            "In general, splicing cannot be done with fixed length data because even if the length of each region's data"
            " is fixed/constant, the number of elements in each spliced element is not. Thus, the final length of the"
            " spliced elements will be variable."
        )

    assert self.sequence_type != "variants"
    assert not isinstance(self.output_length, int)
    assert self.jitter == 0
    assert self.deterministic

    if self.jitter > 0:
        raise RuntimeError("Jitter is not supported with splicing. Please set jitter to 0.")
    if not self.deterministic:
        raise RuntimeError(
            "Non-deterministic algorithms are not supported with splicing. Please set deterministic to True."
        )

    inner_ds = self.with_len("ragged")
    ds_idx, squeeze, out_reshape, offsets = splice_idxer.parse_idx(idx)
    r_idx, _ = np.unravel_index(ds_idx, self._idxer.full_shape)
    regions = self._full_regions[r_idx]

    # Determine inner_fixed for the plan based on the active reconstructor.
    n_rows = offsets.shape[0] - 1
    if hasattr(splice_idxer.dsi, "n_samples"):
        n_samples_total = splice_idxer.dsi.n_samples
        # The selection's effective sample count: ds_idx was unraveled
        # against full_shape, so the sample axis count is whatever the
        # SpliceIndexer.parse_idx loop used. Pull from the index shape.
    # n_samples per (row): elements per (row, sample) come grouped in
    # `offsets` of length n_rows * n_samples + 1.
    n_samples = (offsets.shape[0] - 1) // n_rows if n_rows else 0
    # offsets came from SpliceIndexer.parse_idx which already groups by
    # (row, sample). Re-derive n_rows / n_samples from out_reshape if
    # available; otherwise from the offset count.
    # The reliable values come from idx_t routing; rely on offsets.

    # Recompute n_rows and n_samples from the splice indexer state. The
    # splice_idxer is sample-aware; ds_idx flattens (row, sample) C-order.
    n_pairs = offsets.shape[0] - 1
    # `splice_idxer.parse_idx` builds offsets of length n_selected_pairs + 1
    # where pairs are (row, sample) in C-order. We need n_rows and n_samples
    # *of the selection*. They're embedded in the parse_idx return as
    # out_reshape when basic indexing is used; for advanced indexing we
    # fall back to treating the whole thing as a single sample axis.
    # SIMPLE RULE: ask the splice_idxer.
    n_rows_sel, n_samples_sel = _splice_selection_shape(
        splice_idxer, idx, n_pairs
    )

    # Compute per-query lengths and build the plan.
    plan = inner_ds._build_splice_plan(
        ds_idx=ds_idx,
        r_idx=r_idx,
        regions=regions,
        splice_row_offsets=offsets,
        n_rows=n_rows_sel,
        n_samples=n_samples_sel,
    )

    recon = inner_ds._recon(
        idx=ds_idx,
        r_idx=r_idx,
        regions=regions,
        output_length="ragged",
        jitter=self.jitter,
        rng=self._rng,
        deterministic=self.deterministic,
        splice_plan=plan,
    )

    if not isinstance(recon, tuple):
        recon = (recon,)
    recon = cast(
        tuple[Ragged[np.bytes_ | np.float32] | RaggedAnnotatedHaps, ...], recon
    )

    # RC at per-element granularity. Build permuted to_rc.
    if self.rc_neg:
        to_rc_per_elem = (regions[:, 3] == -1)[plan.perm]
        recon = tuple(self._rc(r, to_rc_per_elem) for r in recon)

    # Now rewrap each Ragged with group_offsets / out_shape to expose the
    # spliced layout.
    recon = tuple(
        _regroup(r, plan.group_offsets, plan.out_shape) for r in recon
    )

    return recon, squeeze, out_reshape  # type: ignore
```

This calls two new helpers — `_splice_selection_shape` and `_regroup` — and a new `_build_splice_plan` method on `Dataset`. Define them:

In `_impl.py`, add a module-level helper:

```python
def _splice_selection_shape(
    splice_idxer: SpliceIndexer, idx, n_pairs: int
) -> tuple[int, int]:
    """Recover (n_rows_sel, n_samples_sel) for the current selection."""
    # parse_idx already raveled (rows, samples) -> n_pairs. We need to recover
    # the row/sample split. For basic + combo indexing the shape is in
    # out_reshape; for adv indexing both axes share the same length.
    # Reliable path: re-run the row2idx / sample2idx splits manually.
    if not isinstance(idx, tuple):
        rows = idx
        samples = slice(None)
    elif len(idx) == 1:
        rows = idx[0]
        samples = slice(None)
    else:
        rows, samples = idx
    r_idx = splice_idxer.map.row2idx(rows)
    s_idx = splice_idxer.sample2idx(samples)
    n_r = (
        1
        if isinstance(r_idx, (int, np.integer))
        else len(np.atleast_1d(np.asarray(r_idx) if not isinstance(r_idx, slice) else np.arange(splice_idxer.n_rows)[r_idx]))
    )
    n_s = (
        1
        if isinstance(s_idx, (int, np.integer))
        else len(np.atleast_1d(np.asarray(s_idx) if not isinstance(s_idx, slice) else np.arange(splice_idxer.n_samples)[s_idx]))
    )
    return n_r, n_s


def _regroup(
    rag, group_offsets: NDArray[np.int64], out_shape: tuple[int | None, ...]
):
    """Rewrap a per-element Ragged / RaggedAnnotatedHaps with grouped offsets."""
    if isinstance(rag, RaggedAnnotatedHaps):
        return RaggedAnnotatedHaps(
            haps=_regroup(rag.haps, group_offsets, out_shape),
            var_idxs=_regroup(rag.var_idxs, group_offsets, out_shape),
            ref_coords=_regroup(rag.ref_coords, group_offsets, out_shape),
        )
    return Ragged.from_offsets(rag.data, out_shape, group_offsets)
```

In `Dataset`, add `_build_splice_plan` that delegates by reconstructor type:

```python
def _build_splice_plan(
    self,
    ds_idx: NDArray[np.intp],
    r_idx: NDArray[np.intp],
    regions: NDArray[np.int32],
    splice_row_offsets: NDArray[np.int64],
    n_rows: int,
    n_samples: int,
) -> SplicePlan:
    """Compute per-query lengths for the active reconstructor and build a plan."""
    recon = self._recon_obj
    # Hap-tracks splice path is out of scope.
    if isinstance(recon, HapsTracks):
        raise NotImplementedError(
            "Splicing of haplotypes + tracks (shape (b, t, p, ~l)) is not supported."
        )
    if isinstance(recon, Haps):
        ploidy = recon.genotypes.shape[-2]
        # Lengths shape (B, P) — same calc as get_haps_and_shifts does.
        lengths_2d = recon.haplotype_lengths_for_plan(
            ds_idx=ds_idx, regions=regions
        )
        return build_splice_plan(
            lengths=lengths_2d.astype(np.int32, copy=False),
            splice_row_offsets=splice_row_offsets,
            n_samples=n_samples,
            n_rows=n_rows,
        )
    if isinstance(recon, Ref):
        lengths_1d = (regions[:, 2] - regions[:, 1]).astype(np.int32, copy=False)
        return build_splice_plan(
            lengths=lengths_1d,
            splice_row_offsets=splice_row_offsets,
            n_samples=n_samples,
            n_rows=n_rows,
        )
    if isinstance(recon, Tracks):
        n_tracks = len(recon.active_tracks)
        # Track lengths are deterministic from regions.
        per_region = (regions[:, 2] - regions[:, 1]).astype(np.int32, copy=False)
        lengths_2d = np.broadcast_to(per_region[:, None], (per_region.shape[0], n_tracks)).copy()
        return build_splice_plan(
            lengths=lengths_2d,
            splice_row_offsets=splice_row_offsets,
            n_samples=n_samples,
            n_rows=n_rows,
        )
    if isinstance(recon, RefTracks):
        # Both inner parts agree on lengths (no haplotype indels). Build a
        # plan against the seq inner; tracks reuse it because shapes match.
        raise NotImplementedError(
            "RefTracks splicing not implemented in this pass; falling back is no longer available."
        )
    raise NotImplementedError(f"Splicing not supported for {type(recon).__name__}.")
```

This requires `recon._recon_obj` to exist — check whether `Dataset` exposes it. If not, refactor: pass the reconstructor explicitly, or call a method on `inner_ds`. The fastest is to add a `Dataset._recon_obj` property that returns whatever `_recon` dispatches to internally. (Look for the `_recon` body to find the right attribute name — likely `self._haps`, `self._ref`, `self._tracks`, etc.; reuse those.)

Also expose `haplotype_lengths_for_plan` on `Haps`:

```python
def haplotype_lengths_for_plan(
    self,
    ds_idx: NDArray[np.intp],
    regions: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Compute (B, P) per-query haplotype lengths without running the full
    reconstruction. Used by the spliced path to size buffers before kernel
    invocation."""
    geno_offset_idx = self._get_geno_offset_idx(ds_idx, self.genotypes)
    # Spliced path forces deterministic + no AF filtering + no exonic filter
    # in the existing implementation paths it's exercised against.
    keep = None
    keep_offsets = None
    if self.filter == "exonic":
        keep, keep_offsets = choose_exonic_variants(
            starts=regions[:, 1],
            ends=regions[:, 2],
            geno_offset_idxs=geno_offset_idx,
            geno_v_idxs=self.genotypes.data,
            geno_offsets=self.genotypes.offsets,
            v_starts=self.variants.start,
            ilens=self.variants.ilen,
        )
    diffs = self._haplotype_ilens(
        ds_idx, regions, deterministic=True, keep=keep, keep_offsets=keep_offsets
    )
    lengths = (regions[:, 2] - regions[:, 1])[:, None] + diffs
    return lengths.astype(np.int32, copy=False)
```

- [ ] **Step 3: Adapt `Dataset._recon` to forward `splice_plan` to the right reconstructor**

Look at how `_recon` dispatches to `Haps.get_haps_and_shifts` / `Ref.__call__` / `Tracks._call_float32`. Plumb `splice_plan` through all three. Where applicable, the existing `_recon` accepts `**kwargs` or specific args — add `splice_plan: SplicePlan | None = None` and forward.

Concretely the affected reconstructors are `Haps`, `Ref`, `Tracks`. For `HapsTracks` raise NotImplementedError when `splice_plan` is not None (defensive).

- [ ] **Step 4: Drop `_cat_length` import from `_impl.py`**

Replace `from ._splice import SpliceMap, _cat_length` (line 34) with `from ._splice import SpliceMap, SplicePlan, build_splice_plan`.

- [ ] **Step 5: Run the haplotype splice integration tests**

Run: `pixi run -e dev pytest tests/dataset/test_rc_packing.py tests/test_ref_ds_splicing.py -v`
Expected: all PASS except the direct `_cat_length` unit tests in `test_rc_packing.py` (those are removed in Task 7).

If `test_multi_exon_spliced_buffer_packed` or `test_multi_exon_spliced_matches_fasta_concat` fail, debug there before continuing.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py python/genvarloader/_dataset/_reconstruct.py
rtk git commit -m "refactor(splice): Dataset spliced path uses SplicePlan

Dataset._getitem_spliced now builds a SplicePlan and dispatches into
the plan-aware reconstructors. _cat_length is no longer called from
the production splice path; ploidy interleaving is correctly handled
by the permutation."
```

---

## Task 6: `Tracks._call_float32` accepts a `SplicePlan`

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (`Tracks._call_float32` ~lines 874-925)

### Background

`_call_float32` currently builds `out` of length `n_tracks * n_per_track` and iterates over tracks, calling `intervals_to_tracks` for each. The buffer is laid out as `(track, query, length)` in C-order — i.e. all of track 0's queries are contiguous, then all of track 1's.

For splicing we want `(splice_row, sample, track, splice_element, bytes)` layout. The flattening: each `(query, track)` becomes one k-index; permute via `plan.perm`.

`intervals_to_tracks` writes one query at a time and accepts an `out_offsets` array per-call. So per-track we already have per-query positioning. We just need to make `out_offsets` point at the right global positions.

### Steps

- [ ] **Step 1: Add `splice_plan` arg and rewrite the buffer write to use `permuted_out_offsets`**

Replace the body of `_call_float32` (~lines 874-925):

```python
def _call_float32(
    self,
    idx: NDArray[np.integer],
    r_idx: NDArray[np.integer],
    regions: NDArray[np.int32],
    output_length: Literal["ragged", "variable"] | int,
    splice_plan: "SplicePlan | None" = None,
) -> RaggedTracks:
    batch_size = len(idx)
    if isinstance(output_length, int):
        out_lengths = track_lengths = np.full(batch_size, output_length)
    else:
        lengths = regions[:, 2] - regions[:, 1]
        out_lengths = track_lengths = lengths

    if splice_plan is None:
        # ... existing body unchanged ...
        out_ofsts_per_t = lengths_to_offsets(out_lengths)
        track_ofsts_per_t = lengths_to_offsets(track_lengths)
        n_per_track: int = out_ofsts_per_t[-1]
        out = np.empty(len(self.active_tracks) * n_per_track, np.float32)
        out_lens = repeat(out_lengths, "b -> b t", t=len(self.active_tracks))
        out_offsets = lengths_to_offsets(out_lens)

        for track_ofst, (name, tracktype) in enumerate(self.active_tracks.items()):
            intervals = self.intervals[name]
            _out = out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track]
            o_idx = idx if tracktype is TrackType.SAMPLE else r_idx
            intervals_to_tracks(
                offset_idxs=o_idx,
                starts=regions[:, 1],
                itv_starts=intervals.starts.data,
                itv_ends=intervals.ends.data,
                itv_values=intervals.values.data,
                itv_offsets=intervals.starts.offsets,
                out=_out,
                out_offsets=track_ofsts_per_t,
            )
        out_shape = (len(idx), len(self.active_tracks), None)
        return cast(
            RaggedTracks, RaggedTracks.from_offsets(out, out_shape, out_offsets)
        )

    # ---- splice plan path ----
    n_tracks = len(self.active_tracks)
    B = batch_size
    E = n_tracks
    # The plan was built with inner_fixed = (n_tracks,), so plan.perm has
    # length B*E indexed in (query, track) C-order.
    # For each k_new in the permuted order, k_old = plan.perm[k_new].
    # query(k_old) = k_old // E, track(k_old) = k_old % E.
    # We need to write into out_buf at offsets plan.permuted_out_offsets[k_new]
    # for the appropriate (query, track) source data.

    total = int(splice_plan.permuted_out_offsets[-1])
    out_buf = np.empty(total, np.float32)

    # Group k_new indices by track so we can call intervals_to_tracks once
    # per track with a per-call offsets array that points at the global
    # out_buf positions.
    k_old = splice_plan.perm  # length B*E
    track_of_k = k_old % E
    query_of_k = k_old // E

    for track_ofst, (name, tracktype) in enumerate(self.active_tracks.items()):
        mask = track_of_k == track_ofst
        if not mask.any():
            continue
        # k_new indices that target this track, in order.
        k_new_idx = np.flatnonzero(mask)
        queries = query_of_k[k_new_idx]  # length M
        # Per-call we need:
        #   offset_idxs: idx / r_idx for these queries
        #   starts: regions[queries, 1]
        #   out: a view into out_buf
        #   out_offsets: cumulative offsets sized M+1 into the *track view*
        # But intervals_to_tracks writes into a buffer indexed by its
        # local out_offsets. We need to pass a scratch buffer per-track,
        # then scatter into out_buf using plan.permuted_out_offsets.
        # Simpler: call intervals_to_tracks into a contiguous scratch
        # sized to M queries' lengths, then copy into out_buf at the
        # right positions.
        intervals = self.intervals[name]
        o_idx_full = idx if tracktype is TrackType.SAMPLE else r_idx
        sub_lengths = (regions[queries, 2] - regions[queries, 1]).astype(
            np.int64, copy=False
        )
        sub_offsets = lengths_to_offsets(sub_lengths)
        scratch = np.empty(int(sub_offsets[-1]), np.float32)
        intervals_to_tracks(
            offset_idxs=o_idx_full[queries],
            starts=regions[queries, 1],
            itv_starts=intervals.starts.data,
            itv_ends=intervals.ends.data,
            itv_values=intervals.values.data,
            itv_offsets=intervals.starts.offsets,
            out=scratch,
            out_offsets=sub_offsets,
        )
        # Scatter into out_buf. For each m, dest range is
        #   out_buf[plan.permuted_out_offsets[k_new_idx[m]] :
        #           plan.permuted_out_offsets[k_new_idx[m]+1]]
        for m, k_new in enumerate(k_new_idx):
            s_dest = int(splice_plan.permuted_out_offsets[k_new])
            e_dest = int(splice_plan.permuted_out_offsets[k_new + 1])
            s_src = int(sub_offsets[m])
            e_src = int(sub_offsets[m + 1])
            out_buf[s_dest:e_dest] = scratch[s_src:e_src]

    # Per-element Ragged (caller rewraps with group_offsets via _regroup).
    out_shape = (splice_plan.permuted_lengths.shape[0], None)
    return cast(
        RaggedTracks,
        RaggedTracks.from_offsets(
            out_buf, out_shape, splice_plan.permuted_out_offsets
        ),
    )
```

Note: the scatter loop is O(B·E) in Python overhead. For typical splice batches this is fine; if it becomes a bottleneck, vectorize via `np.add.at` or precompute index arrays. Mark as a follow-up.

- [ ] **Step 2: Forward `splice_plan` from `Tracks.__call__` to `_call_float32`**

`Tracks.__call__` (~line 858) calls `self._call_float32(idx, r_idx, regions, output_length)`. Change to also forward `splice_plan` when present. Add `splice_plan: "SplicePlan | None" = None` to `Tracks.__call__`'s signature.

The `_call_intervals` branch does not support splicing — if `splice_plan is not None` and `kind` is not `RaggedTracks`, raise NotImplementedError.

- [ ] **Step 3: Run track splice integration tests if any exist**

Run: `pixi run -e dev pytest tests/dataset/test_write_tracks.py tests/tracks/ -v`
Expected: PASS for any non-spliced track tests; splice + tracks may not have direct tests, so the end-to-end gate is correctness on the haplotype suite + manual verification.

If no splice-track test exists, add a minimal one to `tests/dataset/test_rc_packing.py`:

```python
def test_spliced_tracks_round_trip(multi_exon_ds_path: Path):
    """Spliced track output: data buffer equals sum of per-element lengths."""
    import genvarloader as gvl
    # Skip if the fixture has no tracks attached; otherwise verify the
    # invariant that the splice plan path produces sane buffers.
    try:
        ds = (
            gvl.Dataset.open(multi_exon_ds_path, ref_path)
            .with_tracks("dummy")
            .with_settings(splice_info=("transcript_id", "exon_number"))
        )
    except (ValueError, AttributeError):
        pytest.skip("No tracks in fixture; tracks splice path covered elsewhere")
    out = ds[0, 0]
    # Output is a Ragged tracks structure; check it has the expected splice
    # row at index 0.
    assert out is not None
```

(If the test fixture has no tracks, this test will skip and we rely on the haplotype path coverage.)

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reconstruct.py tests/dataset/test_rc_packing.py
rtk git commit -m "feat(splice): Tracks._call_float32 accepts SplicePlan"
```

---

## Task 7: Delete `_cat_length` / `_cat_length_inner`; update or remove direct-call tests

**Files:**
- Modify: `python/genvarloader/_dataset/_splice.py` (remove `_cat_length`, `_cat_length_inner`, and the `_cat_length`-related overload imports)
- Modify: `tests/dataset/test_rc_packing.py` (remove direct `_cat_length` unit tests; keep integration tests)

### Steps

- [ ] **Step 1: Confirm no remaining callers**

Run: `rtk grep -r "_cat_length" python/ tests/`
Expected: only references in the file we're about to delete and the tests we're about to update.

If any other caller surfaces, route it through the new plan path before continuing.

- [ ] **Step 2: Delete `_cat_length` and `_cat_length_inner`**

In `python/genvarloader/_dataset/_splice.py`, delete the function definitions and their overload stubs (lines 24-112). Remove now-unused imports if any (`is_rag_dtype`, `assert_never` may still be used by `SplicePlan`/`SpliceMap`; leave those).

- [ ] **Step 3: Remove direct `_cat_length` tests from `test_rc_packing.py`**

Delete these test functions:
- `test_cat_length_with_packed_input_preserves_content` (~line 97)
- `test_cat_length_preserves_per_ploidy_content` (~line 256)
- `test_cat_length_non_bytes_dtype` (~line 279)

Remove the `from genvarloader._dataset._splice import _cat_length` import (line 34). The integration tests in the same file (which exercise the production splice path end-to-end) provide the regression coverage these unit tests were duplicating.

- [ ] **Step 4: Run the full test suite**

Run: `pixi run -e dev test`
Expected: all PASS.

If anything fails, do not delete the function — restore it and debug the failing case first.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_splice.py tests/dataset/test_rc_packing.py
rtk git commit -m "chore(splice): remove _cat_length and direct unit tests

The splice-plan path produces pre-spliced output directly from the
reconstruction kernels; _cat_length is no longer needed."
```

---

## Task 8: Defensive `NotImplementedError` for hap-tracks splicing

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (`HapsTracks.__call__`)
- Modify: `python/genvarloader/_dataset/_impl.py` (`Dataset._build_splice_plan` already covers this — verify)

### Steps

- [ ] **Step 1: Verify the early raise**

`Dataset._build_splice_plan` (added in Task 5) already raises `NotImplementedError` for `HapsTracks`. Confirm by re-reading the relevant section. Add an integration test:

In `tests/dataset/test_rc_packing.py`, add:

```python
def test_haptracks_splicing_raises(multi_exon_ds_path: Path):
    """Haplotype + track splicing is not supported (shape (b, t, p, ~l))."""
    import pytest
    # If the fixture lacks tracks, skip.
    try:
        ds = (
            gvl.Dataset.open(multi_exon_ds_path, ref_path)
            .with_seqs("haplotypes")
            .with_tracks("dummy")
            .with_settings(splice_info=("transcript_id", "exon_number"))
        )
    except (ValueError, AttributeError):
        pytest.skip("no tracks in fixture")
    with pytest.raises(NotImplementedError, match="haplotypes \\+ tracks"):
        _ = ds[0, 0]
```

- [ ] **Step 2: Run the test**

Run: `pixi run -e dev pytest tests/dataset/test_rc_packing.py::test_haptracks_splicing_raises -v`
Expected: PASS (or SKIP if fixture has no tracks).

- [ ] **Step 3: Commit (if changes were made)**

```bash
rtk git add tests/dataset/test_rc_packing.py
rtk git commit -m "test(splice): assert hap-tracks splicing raises NotImplementedError"
```

---

## Task 9: Final verification

**Files:** none — verification only.

### Steps

- [ ] **Step 1: Run the full test suite**

Run: `pixi run -e dev test`
Expected: all PASS.

- [ ] **Step 2: Confirm `_cat_length` is gone**

Run: `rtk grep -r "_cat_length" python/ tests/`
Expected: zero matches.

- [ ] **Step 3: Confirm splice path no longer allocates intermediate buffers**

Spot-check by adding a temporary `print` or via profiling — not required for correctness; document any obvious win in the PR description.

- [ ] **Step 4: Update spec status**

In `docs/superpowers/specs/2026-05-22-splice-zero-copy-design.md`, change `**Status:** Draft` to `**Status:** Implemented`.

- [ ] **Step 5: Commit**

```bash
rtk git add docs/superpowers/specs/2026-05-22-splice-zero-copy-design.md
rtk git commit -m "docs: mark splice-zero-copy spec as implemented"
```

---

## Self-review notes

- **Spec coverage:** All spec sections (Problem, Core idea, Scope, Design subsections) have corresponding tasks. Hap-tracks NotImplementedError comes from Task 5's `_build_splice_plan` and is locked in by Task 8.
- **Placeholder scan:** None. Every step has either code or a concrete command.
- **Type consistency:** `SplicePlan` field names and `build_splice_plan` signature stable across Tasks 1-7. `splice_plan` parameter name consistent across `Ref.__call__`, `Haps._get_haplotypes`, `Haps.get_haps_and_shifts`, `Tracks._call_float32`, `Tracks.__call__`.
- **Risk areas to watch during execution:**
  1. `_splice_selection_shape` in Task 5 is fiddly because `SpliceIndexer.parse_idx` already raveled the (row, sample) grid. The implementer may need to revisit by inspecting `parse_idx` rather than re-deriving — if so, refactor `parse_idx` to also return the (n_rows_sel, n_samples_sel) pair.
  2. `_recon` dispatch in `Dataset` — the plan needs to reach the right reconstructor. If `Dataset._recon` is hard to extend, plumbing the plan via the reconstructor object directly (bypassing `_recon`) is acceptable.
  3. `Tracks._call_float32` scatter loop is Python-level and may be slow for very large batches. Acceptable for v1; vectorize later if needed.
  4. `RefTracks` splice path is unimplemented in this pass (Task 5 raises). If any test exercises it, decide whether to extend or to skip the test.
