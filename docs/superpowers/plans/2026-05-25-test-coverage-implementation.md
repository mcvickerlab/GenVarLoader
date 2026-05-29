# Test Coverage Initiative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cover risky numba kernels via behavior tests, raise pure-Python coverage on user-facing readers + the `Dataset.with_*` / `_open` / `_indexing` API surface, and configure coverage measurement to exclude numba-shadowed lines.

**Architecture:** Three waves, one bundled PR. Wave 1 extends existing kernel test files in `tests/unit/dataset/genotypes/` and adds an intervals kernel file. Wave 2 extends `tests/unit/test_fasta.py` and `tests/unit/test_bigwig.py`, and creates `tests/unit/test_torch.py`. Wave 3 adds `tests/dataset/test_with_methods.py` and extends `tests/unit/dataset/test_indexing.py`, plus `.coveragerc` updates. No new fixtures; reuse `tests/conftest.py` paths.

**Tech Stack:** pytest, pytest-cases, numpy, numba (`@nb.njit(cache=True)`), torch (optional), pixi env `dev`.

**Spec:** `docs/superpowers/specs/2026-05-25-test-coverage-design.md`

---

## Conventions used throughout

- Run all tests via `pixi run -e dev pytest <path> -v`.
- Coverage runs via `pixi run -e dev pytest --cov=python/genvarloader --cov-report=term-missing`.
- Commit message style: conventional commits (`test:`, `test(scope):`, `chore(cov):`). Per the project memory, this is a solo-maintainer initiative: one bundled PR at the end, but commit-per-task inside.
- Always use `rtk git ...` for git commands (per CLAUDE.md).
- After each task: `pixi run -e dev pytest tests/ -x` (full suite) to confirm no regressions.
- Each task ends with a commit step. Do not skip.

---

## Wave 1 — Numba kernel correctness tests

### Task 1: Extend `reconstruct_haplotype_from_sparse` case matrix

**Files:**
- Modify: `tests/unit/dataset/genotypes/test_reconstruct.py`

Existing file already uses `pytest_cases` with `case_snps`, `case_indels`, `case_spanning_del_pad`, `case_shift_ins`. Add cases that fill the gaps from the spec: deletion spanning region end, overlapping variants (first ALT wins), shift exactly at variant boundary, shift exceeds region length, reference-only (no variants).

- [ ] **Step 1: Add `case_ref_only`**

Append to `tests/unit/dataset/genotypes/test_reconstruct.py` before the `@parametrize_with_cases` decorator:

```python
def case_ref_only():
    """No variants applied — output is pure reference slice."""
    v_starts = np.array([], np.int32)
    ilens = np.array([], np.int32)
    genos = np.zeros((1, 1, 0), dtype=np.int8)  # (s p v)
    var_idxs = np.array([], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"", dtype=np.uint8)
    alt_offsets = np.array([0], dtype=np.uintp)
    ref = np.frombuffer(b"ACGT", dtype=np.uint8)
    ref_start = 0

    desired = np.frombuffer(b"ACGT", dtype="S1")
    annot_v_idxs = np.array([-1, -1, -1, -1], dtype=np.int32)
    annot_pos = np.array([0, 1, 2, 3], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        alt_alleles,
        alt_offsets,
        ref,
        sparse_genos,
        ref_start,
        desired,
        annot_v_idxs,
        annot_pos,
    )
```

- [ ] **Step 2: Add `case_spanning_del_end`**

Mirror of existing `case_spanning_del_pad` but the deletion straddles the END of the requested region rather than the start. Use `ref = b"ACGTA"`, variant at position 3 with `ilen=-2` (deletes "TA"), `ref_start=0`, length=4. Expected: `b"ACGG"` (variant truncated at region end). Compute and verify by hand before writing — the kernel comments say deletions ending after region are clipped via `v_ilen += max(0, v_end - q_ends[query])`. The reconstruction kernel here only takes `out` length, not q_ends; verify behavior with `out` of length 4.

```python
def case_spanning_del_end():
    """Deletion runs past region end — output truncates at out length."""
    v_starts = np.array([2], np.int32)
    ilens = np.array([-2], dtype=np.int32)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"G", dtype=np.uint8)
    alt_offsets = np.array([0, 1], np.uintp)
    ref = np.frombuffer(b"ACGTA", dtype=np.uint8)
    ref_start = 0

    # ACGTA with DEL at pos 2 (atomized: replaces "GTA" with "G") -> "ACG"
    # out length = 4 (len(ref) - ref_start = 5; but kernel uses len(out))
    # NB: test driver below sets out length = len(ref) - ref_start = 5
    desired = np.frombuffer(b"ACGNN", dtype="S1")
    annot_v_idxs = np.array([-1, -1, 0, -1, -1], dtype=np.int32)
    annot_pos = np.array(
        [0, 1, 2, np.iinfo(np.int32).max, np.iinfo(np.int32).max], dtype=np.int32
    )

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        alt_alleles,
        alt_offsets,
        ref,
        sparse_genos,
        ref_start,
        desired,
        annot_v_idxs,
        annot_pos,
    )
```

- [ ] **Step 3: Add `case_overlapping_variants`**

Two variants at the same position; the kernel must keep only the first (mirrors bcftools consensus behavior). Use SNVs at the same pos 1.

```python
def case_overlapping_variants():
    """Two variants at same position — first ALT wins, second skipped."""
    v_starts = np.array([1, 1], np.int32)
    ilens = np.zeros(2, dtype=np.int32)
    genos = np.array([[[0, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=np.int32)

    shift = 0
    alt_alleles = np.frombuffer(b"T" + b"G", dtype=np.uint8)
    alt_offsets = np.array([0, 1, 2], dtype=np.uintp)
    ref = np.frombuffer(b"ACGG", dtype=np.uint8)
    ref_start = 0

    # First ALT at pos 1 is "T"; second (G) ignored.
    desired = np.frombuffer(b"ATGG", dtype="S1")
    annot_v_idxs = np.array([-1, 0, -1, -1], dtype=np.int32)
    annot_pos = np.array([0, 1, 2, 3], dtype=np.int32)

    sparse_genos = dense2sparse(genos, var_idxs)

    return (
        v_starts,
        ilens,
        shift,
        alt_alleles,
        alt_offsets,
        ref,
        sparse_genos,
        ref_start,
        desired,
        annot_v_idxs,
        annot_pos,
    )
```

- [ ] **Step 4: Run tests**

```
pixi run -e dev pytest tests/unit/dataset/genotypes/test_reconstruct.py -v
```

Expected: all cases PASS. If any new case fails, investigate whether the expected output is wrong (re-derive by hand) or whether the kernel has a real bug — surface the latter to the user; do NOT change the kernel without approval.

- [ ] **Step 5: Commit**

```
rtk git add tests/unit/dataset/genotypes/test_reconstruct.py
rtk git commit -m "test(kernels): extend reconstruct_haplotype case matrix"
```

---

### Task 2: New test file for `get_diffs_sparse`

**Files:**
- Create: `tests/unit/dataset/genotypes/test_get_diffs.py`

`get_diffs_sparse` computes per-haplotype length delta vs reference. It has two distinct code paths: the "fast path" with only `keep`/`keep_offsets` (sums `ilens` over kept variants) and the "slow path" with `q_starts`/`q_ends`/`v_starts` (handles spanning deletions).

- [ ] **Step 1: Write `test_get_diffs_fast_path`**

```python
import numpy as np
from genvarloader._dataset._genotypes import get_diffs_sparse


def test_get_diffs_fast_path_no_variants():
    """Empty variant set -> diff is 0."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([], dtype=np.int32)
    geno_offsets = np.array([0, 0], dtype=np.int64)
    ilens = np.array([], dtype=np.int32)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
    )

    np.testing.assert_equal(diffs, np.array([[0]], dtype=np.int32))


def test_get_diffs_fast_path_sum():
    """Without spanning info, diff is sum of ilens for selected variants."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0, 1, 2], dtype=np.int32)
    geno_offsets = np.array([0, 3], dtype=np.int64)
    ilens = np.array([1, -2, 3], dtype=np.int32)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
    )

    np.testing.assert_equal(diffs, np.array([[2]], dtype=np.int32))


def test_get_diffs_fast_path_with_keep():
    """`keep` mask selects subset of variants."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0, 1, 2], dtype=np.int32)
    geno_offsets = np.array([0, 3], dtype=np.int64)
    ilens = np.array([1, -2, 3], dtype=np.int32)
    keep = np.array([True, False, True], dtype=np.bool_)
    keep_offsets = np.array([0, 3], dtype=np.int64)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
        keep=keep,
        keep_offsets=keep_offsets,
    )

    np.testing.assert_equal(diffs, np.array([[4]], dtype=np.int32))
```

- [ ] **Step 2: Write `test_get_diffs_spanning_del`**

The slow path activates when `q_starts`, `q_ends`, `v_starts` are all provided. Spanning deletions at start/end clip via the kernel's `max(0, ...)` arithmetic.

```python
def test_get_diffs_spanning_del_clipped_at_start():
    """Deletion starts before region — only the in-region portion counts."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0], dtype=np.int32)
    geno_offsets = np.array([0, 1], dtype=np.int64)
    v_starts = np.array([0], dtype=np.int32)
    ilens = np.array([-3], dtype=np.int32)  # deletes 3bp starting at pos 0
    q_starts = np.array([2], dtype=np.int32)  # region starts inside the deletion
    q_ends = np.array([10], dtype=np.int32)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
        q_starts=q_starts,
        q_ends=q_ends,
        v_starts=v_starts,
    )

    # Atomized DEL: v_end = 0 - min(0, -3) + 1 = 4
    # v_ilen += max(0, q_starts - v_start - 1) = max(0, 2 - 0 - 1) = 1 -> -2
    # v_ilen += max(0, v_end - q_ends) = max(0, 4 - 10) = 0
    # final -> -2
    np.testing.assert_equal(diffs, np.array([[-2]], dtype=np.int32))


def test_get_diffs_variant_outside_region_skipped():
    """Variants outside [q_start, q_end) do not contribute."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_v_idxs = np.array([0, 1], dtype=np.int32)
    geno_offsets = np.array([0, 2], dtype=np.int64)
    v_starts = np.array([0, 20], dtype=np.int32)
    ilens = np.array([2, 5], dtype=np.int32)  # second variant past q_end
    q_starts = np.array([0], dtype=np.int32)
    q_ends = np.array([10], dtype=np.int32)

    diffs = get_diffs_sparse(
        geno_offset_idx=geno_offset_idx,
        geno_v_idxs=geno_v_idxs,
        geno_offsets=geno_offsets,
        ilens=ilens,
        q_starts=q_starts,
        q_ends=q_ends,
        v_starts=v_starts,
    )

    np.testing.assert_equal(diffs, np.array([[2]], dtype=np.int32))
```

- [ ] **Step 3: Run**

```
pixi run -e dev pytest tests/unit/dataset/genotypes/test_get_diffs.py -v
```

If a case fails, re-derive by hand using the comments in `_genotypes.py:99-101`. Do not modify the kernel.

- [ ] **Step 4: Commit**

```
rtk git add tests/unit/dataset/genotypes/test_get_diffs.py
rtk git commit -m "test(kernels): add get_diffs_sparse behavior tests"
```

---

### Task 3: New test file for `filter_af`

**Files:**
- Create: `tests/unit/dataset/genotypes/test_filter_af.py`

- [ ] **Step 1: Write test file**

```python
import numpy as np
from genvarloader._dataset._genotypes import filter_af


def _basic_inputs():
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_offsets = np.array([0, 4], dtype=np.int64)
    geno_v_idxs = np.array([0, 1, 2, 3], dtype=np.int32)
    afs = np.array([0.001, 0.05, 0.2, 0.5], dtype=np.float32)
    return geno_offset_idx, geno_offsets, geno_v_idxs, afs


def test_filter_af_no_op():
    """min_af=None, max_af=None -> all kept, short-circuits."""
    geno_offset_idx, geno_offsets, geno_v_idxs, afs = _basic_inputs()
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, None, None)
    np.testing.assert_equal(keep, np.array([True, True, True, True]))


def test_filter_af_min_only():
    geno_offset_idx, geno_offsets, geno_v_idxs, afs = _basic_inputs()
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, 0.05, None)
    np.testing.assert_equal(keep, np.array([False, True, True, True]))


def test_filter_af_max_only():
    geno_offset_idx, geno_offsets, geno_v_idxs, afs = _basic_inputs()
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, None, 0.2)
    np.testing.assert_equal(keep, np.array([True, True, True, False]))


def test_filter_af_both():
    geno_offset_idx, geno_offsets, geno_v_idxs, afs = _basic_inputs()
    keep, _ = filter_af(geno_offset_idx, geno_offsets, geno_v_idxs, afs, 0.01, 0.3)
    np.testing.assert_equal(keep, np.array([False, True, True, False]))


def test_filter_af_2d_offsets_layout():
    """(2, n_slices) offsets layout — used when geno_offsets came from slicing."""
    geno_offset_idx = np.array([[0]], dtype=np.intp)
    geno_offsets = np.array([[0], [4]], dtype=np.int64)  # (2, n_slices)
    geno_v_idxs = np.array([0, 1, 2, 3], dtype=np.int32)
    afs = np.array([0.001, 0.05, 0.2, 0.5], dtype=np.float32)
    keep, keep_offsets = filter_af(
        geno_offset_idx, geno_offsets, geno_v_idxs, afs, 0.05, None
    )
    np.testing.assert_equal(keep, np.array([False, True, True, True]))
    # keep_offsets length = n_slices + 1 in this layout
    assert keep_offsets.shape == (2,)
```

- [ ] **Step 2: Run and commit**

```
pixi run -e dev pytest tests/unit/dataset/genotypes/test_filter_af.py -v
rtk git add tests/unit/dataset/genotypes/test_filter_af.py
rtk git commit -m "test(kernels): add filter_af coverage across input layouts"
```

---

### Task 4: Extend `choose_exonic_variants` matrix

**Files:**
- Modify: `tests/unit/dataset/genotypes/test_choose_exonic_variants.py`

- [ ] **Step 1: Read current cases**

```
rtk read tests/unit/dataset/genotypes/test_choose_exonic_variants.py
```

Identify which of these scenarios are already covered: fully inside, spans start, spans end, entirely outside (both before and after region).

- [ ] **Step 2: Add missing cases**

For each scenario not already present, add a parametrized case following the existing file's style. The kernel logic (`_choose_exonic_variants` in `_genotypes.py:502-525`) is: keep variant iff `v_pos >= query_start AND v_ref_end <= query_end`. Each new case should be a single variant with a hand-computed expected `keep` value.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/unit/dataset/genotypes/test_choose_exonic_variants.py -v
rtk git add tests/unit/dataset/genotypes/test_choose_exonic_variants.py
rtk git commit -m "test(kernels): fill choose_exonic_variants scenario gaps"
```

---

### Task 5: New test file for `intervals_to_tracks`

**Files:**
- Create: `tests/unit/dataset/test_intervals_kernel.py`

`intervals_to_tracks` writes piecewise-constant signal into a flat output buffer per query. Insertion-fill interactions belong in `tests/unit/tracks/`, not here.

- [ ] **Step 1: Write test file**

```python
import numpy as np
from genvarloader._dataset._intervals import intervals_to_tracks


def test_intervals_to_tracks_empty():
    """No intervals -> output is zeros."""
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array([0], dtype=np.int32)
    itv_starts = np.array([], dtype=np.int32)
    itv_ends = np.array([], dtype=np.int32)
    itv_values = np.array([], dtype=np.float32)
    itv_offsets = np.array([0, 0], dtype=np.int64)
    out = np.empty(5, dtype=np.float32)
    out_offsets = np.array([0, 5], dtype=np.int64)

    intervals_to_tracks(
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )
    np.testing.assert_equal(out, np.zeros(5, dtype=np.float32))


def test_intervals_to_tracks_single_interval():
    """One interval covers part of the output."""
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array([0], dtype=np.int32)
    itv_starts = np.array([1], dtype=np.int32)
    itv_ends = np.array([4], dtype=np.int32)
    itv_values = np.array([2.5], dtype=np.float32)
    itv_offsets = np.array([0, 1], dtype=np.int64)
    out = np.empty(5, dtype=np.float32)
    out_offsets = np.array([0, 5], dtype=np.int64)

    intervals_to_tracks(
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )
    np.testing.assert_equal(out, np.array([0.0, 2.5, 2.5, 2.5, 0.0], dtype=np.float32))


def test_intervals_to_tracks_multiple_non_overlapping():
    """Two non-overlapping intervals."""
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array([0], dtype=np.int32)
    itv_starts = np.array([0, 3], dtype=np.int32)
    itv_ends = np.array([2, 5], dtype=np.int32)
    itv_values = np.array([1.0, 3.0], dtype=np.float32)
    itv_offsets = np.array([0, 2], dtype=np.int64)
    out = np.empty(5, dtype=np.float32)
    out_offsets = np.array([0, 5], dtype=np.int64)

    intervals_to_tracks(
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )
    np.testing.assert_equal(out, np.array([1.0, 1.0, 0.0, 3.0, 3.0], dtype=np.float32))


def test_intervals_to_tracks_offset_query_start():
    """Query starts at non-zero — intervals are in absolute coords."""
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array([10], dtype=np.int32)
    itv_starts = np.array([11], dtype=np.int32)
    itv_ends = np.array([13], dtype=np.int32)
    itv_values = np.array([7.0], dtype=np.float32)
    itv_offsets = np.array([0, 1], dtype=np.int64)
    out = np.empty(4, dtype=np.float32)
    out_offsets = np.array([0, 4], dtype=np.int64)

    intervals_to_tracks(
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )
    np.testing.assert_equal(out, np.array([0.0, 7.0, 7.0, 0.0], dtype=np.float32))
```

- [ ] **Step 2: Run and commit**

```
pixi run -e dev pytest tests/unit/dataset/test_intervals_kernel.py -v
rtk git add tests/unit/dataset/test_intervals_kernel.py
rtk git commit -m "test(kernels): add intervals_to_tracks behavior tests"
```

---

## Wave 2 — User-facing readers + DataLoader

### Task 6: Extend `test_fasta.py`

**Files:**
- Modify: `tests/unit/test_fasta.py`

Current file covers padding behavior. Add: missing contig, contig-name normalization (chr1 vs 1), `Reader` protocol attributes round-trip.

- [ ] **Step 1: Discover what contig normalization Fasta does**

```
rtk search Fasta python/genvarloader/_fasta.py
```

Identify the contig-name normalization path (`_norm_contig` or similar). If the implementation does NOT normalize, the test should assert the strict behavior (missing contig raises). Do not invent behavior.

- [ ] **Step 2: Add tests**

Append to `tests/unit/test_fasta.py`:

```python
def test_fasta_missing_contig_raises(ref_fasta):
    fasta = Fasta("ref", ref_fasta)
    with pytest.raises(Exception):  # narrow once exception type is known
        fasta.read("nonexistent_contig_zzz", 0, 100)


def test_fasta_reader_protocol_attrs(ref_fasta):
    """The Reader protocol requires name, dtype, contigs."""
    fasta = Fasta("ref", ref_fasta, pad="N")
    assert fasta.name == "ref"
    assert fasta.dtype == np.dtype("S1")
    assert "chr1" in fasta.contigs


def test_fasta_zero_length_range(ref_fasta):
    """start == end -> empty result."""
    fasta = Fasta("ref", ref_fasta, pad="N")
    seq = fasta.read("chr1", 100, 100)
    assert len(seq) == 0
```

Refine `pytest.raises(Exception)` to the actual exception class once `_fasta.py` is read (likely a `KeyError` or custom error). If the read happens to succeed (e.g. silent zero-fill), update the test to assert the actual contract.

- [ ] **Step 3: Run**

```
pixi run -e dev pytest tests/unit/test_fasta.py -v
```

If `test_fasta_missing_contig_raises` does NOT raise, that's a code-vs-spec mismatch — surface to user.

- [ ] **Step 4: Commit**

```
rtk git add tests/unit/test_fasta.py
rtk git commit -m "test(fasta): cover missing contig, protocol attrs, zero-length"
```

---

### Task 7: Create/extend `test_bigwig.py`

**Files:**
- Find or create: `tests/unit/test_bigwig.py` (check first; the Rust-side test lives at `tests/test_bigwig.rs`)

- [ ] **Step 1: Discover state**

```
ls tests/unit/test_bigwig.py 2>/dev/null || echo "does not exist"
rtk search BigWigs python/genvarloader/_bigwig.py
```

Identify the public read API on `BigWigs`. Confirm the `bigwig_dir` fixture in `conftest.py` points to usable test data:

```
ls tests/data/bigwig/
```

- [ ] **Step 2: Write test file (create or extend)**

```python
import numpy as np
import pytest
from genvarloader import BigWigs


def test_bigwigs_read_basic(bigwig_dir):
    """Smoke test: read a small range, get expected shape/dtype."""
    bws = BigWigs.from_paths(name="bw", paths=list(bigwig_dir.glob("*.bw")))
    out = bws.read("chr1", 0, 100)
    assert out.dtype == np.float32
    # shape depends on number of bigwigs; just assert last dim matches length
    assert out.shape[-1] == 100


def test_bigwigs_reader_protocol_attrs(bigwig_dir):
    bws = BigWigs.from_paths(name="bw", paths=list(bigwig_dir.glob("*.bw")))
    assert bws.name == "bw"
    assert bws.dtype == np.float32


def test_bigwigs_missing_path_raises():
    with pytest.raises(Exception):
        BigWigs.from_paths(name="bw", paths=["/does/not/exist.bw"])
```

Refine signatures (`from_paths` vs `from_table` vs constructor) by reading `_bigwig.py` first. The test asserts the contract that exists; do not invent.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/unit/test_bigwig.py -v
rtk git add tests/unit/test_bigwig.py
rtk git commit -m "test(bigwig): cover read smoke path, protocol attrs, missing file"
```

---

### Task 8: New `test_torch.py`

**Files:**
- Create: `tests/unit/test_torch.py`

`_torch.py` exposes `get_dataloader`, `get_sampler`, and torch-compatible wrappers. Skip cleanly if torch is not installed.

- [ ] **Step 1: Write test file**

```python
import pytest
import numpy as np
import genvarloader as gvl

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def small_dataset(phased_vcf_gvl, reference):
    return gvl.Dataset.open(phased_vcf_gvl, reference=reference)


def test_get_dataloader_smoke(small_dataset):
    """DataLoader yields batches with expected length."""
    from genvarloader._torch import get_dataloader

    torch_ds = small_dataset.to_dataset()  # whatever the supported conversion is
    dl = get_dataloader(torch_ds, batch_size=2, shuffle=False, num_workers=0)
    batches = list(dl)
    assert len(batches) >= 1


def test_get_sampler_shuffle_deterministic(small_dataset):
    """Seeded RandomSampler produces deterministic order."""
    from genvarloader._torch import get_sampler

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    s1 = get_sampler(len(small_dataset), batch_size=2, shuffle=True)
    s2 = get_sampler(len(small_dataset), batch_size=2, shuffle=True)
    # If the sampler accepts a generator, pass it; otherwise smoke-check non-empty
    assert list(s1) is not None  # placeholder until sampler API confirmed


def test_get_dataloader_num_workers_warning(small_dataset, caplog):
    """num_workers > 1 should log a warning."""
    from genvarloader._torch import get_dataloader

    torch_ds = small_dataset.to_dataset()
    get_dataloader(torch_ds, batch_size=1, num_workers=2)
    # Loguru-based warning; check the messages list via caplog or capsys as appropriate
```

The exact `to_dataset()` API and the sampler interface need to be confirmed by reading `_impl.py` and `_torch.py` before finalizing assertions. The implementer should:

1. Read `_torch.py` end-to-end.
2. Read the `Dataset.to_dataset` (or equivalent) method in `_impl.py`.
3. Rewrite the assertions to match the real API. Do NOT invent methods.

If torch's `td.Dataset` ABC requires `__len__` / `__getitem__`, the test must use a class that satisfies it — likely the `RaggedDataset` / `ArrayDataset` itself.

- [ ] **Step 2: Run**

```
pixi run -e dev pytest tests/unit/test_torch.py -v
```

If torch is missing in the dev env, the file should `pytest.importorskip` cleanly.

- [ ] **Step 3: Commit**

```
rtk git add tests/unit/test_torch.py
rtk git commit -m "test(torch): cover get_dataloader/get_sampler smoke paths"
```

---

## Wave 3 — API surface

### Task 9: New `test_with_methods.py`

**Files:**
- Create: `tests/dataset/test_with_methods.py`

The `Dataset` class exposes `with_settings`, `with_len`, `with_seqs`, `with_tracks`, `with_insertion_fill` as instance methods (plus typing overloads). Each returns a new lazy view (frozen dataclass — never mutates self).

- [ ] **Step 1: Enumerate `with_*` methods**

```
rtk grep "^    def with_" python/genvarloader/_dataset/_impl.py
```

Confirm the active set: `with_settings`, `with_len`, `with_seqs`, `with_tracks`, `with_insertion_fill`. If others appear (e.g. `with_jitter`, `with_indels`), add a test for each. The implementer must NOT skip any method actually present.

- [ ] **Step 2: Write the test file**

```python
import pytest
import genvarloader as gvl


@pytest.fixture(scope="module")
def base_ds(phased_vcf_gvl, reference):
    return gvl.Dataset.open(phased_vcf_gvl, reference=reference)


def test_with_settings_returns_new_view(base_ds):
    new_ds = base_ds.with_settings()  # no-op call with defaults
    # frozen dataclass: not the same instance
    assert new_ds is not base_ds


def test_with_len_returns_new_view(base_ds):
    original_len = base_ds.output_length  # confirm attribute name from _impl.py
    new_ds = base_ds.with_len(100)
    assert new_ds is not base_ds
    assert new_ds.output_length == 100
    # original is unchanged
    assert base_ds.output_length == original_len


def test_with_seqs_haplotypes(base_ds):
    new_ds = base_ds.with_seqs("haplotypes")
    assert new_ds is not base_ds


def test_with_seqs_invalid_raises(base_ds):
    with pytest.raises((ValueError, KeyError, TypeError)):
        base_ds.with_seqs("not_a_real_mode")


def test_with_tracks_none(base_ds):
    new_ds = base_ds.with_tracks(None)
    assert new_ds is not base_ds


def test_with_insertion_fill_modes(base_ds):
    """Each accepted fill mode produces a new view; invalid raises."""
    # Accepted modes need to be read from _insertion_fill.py before finalizing this list.
    for mode in ["mean", "left", "right"]:  # confirm against source
        try:
            new_ds = base_ds.with_insertion_fill(mode)
        except (ValueError, TypeError):
            pytest.skip(f"mode {mode!r} not supported in this build")
        else:
            assert new_ds is not base_ds

    with pytest.raises((ValueError, KeyError, TypeError)):
        base_ds.with_insertion_fill("not_a_real_mode")
```

Note: an existing `tests/unit/dataset/test_with_insertion_fill.py` already covers the rejection case. Do not duplicate it — link conceptually but the new file is about the `with_*` matrix as a whole.

- [ ] **Step 3: Confirm method names and attribute names**

Read each `with_*` definition in `_impl.py` once before finalizing the test. If an assertion in the draft references an attribute that doesn't exist (`output_length` is plausible but verify), correct it.

- [ ] **Step 4: Run and commit**

```
pixi run -e dev pytest tests/dataset/test_with_methods.py -v
rtk git add tests/dataset/test_with_methods.py
rtk git commit -m "test(dataset): cover with_* method matrix"
```

---

### Task 10: Extend `test_open` error paths

**Files:**
- Create or modify: `tests/dataset/test_open.py` (check if it exists; if not create)

- [ ] **Step 1: Check current state**

```
ls tests/dataset/test_open.py 2>/dev/null || echo "create new"
rtk read python/genvarloader/_dataset/_open.py
```

Identify the actual error conditions: missing directory, missing `metadata.json`, missing reference when intervals or haps need it, unknown variant source.

- [ ] **Step 2: Write tests**

```python
import pytest
import genvarloader as gvl


def test_open_missing_dir_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        gvl.Dataset.open(tmp_path / "does_not_exist")


def test_open_dir_without_metadata_raises(tmp_path):
    """Directory exists but lacks the required metadata.json."""
    (tmp_path / "empty.gvl").mkdir()
    with pytest.raises((FileNotFoundError, KeyError, ValueError)):
        gvl.Dataset.open(tmp_path / "empty.gvl")


def test_open_without_reference_when_required(phased_vcf_gvl):
    """Opening a haps dataset without a reference should be valid (lazy)
    but accessing haplotype sequences without one should raise.

    Verify the actual contract by reading _open.py — if Dataset.open requires
    a reference for haps datasets, assert that. Otherwise, test the deferred
    error point.
    """
    ds = gvl.Dataset.open(phased_vcf_gvl)
    # The behavior here depends on whether _open enforces ref-at-open or
    # ref-at-eager-access. Read _open.py and test the actual contract.
    assert ds is not None
```

The final assertions depend on the exact `_open.py` contract. Implementer reads the module first and writes tests that match the real behavior — do not assume.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/dataset/test_open.py -v
rtk git add tests/dataset/test_open.py
rtk git commit -m "test(open): cover error paths on Dataset.open"
```

---

### Task 11: Extend `test_indexing.py` edge cases

**Files:**
- Modify: `tests/unit/dataset/test_indexing.py`

- [ ] **Step 1: Read existing tests**

```
rtk read tests/unit/dataset/test_indexing.py
```

Identify gaps from spec: slice with step, negative indices, out-of-bounds, fancy boolean mask, empty selection.

- [ ] **Step 2: Add missing scenarios**

For each gap, add a parametrized or named test following the file's existing style. Each test:
- Builds (or fixtures) a small Dataset.
- Indexes it with the edge-case form.
- Asserts shape/contents OR asserts the exception class raised.

Example skeleton:

```python
def test_indexing_negative_region_index(opened_ds):
    """Negative indices wrap from the end."""
    last_via_neg = opened_ds[-1, 0]
    last_via_pos = opened_ds[len(opened_ds.regions) - 1, 0]
    # Compare via numpy-aware equality
    np.testing.assert_array_equal(last_via_neg, last_via_pos)


def test_indexing_out_of_bounds_raises(opened_ds):
    with pytest.raises(IndexError):
        _ = opened_ds[len(opened_ds.regions) + 100, 0]


def test_indexing_boolean_mask(opened_ds):
    n_regions = len(opened_ds.regions)
    mask = np.zeros(n_regions, dtype=bool)
    mask[0] = True
    result = opened_ds[mask, 0]
    # whatever the contract is — match it
```

Implementer reads `_indexing.py` first to confirm which forms are supported. Mark unsupported forms with explicit `pytest.raises` instead of skipping.

- [ ] **Step 3: Run and commit**

```
pixi run -e dev pytest tests/unit/dataset/test_indexing.py -v
rtk git add tests/unit/dataset/test_indexing.py
rtk git commit -m "test(indexing): cover negative, oob, boolean-mask, empty selections"
```

---

### Task 12: Coverage config + final sweep

**Files:**
- Modify or create: `.coveragerc` (or `pyproject.toml` `[tool.coverage.*]` if that's where config lives)

- [ ] **Step 1: Locate current coverage config**

```
ls .coveragerc 2>/dev/null
rtk grep "tool.coverage" pyproject.toml
```

Determine the config location.

- [ ] **Step 2: Add omit + exclude rules**

Add to the existing config (do not overwrite unrelated settings):

```ini
[run]
omit =
    python/genvarloader/_dataset/_intervals.py

[report]
exclude_lines =
    pragma: no cover
    @nb.njit
    @numba.njit
    raise ImportError\("PyTorch is not available
```

Notes:
- `_dataset/_intervals.py` is omitted because every function is `@nb.njit` and coverage.py reports false negatives. Behavior is verified by Task 5.
- The `@nb.njit` exclusion pattern marks numba-decorated functions as not-counted across mixed-Python modules (`_genotypes.py`, `_rag_variants.py`, etc.).

If config lives in `pyproject.toml`, translate the syntax:

```toml
[tool.coverage.run]
omit = ["python/genvarloader/_dataset/_intervals.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "@nb.njit",
    "@numba.njit",
    "raise ImportError\\(\"PyTorch is not available",
]
```

- [ ] **Step 3: Run coverage and compare to baseline**

```
pixi run -e dev pytest tests/ --cov=python/genvarloader --cov-report=term-missing | tee /tmp/cov-after.txt
```

Compare against `docs/superpowers/specs/2026-05-24-test-audit-coverage-baseline.txt`. Confirm:
- Pure-Python target modules (`_torch`, `_bigwig`, `_fasta`, `_open`, `_indexing`, `_impl` excluding repr) at ≥80%.
- No regressions on previously-covered modules.

- [ ] **Step 4: Save new baseline**

```
pixi run -e dev pytest tests/ --cov=python/genvarloader --cov-report=term \
  > docs/superpowers/specs/2026-05-25-test-coverage-after.txt
```

- [ ] **Step 5: Commit**

```
rtk git add .coveragerc pyproject.toml docs/superpowers/specs/2026-05-25-test-coverage-after.txt
rtk git commit -m "chore(cov): exclude numba modules from coverage gate"
```

---

### Task 13: Full-suite sanity + open PR

- [ ] **Step 1: Run the full test suite**

```
pixi run -e dev test
```

Expected: all tests PASS, including the cargo side.

- [ ] **Step 2: Run lint and typecheck**

```
pixi run -e dev ruff check python/ tests/
pixi run -e dev typecheck
```

Fix any new issues introduced by this initiative. Do NOT auto-fix unrelated pre-existing issues.

- [ ] **Step 3: Open the PR**

```
rtk git push -u origin <branch>
gh pr create --title "test: coverage initiative — kernels, readers, with_* API" --body "$(cat <<'EOF'
## Summary
- Wave 1: behavior tests for reconstruct/get_diffs/filter_af/choose_exonic/intervals kernels (coverage.py can't see numba)
- Wave 2: extended fasta/bigwig tests; new torch DataLoader smoke tests
- Wave 3: with_* method matrix, Dataset.open error paths, indexing edge cases
- Coverage config: omit numba-only modules from the gate; exclude numba-decorated functions

## Test plan
- [ ] `pixi run -e dev test` passes
- [ ] Coverage report shows pure-Python target modules ≥80%
- [ ] No regressions vs `2026-05-24-test-audit-coverage-baseline.txt`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

**Spec coverage:**
- Wave 1 (kernels): Tasks 1–5 cover every kernel named in the spec.
- Wave 2 (readers + DataLoader): Tasks 6–8.
- Wave 3 (API surface): Tasks 9–11 cover `with_*`, `_open`, `_indexing`. Task 12 handles coverage config. Spec items left explicitly under-covered (acceptable per spec's "Out of scope"): `_query` AF+exonic combination, `_reference` splice cross-contig, `data_registry` resolve path. If the implementer has time after Task 13, those are good follow-up commits inside the same PR; otherwise defer.

**Open ambiguities the implementer must resolve at execution time** (not placeholder bugs — these are genuinely "read the code first" items):
- Exact exception types raised by `Fasta`/`BigWigs`/`Dataset.open` on bad input.
- Exact `with_*` method set on `Dataset`.
- Whether `Dataset.to_dataset()` is the right method to obtain a torch-compatible view, and what its return type is.
- Accepted modes for `with_insertion_fill`.

Each task flags this and instructs the implementer to read the relevant source before finalizing assertions.
