# Track Insertion Fill Options — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the read-time track insertion fill behavior user-configurable via strategy classes (`Repeat5p`, `Repeat5pNormalized`, `Constant`, `FlankSample`, `Interpolate`) selected per-track via `Dataset.with_insertion_fill(...)`.

**Architecture:** Add a strategy-class hierarchy (`_insertion_fill.py`) that lowers to numba-friendly primitives (int enum + float64 params array). Extend the existing `shift_and_realign_track_sparse` numba kernel with a branching ladder at the single insertion-write line. `Tracks` holds a per-track strategy dict; `HapsTracks.__call__` looks up the right strategy per loop iteration and forwards lowered values into the kernel. `FlankSample` uses inline xorshift64 hashing of `(base_seed, query, hap, position)` for parallel-safe, deterministic randomness.

**Tech Stack:** Python, numba (`@njit(parallel=True)`), attrs, numpy.

**Spec:** [`docs/superpowers/specs/2026-05-11-track-insertion-options-design.md`](../specs/2026-05-11-track-insertion-options-design.md)

**Branch:** `feat/track-insertion-options`

---

## File Structure

- **Create:** `python/genvarloader/_dataset/_insertion_fill.py` — strategy class hierarchy + numba enum + `lower()` helper.
- **Modify:** `python/genvarloader/_dataset/_tracks.py` — extend both kernels with `strategy_id`, `params`, `base_seed`; add inline insertion-fill dispatch and inline xorshift64.
- **Modify:** `python/genvarloader/_dataset/_reconstruct.py` — add `insertion_fill` field + `with_insertion_fill` method on `Tracks`; update `with_tracks` pruning; route strategy through `HapsTracks.__call__`.
- **Modify:** `python/genvarloader/_dataset/_impl.py` — add `with_insertion_fill` method on `Dataset` / `RaggedDataset`.
- **Modify:** `python/genvarloader/__init__.py` — re-export public strategy classes.
- **Create:** `tests/dataset/test_insertion_fill.py` — kernel-level unit tests for every strategy + end-to-end Dataset test.
- **Modify:** `tests/dataset/test_realign.py` — pass the new kernel args (default = `Repeat5p`) so existing tests still call the updated signature.

---

## Task 1: Strategy classes and `lower()` helper

**Files:**
- Create: `python/genvarloader/_dataset/_insertion_fill.py`
- Test: `tests/dataset/test_insertion_fill.py`

- [ ] **Step 1: Create `_insertion_fill.py` with classes and constants**

```python
# python/genvarloader/_dataset/_insertion_fill.py
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from attrs import define, field
from numpy.typing import NDArray

REPEAT_5P = 0
REPEAT_5P_NORM = 1
CONSTANT = 2
FLANK_SAMPLE = 3
INTERPOLATE = 4

MAX_PARAMS = 2  # widest strategy uses 1 slot; keep small buffer for future


@define
class InsertionFill:
    """Base class for track insertion fill strategies. Do not instantiate directly."""


@define
class Repeat5p(InsertionFill):
    """Repeat the value at the variant POS across the entire inserted region. Current default behavior."""


@define
class Repeat5pNormalized(InsertionFill):
    """Repeat track[v_rel_pos] / (v_diff + 1) across the inserted region.

    Preserves the sum: the total written value equals track[v_rel_pos].
    """


@define
class Constant(InsertionFill):
    """Write a fixed value at every inserted position.

    Parameters
    ----------
    value
        Value to write. Defaults to NaN.
    """

    value: float = float("nan")


@define
class FlankSample(InsertionFill):
    """Sample (with replacement) from the 2*flank_width+1 reference values
    centered at the variant POS. Each inserted position samples independently.
    Out-of-bounds neighbors are clamped to in-bounds values.

    Parameters
    ----------
    flank_width
        Half-width of the flanking pool. Must be >= 0.
    """

    flank_width: int = 5

    def __attrs_post_init__(self):
        if self.flank_width < 0:
            raise ValueError(f"flank_width must be >= 0, got {self.flank_width}")


@define
class Interpolate(InsertionFill):
    """Polynomial interpolation across the inserted region.

    order=1: linear between track[v_rel_pos] and track[v_rel_pos + 1].
    order=2,3: Lagrange polynomial through ceil((order+1)/2) reference values
    on each side of the variant, clamped at boundaries.

    Parameters
    ----------
    order
        Polynomial order. Must be in {1, 2, 3}.
    """

    order: int = 1

    def __attrs_post_init__(self):
        if self.order not in (1, 2, 3):
            raise ValueError(f"Interpolate order must be 1, 2, or 3 (got {self.order})")


def lower(
    strategies: Sequence[InsertionFill],
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Pack strategy instances into numba-friendly arrays.

    Returns
    -------
    strategy_ids
        Shape (n,), int8. One enum value per strategy.
    params
        Shape (n, MAX_PARAMS), float64. Per-strategy parameter slots:
        - Repeat5p / Repeat5pNormalized: unused (all zeros).
        - Constant: [value, 0].
        - FlankSample: [flank_width, 0].
        - Interpolate: [order, 0].
    """
    n = len(strategies)
    ids = np.empty(n, dtype=np.int8)
    params = np.zeros((n, MAX_PARAMS), dtype=np.float64)
    for i, s in enumerate(strategies):
        if isinstance(s, Repeat5p):
            ids[i] = REPEAT_5P
        elif isinstance(s, Repeat5pNormalized):
            ids[i] = REPEAT_5P_NORM
        elif isinstance(s, Constant):
            ids[i] = CONSTANT
            params[i, 0] = s.value
        elif isinstance(s, FlankSample):
            ids[i] = FLANK_SAMPLE
            params[i, 0] = float(s.flank_width)
        elif isinstance(s, Interpolate):
            ids[i] = INTERPOLATE
            params[i, 0] = float(s.order)
        else:
            raise TypeError(f"Unknown InsertionFill subclass: {type(s).__name__}")
    return ids, params
```

- [ ] **Step 2: Create test file with `lower()` tests**

```python
# tests/dataset/test_insertion_fill.py
import math

import numpy as np
import pytest

from genvarloader._dataset._insertion_fill import (
    CONSTANT,
    FLANK_SAMPLE,
    INTERPOLATE,
    REPEAT_5P,
    REPEAT_5P_NORM,
    Constant,
    FlankSample,
    Interpolate,
    Repeat5p,
    Repeat5pNormalized,
    lower,
)


def test_lower_all_strategies():
    strategies = [
        Repeat5p(),
        Repeat5pNormalized(),
        Constant(value=0.5),
        FlankSample(flank_width=3),
        Interpolate(order=2),
    ]
    ids, params = lower(strategies)
    assert ids.dtype == np.int8
    assert params.dtype == np.float64
    assert ids.tolist() == [
        REPEAT_5P,
        REPEAT_5P_NORM,
        CONSTANT,
        FLANK_SAMPLE,
        INTERPOLATE,
    ]
    assert params[2, 0] == 0.5
    assert params[3, 0] == 3.0
    assert params[4, 0] == 2.0


def test_lower_empty():
    ids, params = lower([])
    assert ids.shape == (0,)
    assert params.shape == (0, 2)


def test_constant_default_is_nan():
    assert math.isnan(Constant().value)


def test_flank_sample_negative_width_rejected():
    with pytest.raises(ValueError, match="flank_width must be >= 0"):
        FlankSample(flank_width=-1)


def test_interpolate_order_capped():
    Interpolate(order=1)
    Interpolate(order=3)
    with pytest.raises(ValueError, match="order must be 1, 2, or 3"):
        Interpolate(order=4)
    with pytest.raises(ValueError, match="order must be 1, 2, or 3"):
        Interpolate(order=0)


def test_lower_unknown_class_raises():
    class Bogus:
        pass

    with pytest.raises(TypeError, match="Unknown InsertionFill subclass"):
        lower([Bogus()])  # type: ignore[list-item]
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_insertion_fill.py -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_insertion_fill.py tests/dataset/test_insertion_fill.py
rtk git commit -m "feat: add InsertionFill strategy classes and lowering helper"
```

---

## Task 2: Extend inner numba kernel with insertion-fill dispatch

**Files:**
- Modify: `python/genvarloader/_dataset/_tracks.py` (replace `_tracks.py:215`'s single-line write with strategy dispatch; add new function args)
- Modify: `tests/dataset/test_realign.py` (pass new kwargs with `Repeat5p` defaults)
- Test: `tests/dataset/test_insertion_fill.py` (extend with kernel-level strategy tests)

- [ ] **Step 1: Add inline xorshift64 helper and strategy constants to `_tracks.py`**

Add at the top of `python/genvarloader/_dataset/_tracks.py` (after the `__all__ = []` line):

```python
# Strategy enum (mirrors _insertion_fill.py; duplicated to avoid Python-level
# imports inside @njit functions)
_REPEAT_5P = 0
_REPEAT_5P_NORM = 1
_CONSTANT = 2
_FLANK_SAMPLE = 3
_INTERPOLATE = 4


@nb.njit(nogil=True, cache=True, inline="always")
def _xorshift64(x: np.uint64) -> np.uint64:
    """Single round of xorshift64. Pure function — safe in parallel."""
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    return x


@nb.njit(nogil=True, cache=True, inline="always")
def _hash4(a: np.uint64, b: np.uint64, c: np.uint64, d: np.uint64) -> np.uint64:
    """Hash four uint64 values into one. Used as a per-position deterministic seed."""
    h = a
    h = _xorshift64(h ^ b)
    h = _xorshift64(h ^ c)
    h = _xorshift64(h ^ d)
    return h
```

- [ ] **Step 2: Add the strategy-dispatch helper**

Add this `@njit` helper function (also near the top of `_tracks.py`):

```python
@nb.njit(nogil=True, cache=True, inline="always")
def _apply_insertion_fill(
    out: NDArray[np.floating],
    out_idx: int,
    writable_length: int,
    v_len: int,
    track: NDArray[np.floating],
    v_rel_pos: int,
    strategy_id: int,
    params: NDArray[np.float64],
    base_seed: np.uint64,
    query: int,
    hap: int,
):
    """Write `writable_length` values at out[out_idx:] according to strategy.

    v_len is the total length of the insertion stretch (v_diff + 1); the kernel
    may truncate the actual write to writable_length when running out of output.
    """
    track_len = len(track)

    if strategy_id == _REPEAT_5P:
        val = track[v_rel_pos]
        for i in range(writable_length):
            out[out_idx + i] = val

    elif strategy_id == _REPEAT_5P_NORM:
        val = track[v_rel_pos] / v_len
        for i in range(writable_length):
            out[out_idx + i] = val

    elif strategy_id == _CONSTANT:
        val = params[0]
        for i in range(writable_length):
            out[out_idx + i] = val

    elif strategy_id == _FLANK_SAMPLE:
        width = np.int64(params[0])
        pool_lo = max(0, v_rel_pos - width)
        pool_hi = min(track_len - 1, v_rel_pos + width)
        pool_size = pool_hi - pool_lo + 1
        for i in range(writable_length):
            seed = _hash4(
                base_seed,
                np.uint64(query),
                np.uint64(hap),
                np.uint64(out_idx + i),
            )
            offset = np.int64(seed % np.uint64(pool_size))
            out[out_idx + i] = track[pool_lo + offset]

    elif strategy_id == _INTERPOLATE:
        order = np.int64(params[0])
        # Number of anchor values per side: ceil((order+1)/2)
        k = (order + 1 + 1) // 2  # ceil((order+1)/2)
        # Build anchor x and y arrays in *insertion-output* coordinate where the
        # inserted stretch occupies positions [0, v_len - 1]. The 5' anchor
        # (track[v_rel_pos]) corresponds to x = 0; the 3' anchor
        # (track[v_rel_pos + 1]) corresponds to x = v_len. Left flanks lie at
        # negative x = -1, -2, ...; right flanks at x = v_len + 1, v_len + 2, ...
        n_anchors = 2 * k
        xs = np.empty(n_anchors, dtype=np.float64)
        ys = np.empty(n_anchors, dtype=np.float64)
        for j in range(k):
            # Left anchors: positions v_rel_pos - j (clamped)
            ref_idx = v_rel_pos - j
            if ref_idx < 0:
                ref_idx = 0
            xs[j] = -float(j)
            ys[j] = track[ref_idx]
        for j in range(k):
            # Right anchors: positions v_rel_pos + 1 + j (clamped)
            ref_idx = v_rel_pos + 1 + j
            if ref_idx > track_len - 1:
                ref_idx = track_len - 1
            xs[k + j] = float(v_len) + float(j)
            ys[k + j] = track[ref_idx]
        # Lagrange interpolation at each output position (in [0, writable_length))
        for i in range(writable_length):
            x = float(i)
            acc = 0.0
            for a in range(n_anchors):
                term = ys[a]
                for b in range(n_anchors):
                    if b == a:
                        continue
                    term *= (x - xs[b]) / (xs[a] - xs[b])
                acc += term
            out[out_idx + i] = acc
```

- [ ] **Step 3: Update `shift_and_realign_track_sparse` signature and replace the hardcoded write**

In `_tracks.py`, change the function signature (around line 89) by appending these keyword-arg-defaulted parameters:

```python
@nb.njit(nogil=True, cache=True)
def shift_and_realign_track_sparse(
    offset_idx: int,
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    shift: int,
    track: NDArray[np.floating],
    query_start: int,
    out: NDArray[np.floating],
    keep: NDArray[np.bool_] | None = None,
    strategy_id: int = 0,            # _REPEAT_5P
    params: NDArray[np.float64] | None = None,
    base_seed: np.uint64 = np.uint64(0),
    query: int = 0,
    hap: int = 0,
):
```

Find the existing two lines that fill an insertion (around `_tracks.py:214-216`):

```python
        # indels (substitutions are skipped above and then handled by above clause)
        writable_length = min(v_len, length - out_idx)
        out[out_idx : out_idx + writable_length] = track[v_rel_pos]
        out_idx += writable_length
        track_idx = v_rel_end
```

Replace the single-line write with a dispatch call (`v_diff == 0` substitutions are already skipped earlier, so this branch only runs for `v_diff != 0`; insertions are `v_diff > 0`, deletions are `v_diff < 0` with `v_len == 1`). To preserve current semantics for deletions (`v_len == 1`, just writes one value at v_rel_pos), the dispatch should also handle them — `Repeat5p` matches current behavior. Since deletions write a single value equal to `track[v_rel_pos]`, they match `_REPEAT_5P`. All other strategies *only* meaningfully apply to insertions, so we guard:

```python
        # indels (substitutions are skipped above and then handled by above clause)
        writable_length = min(v_len, length - out_idx)
        if v_diff > 0 and strategy_id != _REPEAT_5P and params is not None:
            _apply_insertion_fill(
                out=out,
                out_idx=out_idx,
                writable_length=writable_length,
                v_len=v_len,
                track=track,
                v_rel_pos=v_rel_pos,
                strategy_id=strategy_id,
                params=params,
                base_seed=base_seed,
                query=query,
                hap=hap,
            )
        else:
            # Deletions and Repeat5p insertions: original behavior.
            for i in range(writable_length):
                out[out_idx + i] = track[v_rel_pos]
        out_idx += writable_length
        track_idx = v_rel_end
```

Note: the `for i in range(writable_length)` loop replaces the slice assignment because numba's parallel kernel cannot mix slice broadcast with the conditional dispatch cleanly without dimension-change warnings. The explicit loop is identical in behavior and well-supported.

- [ ] **Step 4: Update `tests/dataset/test_realign.py` to call the new signature**

Existing tests at `tests/dataset/test_realign.py:122-145` call the kernel without the new args. Add a `params=np.zeros((2,), dtype=np.float64)` default at the call site so numba's type inference is happy when the function is jit-compiled:

```python
def test_sparse(
    v_starts,
    ilens,
    shift,
    track,
    sparse_genos: Ragged[V_IDX_TYPE],
    desired,
    query_start,
):
    offset_idx = 0
    actual = np.empty(len(track) - query_start, np.float32)
    shift_and_realign_track_sparse(
        offset_idx=offset_idx,
        geno_v_idxs=sparse_genos.data,
        geno_offsets=sparse_genos.offsets,
        v_starts=v_starts,
        ilens=ilens,
        shift=shift,
        track=track,
        query_start=query_start,
        out=actual,
        params=np.zeros(2, dtype=np.float64),
    )

    np.testing.assert_equal(actual, desired)
```

- [ ] **Step 5: Add kernel-level tests for each strategy in `tests/dataset/test_insertion_fill.py`**

Append to `tests/dataset/test_insertion_fill.py`:

```python
from genoray._svar import dense2sparse
from genvarloader._dataset._tracks import shift_and_realign_track_sparse


def _run_kernel(strategy_id, params, base_seed=np.uint64(0)):
    """Run the kernel on a single insertion at v_rel_pos=1, v_diff=3.

    Track is [0, 10, 20, 30, 40] (5 values). The variant at start=1 (v_rel_pos=1)
    with ilen=3 inserts 3 bases. Output region length matches track length + 3 = 8.
    """
    v_starts = np.array([1], dtype=np.int32)
    ilens = np.array([3], dtype=np.int32)
    track = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    # genotypes (s p v) = (1 1 1)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)
    sparse_genos = dense2sparse(genos, var_idxs)

    out = np.zeros(8, dtype=np.float32)
    shift_and_realign_track_sparse(
        offset_idx=0,
        geno_v_idxs=sparse_genos.data,
        geno_offsets=sparse_genos.offsets,
        v_starts=v_starts,
        ilens=ilens,
        shift=0,
        track=track,
        query_start=0,
        out=out,
        strategy_id=strategy_id,
        params=params,
        base_seed=base_seed,
    )
    return out, track


def test_kernel_repeat_5p_default():
    out, track = _run_kernel(REPEAT_5P, np.zeros(2, dtype=np.float64))
    # Positions 1..4 are the v_len=4 insertion stretch (anchor + 3 inserted bases).
    # All equal track[1] = 10.
    np.testing.assert_array_equal(
        out[1:5], np.array([10, 10, 10, 10], dtype=np.float32)
    )
    # Surrounding values are reference data.
    assert out[0] == 0.0
    np.testing.assert_array_equal(out[5:], track[2:])


def test_kernel_repeat_5p_normalized():
    out, _ = _run_kernel(REPEAT_5P_NORM, np.zeros(2, dtype=np.float64))
    # Sum across v_len=4 positions should equal track[v_rel_pos] = 10.
    assert math.isclose(out[1:5].sum(), 10.0, abs_tol=1e-6)
    # All four positions equal.
    assert np.allclose(out[1:5], out[1])


def test_kernel_constant_nan():
    params = np.zeros(2, dtype=np.float64)
    params[0] = float("nan")
    out, _ = _run_kernel(CONSTANT, params)
    assert np.all(np.isnan(out[1:5]))
    # Outside the insertion: still real values.
    assert out[0] == 0.0
    assert out[5] == 20.0


def test_kernel_flank_sample_pool_membership():
    params = np.zeros(2, dtype=np.float64)
    params[0] = 2.0  # flank_width
    # Track = [0, 10, 20, 30, 40], v_rel_pos=1, flank_width=2 ->
    # pool = track[max(0, -1):min(4, 3)+1] = track[0:4] = [0, 10, 20, 30]
    out, _ = _run_kernel(FLANK_SAMPLE, params, base_seed=np.uint64(42))
    pool = {0.0, 10.0, 20.0, 30.0}
    for v in out[1:5]:
        assert float(v) in pool


def test_kernel_flank_sample_deterministic():
    params = np.zeros(2, dtype=np.float64)
    params[0] = 2.0
    a, _ = _run_kernel(FLANK_SAMPLE, params, base_seed=np.uint64(123))
    b, _ = _run_kernel(FLANK_SAMPLE, params, base_seed=np.uint64(123))
    np.testing.assert_array_equal(a, b)


def test_kernel_interpolate_linear():
    params = np.zeros(2, dtype=np.float64)
    params[0] = 1.0  # order=1
    out, _ = _run_kernel(INTERPOLATE, params)
    # Linear between track[1]=10 (at x=0) and track[2]=20 (at x=v_len=4),
    # evaluated at x=0,1,2,3 (the 4 written positions). Slope = (20-10)/4 = 2.5.
    expected = np.array([10.0, 12.5, 15.0, 17.5], dtype=np.float32)
    np.testing.assert_allclose(out[1:5], expected, atol=1e-5)


def test_kernel_interpolate_cubic_passes_through_anchors():
    """Cubic Lagrange through 4 colinear points must reproduce the linear answer."""
    params = np.zeros(2, dtype=np.float64)
    params[0] = 3.0  # order=3 -> 2 anchors per side
    out, _ = _run_kernel(INTERPOLATE, params)
    # Track is a linear ramp, so cubic interpolation must give the linear result.
    expected = np.array([10.0, 12.5, 15.0, 17.5], dtype=np.float32)
    np.testing.assert_allclose(out[1:5], expected, atol=1e-4)


def test_kernel_flank_sample_edge_clamp():
    """Insertion at the very start of the track — pool should clamp without crash."""
    v_starts = np.array([0], dtype=np.int32)
    ilens = np.array([2], dtype=np.int32)
    track = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)
    sparse_genos = dense2sparse(genos, var_idxs)

    params = np.zeros(2, dtype=np.float64)
    params[0] = 10.0  # flank_width larger than track
    out = np.zeros(5, dtype=np.float32)
    shift_and_realign_track_sparse(
        offset_idx=0,
        geno_v_idxs=sparse_genos.data,
        geno_offsets=sparse_genos.offsets,
        v_starts=v_starts,
        ilens=ilens,
        shift=0,
        track=track,
        query_start=0,
        out=out,
        strategy_id=FLANK_SAMPLE,
        params=params,
        base_seed=np.uint64(7),
    )
    pool = {5.0, 6.0, 7.0}
    for v in out[:3]:  # v_len = 3 positions written from the insertion
        assert float(v) in pool
```

- [ ] **Step 6: Run tests**

```bash
pixi run -e dev pytest tests/dataset/test_insertion_fill.py tests/dataset/test_realign.py -v
```

Expected: all tests pass. If `test_realign.py` had a previously-cached numba dispatch, the first run may re-compile — that's fine.

- [ ] **Step 7: Commit**

```bash
rtk git add python/genvarloader/_dataset/_tracks.py tests/dataset/test_insertion_fill.py tests/dataset/test_realign.py
rtk git commit -m "feat: kernel-level insertion-fill strategy dispatch"
```

---

## Task 3: Propagate strategy args through the parallel wrapper

**Files:**
- Modify: `python/genvarloader/_dataset/_tracks.py` — `shift_and_realign_tracks_sparse` (lines 8–86)

- [ ] **Step 1: Add new args to outer kernel signature**

Update `shift_and_realign_tracks_sparse`'s signature (lines 8–23 of the current file) to:

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def shift_and_realign_tracks_sparse(
    out: NDArray[np.floating],
    out_offsets: NDArray[np.integer],
    regions: NDArray[np.integer],
    shifts: NDArray[np.integer],
    geno_offset_idxs: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    tracks: NDArray[np.floating],
    track_offsets: NDArray[np.integer],
    keep: NDArray[np.bool_] | None = None,
    keep_offsets: NDArray[np.integer] | None = None,
    strategy_id: int = 0,           # _REPEAT_5P
    params: NDArray[np.float64] | None = None,
    base_seed: np.uint64 = np.uint64(0),
):
```

- [ ] **Step 2: Pass strategy args into the inner call**

In the inner loop (currently at `_tracks.py:75-86`), pass through:

```python
            shift_and_realign_track_sparse(
                offset_idx=o_idx,
                geno_v_idxs=geno_v_idxs,
                geno_offsets=geno_offsets,
                v_starts=v_starts,
                ilens=ilens,
                shift=qh_shifts,
                track=q_track,
                query_start=q_start,
                out=qh_out,
                keep=qh_keep,
                strategy_id=strategy_id,
                params=params if params is not None else np.zeros(2, dtype=np.float64),
                base_seed=base_seed,
                query=query,
                hap=hap,
            )
```

Note: numba `@njit` doesn't allow `None` to flow into the inner function if its signature expects an array. The `np.zeros(2, ...)` fallback ensures we always pass a concrete array. (When `strategy_id == _REPEAT_5P`, the inner kernel never reads `params`.)

- [ ] **Step 3: Run existing realign tests to confirm wrapper still works**

```bash
pixi run -e dev pytest tests/dataset/test_realign.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_tracks.py
rtk git commit -m "feat: thread insertion-fill args through parallel track kernel"
```

---

## Task 4: Add `insertion_fill` field and `with_insertion_fill` on `Tracks`

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` — `Tracks` class (lines 691–765)

- [ ] **Step 1: Update imports at the top of `_reconstruct.py`**

Add:

```python
from collections.abc import Mapping

from ._insertion_fill import InsertionFill, Repeat5p
```

(`Mapping` may already be imported via `collections.abc.Iterable` line — verify and add if missing.)

- [ ] **Step 2: Add `insertion_fill` field to `Tracks`**

Add a new field to the `@define class Tracks` (right after `available_tracks`):

```python
@define
class Tracks(Reconstructor[_T]):
    intervals: dict[str, RaggedIntervals]
    active_tracks: dict[str, TrackType]
    available_tracks: dict[str, TrackType]
    kind: type[_T]
    n_regions: int
    n_samples: int
    insertion_fill: dict[str, InsertionFill] = field(factory=dict)
```

- [ ] **Step 3: Initialize `insertion_fill` in `from_path`**

In `Tracks.from_path` (current return statement at line 765), default the field to `Repeat5p()` per active track:

```python
        insertion_fill = {name: Repeat5p() for name in all_tracks}
        return cls(
            intervals,
            all_tracks,
            all_tracks,
            kind,
            n_regions,
            n_samples,
            insertion_fill,
        )
```

- [ ] **Step 4: Add `with_insertion_fill` method**

Insert just below `with_tracks` (around line 715):

```python
    def with_insertion_fill(
        self,
        fill: InsertionFill | Mapping[str, InsertionFill],
    ) -> Tracks:
        """Configure the insertion-fill strategy for each active track.

        Parameters
        ----------
        fill
            Either a single :class:`InsertionFill` strategy applied to every
            active track, or a mapping from track name to strategy. Track names
            not present in the mapping fall back to :class:`Repeat5p`.
        """
        if isinstance(fill, InsertionFill):
            fills = {name: fill for name in self.active_tracks}
        else:
            fills = {name: fill.get(name, Repeat5p()) for name in self.active_tracks}
        return evolve(self, insertion_fill=fills)
```

- [ ] **Step 5: Prune `insertion_fill` in `with_tracks`**

Update `with_tracks` (around line 701-715). The current body ends with:

```python
        tracks = {t: self.available_tracks[t] for t in _tracks}
        return evolve(self, active_tracks=tracks)
```

Replace with:

```python
        tracks = {t: self.available_tracks[t] for t in _tracks}
        fills = {t: self.insertion_fill.get(t, Repeat5p()) for t in _tracks}
        return evolve(self, active_tracks=tracks, insertion_fill=fills)
```

Also update the `tracks is None` branch:

```python
        if tracks is None:
            return evolve(self, active_tracks={}, insertion_fill={})
```

- [ ] **Step 6: Add unit tests for `Tracks.with_insertion_fill`**

Append to `tests/dataset/test_insertion_fill.py`:

```python
from genvarloader._dataset._reconstruct import Tracks, TrackType
from genvarloader._ragged import RaggedIntervals


def _make_tracks(names):
    # Minimal Tracks instance for testing the with_insertion_fill plumbing only.
    starts = ends = values = np.array([0], dtype=np.int32)
    offsets = np.array([0, 1], dtype=np.int64)
    from seqpro.rag import Ragged

    intervals = {
        n: RaggedIntervals(
            Ragged.from_offsets(starts, (1, None), offsets),
            Ragged.from_offsets(ends, (1, None), offsets),
            Ragged.from_offsets(values, (1, None), offsets),
        )
        for n in names
    }
    active = {n: TrackType.SAMPLE for n in names}
    return Tracks(
        intervals=intervals,
        active_tracks=active,
        available_tracks=active,
        kind=type("Dummy", (), {}),  # not exercised here
        n_regions=1,
        n_samples=1,
        insertion_fill={n: Repeat5p() for n in names},
    )


def test_with_insertion_fill_single_applies_to_all():
    tracks = _make_tracks(["a", "b"])
    new = tracks.with_insertion_fill(Constant(0.0))
    assert isinstance(new.insertion_fill["a"], Constant)
    assert isinstance(new.insertion_fill["b"], Constant)
    # original unchanged (evolve returns new instance)
    assert isinstance(tracks.insertion_fill["a"], Repeat5p)


def test_with_insertion_fill_dict_partial_falls_back():
    tracks = _make_tracks(["a", "b"])
    new = tracks.with_insertion_fill({"a": FlankSample(flank_width=2)})
    assert isinstance(new.insertion_fill["a"], FlankSample)
    assert isinstance(new.insertion_fill["b"], Repeat5p)  # fallback


def test_with_tracks_prunes_insertion_fill():
    tracks = _make_tracks(["a", "b"]).with_insertion_fill({
        "a": Constant(0.0),
        "b": FlankSample(),
    })
    new = tracks.with_tracks("a")
    assert set(new.insertion_fill) == {"a"}
    assert isinstance(new.insertion_fill["a"], Constant)
```

- [ ] **Step 7: Run tests**

```bash
pixi run -e dev pytest tests/dataset/test_insertion_fill.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reconstruct.py tests/dataset/test_insertion_fill.py
rtk git commit -m "feat: per-track insertion-fill on Tracks reconstructor"
```

---

## Task 5: Wire strategy through `HapsTracks.__call__`

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` — `HapsTracks.__call__` (per-track loop around line 1151)

- [ ] **Step 1: Import the lowering helper**

At the top of `_reconstruct.py`, add:

```python
from ._insertion_fill import lower as _lower_insertion_fills
```

- [ ] **Step 2: Lower strategies and pass to kernel**

Find the per-track loop in `HapsTracks.__call__` (around `_reconstruct.py:1151-1190`). Just before the `for track_ofst, (name, tracktype) in enumerate(...)` line, lower the strategies once:

```python
strat_list = [self.tracks.insertion_fill[name] for name in self.tracks.active_tracks]
strat_ids, strat_params = _lower_insertion_fills(strat_list)
# Draw a base seed for FlankSample determinism. If deterministic,
# derive it from idx so calls are reproducible.
if deterministic:
    base_seed = np.uint64(int(idx.sum()) & ((1 << 63) - 1))
else:
    base_seed = np.uint64(rng.integers(0, 1 << 63))
```

Inside the existing loop, update the `shift_and_realign_tracks_sparse` call (`_reconstruct.py:1176-1190`) to pass the per-track strategy:

```python
                shift_and_realign_tracks_sparse(
                    out=_out,
                    out_offsets=out_ofsts_per_t,
                    regions=regions,
                    shifts=shifts,
                    geno_offset_idxs=geno_idx,
                    geno_v_idxs=self.haps.genotypes.data,
                    geno_offsets=self.haps.genotypes.offsets,
                    v_starts=self.haps.variants.start,
                    ilens=self.haps.variants.ilen,
                    tracks=_tracks,
                    track_offsets=track_ofsts_per_t,
                    keep=keep,
                    keep_offsets=keep_offsets,
                    strategy_id=int(strat_ids[track_ofst]),
                    params=strat_params[track_ofst],
                    base_seed=base_seed,
                )
```

- [ ] **Step 3: Run the existing dataset tests to confirm no regression**

```bash
pixi run -e dev pytest tests/dataset/test_ds_haps.py tests/dataset/test_dataset.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reconstruct.py
rtk git commit -m "feat: route per-track insertion fill into HapsTracks kernel call"
```

---

## Task 6: Dataset-level `with_insertion_fill` + public API exports + end-to-end test

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` — add method
- Modify: `python/genvarloader/__init__.py` — re-export classes
- Test: `tests/dataset/test_insertion_fill.py` — end-to-end test

- [ ] **Step 1: Add `with_insertion_fill` to `Dataset`**

In `python/genvarloader/_dataset/_impl.py`, just after the `with_tracks` method (around line 775), add:

```python
def with_insertion_fill(
    self,
    fill: "InsertionFill | Mapping[str, InsertionFill]",
):
    """Configure how track values are filled at insertion sites.

    Only meaningful when the dataset returns haplotypes *and* tracks (i.e.
    when the reconstructor is :class:`HapsTracks`). Pure-reference and
    pure-haplotype datasets have no insertion fill to configure.

    Parameters
    ----------
    fill
        Either a single :class:`InsertionFill` strategy applied to every
        active track, or a dict mapping track name to strategy. Tracks not
        in the dict fall back to :class:`Repeat5p`.
    """
    if self._tracks is None:
        raise ValueError("Dataset has no tracks; with_insertion_fill is a no-op.")
    if not isinstance(self._recon, HapsTracks):
        raise ValueError(
            "with_insertion_fill is only meaningful for datasets with both "
            "haplotypes and tracks (reconstructor must be HapsTracks)."
        )
    new_tracks = self._tracks.with_insertion_fill(fill)
    new_recon = evolve(self._recon, tracks=new_tracks)
    return evolve(self, _tracks=new_tracks, _recon=new_recon)
```

Add imports near the top of `_impl.py` (find the existing `from ._reconstruct import` block and append):

```python
from ._insertion_fill import InsertionFill
```

Also add to the `from collections.abc import` block if it doesn't already include `Mapping`:

```python
from collections.abc import Mapping
```

- [ ] **Step 2: Re-export public classes from the top-level package**

Update `python/genvarloader/__init__.py`:

```python
from ._dataset._insertion_fill import (
    Constant,
    FlankSample,
    InsertionFill,
    Interpolate,
    Repeat5p,
    Repeat5pNormalized,
)
```

Add to `__all__`:

```python
("InsertionFill",)
("Repeat5p",)
("Repeat5pNormalized",)
("Constant",)
("FlankSample",)
("Interpolate",)
```

- [ ] **Step 3: Add an end-to-end test**

Append to `tests/dataset/test_insertion_fill.py`:

```python
import genvarloader as gvl


def test_end_to_end_constant_nan(tmp_path):
    """Use the existing dummy dataset to confirm with_insertion_fill plumbing works end-to-end."""
    ds = gvl.get_dummy_dataset()
    # Only haps+tracks datasets support insertion fill.
    if not (ds._tracks is not None and ds._seqs is not None):
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    # Configure NaN fill for the first track.
    first_track = next(iter(ds._tracks.active_tracks))
    ds_nan = ds.with_insertion_fill({first_track: Constant(float("nan"))})
    # The returned object is a new dataset; the configured strategy is recorded.
    assert isinstance(ds_nan._tracks.insertion_fill[first_track], Constant)
    assert math.isnan(ds_nan._tracks.insertion_fill[first_track].value)


def test_with_insertion_fill_rejects_tracks_only_dataset():
    """Datasets without haplotypes cannot have insertion fill applied."""
    ds = gvl.get_dummy_dataset()
    if ds._seqs is not None:
        pytest.skip("dummy dataset has sequences; nothing to test")
    with pytest.raises(ValueError, match="HapsTracks"):
        ds.with_insertion_fill(Repeat5p())
```

- [ ] **Step 4: Run full test suite**

```bash
pixi run -e dev pytest tests/dataset/ -v
```

Expected: all pass. Confirm no existing dataset test regressed.

- [ ] **Step 5: Lint and typecheck**

```bash
pixi run -e dev ruff check python/genvarloader/_dataset/_insertion_fill.py python/genvarloader/_dataset/_tracks.py python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_impl.py python/genvarloader/__init__.py
pixi run -e dev basedpyright python/genvarloader/_dataset/_insertion_fill.py
```

Expected: clean (or pre-existing warnings only).

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py python/genvarloader/__init__.py tests/dataset/test_insertion_fill.py
rtk git commit -m "feat: Dataset.with_insertion_fill + public API exports"
```

---

## Self-Review Checklist

Run through this before opening the PR:

- [ ] Every spec strategy has a kernel test and a per-track lowering test.
- [ ] The default path (`Repeat5p`, no `with_insertion_fill` call) is byte-identical to pre-change behavior (`test_kernel_repeat_5p_default` + existing `test_realign.py` cases).
- [ ] `Constant(NaN)`, `FlankSample`, `Interpolate(1)`, `Interpolate(3)` all have dedicated tests.
- [ ] Per-track dict form is tested (`test_with_insertion_fill_dict_partial_falls_back`).
- [ ] Edge clamping is tested (`test_kernel_flank_sample_edge_clamp`).
- [ ] `with_tracks` prunes `insertion_fill` (`test_with_tracks_prunes_insertion_fill`).
- [ ] Public classes are re-exported and present in `__all__`.
- [ ] No `np.random` globals inside the parallel kernel — only inline xorshift64.
