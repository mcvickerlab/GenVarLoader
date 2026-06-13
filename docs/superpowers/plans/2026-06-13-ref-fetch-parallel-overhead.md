# Reference.fetch parallel-overhead + fetch-fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ~37 ms/call numba `parallel=True` fork-join overhead and the 3-redundant-fetches-per-decode in the `variant-windows` reference read path, so `Reference.fetch` scales with bytes copied instead of dominating the decode.

**Architecture:** (1) Cap numba's worker count to the cgroup-aware core count once at import. (2) Add a thread-aware dispatcher that routes the two reference-copy kernels (`_fetch_impl`, `get_reference`) to a serial njit below a per-thread byte threshold and a parallel njit above it. (3) Collapse the variant-windows flank reads from 3 fetches to 1 by reading `[start−L, end+L)` once and slicing `f5`/`f3` — done *internally* so all `_flat_flanks` function signatures stay identical and existing oracle tests act as byte-identical guards.

**Tech Stack:** Python, numba (`@nb.njit`), numpy, polars; `pixi run -e dev` for all tasks.

**Reference:** spec at `docs/superpowers/specs/2026-06-13-ref-fetch-parallel-overhead-design.md`; issue [#221](https://github.com/mcvickerlab/GenVarLoader/issues/221).

**Design refinement vs. spec:** The spec sketches builders that "take pre-fetched data." Implementation keeps `_flat_flanks` signatures unchanged and does the single fetch+slice *inside* `compute_alt_window`, `compute_flank_tokens`, and `compute_windows`. The hot `ref=window,alt=window` path is routed through `compute_windows` (1 fetch). This is less churn and lets the existing oracle tests (which compute the old separate-fetch result) verify byte-identity automatically.

**Before starting:** ensure test data exists — run `pixi run -e dev gen` once if `tests/data` is empty.

---

### Task 1: Thread cap + dispatch predicate (`_threads.py`)

**Files:**
- Create: `python/genvarloader/_threads.py`
- Modify: `python/genvarloader/__init__.py:1-7` (add early import + cap call)
- Test: `tests/unit/test_threads.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_threads.py`:

```python
import os

import numba
import genvarloader._threads as th


def test_resolve_honors_env_override(monkeypatch):
    monkeypatch.setenv("GVL_NUM_THREADS", "7")
    # env wins, clamped to >= 1 and <= numba hard max
    monkeypatch.setattr(numba, "get_num_threads", lambda: 64)
    assert th._resolve_num_threads() == 7


def test_resolve_env_clamped_to_numba_max(monkeypatch):
    monkeypatch.setenv("GVL_NUM_THREADS", "9999")
    monkeypatch.setattr(numba, "get_num_threads", lambda: 64)
    assert th._resolve_num_threads() == 64


def test_resolve_uses_cgroup_affinity(monkeypatch):
    monkeypatch.delenv("GVL_NUM_THREADS", raising=False)
    # host reports 208 logical CPUs, cgroup allows 52 -> min wins
    monkeypatch.setattr(numba, "get_num_threads", lambda: 208)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(52)))
    assert th._resolve_num_threads() == 52


def test_should_parallelize_threshold(monkeypatch):
    monkeypatch.setattr(numba, "get_num_threads", lambda: 4)
    thresh = 4 * th._MIN_BYTES_PER_THREAD
    assert th.should_parallelize(thresh - 1) is False
    assert th.should_parallelize(thresh) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_threads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'genvarloader._threads'`

- [ ] **Step 3: Write minimal implementation**

Create `python/genvarloader/_threads.py`:

```python
"""Cgroup-aware numba thread cap + a per-thread dispatch predicate.

numba.get_num_threads() reports host logical CPUs, not the cgroup allocation
(e.g. 208 reported vs. 52 allocated). Forking the misdetected count makes
parallel=True regions pay a flat ~37 ms fork-join for trivial work. We cap the
worker count down to the real allocation once at import, and route copy kernels
to a serial variant unless there is enough work to amortize the fork-join.
"""

from __future__ import annotations

import os

import numba

# Parallel only pays off when each worker gets at least this many bytes to copy.
# Below `num_threads * _MIN_BYTES_PER_THREAD` total, the serial kernel wins.
_MIN_BYTES_PER_THREAD = 1 << 20  # 1 MiB


def _resolve_num_threads() -> int:
    hard_max = numba.get_num_threads()
    env = os.environ.get("GVL_NUM_THREADS")
    if env:
        return max(1, min(int(env), hard_max))
    try:
        real = len(os.sched_getaffinity(0))  # respects cgroup cpuset (Linux)
    except AttributeError:
        real = os.cpu_count() or 1  # non-Linux fallback
    return max(1, min(real, hard_max))


def cap_numba_threads() -> int:
    """Cap numba's parallel worker count to the resolved value. Idempotent."""
    n = _resolve_num_threads()
    numba.set_num_threads(n)
    return n


def should_parallelize(total_bytes: int) -> bool:
    """True iff a copy of `total_bytes` is large enough to justify fork-join."""
    return total_bytes >= numba.get_num_threads() * _MIN_BYTES_PER_THREAD
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_threads.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Wire the cap into package import**

Modify `python/genvarloader/__init__.py` — add the cap immediately after the stdlib import, before the seqpro/submodule imports (so it runs before any kernel could fire):

```python
import importlib.metadata

from ._threads import cap_numba_threads

cap_numba_threads()

from seqpro.bed import read as read_bedlike
```

- [ ] **Step 6: Verify import works and nothing regressed**

Run: `pixi run -e dev python -c "import genvarloader; import numba; print(numba.get_num_threads())"`
Expected: prints an integer ≤ the host CPU count, no errors.

- [ ] **Step 7: Commit**

```bash
rtk git add python/genvarloader/_threads.py python/genvarloader/__init__.py tests/unit/test_threads.py
rtk git commit -m "perf(threads): cap numba workers to cgroup cores + add dispatch predicate (#221)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Serial/parallel dispatch for `_fetch_impl` (`Reference.fetch`)

**Files:**
- Modify: `python/genvarloader/_dataset/_reference.py:117-165` (`fetch` + `_fetch_impl`)
- Test: `tests/unit/dataset/test_ref_fetch_dispatch.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dataset/test_ref_fetch_dispatch.py`:

```python
import numpy as np
from seqpro.rag import lengths_to_offsets

from genvarloader._dataset._reference import _fetch_impl_ser, _fetch_impl_par


def _run(kernel, c_idxs, starts, ends, reference, ref_offsets, pad_char):
    out_offsets = lengths_to_offsets(ends - starts)
    out = np.empty(int(out_offsets[-1]), np.uint8)
    kernel(c_idxs, starts, ends, reference, ref_offsets, pad_char, out, out_offsets)
    return out


def test_serial_and_parallel_kernels_agree():
    rng = np.random.default_rng(0)
    reference = rng.integers(65, 85, size=500, dtype=np.uint8)  # ascii A..T
    ref_offsets = np.array([0, 200, 500], dtype=np.int64)  # 2 contigs
    c_idxs = np.array([0, 1, 0, 1], dtype=np.int64)
    starts = np.array([-5, 10, 190, 0], dtype=np.int64)  # includes OOB left
    ends = np.array([10, 30, 205, 300], dtype=np.int64)  # includes OOB right
    pad = ord("N")
    ser = _run(_fetch_impl_ser, c_idxs, starts, ends, reference, ref_offsets, pad)
    par = _run(_fetch_impl_par, c_idxs, starts, ends, reference, ref_offsets, pad)
    np.testing.assert_array_equal(ser, par)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_ref_fetch_dispatch.py -v`
Expected: FAIL — `ImportError: cannot import name '_fetch_impl_ser'`

- [ ] **Step 3: Refactor the kernel + dispatch in `fetch`**

In `python/genvarloader/_dataset/_reference.py`, add the import near the top (after the existing `from ._utils import ...` line at line 26):

```python
from .._threads import should_parallelize
```

Replace the body of `fetch` that currently reads (lines 133-143):

```python
        seqs = np.empty(offsets[-1], np.uint8)
        _fetch_impl(
            c_idxs,
            starts,
            ends,
            self.reference,
            self.offsets,
            self.pad_char,
            seqs,
            offsets,
        )
```

with:

```python
        seqs = np.empty(offsets[-1], np.uint8)
        kernel = (
            _fetch_impl_par
            if should_parallelize(int(offsets[-1]))
            else _fetch_impl_ser
        )
        kernel(
            c_idxs,
            starts,
            ends,
            self.reference,
            self.offsets,
            self.pad_char,
            seqs,
            offsets,
        )
```

Replace the existing `_fetch_impl` definition (lines 150-165):

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def _fetch_impl(
    c_idxs: NDArray[np.integer],
    starts: NDArray[np.integer],
    ends: NDArray[np.integer],
    reference: NDArray[np.integer],
    ref_offsets: NDArray[np.integer],
    pad_char: int,
    out: NDArray[np.uint8],
    out_offsets: NDArray[np.integer],
):
    for i in nb.prange(len(c_idxs)):
        r_s, r_e = ref_offsets[c_idxs[i]], ref_offsets[c_idxs[i] + 1]
        o_s, o_e = out_offsets[i], out_offsets[i + 1]
        padded_slice(reference[r_s:r_e], starts[i], ends[i], pad_char, out[o_s:o_e])
    return out
```

with a shared inner row kernel plus serial and parallel wrappers:

```python
@nb.njit(nogil=True, cache=True, inline="always")
def _fetch_row(i, c_idxs, starts, ends, reference, ref_offsets, pad_char, out, out_offsets):
    r_s, r_e = ref_offsets[c_idxs[i]], ref_offsets[c_idxs[i] + 1]
    o_s, o_e = out_offsets[i], out_offsets[i + 1]
    padded_slice(reference[r_s:r_e], starts[i], ends[i], pad_char, out[o_s:o_e])


@nb.njit(parallel=True, nogil=True, cache=True)
def _fetch_impl_par(
    c_idxs, starts, ends, reference, ref_offsets, pad_char, out, out_offsets
):
    for i in nb.prange(len(c_idxs)):
        _fetch_row(
            i, c_idxs, starts, ends, reference, ref_offsets, pad_char, out, out_offsets
        )
    return out


@nb.njit(nogil=True, cache=True)
def _fetch_impl_ser(
    c_idxs, starts, ends, reference, ref_offsets, pad_char, out, out_offsets
):
    for i in range(len(c_idxs)):
        _fetch_row(
            i, c_idxs, starts, ends, reference, ref_offsets, pad_char, out, out_offsets
        )
    return out
```

- [ ] **Step 4: Run the new test + the existing fetch-backed tests**

Run: `pixi run -e dev pytest tests/unit/dataset/test_ref_fetch_dispatch.py tests/dataset/test_flat_flanks.py -v`
Expected: PASS (new kernel-agreement test + all existing flank/window oracle tests, which call `Reference.fetch`).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reference.py tests/unit/dataset/test_ref_fetch_dispatch.py
rtk git commit -m "perf(reference): dispatch fetch kernel serial/parallel by per-thread bytes (#221)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Serial/parallel dispatch for `get_reference`

**Files:**
- Modify: `python/genvarloader/_dataset/_reference.py:669-684` (`get_reference`)
- Test: `tests/unit/dataset/test_ref_fetch_dispatch.py:1` (extend)

`get_reference` is the kernel used by `RefDataset` / `Ref` (reference & haplotype output modes) and has the identical `parallel=True` pathology. It currently allocates `out` *inside* the njit; we move allocation into a thin Python wrapper so dispatch can pick the kernel. Callers (`_reference.py` `_getitem_unspliced`, `_fetch_spliced_ref`; `_ref.py:70`) call `get_reference(regions=..., out_offsets=..., reference=..., ref_offsets=..., pad_char=...)` and use the returned array — the wrapper keeps that exact signature and return.

- [ ] **Step 1: Write the failing test (extend the dispatch test file)**

Append to `tests/unit/dataset/test_ref_fetch_dispatch.py`:

```python
from genvarloader._dataset._reference import (
    _get_reference_ser,
    _get_reference_par,
)


def test_get_reference_kernels_agree():
    rng = np.random.default_rng(1)
    reference = rng.integers(65, 85, size=500, dtype=np.uint8)
    ref_offsets = np.array([0, 200, 500], dtype=np.int64)
    # regions: (c_idx, start, end, strand)
    regions = np.array(
        [[0, -5, 10, 1], [1, 10, 30, 1], [0, 190, 205, 1], [1, 0, 300, 1]],
        dtype=np.int64,
    )
    out_offsets = lengths_to_offsets(regions[:, 2] - regions[:, 1])
    pad = ord("N")
    ser = np.empty(int(out_offsets[-1]), np.uint8)
    par = np.empty(int(out_offsets[-1]), np.uint8)
    _get_reference_ser(regions, out_offsets, reference, ref_offsets, pad, ser)
    _get_reference_par(regions, out_offsets, reference, ref_offsets, pad, par)
    np.testing.assert_array_equal(ser, par)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_ref_fetch_dispatch.py::test_get_reference_kernels_agree -v`
Expected: FAIL — `ImportError: cannot import name '_get_reference_ser'`

- [ ] **Step 3: Refactor `get_reference`**

Replace the existing definition (lines 669-684):

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def get_reference(
    regions: NDArray[np.integer],
    out_offsets: NDArray[np.integer],
    reference: NDArray[np.integer],
    ref_offsets: NDArray[np.integer],
    pad_char: int,
) -> NDArray[np.uint8]:
    out = np.empty(out_offsets[-1], np.uint8)
    for i in nb.prange(len(regions)):
        o_s, o_e = out_offsets[i], out_offsets[i + 1]
        c_idx, start, end = regions[i, :3]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        padded_slice(reference[c_s:c_e], start, end, pad_char, out[o_s:o_e])
    return out
```

with an inner row kernel, serial/parallel wrappers, and a Python dispatch wrapper of the same name and signature:

```python
@nb.njit(nogil=True, cache=True, inline="always")
def _get_reference_row(i, regions, out_offsets, reference, ref_offsets, pad_char, out):
    o_s, o_e = out_offsets[i], out_offsets[i + 1]
    c_idx, start, end = regions[i, 0], regions[i, 1], regions[i, 2]
    c_s = ref_offsets[c_idx]
    c_e = ref_offsets[c_idx + 1]
    padded_slice(reference[c_s:c_e], start, end, pad_char, out[o_s:o_e])


@nb.njit(parallel=True, nogil=True, cache=True)
def _get_reference_par(regions, out_offsets, reference, ref_offsets, pad_char, out):
    for i in nb.prange(len(regions)):
        _get_reference_row(i, regions, out_offsets, reference, ref_offsets, pad_char, out)
    return out


@nb.njit(nogil=True, cache=True)
def _get_reference_ser(regions, out_offsets, reference, ref_offsets, pad_char, out):
    for i in range(len(regions)):
        _get_reference_row(i, regions, out_offsets, reference, ref_offsets, pad_char, out)
    return out


def get_reference(
    regions: NDArray[np.integer],
    out_offsets: NDArray[np.integer],
    reference: NDArray[np.integer],
    ref_offsets: NDArray[np.integer],
    pad_char: int,
) -> NDArray[np.uint8]:
    out = np.empty(out_offsets[-1], np.uint8)
    kernel = (
        _get_reference_par
        if should_parallelize(int(out_offsets[-1]))
        else _get_reference_ser
    )
    return kernel(regions, out_offsets, reference, ref_offsets, pad_char, out)
```

- [ ] **Step 4: Run the dispatch test + the reference-dataset tests**

Run: `pixi run -e dev pytest tests/unit/dataset/test_ref_fetch_dispatch.py tests/unit/dataset/test_ref_ds.py -v`
Expected: PASS (kernel-agreement + existing `RefDataset` behavior unchanged).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reference.py tests/unit/dataset/test_ref_fetch_dispatch.py
rtk git commit -m "perf(reference): dispatch get_reference kernel serial/parallel (#221)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Fuse the 3 flank fetches into 1 (internal to `_flat_flanks`)

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_flanks.py:30-67` (`compute_flank_tokens`), `:118-148` (`compute_alt_window`), `:168-196` (`compute_windows`)
- Test: `tests/dataset/test_flat_flanks.py` (existing oracle tests are the guard — no edits needed)

The fetch fusion is purely internal: each function does one `[start−L, end+L)` read and derives `f5`/`f3` by slicing. Signatures stay identical, so the existing `_oracle_*` tests (which compute the old separate-fetch result) verify byte-identity for free. `compute_ref_window` already does a single read — leave it unchanged.

- [ ] **Step 1: Run the existing oracle tests to confirm the current green baseline**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py -k "unit or split or oracle" -v`
Expected: PASS (these are the byte-identical guards we must keep green).

- [ ] **Step 2: Add the shared slice helper**

In `python/genvarloader/_dataset/_flat_flanks.py`, add after `build_token_lut` (after line 27):

```python
def _slice_flanks(data: NDArray[np.uint8], rw_off: NDArray[np.int64], flank_len: int):
    """Derive per-variant (f5, f3) flanks from a contiguous ref-window read
    ``rw = [start-L, end+L)``. ``f5`` = first ``L`` bytes of each row, ``f3`` =
    last ``L``. Byte-identical to fetching ``[start-L, start)`` / ``[end, end+L)``
    separately: rows are always ``ref_len + 2L >= 2L + 1`` long so the two
    fixed-``L`` windows never overlap, and ``padded_slice`` pads OOB by absolute
    coordinate so boundary padding matches. Both returned arrays are ``(n, L)``.
    """
    cols = np.arange(flank_len)
    f5 = data[rw_off[:-1, None] + cols]
    f3 = data[rw_off[1:, None] - flank_len + cols]
    return f5, f3
```

- [ ] **Step 3: Rewrite `compute_flank_tokens` to a single fetch**

Replace the body of `compute_flank_tokens` after its docstring (current lines 57-67):

```python
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    n = starts.shape[0]
    f5 = f5.reshape(n, flank_len)
    f3 = f3.reshape(n, flank_len)
    flank_bytes = np.concatenate([f5, f3], axis=1)  # (n, 2L)
    tokens = lut[flank_bytes]  # vectorized 256-LUT gather -> lut.dtype
    return tokens.reshape(-1), np.asarray(row_offsets, np.int64)
```

with:

```python
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    f5, f3 = _slice_flanks(
        rw.data.view(np.uint8), np.asarray(rw.offsets, np.int64), flank_len
    )  # each (n, L)
    flank_bytes = np.concatenate([f5, f3], axis=1)  # (n, 2L)
    tokens = lut[flank_bytes]  # vectorized 256-LUT gather -> lut.dtype
    return tokens.reshape(-1), np.asarray(row_offsets, np.int64)
```

- [ ] **Step 4: Rewrite `compute_alt_window` to a single fetch**

Replace the body of `compute_alt_window` after its docstring (current lines 130-148):

```python
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    alt_bytes, alt_off = _assemble_alt_windows(
        np.ascontiguousarray(f5),
        np.ascontiguousarray(f3),
        np.asarray(alt_data, np.uint8),
        np.asarray(alt_seq_off, np.int64),
        flank_len,
    )
    alt_tok = lut[alt_bytes]
    return _FlatWindow(
        alt_tok,
        alt_off,
        np.asarray(row_offsets, np.int64),
        (None,),
    )
```

with:

```python
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    f5, f3 = _slice_flanks(
        rw.data.view(np.uint8), np.asarray(rw.offsets, np.int64), flank_len
    )  # each (n, L)
    alt_bytes, alt_off = _assemble_alt_windows(
        np.ascontiguousarray(f5).reshape(-1),
        np.ascontiguousarray(f3).reshape(-1),
        np.asarray(alt_data, np.uint8),
        np.asarray(alt_seq_off, np.int64),
        flank_len,
    )
    alt_tok = lut[alt_bytes]
    return _FlatWindow(
        alt_tok,
        alt_off,
        np.asarray(row_offsets, np.int64),
        (None,),
    )
```

- [ ] **Step 5: Rewrite `compute_windows` to a single fetch (the hot path)**

Replace the entire `compute_windows` function (current lines 168-196) with a version that fetches once and builds both windows from the shared read:

```python
def compute_windows(
    reference,
    v_contigs: NDArray[np.integer],
    starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    alt_data: NDArray[np.uint8],
    alt_seq_off: NDArray[np.int64],
    flank_len: int,
    lut: NDArray,
    row_offsets: NDArray[np.int64],
) -> tuple["_FlatWindow", "_FlatWindow"]:
    """ref_window = [start-L, end+L) read; alt_window = flank5 . alt . flank3.

    Single fused fetch: read the ref window once and derive the alt-window flanks
    by slicing it, instead of the previous 3 separate ``reference.fetch`` calls.
    Byte-identical to ``(compute_ref_window, compute_alt_window)``.
    """
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    data = rw.data.view(np.uint8)
    rw_off = np.asarray(rw.offsets, np.int64)
    row_off = np.asarray(row_offsets, np.int64)

    # ref window: tokenize the contiguous read directly.
    ref_w = _FlatWindow(lut[data], rw_off, row_off, (None,))

    # alt window: flank5 . alt . flank3 from sliced flanks.
    f5, f3 = _slice_flanks(data, rw_off, flank_len)
    alt_bytes, alt_off = _assemble_alt_windows(
        np.ascontiguousarray(f5).reshape(-1),
        np.ascontiguousarray(f3).reshape(-1),
        np.asarray(alt_data, np.uint8),
        np.asarray(alt_seq_off, np.int64),
        flank_len,
    )
    alt_w = _FlatWindow(lut[alt_bytes], alt_off, row_off, (None,))
    return ref_w, alt_w
```

- [ ] **Step 6: Run the full flank/window test module**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py -v`
Expected: PASS — every existing oracle/split/end-to-end test stays green, confirming byte-identity after fusion.

- [ ] **Step 7: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_flanks.py
rtk git commit -m "perf(flanks): fuse 3 ref-window fetches into 1 via flank slicing (#221)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Route the both-window decode through `compute_windows`

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py:762-817` (variant-windows branch)
- Test: `tests/dataset/test_flat_flanks.py` (add a fetch-count regression test)

The production hot path is `ref=window, alt=window`. Today the caller invokes `compute_ref_window` (1 fetch) and `compute_alt_window` (now 1 fetch) separately = 2 fetches. Routing the both-window case through the fused `compute_windows` makes it a single fetch.

- [ ] **Step 1: Write the failing fetch-count test**

Append to `tests/dataset/test_flat_flanks.py`:

```python
def test_variant_windows_single_fetch_per_decode(snap_dataset, monkeypatch):
    """ref=window, alt=window decode must call Reference.fetch exactly once."""
    import genvarloader._dataset._reference as refmod
    from genvarloader._dataset._flat_variants import VarWindowOpt

    calls = {"n": 0}
    orig = refmod.Reference.fetch

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(refmod.Reference, "fetch", spy)

    ds = (
        snap_dataset.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    calls["n"] = 0
    out = ds[[0, 1, 2], [0, 1, 2]]
    assert out.ref_window is not None and out.alt_window is not None
    assert calls["n"] == 1, (
        f"expected 1 reference.fetch for both-window decode, got {calls['n']}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py::test_variant_windows_single_fetch_per_decode -v`
Expected: FAIL — `assert 2 == 1` (currently two separate fetches in the both-window branch).

- [ ] **Step 3: Restructure the variant-windows branch**

In `python/genvarloader/_dataset/_flat_variants.py`, update the import inside the branch (currently lines 762-766):

```python
        from ._flat_flanks import (
            compute_alt_window,
            compute_ref_window,
            tokenize_alleles,
        )
```

to add `compute_windows`:

```python
        from ._flat_flanks import (
            compute_alt_window,
            compute_ref_window,
            compute_windows,
            tokenize_alleles,
        )
```

Then replace the ref/alt construction block (current lines 779-810):

```python
        if opt.ref == "window":
            rw = compute_ref_window(
                haps.reference, v_contigs, starts_v, ilens_v, L, lut, row_offsets
            )
            rw.shape = wshape
            win.ref_window = rw
        else:  # "allele": bare tokenized ref allele
            ref_bytes = np.asarray(haps.variants.ref.data).view(np.uint8)
            ref_off = np.asarray(haps.variants.ref.offsets, np.int64)
            ref_data, ref_seq_off = _gather_alleles(v_idxs, ref_bytes, ref_off)
            rw = tokenize_alleles(ref_data, ref_seq_off, lut, row_offsets)
            rw.shape = wshape
            win.ref = rw

        if opt.alt == "window":
            aw = compute_alt_window(
                haps.reference,
                v_contigs,
                starts_v,
                ilens_v,
                alt_data,
                alt_seq_off,
                L,
                lut,
                row_offsets,
            )
            aw.shape = wshape
            win.alt_window = aw
        else:  # "allele": bare tokenized alt allele
            aw = tokenize_alleles(alt_data, alt_seq_off, lut, row_offsets)
            aw.shape = wshape
            win.alt = aw
```

with a version that fuses the common both-window case into one fetch:

```python
        if opt.ref == "window" and opt.alt == "window":
            # Hot path: single fused fetch produces both windows.
            rw, aw = compute_windows(
                haps.reference,
                v_contigs,
                starts_v,
                ilens_v,
                alt_data,
                alt_seq_off,
                L,
                lut,
                row_offsets,
            )
            rw.shape = wshape
            aw.shape = wshape
            win.ref_window = rw
            win.alt_window = aw
        else:
            if opt.ref == "window":
                rw = compute_ref_window(
                    haps.reference, v_contigs, starts_v, ilens_v, L, lut, row_offsets
                )
                rw.shape = wshape
                win.ref_window = rw
            else:  # "allele": bare tokenized ref allele
                ref_bytes = np.asarray(haps.variants.ref.data).view(np.uint8)
                ref_off = np.asarray(haps.variants.ref.offsets, np.int64)
                ref_data, ref_seq_off = _gather_alleles(v_idxs, ref_bytes, ref_off)
                rw = tokenize_alleles(ref_data, ref_seq_off, lut, row_offsets)
                rw.shape = wshape
                win.ref = rw

            if opt.alt == "window":
                aw = compute_alt_window(
                    haps.reference,
                    v_contigs,
                    starts_v,
                    ilens_v,
                    alt_data,
                    alt_seq_off,
                    L,
                    lut,
                    row_offsets,
                )
                aw.shape = wshape
                win.alt_window = aw
            else:  # "allele": bare tokenized alt allele
                aw = tokenize_alleles(alt_data, alt_seq_off, lut, row_offsets)
                aw.shape = wshape
                win.alt = aw
```

- [ ] **Step 4: Run the fetch-count test + the full window matrix**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py -v`
Expected: PASS — the new single-fetch test plus the `test_variant_windows_matrix_fields` parametrization (all 4 ref/alt combos) and the dummy-fill / oracle tests.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py tests/dataset/test_flat_flanks.py
rtk git commit -m "perf(variant-windows): single fused fetch for both-window decode (#221)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the variant/reference test surface**

Run: `pixi run -e dev pytest tests/dataset/ tests/unit/dataset/ tests/unit/test_threads.py -v`
Expected: PASS (no regressions across flat-variants, flanks, windows, reference-dataset, threads).

- [ ] **Step 2: Lint + typecheck**

Run: `pixi run -e dev ruff check python/ && pixi run -e dev typecheck`
Expected: clean (no new findings in the touched files).

- [ ] **Step 3: Sanity-check the perf win locally (optional, not a CI gate)**

Run a quick timing of a both-window decode on the snapshot dataset and confirm `Reference.fetch` is called once and returns promptly (no ~37 ms floor). This is an informal check — production confirmation happens against `gvf-germ-som` per #221's acceptance criteria.

---

## Self-Review

**Spec coverage:**
- Thread resolution + cap (spec §1) → Task 1 (`_threads.py`, `__init__` wiring).
- Shared thread-aware dispatch (spec §2) → Task 1 (`should_parallelize`) + Tasks 2–3 (kernel refactor + dispatch for both `_fetch_impl` and `get_reference`).
- Fetch fusion 3→1 (spec §3) → Tasks 4–5 (internal slice fusion + caller routing).
- Testing (spec §4): byte-identical → existing oracle tests in Task 4/6 (kept green) + kernel-agreement tests (Tasks 2–3); fetch-count spy → Task 5; dispatcher decision → Task 1; thread resolution → Task 1. All covered.
- Scope: only `_fetch_impl`, `get_reference`, the three `_flat_flanks` builders, the `_flat_variants` caller, and the new `_threads` module are touched. `Reference.fetch` public signature unchanged. No dedup, no full LUT-kernel fusion. Matches spec scope.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code; every run step shows the command and expected result.

**Type/signature consistency:** `_threads.should_parallelize` / `cap_numba_threads` / `_resolve_num_threads` / `_MIN_BYTES_PER_THREAD` named identically across Task 1 and their callers in Tasks 2–3. `_fetch_impl_ser`/`_fetch_impl_par`/`_fetch_row` and `_get_reference_ser`/`_get_reference_par`/`_get_reference_row` used consistently between definition and tests. `_slice_flanks(data, rw_off, flank_len) -> (f5, f3)` signature matches all three call sites in Task 4. `compute_windows(...) -> (ref_w, alt_w)` signature unchanged from the existing function and matches the Task 5 call site.
