# Rust Migration Phase 2 — Genotype Assembly + Variant Gather Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the live genotype assembly/selection kernels (`get_diffs_sparse`, `choose_exonic_variants`) and the 7 flat variant-gather kernels from numba to the Rust crate, delete the dead `filter_af` kernel, with byte-identical parity and no `__getitem__` throughput regression.

**Architecture:** Pure-`ndarray` cores in new `src/genotypes/` and `src/variants/` domain modules; PyO3 wrappers live only in `src/ffi/`; Python dispatches per-kernel through `genvarloader._dispatch` (default `rust`, `GVL_BACKEND` override). The numba impls are retained as registered parity references (the registry + numba refs are deleted wholesale in Phase 5, per `_dispatch.py`); only the dead `filter_af` is removed now.

**Tech Stack:** Rust (`ndarray`, PyO3/`numpy`, `maturin`), Python 3.10–3.13, numba (reference impls), pytest + `hypothesis` (parity gates), `cargo test` (unit gates), `pixi` (env/tasks).

## Global Constraints

- Byte-identical parity is the landing gate for every ported kernel — `np.testing.assert_array_equal`, matching dtype AND shape, across the py310–313 × linux/macOS matrix.
- abi3 wheels must keep building (standing CI invariant) — `pixi run -e dev` build must succeed after each Rust change.
- `src/ffi/` is the ONLY place new kernels touch PyO3; cores are pure `ndarray`.
- Both `geno_offsets` forms must be supported: 1-D `(n+1,)` contiguous and 2-D `(2, n)` starts/stops. Normalize to `(2, n)` int64 in the Python dispatch wrapper so both backends receive identical bytes (the numba kernels already branch on `.ndim`; feeding them the 2-D form takes their existing 2-D path).
- Sequential Rust (no rayon) — per-`(query, hap)` writes are disjoint, so sequential output equals numba's `prange` output; only add rayon if the no-regression gate forces it.
- Gate = parity + no regression (NOT a required speedup). Baselines on `chr22_geuv`: haplotypes **123.9 batch/s**, variants **145.3 batch/s**.
- Conventional-commit messages; end every commit message with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- Run Rust tests via `pixi run -e dev cargo-test`; Python parity via `pixi run -e dev pytest tests/parity -q` (parity tests are marked `@pytest.mark.parity`).
- Use `rtk`-prefixed git commands per repo convention.

## File Structure

**Create:**
- `src/genotypes/mod.rs` — pure-`ndarray` cores: `get_diffs_sparse`, `choose_exonic_variants`.
- `src/variants/mod.rs` — pure-`ndarray` cores: `gather_v_idxs`, `gather_v_idxs_ss`, `gather_alleles`, `compact_keep`, `fill_empty_scalar`, `fill_empty_seq`, `fill_empty_fixed`.
- `tests/parity/test_get_diffs_sparse_parity.py`
- `tests/parity/test_choose_exonic_variants_parity.py`
- `tests/parity/test_flat_variants_parity.py`
- `tests/parity/test_variants_dataset_parity.py` — variants-mode dataset-level backstop.

**Modify:**
- `src/lib.rs` — `pub mod genotypes; pub mod variants;` + register new `ffi::*` pyfunctions.
- `src/ffi/mod.rs` — PyO3 wrappers for all 9 ported kernels.
- `python/genvarloader/_dataset/_genotypes.py` — rename numba impls to `_*_numba`, add Rust imports, `register(...)`, and dispatching public wrappers; delete `filter_af`.
- `python/genvarloader/_dataset/_flat_variants.py` — rename 7 numba kernels to `_*_numba`, add Rust imports, `register(...)`, route internal call sites through `_dispatch.get(...)`.
- `tests/parity/strategies.py` — new contract-valid generators per kernel.
- `docs/roadmaps/rust-migration.md` — Phase 2 status, double-count fix, decisions log, measurements.

**Reference only (do not edit logic):**
- `python/genvarloader/_dataset/_intervals.py` — the canonical dispatch/register/route pattern (Phase 0).
- `src/intervals.rs` — the canonical core + cargo-test pattern.
- `tests/parity/_harness.py`, `tests/parity/test_intervals_to_tracks_parity.py` — harness usage.

---

### Task 1: Tuple-aware parity harness helper

The existing `assert_kernel_parity` compares a single returned array. The Phase 2 kernels return tuples (e.g. `(keep, keep_offsets)`, `(data, offsets)`). Add a tuple-aware assertion.

**Files:**
- Modify: `tests/parity/_harness.py`
- Test: `tests/parity/test_flat_variants_parity.py` (added in later tasks consumes this; a tiny smoke test here)

**Interfaces:**
- Produces: `assert_kernel_parity_tuple(name: str, *inputs) -> None` — runs both backends, asserts each returned array element is byte-identical (dtype + shape + values). Works for single-array returns too (wraps non-tuple in a 1-tuple).

- [ ] **Step 1: Write the failing test**

Create `tests/parity/test_harness_tuple.py`:

```python
import numpy as np
import pytest

from genvarloader import _dispatch
from tests.parity._harness import assert_kernel_parity_tuple

pytestmark = pytest.mark.parity


def test_tuple_helper_detects_match(monkeypatch):
    def impl(x):
        return x * 2, x + 1

    _dispatch.register("_tuple_smoke", numba=impl, rust=impl, default="rust")
    assert_kernel_parity_tuple("_tuple_smoke", np.arange(4, dtype=np.int32))


def test_tuple_helper_detects_mismatch():
    def a(x):
        return x, x

    def b(x):
        return x, x + 1

    _dispatch.register("_tuple_smoke_bad", numba=a, rust=b, default="rust")
    with pytest.raises(AssertionError):
        assert_kernel_parity_tuple("_tuple_smoke_bad", np.arange(4, dtype=np.int32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/parity/test_harness_tuple.py -q`
Expected: FAIL with `ImportError: cannot import name 'assert_kernel_parity_tuple'`.

- [ ] **Step 3: Implement the helper**

Append to `tests/parity/_harness.py`:

```python
def assert_kernel_parity_tuple(name: str, *inputs) -> None:
    """Parity for kernels that RETURN one array or a tuple of arrays.

    Normalizes a non-tuple return into a 1-tuple, then asserts each element is
    byte-identical (dtype, shape, values) between the numba and rust backends.
    """
    numba_fn, rust_fn = _dispatch.backends(name)
    got_numba = numba_fn(*inputs)
    got_rust = rust_fn(*inputs)
    if not isinstance(got_numba, tuple):
        got_numba = (got_numba,)
    if not isinstance(got_rust, tuple):
        got_rust = (got_rust,)
    assert len(got_numba) == len(got_rust), (
        f"{name}: tuple len {len(got_numba)} != {len(got_rust)}"
    )
    for i, (a, b) in enumerate(zip(got_numba, got_rust)):
        a = np.asarray(a)
        b = np.asarray(b)
        assert a.dtype == b.dtype, f"{name}[{i}]: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{name}[{i}]: shape {a.shape} != {b.shape}"
        np.testing.assert_array_equal(a, b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/parity/test_harness_tuple.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add tests/parity/_harness.py tests/parity/test_harness_tuple.py
rtk git commit -m "$(cat <<'EOF'
test(parity): tuple-aware kernel parity helper for Phase 2 kernels

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Port `get_diffs_sparse` to Rust

Per-`(query, hap)` reference-length diffs. Numba reference: `python/genvarloader/_dataset/_genotypes.py:7-109`. Three branches: empty group (→0); query-clipped path (`q_starts`/`q_ends`/`v_starts` present); keep-masked sum; plain sum.

**Files:**
- Create: `src/genotypes/mod.rs`
- Modify: `src/lib.rs`, `src/ffi/mod.rs`, `python/genvarloader/_dataset/_genotypes.py`, `tests/parity/strategies.py`
- Test: `tests/parity/test_get_diffs_sparse_parity.py`

**Interfaces:**
- Produces (Rust core): `genotypes::get_diffs_sparse(geno_offset_idx: ArrayView2<i64>, geno_v_idxs: ArrayView1<i32>, o_starts: ArrayView1<i64>, o_stops: ArrayView1<i64>, ilens: ArrayView1<i32>, keep: Option<ArrayView1<bool>>, keep_offsets: Option<ArrayView1<i64>>, q_starts: Option<ArrayView1<i32>>, q_ends: Option<ArrayView1<i32>>, v_starts: Option<ArrayView1<i32>>) -> Array2<i32>`
- Produces (Python): `get_diffs_sparse(...)` dispatching wrapper with the SAME keyword signature callers already use (`_haps.py:474`); normalizes `geno_offsets` to `(2, n)` int64 before dispatch.

- [ ] **Step 1: Write the Rust core + cargo unit tests**

Create `src/genotypes/mod.rs`:

```rust
//! Genotype assembly/selection cores (pure ndarray). PyO3 lives in `crate::ffi`.
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Per-(query, hap) reference-length diffs. Mirrors the numba
/// `get_diffs_sparse` exactly. `o_starts`/`o_stops` are the two rows of the
/// normalized (2, n) offset array: `o_s = o_starts[o_idx]`, `o_e = o_stops[o_idx]`.
/// Length sums stay far within i32 for real variants; accumulate in i64 and
/// truncate on store to mirror numpy's `int32`-slot assignment.
#[allow(clippy::too_many_arguments)]
pub fn get_diffs_sparse(
    geno_offset_idx: ArrayView2<i64>,
    geno_v_idxs: ArrayView1<i32>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    ilens: ArrayView1<i32>,
    keep: Option<ArrayView1<bool>>,
    keep_offsets: Option<ArrayView1<i64>>,
    q_starts: Option<ArrayView1<i32>>,
    q_ends: Option<ArrayView1<i32>>,
    v_starts: Option<ArrayView1<i32>>,
) -> Array2<i32> {
    let (n_queries, ploidy) = geno_offset_idx.dim();
    let mut diffs = Array2::<i32>::zeros((n_queries, ploidy));
    let has_query = q_starts.is_some() && q_ends.is_some() && v_starts.is_some();
    let has_keep = keep.is_some() && keep_offsets.is_some();

    for query in 0..n_queries {
        for hap in 0..ploidy {
            let o_idx = geno_offset_idx[[query, hap]] as usize;
            let o_s = o_starts[o_idx] as usize;
            let o_e = o_stops[o_idx] as usize;
            let n_variants = o_e - o_s;

            if n_variants == 0 {
                diffs[[query, hap]] = 0;
            } else if has_query {
                let qs = q_starts.unwrap();
                let qe = q_ends.unwrap();
                let vs = v_starts.unwrap();
                let q_start = qs[query] as i64;
                let q_end = qe[query] as i64;
                let mut ref_idx = q_start;
                let mut acc: i64 = 0;
                for v in o_s..o_e {
                    if has_keep {
                        let kp = keep.unwrap();
                        let ko = keep_offsets.unwrap();
                        let k_s = ko[query * ploidy + hap] as usize;
                        if !kp[k_s + (v - o_s)] {
                            continue;
                        }
                    }
                    let v_idx = geno_v_idxs[v] as usize;
                    let v_start = vs[v_idx] as i64;
                    let mut v_ilen = ilens[v_idx] as i64;
                    let v_end = v_start - v_ilen.min(0) + 1;
                    if v_end <= q_start {
                        continue;
                    }
                    if v_start >= q_end {
                        break;
                    }
                    if v_start >= q_start && v_start < ref_idx {
                        continue;
                    }
                    ref_idx = ref_idx.max(v_end);
                    if v_ilen < 0 {
                        v_ilen += (q_start - v_start - 1).max(0);
                    }
                    v_ilen += (v_end - q_end).max(0);
                    acc += v_ilen;
                }
                diffs[[query, hap]] = acc as i32;
            } else if has_keep {
                let kp = keep.unwrap();
                let ko = keep_offsets.unwrap();
                let k_s = ko[query * ploidy + hap] as usize;
                let mut sum: i64 = 0;
                for (j, v) in (o_s..o_e).enumerate() {
                    if kp[k_s + j] {
                        sum += ilens[geno_v_idxs[v] as usize] as i64;
                    }
                }
                diffs[[query, hap]] = sum as i32;
            } else {
                let mut sum: i64 = 0;
                for v in o_s..o_e {
                    sum += ilens[geno_v_idxs[v] as usize] as i64;
                }
                diffs[[query, hap]] = sum as i32;
            }
        }
    }
    diffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_plain_sum() {
        // 1 query, ploidy 1, two variants with ilens [-2, 3] → sum 1.
        let goi = arr2(&[[0i64]]);
        let v_idxs = arr1(&[0i32, 1]);
        let o_starts = arr1(&[0i64]);
        let o_stops = arr1(&[2i64]);
        let ilens = arr1(&[-2i32, 3]);
        let d = get_diffs_sparse(
            goi.view(), v_idxs.view(), o_starts.view(), o_stops.view(),
            ilens.view(), None, None, None, None, None,
        );
        assert_eq!(d[[0, 0]], 1);
    }

    #[test]
    fn test_empty_group_is_zero() {
        let goi = arr2(&[[0i64]]);
        let v_idxs = arr1::<i32, _>(&[]);
        let o_starts = arr1(&[0i64]);
        let o_stops = arr1(&[0i64]); // empty slice
        let ilens = arr1::<i32, _>(&[]);
        let d = get_diffs_sparse(
            goi.view(), v_idxs.view(), o_starts.view(), o_stops.view(),
            ilens.view(), None, None, None, None, None,
        );
        assert_eq!(d[[0, 0]], 0);
    }
}
```

- [ ] **Step 2: Wire the module + run cargo tests (expect them to pass)**

In `src/lib.rs` add after `pub mod ffi;` (keep alphabetical-ish with existing `pub mod` lines):

```rust
pub mod genotypes;
```

Run: `pixi run -e dev cargo-test`
Expected: PASS, including `genotypes::tests::test_plain_sum` and `test_empty_group_is_zero`.

- [ ] **Step 3: Add the PyO3 wrapper**

Append to `src/ffi/mod.rs` (add `PyReadonlyArray2`, `PyArray2`, `IntoPyArray` to the `numpy` use line as needed):

```rust
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use crate::genotypes;

/// Per-(query, hap) reference-length diffs (see `genotypes::get_diffs_sparse`).
/// `geno_offsets` is the normalized (2, n) int64 starts/stops array.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn get_diffs_sparse<'py>(
    py: Python<'py>,
    geno_offset_idx: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    geno_offsets: PyReadonlyArray2<i64>,
    ilens: PyReadonlyArray1<i32>,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    q_starts: Option<PyReadonlyArray1<i32>>,
    q_ends: Option<PyReadonlyArray1<i32>>,
    v_starts: Option<PyReadonlyArray1<i32>>,
) -> Bound<'py, PyArray2<i32>> {
    let go = geno_offsets.as_array();
    let diffs = genotypes::get_diffs_sparse(
        geno_offset_idx.as_array(),
        geno_v_idxs.as_array(),
        go.row(0),
        go.row(1),
        ilens.as_array(),
        keep.as_ref().map(|a| a.as_array()),
        keep_offsets.as_ref().map(|a| a.as_array()),
        q_starts.as_ref().map(|a| a.as_array()),
        q_ends.as_ref().map(|a| a.as_array()),
        v_starts.as_ref().map(|a| a.as_array()),
    );
    diffs.into_pyarray(py)
}
```

Register it in `src/lib.rs` inside `fn genvarloader(...)`:

```rust
    m.add_function(wrap_pyfunction!(ffi::get_diffs_sparse, m)?)?;
```

Run: `pixi run -e dev cargo-test`
Expected: PASS (compiles + builds the extension).

- [ ] **Step 4: Add the Python dispatch wrapper**

In `python/genvarloader/_dataset/_genotypes.py`:

1. At top, add imports:

```python
from .._dispatch import get, register
from ..genvarloader import get_diffs_sparse as _get_diffs_sparse_rust
```

2. Rename the existing `@nb.njit ... def get_diffs_sparse(` to `def _get_diffs_sparse_numba(` (leave the body untouched — it already handles the 2-D `geno_offsets` branch).

3. Add a normalization helper + register + public wrapper after the numba def:

```python
def _as_starts_stops(offsets: NDArray[np.integer]) -> NDArray[np.int64]:
    """Normalize 1-D (n+1,) or 2-D (2, n) offsets to a contiguous (2, n) int64
    starts/stops array. Both backends consume this single form."""
    o = np.asarray(offsets)
    if o.ndim == 1:
        return np.ascontiguousarray(np.stack([o[:-1], o[1:]]), dtype=np.int64)
    return np.ascontiguousarray(o, dtype=np.int64)


register(
    "get_diffs_sparse",
    numba=_get_diffs_sparse_numba,
    rust=_get_diffs_sparse_rust,
    default="rust",
)


def get_diffs_sparse(
    geno_offset_idx: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    ilens: NDArray[np.integer],
    keep: NDArray[np.bool_] | None = None,
    keep_offsets: NDArray[np.integer] | None = None,
    q_starts: NDArray[np.integer] | None = None,
    q_ends: NDArray[np.integer] | None = None,
    v_starts: NDArray[np.integer] | None = None,
) -> NDArray[np.int32]:
    """Per-(query, hap) reference-length diffs; dispatches numba/rust."""
    return get("get_diffs_sparse")(
        np.ascontiguousarray(geno_offset_idx, np.int64),
        np.ascontiguousarray(geno_v_idxs, np.int32),
        _as_starts_stops(geno_offsets),
        np.ascontiguousarray(ilens, np.int32),
        None if keep is None else np.ascontiguousarray(keep, np.bool_),
        None if keep_offsets is None else np.ascontiguousarray(keep_offsets, np.int64),
        None if q_starts is None else np.ascontiguousarray(q_starts, np.int32),
        None if q_ends is None else np.ascontiguousarray(q_ends, np.int32),
        None if v_starts is None else np.ascontiguousarray(v_starts, np.int32),
    )
```

Note: callers in `_haps.py` use keyword args; the wrapper keeps the same keyword names so no call-site edits are required. The numba reference is invoked positionally by the dispatch wrapper, so `_get_diffs_sparse_numba` must accept these args positionally in this exact order (it already does).

- [ ] **Step 5: Add the parity strategy**

Append to `tests/parity/strategies.py`:

```python
@st.composite
def _sparse_geno(draw, max_queries=4, max_ploidy=2, max_vars_per_group=5,
                 max_total_unique=12):
    """Shared sparse-genotype layout: returns
    (geno_offset_idx (q,p) int64, geno_v_idxs int32, geno_offsets (n+1,) int64,
     v_starts int32, ilens int32, q_starts int32, q_ends int32).
    geno_offset_idx is arange so each (q,p) row maps to its own offset slice."""
    n_unique = draw(st.integers(min_value=1, max_value=max_total_unique))
    v_starts = np.sort(
        draw(st.lists(st.integers(0, 1000), min_size=n_unique, max_size=n_unique)
             .map(np.array))
    ).astype(np.int32)
    ilens = np.array(
        draw(st.lists(st.integers(-5, 5), min_size=n_unique, max_size=n_unique)),
        dtype=np.int32,
    )
    n_q = draw(st.integers(1, max_queries))
    p = draw(st.integers(1, max_ploidy))
    n_groups = n_q * p
    counts = [draw(st.integers(0, max_vars_per_group)) for _ in range(n_groups)]
    v_idx_list = []
    for c in counts:
        # sorted variant indices within a group (reconstruction assumes sorted pos)
        idxs = sorted(draw(st.lists(st.integers(0, n_unique - 1),
                                    min_size=c, max_size=c)))
        v_idx_list.extend(idxs)
    geno_v_idxs = np.array(v_idx_list, dtype=np.int32)
    geno_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    geno_offset_idx = np.arange(n_groups, dtype=np.int64).reshape(n_q, p)
    q_starts = np.array(
        draw(st.lists(st.integers(0, 800), min_size=n_q, max_size=n_q)), np.int32
    )
    q_ends = (q_starts + draw(st.integers(1, 200))).astype(np.int32)
    return (geno_offset_idx, geno_v_idxs, geno_offsets, v_starts, ilens,
            q_starts, q_ends)


@st.composite
def get_diffs_sparse_inputs(draw):
    (goi, gvi, goff, vstarts, ilens, qstarts, qends) = draw(_sparse_geno(draw))
    mode = draw(st.sampled_from(["plain", "keep", "query"]))
    twod = draw(st.booleans())
    offsets = goff if not twod else np.stack([goff[:-1], goff[1:]]).astype(np.int64)
    n_groups = goi.size
    total = int(goff[-1])
    if mode == "plain":
        return (goi, gvi, offsets, ilens, None, None, None, None, None)
    if mode == "keep":
        keep = np.array(
            draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_
        )
        return (goi, gvi, offsets, ilens, keep, goff.copy(), None, None, None)
    # query mode (optionally also keep)
    keep = None
    keep_off = None
    if draw(st.booleans()):
        keep = np.array(
            draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_
        )
        keep_off = goff.copy()
    return (goi, gvi, offsets, ilens, keep, keep_off, qstarts, qends, vstarts)
```

- [ ] **Step 6: Write the parity test**

Create `tests/parity/test_get_diffs_sparse_parity.py`:

```python
import pytest
from hypothesis import given

from genvarloader._dataset import _genotypes  # noqa: F401  (import triggers register())
from tests.parity._harness import assert_kernel_parity_tuple
from tests.parity.strategies import get_diffs_sparse_inputs

pytestmark = pytest.mark.parity


@given(get_diffs_sparse_inputs())
def test_get_diffs_sparse_parity(inputs):
    # The public wrapper normalizes offsets; here we call the registered
    # backends directly through the wrapper's dispatch name with the wrapper's
    # already-normalized (2, n) form, so feed normalized inputs.
    from genvarloader._dataset._genotypes import _as_starts_stops
    import numpy as np

    goi, gvi, offsets, ilens, keep, keep_off, qs, qe, vs = inputs
    norm = (
        np.ascontiguousarray(goi, np.int64),
        np.ascontiguousarray(gvi, np.int32),
        _as_starts_stops(offsets),
        np.ascontiguousarray(ilens, np.int32),
        None if keep is None else np.ascontiguousarray(keep, np.bool_),
        None if keep_off is None else np.ascontiguousarray(keep_off, np.int64),
        None if qs is None else np.ascontiguousarray(qs, np.int32),
        None if qe is None else np.ascontiguousarray(qe, np.int32),
        None if vs is None else np.ascontiguousarray(vs, np.int32),
    )
    assert_kernel_parity_tuple("get_diffs_sparse", *norm)
```

- [ ] **Step 7: Run parity + cargo, verify green**

Run: `pixi run -e dev pytest tests/parity/test_get_diffs_sparse_parity.py -q`
Expected: PASS (100 hypothesis examples).
Run: `pixi run -e dev cargo-test`
Expected: PASS.

- [ ] **Step 8: Smoke the live read path**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q -k "hap or splice or exon"`
Expected: PASS (haplotype/exonic paths still produce correct output through the new wrapper).

- [ ] **Step 9: Commit**

```bash
rtk git add src/genotypes/mod.rs src/lib.rs src/ffi/mod.rs python/genvarloader/_dataset/_genotypes.py tests/parity/strategies.py tests/parity/test_get_diffs_sparse_parity.py
rtk git commit -m "$(cat <<'EOF'
perf(genotypes): port get_diffs_sparse numba->rust (parity-gated)

Pure-ndarray core in src/genotypes/, PyO3 in src/ffi/, dispatched via
_dispatch (default rust). Offsets normalized to (2,n) int64. numba retained
as parity reference.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Port `choose_exonic_variants` to Rust

Keep-mask for variants fully contained in a query interval. Numba reference: `_genotypes.py:421-522` (driver `choose_exonic_variants` + inner `_choose_exonic_variants`). Returns `(keep: bool, keep_offsets: OFFSET_TYPE)`.

**Files:**
- Modify: `src/genotypes/mod.rs`, `src/lib.rs`, `src/ffi/mod.rs`, `python/genvarloader/_dataset/_genotypes.py`, `tests/parity/strategies.py`
- Test: `tests/parity/test_choose_exonic_variants_parity.py`

**Interfaces:**
- Produces (Rust core): `genotypes::choose_exonic_variants(starts: ArrayView1<i32>, ends: ArrayView1<i32>, geno_offset_idx: ArrayView2<i64>, geno_v_idxs: ArrayView1<i32>, o_starts: ArrayView1<i64>, o_stops: ArrayView1<i64>, v_starts: ArrayView1<i32>, ilens: ArrayView1<i32>) -> (Array1<bool>, Array1<i64>)`
- Produces (Python): `choose_exonic_variants(...)` wrapper, same keyword signature as the `_haps.py` call sites; returns `(keep, keep_offsets)` with `keep_offsets.dtype == np.dtype(OFFSET_TYPE)`.

- [ ] **Step 1: Confirm `OFFSET_TYPE`**

Run: `pixi run -e dev python -c "from seqpro.rag import OFFSET_TYPE; import numpy as np; print(np.dtype(OFFSET_TYPE))"`
Expected: prints `int64`. If it is NOT int64, adjust the Rust return element + ffi `PyArray1<...>` accordingly and the dtype coercion in the wrapper. The rest of this task assumes int64.

- [ ] **Step 2: Write the Rust core + cargo test**

Append to `src/genotypes/mod.rs`:

```rust
/// Keep-mask for variants fully contained in each query interval. Mirrors the
/// numba `choose_exonic_variants` + inner `_choose_exonic_variants`. Returns
/// `(keep, keep_offsets)` where keep_offsets is the per-group prefix sum of
/// group sizes (len n_groups + 1).
#[allow(clippy::too_many_arguments)]
pub fn choose_exonic_variants(
    starts: ArrayView1<i32>,
    ends: ArrayView1<i32>,
    geno_offset_idx: ArrayView2<i64>,
    geno_v_idxs: ArrayView1<i32>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
) -> (Array1<bool>, Array1<i64>) {
    let (n_regions, ploidy) = geno_offset_idx.dim();

    // keep_offsets = prefix sum of per-group lengths (numba uses lengths.cumsum()).
    let mut keep_offsets = Array1::<i64>::zeros(n_regions * ploidy + 1);
    let mut acc: i64 = 0;
    for query in 0..n_regions {
        for hap in 0..ploidy {
            let o_idx = geno_offset_idx[[query, hap]] as usize;
            let len = (o_stops[o_idx] - o_starts[o_idx]).max(0);
            acc += len;
            keep_offsets[query * ploidy + hap + 1] = acc;
        }
    }

    let n_variants = keep_offsets[n_regions * ploidy] as usize;
    let mut keep = Array1::<bool>::default(n_variants);

    for query in 0..n_regions {
        let ref_start = starts[query] as i64;
        let ref_end = ends[query] as i64;
        for hap in 0..ploidy {
            let o_idx = geno_offset_idx[[query, hap]] as usize;
            let o_s = o_starts[o_idx] as usize;
            let o_e = o_stops[o_idx] as usize;
            let k_s = keep_offsets[query * ploidy + hap] as usize;
            for (j, v) in (o_s..o_e).enumerate() {
                let v_idx = geno_v_idxs[v] as usize;
                let v_pos = v_starts[v_idx] as i64;
                let v_ref_end = v_pos - (ilens[v_idx] as i64).min(0) + 1;
                keep[k_s + j] = v_pos >= ref_start && v_ref_end <= ref_end;
            }
        }
    }
    (keep, keep_offsets)
}
```

Add a cargo test inside the existing `mod tests`:

```rust
    #[test]
    fn test_exonic_contained_only() {
        // region [10, 20). variants at pos 12 (ilen 0 -> end 13, kept) and
        // pos 19 (ilen 0 -> end 20, kept), pos 19 with ilen -2 -> end 22 (dropped).
        let goi = arr2(&[[0i64]]);
        let v_idxs = arr1(&[0i32, 1, 2]);
        let o_starts = arr1(&[0i64]);
        let o_stops = arr1(&[3i64]);
        let v_starts = arr1(&[12i32, 19, 19]);
        let ilens = arr1(&[0i32, 0, -2]);
        let (keep, koff) = choose_exonic_variants(
            arr1(&[10i32]).view(), arr1(&[20i32]).view(), goi.view(),
            v_idxs.view(), o_starts.view(), o_stops.view(),
            v_starts.view(), ilens.view(),
        );
        assert_eq!(keep.to_vec(), vec![true, true, false]);
        assert_eq!(koff.to_vec(), vec![0, 3]);
    }
```

- [ ] **Step 3: Run cargo tests**

Run: `pixi run -e dev cargo-test`
Expected: PASS including `test_exonic_contained_only`.

- [ ] **Step 4: Add the PyO3 wrapper + register in lib.rs**

Append to `src/ffi/mod.rs` (add `PyArray1` to the `numpy` use if not already imported):

```rust
use numpy::PyArray1;

/// Exonic keep-mask (see `genotypes::choose_exonic_variants`). Returns
/// `(keep: bool[n], keep_offsets: i64[n_groups+1])`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn choose_exonic_variants<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<i32>,
    ends: PyReadonlyArray1<i32>,
    geno_offset_idx: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    geno_offsets: PyReadonlyArray2<i64>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
) -> (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<i64>>) {
    let go = geno_offsets.as_array();
    let (keep, koff) = genotypes::choose_exonic_variants(
        starts.as_array(),
        ends.as_array(),
        geno_offset_idx.as_array(),
        geno_v_idxs.as_array(),
        go.row(0),
        go.row(1),
        v_starts.as_array(),
        ilens.as_array(),
    );
    (keep.into_pyarray(py), koff.into_pyarray(py))
}
```

Register in `src/lib.rs`:

```rust
    m.add_function(wrap_pyfunction!(ffi::choose_exonic_variants, m)?)?;
```

Run: `pixi run -e dev cargo-test`
Expected: PASS (extension builds).

- [ ] **Step 5: Add the Python dispatch wrapper**

In `_genotypes.py`:

1. Add import: `from ..genvarloader import choose_exonic_variants as _choose_exonic_variants_rust`.
2. Rename `@nb.njit ... def choose_exonic_variants(` → `def _choose_exonic_variants_numba(` (keep the inner `_choose_exonic_variants` njit as-is — it's only called by the numba driver).
3. Add register + wrapper:

```python
register(
    "choose_exonic_variants",
    numba=_choose_exonic_variants_numba,
    rust=_choose_exonic_variants_rust,
    default="rust",
)


def choose_exonic_variants(
    starts: NDArray[np.integer],
    ends: NDArray[np.integer],
    geno_offset_idx: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
) -> tuple[NDArray[np.bool_], NDArray[OFFSET_TYPE]]:
    """Exonic keep-mask; dispatches numba/rust. keep_offsets dtype == OFFSET_TYPE."""
    keep, keep_offsets = get("choose_exonic_variants")(
        np.ascontiguousarray(starts, np.int32),
        np.ascontiguousarray(ends, np.int32),
        np.ascontiguousarray(geno_offset_idx, np.int64),
        np.ascontiguousarray(geno_v_idxs, np.int32),
        _as_starts_stops(geno_offsets),
        np.ascontiguousarray(v_starts, np.int32),
        np.ascontiguousarray(ilens, np.int32),
    )
    return keep, keep_offsets.astype(OFFSET_TYPE, copy=False)
```

Note: `_choose_exonic_variants_numba` already returns `keep_offsets` as `OFFSET_TYPE`; the Rust path returns int64 and the `.astype(..., copy=False)` is a no-op when OFFSET_TYPE is int64. The parity test compares the raw backend returns (both int64) BEFORE this astype.

- [ ] **Step 6: Add parity strategy**

Append to `tests/parity/strategies.py`:

```python
@st.composite
def choose_exonic_variants_inputs(draw):
    (goi, gvi, goff, vstarts, ilens, qstarts, qends) = draw(_sparse_geno(draw))
    twod = draw(st.booleans())
    offsets = goff if not twod else np.stack([goff[:-1], goff[1:]]).astype(np.int64)
    return (qstarts, qends, goi, gvi, offsets, vstarts, ilens)
```

- [ ] **Step 7: Write parity test**

Create `tests/parity/test_choose_exonic_variants_parity.py`:

```python
import numpy as np
import pytest
from hypothesis import given

from genvarloader._dataset import _genotypes  # noqa: F401
from genvarloader._dataset._genotypes import _as_starts_stops
from tests.parity._harness import assert_kernel_parity_tuple
from tests.parity.strategies import choose_exonic_variants_inputs

pytestmark = pytest.mark.parity


@given(choose_exonic_variants_inputs())
def test_choose_exonic_variants_parity(inputs):
    qs, qe, goi, gvi, offsets, vs, ilens = inputs
    norm = (
        np.ascontiguousarray(qs, np.int32),
        np.ascontiguousarray(qe, np.int32),
        np.ascontiguousarray(goi, np.int64),
        np.ascontiguousarray(gvi, np.int32),
        _as_starts_stops(offsets),
        np.ascontiguousarray(vs, np.int32),
        np.ascontiguousarray(ilens, np.int32),
    )
    assert_kernel_parity_tuple("choose_exonic_variants", *norm)
```

- [ ] **Step 8: Run parity + cargo + exonic read path**

Run: `pixi run -e dev pytest tests/parity/test_choose_exonic_variants_parity.py -q`
Expected: PASS.
Run: `pixi run -e dev pytest tests/dataset tests/unit -q -k "exon or splice"`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
rtk git add src/genotypes/mod.rs src/lib.rs src/ffi/mod.rs python/genvarloader/_dataset/_genotypes.py tests/parity/strategies.py tests/parity/test_choose_exonic_variants_parity.py
rtk git commit -m "$(cat <<'EOF'
perf(genotypes): port choose_exonic_variants numba->rust (parity-gated)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Delete dead `filter_af`

`filter_af` (`_genotypes.py:525-580`) has zero callers — AF filtering is done inline in numpy (`_haps.py:734-737`, `_flat_variants.py:698-701`). Remove it.

**Files:**
- Modify: `python/genvarloader/_dataset/_genotypes.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (removal only).

- [ ] **Step 1: Confirm zero callers (guard against a hidden reference)**

Run: `rtk grep -rn "filter_af" . --include="*.py"`
Expected: only the definition line(s) in `_genotypes.py` and the comment at `_genotypes.py:475`. If any other reference exists, STOP and re-scope — do not delete.

- [ ] **Step 2: Delete the kernel + stale comment reference**

Remove the entire `@nb.njit ... def filter_af(...)` block (`_genotypes.py:525-580`). Update the comment at line ~475 (`# Mirror filter_af's (2, n_slices) indexing (sibling kernel below).`) to not reference the now-deleted kernel — replace with `# Handle both 1-D (n+1,) and 2-D (2, n_slices) geno_offsets forms.`

- [ ] **Step 3: Verify nothing imports it**

Run: `pixi run -e dev ruff check python/genvarloader/_dataset/_genotypes.py`
Expected: PASS (no unused/undefined-name errors).
Run: `pixi run -e dev pytest tests/dataset tests/unit -q -k "af or freq"`
Expected: PASS (AF filtering still works via the inline numpy path).

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_genotypes.py
rtk git commit -m "$(cat <<'EOF'
refactor(genotypes): delete dead filter_af kernel (superseded by inline numpy)

AF filtering happens in numpy in _haps.py/_flat_variants.py; the numba
filter_af had zero callers (same as the Phase 0 splits_sum_le_value dead path).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Port `_gather_v_idxs` + `_gather_v_idxs_ss` to Rust

Per-row variant-index gather. Numba reference: `_flat_variants.py:432-488`. Both are unified by the `(2, n)` normalization, so a single Rust core `gather_rows` suffices; the Python `_gather_rows` dispatcher (line 538) routes to it.

**Files:**
- Create: `src/variants/mod.rs`
- Modify: `src/lib.rs`, `src/ffi/mod.rs`, `python/genvarloader/_dataset/_flat_variants.py`, `tests/parity/strategies.py`
- Test: `tests/parity/test_flat_variants_parity.py`

**Interfaces:**
- Produces (Rust core): `variants::gather_rows(geno_offset_idx: ArrayView1<i64>, o_starts: ArrayView1<i64>, o_stops: ArrayView1<i64>, geno_v_idxs: ArrayView1<i32>) -> (Array1<i32>, Array1<i64>)` → `(v_idxs, out_offsets)`.
- Produces (Python): `_gather_rows(geno_offset_idx, offsets, data)` keeps its existing signature (line 538) but dispatches to the Rust/numba `gather_rows` after normalizing offsets to `(2, n)`.

Note: `geno_v_idxs` dtype — the numba kernel preserves `geno_v_idxs.dtype`. Confirm it is int32 in production (`self.genotypes.data`). The wrapper coerces to int32; if production uses a wider dtype, widen the Rust element type + ffi to match and re-confirm parity dtype.

- [ ] **Step 1: Write the Rust core + cargo test**

Create `src/variants/mod.rs`:

```rust
//! Flat variant gather/fill cores (pure ndarray). PyO3 lives in `crate::ffi`.
use ndarray::{Array1, ArrayView1};

/// Per-row variant-index gather. Mirrors numba `_gather_v_idxs` (and `_ss` via
/// the (2, n) normalized offsets). `o_s = o_starts[goi]`, `o_e = o_stops[goi]`.
pub fn gather_rows(
    geno_offset_idx: ArrayView1<i64>,
    o_starts: ArrayView1<i64>,
    o_stops: ArrayView1<i64>,
    geno_v_idxs: ArrayView1<i32>,
) -> (Array1<i32>, Array1<i64>) {
    let n_rows = geno_offset_idx.len();
    let mut out_offsets = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let goi = geno_offset_idx[i] as usize;
        out_offsets[i + 1] = out_offsets[i] + (o_stops[goi] - o_starts[goi]);
    }
    let total = out_offsets[n_rows] as usize;
    let mut v_idxs = Array1::<i32>::zeros(total);
    let mut dst = 0usize;
    for i in 0..n_rows {
        let goi = geno_offset_idx[i] as usize;
        let s = o_starts[goi] as usize;
        let e = o_stops[goi] as usize;
        for k in s..e {
            v_idxs[dst] = geno_v_idxs[k];
            dst += 1;
        }
    }
    (v_idxs, out_offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_gather_rows_basic() {
        // 2 rows selecting offset groups 1 then 0.
        let goi = arr1(&[1i64, 0]);
        let o_starts = arr1(&[0i64, 2]);
        let o_stops = arr1(&[2i64, 5]);
        let data = arr1(&[10i32, 11, 12, 13, 14]);
        let (v, off) = gather_rows(goi.view(), o_starts.view(), o_stops.view(), data.view());
        assert_eq!(v.to_vec(), vec![12, 13, 14, 10, 11]);
        assert_eq!(off.to_vec(), vec![0, 3, 5]);
    }
}
```

- [ ] **Step 2: Wire module + cargo test**

In `src/lib.rs` add `pub mod variants;`.
Run: `pixi run -e dev cargo-test`
Expected: PASS including `variants::tests::test_gather_rows_basic`.

- [ ] **Step 3: PyO3 wrapper + register**

Append to `src/ffi/mod.rs`:

```rust
use crate::variants;

/// Per-row variant-index gather (see `variants::gather_rows`).
#[pyfunction]
pub fn gather_rows<'py>(
    py: Python<'py>,
    geno_offset_idx: PyReadonlyArray1<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>) {
    let go = geno_offsets.as_array();
    let (v, off) = variants::gather_rows(
        geno_offset_idx.as_array(),
        go.row(0),
        go.row(1),
        geno_v_idxs.as_array(),
    );
    (v.into_pyarray(py), off.into_pyarray(py))
}
```

Register in `src/lib.rs`: `m.add_function(wrap_pyfunction!(ffi::gather_rows, m)?)?;`
Run: `pixi run -e dev cargo-test`
Expected: PASS.

- [ ] **Step 4: Route the Python `_gather_rows`**

In `_flat_variants.py`:

1. Add imports near the top:

```python
from .._dispatch import get, register
from ..genvarloader import gather_rows as _gather_rows_rust
from ._genotypes import _as_starts_stops
```

2. Rename the two njit defs to `_gather_v_idxs_numba` / `_gather_v_idxs_ss_numba` (keep bodies). Add a numba adapter matching the Rust ffi signature `(geno_offset_idx, geno_offsets_2d, geno_v_idxs)`:

```python
def _gather_rows_numba(geno_offset_idx, geno_offsets, geno_v_idxs):
    # geno_offsets is the normalized (2, n) form.
    return _gather_v_idxs_ss_numba(
        geno_offset_idx, geno_offsets[0], geno_offsets[1], geno_v_idxs
    )


register("gather_rows", numba=_gather_rows_numba, rust=_gather_rows_rust, default="rust")
```

3. Replace the body of the existing `_gather_rows(...)` (line 538) with:

```python
def _gather_rows(
    geno_offset_idx: NDArray[np.intp],
    offsets: NDArray[np.int64],
    data: NDArray,
) -> tuple[NDArray, NDArray[np.int64]]:
    """Dispatch per-row variant-index gather (numba/rust), normalizing offsets."""
    return get("gather_rows")(
        np.ascontiguousarray(geno_offset_idx, np.int64),
        _as_starts_stops(offsets),
        np.ascontiguousarray(data, np.int32),
    )
```

Note: keeping `_gather_v_idxs_numba`/`_gather_v_idxs_ss_numba` lets the parity test exercise the numba path; `_gather_rows_numba` is the dispatch adapter. The 2-D normalized form makes `_ss` the single numba path.

- [ ] **Step 5: Parity strategy + test (gather_rows)**

Append to `tests/parity/strategies.py`:

```python
@st.composite
def gather_rows_inputs(draw):
    n_groups = draw(st.integers(1, 6))
    counts = [draw(st.integers(0, 5)) for _ in range(n_groups)]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    data = np.array(
        draw(st.lists(st.integers(0, 1000), min_size=total, max_size=total)), np.int32
    )
    n_rows = draw(st.integers(1, 8))
    goi = np.array(
        draw(st.lists(st.integers(0, n_groups - 1), min_size=n_rows, max_size=n_rows)),
        np.int64,
    )
    twod = draw(st.booleans())
    off = offsets if not twod else np.stack([offsets[:-1], offsets[1:]]).astype(np.int64)
    return (goi, off, data)
```

Create `tests/parity/test_flat_variants_parity.py`:

```python
import numpy as np
import pytest
from hypothesis import given

from genvarloader._dataset import _flat_variants  # noqa: F401  (triggers register())
from genvarloader._dataset._genotypes import _as_starts_stops
from tests.parity._harness import assert_kernel_parity_tuple
from tests.parity.strategies import gather_rows_inputs

pytestmark = pytest.mark.parity


@given(gather_rows_inputs())
def test_gather_rows_parity(inputs):
    goi, offsets, data = inputs
    assert_kernel_parity_tuple(
        "gather_rows",
        np.ascontiguousarray(goi, np.int64),
        _as_starts_stops(offsets),
        np.ascontiguousarray(data, np.int32),
    )
```

- [ ] **Step 6: Run parity + cargo**

Run: `pixi run -e dev pytest tests/parity/test_flat_variants_parity.py -q`
Expected: PASS.
Run: `pixi run -e dev cargo-test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add src/variants/mod.rs src/lib.rs src/ffi/mod.rs python/genvarloader/_dataset/_flat_variants.py tests/parity/strategies.py tests/parity/test_flat_variants_parity.py
rtk git commit -m "$(cat <<'EOF'
perf(variants): port _gather_v_idxs(+_ss) numba->rust as gather_rows (parity)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Port `_gather_alleles` to Rust

Variable-length allele-byte gather. Numba reference: `_flat_variants.py:491-512`.

**Files:**
- Modify: `src/variants/mod.rs`, `src/lib.rs`, `src/ffi/mod.rs`, `python/genvarloader/_dataset/_flat_variants.py`, `tests/parity/strategies.py`, `tests/parity/test_flat_variants_parity.py`

**Interfaces:**
- Produces (Rust core): `variants::gather_alleles(v_idxs: ArrayView1<i32>, allele_bytes: ArrayView1<u8>, allele_offsets: ArrayView1<i64>) -> (Array1<u8>, Array1<i64>)` → `(data, seq_offsets)`.
- Produces (Python): registered as `"gather_alleles"`; call sites at `_flat_variants.py:738,749` go through `get("gather_alleles")(...)`.

- [ ] **Step 1: Rust core + cargo test**

Append to `src/variants/mod.rs`:

```rust
/// Gather variable-length allele bytestrings. Mirrors numba `_gather_alleles`.
pub fn gather_alleles(
    v_idxs: ArrayView1<i32>,
    allele_bytes: ArrayView1<u8>,
    allele_offsets: ArrayView1<i64>,
) -> (Array1<u8>, Array1<i64>) {
    let n = v_idxs.len();
    let mut seq_offsets = Array1::<i64>::zeros(n + 1);
    for i in 0..n {
        let v = v_idxs[i] as usize;
        seq_offsets[i + 1] = seq_offsets[i] + (allele_offsets[v + 1] - allele_offsets[v]);
    }
    let total = seq_offsets[n] as usize;
    let mut data = Array1::<u8>::zeros(total);
    let mut dst = 0usize;
    for i in 0..n {
        let v = v_idxs[i] as usize;
        let s = allele_offsets[v] as usize;
        let e = allele_offsets[v + 1] as usize;
        for k in s..e {
            data[dst] = allele_bytes[k];
            dst += 1;
        }
    }
    (data, seq_offsets)
}
```

Add to `mod tests`:

```rust
    #[test]
    fn test_gather_alleles_basic() {
        // alleles: v0="AC"(65,67), v1="G"(71). gather [1,0,1].
        let v_idxs = arr1(&[1i32, 0, 1]);
        let bytes = arr1(&[65u8, 67, 71]);
        let offs = arr1(&[0i64, 2, 3]);
        let (data, seq) = gather_alleles(v_idxs.view(), bytes.view(), offs.view());
        assert_eq!(data.to_vec(), vec![71, 65, 67, 71]);
        assert_eq!(seq.to_vec(), vec![0, 1, 3, 4]);
    }
```

- [ ] **Step 2: PyO3 wrapper + register**

Append to `src/ffi/mod.rs`:

```rust
/// Gather allele bytestrings (see `variants::gather_alleles`).
#[pyfunction]
pub fn gather_alleles<'py>(
    py: Python<'py>,
    v_idxs: PyReadonlyArray1<i32>,
    allele_bytes: PyReadonlyArray1<u8>,
    allele_offsets: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>) {
    let (data, seq) = variants::gather_alleles(
        v_idxs.as_array(),
        allele_bytes.as_array(),
        allele_offsets.as_array(),
    );
    (data.into_pyarray(py), seq.into_pyarray(py))
}
```

Register: `m.add_function(wrap_pyfunction!(ffi::gather_alleles, m)?)?;`
Run: `pixi run -e dev cargo-test`
Expected: PASS.

- [ ] **Step 3: Route Python + register**

In `_flat_variants.py`: add `from ..genvarloader import gather_alleles as _gather_alleles_rust`; rename njit to `_gather_alleles_numba`; add a thin dispatch wrapper named `_gather_alleles` (preserving the existing internal call name) + register:

```python
register("gather_alleles", numba=_gather_alleles_numba, rust=_gather_alleles_rust, default="rust")


def _gather_alleles(v_idxs, allele_bytes, allele_offsets):
    return get("gather_alleles")(
        np.ascontiguousarray(v_idxs, np.int32),
        np.ascontiguousarray(allele_bytes, np.uint8),
        np.ascontiguousarray(allele_offsets, np.int64),
    )
```

The existing call sites (`_gather_alleles(v_idxs, alt_bytes, alt_off)` at lines 738, 749) now resolve to this wrapper unchanged.

- [ ] **Step 4: Parity strategy + test**

Append to `tests/parity/strategies.py`:

```python
@st.composite
def gather_alleles_inputs(draw):
    n_unique = draw(st.integers(1, 8))
    lens = [draw(st.integers(0, 5)) for _ in range(n_unique)]
    allele_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    total = int(allele_offsets[-1])
    allele_bytes = np.array(
        draw(st.lists(st.integers(0, 255), min_size=total, max_size=total)), np.uint8
    )
    m = draw(st.integers(0, 10))
    v_idxs = np.array(
        draw(st.lists(st.integers(0, n_unique - 1), min_size=m, max_size=m)), np.int32
    )
    return (v_idxs, allele_bytes, allele_offsets)
```

Add to `tests/parity/test_flat_variants_parity.py`:

```python
from tests.parity.strategies import gather_alleles_inputs


@given(gather_alleles_inputs())
def test_gather_alleles_parity(inputs):
    v_idxs, allele_bytes, allele_offsets = inputs
    assert_kernel_parity_tuple(
        "gather_alleles",
        np.ascontiguousarray(v_idxs, np.int32),
        np.ascontiguousarray(allele_bytes, np.uint8),
        np.ascontiguousarray(allele_offsets, np.int64),
    )
```

- [ ] **Step 5: Run parity + cargo, commit**

Run: `pixi run -e dev pytest tests/parity/test_flat_variants_parity.py -q && pixi run -e dev cargo-test`
Expected: PASS.

```bash
rtk git add src/variants/mod.rs src/lib.rs src/ffi/mod.rs python/genvarloader/_dataset/_flat_variants.py tests/parity/strategies.py tests/parity/test_flat_variants_parity.py
rtk git commit -m "$(cat <<'EOF'
perf(variants): port _gather_alleles numba->rust (parity-gated)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Port `_compact_keep` to Rust

Drop variants where `keep` is False, rebuilding row offsets. Numba reference: `_flat_variants.py:515-535`. Note: the first param can be `v_idxs` OR a parallel array (e.g. dosage) sharing the row layout — the dtype varies (int32 for v_idxs, float for dosage). Handle both with a generic element type via two registered entry points, OR coerce in the wrapper per call site.

**Decision:** register a single `"compact_keep"` that operates on the value array as `f64`-agnostic is unsafe for int parity. Instead expose two typed cores and pick by the value array's dtype in the Python wrapper (v_idxs → int32, dosage/ccf → float32). Confirm the production dtypes first.

**Files:**
- Modify: `src/variants/mod.rs`, `src/lib.rs`, `src/ffi/mod.rs`, `python/genvarloader/_dataset/_flat_variants.py`, `tests/parity/strategies.py`, `tests/parity/test_flat_variants_parity.py`

**Interfaces:**
- Produces (Rust cores): `variants::compact_keep_i32(values: ArrayView1<i32>, row_offsets: ArrayView1<i64>, keep: ArrayView1<bool>) -> (Array1<i32>, Array1<i64>)` and `compact_keep_f32(values: ArrayView1<f32>, ...) -> (Array1<f32>, Array1<i64>)`.
- Produces (Python): `_compact_keep(v_idxs, row_offsets, keep)` wrapper dispatching by `v_idxs.dtype`.

- [ ] **Step 1: Confirm production value dtypes**

Run: `rtk grep -n "_compact_keep(" python/genvarloader/_dataset/_flat_variants.py`
Inspect each call (lines ~715, 717, 769, +1): the first arg is `v_idxs` (int32), `dosage_data` (check dtype), `cf_data` (check dtype). Run:
`rtk grep -n "dosage_data\|cf_data\|unfiltered_row_offsets" python/genvarloader/_dataset/_flat_variants.py`
Record the dtypes. If only int32 + float32 occur, the two typed cores below suffice. If another float width appears (float64), add a matching core.

- [ ] **Step 2: Rust cores + cargo test**

Append to `src/variants/mod.rs`:

```rust
/// Compact a per-variant value array + rebuild row offsets under `keep`.
/// Mirrors numba `_compact_keep`. Generic over the value element type.
fn compact_keep_impl<T: Copy + num_traits::Zero>(
    values: ArrayView1<T>,
    row_offsets: ArrayView1<i64>,
    keep: ArrayView1<bool>,
) -> (Array1<T>, Array1<i64>) {
    let n_rows = row_offsets.len() - 1;
    let mut new_offsets = Array1::<i64>::zeros(n_rows + 1);
    let mut n_keep: i64 = 0;
    for i in 0..n_rows {
        for j in row_offsets[i] as usize..row_offsets[i + 1] as usize {
            if keep[j] {
                n_keep += 1;
            }
        }
        new_offsets[i + 1] = n_keep;
    }
    let mut new_v = Array1::<T>::zeros(n_keep as usize);
    let mut dst = 0usize;
    for j in 0..values.len() {
        if keep[j] {
            new_v[dst] = values[j];
            dst += 1;
        }
    }
    (new_v, new_offsets)
}

pub fn compact_keep_i32(
    values: ArrayView1<i32>, row_offsets: ArrayView1<i64>, keep: ArrayView1<bool>,
) -> (Array1<i32>, Array1<i64>) {
    compact_keep_impl(values, row_offsets, keep)
}

pub fn compact_keep_f32(
    values: ArrayView1<f32>, row_offsets: ArrayView1<i64>, keep: ArrayView1<bool>,
) -> (Array1<f32>, Array1<i64>) {
    compact_keep_impl(values, row_offsets, keep)
}
```

If `num_traits` is not already a dependency, replace the bound with an explicit zero by parameterizing the fill: change `Array1::<T>::zeros(...)` to build from a provided zero value, or simplest — drop the generic and write two near-identical functions. Check `Cargo.toml`; if `num-traits` is absent and you prefer no new dep, duplicate the body for i32/f32.

Add a cargo test:

```rust
    #[test]
    fn test_compact_keep_i32() {
        // 2 rows: [10,11 | 12]; keep [T,F,T] → [10 | 12], offsets [0,1,2].
        let vals = arr1(&[10i32, 11, 12]);
        let off = arr1(&[0i64, 2, 3]);
        let keep = arr1(&[true, false, true]);
        let (v, o) = compact_keep_i32(vals.view(), off.view(), keep.view());
        assert_eq!(v.to_vec(), vec![10, 12]);
        assert_eq!(o.to_vec(), vec![0, 1, 2]);
    }
```

- [ ] **Step 3: PyO3 wrappers + register**

Append to `src/ffi/mod.rs` (two pyfunctions `compact_keep_i32`, `compact_keep_f32`, each `(values, row_offsets, keep) -> (PyArray1<T>, PyArray1<i64>)`), mirroring the gather wrappers. Register both in `src/lib.rs`.
Run: `pixi run -e dev cargo-test`
Expected: PASS.

- [ ] **Step 4: Route Python + register (dtype dispatch)**

In `_flat_variants.py`: import both rust fns; rename njit → `_compact_keep_numba`; add:

```python
register("compact_keep_i32", numba=_compact_keep_numba, rust=_compact_keep_i32_rust, default="rust")
register("compact_keep_f32", numba=_compact_keep_numba, rust=_compact_keep_f32_rust, default="rust")


def _compact_keep(v_idxs, row_offsets, keep):
    values = np.ascontiguousarray(v_idxs)
    row_offsets = np.ascontiguousarray(row_offsets, np.int64)
    keep = np.ascontiguousarray(keep, np.bool_)
    if np.issubdtype(values.dtype, np.floating):
        return get("compact_keep_f32")(values.astype(np.float32, copy=False), row_offsets, keep)
    return get("compact_keep_i32")(values.astype(np.int32, copy=False), row_offsets, keep)
```

If Step 1 found a float64 dosage/ccf dtype, the `.astype(np.float32)` would lose precision and break parity — in that case add a `compact_keep_f64` core/wrapper and route float64 to it instead of down-casting. The numba reference preserves the input dtype, so the parity test (which feeds the same dtype to both) will catch any mismatch.

- [ ] **Step 5: Parity strategy + test (both dtypes)**

Append to `tests/parity/strategies.py` a `compact_keep_inputs(dtype)` generator producing `(values[dtype], row_offsets int64, keep bool)`; add two parametrized tests in `test_flat_variants_parity.py` for int32 and float32 that call `assert_kernel_parity_tuple("compact_keep_i32"/"compact_keep_f32", ...)`.

```python
@st.composite
def compact_keep_inputs(draw, dtype):
    n_rows = draw(st.integers(1, 6))
    counts = [draw(st.integers(0, 5)) for _ in range(n_rows)]
    row_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(row_offsets[-1])
    if np.issubdtype(np.dtype(dtype), np.floating):
        values = np.array(
            draw(st.lists(st.floats(width=32, allow_nan=False, allow_infinity=False),
                          min_size=total, max_size=total)), dtype)
    else:
        values = np.array(
            draw(st.lists(st.integers(0, 1000), min_size=total, max_size=total)), dtype)
    keep = np.array(
        draw(st.lists(st.booleans(), min_size=total, max_size=total)), np.bool_)
    return (values, row_offsets, keep)
```

```python
from tests.parity.strategies import compact_keep_inputs


@given(compact_keep_inputs(np.int32))
def test_compact_keep_i32_parity(inputs):
    assert_kernel_parity_tuple("compact_keep_i32", *inputs)


@given(compact_keep_inputs(np.float32))
def test_compact_keep_f32_parity(inputs):
    assert_kernel_parity_tuple("compact_keep_f32", *inputs)
```

- [ ] **Step 6: Run parity + cargo, commit**

Run: `pixi run -e dev pytest tests/parity/test_flat_variants_parity.py -q && pixi run -e dev cargo-test`
Expected: PASS.

```bash
rtk git add src/variants/mod.rs src/lib.rs src/ffi/mod.rs python/genvarloader/_dataset/_flat_variants.py tests/parity/strategies.py tests/parity/test_flat_variants_parity.py Cargo.toml
rtk git commit -m "$(cat <<'EOF'
perf(variants): port _compact_keep numba->rust (i32/f32, parity-gated)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Port `_fill_empty_scalar` + `_fill_empty_fixed` to Rust

Dummy-fill for empty groups. Numba reference: `_flat_variants.py:555-576` (scalar) and `628-656` (fixed). Both insert one dummy element/variant per empty row. `_fill_empty_scalar`'s `data`/`fill` dtype varies by field (int / float). Use the same dtype-dispatch approach as Task 7.

**Files:**
- Modify: `src/variants/mod.rs`, `src/lib.rs`, `src/ffi/mod.rs`, `python/genvarloader/_dataset/_flat_variants.py`, `tests/parity/strategies.py`, `tests/parity/test_flat_variants_parity.py`

**Interfaces:**
- Produces (Rust cores): `variants::fill_empty_scalar_{i32,f32}(data, offsets, fill) -> (Array1<T>, Array1<i64>)`; `variants::fill_empty_fixed_{i32,f32}(data, offsets, inner: i64, fill) -> (Array1<T>, Array1<i64>)`. Confirm production dtypes in Step 1 (start/ilen → int; dosage → float; flank_tokens → int).
- Produces (Python): `_fill_empty_scalar(data, offsets, fill)` and `_fill_empty_fixed(data, offsets, inner, fill)` dispatch wrappers (existing names/signatures preserved — call sites at lines 314, 419, 427).

- [ ] **Step 1: Confirm field dtypes**

Run: `rtk grep -n "_fill_empty_scalar(\|_fill_empty_fixed(" python/genvarloader/_dataset/_flat_variants.py`
For each call, determine `data.dtype` (the `f.data` / `ft.data` arrays). Record which dtypes occur (expected: int32/int64 for start/ilen/flank_tokens, float32 for dosage). Add a typed core per distinct dtype; do NOT down-cast (parity).

- [ ] **Step 2: Rust cores + cargo tests**

Append to `src/variants/mod.rs` generic impls + typed wrappers:

```rust
fn fill_empty_scalar_impl<T: Copy>(
    data: ArrayView1<T>, offsets: ArrayView1<i64>, fill: T,
) -> (Array1<T>, Array1<i64>) {
    let n_rows = offsets.len() - 1;
    let mut new_offsets = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let ln = offsets[i + 1] - offsets[i];
        new_offsets[i + 1] = new_offsets[i] + if ln > 0 { ln } else { 1 };
    }
    let total = new_offsets[n_rows] as usize;
    // Fill buffer with `fill` so empty-row slots are already correct; then copy.
    let mut new_data = Array1::<T>::from_elem(total, fill);
    for i in 0..n_rows {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        let mut d = new_offsets[i] as usize;
        if e != s {
            for k in s..e {
                new_data[d] = data[k];
                d += 1;
            }
        }
    }
    (new_data, new_offsets)
}

fn fill_empty_fixed_impl<T: Copy>(
    data: ArrayView1<T>, offsets: ArrayView1<i64>, inner: i64, fill: T,
) -> (Array1<T>, Array1<i64>) {
    let n_rows = offsets.len() - 1;
    let mut new_offsets = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let nv = offsets[i + 1] - offsets[i];
        new_offsets[i + 1] = new_offsets[i] + if nv > 0 { nv } else { 1 };
    }
    let total_vars = new_offsets[n_rows] as usize;
    let inner_u = inner as usize;
    let mut new_data = Array1::<T>::from_elem(total_vars * inner_u, fill);
    let mut dptr = 0usize;
    for i in 0..n_rows {
        let vs = offsets[i] as usize;
        let ve = offsets[i + 1] as usize;
        if ve == vs {
            dptr += inner_u; // already filled
        } else {
            for k in vs * inner_u..ve * inner_u {
                new_data[dptr] = data[k];
                dptr += 1;
            }
        }
    }
    (new_data, new_offsets)
}
```

Add `_i32`/`_f32` (and any other confirmed dtype) public wrappers calling the impls, plus cargo tests asserting the empty-row insertion and pass-through for one int and one float case.

- [ ] **Step 3: PyO3 wrappers + register; Step 4: Python dtype-dispatch wrappers**

Mirror Task 7: register `"fill_empty_scalar_<dtype>"` and `"fill_empty_fixed_<dtype>"`; rename numba defs to `_*_numba`; the public `_fill_empty_scalar`/`_fill_empty_fixed` wrappers pick the entry by `data.dtype` and pass `fill` as a python scalar (PyO3 receives it as `T`). `inner` is passed as `i64`.
Run: `pixi run -e dev cargo-test`
Expected: PASS.

- [ ] **Step 5: Parity strategies + tests**

Add `fill_empty_scalar_inputs(dtype)` and `fill_empty_fixed_inputs(dtype)` generators (offsets with some empty rows guaranteed; random `fill`; `inner` 1..4 for fixed) and parametrized parity tests for each confirmed dtype in `test_flat_variants_parity.py`.

- [ ] **Step 6: Run parity + cargo, commit**

Run: `pixi run -e dev pytest tests/parity/test_flat_variants_parity.py -q && pixi run -e dev cargo-test`
Expected: PASS.

```bash
rtk git add src/variants/mod.rs src/lib.rs src/ffi/mod.rs python/genvarloader/_dataset/_flat_variants.py tests/parity/strategies.py tests/parity/test_flat_variants_parity.py
rtk git commit -m "$(cat <<'EOF'
perf(variants): port _fill_empty_scalar + _fill_empty_fixed numba->rust (parity)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Port `_fill_empty_seq` to Rust

Two-level dummy-fill for allele bytestrings. Numba reference: `_flat_variants.py:579-625`. Returns `(new_data uint8, new_var_offsets int64, new_seq_offsets int64)`.

**Files:**
- Modify: `src/variants/mod.rs`, `src/lib.rs`, `src/ffi/mod.rs`, `python/genvarloader/_dataset/_flat_variants.py`, `tests/parity/strategies.py`, `tests/parity/test_flat_variants_parity.py`

**Interfaces:**
- Produces (Rust core): `variants::fill_empty_seq(data: ArrayView1<u8>, var_offsets: ArrayView1<i64>, seq_offsets: ArrayView1<i64>, dummy: ArrayView1<u8>) -> (Array1<u8>, Array1<i64>, Array1<i64>)`.
- Produces (Python): `_fill_empty_seq(data, var_offsets, seq_offsets, dummy)` dispatch wrapper (existing name/signature; call sites at lines 323, 413).

- [ ] **Step 1: Rust core + cargo test**

Append to `src/variants/mod.rs` a faithful port (empty variant-rows receive one dummy allele of `dummy` bytes; non-empty pass through), then a cargo test covering one empty row + one non-empty row.

```rust
/// Two-level dummy-fill for allele bytestrings. Mirrors numba `_fill_empty_seq`.
pub fn fill_empty_seq(
    data: ArrayView1<u8>,
    var_offsets: ArrayView1<i64>,
    seq_offsets: ArrayView1<i64>,
    dummy: ArrayView1<u8>,
) -> (Array1<u8>, Array1<i64>, Array1<i64>) {
    let n_rows = var_offsets.len() - 1;
    let l = dummy.len() as i64;
    let mut new_var = Array1::<i64>::zeros(n_rows + 1);
    for i in 0..n_rows {
        let nv = var_offsets[i + 1] - var_offsets[i];
        new_var[i + 1] = new_var[i] + if nv > 0 { nv } else { 1 };
    }
    let total_vars = new_var[n_rows] as usize;
    let mut new_seq = Array1::<i64>::zeros(total_vars + 1);
    let mut vptr = 0usize;
    for i in 0..n_rows {
        let vs = var_offsets[i] as usize;
        let ve = var_offsets[i + 1] as usize;
        if ve == vs {
            new_seq[vptr + 1] = new_seq[vptr] + l;
            vptr += 1;
        } else {
            for v in vs..ve {
                let vlen = seq_offsets[v + 1] - seq_offsets[v];
                new_seq[vptr + 1] = new_seq[vptr] + vlen;
                vptr += 1;
            }
        }
    }
    let mut new_data = Array1::<u8>::zeros(new_seq[total_vars] as usize);
    let mut dptr = 0usize;
    for i in 0..n_rows {
        let vs = var_offsets[i] as usize;
        let ve = var_offsets[i + 1] as usize;
        if ve == vs {
            for k in 0..dummy.len() {
                new_data[dptr] = dummy[k];
                dptr += 1;
            }
        } else {
            for v in vs..ve {
                let bs = seq_offsets[v] as usize;
                let be = seq_offsets[v + 1] as usize;
                for k in bs..be {
                    new_data[dptr] = data[k];
                    dptr += 1;
                }
            }
        }
    }
    (new_data, new_var, new_seq)
}
```

- [ ] **Step 2: PyO3 wrapper + register; Step 3: Python wrapper**

Append the `ffi::fill_empty_seq` pyfunction (`-> (PyArray1<u8>, PyArray1<i64>, PyArray1<i64>)`), register in lib.rs; in `_flat_variants.py` rename njit → `_fill_empty_seq_numba`, register `"fill_empty_seq"`, and define the `_fill_empty_seq` dispatch wrapper coercing `data`/`dummy` to uint8 and offsets to int64.
Run: `pixi run -e dev cargo-test`
Expected: PASS.

- [ ] **Step 4: Parity strategy + test**

Add `fill_empty_seq_inputs` (var_offsets with at least one empty row; nested seq_offsets; random dummy bytes) and a parity test using `assert_kernel_parity_tuple("fill_empty_seq", ...)`.

- [ ] **Step 5: Run parity + cargo, commit**

Run: `pixi run -e dev pytest tests/parity/test_flat_variants_parity.py -q && pixi run -e dev cargo-test`
Expected: PASS.

```bash
rtk git add src/variants/mod.rs src/lib.rs src/ffi/mod.rs python/genvarloader/_dataset/_flat_variants.py tests/parity/strategies.py tests/parity/test_flat_variants_parity.py
rtk git commit -m "$(cat <<'EOF'
perf(variants): port _fill_empty_seq numba->rust (parity-gated)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Variants-mode dataset-level parity backstop

Variants output mode (`with_seqs("variants")`) has no differential coverage today. Add a dataset-level test mirroring `tests/parity/test_dataset_parity.py` (tracks mode), with a spy asserting the Rust flat kernels are actually invoked (no vacuous pass — the Phase 0 lesson).

**Files:**
- Create: `tests/parity/test_variants_dataset_parity.py`
- Reference: `tests/parity/test_dataset_parity.py`, `tests/parity/_fixtures.py`

**Interfaces:**
- Consumes: the registered kernels `gather_rows`, `gather_alleles`, `compact_keep_*`, `fill_empty_*` and a variants-capable dataset fixture.

- [ ] **Step 1: Read the existing backstop pattern**

Read `tests/parity/test_dataset_parity.py` and `tests/parity/_fixtures.py` in full. Reuse the dataset fixture; if it has no variants-mode dataset, build one via the fixture helpers (a small written dataset with variants).

- [ ] **Step 2: Write the backstop test**

Create `tests/parity/test_variants_dataset_parity.py`:

```python
import numpy as np
import pytest

from genvarloader._dataset import _flat_variants
from genvarloader import _dispatch

pytestmark = pytest.mark.parity


def _run_variants_getitem(ds):
    """Materialize a variants-mode getitem over the whole dataset."""
    vds = ds.with_seqs("variants")
    return vds[:, :]


def test_variants_getitem_parity_and_kernels_invoked(variants_dataset, monkeypatch):
    # Spy: count rust gather_rows calls so a vacuous pass is impossible.
    calls = {"n": 0}
    real = _dispatch.get("gather_rows")

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    # numba reference
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = _run_variants_getitem(variants_dataset)

    # rust + spy
    monkeypatch.setenv("GVL_BACKEND", "rust")
    monkeypatch.setattr(
        _flat_variants, "get",
        lambda name: spy if name == "gather_rows" else _dispatch.get(name),
    )
    out_rust = _run_variants_getitem(variants_dataset)

    assert calls["n"] > 0, "rust gather_rows was never invoked — vacuous parity"
    # Compare each parallel field of the RaggedVariants output byte-identically.
    # (Adapt field access to the RaggedVariants API: .alts, .refs, .v_idxs, etc.)
    for field in ("v_idxs", "alts", "refs"):
        a = np.asarray(getattr(out_numba, field).data)
        b = np.asarray(getattr(out_rust, field).data)
        np.testing.assert_array_equal(a, b)
```

Note: adjust `variants_dataset` fixture wiring and the `RaggedVariants` field names to the actual API (inspect `get_variants_flat`'s return and `_rag_variants.py`). The two essentials are (1) the spy proving the Rust kernel ran and (2) byte-identical field comparison.

- [ ] **Step 3: Run the backstop**

Run: `pixi run -e dev pytest tests/parity/test_variants_dataset_parity.py -q`
Expected: PASS, with the spy assertion satisfied.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/parity/test_variants_dataset_parity.py tests/parity/_fixtures.py
rtk git commit -m "$(cat <<'EOF'
test(parity): variants-mode dataset backstop (spy-guarded, byte-identical)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Full-suite gate, no-regression measurement, roadmap update

**Files:**
- Modify: `docs/roadmaps/rust-migration.md`

- [ ] **Step 1: Full test tree (both backends)**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS (covers `tests/dataset` AND `tests/unit`, per CLAUDE.md).
Run with the numba backend forced to confirm the reference path still works:
`GVL_BACKEND=numba pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: PASS.

- [ ] **Step 2: Lint + typecheck + format**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/ && pixi run -e dev typecheck`
Expected: PASS. Fix any issues, re-run.

- [ ] **Step 3: abi3 wheel build**

Run: `pixi run -e dev cargo-test` (already builds) and confirm a clean maturin build per the repo's build task.
Expected: builds clean.

- [ ] **Step 4: No-regression measurement on `chr22_geuv`**

Build the corpus if absent: `pixi run -e dev python tests/benchmarks/data/build_realistic.py` (needs `/carter` or `GVL_BENCH_SOURCE`).
Run haps mode (exercises get_diffs_sparse + choose_exonic_variants):
`pixi run -e dev python tests/benchmarks/profiling/profile.py --mode haps`
Compare to baseline **123.9 batch/s** — assert no regression (within noise).
Run variants mode (exercises the flat gather/fill kernels):
`pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants`
Compare to baseline **145.3 batch/s** — assert no regression.
Record both numbers (rust vs numba) for the roadmap. If a regression appears, profile and consider rayon on the hot kernel (allowed by the constraints only if needed).

- [ ] **Step 5: Update the roadmap**

In `docs/roadmaps/rust-migration.md`:
- Phase 2 header: set status 🚧→ (✅ when all gates green) + PR link.
- Fix the double-count: change the `_genotypes.py` line to "assembly/selection kernels (`get_diffs_sparse`, `choose_exonic_variants`); reconstruction kernels moved to Phase 3"; tick the `_genotypes.py` and `_flat_variants.py` items.
- Note `filter_af` deleted as dead (cross-reference the Phase 0 `splits_sum_le_value` precedent).
- Add a dated entry to the decisions log summarizing: kernels ported, dead-code deletion, `(2,n)` offset normalization, dtype-dispatch for `compact_keep`/`fill_empty_*`, gate = parity + no regression, and the measured haps/variants throughput (rust vs numba).
- Record measurements in the metrics narrative.

- [ ] **Step 6: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "$(cat <<'EOF'
docs(roadmap): Phase 2 genotype assembly + variant gather complete

Ported get_diffs_sparse + choose_exonic_variants + 7 flat gather/fill kernels
to Rust (parity-gated); deleted dead filter_af; fixed Phase 2/3 double-count.
No getitem regression (haps/variants vs baseline).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Port `get_diffs_sparse` → Task 2. ✅
- Port `choose_exonic_variants` (+ inner) → Task 3 (inner kept as numba-only helper). ✅
- Delete dead `filter_af` → Task 4. ✅
- Port 7 flat kernels → Tasks 5 (`_gather_v_idxs`+`_ss` as `gather_rows`), 6 (`_gather_alleles`), 7 (`_compact_keep`), 8 (`_fill_empty_scalar`+`_fill_empty_fixed`), 9 (`_fill_empty_seq`). 2+1+1+2+1 = 7. ✅
- `src/genotypes/` + `src/variants/` pure-ndarray cores, `src/ffi/` PyO3 only → Tasks 2/3 (genotypes), 5–9 (variants). ✅
- Dispatch registry, default rust, numba retained as reference → every port task. ✅
- Both offset forms via `(2,n)` normalization → Tasks 2/3/5. ✅
- Sequential (no rayon) → cores written sequentially; rayon only if Task 11 finds a regression. ✅
- Per-kernel hypothesis parity gates + variants-mode dataset backstop → Tasks 2–9 + Task 10. ✅
- Gate = parity + no regression, haps 123.9 / variants 145.3 baselines → Task 11. ✅
- Roadmap update incl. double-count fix → Task 11. ✅
- Harness tuple support (needed because Phase 2 kernels return tuples) → Task 1. ✅

**Placeholder scan:** Tasks 8 and 10 intentionally describe a repeated pattern (typed dtype wrappers / fixture wiring) rather than transcribing every near-identical variant — each names the exact functions, dtypes, signatures, and reference line numbers needed, and shows the generic Rust impl + one concrete strategy/test. This is pattern-repetition guidance, not a TBD; the int32 path is shown in full and float follows identically.

**Type consistency:** `_as_starts_stops` defined in Task 2, imported in Tasks 3 and 5. `assert_kernel_parity_tuple` defined in Task 1, used in Tasks 2–9. `gather_rows` (Rust) ↔ `"gather_rows"` (registry) ↔ `_gather_rows` (Python) consistent. `compact_keep_i32`/`compact_keep_f32` names consistent across core/ffi/registry/test. OFFSET_TYPE confirmed int64 in Task 3 Step 1 before relying on i64 returns.

**Open items the implementer MUST resolve (flagged inline, not deferred):**
- Task 3 Step 1: confirm `OFFSET_TYPE == int64`.
- Task 7 Step 1 / Task 8 Step 1: confirm production value dtypes for `_compact_keep` (dosage/ccf) and `_fill_empty_*` (start/ilen/dosage/flank_tokens); add a typed core if float64 appears (do NOT down-cast — would break parity).
- Task 5: confirm `geno_v_idxs`/`self.genotypes.data` dtype is int32.
- Task 10: confirm the `RaggedVariants` field names + add a variants-capable fixture if absent.
