# Rust Migration Phase 0 — Foundation & Differential-Test Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the reusable backend-dispatch registry, the `ffi/` seam, the both-layer differential-test harness, and the four perf baselines that gate every later Rust-migration phase — proven end-to-end by migrating one real Python-entry kernel (`splits_sum_le_value`) to Rust.

**Architecture:** A Python `_dispatch` registry maps each migratable kernel name to `{numba_fn, rust_fn, default}`; production calls `dispatch.get(name)(...)` and a `GVL_BACKEND` env var force-overrides for CI parity sweeps. New `#[pyfunction]` wrappers live only in `src/ffi/`; core Rust logic lives in lazily-created domain modules (Phase 0 grows just `src/utils.rs`). A `tests/parity/` harness asserts byte-identical output between the two backends at both per-kernel and dataset levels.

**Tech Stack:** Rust (PyO3 0.28, ndarray 0.17, numpy 0.28, abi3-py310), Python 3.10–3.13, numba 0.59.1, numpy 1.26, pytest + hypothesis, vcfixture ≥0.5.0, memray + py-spy, pixi, maturin.

## Global Constraints

- **Byte-identical parity is the landing gate.** A migrated kernel only "lands" when its output exactly matches the numba impl (same dtype, shape, bytes) across the test matrix.
- **Only Python-entry kernels get a registry entry.** njit-internal leaf kernels (called from inside other `@njit` functions, e.g. `padded_slice`) cannot be dispatched individually — they migrate with their Python-entry caller's subtree. This is why the proof-point is `splits_sum_le_value` (called from Python at `_write.py:1280`), not `padded_slice`.
- **Grow `src/` lazily.** No empty domain modules. Create a module only when it holds real code. Phase 0 creates exactly `src/utils.rs` + `src/ffi/`.
- **`bigwig.rs` / `tables.rs` / `ragged/` untouched** — no churn of existing Rust.
- **abi3 wheels must keep building** py310–313 × linux/macOS (standing invariant).
- **Pixi for everything:** dev tasks run via `pixi run -e dev <task>`. The extension must be rebuilt (`pixi run -e dev python -c "import genvarloader"` triggers maturin build, or `pixi run -e dev maturin develop --release`) after any Rust change before Python sees it.
- **Proof-point default = `rust`.** `splits_sum_le_value`'s registry `default` is set to `"rust"` so Phase 0 exercises the real Rust production path. The numba impl is **retained** as the registered parity reference (the harness needs both); its deletion is deferred to when the `_utils`/write phase (Phase 4) closes — NOT in this PR.
- **Commit conventionally** (commitizen). Use `rtk` prefix for git per CLAUDE.md.

---

## File Structure

**Created:**
- `python/genvarloader/_dispatch.py` — backend registry (`register`, `get`, `backends`, `GVL_BACKEND` resolution).
- `src/utils.rs` — Rust `splits_sum_le_value` core logic (plain ndarray, no PyO3).
- `src/ffi/mod.rs` — `#[pyfunction]` wrappers; the only place PyO3 touches new kernels.
- `tests/unit/test_dispatch.py` — registry unit tests.
- `tests/parity/__init__.py`, `tests/parity/_harness.py` — `assert_kernel_parity` helper.
- `tests/parity/strategies.py` — hypothesis input strategies (template: `splits_sum_le_value`).
- `tests/parity/test_splits_sum_le_value_parity.py` — per-kernel parity test.
- `tests/parity/test_dataset_parity.py` — dataset-level backstop (write round-trip across backends).
- `tests/benchmarks/profiling/profile_write.py` — write/update baseline driver (1kg corpus).
- `tests/benchmarks/profiling/baseline_getitem.sh` — py-spy hand-off script for David (sudo on macOS).

**Modified:**
- `src/lib.rs` — register the `ffi` module + new pyfunction in the pymodule.
- `python/genvarloader/_dataset/_utils.py` — rename numba body, import Rust fn, `register(...)`, keep `splits_sum_le_value` as the dispatching wrapper.
- `pyproject.toml` — register the `parity` pytest marker.
- `pixi.toml` — add standalone `cargo-test` + `memray-write` tasks.
- `docs/roadmaps/rust-migration.md` — tick Phase 0, fill baseline table, add decisions-log entry.

---

## Task 1: Backend dispatch registry

**Files:**
- Create: `python/genvarloader/_dispatch.py`
- Test: `tests/unit/test_dispatch.py`

**Interfaces:**
- Produces:
  - `register(name: str, *, numba: Callable, rust: Callable, default: Literal["numba","rust"] = "numba") -> None`
  - `get(name: str) -> Callable` — returns active backend's callable (env `GVL_BACKEND` override wins, else entry default).
  - `backends(name: str) -> tuple[Callable, Callable]` — returns `(numba_fn, rust_fn)`.
  - `registered_names() -> list[str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_dispatch.py
import pytest
from genvarloader import _dispatch


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    # Isolate each test: fresh registry + no inherited GVL_BACKEND.
    monkeypatch.setattr(_dispatch, "_REGISTRY", {})
    monkeypatch.delenv("GVL_BACKEND", raising=False)
    yield


def _reg():
    _dispatch.register("k", numba=lambda: "numba", rust=lambda: "rust", default="numba")


def test_get_returns_default_backend():
    _reg()
    assert _dispatch.get("k")() == "numba"


def test_get_respects_per_kernel_rust_default():
    _dispatch.register("k", numba=lambda: "n", rust=lambda: "r", default="rust")
    assert _dispatch.get("k")() == "r"


def test_env_override_forces_all_kernels(monkeypatch):
    _reg()
    monkeypatch.setenv("GVL_BACKEND", "rust")
    assert _dispatch.get("k")() == "rust"


def test_backends_returns_both_regardless_of_default():
    _reg()
    numba_fn, rust_fn = _dispatch.backends("k")
    assert numba_fn() == "numba" and rust_fn() == "rust"


def test_unknown_name_raises_keyerror_listing_names():
    _reg()
    with pytest.raises(KeyError, match="k"):
        _dispatch.get("missing")


def test_invalid_env_backend_raises(monkeypatch):
    _reg()
    monkeypatch.setenv("GVL_BACKEND", "julia")
    with pytest.raises(ValueError, match="GVL_BACKEND"):
        _dispatch.get("k")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/unit/test_dispatch.py -v`
Expected: FAIL with `ModuleNotFoundError: genvarloader._dispatch` (or AttributeError).

- [ ] **Step 3: Implement the registry**

```python
# python/genvarloader/_dispatch.py
"""Backend dispatch registry for the Rust migration strangler window.

Each migratable Python-entry kernel registers a numba and a rust implementation.
Production code calls ``get(name)(...)``; ``GVL_BACKEND=numba|rust`` force-overrides
all kernels (used by CI parity sweeps). Deleted wholesale in migration Phase 5.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal

_Backend = Literal["numba", "rust"]
_REGISTRY: dict[str, dict[str, object]] = {}


def register(
    name: str,
    *,
    numba: Callable,
    rust: Callable,
    default: _Backend = "numba",
) -> None:
    if default not in ("numba", "rust"):
        raise ValueError(f"default must be 'numba' or 'rust', got {default!r}")
    _REGISTRY[name] = {"numba": numba, "rust": rust, "default": default}


def _entry(name: str) -> dict[str, object]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no kernel registered as {name!r}; registered: {registered_names()}"
        ) from None


def get(name: str) -> Callable:
    entry = _entry(name)
    backend = os.environ.get("GVL_BACKEND")
    if backend is None:
        backend = entry["default"]  # type: ignore[assignment]
    elif backend not in ("numba", "rust"):
        raise ValueError(
            f"GVL_BACKEND must be 'numba' or 'rust', got {backend!r}"
        )
    return entry[backend]  # type: ignore[return-value]


def backends(name: str) -> tuple[Callable, Callable]:
    entry = _entry(name)
    return entry["numba"], entry["rust"]  # type: ignore[return-value]


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/test_dispatch.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dispatch.py tests/unit/test_dispatch.py
rtk git commit -m "feat(dispatch): backend registry for Rust migration strangler window"
```

---

## Task 2: Rust `splits_sum_le_value` + `ffi/` seam

**Files:**
- Create: `src/utils.rs`, `src/ffi/mod.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces (Rust): `pub fn splits_sum_le_value(arr: ArrayView1<i64>, max_value: f64) -> Array1<i64>` in `src/utils.rs`.
- Produces (Python-visible): `genvarloader.genvarloader.splits_sum_le_value(arr: NDArray[int64], max_value: float) -> NDArray[int64]`.

Parity contract — mirror the numba body exactly (`_utils.py:142`): start `indices=[0]`, `current_sum=0`; for each value accumulate, and when `current_sum > max_value` push the current index and reset `current_sum = value`; append `len(arr)`. Accumulate in `i64`, compare `current_sum as f64 > max_value`.

- [ ] **Step 1: Write the failing Rust unit test**

```rust
// src/utils.rs
use ndarray::{Array1, ArrayView1};

/// Greedy split offsets for groups summing to no more than `max_value`.
/// Byte-identical to the numba `splits_sum_le_value` in `_utils.py`.
pub fn splits_sum_le_value(arr: ArrayView1<i64>, max_value: f64) -> Array1<i64> {
    let mut indices: Vec<i64> = vec![0];
    let mut current_sum: i64 = 0;
    for (idx, &value) in arr.iter().enumerate() {
        current_sum += value;
        if current_sum as f64 > max_value {
            indices.push(idx as i64);
            current_sum = value;
        }
    }
    indices.push(arr.len() as i64);
    Array1::from(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn docstring_example() {
        // splits_sum_le_value([5,5,11,9,2,7], 10) -> [0,2,3,4,6]
        let a = array![5_i64, 5, 11, 9, 2, 7];
        assert_eq!(splits_sum_le_value(a.view(), 10.0), array![0_i64, 2, 3, 4, 6]);
    }

    #[test]
    fn empty_array() {
        let a: Array1<i64> = Array1::from(vec![]);
        assert_eq!(splits_sum_le_value(a.view(), 10.0), array![0_i64, 0]);
    }

    #[test]
    fn single_over_max_kept_in_own_group() {
        let a = array![3_i64, 100, 3];
        // 3<=10; +100 ->103>10 push 1 reset 100; +3=103>10 push 2 reset 3; end push 3
        assert_eq!(splits_sum_le_value(a.view(), 10.0), array![0_i64, 1, 2, 3]);
    }
}
```

- [ ] **Step 2: Run the Rust test to verify it fails to compile (module not wired)**

Run: `cargo test splits_sum_le_value 2>&1 | head -20`
Expected: FAIL — `src/utils.rs` not declared in `lib.rs` (`file not found for module` / unresolved).

- [ ] **Step 3: Create the `ffi/` wrapper and wire both modules into `lib.rs`**

```rust
// src/ffi/mod.rs
//! PyO3 boundary for migrated core kernels. The ONLY place new kernels touch Python.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils;

/// Greedy split offsets for groups summing to no more than `max_value`.
#[pyfunction]
pub fn splits_sum_le_value<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<i64>,
    max_value: f64,
) -> Bound<'py, PyArray1<i64>> {
    utils::splits_sum_le_value(arr.as_array(), max_value).into_pyarray(py)
}
```

In `src/lib.rs`, add the module declarations at the top (next to the existing `pub mod` lines):

```rust
pub mod bigwig;
pub mod ffi;
pub mod ragged;
pub mod tables;
pub mod utils;
```

And register the new pyfunction inside the `#[pymodule] fn genvarloader(...)` body, after the existing `ragged::ragged_to_padded` line:

```rust
    m.add_function(wrap_pyfunction!(ffi::splits_sum_le_value, m)?)?;
```

- [ ] **Step 4: Run the Rust tests to verify they pass**

Run: `cargo test 2>&1 | tail -20`
Expected: PASS — `docstring_example`, `empty_array`, `single_over_max_kept_in_own_group` pass; existing tests still pass.

- [ ] **Step 5: Rebuild the extension and confirm the symbol is importable from Python**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev python -c "from genvarloader.genvarloader import splits_sum_le_value as f; import numpy as np; print(f(np.array([5,5,11,9,2,7], np.int64), 10.0))"
```
Expected: prints `[0 2 3 4 6]`.

- [ ] **Step 6: Commit**

```bash
rtk git add src/utils.rs src/ffi/mod.rs src/lib.rs
rtk git commit -m "feat(ffi): Rust splits_sum_le_value + ffi seam module"
```

---

## Task 3: Route the production call site through dispatch

**Files:**
- Modify: `python/genvarloader/_dataset/_utils.py:142-173` (the numba kernel)
- Test: `tests/unit/dataset/test_splits_dispatch.py` (create)

**Interfaces:**
- Consumes: `genvarloader._dispatch.register/get` (Task 1); `genvarloader.genvarloader.splits_sum_le_value` (Task 2).
- Produces: `genvarloader._dataset._utils.splits_sum_le_value(arr, max_value) -> NDArray[int64]` — now a dispatching wrapper; same name/import path, so `_write.py`'s `from ._utils import ... splits_sum_le_value` is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/dataset/test_splits_dispatch.py
import numpy as np
import pytest
from genvarloader._dataset._utils import splits_sum_le_value


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_wrapper_matches_known_result(backend, monkeypatch):
    monkeypatch.setenv("GVL_BACKEND", backend)
    out = splits_sum_le_value(np.array([5, 5, 11, 9, 2, 7]), 10)
    np.testing.assert_array_equal(out, np.array([0, 2, 3, 4, 6]))
    assert out.dtype == np.intp


def test_wrapper_is_registered():
    from genvarloader import _dispatch

    assert "splits_sum_le_value" in _dispatch.registered_names()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_splits_dispatch.py -v`
Expected: FAIL — `splits_sum_le_value` not registered / wrapper not present.

- [ ] **Step 3: Convert the kernel into numba-body + Rust + dispatching wrapper**

In `python/genvarloader/_dataset/_utils.py`, rename the existing `@nb.njit` `splits_sum_le_value` to `_splits_sum_le_value_numba` (keep the body byte-for-byte), then add the import, registration, and wrapper. Place the wrapper where `splits_sum_le_value` used to be so callers are unaffected:

```python
# near the other imports at the top of _utils.py
from ..genvarloader import splits_sum_le_value as _splits_sum_le_value_rust
from .._dispatch import get, register


@nb.njit(nogil=True, cache=True)
def _splits_sum_le_value_numba(
    arr: NDArray[np.number], max_value: float
) -> NDArray[np.intp]:
    # (unchanged body, formerly `splits_sum_le_value`)
    indices = [0]
    current_sum = 0
    for idx, value in enumerate(arr):
        current_sum += value
        if current_sum > max_value:
            indices.append(idx)
            current_sum = value
    indices.append(len(arr))
    return np.array(indices, np.intp)


register(
    "splits_sum_le_value",
    numba=_splits_sum_le_value_numba,
    rust=_splits_sum_le_value_rust,
    default="rust",
)


def splits_sum_le_value(
    arr: NDArray[np.number], max_value: float
) -> NDArray[np.intp]:
    """Greedy split offsets for groups summing to no more than ``max_value``.

    Dispatches to the numba or Rust backend via :mod:`genvarloader._dispatch`.
    Both backends receive an int64-contiguous array + float scalar so their
    output is byte-identical (see tests/parity).

    Examples
    --------
    >>> splits_sum_le_value(np.array([5, 5, 11, 9, 2, 7]), 10)
    array([0, 2, 3, 4, 6])
    """
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    return get("splits_sum_le_value")(arr, float(max_value))
```

Note: the production input (`mem_per_r` at `_write.py:1269`) is already int64, so the `ascontiguousarray` cast is a no-op there; it exists to guarantee both backends get identical bytes for arbitrary `np.number` callers.

- [ ] **Step 4: Run the dispatch test + the write-path tests**

Run:
```bash
pixi run -e dev pytest tests/unit/dataset/test_splits_dispatch.py -v
pixi run -e dev pytest tests/dataset tests/unit -k "write or split or chunk" -q
```
Expected: PASS — dispatch tests pass; write-path tests unaffected by `default="rust"`.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_utils.py tests/unit/dataset/test_splits_dispatch.py
rtk git commit -m "feat(utils): route splits_sum_le_value through backend dispatch (default rust)"
```

---

## Task 4: Per-kernel parity harness + `splits_sum_le_value` parity test

**Files:**
- Create: `tests/parity/__init__.py`, `tests/parity/_harness.py`, `tests/parity/strategies.py`, `tests/parity/test_splits_sum_le_value_parity.py`
- Modify: `pyproject.toml` (register `parity` marker)

**Interfaces:**
- Consumes: `genvarloader._dispatch.backends` (Task 1).
- Produces:
  - `tests.parity._harness.assert_kernel_parity(name: str, *inputs) -> None`
  - `tests.parity.strategies.splits_inputs() -> SearchStrategy[tuple[NDArray[int64], float]]`

- [ ] **Step 1: Write the failing tests (harness + per-kernel)**

```python
# tests/parity/__init__.py
```

```python
# tests/parity/_harness.py
"""Run both registered backends and assert byte-identical output."""

from __future__ import annotations

import numpy as np

from genvarloader import _dispatch


def assert_kernel_parity(name: str, *inputs) -> None:
    numba_fn, rust_fn = _dispatch.backends(name)
    got_numba = numba_fn(*inputs)
    got_rust = rust_fn(*inputs)
    assert got_numba.dtype == got_rust.dtype, (
        f"{name}: dtype {got_numba.dtype} != {got_rust.dtype}"
    )
    assert got_numba.shape == got_rust.shape, (
        f"{name}: shape {got_numba.shape} != {got_rust.shape}"
    )
    np.testing.assert_array_equal(got_numba, got_rust)
```

```python
# tests/parity/strategies.py
"""Hypothesis input strategies per migrated kernel (byte-identical generators)."""

from __future__ import annotations

import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import strategies as st


def splits_inputs():
    arrays = hnp.arrays(
        dtype=np.int64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=64),
        # non-negative values mirror the production input (memory-per-region counts)
        elements=st.integers(min_value=0, max_value=10_000),
    )
    max_values = st.floats(
        min_value=0.0, max_value=50_000.0, allow_nan=False, allow_infinity=False
    )
    return st.tuples(arrays, max_values)
```

```python
# tests/parity/test_splits_sum_le_value_parity.py
import pytest
from hypothesis import given

from genvarloader._dataset import _utils  # noqa: F401  (import triggers register())
from tests.parity._harness import assert_kernel_parity
from tests.parity.strategies import splits_inputs

pytestmark = pytest.mark.parity


@given(splits_inputs())
def test_splits_sum_le_value_parity(inputs):
    arr, max_value = inputs
    assert_kernel_parity("splits_sum_le_value", arr, max_value)
```

- [ ] **Step 2: Verify the harness self-check fails on a mismatch first**

Add a throwaway sanity test, run it, then delete it:
```python
# (temporary, in test_splits_sum_le_value_parity.py)
def test_harness_detects_mismatch():
    import numpy as np
    from genvarloader import _dispatch
    _dispatch.register("bad", numba=lambda a: np.array([1]), rust=lambda a: np.array([2]), default="numba")
    with pytest.raises(AssertionError):
        assert_kernel_parity("bad", np.array([0]))
```
Run: `pixi run -e dev pytest tests/parity/test_splits_sum_le_value_parity.py::test_harness_detects_mismatch -v`
Expected: PASS (the harness correctly raises). Then delete `test_harness_detects_mismatch`.

- [ ] **Step 3: Register the `parity` marker**

In `pyproject.toml`, under the pytest config (`[tool.pytest.ini_options]` `markers = [...]`), add:
```toml
    "parity: byte-identical numba-vs-rust differential tests (Rust migration)",
```
(If a `markers` list does not yet exist, create it under `[tool.pytest.ini_options]`.)

- [ ] **Step 4: Run the parity test to verify it passes**

Run: `pixi run -e dev pytest tests/parity/test_splits_sum_le_value_parity.py -v`
Expected: PASS — `test_splits_sum_le_value_parity` passes (hypothesis explores; no `PytestUnknownMarkWarning`).

- [ ] **Step 5: Commit**

```bash
rtk git add tests/parity/ pyproject.toml
rtk git commit -m "test(parity): per-kernel differential harness + splits_sum_le_value gate"
```

---

## Task 5: Dataset-level parity backstop

**Files:**
- Create: `tests/parity/test_dataset_parity.py`

**Interfaces:**
- Consumes: `vcfixture` for fixture data; `GVL_BACKEND` env override (Task 1); `gvl.write` (write path uses `splits_sum_le_value`).

Goal: prove that flipping `GVL_BACKEND` between `numba` and `rust` produces an identical written dataset, exercising `splits_sum_le_value` through the real `gvl.write` path with a bigWig track (the only path that calls it).

- [ ] **Step 1: Write the failing test**

```python
# tests/parity/test_dataset_parity.py
"""Dataset-level backstop: a write round-trip must be identical across backends."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.parity

# Run gvl.write in a subprocess per backend so GVL_BACKEND is read at import time
# and numba caching can't leak the wrong impl across runs.
_WRITE_SNIPPET = """
import os, sys
from pathlib import Path
import genvarloader as gvl
from tests.parity._fixtures import build_write_inputs
out = Path(sys.argv[1])
bed, variants, bigwigs = build_write_inputs(Path(sys.argv[2]))
gvl.write(path=out, bed=bed, variants=variants, bigwigs=bigwigs)
"""


def _run_write(tmp_path: Path, backend: str, shared: Path) -> Path:
    out = tmp_path / f"ds_{backend}.gvl"
    env = {**os.environ, "GVL_BACKEND": backend}
    subprocess.run(
        [sys.executable, "-c", _WRITE_SNIPPET, str(out), str(shared)],
        check=True,
        env=env,
    )
    return out


def test_write_round_trip_identical_across_backends(tmp_path):
    shared = tmp_path / "inputs"
    numba_ds = _run_write(tmp_path, "numba", shared)
    rust_ds = _run_write(tmp_path, "rust", shared)
    # The intervals/offsets npy files written via the split path must match byte-for-byte.
    for rel in ["intervals/intervals.npy", "intervals/offsets.npy"]:
        a = np.load(numba_ds / rel)
        b = np.load(rust_ds / rel)
        np.testing.assert_array_equal(a, b)
```

> Note for the implementer: `tests/parity/_fixtures.build_write_inputs(dir)` must produce a small bed + variants + a multi-sample BigWigs track using `vcfixture` and the synthetic bigWig corpus helpers already in `tests/_bigwig_corpus.py` (see `tests/benchmarks/profiling/profile_bigwig_write.py:43-49` for the `DEFAULT_CONTIGS` / `make_regions` pattern). The track must be large enough that `mem_per_r` produces more than one split (so `splits_sum_le_value` actually partitions). Confirm the exact written paths (`intervals/intervals.npy`, `intervals/offsets.npy`) against an existing written dataset under `tests/benchmarks/data/chr22_geuv.gvl/` and adjust `rel` if the layout differs.

- [ ] **Step 2: Implement `tests/parity/_fixtures.py`**

Build the smallest inputs that force ≥2 splits. Model the BigWigs corpus + regions on `tests/_bigwig_corpus.py` and the bed on `vcfixture`. Set `max_mem` low (e.g. pass via `gvl.write(..., max_mem=...)`) so `splits_sum_le_value` splits. Return `(bed, variants, bigwigs)`.

```python
# tests/parity/_fixtures.py  (skeleton — fill paths against the real corpus helpers)
from __future__ import annotations
from pathlib import Path


def build_write_inputs(work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    # 1. Build a few-sample BigWigs track + regions (see tests/_bigwig_corpus.py).
    # 2. Build a tiny variants source via vcfixture.
    # 3. Return (bed, variants, bigwigs) suitable for gvl.write.
    raise NotImplementedError
```

- [ ] **Step 3: Run the backstop test**

Run: `pixi run -e dev pytest tests/parity/test_dataset_parity.py -v`
Expected: PASS — both backends write byte-identical interval/offset arrays.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/parity/test_dataset_parity.py tests/parity/_fixtures.py
rtk git commit -m "test(parity): dataset-level write round-trip backstop across backends"
```

---

## Task 6: Build/test wiring + abi3 wheel confirmation

**Files:**
- Modify: `pixi.toml` (`[tasks]`)

**Interfaces:** none new — verification + convenience tasks.

- [ ] **Step 1: Add convenience tasks to `pixi.toml` `[tasks]`**

```toml
cargo-test = { cmd = "cargo test --release" }
memray-write = { cmd = "memray run -fo tests/benchmarks/profiling/write.memray.bin tests/benchmarks/profiling/profile_write.py --op write" }
```

(`test` already runs `pytest tests && cargo test --release`, so cargo is wired; `cargo-test` is a standalone shortcut.)

- [ ] **Step 2: Verify the whole tree passes, including the new parity dir + cargo**

Run: `pixi run -e dev test`
Expected: PASS — `pytest tests` (now including `tests/parity/`) green, `cargo test --release` green.

- [ ] **Step 3: Confirm the abi3 wheel still builds with the new `ffi/` seam**

Run: `pixi run -e dev maturin build --release 2>&1 | tail -15`
Expected: a single `*_cp310_abi3_*.whl` (or platform-tagged abi3 wheel) is produced with no errors. Note in the commit message that the release wheel matrix (`.github/workflows/publish.yaml`, py310–313 × linux/macOS) is release-gated and unaffected by the new pure-Rust module.

- [ ] **Step 4: Commit**

```bash
rtk git add pixi.toml
rtk git commit -m "build(pixi): cargo-test + memray-write tasks; confirm abi3 wheel builds"
```

---

## Task 7: Capture baselines

**Files:**
- Create: `tests/benchmarks/profiling/profile_write.py`, `tests/benchmarks/profiling/baseline_getitem.sh`

**Interfaces:**
- Consumes: existing `tests/benchmarks/profiling/profile.py` (getitem throughput); 1kg corpus from `tests/data/generate_1kg_ground_truth.py` (`pixi run -e dev gen-1kg`); `gvl.write` / `gvl.update`.

Baseline tooling (per approved spec decision): reuse `profile.py` for getitem on its realistic `chr22_geuv` corpus; write/update on the 1kg corpus via a new memray-able driver. py-spy steps are handed to David (sudo on macOS) — the agent does NOT invoke py-spy. memray runs directly.

- [ ] **Step 1: Write the write/update baseline driver**

```python
# tests/benchmarks/profiling/profile_write.py
"""Time + measure gvl.write() and gvl.update() on the 1kg chr21/chr22 corpus.

  pixi run -e dev python tests/benchmarks/profiling/profile_write.py --op write
  pixi run -e dev python tests/benchmarks/profiling/profile_write.py --op update

Reports wall-clock; run under memray (pixi run -e dev memray-write) for peak RSS.
Requires the 1kg inputs: `pixi run -e dev gen-1kg` first.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["write", "update"], required=True)
    args = p.parse_args()

    # The 1kg generator builds the filtered bcf + region bed it writes from;
    # reuse those exact inputs (see tests/data/generate_1kg_ground_truth.py).
    from tests.data import generate_1kg_ground_truth as g1k

    bcf = g1k.ONE_KG_DIR / "filtered.bcf"   # confirm the actual filename in g1k
    bed = g1k.ONE_KG_DIR / "regions.bed"    # confirm the actual filename in g1k
    if not bcf.exists():
        raise SystemExit("Run `pixi run -e dev gen-1kg` first to build 1kg inputs.")

    import genvarloader as gvl
    from genoray import VCF

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "phased_1kg.gvl"
        t0 = time.perf_counter()
        gvl.write(path=out, bed=bed, variants=VCF(bcf))
        dt_write = time.perf_counter() - t0
        if args.op == "write":
            print(f"op=write corpus=1kg-chr21chr22 wall={dt_write:.3f}s")
            return
        # update: re-run gvl.update on the freshly written dataset.
        t0 = time.perf_counter()
        gvl.update(out)  # confirm gvl.update's signature/args in _write.py:360
        dt_update = time.perf_counter() - t0
        print(f"op=update corpus=1kg-chr21chr22 wall={dt_update:.3f}s")


if __name__ == "__main__":
    main()
```

> Implementer note: open `tests/data/generate_1kg_ground_truth.py` and `python/genvarloader/_dataset/_write.py:360` to confirm (a) the exact 1kg input filenames/`ONE_KG_DIR` constant and (b) `gvl.update`'s real signature, then fix the three `# confirm ...` lines. Do not guess — read and match.

- [ ] **Step 2: Build the 1kg corpus and capture write/update wall-clock + RSS (agent runs these)**

Run:
```bash
pixi run -e dev gen-1kg
pixi run -e dev python tests/benchmarks/profiling/profile_write.py --op write
pixi run -e dev python tests/benchmarks/profiling/profile_write.py --op update
pixi run -e dev memray-write
pixi run -e dev memray stats tests/benchmarks/profiling/write.memray.bin | grep -i "peak"
```
Expected: wall-clock seconds for write + update; peak RSS from memray stats. Record the numbers.

- [ ] **Step 3: Write the getitem py-spy hand-off script for David**

```bash
# tests/benchmarks/profiling/baseline_getitem.sh
#!/usr/bin/env bash
# __getitem__ throughput baseline. py-spy needs sudo on macOS — David runs this.
# Reuses the existing profile.py harness on the realistic chr22_geuv corpus.
set -euo pipefail
cd "$(dirname "$0")/../../.."
for mode in haplotypes tracks variants; do
  echo "=== getitem baseline: $mode ==="
  sudo pixi run -e dev py-spy record -d 30 -r 250 -f speedscope \
    -o "tests/benchmarks/profiling/baseline_getitem_${mode}.speedscope.json" \
    -- python tests/benchmarks/profiling/profile.py --mode "$mode"
done
echo "Report wall-clock from profile.py stdout + speedscope total samples per mode."
```

- [ ] **Step 4: Hand off the getitem baseline to David and record the result**

Per [[feedback_macos_profiling_handoff]], do NOT run py-spy directly. Tell David:
> "Run `bash tests/benchmarks/profiling/baseline_getitem.sh` (it uses sudo for py-spy) and paste the per-mode wall-clock + speedscope sample totals so I can fill the baseline table."

Record the returned getitem throughput numbers.

- [ ] **Step 5: Commit the bench scripts**

```bash
rtk git add tests/benchmarks/profiling/profile_write.py tests/benchmarks/profiling/baseline_getitem.sh
rtk git commit -m "bench: write/update (1kg) + getitem (profile.py) baseline drivers"
```

---

## Task 8: Update the roadmap

**Files:**
- Modify: `docs/roadmaps/rust-migration.md`

- [ ] **Step 1: Tick Phase 0 + fill the baseline table**

In `docs/roadmaps/rust-migration.md`:
- Change the Phase 0 header marker `⬜` → `✅` (or `🚧` if the getitem hand-off is still pending David's run) and add the PR link.
- Tick the completed Phase 0 checkboxes (harness built; cargo test wired/verified; abi3 wheels confirmed; baselines captured). For the `src/` skeleton item, note it was scoped to the `ffi/` seam only (lazy growth), per the approved spec.
- Fill the baseline table rows with the captured numbers. Label corpora explicitly: write/update rows = "1kg chr21/chr22 (vcfixture tier)"; the `__getitem__` row = "realistic chr22 geuvadis benchmark dataset (profile.py)". Flip the "Captured" marks to ✅.

- [ ] **Step 2: Add a decisions-log entry**

Append to the "Notes & decisions log":
```markdown
- 2026-06-23: Phase 0 foundation landed. Backend dispatch registry
  (`python/genvarloader/_dispatch.py`, `GVL_BACKEND` global override + per-kernel
  default, deleted in Phase 5). New `src/ffi/` seam holds all PyO3 wrappers; core
  Rust logic in lazily-grown domain modules (`src/utils.rs` first). Both-layer
  parity harness in `tests/parity/` (per-kernel hypothesis gate + dataset-level
  write round-trip backstop). Dispatch rule: only Python-entry kernels register;
  njit-internal leaves (e.g. `padded_slice`) migrate with their caller's subtree.
  Proof-point: `splits_sum_le_value` migrated end-to-end (default=rust, numba
  retained as parity reference; deletion deferred to Phase 4). Baselines captured
  (see table). Spec: docs/superpowers/specs/2026-06-23-rust-migration-phase-0-foundation-design.md.
```

- [ ] **Step 3: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): Phase 0 foundation done; baselines + dispatch/ffi/harness"
```

---

## Self-Review Notes

- **Spec coverage:** registry (Task 1), ffi seam (Task 2), proof-point migration (Tasks 2–3), both-layer harness (Tasks 4–5), cargo/pixi wiring (Task 6), abi3 confirmation (Task 6), all-four baselines (Task 7), roadmap update (Task 8). Lazy `src/` growth honored (only `utils.rs` + `ffi/`).
- **Proof-point default:** `rust`, numba retained as parity reference (Global Constraints) — resolves the spec's deferred plan-level decision.
- **Two flagged deviations from the written spec (approved with the user during planning):** (1) getitem baseline reuses the existing `profile.py` harness instead of pulling the `prefetching-dataloader` bench; (2) baseline corpora are mixed + labeled (write/update on 1kg, getitem on chr22_geuv) rather than single-corpus.
- **Known confirm-before-coding spots** (read, don't guess): `_fixtures.build_write_inputs` written-path names (Task 5), 1kg input filenames + `gvl.update` signature (Task 7).
