# Rayon Multithread Verification & Thread-Cap Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make gvl's rayon-parallel paths runnable and verified under real multithreaded load, fix the `cap_threads()` oversubscription bugs behind issue #263, release the GIL around the rayon FFI calls, and unblock the pre-commit pyrefly hook.

**Architecture:** Four Python-side changes in `_threads.py` (force-parallel override, CFS-quota-aware CPU detection, `RAYON_NUM_THREADS` overwrite) and `tests/parity/` (pyrefly fix); one Rust change wrapping every parallel FFI entry point in `py.allow_threads`; and two new test modules (forced-parallel equivalence + a spawn-worker stress reproducer). The size gate in `should_parallelize()` is the single chokepoint the tests flip via `GVL_FORCE_PARALLEL`.

**Tech Stack:** Python 3.10–3.13, Rust + PyO3 + rayon (via maturin), pytest, pixi (`-e dev`).

**Spec:** `docs/superpowers/specs/2026-06-30-rayon-multithread-verification-design.md`
**Issue:** https://github.com/mcvickerlab/GenVarLoader/issues/263

## Global Constraints

- **Rebuild Rust before any Python test that imports the extension:** `pixi run -e dev maturin develop --release`. `pytest` does NOT rebuild; a stale `.so` silently masks Rust changes. (`cargo test` compiles from source and is exempt.)
- All dev commands run under pixi: `pixi run -e dev <cmd>`. Platform is linux-64.
- Lint/format cover BOTH trees: `ruff check python/ tests/` and `ruff format python/ tests/`. E501 is ignored.
- Typecheck: `pixi run -e dev typecheck` (pyrefly) must report **zero** errors — the pre-commit hook runs it and blocks commits.
- **No inline `# pyrefly: ignore`** — an ignore that becomes unused in another environment breaks CI via `unused-ignore`. Fix by restructuring code.
- Conventional-commit messages (commitizen). End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- prek/pre-commit hooks must be installed and must pass **without** `--no-verify` by the end of the work.
- `should_parallelize()` / `_force_parallel()` must read `os.environ` **live** on every call (no import-time capture) so tests can toggle env vars with `monkeypatch.setenv`.

---

## File Structure

- `python/genvarloader/_threads.py` — MODIFY. Add `_force_parallel()`, `_cgroup_cpu_quota()`; edit `should_parallelize()`, `_detect_cpus()`, `cap_threads()`. Update module docstring.
- `tests/unit/test_threads.py` — MODIFY. Add tests for the three `_threads.py` changes.
- `tests/parity/test_gen_dataset_goldens.py` — MODIFY. Fix `unbound-name` (lines ~262–270).
- `tests/parity/test_variants_dataset_parity.py` — MODIFY. Fix `unbound-name` (lines ~85–115).
- `src/ffi/mod.rs` — MODIFY. Wrap the 11 parallel FFI entry points in `py.allow_threads`.
- `tests/integration/test_rayon_forced_parallel.py` — CREATE. End-to-end forced-parallel == serial equivalence.
- `tests/integration/test_rayon_stress.py` — CREATE. Spawn-worker stress reproducer (`slow`).

---

## Task 1: `GVL_FORCE_PARALLEL` override

**Files:**
- Modify: `python/genvarloader/_threads.py`
- Test: `tests/unit/test_threads.py`

**Interfaces:**
- Produces: `_force_parallel() -> bool` (reads `GVL_FORCE_PARALLEL` live); `should_parallelize(total_bytes: int) -> bool` short-circuits to `True` when `_force_parallel()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_threads.py`:

```python
import pytest


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
def test_force_parallel_truthy(monkeypatch, val):
    monkeypatch.setenv("GVL_FORCE_PARALLEL", val)
    # Below the byte threshold, but forced on → parallel.
    assert th.should_parallelize(0) is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "banana"])
def test_force_parallel_falsy_falls_back_to_threshold(monkeypatch, val):
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.delenv("GVL_NUM_THREADS", raising=False)
    monkeypatch.setenv("GVL_FORCE_PARALLEL", val)
    _constrain_detected_cpus(monkeypatch, 4)
    # Not forced → normal size gate applies.
    assert th.should_parallelize(0) is False
    assert th.should_parallelize(4 * th._MIN_BYTES_PER_THREAD) is True


def test_force_parallel_unset(monkeypatch):
    monkeypatch.delenv("GVL_FORCE_PARALLEL", raising=False)
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.delenv("GVL_NUM_THREADS", raising=False)
    _constrain_detected_cpus(monkeypatch, 4)
    assert th.should_parallelize(0) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/test_threads.py -k force_parallel -v`
Expected: FAIL — `AttributeError`/wrong result (`_force_parallel` / short-circuit not implemented).

- [ ] **Step 3: Implement**

In `python/genvarloader/_threads.py`, add near the top after `_NUM_THREADS`:

```python
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _force_parallel() -> bool:
    """True iff GVL_FORCE_PARALLEL is set to a truthy value (read live)."""
    return os.environ.get("GVL_FORCE_PARALLEL", "").strip().lower() in _TRUTHY
```

Replace `should_parallelize`:

```python
def should_parallelize(total_bytes: int) -> bool:
    # GVL_FORCE_PARALLEL bypasses the size gate so the multithreaded paths run
    # on small inputs (tests, repro harnesses). See issue #263.
    if _force_parallel():
        return True
    return total_bytes >= num_threads() * _MIN_BYTES_PER_THREAD
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest tests/unit/test_threads.py -k force_parallel -v`
Expected: PASS (all parametrizations).

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_threads.py tests/unit/test_threads.py
git commit -m "feat(threads): add GVL_FORCE_PARALLEL to bypass the size gate"
```

---

## Task 2: CFS-quota-aware CPU detection

**Files:**
- Modify: `python/genvarloader/_threads.py`
- Test: `tests/unit/test_threads.py`

**Interfaces:**
- Produces: `_cgroup_cpu_quota() -> int | None` (effective CPU count from a CFS quota, or `None` when no quota / unreadable). `_detect_cpus()` returns `min(affinity, quota)` when a quota is present.
- Consumes: nothing from Task 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_threads.py`:

```python
import math


def test_cgroup_quota_v2_parses_cpu_max(monkeypatch, tmp_path):
    f = tmp_path / "cpu.max"
    f.write_text("1530000 100000\n")  # 15.3 cores
    monkeypatch.setattr(th, "_CGROUP_V2_CPU_MAX", f)
    monkeypatch.setattr(th, "_CGROUP_V1_QUOTA", tmp_path / "nope_quota")
    monkeypatch.setattr(th, "_CGROUP_V1_PERIOD", tmp_path / "nope_period")
    assert th._cgroup_cpu_quota() == math.ceil(1530000 / 100000)  # 16


def test_cgroup_quota_v2_max_is_none(monkeypatch, tmp_path):
    f = tmp_path / "cpu.max"
    f.write_text("max 100000\n")  # unlimited
    monkeypatch.setattr(th, "_CGROUP_V2_CPU_MAX", f)
    monkeypatch.setattr(th, "_CGROUP_V1_QUOTA", tmp_path / "nope_quota")
    monkeypatch.setattr(th, "_CGROUP_V1_PERIOD", tmp_path / "nope_period")
    assert th._cgroup_cpu_quota() is None


def test_cgroup_quota_v1_fallback(monkeypatch, tmp_path):
    q = tmp_path / "cfs_quota_us"
    p = tmp_path / "cfs_period_us"
    q.write_text("800000\n")
    p.write_text("100000\n")
    monkeypatch.setattr(th, "_CGROUP_V2_CPU_MAX", tmp_path / "absent")
    monkeypatch.setattr(th, "_CGROUP_V1_QUOTA", q)
    monkeypatch.setattr(th, "_CGROUP_V1_PERIOD", p)
    assert th._cgroup_cpu_quota() == 8


def test_cgroup_quota_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(th, "_CGROUP_V2_CPU_MAX", tmp_path / "absent1")
    monkeypatch.setattr(th, "_CGROUP_V1_QUOTA", tmp_path / "absent2")
    monkeypatch.setattr(th, "_CGROUP_V1_PERIOD", tmp_path / "absent3")
    assert th._cgroup_cpu_quota() is None


def test_detect_cpus_takes_min_of_affinity_and_quota(monkeypatch, tmp_path):
    _constrain_detected_cpus(monkeypatch, 16)  # affinity reports 16
    monkeypatch.setattr(th, "_cgroup_cpu_quota", lambda: 15)
    assert th._detect_cpus() == 15


def test_detect_cpus_ignores_quota_when_none(monkeypatch):
    _constrain_detected_cpus(monkeypatch, 16)
    monkeypatch.setattr(th, "_cgroup_cpu_quota", lambda: None)
    assert th._detect_cpus() == 16
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/test_threads.py -k cgroup or detect_cpus -v`
Expected: FAIL — `_cgroup_cpu_quota` / module path constants not defined.

- [ ] **Step 3: Implement**

In `python/genvarloader/_threads.py`, add imports and module-level path constants near the top:

```python
import math
from pathlib import Path

# cgroup CPU-quota files (module-level so tests can repoint them).
_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota() -> int | None:
    """Effective CPU count implied by a CFS quota, or None if unlimited/unreadable.

    A CFS *quota* (cpu.max / cpu.cfs_quota_us) is invisible to
    sched_getaffinity, so a 15.3-core container still reports 16 cores by
    affinity. See issue #263.
    """
    # cgroup v2: "<quota> <period>" or "max <period>".
    try:
        raw = _CGROUP_V2_CPU_MAX.read_text().split()
    except OSError:
        raw = None
    if raw and len(raw) == 2:
        quota_s, period_s = raw
        if quota_s != "max":
            try:
                quota, period = int(quota_s), int(period_s)
            except ValueError:
                quota = period = 0
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
        else:
            return None  # explicitly unlimited

    # cgroup v1 fallback.
    quota = _read_int(_CGROUP_V1_QUOTA)
    period = _read_int(_CGROUP_V1_PERIOD)
    if quota is not None and quota > 0 and period:
        return max(1, math.ceil(quota / period))
    return None
```

Replace `_detect_cpus`:

```python
def _detect_cpus() -> int:
    try:
        affinity = max(1, len(os.sched_getaffinity(0)))  # respects cgroup cpuset (Linux)
    except AttributeError:
        affinity = max(1, os.cpu_count() or 1)
    quota = _cgroup_cpu_quota()
    if quota is not None:
        return max(1, min(affinity, quota))
    return affinity
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest tests/unit/test_threads.py -k "cgroup or detect_cpus" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_threads.py tests/unit/test_threads.py
git commit -m "fix(threads): honor CFS cpu quota in CPU detection (#263)"
```

---

## Task 3: Overwrite `RAYON_NUM_THREADS` in `cap_threads()`

**Files:**
- Modify: `python/genvarloader/_threads.py`
- Test: `tests/unit/test_threads.py`

**Interfaces:**
- Consumes: `_resolve_num_threads()` (existing), `_detect_cpus()` (Task 2).
- Produces: `cap_threads()` sets `os.environ["RAYON_NUM_THREADS"]` unconditionally (overwrite, not `setdefault`).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_threads.py`:

```python
def test_cap_threads_overwrites_ambient_rayon(monkeypatch):
    # An ambient RAYON_NUM_THREADS (base image) must NOT win over GVL's count.
    monkeypatch.setenv("RAYON_NUM_THREADS", "16")
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.setenv("GVL_NUM_THREADS", "4")
    n = th.cap_threads()
    assert n == 4
    assert os.environ["RAYON_NUM_THREADS"] == "4"


def test_cap_threads_sets_when_unset(monkeypatch):
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.setattr(th, "_NUM_THREADS", None)
    monkeypatch.setenv("GVL_NUM_THREADS", "3")
    th.cap_threads()
    assert os.environ["RAYON_NUM_THREADS"] == "3"
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/test_threads.py -k cap_threads -v`
Expected: FAIL on `test_cap_threads_overwrites_ambient_rayon` — `setdefault` leaves `"16"`.

- [ ] **Step 3: Implement**

In `python/genvarloader/_threads.py`, replace the assignment in `cap_threads`:

```python
def cap_threads() -> int:
    """Resolve worker count once and pin rayon's pool via RAYON_NUM_THREADS.

    Overwrites any ambient RAYON_NUM_THREADS: an inherited value (e.g. from a
    base image) must not defeat GVL's cgroup-aware cap (issue #263). Users who
    want explicit control set GVL_NUM_THREADS. Must run before the first rust
    parallel call (rayon reads RAYON_NUM_THREADS at global-pool init).
    Idempotent.
    """
    global _NUM_THREADS
    if _NUM_THREADS is None:
        _NUM_THREADS = _resolve_num_threads()
        os.environ["RAYON_NUM_THREADS"] = str(_NUM_THREADS)
    return _NUM_THREADS
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest tests/unit/test_threads.py -v`
Expected: PASS (whole module, including Task 1 & 2 tests).

- [ ] **Step 5: Update module docstring & commit**

In `python/genvarloader/_threads.py`, extend the module docstring to mention the two env knobs: `GVL_NUM_THREADS` (explicit worker count) and `GVL_FORCE_PARALLEL` (bypass the size gate), and that `RAYON_NUM_THREADS` is overwritten with GVL's resolved count.

```bash
git add python/genvarloader/_threads.py tests/unit/test_threads.py
git commit -m "fix(threads): overwrite ambient RAYON_NUM_THREADS with resolved cap (#263)"
```

---

## Task 4: Fix pre-existing pyrefly `unbound-name` in parity tests

**Files:**
- Modify: `tests/parity/test_gen_dataset_goldens.py`
- Modify: `tests/parity/test_variants_dataset_parity.py`

**Interfaces:** none (test-only correctness fix).

Root cause: a variable assigned in a `try` whose `except` calls `pytest.skip(...)`; pyrefly does not model `pytest.skip` as `NoReturn`, so it treats the variable as possibly-unbound at later use. Fix: make each skipping `except` branch visibly terminate with a trailing `raise` (unreachable at runtime — `skip` raises — but statically `NoReturn`).

- [ ] **Step 1: Confirm the current failure**

Run: `pixi run -e dev typecheck`
Expected: FAIL — 4 `unbound-name` errors (`ds`, `out_numba`, `golden`) in the two files.

- [ ] **Step 2: Edit `test_gen_dataset_goldens.py`**

In `test_gen_variants_af`, add a trailing `raise` after each `pytest.skip(...)` inside an `except`:

```python
    try:
        ds = ds_base.with_seqs("variants").with_settings(min_af=0.1, max_af=0.9)
    except Exception as e:
        pytest.skip(f"AF filtering unavailable: {e}")
        raise  # unreachable (skip raises); tells pyrefly this branch is NoReturn
    try:
        monkeypatch.setenv("GVL_BACKEND", "numba")
        out_numba = ds[:, :]
    except KeyError as e:
        pytest.skip(f"AF key missing: {e}")
        raise  # unreachable; NoReturn marker for pyrefly
```

- [ ] **Step 3: Edit `test_variants_dataset_parity.py`**

Apply the same trailing-`raise` fix to both skipping `except` branches (the `with_seqs(...).with_settings(...)` block and the `load_flat_golden(...)` `FileNotFoundError` block):

```python
    try:
        ds = ds_base.with_seqs("variants").with_settings(min_af=0.1, max_af=0.9)
    except Exception as e:
        pytest.skip(
            f"AF filtering unavailable on this dataset — skipping compact_keep "
            f"exercise ({type(e).__name__}: {e})"
        )
        raise  # unreachable; NoReturn marker for pyrefly
```

```python
    try:
        golden = _golden.load_flat_golden("ds_variants_af")
    except FileNotFoundError:
        pytest.skip("ds_variants_af golden not generated (AF unavailable at gen time)")
        raise  # unreachable; NoReturn marker for pyrefly
```

- [ ] **Step 4: Verify typecheck is clean**

Run: `pixi run -e dev typecheck`
Expected: PASS — zero errors.

- [ ] **Step 5: Commit (hook must pass without --no-verify)**

```bash
git add tests/parity/test_gen_dataset_goldens.py tests/parity/test_variants_dataset_parity.py
git commit -m "test(parity): mark pytest.skip except-branches NoReturn for pyrefly"
```

Expected: the pre-commit `pyrefly-check` hook passes (no `--no-verify`).

---

## Task 5: Release the GIL around the rayon FFI entry points

**Files:**
- Modify: `src/ffi/mod.rs`
- Test: `tests/parity/test_rayon_equivalence.py` (existing — used as the correctness gate, no edits)

**Interfaces:**
- The 11 parallel entry points keep their Python-visible signatures unchanged. Functions lacking a `py` token gain `py: Python<'py>` as the **first** parameter (PyO3 injects it; invisible to Python callers).
- The rayon closures touch **no Python** (verified: the load path is lock-free, operates only on `ndarray` views), so running them under `allow_threads` is sound.

**The 11 entry points** (by `pub fn`): `get_diffs_sparse`, `intervals_to_tracks`, `reconstruct_haplotypes_from_sparse`, `reconstruct_haplotypes_fused`, `reconstruct_haplotypes_spliced_fused`, `reconstruct_annotated_haplotypes_spliced_fused`, `reconstruct_annotated_haplotypes_fused`, `get_reference`, `shift_and_realign_tracks_sparse`, `tracks_to_intervals`, `intervals_and_realign_track_fused`.

Rule: resolve every `PyReadonly*/PyReadwrite*` guard to an `ndarray` view (`.as_array()` / `.as_array_mut()`) in locals **before** the closure; capture only the views (they are `Ungil`); keep the guards in the outer scope; run every `into_pyarray(py)` **after** the closure. Two shape templates follow.

- [ ] **Step 1: Baseline — confirm parity is green before the change**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/parity/test_rayon_equivalence.py -v
```
Expected: PASS (byte-identical serial==parallel==golden). This is the invariant Task 5 must preserve.

- [ ] **Step 2: Template A — in-place / no-return kernels (add `py`, wrap core)**

For `intervals_to_tracks` (currently no `py`), transform to:

```rust
pub fn intervals_to_tracks(
    py: Python<'_>,
    offset_idxs: PyReadonlyArray1<i64>,
    starts: PyReadonlyArray1<i32>,
    itv_starts: PyReadonlyArray1<i32>,
    itv_ends: PyReadonlyArray1<i32>,
    itv_values: PyReadonlyArray1<f32>,
    itv_offsets: PyReadonlyArray1<i64>,
    mut out: PyReadwriteArray1<f32>,
    out_offsets: PyReadonlyArray1<i64>,
    parallel: bool,
) {
    let offset_idxs_a = offset_idxs.as_array();
    let starts_a = starts.as_array();
    let itv_starts_a = itv_starts.as_array();
    let itv_ends_a = itv_ends.as_array();
    let itv_values_a = itv_values.as_array();
    let itv_offsets_a = itv_offsets.as_array();
    let out_a = out.as_array_mut();
    let out_offsets_a = out_offsets.as_array();
    py.allow_threads(move || {
        intervals::intervals_to_tracks(
            offset_idxs_a, starts_a, itv_starts_a, itv_ends_a, itv_values_a,
            itv_offsets_a, out_a, out_offsets_a, parallel,
        );
    });
}
```

Apply the same shape to: `reconstruct_haplotypes_from_sparse`, `shift_and_realign_tracks_sparse`, `intervals_and_realign_track_fused` (each gains `py: Python<'_>` as the first param; move all `.as_array()/.as_array_mut()` above a `py.allow_threads(move || { <core call> })`).

- [ ] **Step 3: Template B — functions that already have `py` and return arrays**

For these, run the compute into **owned** Rust arrays inside `allow_threads`, then `into_pyarray(py)` outside. Example for `reconstruct_haplotypes_fused` (the body from `get_diffs_sparse` through the optional RC is all Python-free):

```rust
    // ... resolve all `.as_array()` views into locals (as today) ...
    let (out_data, out_offsets_vec) = py.allow_threads(move || {
        // Steps 1–4b unchanged, but returning owned arrays instead of calling into_pyarray:
        // let diffs = genotypes::get_diffs_sparse(...);
        // ... offset loop building out_offsets_vec ...
        // let mut out_data = uninit_output(total);
        // reconstruct::reconstruct_haplotypes_from_sparse(out_data.view_mut(), ...);
        // if let Some(to_rc) = ... { rc_flat_rows_inplace(...); }
        (out_data, out_offsets_vec)
    });
    (out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py))
```

Apply the same restructuring (compute-in-closure, `into_pyarray` after) to: `get_diffs_sparse`, `reconstruct_haplotypes_spliced_fused`, `reconstruct_annotated_haplotypes_spliced_fused`, `reconstruct_annotated_haplotypes_fused`, `get_reference`, `tracks_to_intervals`. Keep the `geno_offsets.as_array()` + `.row(0)/.row(1)` extraction outside the closure and pass the resulting `ArrayView`s in.

- [ ] **Step 4: Compile & iterate**

Run: `pixi run -e dev cargo build --release 2>&1 | rtk err`
Expected: clean build. Common fixes if it errors:
- "closure may outlive / not `Ungil`": ensure no `PyReadonlyArray`/`PyReadwriteArray` guard and no `py` token is captured — only extracted views and POD locals.
- "cannot move out of `out` because borrowed": use `let out_a = out.as_array_mut();` then `move` the view (not `out`) into the closure.
- borrow conflicts on `geno_offsets`: bind `let go = geno_offsets.as_array();` and derive `go.row(0)/go.row(1)` into locals before the closure.

- [ ] **Step 5: Rebuild and run the parity gate**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/parity/test_rayon_equivalence.py -v
pixi run -e dev cargo-test
```
Expected: PASS — serial==parallel==golden still byte-identical; Rust unit tests green.

- [ ] **Step 6: Commit**

```bash
git add src/ffi/mod.rs
git commit -m "perf(ffi): release the GIL around rayon parallel regions"
```

---

## Task 6: Forced-parallel end-to-end equivalence test

**Files:**
- Create: `tests/integration/test_rayon_forced_parallel.py`

**Interfaces:**
- Consumes: `GVL_FORCE_PARALLEL` (Task 1); the rebuilt extension (Task 5); the `snap_dataset` fixture is in `tests/dataset/conftest.py` — this test lives under `tests/integration/`, so it builds its own dataset from the shared session fixtures (`source_bed`, `vcf_dir`, `reference` from `tests/conftest.py`).

- [ ] **Step 1: Write the test**

Create `tests/integration/test_rayon_forced_parallel.py`:

```python
"""Forced-parallel dispatch must be byte-identical to the serial size-gated path.

GVL_FORCE_PARALLEL=1 makes should_parallelize() return True, so dataset[...]
runs the real rayon path end-to-end on the small test corpus — coverage the
tiny-golden parity suite cannot reach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyBigWig
import pytest
from genoray import VCF

import genvarloader as gvl


@pytest.fixture()
def variant_track_dataset(source_bed, vcf_dir, reference, tmp_path: Path):
    """A haplotypes+track dataset whose getitem hits reconstruct + intervals_to_tracks."""
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(["s0", "s1", "s2"]):
        bw_path = tmp_path / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            value = float(i + 1)
            bw.addEntries(
                ["chr1", "chr1", "chr2", "chr2"],
                [499_990, 1_010_686, 17_320, 1_234_560],
                ends=[500_030, 1_010_706, 17_340, 1_234_580],
                values=[value, value, value, value],
            )
        bw_paths[sample] = str(bw_path)
    out = tmp_path / "ds.gvl"
    gvl.write(
        path=out,
        bed=source_bed,
        variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        tracks=gvl.BigWigs("5ss", bw_paths),
        max_jitter=2,
    )
    return out


def _materialize(ds):
    """Reduce a getitem result to a list of numpy arrays for comparison."""
    out = ds[:, :]
    items = out if isinstance(out, tuple) else (out,)
    arrays = []
    for it in items:
        # Ragged-like objects expose .data; dense are ndarrays already.
        arrays.append(np.asarray(getattr(it, "data", it)))
    return arrays


@pytest.mark.parametrize("seq_kind", ["haplotypes", "variants"])
def test_forced_parallel_matches_serial(variant_track_dataset, reference, monkeypatch, seq_kind):
    open_ds = lambda: gvl.Dataset.open(variant_track_dataset, reference=reference).with_seqs(seq_kind)

    monkeypatch.delenv("GVL_FORCE_PARALLEL", raising=False)
    serial = _materialize(open_ds())

    monkeypatch.setenv("GVL_FORCE_PARALLEL", "1")
    parallel = _materialize(open_ds())

    assert len(serial) == len(parallel)
    for s, p in zip(serial, parallel):
        np.testing.assert_array_equal(s, p)
```

- [ ] **Step 2: Rebuild the extension (if not already current for Task 5)**

Run: `pixi run -e dev maturin develop --release`

- [ ] **Step 3: Run the test**

Run: `pixi run -e dev pytest tests/integration/test_rayon_forced_parallel.py -v`
Expected: PASS. If `.with_seqs("variants")` is unsupported for this fixture, the assertion still runs on the haplotypes parametrization; if a parametrization errors on dataset construction, narrow `seq_kind` to `["haplotypes"]` and note it in the test docstring.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_rayon_forced_parallel.py
git commit -m "test(integration): forced-parallel getitem == serial (end-to-end)"
```

---

## Task 7: Spawn-worker stress reproducer

**Files:**
- Create: `tests/integration/test_rayon_stress.py`

**Interfaces:**
- Consumes: `GVL_FORCE_PARALLEL` (Task 1); the rebuilt extension (Task 5); the `variant_track_dataset`-style build (repeated here — workers open by path).
- A module-level worker function (picklable under `spawn`) that opens the dataset by path and iterates it.

- [ ] **Step 1: Write the stress test**

Create `tests/integration/test_rayon_stress.py`:

```python
"""Stress reproducer for issue #263: concurrent spawn workers iterating a
Dataset under forced-parallel + oversubscribed rayon must not deadlock.

A hang (futex-parked workers, per #263) surfaces as future.result(timeout=...)
raising TimeoutError → test failure. Clean completion across repeated launches
is evidence the cause was oversubscription (fixed by the cap_threads changes).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeoutError
from pathlib import Path

import numpy as np
import pyBigWig
import pytest
from genoray import VCF

import genvarloader as gvl

pytestmark = pytest.mark.slow

N_WORKERS = 5
ITERS_PER_WORKER = 40
LAUNCHES = 4
PER_LAUNCH_TIMEOUT_S = 120


def _iterate_dataset(ds_path: str, reference_path: str, iters: int) -> int:
    """Worker body (must be importable/picklable for spawn). Returns bytes touched."""
    # Force the parallel path and oversubscribe: many rayon threads per worker.
    os.environ["GVL_FORCE_PARALLEL"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "8"
    ds = gvl.Dataset.open(Path(ds_path), reference=Path(reference_path)).with_seqs("haplotypes")
    total = 0
    n = len(ds)
    for _ in range(iters):
        out = ds[:n, :]
        first = out[0] if isinstance(out, tuple) else out
        total += int(np.asarray(getattr(first, "data", first)).size)
    return total


@pytest.fixture()
def stress_dataset(source_bed, vcf_dir, reference, tmp_path: Path) -> tuple[Path, Path]:
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(["s0", "s1", "s2"]):
        bw_path = tmp_path / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            v = float(i + 1)
            bw.addEntries(["chr1", "chr2"], [499_990, 17_320],
                          ends=[500_030, 17_340], values=[v, v])
        bw_paths[sample] = str(bw_path)
    out = tmp_path / "stress.gvl"
    gvl.write(
        path=out, bed=source_bed, variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        tracks=gvl.BigWigs("5ss", bw_paths), max_jitter=2,
    )
    return out, Path(reference)


def test_concurrent_spawn_workers_do_not_deadlock(stress_dataset):
    ds_path, ref_path = stress_dataset
    ctx = mp.get_context("spawn")
    for launch in range(LAUNCHES):
        with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
            futs = [
                ex.submit(_iterate_dataset, str(ds_path), str(ref_path), ITERS_PER_WORKER)
                for _ in range(N_WORKERS)
            ]
            try:
                results = [f.result(timeout=PER_LAUNCH_TIMEOUT_S) for f in futs]
            except FTimeoutError:
                pytest.fail(
                    f"launch {launch}: worker did not finish within "
                    f"{PER_LAUNCH_TIMEOUT_S}s — likely the #263 rayon deadlock."
                )
            assert all(r > 0 for r in results)
```

- [ ] **Step 2: Rebuild (ensure current) and run**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/integration/test_rayon_stress.py -v -m slow
```
Expected: PASS (all launches complete under timeout). If it HANGS/fails → the #263 deadlock reproduced under test: **stop and escalate to Task 8** (do not weaken the test to make it pass).

- [ ] **Step 3: Tune only if flaky-slow (not hanging)**

If workers complete but the box is too slow for the timeout under CI load, reduce `ITERS_PER_WORKER`/`LAUNCHES` or raise `PER_LAUNCH_TIMEOUT_S` — never remove the timeout (it is the deadlock detector). Keep `N_WORKERS × RAYON_NUM_THREADS` comfortably above the core count so oversubscription is still exercised.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_rayon_stress.py
git commit -m "test(integration): spawn-worker rayon deadlock stress reproducer (#263)"
```

---

## Task 8 (contingent): root-cause fix if the stress test reproduces a hang

**Only if Task 7 reproduces a deadlock after Tasks 1–6.** Use superpowers:systematic-debugging. Given the lock-free audit, investigate in this order and add the minimal fix + a regression assertion:
1. A dependency's rayon usage on the shared global pool during the read path (e.g. `bigtools` track decode) — capture a native backtrace of a parked worker (`py-spy dump --native` / `gdb` on the hung PID; on macOS hand David a script per the profiling-handoff note).
2. rayon global-pool init races under contention — try an explicit, once-only `ThreadPoolBuilder::num_threads(cap).build_global()` seeded from `RAYON_NUM_THREADS` at import.

Do not write speculative code without a captured backtrace pinning the wait site. If Task 7 stays green across many launches, **skip this task** and record in the PR that oversubscription (fixed in Tasks 2–3) was the driver.

---

## Final integration gate (before PR)

- [ ] Rebuild: `pixi run -e dev maturin develop --release`
- [ ] Rust: `pixi run -e dev cargo-test`
- [ ] Lint/format: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/`
- [ ] Types: `pixi run -e dev typecheck` (zero errors; hook passes without `--no-verify`)
- [ ] Full tree: `pixi run -e dev pytest tests -q` (covers dataset + unit + parity; shared `_threads.py` change)
- [ ] Slow tier: `pixi run -e dev pytest tests/integration/test_rayon_stress.py -m slow -q`
- [ ] Confirm no `skills/genvarloader/SKILL.md` change needed (env-only knobs, no `__all__` change).
```
