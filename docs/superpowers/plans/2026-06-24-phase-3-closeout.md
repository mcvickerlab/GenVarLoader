# Phase 3 Close-out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `phase-3-reconstruction` to an honest, fully-rust-default state — merge the bug fixes that landed on `main` during Phase 3, lift the now-obsolete #242 test exclusions, port the one genuinely-missing kernel (`Reference.fetch`), fuse the annotated/splice haps read paths, bump seqpro to 0.20.0, and reconcile the roadmap.

**Architecture:** GVL is a Python/Rust hybrid. Hot kernels live in `src/` (pure `ndarray` cores in domain modules, PyO3 wrappers in `src/ffi/mod.rs`), exposed to Python and routed through a backend-dispatch registry (`python/genvarloader/_dispatch.py`) where each kernel registers a `numba` parity reference and a `rust` impl with `default="rust"`. The migration contract is **byte-identical parity** between backends, gated by `@pytest.mark.parity` suites that flip `GVL_BACKEND`. This plan adds two fused kernels (reuse existing cores), reroutes one path through an existing kernel, and merges upstream fixes.

**Tech Stack:** Rust (`ndarray`, `rayon`, PyO3 0.28, `numpy` 0.28, `seqpro-core` 0.1.0), Python 3.10–3.13, numba (parity refs only), pytest + hypothesis, maturin, pixi.

## Global Constraints

- **No public API change.** Nothing in `python/genvarloader/__init__.py` `__all__`, `gvl.write`, `Dataset.open`, or `Dataset.with_*` signatures changes. (Per CLAUDE.md, a public-API change would also require a `skills/genvarloader/SKILL.md` update — not expected here.)
- **Byte-identical parity** is the landing gate for every new/rerouted kernel — verified across `GVL_BACKEND=rust` and `GVL_BACKEND=numba`.
- **Do NOT delete numba parity references** (Phase 5 owns that). Exception: code with *zero callers* may be deleted (precedent: `filter_af`, `splits_sum_le_value`).
- **No new perf gate.** Phase 3 is parity-gated; throughput is recorded only.
- **seqpro version floor:** `pixi.toml` pin `==0.20.0`; `pyproject.toml` floor `>=0.20`.
- **Merge style:** merge commit, never squash (preserve history).
- **HPC test env:** dataset tests require `--basetemp=$(pwd)/.pytest_tmp` on Carter (os.link cross-device Errno 18).
- **Commands run under pixi:** `pixi run -e dev <task>`. Build the Rust ext with `pixi run -e dev maturin develop --release` (or the project's `develop` task) after Rust changes.
- **Lint/format/typecheck scope:** `ruff check python/ tests/`, `ruff format python/ tests/`, `pixi run -e dev typecheck`.
- **RTK:** prefix shell commands with `rtk` (e.g. `rtk git commit`).

---

## File-touch map

| File | Responsibility | Tasks |
|---|---|---|
| (git merge) `python/genvarloader/_dataset/_intervals.py` | resolve #242 clip-fix vs Phase 3 conflict | 1 |
| `tests/dataset/test_flat_intervals.py`, `test_seqs_tracks.py`, `test_realign_tracks.py`; `tests/unit/dataset/test_output_bytes_per_instance.py`; `tests/integration/dataset/test_dummy_dataset_insertion_fill.py` | drop `_REASON_242` xfails | 2 |
| `tests/parity/test_reconstruct_haplotypes_parity.py`, `test_shift_and_realign_tracks_parity.py` | drop #242-domain `assume(False)` guards (keep trailing-under-write guard) | 2 |
| `python/genvarloader/_dataset/_reference.py` | reroute `Reference.fetch` through dispatched `get_reference`; retire dead `_fetch_*` | 3 |
| `tests/parity/test_reference_fetch_parity.py` (new) | fetch parity backstop | 3 |
| `src/ffi/mod.rs` | add `reconstruct_annotated_haplotypes_fused`, `reconstruct_haplotypes_spliced_fused` | 4, 5 |
| `src/lib.rs` | register the two new pyfunctions | 4, 5 |
| `python/genvarloader/_dataset/_haps.py` | route annotated/splice branches to the fused entries | 4, 5 |
| `python/genvarloader/genvarloader.pyi` | stub the new pyfunctions | 4, 5 |
| `tests/parity/test_haplotypes_dataset_parity.py` | move annotated spy to fused entry; add splice fixture coverage | 4, 5 |
| `pixi.toml`, `pyproject.toml` | seqpro 0.20 bump | 6 |
| (read-path materialization sites, TBD by inventory) | `to_numpy(validate=False)` adoption | 6 |
| `docs/roadmaps/rust-migration.md` | honesty pass | 7 |

---

## Task 1: Merge `origin/main` into the branch

**Files:**
- Modify (conflict): `python/genvarloader/_dataset/_intervals.py`

**Interfaces:**
- Consumes: nothing.
- Produces: branch containing #242 clip fix (`src/intervals.rs` `intervals_to_tracks` left-clamp) + #243 SpliceIndexer fix. The fused tracks kernel `intervals_and_realign_track_fused` inherits the clip fix automatically (it calls `intervals::intervals_to_tracks`).

- [ ] **Step 1: Confirm fetch is current and review the incoming fixes**

```bash
rtk git fetch origin
rtk proxy git log --oneline HEAD..origin/main
```
Expected: the 9 commits incl. `fe83436 fix(intervals): clip sub-query interval starts` and `d814965 fix(indexing): SpliceIndexer.parse_idx double-applies sample-subset map`.

- [ ] **Step 2: Start the merge**

```bash
rtk git merge origin/main --no-edit
```
Expected: conflict in `python/genvarloader/_dataset/_intervals.py` (others auto-merge). If it reports more conflicts, resolve each by keeping BOTH main's fix and Phase 3's additions.

- [ ] **Step 3: Resolve `_intervals.py`**

Open the file. The conflict is between main's clip logic (clamp `itv.start` up to `query_start` in `_intervals_to_tracks_numba`) and Phase 3's additions (the registered `intervals_to_tracks` dispatcher block, +45 lines). Keep main's clamp inside the numba kernel AND Phase 3's dispatch registration. Verify no `<<<<<<<`/`=======`/`>>>>>>>` markers remain:

```bash
rtk proxy grep -n "<<<<<<<\|=======\|>>>>>>>" python/genvarloader/_dataset/_intervals.py
```
Expected: no output.

- [ ] **Step 4: Build and smoke-check**

```bash
rtk git add python/genvarloader/_dataset/_intervals.py
pixi run -e dev maturin develop --release 2>&1 | tail -5
```
Expected: build succeeds (`src/intervals.rs` carries the clip fix; clean Rust merge).

- [ ] **Step 5: Run the #242 kernel test from main + the intervals parity test (still xfailed at this point)**

```bash
pixi run -e dev pytest tests/unit/dataset/test_intervals_kernel.py tests/parity -k intervals -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS (this is the test PR #244 added to lock the clip fix).

- [ ] **Step 6: Complete the merge commit**

```bash
rtk git commit --no-edit
```
Expected: merge commit recorded (no squash).

---

## Task 2: Lift the now-obsolete #242 test exclusions

**Files:**
- Modify: `tests/dataset/test_flat_intervals.py`, `tests/dataset/test_seqs_tracks.py`, `tests/dataset/test_realign_tracks.py`
- Modify: `tests/unit/dataset/test_output_bytes_per_instance.py`
- Modify: `tests/integration/dataset/test_dummy_dataset_insertion_fill.py`
- Modify: `tests/parity/test_reconstruct_haplotypes_parity.py`, `tests/parity/test_shift_and_realign_tracks_parity.py`

**Interfaces:**
- Consumes: Task 1's merged #242 fix.
- Produces: the `max_jitter>0` interval domain is now real, passing coverage (no xfail).

- [ ] **Step 1: Confirm these tests now PASS as xpass (fix is in)**

```bash
pixi run -e dev pytest tests/dataset/test_realign_tracks.py tests/dataset/test_seqs_tracks.py tests/dataset/test_flat_intervals.py tests/unit/dataset/test_output_bytes_per_instance.py tests/integration/dataset/test_dummy_dataset_insertion_fill.py -q --basetemp=$(pwd)/.pytest_tmp -rX
```
Expected: the `_REASON_242`-marked tests report **XPASS** (they pass despite the xfail marker) — proof the fix resolves them. If any still genuinely FAIL, STOP and investigate (the clip fix did not cover that case — that is a real signal, do not re-xfail).

- [ ] **Step 2: Remove the `xfail` markers + `_REASON_242` constants**

In each of the 5 test files, delete the `_REASON_242 = (...)` constant and every `@pytest.mark.xfail(strict=False, reason=_REASON_242)` decorator that references it. Leave the test bodies unchanged. Example diff shape (apply per occurrence):

```python
# DELETE these lines:
_REASON_242 = (
    "mcvickerlab/GenVarLoader#242 — intervals_to_tracks itv.start<query_start "
    "..."
)
...
@pytest.mark.xfail(strict=False, reason=_REASON_242)   # DELETE this decorator
def test_something(...):
    ...
```

Verify none remain:
```bash
rtk proxy grep -rn "_REASON_242" tests/
```
Expected: no output.

- [ ] **Step 3: Remove ONLY the #242-domain `assume(False)` guards in parity tests**

In `tests/parity/test_shift_and_realign_tracks_parity.py` and `tests/parity/test_reconstruct_haplotypes_parity.py`, remove the `assume(False)` branches whose comments tie them to the `itv.start < query_start` / `start>=clen` / #242 family. **KEEP** the *reconstruct trailing-under-write* overshoot pre-check + double-init guard (that excludes a genuine numba-undefined domain, not #242). Read each `assume(False)` site's comment before deleting — when in doubt, keep it.

- [ ] **Step 4: Run the full affected set on BOTH backends**

```bash
GVL_BACKEND=rust pixi run -e dev pytest tests/dataset tests/unit/dataset tests/integration/dataset tests/parity -q --basetemp=$(pwd)/.pytest_tmp
GVL_BACKEND=numba pixi run -e dev pytest tests/dataset tests/unit/dataset tests/integration/dataset tests/parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: all PASS, 0 xfail from `_REASON_242`. (Numba may still legitimately skip the trailing-under-write domain via the retained guard.)

- [ ] **Step 5: Commit**

```bash
rtk git add tests/
rtk git commit -m "test(parity): lift obsolete #242 xfails after main clip-fix merge

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Reroute `Reference.fetch` through the dispatched rust `get_reference`

**Files:**
- Modify: `python/genvarloader/_dataset/_reference.py:119-183`
- Create: `tests/parity/test_reference_fetch_parity.py`

**Interfaces:**
- Consumes: existing `get_reference(regions, out_offsets, reference, ref_offsets, pad_char)` dispatcher (`_reference.py:743`, `default="rust"`), which packs `regions[i] = (contig_idx, start, end)` and calls the rust `reference::get_reference` core (same `padded_slice` row op as `_fetch_row`).
- Produces: `Reference.fetch` runs rust by default; numba `_fetch_impl_*` become zero-caller dead code.

- [ ] **Step 1: Write the failing parity test**

Create `tests/parity/test_reference_fetch_parity.py`:

```python
"""Parity backstop for Reference.fetch (rerouted through dispatched get_reference).

fetch builds regions=(contig_idx, start, end) and out_offsets, then calls the
same get_reference core used by the main reference read path. This test flips
GVL_BACKEND and asserts byte-identical fetched sequence across backends, with a
spy proving the rust get_reference kernel is actually invoked.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader._dataset._reference as _ref_mod
import genvarloader._dispatch as _dispatch

pytestmark = pytest.mark.parity


def test_reference_fetch_parity(reference, monkeypatch):
    ref = _ref_mod.Reference.from_path_and_contigs(reference, None) \
        if hasattr(_ref_mod.Reference, "from_path_and_contigs") \
        else _ref_mod.Reference.from_path(reference)
    contigs = ref.contigs[:1]
    starts = np.array([0], dtype=np.int64)
    ends = np.array([50], dtype=np.int64)

    numba_fn, rust_fn = _dispatch.backends("get_reference")
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return rust_fn(*a, **k)

    orig = dict(_dispatch._REGISTRY["get_reference"])
    _dispatch.register("get_reference", numba=numba_fn, rust=_spy, default="numba")
    try:
        monkeypatch.setenv("GVL_BACKEND", "rust")
        out_rust = ref.fetch(contigs, starts, ends)
        rust_calls = calls["n"]
        monkeypatch.setenv("GVL_BACKEND", "numba")
        out_numba = ref.fetch(contigs, starts, ends)
        assert calls["n"] == rust_calls, "rust spy fired during numba read"
    finally:
        _dispatch._REGISTRY["get_reference"] = orig

    assert rust_calls > 0, "rust get_reference never invoked via fetch — vacuous"
    np.testing.assert_array_equal(
        np.asarray(out_numba.data), np.asarray(out_rust.data)
    )
    np.testing.assert_array_equal(
        np.asarray(out_numba.offsets, np.int64),
        np.asarray(out_rust.offsets, np.int64),
    )
```

> Note: adapt the `Reference` construction line to the actual constructor in `_reference.py` (check `Reference.from_path*`/`__init__` and the `reference` fixture in `tests/conftest.py` before running — replace the `hasattr` shim with the real call).

- [ ] **Step 2: Run it to confirm it fails (fetch still bypasses get_reference)**

```bash
pixi run -e dev pytest tests/parity/test_reference_fetch_parity.py -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: FAIL — `rust get_reference never invoked via fetch` (fetch currently calls `_fetch_impl_*` directly).

- [ ] **Step 3: Reroute `Reference.fetch`**

In `_reference.py`, replace the kernel-selection block inside `fetch` (currently lines 135-148) with a call to the dispatched `get_reference`, assembling a `(n,3)` regions array:

```python
        lengths = ends - starts
        offsets = lengths_to_offsets(lengths)
        regions = np.stack(
            [
                np.asarray(c_idxs, np.int32),
                np.asarray(starts, np.int32),
                np.asarray(ends, np.int32),
            ],
            axis=1,
        )
        seqs = get_reference(
            regions, offsets, self.reference, self.offsets, int(self.pad_char)
        )
        seqs = Ragged.from_offsets(seqs.view("S1"), (len(contigs), None), offsets)
        return seqs
```

(`get_reference` is defined later in the same module; it is module-level, so the forward reference resolves at call time.)

- [ ] **Step 4: Delete the now-dead `_fetch_row`/`_fetch_impl_par`/`_fetch_impl_ser`**

Confirm zero callers, then remove all three numba functions (`_reference.py:155-183`):
```bash
rtk proxy grep -rn "_fetch_impl_par\|_fetch_impl_ser\|_fetch_row" python/ tests/
```
Expected after edit: no production/test references (only the definitions, which you then delete). This is zero-caller dead-code removal (allowed by the Global Constraints exception).

- [ ] **Step 5: Build + run the parity test**

```bash
pixi run -e dev maturin develop --release 2>&1 | tail -3
pixi run -e dev pytest tests/parity/test_reference_fetch_parity.py -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS.

- [ ] **Step 6: Run the spliced-ref + flat-flanks paths that use fetch**

```bash
pixi run -e dev pytest tests/ -k "splice or flank or ref" -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS (RefDataset spliced path + `_flat_flanks.py` now use rust via get_reference).

- [ ] **Step 7: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reference.py tests/parity/test_reference_fetch_parity.py
rtk git commit -m "perf(reference): route Reference.fetch through rust get_reference; drop dead _fetch_* numba

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Fuse the annotated-haps path

**Files:**
- Modify: `src/ffi/mod.rs` (add `reconstruct_annotated_haplotypes_fused`)
- Modify: `src/lib.rs` (register pyfunction)
- Modify: `python/genvarloader/_dataset/_haps.py:884-...` (route annotated non-splice branch)
- Modify: `python/genvarloader/genvarloader.pyi` (stub)
- Modify: `tests/parity/test_haplotypes_dataset_parity.py` (move annotated spy to fused entry)

**Interfaces:**
- Consumes: `reconstruct::reconstruct_haplotypes_from_sparse` core, which **already accepts `annot_v_idxs`/`annot_ref_pos`** (`src/ffi/mod.rs:474-475` currently passes `None`). Also `genotypes::get_diffs_sparse` (for output-length computation).
- Produces (exact signature, mirrors `reconstruct_haplotypes_fused` but returns 3 arrays):
  ```rust
  pub fn reconstruct_annotated_haplotypes_fused<'py>(
      py: Python<'py>,
      regions: PyReadonlyArray2<i32>, shifts: PyReadonlyArray2<i32>,
      geno_offset_idx: PyReadonlyArray2<i64>, geno_offsets: PyReadonlyArray2<i64>,
      geno_v_idxs: PyReadonlyArray1<i32>, v_starts: PyReadonlyArray1<i32>,
      ilens: PyReadonlyArray1<i32>, alt_alleles: PyReadonlyArray1<u8>,
      alt_offsets: PyReadonlyArray1<i64>, ref_: PyReadonlyArray1<u8>,
      ref_offsets: PyReadonlyArray1<i64>, pad_char: u8, output_length: i64,
      keep: Option<PyReadonlyArray1<bool>>, keep_offsets: Option<PyReadonlyArray1<i64>>,
  ) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>)
  ```
  Returns `(out_data, annot_v_idxs_data, annot_ref_pos_data, out_offsets)` — actually return 4 arrays: bytes, var_idxs (i32), ref_coords (i32), offsets (i64). The Python wrapper builds three Ragged from the shared offsets.

- [ ] **Step 1: Add the failing parity assertion (update existing annotated test to spy the fused entry)**

In `tests/parity/test_haplotypes_dataset_parity.py::test_annotated_haplotypes_mode_dataset_parity`, change the spy from the dispatched `reconstruct_haplotypes_from_sparse` to the new module-level fused entry, mirroring `test_haplotypes_mode_dataset_parity` (which spies `_haps_mod.reconstruct_haplotypes_fused`):

```python
    import genvarloader._dataset._haps as _haps_mod
    orig_fused = _haps_mod.reconstruct_annotated_haplotypes_fused
    calls = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    monkeypatch.setattr(
        _haps_mod, "reconstruct_annotated_haplotypes_fused", _spy_fused
    )
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]
    rust_call_count = calls["n"]
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]
    assert calls["n"] == rust_call_count, "fused spy fired during numba read"
    assert calls["n"] > 0, "rust annotated fused entry never invoked — vacuous"
```
Keep the existing three-array byte-identical comparison (`_compare_ragged_bytes` + two `_compare_ragged_int`).

- [ ] **Step 2: Run it to confirm it fails**

```bash
pixi run -e dev pytest tests/parity/test_haplotypes_dataset_parity.py::test_annotated_haplotypes_mode_dataset_parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: FAIL — `AttributeError: ... has no attribute 'reconstruct_annotated_haplotypes_fused'`.

- [ ] **Step 3: Implement the rust fused kernel**

In `src/ffi/mod.rs`, add `reconstruct_annotated_haplotypes_fused` by copying `reconstruct_haplotypes_fused` (lines 373-480) and making exactly these changes:
1. Add the 4-array return type (bytes, i32 var_idxs, i32 ref_coords, i64 offsets).
2. After allocating `out_data`, also allocate `let mut annot_v: Array1<i32> = Array1::zeros(total);` and `let mut annot_pos: Array1<i32> = Array1::zeros(total);`.
3. In the `reconstruct::reconstruct_haplotypes_from_sparse(...)` call, replace the two trailing `None,  // annot_*` args with `Some(annot_v.view_mut()), Some(annot_pos.view_mut())` (match the core's expected `Option<ArrayViewMut1<i32>>` param types — check `src/reconstruct/mod.rs:282` signature and adapt).
4. Return `(out_data.into_pyarray(py), annot_v.into_pyarray(py), annot_pos.into_pyarray(py), out_offsets_vec.into_pyarray(py))`.

- [ ] **Step 4: Register the pyfunction**

In `src/lib.rs` after line 38 (`reconstruct_haplotypes_fused`):
```rust
    m.add_function(wrap_pyfunction!(ffi::reconstruct_annotated_haplotypes_fused, m)?)?;
```

- [ ] **Step 5: Add the `.pyi` stub**

In `python/genvarloader/genvarloader.pyi`, add a stub mirroring the existing `reconstruct_haplotypes_fused` stub but with the 4-tuple return (`tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.int64]]`).

- [ ] **Step 6: Route the Python annotated branch to the fused entry**

In `_haps.py::_reconstruct_annotated_haplotypes` (non-splice branch, currently lines 895-919), add a `_backend = os.environ.get("GVL_BACKEND", "rust")` check mirroring `_reconstruct_haplotypes` (lines 773-817). When rust: call `reconstruct_annotated_haplotypes_fused(...)` (import it at module top alongside `reconstruct_haplotypes_fused`), wrap the 3 returned data arrays into Ragged via the shared `out_offsets`, and return the `RaggedAnnotatedHaps`-equivalent tuple. When numba: keep the existing composed `reconstruct_haplotypes_from_sparse(...)` call unchanged.

- [ ] **Step 7: Build + run the parity test**

```bash
pixi run -e dev maturin develop --release 2>&1 | tail -3
pixi run -e dev pytest tests/parity/test_haplotypes_dataset_parity.py::test_annotated_haplotypes_mode_dataset_parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS (byte-identical haps + var_idxs + ref_coords; fused spy fired).

- [ ] **Step 8: Run cargo + annotated integration tests**

```bash
rtk cargo test 2>&1 | tail -5
pixi run -e dev pytest tests/ -k "annot" -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
rtk git add src/ffi/mod.rs src/lib.rs python/genvarloader/genvarloader.pyi python/genvarloader/_dataset/_haps.py tests/parity/test_haplotypes_dataset_parity.py
rtk git commit -m "perf(reconstruct): fused annotated-haps __getitem__ kernel (dataset parity)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Fuse the splice haps path

**Files:**
- Modify: `src/ffi/mod.rs` (add `reconstruct_haplotypes_spliced_fused`)
- Modify: `src/lib.rs` (register)
- Modify: `python/genvarloader/_dataset/_haps.py:846-882` (route splice branch)
- Modify: `python/genvarloader/genvarloader.pyi` (stub)
- Create: `tests/parity/test_spliced_haplotypes_parity.py`

**Interfaces:**
- Consumes: `reconstruct::reconstruct_haplotypes_from_sparse` core. The Python side already computes the splice permutation (`_permute_request_for_splice` → `flat_geno_idx`, `flat_shifts`, `permuted_regions`, `keep_perm`, `keep_offsets_perm`) and `splice_plan.permuted_out_offsets`. **The permutation stays in Python**; only the reconstruction FFI crossing fuses.
- Produces (the splice variant takes precomputed `out_offsets` instead of computing diffs):
  ```rust
  pub fn reconstruct_haplotypes_spliced_fused<'py>(
      py: Python<'py>,
      permuted_regions: PyReadonlyArray2<i32>,   // (n_perm, 3)
      flat_shifts: PyReadonlyArray2<i32>,        // (n_perm, 1)
      flat_geno_offset_idx: PyReadonlyArray2<i64>, // (n_perm, 1)
      out_offsets: PyReadonlyArray1<i64>,        // permuted_out_offsets (n_perm+1)
      geno_offsets: PyReadonlyArray2<i64>, geno_v_idxs: PyReadonlyArray1<i32>,
      v_starts: PyReadonlyArray1<i32>, ilens: PyReadonlyArray1<i32>,
      alt_alleles: PyReadonlyArray1<u8>, alt_offsets: PyReadonlyArray1<i64>,
      ref_: PyReadonlyArray1<u8>, ref_offsets: PyReadonlyArray1<i64>, pad_char: u8,
      keep: Option<PyReadonlyArray1<bool>>, keep_offsets: Option<PyReadonlyArray1<i64>>,
  ) -> Bound<'py, PyArray1<u8>>   // out_data only; caller already has out_offsets
  ```

- [ ] **Step 1: Write the failing splice parity test**

Create `tests/parity/test_spliced_haplotypes_parity.py`. It needs a spliced dataset fixture. Check `tests/conftest.py` / `tests/parity/conftest.py` for an existing `splice_info`-bearing fixture; if none exists, build one from the existing `phased_svar_gvl` by opening with a minimal synthetic `splice_info` (transcript-ID grouping over the BED regions). Mirror `test_haplotypes_dataset_parity.py` structure, spying `_haps_mod.reconstruct_haplotypes_spliced_fused`:

```python
"""Spliced-haplotypes dataset parity backstop (fused rust splice entry)."""
from __future__ import annotations
import numpy as np
import pytest
import genvarloader as gvl
import genvarloader._dataset._haps as _haps_mod

pytestmark = pytest.mark.parity


def test_spliced_haplotypes_parity(spliced_gvl, reference, monkeypatch):
    ds = gvl.Dataset.open(spliced_gvl, reference=reference).with_seqs("haplotypes")
    orig = _haps_mod.reconstruct_haplotypes_spliced_fused
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_haps_mod, "reconstruct_haplotypes_spliced_fused", _spy)
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]
    rc = calls["n"]
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]
    assert calls["n"] == rc, "fused splice spy fired during numba read"
    assert calls["n"] > 0, "rust spliced fused entry never invoked — vacuous"
    np.testing.assert_array_equal(
        np.asarray(out_numba.data), np.asarray(out_rust.data)
    )
    np.testing.assert_array_equal(
        np.asarray(out_numba.offsets, np.int64),
        np.asarray(out_rust.offsets, np.int64),
    )
```

> If building a synthetic spliced fixture proves disproportionate, STOP and report — per the spec, splice fusion may fall back to the documented unfused-rust path with an honest roadmap note rather than blocking the plan.

- [ ] **Step 2: Run it to confirm it fails**

```bash
pixi run -e dev pytest tests/parity/test_spliced_haplotypes_parity.py -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: FAIL — `AttributeError: ... reconstruct_haplotypes_spliced_fused`.

- [ ] **Step 3: Implement the rust splice fused kernel**

In `src/ffi/mod.rs`, add `reconstruct_haplotypes_spliced_fused`. It is `reconstruct_haplotypes_fused` **without** the diff/out-offset computation (Steps 1-2 of that fn): the caller passes `out_offsets` directly. Body:
1. `let out_offsets_a = out_offsets.as_array();` `let total = out_offsets_a[out_offsets_a.len()-1] as usize;`
2. `let mut out_data: Array1<u8> = Array1::zeros(total);`
3. Call `reconstruct::reconstruct_haplotypes_from_sparse(out_data.view_mut(), out_offsets_a, permuted_regions.as_array(), flat_shifts.as_array(), flat_geno_offset_idx.as_array(), go_starts, go_stops, geno_v_idxs.as_array(), v_starts.as_array(), ilens.as_array(), alt_alleles.as_array(), alt_offsets.as_array(), ref_.as_array(), ref_offsets.as_array(), pad_char, keep.as_ref().map(|k| k.as_array()), keep_offsets.as_ref().map(|ko| ko.as_array()), None, None);`
4. `out_data.into_pyarray(py)`

- [ ] **Step 4: Register + stub**

`src/lib.rs`: `m.add_function(wrap_pyfunction!(ffi::reconstruct_haplotypes_spliced_fused, m)?)?;`
`genvarloader.pyi`: stub returning `NDArray[np.uint8]`.

- [ ] **Step 5: Route the Python splice branch**

In `_haps.py::_reconstruct_haplotypes` splice-plan branch (lines 846-882), add a `_backend` check. When rust: after `_permute_request_for_splice`, call `reconstruct_haplotypes_spliced_fused(...)` (import at top) with the permuted arrays + `splice_plan.permuted_out_offsets`, then wrap into the `_Flat.from_offsets(out_buf, per_elem_shape, splice_plan.permuted_out_offsets).view("S1")` as today. When numba: keep the existing composed `reconstruct_haplotypes_from_sparse(...)` call unchanged.

- [ ] **Step 6: Build + run the splice parity test**

```bash
pixi run -e dev maturin develop --release 2>&1 | tail -3
pixi run -e dev pytest tests/parity/test_spliced_haplotypes_parity.py -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS.

- [ ] **Step 7: Cargo + splice integration tests**

```bash
rtk cargo test 2>&1 | tail -5
pixi run -e dev pytest tests/ -k splice -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add src/ffi/mod.rs src/lib.rs python/genvarloader/genvarloader.pyi python/genvarloader/_dataset/_haps.py tests/parity/test_spliced_haplotypes_parity.py tests/conftest.py
rtk git commit -m "perf(reconstruct): fused spliced-haps __getitem__ kernel (dataset parity)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Bump seqpro to 0.20.0 + adopt `to_numpy(validate=False)`

**Files:**
- Modify: `pixi.toml:91`, `pyproject.toml:13`
- Modify: read-path materialization sites (determined by inventory in Step 3)

**Interfaces:**
- Consumes: seqpro 0.20.0's `to_numpy(validate=False)` (skips the uniformity scan).
- Produces: faster fixed-length materialization where row uniformity is guaranteed.

- [ ] **Step 1: Bump the pins**

`pixi.toml:91`: `seqpro = "==0.18.0"` → `seqpro = "==0.20.0"`.
`pyproject.toml:13`: `"seqpro>=0.18",` → `"seqpro>=0.20",`.

```bash
pixi install -e dev 2>&1 | tail -5
pixi run -e dev python -c "import seqpro; print(seqpro.__version__)"
```
Expected: `0.20.0`.

- [ ] **Step 2: Verify seqpro-core Rust layout still matches**

```bash
pixi run -e dev maturin develop --release 2>&1 | tail -3
rtk cargo test 2>&1 | tail -5
GVL_BACKEND=rust pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: build + cargo + parity all PASS (proves the `seqpro-core` 0.1.0 `Ragged` layout still matches 0.20.0). If parity breaks, STOP — the layout drifted and needs a `seqpro-core` bump (out of this plan's scope; report).

- [ ] **Step 3: Inventory guaranteed-uniform `.to_numpy()` / materialization sites**

```bash
rtk proxy grep -rn "to_numpy\|to_padded\|to_fixed\|\.to_fixed(" python/genvarloader/
```
Identify sites on the read path where row lengths are uniform *by construction* (fixed-length / `with_len(L)` output, padded materialization). Produce a short list with file:line and a one-line justification each. **Do not edit yet** — these are the propose-then-approve candidates per the spec.

- [ ] **Step 4: STOP and present the candidate list to the maintainer for approval**

Present the inventory. Apply `validate=False` only to approved sites. (If the maintainer defers, skip to Step 6 with just the version bump.)

- [ ] **Step 5: Apply `validate=False` at approved sites + re-verify parity**

For each approved site, add `validate=False` to the `to_numpy(...)` call. Then:
```bash
GVL_BACKEND=rust pixi run -e dev pytest tests/dataset tests/unit/dataset tests/parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS (output unchanged — `validate=False` only skips the scan, never changes data).

- [ ] **Step 6: Commit**

```bash
rtk git add pixi.toml pyproject.toml pixi.lock python/genvarloader/
rtk git commit -m "build(seqpro): bump to 0.20.0; adopt to_numpy(validate=False) on uniform read-path sites

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Roadmap honesty pass + full-tree verification

**Files:**
- Modify: `docs/roadmaps/rust-migration.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: roadmap consistent with reality; full green tree on both backends.

- [ ] **Step 1: Full-tree verification on BOTH backends**

```bash
GVL_BACKEND=rust pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | tail -15
GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | tail -15
rtk cargo test 2>&1 | tail -5
```
Expected: all PASS; the only remaining xfails are the genuine non-#242 ones (trailing-under-write numba domain, `test_e2e_variants` if still pre-existing). Record counts.

- [ ] **Step 2: Lint / format / typecheck**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck 2>&1 | tail -10
```
Expected: clean.

- [ ] **Step 3: Confirm abi3 wheel builds**

```bash
pixi run -e dev maturin build --release 2>&1 | tail -5
```
Expected: wheel builds.

- [ ] **Step 4: Reconcile the Phase 3 section of the roadmap**

In `docs/roadmaps/rust-migration.md` Phase 3 section (lines ~270-312):
- Check off item "Migrate `_dataset/_reconstruct.py` + `_dataset/_haps.py` remaining paths" — note annotated + splice now fused (Tasks 4-5).
- Reword the `_tracks.py`/`_intervals.py` item: rust-default + fused; remaining numba are Phase-5-deletion parity refs.
- Check off the `_reference.py` item — note `Reference.fetch` rerouted through rust `get_reference`; `_fetch_*` numba deleted (zero callers).
- Check off the `_insertion_fill.py` + `_splice.py` item (no numba kernels; splice fused via Task 5) — OR, if splice fusion fell back per Task 5 Step 1, mark it "rust-default, fusion deferred to Phase 5" with the honest note.
- Resolve the `✅`-header / unchecked-box contradiction so the marker matches the boxes.

- [ ] **Step 5: Add a dated decisions-log entry**

Append to the "Notes & decisions log" (top entry, dated 2026-06-24):
```
- 2026-06-24 (Phase 3 close-out): Merged origin/main (#242 intervals_to_tracks
  clip fix via PR #244; SpliceIndexer subset double-apply fix via PR #243) into
  the branch — the fused tracks kernel inherits the clip fix (shared
  intervals::intervals_to_tracks core). Lifted ~10 obsolete #242 xfails +
  #242-domain assume(False) guards → real passing max_jitter>0 coverage.
  Rerouted Reference.fetch through the dispatched rust get_reference (deleted
  zero-caller _fetch_* numba). Fused the annotated-haps
  (reconstruct_annotated_haplotypes_fused) and spliced-haps
  (reconstruct_haplotypes_spliced_fused) read paths — both byte-identical to the
  composed numba oracle. Bumped seqpro 0.18->0.20.0 with to_numpy(validate=False)
  on guaranteed-uniform read-path sites. Full tree green on both backends.
```

- [ ] **Step 6: Confirm no public-API change (skill check)**

```bash
rtk proxy git diff origin/main..HEAD -- python/genvarloader/__init__.py
```
Expected: no change to `__all__` / exports → `skills/genvarloader/SKILL.md` needs no update (per CLAUDE.md). If anything changed, update the skill.

- [ ] **Step 7: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): Phase 3 close-out — honest item status, decisions log

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** Step1→Task1 (merge), Step2→Task2 (xfails), Step3→Task3 (Reference.fetch), Step4→Tasks4-5 (fusion), Step5→Task6 (seqpro), Step6→Task7 (roadmap/skill). All spec steps mapped.
- **Simplifications found during planning (vs spec):** (a) the #242 fix needs **no** manual Rust propagation — the fused tracks kernel reuses the shared core; (b) `Reference.fetch` needs **no new rust kernel** — it reroutes through the existing dispatched `get_reference`; (c) the reconstruct core **already** accepts annot buffers, so annotated fusion is a thin wrapper. These reduce risk; the spec's more cautious framing still holds.
- **Fallback honored:** Task 5 Step 1 explicitly allows splice fusion to fall back to documented unfused-rust if a synthetic spliced fixture is disproportionate (matches spec risk mitigation).
- **Type consistency:** new entries named consistently — `reconstruct_annotated_haplotypes_fused` (Task 4) and `reconstruct_haplotypes_spliced_fused` (Task 5) used identically in ffi/lib.rs/_haps.py/pyi/tests.
