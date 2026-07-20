# SVAR2 streaming Phase-2 PR 1 (fast synchronous path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the synchronous SVAR2 `StreamingDataset` path fast — pass `parallel=False` for the per-batch reconstruct (PR 1a), and move the per-window range query into GIL-free Rust via `genoray_core::query::find_ranges` (PR 1b) — with byte-identical parity preserved.

**Architecture:** Two stacked PRs on top of the landed Phase-1 synchronous backend. PR 1a is a one-line, streaming-only flag flip guarded by the existing parity suite. PR 1b adds a new `svar2_read_window` Rust FFI function (mirroring the existing `svar1_read_window`) that calls genoray's public Rust `find_ranges`, then rewires `_Svar2Backend.read_window` to use it and deletes the Python `SparseVar2._find_ranges` call + numpy reshape glue. No new Rust *kernels* — the read-bound reconstruction FFI is reused unchanged.

**Tech Stack:** Python 3.10+ (abi3), Rust via PyO3/maturin, `genoray_core::query` (Rust, already linked at the pinned rev), `genoray.SparseVar2` (Phase-1 comparison oracle), numpy, `seqpro.rag.Ragged`, pytest.

## Global Constraints

- **Target branch:** `streaming` (not `main`). Streaming PRs merge into `streaming` per `CLAUDE.md`.
- **Correctness oracle (hard gate):** byte-identical parity vs `gvl.write()` + `Dataset.open()[r, s]` under `.with_seqs("haplotypes")`, jitter=0. `tests/dataset/test_streaming_parity_svar2.py` and the #284 gate `tests/dataset/test_streaming_scale.py::test_svar2_generate_batch_output_is_flat_in_cohort_size` must stay green. A faster variant that fails parity is a bug, not a feature.
- **No genoray rev bump.** `Cargo.toml` stays pinned at `rev = e07477e687c913f9605fc79ea251f1bb3b177aa9`. `genoray_core::query::find_ranges` is public at that rev.
- **Streaming-only flag.** The written-`Dataset` path (`_svar2_haps.py`) keeps `parallel=True`; only the streaming call site changes. Do **not** change any kernel default.
- **Rebuild Rust before testing Rust changes (PR 1b only):** after editing `src/`, run `pixi run -e dev maturin develop --release` **before** `pixi run -e dev pytest`, or pytest imports the stale extension (`CLAUDE.md`). PR 1a is pure-Python — no rebuild.
- **Testing commands:** `pixi run -e dev pytest <path> -v`. Before pushing: `pixi run -e dev pytest tests/dataset tests/unit -q`. Lint/format/types: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck`.
- **Commit hooks:** ensure `prek install` has run in this worktree before the first commit (`.pre-commit-config.yaml` present).
- **Sample-index convention:** public `sample_idx` indexes the **lexicographically sorted** sample-name order; translate to the store's physical column before any Rust query (PR 1b introduces `_phys_sample_idx` for exactly this).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `python/genvarloader/_dataset/_streaming.py` | 1a: flip the `parallel` arg in `_Svar2Backend.generate_batch` (`:1113`). 1b: add `_phys_sample_idx` to `_Svar2Backend.__init__`; rewire `_Svar2Backend.read_window` (`:1008-1047`) to call `svar2_read_window`; delete the `_find_ranges` glue. | 1, 3 |
| `src/ffi/mod.rs` | 1b: add `pub fn svar2_read_window` (mirror `svar1_read_window`), calling `genoray_core::query::find_ranges`. | 2 |
| `src/lib.rs` | 1b: register `ffi::svar2_read_window` (after `svar1_generate_batch`, `:53`). | 2 |
| `tests/dataset/test_streaming_phase2_pr1.py` | 1b: smoke (shape/dtype of `svar2_read_window`) + byte-equivalence (Rust path == old `_find_ranges` path). | 2, 3 |

**Parallelism (per project convention — dispatch with `superpowers:dispatching-parallel-agents` + `subagent-driven-development`, Sonnet or weaker for implementers):**

- Task 1 (PR 1a, pure-Python flag) and Task 2's **Rust function** (new symbol in `src/ffi/mod.rs` + `src/lib.rs` + a self-contained smoke test) touch disjoint code and can be implemented in parallel by two subagents.
- Task 3 (Python rewire) depends on Task 2's compiled symbol and rebases on Task 1 (same file, different functions: `generate_batch` vs `read_window`/`__init__`). Land order is stacked: PR 1a (Task 1) → PR 1b (Tasks 2 + 3).
- This PR-1 plan is deliberately small; the larger parallelizable work (super-batch reconstruction) is PR 2's separate plan.

---

## Task 1: PR 1a — `parallel=False` for the streaming per-batch reconstruct

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py:1113` (the `parallel` arg of `reconstruct_haplotypes_from_svar2_readbound` inside `_Svar2Backend.generate_batch`)

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature change; behavior change only (streaming reconstruct runs single-threaded).

**Why:** measured this session — `parallel=True` forks the ~96-thread rayon pool for ~64 tiny haplotypes/call; `parallel=False` is 1.2–1.8× faster and uses one core, byte-parity-identical (bs=8 1.79×, bs=32 1.44×, bs=128 1.22×). This is a per-call flag, streaming-only. Parity is the invariant, so the "test" is the parity suite as a regression guard (a perf flag is not unit-testable, but it MUST NOT change output).

- [ ] **Step 1: Confirm the parity suite is green before the change (baseline)**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py -v`
Expected: PASS (2 tests). This is the invariant the flag must not break.

- [ ] **Step 2: Flip the flag**

In `_Svar2Backend.generate_batch`, change the `parallel` argument (currently `True,  # parallel` at `:1113`) to:

```python
            False,  # parallel: streaming per-batch reconstruct is tiny (~batch_size*ploidy
            #        haplotypes); the 96-thread rayon fork/join costs more than it saves
            #        here (measured 1.2-1.8x faster serial). The written-Dataset path
            #        (_svar2_haps.py) keeps parallel=True (its getitem chunks are large).
```

(Keep the following `False,  # filter_exonic` line unchanged.)

- [ ] **Step 3: Run parity to verify output is unchanged**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py tests/dataset/test_streaming_scale.py -k "svar2 or streaming" -v`
Expected: PASS. Byte-identical parity + #284 scale gate green — proves the flag does not change output.

- [ ] **Step 4: Lint/format/type gate**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py
git commit -m "perf(streaming): SVAR2 per-batch reconstruct is serial (parallel=False) (#278)"
```

---

## Task 2: PR 1b(i) — `svar2_read_window` Rust FFI over `genoray_core::query::find_ranges`

**Files:**
- Modify: `src/ffi/mod.rs` (add `pub fn svar2_read_window`, mirroring `svar1_read_window`)
- Modify: `src/lib.rs:53` (register the function after `ffi::svar1_generate_batch`)
- Create: `tests/dataset/test_streaming_phase2_pr1.py` (shape/dtype smoke)

**Interfaces:**
- Consumes: `Svar2Store` pyclass (`src/svar2/store.rs`, exposes `reader(contig) -> Option<&ContigReader>`); `genoray_core::query::find_ranges(reader: &ContigReader, regions: &[(u32,u32)], samples: Option<&[usize]>) -> RangesBundle` (fields used: `vk_snp_range`/`vk_indel_range`/`dense_snp_range`/`dense_indel_range: Vec<Range<usize>>`, `sample_cols: Vec<usize>`); the existing `require_contiguous_1d` helper and imports already used by `svar1_read_window` (`PyReadonlyArray1`, `PyArray1`, `Array1`, `into_pyarray`).
- Produces: `svar2_read_window(store, contig, starts, ends, sample_idx) -> (vk_snp, vk_indel, dense_snp, dense_indel, sample_cols)` — five `PyArray1<i64>`. `vk_snp`/`vk_indel` are flat `[start, stop, ...]` of length `n_reg*n_s*P*2` in `(region, sample, ploid)` C-order; `dense_snp`/`dense_indel` flat length `n_reg*2`; `sample_cols` length `n_s`. Consumed by Task 3's rewired `read_window`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/dataset/test_streaming_phase2_pr1.py
"""PR 1b: svar2_read_window Rust FFI — shape/dtype smoke + byte-equivalence vs the
Phase-1 name-based SparseVar2._find_ranges path."""
from __future__ import annotations

import numpy as np


def test_svar2_read_window_shapes(svar2_multicontig_fixture) -> None:
    from genoray import SparseVar2
    from genvarloader.genvarloader import Svar2Store, svar2_read_window

    fx = svar2_multicontig_fixture
    sv = SparseVar2(str(fx.svar2_path))
    ploidy = int(sv.ploidy)
    store = Svar2Store(str(fx.svar2_path), sv.contigs, sv.n_samples, ploidy)

    # One contig window: chr1 regions [0,20) and [4,24); all physical samples 0..n.
    contig = "chr1"
    starts = np.array([0, 4], np.uint32)
    ends = np.array([20, 24], np.uint32)
    phys = np.arange(sv.n_samples, dtype=np.int64)
    n_reg, n_s = len(starts), len(phys)

    vk_snp, vk_indel, dense_snp, dense_indel, sample_cols = svar2_read_window(
        store, contig, starts, ends, phys
    )
    for a in (vk_snp, vk_indel, dense_snp, dense_indel, sample_cols):
        assert np.asarray(a).dtype == np.int64
    assert np.asarray(vk_snp).size == n_reg * n_s * ploidy * 2
    assert np.asarray(vk_indel).size == n_reg * n_s * ploidy * 2
    assert np.asarray(dense_snp).size == n_reg * 2
    assert np.asarray(dense_indel).size == n_reg * 2
    assert np.asarray(sample_cols).size == n_s
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr1.py::test_svar2_read_window_shapes -v`
Expected: FAIL — `ImportError: cannot import name 'svar2_read_window'` (not yet defined).

- [ ] **Step 3: Add the Rust function**

In `src/ffi/mod.rs`, add (mirrors `svar1_read_window`; place adjacent to it):

```rust
/// Compute a window's SVAR2 read-bound ranges in Rust (GIL released), replacing the
/// Python `SparseVar2._find_ranges` call. `sample_idx` are PHYSICAL store columns
/// (public sorted-name -> physical translation happens Python-side). Returns flat i64
/// range arrays: vk_snp/vk_indel are `[start, stop, ...]` in (region, sample, ploid)
/// C-order (len n_reg*n_s*P*2); dense_snp/dense_indel per region (len n_reg*2);
/// sample_cols len n_s. No genoray rev bump: `find_ranges` is public at the pinned rev.
#[pyfunction]
pub fn svar2_read_window<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar2::store::Svar2Store>,
    contig: &str,
    starts: PyReadonlyArray1<u32>,
    ends: PyReadonlyArray1<u32>,
    sample_idx: PyReadonlyArray1<i64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    require_contiguous_1d(&starts, "starts")?;
    require_contiguous_1d(&ends, "ends")?;

    let starts_a = starts.as_array();
    let ends_a = ends.as_array();
    let regions_v: Vec<(u32, u32)> = starts_a
        .iter()
        .zip(ends_a.iter())
        .map(|(&s, &e)| (s, e))
        .collect();
    let samples_v: Vec<usize> = sample_idx.as_array().iter().map(|&s| s as usize).collect();

    let store_ref: &crate::svar2::store::Svar2Store = &store;

    let result = py.detach(
        move || -> anyhow::Result<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>)> {
            let reader = store_ref
                .reader(contig)
                .ok_or_else(|| anyhow::anyhow!("no reader for contig {contig}"))?;
            let rb = genoray_core::query::find_ranges(reader, &regions_v, Some(&samples_v));
            let flat = |v: &[std::ops::Range<usize>]| -> Vec<i64> {
                let mut out = Vec::with_capacity(v.len() * 2);
                for r in v {
                    out.push(r.start as i64);
                    out.push(r.end as i64);
                }
                out
            };
            Ok((
                flat(&rb.vk_snp_range),
                flat(&rb.vk_indel_range),
                flat(&rb.dense_snp_range),
                flat(&rb.dense_indel_range),
                rb.sample_cols.iter().map(|&c| c as i64).collect(),
            ))
        },
    );

    let (vk_snp, vk_indel, dense_snp, dense_indel, sample_cols) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        Array1::from_vec(vk_snp).into_pyarray(py),
        Array1::from_vec(vk_indel).into_pyarray(py),
        Array1::from_vec(dense_snp).into_pyarray(py),
        Array1::from_vec(dense_indel).into_pyarray(py),
        Array1::from_vec(sample_cols).into_pyarray(py),
    ))
}
```

- [ ] **Step 4: Register the function in `src/lib.rs`**

After `m.add_function(wrap_pyfunction!(ffi::svar1_generate_batch, m)?)?;` (`:53`), add:

```rust
    m.add_function(wrap_pyfunction!(ffi::svar2_read_window, m)?)?;
```

- [ ] **Step 5: Rebuild the Rust extension**

Run: `pixi run -e dev maturin develop --release`
Expected: builds cleanly (no errors). If `genoray_core::query::find_ranges` is unresolved, confirm the import path (`gather_haps_readbound` is already used from `genoray_core::query` in this file — `find_ranges` is re-exported alongside it).

- [ ] **Step 6: Run the smoke test to green**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr1.py::test_svar2_read_window_shapes -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ffi/mod.rs src/lib.rs tests/dataset/test_streaming_phase2_pr1.py
git commit -m "feat(streaming): svar2_read_window Rust FFI over genoray find_ranges (#278)"
```

---

## Task 3: PR 1b(ii) — rewire `_Svar2Backend.read_window` to the Rust FFI; delete the Python glue

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_Svar2Backend.__init__` — add `_phys_sample_idx`; `_Svar2Backend.read_window` `:1008-1047` — call `svar2_read_window`, drop the `_find_ranges` call + reshape glue)
- Modify: `tests/dataset/test_streaming_phase2_pr1.py` (add the byte-equivalence test)

**Interfaces:**
- Consumes: `svar2_read_window` (Task 2); `self._sv.available_samples` (native/VCF column order), `self._sample_names` (sorted), `self._sv.ploidy`, `self._store`, `self._regions`, `self._contig_of` (all already on `_Svar2Backend`).
- Produces: `read_window(r_idx, s_idx) -> dict` with the SAME keys/shapes as before (`contig_idx`, `region_bounds` `(n_reg,2)` i32, `orig_samples` `(n_s,)` i64, `vk_snp`/`vk_indel` `(n_reg,n_s,P,2)` i64, `dense_snp`/`dense_indel` `(n_reg,2)` i64) — so `generate_batch` is untouched. New attr `_phys_sample_idx: NDArray[np.int64]` (sorted-name position -> physical store column).

**The correctness seam:** the old path passed sample NAMES to `SparseVar2._find_ranges`, which resolved them to physical columns internally. The Rust `find_ranges` takes physical `usize` indices, so `__init__` must build `_phys_sample_idx` (`available_samples` position of each sorted name) and `read_window` passes `_phys_sample_idx[s_idx]`. The byte-equivalence test below is the oracle that this mapping matches genoray's name resolution; if it fails, the native-order source (`available_samples`) or the mapping is wrong.

- [ ] **Step 1: Write the failing byte-equivalence test**

Add to `tests/dataset/test_streaming_phase2_pr1.py`:

```python
def test_svar2_read_window_matches_find_ranges(svar2_multicontig_fixture) -> None:
    """The rewired Rust read_window is byte-identical to the Phase-1 name-based
    SparseVar2._find_ranges path for the same window."""
    import genvarloader as gvl

    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert backend is not None

    # Reference (old) implementation: name-based _find_ranges, reshaped as Phase 1 did.
    def old_read_window(r_idx, s_idx):
        r_idx = np.asarray(r_idx, np.intp)
        s_idx = np.asarray(s_idx, np.intp)
        contig_idx, contig = backend._contig_of(r_idx)
        rb = backend._regions[r_idx, 1:3]
        starts = np.ascontiguousarray(rb[:, 0])
        ends = np.ascontiguousarray(rb[:, 1])
        names = [backend._sample_names[i] for i in s_idx]
        d = backend._sv._find_ranges(contig, starts, ends, samples=names)
        n_reg, n_s, P = len(r_idx), len(s_idx), backend.ploidy
        return {
            "orig_samples": np.ascontiguousarray(d["sample_cols"], np.int64),
            "vk_snp": np.asarray(d["vk_snp_range"], np.int64).reshape(n_reg, n_s, P, 2),
            "vk_indel": np.asarray(d["vk_indel_range"], np.int64).reshape(n_reg, n_s, P, 2),
            "dense_snp": np.asarray(d["dense_snp_range"], np.int64).reshape(n_reg, 2),
            "dense_indel": np.asarray(d["dense_indel_range"], np.int64).reshape(n_reg, 2),
        }

    for r_idx, s_idx in sds._plan():
        new = backend.read_window(r_idx, s_idx)   # rewired (Rust) path
        old = old_read_window(r_idx, s_idx)
        for k in ("orig_samples", "vk_snp", "vk_indel", "dense_snp", "dense_indel"):
            np.testing.assert_array_equal(
                np.asarray(new[k]), old[k], err_msg=f"mismatch in {k}"
            )
```

- [ ] **Step 2: Run the equivalence test as a characterization baseline (should PASS now)**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr1.py::test_svar2_read_window_matches_find_ranges -v`
Expected: PASS. Before the rewire, `backend.read_window` *is* the old name-based path, so `new == old` is tautologically green — this is a **refactor-under-test / characterization test**, not a red-first test (Task 2's `ImportError` smoke was the red-first one). Its real job is Step 5: it must **stay green after** the rewire. If it goes red after Step 4, the `_phys_sample_idx` mapping does not match genoray's name resolution (see the seam note above).

- [ ] **Step 3: Add `_phys_sample_idx` to `_Svar2Backend.__init__`**

Immediately after `self._sample_names = sorted(native)` (and after `native = list(self._sv.available_samples)`), add:

```python
        # Public sorted-name position -> physical store (VCF) column. genoray's Rust
        # find_ranges takes physical usize indices (the Python _find_ranges resolved
        # names internally); we translate here so the Rust read_window gets columns.
        _col_of = {name: i for i, name in enumerate(native)}
        self._phys_sample_idx = np.array(
            [_col_of[n] for n in self._sample_names], dtype=np.int64
        )
```

- [ ] **Step 4: Rewire `read_window` to call the Rust FFI**

Replace the body of `_Svar2Backend.read_window` (`:1008-1047`) with:

```python
    def read_window(
        self, r_idx: NDArray[np.intp], s_idx: NDArray[np.intp]
    ) -> dict[str, object]:
        """Compute the window's live ranges via the GIL-free Rust `svar2_read_window`
        (genoray_core::query::find_ranges), replacing the Python SparseVar2._find_ranges
        call + numpy glue. `s_idx` (public sorted-name order) is translated to physical
        store columns via `_phys_sample_idx` before crossing into Rust.
        """
        from ..genvarloader import svar2_read_window

        r_idx = np.asarray(r_idx, np.intp)
        s_idx = np.asarray(s_idx, np.intp)
        contig_idx, contig = self._contig_of(r_idx)
        rb = self._regions[r_idx, 1:3]  # (n_reg, 2) int
        starts = np.ascontiguousarray(rb[:, 0], np.uint32)
        ends = np.ascontiguousarray(rb[:, 1], np.uint32)
        phys = np.ascontiguousarray(self._phys_sample_idx[s_idx], np.int64)
        n_reg, n_s, P = len(r_idx), len(s_idx), self.ploidy
        vk_snp, vk_indel, dense_snp, dense_indel, sample_cols = svar2_read_window(
            self._store, contig, starts, ends, phys
        )
        return {
            "contig_idx": contig_idx,
            "region_bounds": np.ascontiguousarray(rb, np.int32),  # (n_reg, 2)
            "orig_samples": np.asarray(sample_cols, np.int64),  # (n_s,)
            "vk_snp": np.asarray(vk_snp, np.int64).reshape(n_reg, n_s, P, 2),
            "vk_indel": np.asarray(vk_indel, np.int64).reshape(n_reg, n_s, P, 2),
            "dense_snp": np.asarray(dense_snp, np.int64).reshape(n_reg, 2),
            "dense_indel": np.asarray(dense_indel, np.int64).reshape(n_reg, 2),
        }
```

- [ ] **Step 5: Rebuild not required (no `src/` change), run the equivalence test**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr1.py -v`
Expected: PASS (both smoke + equivalence). If `test_..._matches_find_ranges` fails, the `_phys_sample_idx` mapping does not match genoray's name resolution — verify `available_samples` is the native VCF column order and that `sample_cols` returned by `find_ranges(Some(phys))` equals `phys` in order.

- [ ] **Step 6: Run the hard gates — parity + #284 scale**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py tests/dataset/test_streaming_scale.py -k "svar2 or streaming" -v`
Expected: PASS. Byte-identical parity + cohort-scale flatness both green through the Rust rewire.

- [ ] **Step 7: Lint/format/type gate**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_phase2_pr1.py
git commit -m "feat(streaming): SVAR2 read_window uses GIL-free Rust find_ranges (#278)"
```

---

## Final verification (before opening the PRs)

- [ ] **Full parity + scale + unit sweep** (shared code — cover both trees):

Run: `pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: green, including all SVAR1 + SVAR2 parity/scale tests (no regression from the additive changes).

- [ ] **Lint/format/type gate:**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Rebuild sanity (Rust changed in PR 1b):**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr1.py -q`
Expected: clean build, tests pass against the freshly-built extension.

- [ ] **Open the stacked draft PRs into `streaming`** (not `main`):
  - PR 1a: `streaming: SVAR2 per-batch reconstruct serial (parallel=False)` — Task 1 (relates to #278).
  - PR 1b: `streaming: SVAR2 read_window via GIL-free Rust find_ranges` — Tasks 2+3, stacked on PR 1a (relates to #278).
  Add both to the StreamingDataset project board.

---

## Out of scope (separate plans)

- **PR 2 — super-batch parallel reconstruction** (the multi-core lever): decouple the reconstruct super-batch from `batch_size`, one coarse `parallel=True` dispatch per `max_mem`-sized super-batch, drain per batch. Its own plan after PR 1's baseline is recorded.
- **PR 3 (gated) — relaxed-order multi-window pipeline**: bounded-queue producer/consumer; ships as default only on a cold-cache win over PR 2. Its own plan.
- Design reference: `docs/superpowers/specs/2026-07-18-streaming-svar2-phase2-design.md`.
