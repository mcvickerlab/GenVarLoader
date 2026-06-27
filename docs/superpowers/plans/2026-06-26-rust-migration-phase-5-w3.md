# Rust Migration Phase 5 — PR3 (W3): Fuse the deferred annotated+spliced reconstruction path

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the last un-fused FFI seam in haplotype reconstruction by adding a fused Rust kernel `reconstruct_annotated_haplotypes_spliced_fused` for the annotated **and** spliced path, wiring it into `_haps.py`, and parity-gating it byte-identically against the composed numba oracle.

**Architecture:** Three of the four annotated×spliced combinations are already fused into single-FFI-crossing Rust kernels (`reconstruct_haplotypes_fused`, `reconstruct_annotated_haplotypes_fused`, `reconstruct_haplotypes_spliced_fused`). The fourth — annotated **and** spliced — was deferred to Phase 5: on the rust backend it currently runs the un-fused dispatched `reconstruct_haplotypes_from_sparse` core and then folds reverse-complement (RC) in a Python post-pass (`_FlatAnnotatedHaps.reverse_masked`). This PR adds the missing fused kernel — a faithful **merge** of the two existing kernels: the spliced scaffolding (precomputed `out_offsets`, permuted ploidy-1 inputs, no `get_diffs_sparse`) from `reconstruct_haplotypes_spliced_fused`, plus the annotation buffers and the in-kernel RC triple from `reconstruct_annotated_haplotypes_fused`. Every primitive it composes (`reconstruct::reconstruct_haplotypes_from_sparse` with `Some` annotation views, `rc_flat_rows_inplace`, `reverse_flat_rows_inplace`) is already cargo-tested and parity-proven, so correctness reduces to wiring + a dataset-level parity gate.

**Tech Stack:** Rust (PyO3/maturin, `ndarray`), Python (NumPy, Polars), pytest parity suite, numba as the differential oracle.

## Global Constraints

- **Byte-identical numba/rust parity is the landing gate.** numba is the oracle and is NOT deleted in this PR (deletion is W5/W6). Every code path must remain comparable across `GVL_BACKEND=numba|rust`.
- **RC accounting (the parity-critical invariant):** for the spliced path, RC is applied per **permuted element**. On the **numba** backend RC is applied *externally* in `_query.py::_getitem_spliced` (the `if _active_backend() == "numba"` branch). On the **rust** backend the reconstructor must return output that is **already RC'd**, so `_getitem_spliced` treats rust as a no-op. The new fused kernel therefore folds RC *in-kernel*: `rc_flat_rows_inplace` on the sequence bytes (reverse + complement) and `reverse_flat_rows_inplace` on **both** annotation arrays (reverse only, **no** complement). This is byte-identical to `_FlatAnnotatedHaps.reverse_masked(mask, _COMP)` in `python/genvarloader/_flat.py:170-176`.
- The `to_rc` mask reaching the reconstructor is already in permuted per-element order (`to_rc_per_elem = to_rc_flat[plan.permutation]` from `_getitem_spliced`); pass it straight through. Its length must equal `out_offsets.len() - 1`.
- **maturin rebuild gotcha:** `pixi run -e dev pytest` does NOT rebuild the Rust extension. After ANY edit under `src/`, run `pixi run -e dev maturin develop --release` before pytest, or pytest imports the stale binary. `cargo test` compiles from source and is unaffected.
- **All pytest commands MUST include** `--basetemp=$(pwd)/.pytest_tmp` (os.link cross-device Errno 18 on this HPC otherwise).
- Conventional commits; co-author trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. No squash on merge; topic branch `phase-5-w3` (off `rust-migration`) → PR into `rust-migration`.

## Reference: the two existing kernels this one merges

- `src/ffi/mod.rs:689-762` `reconstruct_haplotypes_spliced_fused` — takes precomputed `out_offsets`, permuted inputs, ploidy-1 `flat_shifts`/`flat_geno_offset_idx`; allocates only `out_data`; calls the core with `None, None` for the annotation views; RCs sequence bytes in place via `rc_flat_rows_inplace`; returns `out_data` only (caller holds offsets).
- `src/ffi/mod.rs:789-920` `reconstruct_annotated_haplotypes_fused` — allocates `out_data` + `annot_v` (i32) + `annot_pos` (i32); calls the core with `Some(annot_v.view_mut()), Some(annot_pos.view_mut())`; on RC does `rc_flat_rows_inplace(out_data)` + `reverse_flat_rows_inplace(annot_v)` + `reverse_flat_rows_inplace(annot_pos)`. (It *computes* its own offsets via `get_diffs_sparse`; the spliced kernel does NOT — it receives them.)
- Python caller to mirror: the non-annotated spliced **rust branch** at `python/genvarloader/_dataset/_haps.py:910-942` shows the exact input prep (`np.ascontiguousarray(...)`, `_as_starts_stops`, `_ffi_array`, `self.ffi_static.*`, `reshape(-1, 1)`, `to_rc` passthrough).
- Exemplar parity tests: `tests/parity/test_spliced_haplotypes_parity.py` (spy + byte-identity pattern) and `tests/parity/test_haplotypes_dataset_parity.py::test_annotated_haplotypes_mode_dataset_parity` (annotated 3-array comparison via `.haps`/`.var_idxs`/`.ref_coords`).

---

## Task 1: Add the fused `reconstruct_annotated_haplotypes_spliced_fused` kernel, wire it into `_haps.py`, and parity-gate it

**Files:**
- Modify: `src/ffi/mod.rs` — add `reconstruct_annotated_haplotypes_spliced_fused` (insert after `reconstruct_haplotypes_spliced_fused`, i.e. after line 762).
- Modify: `src/lib.rs` — register the new pyfunction (after line 44).
- Modify: `python/genvarloader/_dataset/_haps.py` — add the module-level import (after line 42); rewrite the splice branch of `_reconstruct_annotated_haplotypes` (current lines 1100-1157) to call the fused kernel on the rust backend and drop the Python RC post-pass.
- Create: `tests/parity/test_annotated_spliced_haplotypes_parity.py` — the parity gate.

**Interfaces:**
- Produces (Rust → Python FFI): `reconstruct_annotated_haplotypes_spliced_fused(permuted_regions: i32[n,3], flat_shifts: i32[n,1], flat_geno_offset_idx: i64[n,1], out_offsets: i64[n+1], geno_offsets: i64[2,m], geno_v_idxs: i32[], v_starts: i32[], ilens: i32[], alt_alleles: u8[], alt_offsets: i64[], ref_: u8[], ref_offsets: i64[], pad_char: u8, keep: Optional[bool[]], keep_offsets: Optional[i64[]], to_rc: Optional[bool[n]]) -> (out_data: u8[], annot_v: i32[], annot_pos: i32[])`. Note: `out_offsets` is an INPUT (the caller holds the splice plan's `permuted_out_offsets`) and is NOT returned — matching `reconstruct_haplotypes_spliced_fused`.

- [ ] **Step 1: Write the failing parity test**

Create `tests/parity/test_annotated_spliced_haplotypes_parity.py`:

```python
"""Annotated+spliced haplotypes dataset parity backstop (fused rust entry, Phase 5 W3).

Proves the fused Rust entry ``reconstruct_annotated_haplotypes_spliced_fused`` produces
byte-identical (haps, var_idxs, ref_coords) output to the composed numba oracle for the
annotated AND spliced path — including a negative-strand transcript, which exercises the
in-kernel RC triple (reverse-complement of the sequence bytes + reverse of the two
annotation arrays, no complement).

Asserts:
  1. The fused entry actually fires on the rust path and NOT on the numba path (spy).
  2. All three arrays are byte-identical across backends (haps + var_idxs + ref_coords + offsets).
  3. RC actually changes the output (rc_neg=True vs rc_neg=False differ) — proves the
     negative-strand transcript exercises the in-kernel RC path (non-vacuous RC coverage).
  4. Output is non-trivial (contains non-N bases).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
import genvarloader._dataset._haps as _haps_mod
from genvarloader._ragged import RaggedAnnotatedHaps
from seqpro.rag import Ragged

pytestmark = pytest.mark.parity


def _compare_ragged(numba_out: Ragged, rust_out: Ragged, name: str) -> None:
    n_data = np.asarray(numba_out.data)
    r_data = np.asarray(rust_out.data)
    assert n_data.dtype == r_data.dtype, (
        f"dtype mismatch for {name}: numba={n_data.dtype}, rust={r_data.dtype}"
    )
    np.testing.assert_array_equal(
        n_data, r_data, err_msg=f"data differs across backends for '{name}'"
    )
    np.testing.assert_array_equal(
        np.asarray(numba_out.offsets, np.int64),
        np.asarray(rust_out.offsets, np.int64),
        err_msg=f"offsets differ across backends for '{name}'",
    )


def test_annotated_spliced_haplotypes_parity(phased_svar_gvl, reference, monkeypatch):
    # --- open in annotated mode, build a spliced dataset with mixed strands inline ---
    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = ds.with_seqs("annotated").with_tracks(False)

    n = 4
    # Group regions 0+1 -> T1 (+ strand), 2+3 -> T2 (- strand). The '-' transcript
    # exercises the in-kernel RC triple (rc bytes + reverse var_idxs/ref_coords).
    sub_bed = ds._full_bed[:n].with_columns(
        pl.Series("transcript_id", ["T1", "T1", "T2", "T2"]),
        pl.Series("strand", ["+", "+", "-", "-"]),
    )
    assert (sub_bed["strand"] == "-").any(), "need a '-' transcript to cover RC"
    ds = replace(ds, _full_bed=sub_bed).with_settings(splice_info="transcript_id")
    assert ds.is_spliced, "Dataset should be in spliced mode"

    # --- spy on the fused annotated-spliced entry ---
    orig = getattr(_haps_mod, "reconstruct_annotated_haplotypes_spliced_fused", None)
    assert orig is not None, (
        "reconstruct_annotated_haplotypes_spliced_fused not found on _haps_mod — "
        "ensure it is imported at module level in _haps.py"
    )
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(
        _haps_mod, "reconstruct_annotated_haplotypes_spliced_fused", _spy
    )

    # --- rust read (fused path) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]
    rust_calls = calls["n"]

    # --- numba read (composed oracle; spy must NOT fire) ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    assert calls["n"] == rust_calls, (
        "fused annotated-spliced spy fired during the numba read — "
        "the fused entry is being called on the numba path."
    )
    assert rust_calls > 0, (
        "reconstruct_annotated_haplotypes_spliced_fused was NEVER invoked on the rust "
        "read — the backstop is vacuous. Ensure _haps._reconstruct_annotated_haplotypes "
        "calls it on the splice path when GVL_BACKEND=rust."
    )

    assert isinstance(out_rust, RaggedAnnotatedHaps), type(out_rust)
    assert isinstance(out_numba, RaggedAnnotatedHaps), type(out_numba)

    # --- non-trivial output ---
    data_u8 = np.asarray(out_rust.haps.data).view(np.uint8)
    assert data_u8.size > 0 and np.any(data_u8 != np.uint8(ord("N"))), (
        "annotated-spliced output is empty or all-N padding — comparison is vacuous."
    )

    # --- RC non-vacuity: rc_neg flips the '-' transcript output (rust backend) ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_norc = ds.with_settings(rc_neg=False)[:, :]
    assert not np.array_equal(
        np.asarray(out_rust.haps.data), np.asarray(out_norc.haps.data)
    ), (
        "RC made no difference — the negative-strand transcript is not exercising the "
        "in-kernel RC path (check strand propagation / rc_neg default)."
    )

    # --- byte-identity across backends on all three arrays ---
    _compare_ragged(out_numba.haps, out_rust.haps, "annotated-spliced.haps")
    _compare_ragged(out_numba.var_idxs, out_rust.var_idxs, "annotated-spliced.var_idxs")
    _compare_ragged(
        out_numba.ref_coords, out_rust.ref_coords, "annotated-spliced.ref_coords"
    )
```

If any attribute used above (`_full_bed`, `is_spliced`, `with_seqs("annotated")`, `with_settings(rc_neg=...)`, `RaggedAnnotatedHaps`, `.haps`/`.var_idxs`/`.ref_coords`) does not exist with these exact names, reconcile against the two exemplar tests in the "Reference" section above — do NOT invent names. (`ds._full_bed` and `ds.is_spliced` are used verbatim in `test_spliced_haplotypes_parity.py:87,92`.)

- [ ] **Step 2: Run the test to verify it fails for the right reason**

Run: `pixi run -e dev pytest tests/parity/test_annotated_spliced_haplotypes_parity.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL at the `orig is not None` assertion (the symbol `reconstruct_annotated_haplotypes_spliced_fused` is not yet imported on `_haps_mod`). This confirms the gate targets the new kernel.

- [ ] **Step 3: Add the fused Rust kernel**

In `src/ffi/mod.rs`, insert immediately after `reconstruct_haplotypes_spliced_fused` (after line 762):

```rust
/// Fused annotated spliced-haplotype reconstruction: the annotated counterpart of
/// `reconstruct_haplotypes_spliced_fused`. Reconstructs in one FFI crossing using
/// precomputed splice output offsets AND fills the two per-nucleotide annotation
/// arrays (variant index, reference coordinate).
///
/// Like the non-annotated splice entry, the Python splice plan already computes the
/// permutation and `out_offsets` (`splice_plan.permuted_out_offsets`), so this kernel
/// takes `out_offsets` directly and skips `get_diffs_sparse` / the offset loop.
///
/// On `to_rc`, each masked permuted element is reverse-complemented in place
/// (`rc_flat_rows_inplace` on the sequence bytes) and its annotation rows are reversed
/// in place (`reverse_flat_rows_inplace`, no complement) — byte-identical to
/// `_FlatAnnotatedHaps.reverse_masked(mask, _COMP)`.
///
/// Returns `(out_data, annot_v, annot_pos)`. `out_offsets` is held by the caller and
/// not returned (matches `reconstruct_haplotypes_spliced_fused`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_annotated_haplotypes_spliced_fused<'py>(
    py: Python<'py>,
    permuted_regions: PyReadonlyArray2<i32>,
    flat_shifts: PyReadonlyArray2<i32>,
    flat_geno_offset_idx: PyReadonlyArray2<i64>,
    out_offsets: PyReadonlyArray1<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    to_rc: Option<PyReadonlyArray1<bool>>,
) -> (
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
) {
    use crate::reconstruct;

    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);

    // out_offsets are precomputed by the Python splice plan — use them directly.
    let out_offsets_a = out_offsets.as_array();
    let total = out_offsets_a[out_offsets_a.len() - 1] as usize;

    // Allocate the sequence + annotation buffers.
    let mut out_data: Array1<u8> = uninit_output(total);
    let mut annot_v: Array1<i32> = uninit_output(total);
    let mut annot_pos: Array1<i32> = uninit_output(total);

    // Reconstruct all haplotypes + annotations into the owned buffers (reuses batch core).
    reconstruct::reconstruct_haplotypes_from_sparse(
        out_data.view_mut(),
        out_offsets_a,
        permuted_regions.as_array(),
        flat_shifts.as_array(),
        flat_geno_offset_idx.as_array(),
        go_starts,
        go_stops,
        geno_v_idxs.as_array(),
        v_starts.as_array(),
        ilens.as_array(),
        alt_alleles.as_array(),
        alt_offsets.as_array(),
        ref_.as_array(),
        ref_offsets.as_array(),
        pad_char,
        keep.as_ref().map(|k| k.as_array()),
        keep_offsets.as_ref().map(|ko| ko.as_array()),
        Some(annot_v.view_mut()),   // annot_v_idxs — variant index per nucleotide
        Some(annot_pos.view_mut()), // annot_ref_pos — reference coordinate per nucleotide
    );

    // Optional in-place RC per permuted element. Sequence bytes are reverse-complemented;
    // annotation rows are reversed only (no complement) — matching
    // _FlatAnnotatedHaps.reverse_masked. out_offsets_a is the permuted per-element
    // offsets array, so each masked element is transformed in its own byte range.
    if let Some(to_rc) = to_rc.as_ref() {
        let m = to_rc.as_array();
        debug_assert_eq!(
            m.len(),
            out_offsets_a.len() - 1,
            "to_rc mask length must equal number of output rows (offsets.len() - 1)"
        );
        crate::reverse::rc_flat_rows_inplace(out_data.as_slice_mut().unwrap(), out_offsets_a, m);
        crate::reverse::reverse_flat_rows_inplace(annot_v.as_slice_mut().unwrap(), out_offsets_a, m);
        crate::reverse::reverse_flat_rows_inplace(annot_pos.as_slice_mut().unwrap(), out_offsets_a, m);
    }

    (
        out_data.into_pyarray(py),
        annot_v.into_pyarray(py),
        annot_pos.into_pyarray(py),
    )
}
```

Verify against the source: confirm `uninit_output`, `crate::reverse::rc_flat_rows_inplace`, and `crate::reverse::reverse_flat_rows_inplace` are the same symbols used by `reconstruct_annotated_haplotypes_fused` (`src/ffi/mod.rs:875-911`) and that `reconstruct::reconstruct_haplotypes_from_sparse`'s parameter order matches the call in `reconstruct_haplotypes_spliced_fused` (`src/ffi/mod.rs:722-742`). If a helper name differs in your tree, use the name the two reference kernels actually use.

- [ ] **Step 4: Register the pyfunction**

In `src/lib.rs`, after line 44 (`reconstruct_haplotypes_spliced_fused`), add:

```rust
    m.add_function(wrap_pyfunction!(ffi::reconstruct_annotated_haplotypes_spliced_fused, m)?)?;
```

- [ ] **Step 5: Import the symbol in `_haps.py`**

In `python/genvarloader/_dataset/_haps.py`, in the extension-import block (after line 42, `reconstruct_haplotypes_spliced_fused as reconstruct_haplotypes_spliced_fused,`), add:

```python
    reconstruct_annotated_haplotypes_spliced_fused as reconstruct_annotated_haplotypes_spliced_fused,
```

(Match the existing `import X as X` re-export style used by its siblings in that block.)

- [ ] **Step 6: Rewrite the splice branch of `_reconstruct_annotated_haplotypes`**

Replace the current splice-plan block (`python/genvarloader/_dataset/_haps.py:1100-1157`, from the `# ---- splice plan path ----` comment through the final `return haps_rag, annot_v_rag, annot_pos_rag`) with:

```python
        # ---- splice plan path ----
        flat_geno_idx, flat_shifts, permuted_regions, keep_perm, keep_offsets_perm = (
            self._permute_request_for_splice(req)
        )
        splice_plan = req.splice_plan
        per_elem_shape = (splice_plan.permuted_lengths.shape[0], None)
        off = splice_plan.permuted_out_offsets

        _backend = os.environ.get("GVL_BACKEND", "rust")
        if _backend == "rust":
            # Fused path: one FFI crossing. RC is folded in-kernel (sequence bytes
            # reverse-complemented, annotation rows reversed), so there is NO Python
            # reverse_masked post-pass. to_rc is already in permuted per-element order
            # (from _getitem_spliced), and _getitem_spliced treats the rust output as
            # already-RC'd (its post-pass is numba-only).
            _to_rc_spliced = (
                None if to_rc is None else np.ascontiguousarray(to_rc, np.bool_)
            )
            out_buf, annot_v_buf, annot_pos_buf = (
                reconstruct_annotated_haplotypes_spliced_fused(
                    permuted_regions=np.ascontiguousarray(permuted_regions, np.int32),
                    flat_shifts=np.ascontiguousarray(
                        flat_shifts.reshape(-1, 1), np.int32
                    ),
                    flat_geno_offset_idx=np.ascontiguousarray(
                        flat_geno_idx.reshape(-1, 1), np.int64
                    ),
                    out_offsets=np.ascontiguousarray(off, np.int64),
                    geno_offsets=_as_starts_stops(self.genotypes.offsets),
                    geno_v_idxs=_ffi_array(self.genotypes.data, np.int32, "geno_v_idxs"),
                    v_starts=self.ffi_static.v_starts,
                    ilens=self.ffi_static.ilens,
                    alt_alleles=self.ffi_static.alt_alleles,
                    alt_offsets=self.ffi_static.alt_offsets,
                    ref_=self.ffi_static.ref,
                    ref_offsets=self.ffi_static.ref_offsets,
                    pad_char=np.uint8(self.reference.pad_char),
                    keep=None
                    if keep_perm is None
                    else np.ascontiguousarray(keep_perm, np.bool_),
                    keep_offsets=None
                    if keep_offsets_perm is None
                    else np.ascontiguousarray(keep_offsets_perm, np.int64),
                    to_rc=_to_rc_spliced,
                )
            )
        else:
            # Numba composed oracle path. RC is applied externally in
            # _getitem_spliced (numba branch), so no to_rc / RC is applied here.
            total = int(off[-1])
            out_buf = np.empty(total, np.uint8)
            annot_v_buf = np.empty(total, V_IDX_TYPE)
            annot_pos_buf = np.empty(total, np.int32)
            reconstruct_haplotypes_from_sparse(
                geno_offset_idx=flat_geno_idx.reshape(-1, 1),
                out=out_buf,
                out_offsets=off,
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

        haps_rag = cast(
            "Ragged[np.bytes_]",
            _Flat.from_offsets(out_buf, per_elem_shape, off).view("S1"),
        )
        annot_v_rag = cast(
            "Ragged[V_IDX_TYPE]",
            _Flat.from_offsets(annot_v_buf, per_elem_shape, off),
        )
        annot_pos_rag = cast(
            "Ragged[np.int32]",
            _Flat.from_offsets(annot_pos_buf, per_elem_shape, off),
        )
        return haps_rag, annot_v_rag, annot_pos_rag
```

This deletes the old unconditional `reconstruct_haplotypes_from_sparse` call (it now lives only in the numba `else` branch) and the `if ... == "rust" and to_rc is not None: ... reverse_masked(...)` post-pass block (RC is now in-kernel on rust). If removing that block leaves `_FlatAnnotatedHaps` and/or the local `from .._ragged import _COMP` unused in the file, the lint step in Task 2 will catch it — remove the now-dead import(s). Do NOT change `_query.py::_getitem_spliced`: its `if _active_backend() == "numba"` RC guard remains correct (rust output is already RC'd, numba is post-passed there).

- [ ] **Step 7: Rebuild the Rust extension**

Run: `pixi run -e dev maturin develop --release`
Expected: builds cleanly (the new kernel + registration compile).

- [ ] **Step 8: Run the parity test under both backends**

```bash
pixi run -e dev pytest tests/parity/test_annotated_spliced_haplotypes_parity.py -v --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS — the spy fires on rust only, RC non-vacuity holds, and all three arrays are byte-identical to numba.

- [ ] **Step 9: Run the broader haplotype parity + reconstruct suites to confirm no regression**

```bash
pixi run -e dev cargo test --release reconstruct
pixi run -e dev pytest tests/parity/test_spliced_haplotypes_parity.py tests/parity/test_haplotypes_dataset_parity.py tests/parity/test_annotated_spliced_haplotypes_parity.py -q --basetemp=$(pwd)/.pytest_tmp
GVL_BACKEND=numba pixi run -e dev pytest tests/parity/test_spliced_haplotypes_parity.py tests/parity/test_haplotypes_dataset_parity.py tests/parity/test_annotated_spliced_haplotypes_parity.py -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: all green on both backends; cargo reconstruct tests pass.

- [ ] **Step 10: Commit**

```bash
rtk git add src/ffi/mod.rs src/lib.rs python/genvarloader/_dataset/_haps.py tests/parity/test_annotated_spliced_haplotypes_parity.py
rtk git commit -m "feat(rust): fuse annotated+spliced haplotype reconstruction into one FFI crossing (Phase 5 W3)

Add reconstruct_annotated_haplotypes_spliced_fused — the annotated counterpart of
reconstruct_haplotypes_spliced_fused. Folds RC in-kernel (bytes RC'd, annotation rows
reversed) so the Python _FlatAnnotatedHaps.reverse_masked post-pass is dropped on the
rust backend. Byte-identical to the composed numba oracle (new parity backstop).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Resolve the roadmap deferral note + full-tree both-backend verification

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` — update the deferral note (around line 285) and add a dated Phase 5 W3 entry.

- [ ] **Step 1: Update the roadmap**

Find the note (near `docs/roadmaps/rust-migration.md:285`) that reads, in part: "*(The annotated+spliced intersection remains on the unfused dispatched rust core — still parity-gated and rust-by-default — with fusion deferred to Phase 5.)*". Rewrite it to state the intersection is now fused via `reconstruct_annotated_haplotypes_spliced_fused` (one FFI crossing, RC folded in-kernel), byte-identical to the composed numba oracle, covered by `tests/parity/test_annotated_spliced_haplotypes_parity.py`. Then add a dated Phase 5 W3 entry to the Notes & decisions log recording: the fourth (and final) annotated×spliced combination is now fused; all four reconstruction combinations cross the FFI boundary exactly once on the rust backend; numba remains the oracle (deletion is W5/W6); Phase 5 stays 🚧 (W4–W9 remain). Reference the new test and the PR. Do NOT mark Phase 5 ✅.

- [ ] **Step 2: Full parity suite, both backends**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp
GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: green on both backends, matching pass/skip profiles.

- [ ] **Step 3: Full tree (catch stale references in tests/unit and tests/dataset), both backends not required but rust must be green**

```bash
pixi run -e dev pytest tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: green (no stale references to the deleted post-pass / changed branch).

- [ ] **Step 4: Lint, format, typecheck, cargo**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
pixi run -e dev cargo clippy
```
Expected: clean. (If Task 1 left `_FlatAnnotatedHaps`/`_COMP` unused, ruff flags it here — remove the dead import and re-run.)

- [ ] **Step 5: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): record annotated+spliced fusion; all 4 reconstruction combos now single-FFI (Phase 5 W3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Finish (controller, after final whole-branch review + user confirm)

- Re-verify the load-bearing gate against a fresh `pixi run -e dev maturin develop --release` build (the parity test + full parity suite, both backends) before the final review.
- Confirm co-author trailers on every commit.
- File a GVL issue if any follow-up surfaces (e.g. a Minor deferred); otherwise none required.
- Push `phase-5-w3`; open PR into `rust-migration` (no squash). Reference the W3 plan and the new parity test.

## Self-Review

- **Spec coverage:** PR3's three spec clauses are all covered — "add a fused rust kernel collapsing its remaining FFI crossings (pattern `reconstruct_*_fused`)" → Task 1 Steps 3-6; "parity-gate against the composed numba oracle while numba still exists" → Task 1 Steps 1, 8, 9 (numba branch retained as `else`); "extend the parity suite to cover it" → new `tests/parity/test_annotated_spliced_haplotypes_parity.py`. The deferral note (roadmap) is resolved in Task 2.
- **Placeholder scan:** every code step contains complete code (the Rust kernel, the Python branch rewrite, the full test). The only deliberately non-transcribed item is the roadmap prose (Task 2 Step 1), which is a documentation edit with the exact target line and required content enumerated.
- **Type consistency:** the kernel returns `(u8[], i32[], i32[])` with `out_offsets` as input-only — matching `reconstruct_haplotypes_spliced_fused` (offsets in, not returned) and `reconstruct_annotated_haplotypes_fused` (annotation buffers, RC triple). The Python caller wraps the three buffers with the shared `off`/`per_elem_shape`, identical to the deleted code's wrapping. `V_IDX_TYPE` (Python) ↔ `i32` (Rust `annot_v`) match the existing annotated kernels.
