# Rust Variant-Allele Reverse-Complement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-batch Python object churn in the variant-allele reverse-complement post-pass with a thin gvl-owned Rust kernel (`rc_alleles_inplace`) operating on the raw `_FlatAlleles` buffers, byte-identical to the existing seqpro path.

**Architecture:** A pure-`ndarray` core (`src/variants/mod.rs`) reuses the Target-6 `reverse::{rc_flat_rows_inplace, COMP}` primitives; a PyO3 in-place wrapper (`src/ffi/mod.rs`) exposes it; it is registered in `_dispatch` as `rc_alleles` (rust default, the existing seqpro implementation retained as the reference backend). The two Python RC methods (`_FlatAlleles.reverse_masked`, `RaggedVariants.rc_`) route their inner RC through the dispatched kernel. RC stays positioned **after** dummy-fill (same as today), so ordering is byte-identical even for custom non-palindromic dummy alleles.

**Tech Stack:** Rust (PyO3 + ndarray), Python (numpy), pytest + hypothesis (parity), cargo test, pixi (`-e dev`).

## Global Constraints

- **Byte-identical parity** is the migration contract: the new rust kernel must produce output identical to the existing seqpro reference across the parity matrix. A unit only lands when parity holds.
- **Do NOT delete the seqpro reference / numba backends.** `rust-migration` is not ready to merge; the reference is retained for parity + performance gating (deletion is Phase 5). Per `[[numba-oracle-bug-policy]]` and the roadmap.
- **No on-disk format change.** No change to `_FlatVariantWindows` (still never RC'd). No change to `flank_tokens` (the post-pass RCs only `alt`/`ref`).
- Dispatch registry API: `register(name, *, numba=, rust=, default=)`, `get(name)(...)`, `backends(name) -> (numba, rust)`. `GVL_BACKEND=numba|rust` force-overrides.
- Complement LUT is `_COMP = np.frombuffer(bytes.maketrans(b"ACGT", b"TGCA"), np.uint8)` (Python) ≡ `crate::reverse::COMP` (Rust). Both reverse THEN complement per allele.
- Mask broadcast convention (must match exactly): per-region mask → per-`(b*p)` row via `np.repeat(mask, ploidy)` (done Python-side) → per-allele via `np.repeat(per_bp, np.diff(var_offsets))` (done inside the kernel).
- Dataset tests on the HPC need `--basetemp=$(pwd)/.pytest_tmp` (os.link cross-device Errno 18).
- Build/test commands: `pixi run -e dev cargo test`, `pixi run -e dev pytest <path> -q`, `pixi run -e dev test` (full tree), `pixi run -e dev ruff check python/ tests/`, `pixi run -e dev ruff format python/ tests/`, `pixi run -e dev typecheck`.

---

### Task 1: Rust core `rc_alleles_inplace` + cargo unit tests

**Files:**
- Modify: `src/variants/mod.rs` (add `rc_alleles_inplace` after `gather_alleles` ~line 52; add tests to the existing `#[cfg(test)] mod tests` or create one)

**Interfaces:**
- Consumes: `crate::reverse::{rc_flat_rows_inplace, COMP}` (existing, from Target 6).
- Produces: `pub fn rc_alleles_inplace(byte_data: &mut [u8], seq_offsets: ArrayView1<i64>, var_offsets: ArrayView1<i64>, to_rc_row: ArrayView1<bool>)`.
  - `byte_data`: contiguous allele bytes, mutated in place.
  - `seq_offsets`: per-allele byte boundaries, len `n_alleles + 1`.
  - `var_offsets`: per-`(b*p)`-row allele boundaries, len `n_rows + 1`. `to_rc_row` has len `n_rows`.
  - For each row `g` with `to_rc_row[g]==true`, every allele `a` in `var_offsets[g]..var_offsets[g+1]` is reverse-complemented over `seq_offsets[a]..seq_offsets[a+1]` via `COMP`.

- [ ] **Step 1: Write the failing tests**

Add to `src/variants/mod.rs` (inside the test module; if none exists, add `#[cfg(test)] mod rc_tests { use super::*; use ndarray::array; ... }`):

```rust
#[test]
fn rc_alleles_rcs_only_masked_rows() {
    // 2 rows. row0 (masked) has 2 alleles: "AC","G". row1 (unmasked): "TT".
    // seq_offsets delimit alleles: [0,2,3,5]; var_offsets delimit rows: [0,2,3].
    let mut data = b"ACGTT".to_vec();
    let seq_offsets = ndarray::array![0i64, 2, 3, 5];
    let var_offsets = ndarray::array![0i64, 2, 3];
    let to_rc_row = ndarray::array![true, false];
    rc_alleles_inplace(&mut data, seq_offsets.view(), var_offsets.view(), to_rc_row.view());
    // row0: "AC"->"GT", "G"->"C"; row1 "TT" untouched.
    assert_eq!(&data, b"GTCTT");
}

#[test]
fn rc_alleles_all_false_is_noop() {
    let mut data = b"ACG".to_vec();
    let seq_offsets = ndarray::array![0i64, 1, 3];
    let var_offsets = ndarray::array![0i64, 2];
    let to_rc_row = ndarray::array![false];
    rc_alleles_inplace(&mut data, seq_offsets.view(), var_offsets.view(), to_rc_row.view());
    assert_eq!(&data, b"ACG");
}

#[test]
fn rc_alleles_handles_empty_allele_and_n() {
    // 1 masked row, 2 alleles: "" (empty) and "ACN".
    let mut data = b"ACN".to_vec();
    let seq_offsets = ndarray::array![0i64, 0, 3];
    let var_offsets = ndarray::array![0i64, 2];
    let to_rc_row = ndarray::array![true];
    rc_alleles_inplace(&mut data, seq_offsets.view(), var_offsets.view(), to_rc_row.view());
    // "" stays ""; "ACN" -> revcomp -> "NGT".
    assert_eq!(&data, b"NGT");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev cargo test --lib rc_alleles`
Expected: FAIL — `rc_alleles_inplace` not found (cannot resolve function).

- [ ] **Step 3: Implement the core**

Add to `src/variants/mod.rs` (after `gather_alleles`). Ensure `use crate::reverse::{rc_flat_rows_inplace, COMP};` is available — `COMP` is unused directly here (delegated), so import only what is used:

```rust
/// Reverse-complement the alleles of mask-selected `(b*p)` rows, in place.
///
/// `byte_data`   contiguous allele bytes (mutated in place)
/// `seq_offsets` per-allele byte boundaries (len n_alleles + 1)
/// `var_offsets` per-(b*p)-row allele boundaries (len n_rows + 1)
/// `to_rc_row`   per-(b*p)-row bool mask (len n_rows)
///
/// Expands the row mask to a per-allele mask via `var_offsets`, then delegates
/// to `reverse::rc_flat_rows_inplace` (reverse + `COMP`), matching the Python
/// `np.repeat(per_bp, np.diff(var_offsets))` expansion byte-for-byte.
pub fn rc_alleles_inplace(
    byte_data: &mut [u8],
    seq_offsets: ndarray::ArrayView1<i64>,
    var_offsets: ndarray::ArrayView1<i64>,
    to_rc_row: ndarray::ArrayView1<bool>,
) {
    let n_alleles = seq_offsets.len() - 1;
    let mut per_allele = vec![false; n_alleles];
    for g in 0..to_rc_row.len() {
        if !to_rc_row[g] {
            continue;
        }
        let a0 = var_offsets[g] as usize;
        let a1 = var_offsets[g + 1] as usize;
        for a in a0..a1 {
            per_allele[a] = true;
        }
    }
    let per_allele = ndarray::Array1::from_vec(per_allele);
    crate::reverse::rc_flat_rows_inplace(byte_data, seq_offsets, per_allele.view());
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev cargo test --lib rc_alleles`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
rtk git add src/variants/mod.rs
rtk git commit -m "feat(rust): rc_alleles_inplace core for variant-allele RC

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: PyO3 wrapper `rc_alleles` + registration

**Files:**
- Modify: `src/ffi/mod.rs` (add `rc_alleles` pyfunction, follow the `intervals_to_tracks` in-place pattern ~line 67)
- Modify: `src/lib.rs` (register `ffi::rc_alleles` in the `#[pymodule]`, after `assemble_variant_buffers_i32` ~line 38)

**Interfaces:**
- Consumes: `crate::variants::rc_alleles_inplace` (Task 1).
- Produces: pyfunction `rc_alleles(byte_data: PyReadwriteArray1<u8>, seq_offsets: PyReadonlyArray1<i64>, var_offsets: PyReadonlyArray1<i64>, to_rc_row: PyReadonlyArray1<bool>)` — mutates `byte_data` in place, returns `None`.

- [ ] **Step 1: Write the failing test (Python smoke via the rust symbol)**

Create `tests/unit/test_rc_alleles_ffi.py`. The compiled extension is
`genvarloader.genvarloader` (see `_flat_variants.py:20`, `from ..genvarloader import ...`):

```python
import numpy as np
import genvarloader.genvarloader as _gvl  # compiled rust extension module


def test_rc_alleles_ffi_inplace():
    # 2 rows. row0 (masked): alleles "AC","G". row1 (unmasked): "TT".
    data = np.frombuffer(b"ACGTT", np.uint8).copy()
    seq_offsets = np.array([0, 2, 3, 5], np.int64)
    var_offsets = np.array([0, 2, 3], np.int64)
    to_rc_row = np.array([True, False], np.bool_)
    _gvl.rc_alleles(data, seq_offsets, var_offsets, to_rc_row)
    assert data.tobytes() == b"GTCTT"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_rc_alleles_ffi.py -v`
Expected: FAIL — `module ... has no attribute 'rc_alleles'`.

- [ ] **Step 3: Implement the wrapper**

In `src/ffi/mod.rs` (mirror `intervals_to_tracks`):

```rust
/// In-place reverse-complement of the alleles of mask-selected `(b*p)` rows.
/// See `crate::variants::rc_alleles_inplace`.
#[pyfunction]
pub fn rc_alleles(
    mut byte_data: PyReadwriteArray1<u8>,
    seq_offsets: PyReadonlyArray1<i64>,
    var_offsets: PyReadonlyArray1<i64>,
    to_rc_row: PyReadonlyArray1<bool>,
) {
    crate::variants::rc_alleles_inplace(
        byte_data.as_slice_mut().unwrap(),
        seq_offsets.as_array(),
        var_offsets.as_array(),
        to_rc_row.as_array(),
    );
}
```

In `src/lib.rs`, after line 38 (`assemble_variant_buffers_i32`):

```rust
    m.add_function(wrap_pyfunction!(ffi::rc_alleles, m)?)?;
```

- [ ] **Step 4: Rebuild + run to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_rc_alleles_ffi.py -v`
(pixi rebuilds the extension via maturin automatically.)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/ffi/mod.rs src/lib.rs tests/unit/test_rc_alleles_ffi.py
rtk git commit -m "feat(rust): rc_alleles PyO3 wrapper + registration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `rc_alleles` dispatch entry (rust default + seqpro reference)

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (add the dispatch shims + `register("rc_alleles", ...)` near the existing `register("assemble_variant_buffers", ...)` ~line 931)

**Interfaces:**
- Consumes: the rust `rc_alleles` pyfunction (Task 2); `_dispatch.register`; `genvarloader._ragged.reverse_complement_masked` + `seqpro.rag.Ragged` (reference).
- Produces: registry entry `"rc_alleles"` with signature `(byte_data, seq_offsets, var_offsets, to_rc_row)`, both backends mutating `byte_data` in place and returning `None`. `default="rust"`.
  - `byte_data`: `uint8` array. `seq_offsets`/`var_offsets`: `int64`. `to_rc_row`: per-`(b*p)` bool mask (already ploidy-broadcast by the caller).

- [ ] **Step 1: Write the failing parity test**

Create `tests/parity/test_rc_alleles_parity.py`:

```python
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from genvarloader._dataset import _flat_variants  # noqa: F401  (registers rc_alleles)
from genvarloader import _dispatch

_ACGTN = np.frombuffer(b"ACGTN", np.uint8)


@st.composite
def _allele_batch(draw):
    n_rows = draw(st.integers(1, 4))
    alleles_per_row = [draw(st.integers(0, 3)) for _ in range(n_rows)]
    var_offsets = np.concatenate([[0], np.cumsum(alleles_per_row)]).astype(np.int64)
    n_alleles = int(var_offsets[-1])
    lens = [draw(st.integers(0, 5)) for _ in range(n_alleles)]
    seq_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    total = int(seq_offsets[-1])
    data = _ACGTN[draw(st.lists(st.integers(0, 4), min_size=total, max_size=total))] \
        if total else np.zeros(0, np.uint8)
    data = np.ascontiguousarray(data, np.uint8)
    mask = np.array([draw(st.booleans()) for _ in range(n_rows)], np.bool_)
    return data, seq_offsets, var_offsets, mask


@settings(max_examples=200, deadline=None)
@given(batch=_allele_batch())
def test_rc_alleles_rust_matches_reference(batch):
    data, seq_offsets, var_offsets, mask = batch
    numba_fn, rust_fn = _dispatch.backends("rc_alleles")
    a = data.copy()
    b = data.copy()
    numba_fn(a, seq_offsets, var_offsets, mask)
    rust_fn(b, seq_offsets, var_offsets, mask)
    assert a.tobytes() == b.tobytes()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/parity/test_rc_alleles_parity.py -q`
Expected: FAIL — `KeyError: no kernel registered as 'rc_alleles'`.

- [ ] **Step 3: Implement the shims + registration**

In `python/genvarloader/_dataset/_flat_variants.py`, near the `assemble_variant_buffers` registration (~line 931), add:

```python
def _rc_alleles_reference(byte_data, seq_offsets, var_offsets, to_rc_row):
    """Reference backend: seqpro reverse_complement_masked on a flat allele view.

    `to_rc_row` is the per-(b*p) row mask (already ploidy-broadcast); expand to
    per-allele via `var_offsets`, then RC each masked allele in place. Mutates
    `byte_data` in place; byte-identical to `rc_alleles_inplace`.
    """
    from seqpro.rag import Ragged

    from .._ragged import reverse_complement_masked

    seq_off = np.ascontiguousarray(seq_offsets, np.int64)
    var_off = np.ascontiguousarray(var_offsets, np.int64)
    row_mask = np.ascontiguousarray(to_rc_row, np.bool_).reshape(-1)
    if not row_mask.any():
        return
    per_allele = np.repeat(row_mask, np.diff(var_off))
    n_alleles = len(seq_off) - 1
    view = Ragged.from_offsets(byte_data.view("S1"), (n_alleles, None), seq_off)
    reverse_complement_masked(view, per_allele)  # mutates byte_data in place


def _rc_alleles_rust(byte_data, seq_offsets, var_offsets, to_rc_row):
    _rc_alleles_rust_kernel(
        np.ascontiguousarray(byte_data, np.uint8),  # in-place: see note below
        np.ascontiguousarray(seq_offsets, np.int64),
        np.ascontiguousarray(var_offsets, np.int64),
        np.ascontiguousarray(to_rc_row, np.bool_),
    )


register(
    "rc_alleles",
    numba=_rc_alleles_reference,
    rust=_rc_alleles_rust,
    default="rust",
)
```

> **In-place caveat:** `np.ascontiguousarray` returns the SAME object when input is already contiguous `uint8`, but a COPY otherwise — which would silently drop the in-place mutation. The callers (Task 4) pass contiguous `uint8` `byte_data` directly, so guard it: assert contiguity instead of coercing. Replace the `_rc_alleles_rust` body with:
> ```python
> def _rc_alleles_rust(byte_data, seq_offsets, var_offsets, to_rc_row):
>     assert byte_data.dtype == np.uint8 and byte_data.flags.c_contiguous, (
>         "rc_alleles requires a contiguous uint8 byte_data for in-place RC"
>     )
>     _rc_alleles_rust_kernel(
>         byte_data,
>         np.ascontiguousarray(seq_offsets, np.int64),
>         np.ascontiguousarray(var_offsets, np.int64),
>         np.ascontiguousarray(to_rc_row, np.bool_),
>     )
> ```

Add the rust import at the top of `_flat_variants.py`, alongside the existing
`assemble_variant_buffers_*` imports (~lines 20–24, which use `from ..genvarloader import ...`):

```python
from ..genvarloader import rc_alleles as _rc_alleles_rust_kernel
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/parity/test_rc_alleles_parity.py -q`
Expected: PASS (200 examples).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py tests/parity/test_rc_alleles_parity.py
rtk git commit -m "feat: register rc_alleles dispatch (rust default, seqpro reference)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Route `_FlatAlleles.reverse_masked` + `RaggedVariants.rc_` through dispatch

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (`_FlatAlleles.reverse_masked`, ~lines 119-142)
- Modify: `python/genvarloader/_dataset/_rag_variants.py` (`RaggedVariants.rc_`, ~lines 296-351; replace only the inner `_sp_reverse_complement` call)

**Interfaces:**
- Consumes: `get("rc_alleles")` (Task 3).
- Produces: unchanged public signatures `_FlatAlleles.reverse_masked(self, mask) -> _FlatAlleles` and `RaggedVariants.rc_(self, to_rc=None) -> RaggedVariants`; output byte-identical to before, now backend-dispatched.

- [ ] **Step 1: Write the failing test (behavior pin on the rust backend)**

Add to `tests/parity/test_rc_alleles_parity.py`:

```python
def test_flat_alleles_reverse_masked_uses_rc_alleles(monkeypatch):
    """_FlatAlleles.reverse_masked must call the dispatched rc_alleles kernel."""
    from genvarloader._dataset._flat_variants import _FlatAlleles
    from genvarloader._dataset import _flat_variants as fv

    calls = {"n": 0}
    real = _dispatch.get

    def spy(name):
        if name == "rc_alleles":
            calls["n"] += 1
        return real(name)

    monkeypatch.setattr(fv, "get", spy)

    # one row (b=1, ploidy=1), two alleles "AC","G".
    byte_data = np.frombuffer(b"ACG", np.uint8).copy()
    seq_offsets = np.array([0, 2, 3], np.int64)
    var_offsets = np.array([0, 2], np.int64)
    fa = _FlatAlleles(byte_data, seq_offsets, var_offsets, (1, 1, None))
    fa.reverse_masked(np.array([True], np.bool_))
    assert calls["n"] == 1
    # "AC"->"GT", "G"->"C"
    assert fa.byte_data.tobytes() == b"GTC"
```

> Confirm `get` is imported into `_flat_variants.py` as a module-level name (it is used by the `assemble_variant_buffers` call site at ~line 1085 via `get("assemble_variant_buffers")`). If it is imported as `from .._dispatch import get`, the monkeypatch target `fv.get` is correct.

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/parity/test_rc_alleles_parity.py::test_flat_alleles_reverse_masked_uses_rc_alleles -q`
Expected: FAIL — `calls["n"] == 0` (still calls seqpro directly).

- [ ] **Step 3: Implement the routing**

Replace `_FlatAlleles.reverse_masked` body (`_flat_variants.py` ~lines 119-142) with:

```python
    def reverse_masked(self, mask: NDArray[np.bool_]) -> "_FlatAlleles":
        """DNA reverse-complement the mask-selected rows' alleles, in place.

        ``mask`` is one entry per region (length ``b``); broadcast across ploidy
        to a per-(b*p) row mask, then expanded per-allele inside the dispatched
        ``rc_alleles`` kernel (rust default, seqpro reference).
        """
        m = np.ascontiguousarray(mask, np.bool_).reshape(-1)
        per_bp = np.repeat(m, self.ploidy)  # per-(b*p) row mask
        get("rc_alleles")(
            self.byte_data,
            np.asarray(self.seq_offsets, np.int64),
            np.asarray(self.var_offsets, np.int64),
            per_bp,
        )
        return self
```

In `RaggedVariants.rc_` (`_rag_variants.py` ~line 333), replace the single line:

```python
                _sp_reverse_complement(view, _COMP, mask=allele_mask, copy=False)
```

with a call to the dispatched kernel on the same `data` buffer. Two details:
1. `data` is `S1` dtype (`chars.data.copy()`), but `rc_alleles` requires `uint8` — pass
   `data.view(np.uint8)` (shares the buffer, so the in-place RC propagates back into
   `data`, which `Ragged.from_offsets(data, ...)` then consumes at the next line).
2. `rc_` already computed the per-allele `allele_mask` (length `n_alleles`), so make each
   allele its own row via `var_offsets = arange(n_alleles+1)` — the kernel's row→allele
   expansion is then the identity, reproducing the prior `mask=allele_mask` semantics:

```python
                get("rc_alleles")(
                    data.view(np.uint8),
                    np.asarray(char_off, np.int64),
                    np.arange(n_alleles + 1, dtype=np.int64),
                    allele_mask,
                )
```

Remove the now-unused `from seqpro.rag import reverse_complement as _sp_reverse_complement`
import at the top of `rc_` if it has no other use in that method (keep `_COMP` import
only if still referenced; otherwise drop it). Add `from .._dispatch import get` and
`import numpy as np` if not already imported at module scope in `_rag_variants.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/parity/test_rc_alleles_parity.py -q`
Expected: PASS (all, incl. the new spy test).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py python/genvarloader/_dataset/_rag_variants.py tests/parity/test_rc_alleles_parity.py
rtk git commit -m "refactor: route variant-allele RC through dispatched rc_alleles kernel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Remove the dead spliced variant guard in `_query.py`

**Files:**
- Modify: `python/genvarloader/_dataset/_query.py` (`_getitem_spliced`, ~lines 306-321)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_getitem_spliced` no longer references `_VARIANT_TYPES_S`; spliced RC post-pass remains for the seq/annotated kinds only (the only kinds reachable on the spliced path).

- [ ] **Step 1: Write the failing test (assert the guard is gone / spliced variants still rejected)**

Add to `tests/dataset/test_query_spliced.py` (create if absent; otherwise append):

```python
import inspect

from genvarloader._dataset import _query


def test_spliced_has_no_dead_variant_guard():
    src = inspect.getsource(_query._getitem_spliced)
    assert "_VARIANT_TYPES_S" not in src, (
        "spliced variant RC guard is unreachable (spliced variants are rejected "
        "upstream) and must be removed"
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_query_spliced.py -q`
Expected: FAIL — `_VARIANT_TYPES_S` still present in source.

- [ ] **Step 3: Implement the removal**

In `_getitem_spliced` (`_query.py` ~lines 306-321), replace the backend-split block:

```python
    if view.rc_neg and to_rc_per_elem is not None:
        if _active_backend() == "numba":
            # Numba: RC handled entirely by post-pass for all kinds.
            recon = tuple(reverse_complement_ragged(r, to_rc_per_elem) for r in recon)
        else:
            # Rust: flat-seq kinds folded RC in-kernel (or Python-side inside the
            # reconstructor).  Spliced output is never a variant type, so this
            # branch is effectively a no-op, but we keep the guard symmetric
            # with the unspliced path for correctness.
            _VARIANT_TYPES_S = (RaggedVariants, _FlatVariants, _FlatVariantWindows)
            recon = tuple(
                reverse_complement_ragged(r, to_rc_per_elem)
                if isinstance(r, _VARIANT_TYPES_S)
                else r
                for r in recon
            )
```

with:

```python
    if view.rc_neg and to_rc_per_elem is not None:
        # Spliced output is never a variant type (spliced variants are rejected
        # upstream in Haps.__call__). On numba the post-pass RCs the seq/annotated
        # kinds; on rust those kinds fold RC in-kernel, so this is a no-op there.
        if _active_backend() == "numba":
            recon = tuple(reverse_complement_ragged(r, to_rc_per_elem) for r in recon)
```

Then remove any now-unused imports in `_query.py` that were referenced ONLY by the
deleted branch (`_FlatVariants`, `RaggedVariants`, `_FlatVariantWindows` may still be
used by the unspliced path / overloads — check with `rg` before deleting; only drop
truly unused names).

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_query_spliced.py -q && pixi run -e dev ruff check python/genvarloader/_dataset/_query.py`
Expected: PASS; ruff clean (no unused-import error).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_query.py tests/dataset/test_query_spliced.py
rtk git commit -m "refactor: drop unreachable spliced variant-RC guard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: End-to-end neg-strand variants parity + dummy-fill / custom-allele coverage

**Files:**
- Modify: `tests/parity/test_variants_dataset_parity.py` (add neg-strand variant-RC cases + `rc_alleles` spy)

**Context (read before writing):** the existing `tests/parity/test_dataset_parity.py::test_neg_strand_parity` already proves byte-identical neg-strand output across backends for `["reference","haplotypes","annotated","tracks","tracks-seqs","haps-tracks"]` — but **not `variants`**. That is the gap this task fills, reusing the same fixture (`tests/parity/_fixtures.py::build_strand_mixed_dataset`, which has −strand regions at indices 1 and 3) and the `_compare_ragged_field` helper already in `test_variants_dataset_parity.py`.

**Design note (why dummy-fill is NOT a divergence risk here):** RC is applied via the dispatched `rc_alleles` kernel at the **same call site on both backends** (the `_query.py` post-pass → `reverse_masked`), which runs **after** dummy-fill. So dummy alleles are RC'd identically by rust and reference. The custom non-palindromic dummy case below is therefore regression-locking coverage (rust kernel handles dummy-filled buffers exactly like the seqpro reference), not a hunt for an ordering bug.

**Interfaces:**
- Consumes: `build_strand_mixed_dataset` (`tests/parity/_fixtures.py`); `synthetic_case` fixture (provides `.svar_path`, `.ref_path`); `_compare_ragged_field` (same file); `DummyVariant` (`genvarloader._dataset._flat_variants`); `_dispatch._REGISTRY` / `backends` (spy pattern, mirror `test_variants_getitem_parity_and_kernels_invoked`).
- Produces: byte-identical alt/ref assertions (rust vs reference) for a neg-strand variants read, with a non-vacuity guard that `rc_alleles` actually fires, plus a custom-dummy variant case.

- [ ] **Step 1: Write the failing tests**

Append to `tests/parity/test_variants_dataset_parity.py` (imports at top: add
`from genvarloader._dataset._flat_variants import DummyVariant` and
`from ._fixtures import build_strand_mixed_dataset` — match the import style already
used by `test_dataset_parity.py:33`):

```python
def _read_variants_both_backends(ds, monkeypatch):
    """Read ds[:, :] under numba then rust; return (out_numba, out_rust)."""
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]
    return out_numba, out_rust


def test_neg_strand_variants_rc_parity_and_kernel_invoked(
    tmp_path, synthetic_case, monkeypatch
):
    """variants-mode neg-strand RC is byte-identical across backends, and the
    rust rc_alleles kernel actually fires on the live read (non-vacuous)."""
    import genvarloader as gvl

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = gvl.Dataset.open(ds_dir, reference=ref).with_tracks(False).with_seqs("variants")

    # Non-vacuity: fixture must carry −strand regions (rc_neg defaults True).
    assert np.any(ds._full_regions[:, 3] == -1), "fixture has no −strand regions"

    # Spy on the rust rc_alleles to prove it runs on the live neg-strand path.
    numba_fn, rust_fn = _dispatch.backends("rc_alleles")
    calls = {"n": 0}

    def _spy_rust(*a, **k):
        calls["n"] += 1
        return rust_fn(*a, **k)

    orig_entry = dict(_dispatch._REGISTRY["rc_alleles"])
    _dispatch.register("rc_alleles", numba=numba_fn, rust=_spy_rust, default="rust")
    try:
        out_numba, out_rust = _read_variants_both_backends(ds, monkeypatch)
    finally:
        _dispatch._REGISTRY["rc_alleles"] = orig_entry

    assert calls["n"] > 0, (
        "rust rc_alleles was never invoked on the neg-strand variants read — "
        "the backstop is vacuous. Confirm a variant overlaps a −strand region; if "
        "the synthetic variant set does not, extend build_strand_mixed_dataset with a "
        "−strand region positioned over a known variant."
    )
    for field_name in out_numba.fields:
        _compare_ragged_field(out_numba[field_name], out_rust[field_name], field_name)


def test_neg_strand_variants_custom_dummy_parity(tmp_path, synthetic_case, monkeypatch):
    """A custom non-palindromic dummy (alt/ref = b'AC') filled into empty groups on
    a −strand read is RC'd identically by rust and the seqpro reference."""
    import genvarloader as gvl

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = (
        gvl.Dataset.open(ds_dir, reference=ref)
        .with_tracks(False)
        .with_seqs("variants")
        .with_settings(dummy_variant=DummyVariant(alt=b"AC", ref=b"AC"))
    )
    assert np.any(ds._full_regions[:, 3] == -1), "fixture has no −strand regions"

    out_numba, out_rust = _read_variants_both_backends(ds, monkeypatch)
    for field_name in out_numba.fields:
        _compare_ragged_field(out_numba[field_name], out_rust[field_name], field_name)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/parity/test_variants_dataset_parity.py -k "neg_strand_variants" -q --basetemp=$(pwd)/.pytest_tmp`
Expected: with Tasks 1-4 already landed this should PASS; run it FIRST against the
pre-Task-4 state to confirm it would fail (e.g. temporarily on the prior commit it
errors on the missing `rc_alleles` registry entry). If both already pass because
Tasks 1-4 are merged, treat this task as adding the missing live-path coverage and
proceed to Step 4. If `calls["n"] == 0`, apply the fixture fallback in the assert msg.

- [ ] **Step 3: (only if vacuous) extend the fixture**

If the spy reports 0 calls, the synthetic variant set has no variant over a −strand
region. In `tests/parity/_fixtures.py::build_strand_mixed_dataset`, add a −strand BED
row positioned over a known variant from `synthetic_case` (e.g. the GAGA→G chr1
deletion region is at +; mirror its coordinates as a −strand region) so a −strand
group is non-empty. Re-run Step 2. (No production code changes.)

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/parity/test_variants_dataset_parity.py -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (existing tests + the two new neg-strand cases).

- [ ] **Step 5: Commit**

```bash
rtk git add tests/parity/test_variants_dataset_parity.py tests/parity/_fixtures.py
rtk git commit -m "test(parity): e2e neg-strand variants RC + custom-dummy, rc_alleles live spy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Full-tree verification + roadmap update

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (Target 6 section: tick the deferred variant-RC follow-up; record the new gvl `rc_alleles` kernel + retained seqpro reference)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: green full tree on both backends; roadmap reflecting reality.

- [ ] **Step 1: Lint, format, typecheck**

Run:
```bash
pixi run -e dev ruff format python/ tests/
pixi run -e dev ruff check python/ tests/
pixi run -e dev typecheck
```
Expected: all clean (format may rewrite the new test files — re-stage if so).

- [ ] **Step 2: cargo tests**

Run: `pixi run -e dev cargo test`
Expected: all pass (incl. the 3 new `rc_alleles_inplace` tests).

- [ ] **Step 3: Full pytest tree on BOTH backends**

Run:
```bash
pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: both green (same passed/xfailed counts as the Target-7 baseline `967 passed / 21 skipped / 4 xfailed`, modulo the new tests added here). Investigate any new failure before proceeding — do NOT claim success without reading the output.

- [ ] **Step 4: Update the roadmap**

In `docs/roadmaps/rust-migration.md`, under Target 6 (~lines 468-489), add a follow-up note (and tick the deferred variant-RC item):

```markdown
   **✅ Variant-allele RC folded (follow-up, 2026-06-25).** The two deferred kinds
   (`RaggedVariants` + `_FlatVariants`) no longer route variant-allele RC through the
   seqpro post-pass with per-batch ragged object churn; a gvl rust kernel
   (`variants::rc_alleles_inplace`, FFI `rc_alleles`, dispatch `rc_alleles` default
   rust) RCs the raw `_FlatAlleles` buffers in place, applied AFTER dummy-fill so
   ordering stays byte-identical (custom non-palindromic dummy alleles covered). The
   seqpro implementation is retained as the registered reference backend (parity + perf
   gating; deletion is Phase 5). `_FlatVariantWindows` remains never-RC'd. Plan:
   `docs/superpowers/plans/2026-06-25-rust-variant-rc-fold.md`.
```

- [ ] **Step 5: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): variant-allele RC folded onto gvl rust kernel (Target 6 follow-up)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Extension import path:** the compiled rust module is `genvarloader.genvarloader`,
  imported in `_flat_variants.py` (line ~20) as `from ..genvarloader import <name>`. Reuse
  that verbatim for `rc_alleles`; tests import `genvarloader.genvarloader` directly.
- **In-place is load-bearing:** `rc_alleles` mutates `byte_data`. Never wrap the caller's
  `byte_data` in `np.ascontiguousarray` on a path that could copy (non-contiguous/non-uint8)
  — assert contiguity instead (Task 3). The `_FlatAlleles.byte_data` buffer is contiguous
  `uint8` by construction.
- **The reference IS the oracle:** there is no numba `rc_helper`; the seqpro path is the
  byte-identical reference. Parity tests compare rust vs that reference, not vs a numba
  kernel.
- **Don't touch `flank_tokens` or windows:** RC applies only to `alt`/`ref` allele bytes,
  matching the current post-pass exactly.
```
