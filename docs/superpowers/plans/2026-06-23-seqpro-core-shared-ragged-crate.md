# Shared `seqpro-core` Rust Ragged Crate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract seqpro's Rust ragged kernels into a pyo3-free `seqpro-core` rlib that owns a `Ragged` layout type, port its last two numba ops (`to_padded`, `reverse_complement`) to Rust, bump seqpro's pymodule to pyo3 0.28, and have GenVarLoader consume the core via an rlib path-dep with one proof-point op.

**Architecture:** The shared substrate is a normal Rust `rlib` with **zero pyo3**. seqpro's pymodule (pyo3 0.28 after the bump) and GVL's pymodule (pyo3 0.28) each link `seqpro-core` and do their own numpy↔core bridging. The ops GVL consumes take **plain slices** (`&[i64]`, `&[u8]`) in their public signatures so no `ndarray` version coupling crosses the FFI boundary.

**Tech Stack:** Rust (ndarray 0.17, rayon, pyo3 0.28, numpy 0.28), maturin, Python 3.10–3.13, numba (being removed from seqpro's rag layer), pixi, pytest, proptest.

## Global Constraints

- **seqpro repo path:** `~/projects/seqpro`. **GVL repo path:** `/Users/david/projects/GenVarLoader`. (Same case-insensitive repo as `~/projects/SeqPro` — use the lowercase path.)
- **`seqpro-core` is pyo3-free.** No `pyo3`/`numpy`-crate dependency in `crates/seqpro-core/Cargo.toml`. Ever.
- **Public API of ops GVL consumes (`to_padded`, `reverse_complement`) takes plain slices**, not `ndarray` views — keeps GVL decoupled from seqpro-core's ndarray version.
- **seqpro pymodule:** bump `pyo3` 0.20→**0.28**, `numpy` 0.20→**0.28**, `ndarray` 0.15→**0.17** (match GVL). Retain `abi3-py39`. GVL stays `pyo3 0.28 / abi3-py310`.
- **Byte-identical parity** is the landing gate for every ported op. Delete a numba impl only after its Rust replacement passes parity.
- **Local editable mode:** GVL takes `seqpro-core = { path = "../seqpro/crates/seqpro-core" }`. Flip to a git/release dep before GVL ships (tracked, not done here).
- **Building seqpro's Rust ext needs** `PYO3_PYTHON=~/projects/seqpro/.pixi/envs/dev/bin/python` (see [[reference_gvl_dev_env_gotchas]]). Pure-Rust `seqpro-core` tests run via plain `cargo test -p seqpro-core` (no Python).
- **Conventional commits.** End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Commands shown unprefixed; prefix with `rtk` where a filter exists (per CLAUDE.md).

---

## File Structure

**seqpro repo (`~/projects/seqpro`):**
- Create: `crates/seqpro-core/Cargo.toml` — pyo3-free crate manifest.
- Create: `crates/seqpro-core/src/lib.rs` — module root re-exporting the layout + ops.
- Create: `crates/seqpro-core/src/ragged.rs` — `Ragged` layout type + all ragged ops + unit/proptest.
- Modify: `Cargo.toml` — add `[workspace]`, `seqpro-core` path dep; bump pyo3/numpy/ndarray.
- Modify: `src/lib.rs` — pyo3 0.28 Bound API; `#[pyfunction]` wrappers delegate to `seqpro_core`; register two new ones.
- Modify: `src/ragged.rs` — gutted to a re-export shim (or deleted; logic now lives in the core crate).
- Modify: `src/{translate,kshuffle,kmer_encode}.rs` — pyo3 0.28 Bound API adjustments.
- Modify: `python/seqpro/rag/_ops.py` — `to_padded`/`reverse_complement` call the new Rust pyfunctions; delete the two numba kernels.

**GVL repo (`/Users/david/projects/GenVarLoader`):**
- Modify: `Cargo.toml` — add `seqpro-core` path dep.
- Create: `src/ragged/mod.rs` — pyo3 0.28 bridge: numpy → `seqpro_core` → numpy.
- Modify: `src/lib.rs` — `pub mod ragged;` + register `ragged_to_padded`.
- Modify: `python/genvarloader/_ragged.py` — `to_padded()` routes to the Rust bridge.
- Create: `tests/unit/test_ragged_to_padded_rust.py` — differential parity vs seqpro Python.
- Modify: `docs/roadmaps/rust-migration.md` — Phase 1/6 reframe + notes log.

---

# REPO A — seqpro (`~/projects/seqpro`)

### Task 1: Bump seqpro pymodule to pyo3 0.28 / numpy 0.28 / ndarray 0.17

This is the "bump" landed as its own reviewable unit, ahead of the extraction, so a parity failure is easy to localize. No behavior change — only API migration.

**Files:**
- Modify: `Cargo.toml` (deps block)
- Modify: `src/lib.rs` (pymodule + all `#[pyfunction]` signatures)
- Modify: `src/translate.rs`, `src/kshuffle.rs`, `src/kmer_encode.rs` (only where pyo3/numpy types appear)

**Interfaces:**
- Produces: an importable `seqpro` extension built against pyo3 0.28; all existing Python-visible function names/signatures unchanged.

- [ ] **Step 1: Bump the dependency versions in `Cargo.toml`**

```toml
[dependencies]
anyhow = "1.0.99"
ndarray = { version = "0.17", features = ["rayon"] }
numpy = "0.28.0"
rand = { version = "0.8.5", features = ["small_rng"] }
rayon = "1.11.0"
thiserror = "1.0.69"
xxhash-rust = { version = "0.8.15", features = ["xxh3"] }

[dependencies.pyo3]
version = "0.28"
features = ["abi3-py39"]
```

- [ ] **Step 2: Migrate `src/lib.rs` to the pyo3 0.28 Bound API**

The 0.20→0.28 migration is mechanical. Use GVL's `src/lib.rs` (already pyo3 0.28) as the reference style. Apply these exact deltas:

- pymodule signature: `fn seqpro(_py: Python, m: &PyModule)` → `fn seqpro(m: &Bound<'_, PyModule>)`. Drop the unused `_py` param; `wrap_pyfunction!(f, m)` still works with `&Bound`.
- Return types `&'py PyArray<T, D>` → `Bound<'py, PyArray<T, D>>`; `IntoPyArray::into_pyarray(py)` now returns `Bound` (same call site, new type). Update the `type` aliases accordingly, e.g.:

```rust
type RaggedSelectResult<'py> = PyResult<(Bound<'py, PyArray<i64, Ix1>>, Bound<'py, PyArray<i64, Ix1>>)>;
type NestedPackResult<'py> = PyResult<(
    Bound<'py, PyArray<i64, Ix1>>,
    Bound<'py, PyArray<i64, Ix1>>,
    Bound<'py, PyArray<u8, Ix1>>,
)>;
type RaggedConcatResult<'py> = PyResult<(Bound<'py, PyArray<u8, Ix1>>, Bound<'py, PyArray<i64, Ix1>>)>;
```

- `&'py PyList` parameter → `&Bound<'py, PyList>`; iterate with `.iter()` (yields `Bound<PyAny>`), `.extract::<PyReadonlyArray1<u8>>()` unchanged.
- `PyReadonlyArray1<'py, T>` / `PyReadwriteArray1` / `PyReadonlyArray<'py, T, IxDyn>` params: unchanged names; `.as_array()`, `.as_array_mut()`, `.as_slice()` unchanged.
- `&'py PyArray<u8, IxDyn>` return (the `_k_shuffle` fn) → `Bound<'py, PyArray<u8, IxDyn>>`.

- [ ] **Step 3: Migrate `translate.rs`, `kshuffle.rs`, `kmer_encode.rs`**

These are mostly pure Rust. Apply the same `&'py PyArray` → `Bound<'py, PyArray>` and `&PyModule`/`Python` adjustments only where pyo3/numpy types appear in signatures. Pure-Rust internals are untouched.

- [ ] **Step 4: Build the extension into the dev env**

Run: `cd ~/projects/seqpro && PYO3_PYTHON=$PWD/.pixi/envs/dev/bin/python pixi run -e dev maturin develop`
Expected: builds and installs `seqpro` without error.

- [ ] **Step 5: Run cargo + Python tests to confirm no behavior change**

Run: `cd ~/projects/seqpro && cargo test`
Expected: PASS (existing Rust unit tests in `ragged.rs`, `kshuffle` etc.).

Run: `cd ~/projects/seqpro && pixi run -e dev pytest tests/ -q`
Expected: PASS — same counts as before the bump (the ragged/translate/kshuffle Python suites exercise every migrated `#[pyfunction]`).

- [ ] **Step 6: Commit**

```bash
cd ~/projects/seqpro
git add Cargo.toml Cargo.lock src/
git commit -m "build(deps): bump pyo3 0.20->0.28, numpy/ndarray to match GVL

Mechanical Bound-API migration; no behavior change. Aligns seqpro's
pymodule with GenVarLoader's pyo3 generation ahead of the seqpro-core
extraction.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Extract the pyo3-free `seqpro-core` crate with a `Ragged` layout type

Move the existing pure-Rust ragged kernels (already pyo3-free in `src/ragged.rs`) into a new workspace member, wrapped behind a `Ragged` borrowed-view layout type. seqpro's `#[pyfunction]` wrappers then delegate to `seqpro_core`. No Python-visible change.

**Files:**
- Create: `crates/seqpro-core/Cargo.toml`
- Create: `crates/seqpro-core/src/lib.rs`
- Create: `crates/seqpro-core/src/ragged.rs` (logic moved from `src/ragged.rs`)
- Modify: `Cargo.toml` (workspace + dep)
- Modify: `src/lib.rs` (wrappers delegate to `seqpro_core`)
- Modify/Delete: `src/ragged.rs` (becomes a re-export of `seqpro_core::ragged`, or is deleted with `lib.rs` importing the core crate directly)

**Interfaces:**
- Produces (consumed by Task 3/4 and seqpro's wrappers):
  - `seqpro_core::ragged::Ragged<'a>` — `{ offsets: &'a [i64], data: &'a [u8], elem: usize }` borrowed view, with `Ragged::new(offsets, data, elem)`, `.n_rows() -> usize`, `.lengths() -> Vec<i64>`, `.validate() -> Result<(), String>`.
  - Free fns (moved verbatim, signatures preserved): `validate`, `select`, `nested_gather`, `nested_pack`, `pack_into`, `pack`, `ragged_concat`.

- [ ] **Step 1: Create the crate manifest `crates/seqpro-core/Cargo.toml`**

```toml
[package]
name = "seqpro-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "seqpro_core"

[dependencies]
ndarray = { version = "0.17", features = ["rayon"] }
rayon = "1.11.0"

[dev-dependencies]
proptest = "1.4"
```

- [ ] **Step 2: Move the ragged logic into `crates/seqpro-core/src/ragged.rs`**

Move the entire body of the current `~/projects/seqpro/src/ragged.rs` (the pure-Rust `ragged_concat`, `nested_gather`, `select`, `validate`, `nested_pack`, `pack_into`, `pack`, and the `#[cfg(test)] mod tests`) into this file unchanged. Then add the `Ragged` layout type at the top:

```rust
use ndarray::prelude::*;

/// Borrowed, zero-copy view of a single-axis ragged array over a flat byte buffer.
/// `offsets` are in element units (length n_rows+1); `data` is the packed bytes;
/// `elem` is the byte size of one logical element.
pub struct Ragged<'a> {
    pub offsets: &'a [i64],
    pub data: &'a [u8],
    pub elem: usize,
}

impl<'a> Ragged<'a> {
    pub fn new(offsets: &'a [i64], data: &'a [u8], elem: usize) -> Self {
        Self { offsets, data, elem }
    }

    pub fn n_rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn lengths(&self) -> Vec<i64> {
        self.offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Validate monotonic offsets covering exactly `data.len()/elem` elements.
    pub fn validate(&self) -> Result<(), String> {
        if self.elem == 0 {
            return Err("elem must be positive".into());
        }
        let n_data = (self.data.len() / self.elem) as i64;
        validate(ArrayView1::from(self.offsets), n_data, self.n_rows() as i64)
    }
}
```

(The two ported ops in Task 3/4 are added as methods on this `impl`.)

- [ ] **Step 3: Create `crates/seqpro-core/src/lib.rs`**

```rust
pub mod ragged;
pub use ragged::Ragged;
```

- [ ] **Step 4: Wire the workspace in seqpro's root `Cargo.toml`**

Add at the top of `~/projects/seqpro/Cargo.toml`:

```toml
[workspace]
members = ["crates/seqpro-core"]
```

And add to its `[dependencies]`:

```toml
seqpro-core = { path = "crates/seqpro-core" }
```

- [ ] **Step 5: Delegate seqpro's wrappers to the core crate**

In `~/projects/seqpro/src/ragged.rs`, replace the whole file with a re-export so existing `ragged::*` paths in `lib.rs` keep resolving:

```rust
pub use seqpro_core::ragged::*;
```

`src/lib.rs` is unchanged from Task 1 — its `ragged::select(...)` etc. calls now resolve through the re-export to `seqpro_core`.

- [ ] **Step 6: Run cargo tests for both crates**

Run: `cd ~/projects/seqpro && cargo test -p seqpro-core`
Expected: PASS — the moved `mod tests` (e.g. `test_pack_normal_multi_row`, `test_select_gathers`) all pass, no Python needed.

Run: `cd ~/projects/seqpro && cargo test`
Expected: PASS for the whole workspace.

- [ ] **Step 7: Rebuild ext + run Python ragged suite (no behavior change)**

Run: `cd ~/projects/seqpro && PYO3_PYTHON=$PWD/.pixi/envs/dev/bin/python pixi run -e dev maturin develop && pixi run -e dev pytest tests/test_ragged.py tests/test_concatenate.py tests/test_ragged_nested_consumers.py -q`
Expected: PASS — identical counts to Task 1.

- [ ] **Step 8: Commit**

```bash
cd ~/projects/seqpro
git add Cargo.toml Cargo.lock crates/ src/
git commit -m "refactor(rust): extract pyo3-free seqpro-core crate owning the Ragged layout

Move the ragged kernels into crates/seqpro-core (no pyo3); add a borrowed
Ragged layout view; pymodule wrappers delegate via re-export. Pure-Rust
core is cargo-testable standalone and consumable by sibling crates.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Port `to_padded` into `seqpro-core` and remove its numba kernel

**Files:**
- Modify: `crates/seqpro-core/src/ragged.rs` (add `to_padded_into` method + proptest)
- Modify: `src/lib.rs` (add `_ragged_to_padded` pyfunction + register it)
- Modify: `python/seqpro/rag/_ops.py` (call the Rust fn; delete `_to_padded_copy`)

**Interfaces:**
- Consumes: `seqpro_core::Ragged` (Task 2).
- Produces:
  - `seqpro_core::Ragged::to_padded_into(&self, out: &mut [u8], itemsize: usize, out_len: usize) -> Result<(), String>` — copies each row's `min(row_len, out_len)` elements into a pre-filled `out` (flat uint8 of `n_rows*out_len*itemsize` bytes). Mirrors `_to_padded_copy` exactly.
  - Python: `seqpro.seqpro._ragged_to_padded(data_u1, offsets, out_u1, itemsize, out_len)`.

- [ ] **Step 1: Write the failing Rust proptest in `crates/seqpro-core/src/ragged.rs`**

Add inside `#[cfg(test)] mod tests`:

```rust
use proptest::prelude::*;

// Naive reference: row-major copy with truncation into a pre-filled buffer.
fn to_padded_ref(offsets: &[i64], data: &[u8], elem: usize, out_len: usize, fill: u8) -> Vec<u8> {
    let n = offsets.len() - 1;
    let mut out = vec![fill; n * out_len * elem];
    for i in 0..n {
        let row_len = (offsets[i + 1] - offsets[i]) as usize;
        let ncopy = row_len.min(out_len);
        let src = offsets[i] as usize * elem;
        let dst = i * out_len * elem;
        out[dst..dst + ncopy * elem].copy_from_slice(&data[src..src + ncopy * elem]);
    }
    out
}

proptest! {
    #[test]
    fn to_padded_matches_reference(
        rows in proptest::collection::vec(0usize..6, 1..8),
        elem in 1usize..4,
        out_len in 0usize..7,
    ) {
        let mut offsets = vec![0i64];
        for r in &rows { offsets.push(offsets.last().unwrap() + *r as i64); }
        let n_data = *offsets.last().unwrap() as usize;
        let data: Vec<u8> = (0..n_data * elem).map(|x| (x % 251) as u8).collect();

        let mut out = vec![0xAAu8; rows.len() * out_len * elem];
        Ragged::new(&offsets, &data, elem)
            .to_padded_into(&mut out, elem, out_len)
            .unwrap();

        let expected = to_padded_ref(&offsets, &data, elem, out_len, 0xAA);
        prop_assert_eq!(out, expected);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/projects/seqpro && cargo test -p seqpro-core to_padded`
Expected: FAIL — `no method named to_padded_into`.

- [ ] **Step 3: Implement `to_padded_into` on `Ragged`**

Add to the `impl<'a> Ragged<'a>` block:

```rust
/// Copy each row's first `min(row_len, out_len)` elements into a pre-filled
/// `out` (flat uint8 view of a row-major (n_rows, out_len) buffer already
/// filled with the pad value). Parallel across rows.
pub fn to_padded_into(&self, out: &mut [u8], itemsize: usize, out_len: usize) -> Result<(), String> {
    use rayon::prelude::*;
    let n = self.n_rows();
    let row_stride = out_len * itemsize;
    if out.len() != n * row_stride {
        return Err(format!("out has {} bytes, expected {}", out.len(), n * row_stride));
    }
    if row_stride == 0 {
        return Ok(());
    }
    out.par_chunks_mut(row_stride).enumerate().for_each(|(i, dst)| {
        let row_len = (self.offsets[i + 1] - self.offsets[i]) as usize;
        let ncopy = row_len.min(out_len);
        let nbytes = ncopy * itemsize;
        let src = self.offsets[i] as usize * itemsize;
        dst[..nbytes].copy_from_slice(&self.data[src..src + nbytes]);
    });
    Ok(())
}
```

- [ ] **Step 4: Run the proptest to verify it passes**

Run: `cd ~/projects/seqpro && cargo test -p seqpro-core to_padded`
Expected: PASS.

- [ ] **Step 5: Add the `#[pyfunction]` wrapper in `src/lib.rs`**

Add the function and register it in the `#[pymodule]` (`m.add_function(wrap_pyfunction!(_ragged_to_padded, m)?)?;`):

```rust
#[pyfunction]
fn _ragged_to_padded(
    data: PyReadonlyArray1<u8>,
    offsets: PyReadonlyArray1<i64>,
    mut out: PyReadwriteArray1<u8>,
    itemsize: usize,
    out_len: usize,
) -> PyResult<()> {
    let data = data.as_slice()?;
    let offsets = offsets.as_slice()?;
    let out = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("out must be contiguous"))?;
    seqpro_core::Ragged::new(offsets, data, itemsize)
        .to_padded_into(out, itemsize, out_len)
        .map_err(PyValueError::new_err)
}
```

- [ ] **Step 6: Swap the numba call in `python/seqpro/rag/_ops.py`**

In `to_padded`, replace the kernel call (currently `_to_padded_copy(data_u1, offsets, out_u1, itemsize, out_len)`) with:

```python
from seqpro.seqpro import _ragged_to_padded  # type: ignore[missing-import]  # rust
_ragged_to_padded(data_u1, offsets, out_u1, itemsize, out_len)
```

Then **delete** the `@nb.njit ... def _to_padded_copy(...)` definition (lines 327–353).

- [ ] **Step 7: Rebuild and run the existing to_padded suite as the parity gate**

Run: `cd ~/projects/seqpro && PYO3_PYTHON=$PWD/.pixi/envs/dev/bin/python pixi run -e dev maturin develop && pixi run -e dev pytest tests/test_ragged_to_padded.py tests/test_shape_matrix.py -q`
Expected: PASS — the existing behavioral suite confirms byte-identical output with the Rust kernel.

- [ ] **Step 8: Commit**

```bash
cd ~/projects/seqpro
git add crates/ src/lib.rs python/seqpro/rag/_ops.py
git commit -m "feat(rag): port to_padded to seqpro-core Rust, drop numba kernel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Port `reverse_complement` into `seqpro-core` and remove its numba kernel

**Files:**
- Modify: `crates/seqpro-core/src/ragged.rs` (add `reverse_complement_inplace` + proptest)
- Modify: `src/lib.rs` (add `_ragged_reverse_complement` pyfunction + register)
- Modify: `python/seqpro/rag/_ops.py` (call the Rust fn; delete `_reverse_complement_ragged`)

**Interfaces:**
- Produces:
  - `seqpro_core::ragged::reverse_complement_inplace(data: &mut [u8], offsets: &[i64], comp_lut: &[u8], mask: &[bool]) -> Result<(), String>` — in-place per-row reverse-complement over masked rows; `comp_lut` is a 256-entry byte→byte table. (Free fn, not a `Ragged` method, because it mutates `data`.)
  - Python: `seqpro.seqpro._ragged_reverse_complement(u1, offsets, comp_lut, mask)`.

- [ ] **Step 1: Write the failing Rust proptest in `crates/seqpro-core/src/ragged.rs`**

```rust
fn rc_ref(data: &[u8], offsets: &[i64], comp: &[u8], mask: &[bool]) -> Vec<u8> {
    let mut out = data.to_vec();
    for i in 0..offsets.len() - 1 {
        if !mask[i] { continue; }
        let lo = offsets[i] as usize;
        let hi = offsets[i + 1] as usize;
        let row: Vec<u8> = data[lo..hi].iter().rev().map(|&b| comp[b as usize]).collect();
        out[lo..hi].copy_from_slice(&row);
    }
    out
}

proptest! {
    #[test]
    fn rc_matches_reference(rows in proptest::collection::vec(0usize..6, 1..8)) {
        let mut offsets = vec![0i64];
        for r in &rows { offsets.push(offsets.last().unwrap() + *r as i64); }
        let n = *offsets.last().unwrap() as usize;
        let data: Vec<u8> = (0..n).map(|x| (x % 4) as u8).collect();      // 0..3 = A,C,G,T
        let comp: Vec<u8> = (0..256u32).map(|b| match b { 0=>3,1=>2,2=>1,3=>0, o=>o as u8 }).collect();
        let mask: Vec<bool> = rows.iter().enumerate().map(|(i,_)| i % 2 == 0).collect();

        let mut got = data.clone();
        super::reverse_complement_inplace(&mut got, &offsets, &comp, &mask).unwrap();
        prop_assert_eq!(got, rc_ref(&data, &offsets, &comp, &mask));
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/projects/seqpro && cargo test -p seqpro-core rc_matches`
Expected: FAIL — `cannot find function reverse_complement_inplace`.

- [ ] **Step 3: Implement `reverse_complement_inplace` (free fn in `ragged.rs`)**

```rust
/// In-place per-row reverse-complement over a flat S1 buffer for masked rows.
/// `comp_lut` is a 256-entry byte->byte complement table. Length-preserving;
/// offsets are unchanged. Parallel across rows (rows write disjoint spans).
pub fn reverse_complement_inplace(
    data: &mut [u8],
    offsets: &[i64],
    comp_lut: &[u8],
    mask: &[bool],
) -> Result<(), String> {
    use rayon::prelude::*;
    if comp_lut.len() != 256 {
        return Err(format!("comp_lut must have 256 entries, got {}", comp_lut.len()));
    }
    let n = offsets.len() - 1;
    if mask.len() != n {
        return Err(format!("mask has {} entries, expected {}", mask.len(), n));
    }
    // Build disjoint per-row mutable slices, then process in parallel.
    let mut rows: Vec<&mut [u8]> = Vec::with_capacity(n);
    let mut rest: &mut [u8] = data;
    for i in 0..n {
        let len = (offsets[i + 1] - offsets[i]) as usize;
        let (head, tail) = rest.split_at_mut(len);
        rows.push(head);
        rest = tail;
    }
    rows.par_iter_mut().enumerate().for_each(|(i, row)| {
        if !mask[i] { return; }
        let len = row.len();
        for j in 0..len / 2 {
            let a = row[j];
            let b = row[len - 1 - j];
            row[j] = comp_lut[b as usize];
            row[len - 1 - j] = comp_lut[a as usize];
        }
        if len % 2 == 1 {
            row[len / 2] = comp_lut[row[len / 2] as usize];
        }
    });
    Ok(())
}
```

- [ ] **Step 4: Run the proptest to verify it passes**

Run: `cd ~/projects/seqpro && cargo test -p seqpro-core rc_matches`
Expected: PASS.

- [ ] **Step 5: Add the `#[pyfunction]` wrapper in `src/lib.rs` + register it**

```rust
#[pyfunction]
fn _ragged_reverse_complement(
    mut data: PyReadwriteArray1<u8>,
    offsets: PyReadonlyArray1<i64>,
    comp_lut: PyReadonlyArray1<u8>,
    mask: PyReadonlyArray1<bool>,
) -> PyResult<()> {
    let data = data.as_slice_mut().map_err(|_| PyValueError::new_err("data must be contiguous"))?;
    seqpro_core::ragged::reverse_complement_inplace(
        data,
        offsets.as_slice()?,
        comp_lut.as_slice()?,
        mask.as_slice()?,
    )
    .map_err(PyValueError::new_err)
}
```

Register: `m.add_function(wrap_pyfunction!(_ragged_reverse_complement, m)?)?;`

- [ ] **Step 6: Swap the numba call in `python/seqpro/rag/_ops.py`**

In `reverse_complement`, replace `_reverse_complement_ragged(u1, offsets, comp_lut, mask_flat)` with:

```python
from seqpro.seqpro import _ragged_reverse_complement  # type: ignore[missing-import]  # rust
_ragged_reverse_complement(u1, offsets, comp_lut, mask_flat)
```

Then **delete** the `@nb.njit ... def _reverse_complement_ragged(...)` definition (lines 23–55). If `import numba as nb` is now unused in the file, remove it.

- [ ] **Step 7: Rebuild and run the rc suite as the parity gate**

Run: `cd ~/projects/seqpro && PYO3_PYTHON=$PWD/.pixi/envs/dev/bin/python pixi run -e dev maturin develop && pixi run -e dev pytest tests/test_ragged_rc.py -q`
Expected: PASS.

- [ ] **Step 8: Confirm seqpro's rag layer is numba-free**

Run: `cd ~/projects/seqpro && grep -rn "import numba\|nb.njit\|@nb\." python/seqpro/rag/`
Expected: no matches.

- [ ] **Step 9: Commit**

```bash
cd ~/projects/seqpro
git add crates/ src/lib.rs python/seqpro/rag/_ops.py
git commit -m "feat(rag): port reverse_complement to seqpro-core Rust; rag layer numba-free

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

# REPO B — GenVarLoader (`/Users/david/projects/GenVarLoader`)

### Task 5: Add the `seqpro-core` rlib dep and a GVL `ragged` bridge

**Files:**
- Modify: `Cargo.toml`
- Create: `src/ragged/mod.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `seqpro_core::Ragged::to_padded_into` (Task 3).
- Produces: Python `genvarloader.genvarloader.ragged_to_padded(data_u1, offsets, out_u1, itemsize, out_len)` — same contract as seqpro's `_ragged_to_padded`, linked through the rlib (no Python-seqpro round-trip).

- [ ] **Step 1: Add the path dep to `Cargo.toml`**

In `[dependencies]`:

```toml
seqpro-core = { path = "../seqpro/crates/seqpro-core" }
```

(Local editable mode. Must use the same `ndarray = "0.17"` major as `seqpro-core` — already true; verify with `cargo tree -p ndarray` showing a single version.)

- [ ] **Step 2: Create `src/ragged/mod.rs`**

```rust
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Copy each ragged row into a pre-filled (n_rows, out_len) uint8 buffer via
/// the shared seqpro-core kernel. Mirrors seqpro's `_ragged_to_padded`.
#[pyfunction]
pub fn ragged_to_padded(
    data: PyReadonlyArray1<u8>,
    offsets: PyReadonlyArray1<i64>,
    mut out: PyReadwriteArray1<u8>,
    itemsize: usize,
    out_len: usize,
) -> PyResult<()> {
    let data = data.as_slice()?;
    let offsets = offsets.as_slice()?;
    let out = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("out must be contiguous"))?;
    seqpro_core::Ragged::new(offsets, data, itemsize)
        .to_padded_into(out, itemsize, out_len)
        .map_err(PyValueError::new_err)
}
```

- [ ] **Step 3: Register the module + function in `src/lib.rs`**

Add `pub mod ragged;` near the other `pub mod` lines, and inside `fn genvarloader(...)`:

```rust
m.add_function(wrap_pyfunction!(ragged::ragged_to_padded, m)?)?;
```

- [ ] **Step 4: Build GVL's extension**

Run: `cd /Users/david/projects/GenVarLoader && pixi run -e dev maturin develop`
Expected: builds and links `seqpro-core` (a single `ndarray` version in the tree); installs `genvarloader`.

- [ ] **Step 5: Smoke-test the bridge from Python**

Run:
```bash
cd /Users/david/projects/GenVarLoader && pixi run -e dev python -c "
import numpy as np
from genvarloader.genvarloader import ragged_to_padded
data = np.arange(5, dtype=np.uint8)            # rows [0,1],[2,3,4]
offsets = np.array([0,2,5], dtype=np.int64)
out = np.full(2*3, 255, np.uint8)              # n_rows=2, out_len=3, itemsize=1
ragged_to_padded(data, offsets, out, 1, 3)
print(out.reshape(2,3))
"
```
Expected: `[[0 1 255] [2 3 4]]`.

- [ ] **Step 6: Commit**

```bash
cd /Users/david/projects/GenVarLoader
git add Cargo.toml Cargo.lock src/lib.rs src/ragged/
git commit -m "feat(rust): consume seqpro-core via rlib; add ragged_to_padded bridge

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Reroute GVL's `to_padded` chokepoint to the Rust bridge with parity

GVL's `python/genvarloader/_ragged.py:294 to_padded()` is the single pass-through every caller (`RaggedSeqs`, `RaggedAnnotatedHaps`, `_reference.py`, `_query.py`) routes through. Reroute it; prove byte-identical vs seqpro's Python path.

**Files:**
- Modify: `python/genvarloader/_ragged.py` (the module-level `to_padded`)
- Create: `tests/unit/test_ragged_to_padded_rust.py`

**Interfaces:**
- Consumes: `genvarloader.genvarloader.ragged_to_padded` (Task 5).
- Produces: GVL `to_padded(rag, pad_value)` returns the same dense array as `seqpro.rag.to_padded`, now computed via the GVL Rust bridge.

- [ ] **Step 1: Write the failing differential test `tests/unit/test_ragged_to_padded_rust.py`**

```python
import numpy as np
import pytest
from seqpro.rag import Ragged
from seqpro.rag import to_padded as sp_to_padded
from genvarloader._ragged import to_padded as gvl_to_padded


@pytest.mark.parametrize("dtype,pad", [("S1", b"N"), ("i4", -1), ("f4", 0.0)])
@pytest.mark.parametrize("rows", [[0, 1, 3, 2], [5], [0, 0, 4]])
def test_gvl_to_padded_matches_seqpro(dtype, pad, rows):
    offsets = np.concatenate([[0], np.cumsum(rows)]).astype(np.int64)
    n = int(offsets[-1])
    data = (np.arange(n, dtype=np.int64) % 4).astype(dtype) if dtype != "S1" \
        else np.frombuffer(b"ACGT" * (n // 4 + 1), dtype="S1")[:n]
    rag = Ragged.from_offsets(np.ascontiguousarray(data), (len(rows),), offsets)
    np.testing.assert_array_equal(gvl_to_padded(rag, pad), sp_to_padded(rag, pad))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/david/projects/GenVarLoader && pixi run -e dev pytest tests/unit/test_ragged_to_padded_rust.py -q`
Expected: FAIL — GVL `to_padded` still equals seqpro's by delegation, so this *passes trivially today*. To make it a real gate, first confirm it passes (delegation), THEN reroute in Step 3 and confirm it still passes against the Rust path. (This test is the regression guard, not a red-then-green TDD pair — the behavior already exists; we are swapping its implementation.)

Run it now and expect: PASS (current delegation).

- [ ] **Step 3: Reroute `to_padded` in `python/genvarloader/_ragged.py`**

Replace the body of the module-level `to_padded` (currently `return _sp_to_padded(rag, pad_value)`) with a GVL-owned wrapper that mirrors seqpro's orchestration but calls the Rust bridge for the inner copy. Add the import `from .genvarloader import ragged_to_padded` near the top, then:

```python
def to_padded(rag: Ragged[RDTYPE], pad_value: Any) -> NDArray[RDTYPE]:
    """Densify a Ragged into a right-padded array via GVL's seqpro-core Rust bridge.

    Byte-identical to :func:`seqpro.rag.to_padded`; the inner row-copy runs in
    the shared seqpro-core kernel (Rust->Rust, no Python-seqpro round-trip).
    """
    if rag._is_record:
        raise NotImplementedError("to_padded is not defined on record-layout Ragged arrays.")
    rag_dim = rag.rag_dim
    if any(d is not None for d in rag.shape[rag_dim + 1 :]):
        raise ValueError(
            f"to_padded requires the ragged axis to be last, got shape {rag.shape}."
        )
    if not rag.is_contiguous:
        rag = spr.to_packed(rag)

    offsets = np.ascontiguousarray(rag.offsets, dtype=np.int64)
    n_rows = offsets.shape[0] - 1
    out_len = int(rag.lengths.max()) if n_rows else 0

    rag_data: NDArray[Any] = rag.data  # record layout rejected above
    dtype = rag_data.dtype
    out = np.full((n_rows, out_len), pad_value, dtype=dtype)
    if n_rows and out_len:
        data_u1 = np.ascontiguousarray(rag_data).reshape(-1).view(np.uint8)
        out_u1 = out.reshape(-1).view(np.uint8)
        ragged_to_padded(data_u1, offsets, out_u1, dtype.itemsize, out_len)

    leading = rag.shape[:rag_dim]
    if leading:
        out = out.reshape((*leading, out_len))  # pyrefly: ignore[no-matching-overload]
    return out
```

Keep the `_sp_to_padded` import only if still used elsewhere in the file; otherwise remove it. (Note: GVL's `to_padded` has no `length=` parameter today — preserve that; only `pad_value` is passed by all call sites.)

- [ ] **Step 4: Run the differential test against the Rust path**

Run: `cd /Users/david/projects/GenVarLoader && pixi run -e dev pytest tests/unit/test_ragged_to_padded_rust.py -q`
Expected: PASS — byte-identical to seqpro across S1/i4/f4 and all row shapes.

- [ ] **Step 5: Run the broader GVL ragged + dataset suites (no regressions)**

Run: `cd /Users/david/projects/GenVarLoader && pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: PASS — `RaggedSeqs.to_padded`, `RaggedAnnotatedHaps.to_padded`, and `_reference.py`/`_query.py` consumers all route through the rerouted function.

- [ ] **Step 6: Lint + format**

Run: `cd /Users/david/projects/GenVarLoader && pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/`
Expected: clean (ruff format makes no changes, or commit the changes it makes). See [[feedback_ruff_format_hook]].

- [ ] **Step 7: Commit**

```bash
cd /Users/david/projects/GenVarLoader
git add python/genvarloader/_ragged.py tests/unit/test_ragged_to_padded_rust.py
git commit -m "feat(ragged): route to_padded through seqpro-core Rust bridge

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Refresh the Rust-migration roadmap

**Files:**
- Modify: `docs/roadmaps/rust-migration.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Reframe Phase 6 and the goal/end-state**

In `## Goal & end state`, change the "Eventual scope" paragraph: seqpro's ragged primitives are the **shared Rust substrate** (`seqpro-core` rlib), not absorbed. In `### Phase 6`, delete the bullet "Bring ragged primitives fully in-house — drop the seqpro hot-path dependency" and narrow Phase 6 to genoray (variant IO) only; add a line: "seqpro-core remains the ragged substrate (decision 2026-06-23)."

- [ ] **Step 2: Reframe Phase 1**

Rewrite the Phase 1 bullets to: extract pyo3-free `seqpro-core` owning the `Ragged` layout; port the last two numba ops (`to_padded`, `reverse_complement`); GVL consumes via rlib path-dep; one proof-point op (`to_padded`) rerouted with byte-identical parity. Tick the items completed by Tasks 2–6. Keep the awkward-removal item checked. Set the Phase 1 marker to ✅ (or 🚧 if you split follow-ups) with the PR links.

- [ ] **Step 3: Update the target crate layout + migration contract**

Under `### Target crate layout`, note GVL's `ragged/` is a **bridge** to `seqpro-core`, not a reimplementation. In `## The migration contract`, add a line clarifying the ragged layer is *consumed* from seqpro-core, not reimplemented in GVL's crate.

- [ ] **Step 4: Add the notes-log entry**

Append under `## Notes & decisions log`:

```markdown
- 2026-06-23: seqpro is the shared Rust ragged substrate. Extracted a pyo3-free
  `seqpro-core` rlib (crates/seqpro-core) owning a borrowed `Ragged` layout +
  ops; ported its last two numba kernels (`to_padded`, `reverse_complement`) to
  Rust (seqpro rag layer now numba-free). Bumped seqpro's pymodule to pyo3 0.28 /
  numpy 0.28 / ndarray 0.17 (hygiene; NOT required for the link — two pymodules
  with different pyo3 versions coexist; the single-version rule is per-cdylib, and
  the shared core is pyo3-free). GVL links seqpro-core via a path dep (editable;
  flip to git/release before shipping) and routes its `to_padded` chokepoint
  through the shared kernel (proof-point, byte-identical parity). Inverts Phase 6
  (seqpro stays the substrate). PRs: seqpro TBD, GVL TBD.
```

- [ ] **Step 5: Commit**

```bash
cd /Users/david/projects/GenVarLoader
git add docs/roadmaps/rust-migration.md
git commit -m "docs(roadmap): seqpro-core shared substrate; Phase 1 realized, Phase 6 inverted

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Cross-repo ordering is strict:** Tasks 1–4 (seqpro) must land before Tasks 5–6 (GVL) build, because GVL's `cargo build` links `seqpro-core` and calls `to_padded_into` (Task 3). Rebuild seqpro's ext (`maturin develop`) after Tasks 3/4 so seqpro's own Python tests see the new `#[pyfunction]`s.
- **One ndarray version:** after Task 5, run `cargo tree -p ndarray` in the GVL repo and confirm a single `ndarray 0.17.x` resolves. Two versions = the FFI types won't unify; pin seqpro-core to match GVL.
- **Parity philosophy:** Tasks 3/4 lean on seqpro's existing behavioral suites (`test_ragged_to_padded.py`, `test_ragged_rc.py`) plus the new core-level proptests as the oracle; the numba impls are deleted in the same task once green. Task 6's differential test compares GVL's rerouted path directly against `seqpro.rag.to_padded`.
- **Full-tree check before pushing GVL:** per CLAUDE.md, run `pixi run -e dev pytest tests -q` once at the end — scoped runs skip `tests/unit/`.
