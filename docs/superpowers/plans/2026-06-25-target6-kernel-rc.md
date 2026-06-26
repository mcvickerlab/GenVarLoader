# Target 6 — Kernel Reverse-Complement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit negative-strand read-path output already reverse-complemented from the Rust fused kernels, removing the cold batch-wide seqpro RC post-pass for the rust backend while keeping the numba path (the parity oracle) byte-identical.

**Architecture:** Add two generic in-place primitives in a new `src/reverse.rs` that reverse (optionally complement) each masked row of a flat `(data, offsets)` buffer. Thread an optional per-row `to_rc` mask into each fused kernel; when present, the kernel RC's each negative-strand query/element's slice **in place, immediately after it is written, inside the existing per-query loop** (hot in cache). Python computes the mask (reusing the existing strand and splice-permutation logic) and, on the rust backend only, stops applying the Python RC post-pass to the five flat output kinds. The numba composed path keeps the existing `reverse_complement_ragged` post-pass unchanged. `RaggedVariants` RC is deferred to Target 7 and continues to use the Python post-pass on both backends.

**Tech Stack:** Rust (PyO3, ndarray) for kernels; Python (numpy) for orchestration; pixi for env/build (`maturin develop`); pytest + cargo for tests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-25-target6-kernel-rc-design.md` (read before starting).
- Roadmap: `docs/roadmaps/rust-migration.md` — Phase 5, round-2 optimization block. Tick Target 6, record re-measured ratios, set PR link, set the "Target 6 must merge before rayon" marker as part of this work.
- **Parity is the landing gate: output must be byte-identical between backends.** Run both:
  `pixi run -e dev pytest tests/parity -q` (rust default) and `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q` (oracle).
- `_COMP` LUT contract (reproduce exactly from `python/genvarloader/_ragged.py:330`, `bytes.maketrans(b"ACGT", b"TGCA")`): a `[u8; 256]` that is **identity for everything** except `A(0x41)↔T(0x54)` and `C(0x43)↔G(0x47)` (uppercase only). `N`, IUPAC codes, and lowercase `a/c/g/t` are pass-through.
- Scope: five flat-buffer kinds (haplotypes, reference, tracks, annotated, splice). **Out of scope:** `RaggedVariants` (deferred to Target 7), `variant-windows`/`intervals` (no-op).
- Do **not** delete `reverse_complement_ragged` or its `_query.py`/`_reference.py` call — it remains the numba oracle. It becomes backend-and-kind-conditional only.
- Do not reintroduce per-batch `np.ascontiguousarray` on sample-scale memmaps (keeps `tests/integration/test_scale_guard.py` green).
- Build before any test run in this worktree: `pixi run -e dev maturin develop --release` (the shared `.pixi` env's installed extension points at the original checkout until rebuilt here).
- HPC: run pytest with `--basetemp=$(pwd)/.pytest_tmp` so the write path's `os.link` hardlink does not fail cross-device (Errno 18).
- Commit message style: conventional commits; end with the `Co-Authored-By` trailer.
- TDD order across kernels: reference → haplotypes → tracks → annotated → splice.

---

## File Structure

**Rust (create):**
- `src/reverse.rs` — the two in-place primitives + the `_COMP` LUT + cargo unit tests. One responsibility: reverse/reverse-complement masked rows of a flat buffer. Registered as a module in `src/lib.rs`.

**Rust (modify):**
- `src/ffi/mod.rs` — add an optional `to_rc` param to 5 fused kernels and call the primitive after the write.
- `src/reference/mod.rs` — `get_reference` core: accept `to_rc` and apply primitive (covers reference, spliced reference).
- Reconstruct/track cores under `src/{reconstruct,tracks}/` are **not** modified — RC is applied at the FFI layer over the assembled flat buffer, after the core returns, so cores stay untouched.

**Python (modify):**
- `python/genvarloader/_dataset/_query.py` — compute `to_rc`, thread it into `view.recon(...)`, make the post-pass backend-and-kind-conditional.
- `python/genvarloader/_dataset/_reference.py`, `_ref.py` — thread `to_rc` into `get_reference`/`_fetch_spliced_ref`; make the standalone RefDataset RC backend-conditional.
- `python/genvarloader/_dataset/_haps.py` — pass `to_rc` into the three haplotype fused kernels.
- `python/genvarloader/_dataset/_reconstruct.py` — pass `to_rc` into the track fused kernel; thread `to_rc` through `SeqsTracks`/`HapsTracks`/`Tracks.__call__`.
- `python/genvarloader/_dataset/_protocol.py` — add `to_rc` to the `Reconstructor.__call__` protocol signature.
- `python/genvarloader/_dataset/_ref.py` — `Ref.__call__` / wherever `get_reference` is called for an in-Dataset reference reconstructor.

**Tests (create/modify):**
- `src/reverse.rs` `#[cfg(test)]` — primitive unit tests.
- Per-kernel cargo tests in `src/ffi/` or alongside cores — synthetic reconstruct-then-RC checks (where the core is callable in pure Rust).
- `tests/parity/test_dataset_parity.py` — new strand=−1 fixtures + non-vacuity assertions for every in-scope kind.

---

## Task 1: `src/reverse.rs` in-place primitives + `_COMP` LUT

**Files:**
- Create: `src/reverse.rs`
- Modify: `src/lib.rs` (add `mod reverse;`)
- Test: `src/reverse.rs` `#[cfg(test)]`

**Interfaces:**
- Produces:
  - `pub const COMP: [u8; 256]` — ACGT↔TGCA, identity elsewhere.
  - `pub fn reverse_flat_rows_inplace<T: Copy>(data: &mut [T], offsets: ndarray::ArrayView1<i64>, to_rc: ndarray::ArrayView1<bool>)` — reverses element order within each masked row.
  - `pub fn rc_flat_rows_inplace(data: &mut [u8], offsets: ndarray::ArrayView1<i64>, to_rc: ndarray::ArrayView1<bool>)` — reverses **and** complements bytes via `COMP`.
- Contract: `offsets.len() == to_rc.len() + 1`. Row `i` spans `data[offsets[i]..offsets[i+1]]`. When `to_rc[i]` is false the row is untouched. Empty rows (`offsets[i] == offsets[i+1]`) are no-ops.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn comp_lut_matches_maketrans() {
        // identity except ACGT<->TGCA uppercase
        assert_eq!(COMP[b'A' as usize], b'T');
        assert_eq!(COMP[b'T' as usize], b'A');
        assert_eq!(COMP[b'C' as usize], b'G');
        assert_eq!(COMP[b'G' as usize], b'C');
        assert_eq!(COMP[b'N' as usize], b'N');
        assert_eq!(COMP[b'a' as usize], b'a'); // lowercase pass-through
        assert_eq!(COMP[b'c' as usize], b'c');
        assert_eq!(COMP[b'R' as usize], b'R'); // IUPAC pass-through
        assert_eq!(COMP[0u8 as usize], 0u8);
    }

    #[test]
    fn rc_reverses_and_complements_masked_rows_only() {
        // two rows: "ACGT" (rc -> "ACGT") and "AACG" (not rc)
        let mut data = b"ACGTAACG".to_vec();
        let offsets = array![0i64, 4, 8];
        let to_rc = array![true, false];
        rc_flat_rows_inplace(&mut data, offsets.view(), to_rc.view());
        assert_eq!(&data[0..4], b"ACGT"); // revcomp of ACGT is ACGT
        assert_eq!(&data[4..8], b"AACG"); // untouched
    }

    #[test]
    fn rc_handles_odd_length_and_n() {
        let mut data = b"ACN".to_vec(); // revcomp -> "NGT"
        let offsets = array![0i64, 3];
        let to_rc = array![true];
        rc_flat_rows_inplace(&mut data, offsets.view(), to_rc.view());
        assert_eq!(&data, b"NGT");
    }

    #[test]
    fn reverse_only_no_complement_f32() {
        let mut data = vec![1.0f32, 2.0, 3.0, 9.0];
        let offsets = array![0i64, 3, 4];
        let to_rc = array![true, false];
        reverse_flat_rows_inplace(&mut data, offsets.view(), to_rc.view());
        assert_eq!(data, vec![3.0, 2.0, 1.0, 9.0]);
    }

    #[test]
    fn reverse_only_i32_for_annot_arrays() {
        let mut data = vec![10i32, 11, 12];
        let offsets = array![0i64, 3];
        let to_rc = array![true];
        reverse_flat_rows_inplace(&mut data, offsets.view(), to_rc.view());
        assert_eq!(data, vec![12, 11, 10]);
    }

    #[test]
    fn empty_row_and_all_false_are_noops() {
        let mut data = b"AC".to_vec();
        let offsets = array![0i64, 0, 2]; // first row empty
        rc_flat_rows_inplace(&mut data, offsets.view(), array![true, false].view());
        assert_eq!(&data, b"AC");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev cargo test --lib reverse`
Expected: FAIL — `reverse.rs` / functions not defined (compile error).

- [ ] **Step 3: Write minimal implementation**

```rust
//! In-place reverse / reverse-complement of masked rows in a flat (data, offsets)
//! buffer. Used by the read-path kernels to emit negative-strand output already
//! reverse-complemented, replacing the Python RC post-pass on the rust backend.

use ndarray::ArrayView1;

/// ACGT<->TGCA complement, identity for every other byte. Mirrors
/// `bytes.maketrans(b"ACGT", b"TGCA")` (python/genvarloader/_ragged.py).
pub const COMP: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = i as u8;
        i += 1;
    }
    t[b'A' as usize] = b'T';
    t[b'T' as usize] = b'A';
    t[b'C' as usize] = b'G';
    t[b'G' as usize] = b'C';
    t
};

/// Reverse element order within each masked row (no complement). Generic over
/// element width so it serves f32 tracks and i32/i64 annotation arrays.
pub fn reverse_flat_rows_inplace<T: Copy>(
    data: &mut [T],
    offsets: ArrayView1<i64>,
    to_rc: ArrayView1<bool>,
) {
    for i in 0..to_rc.len() {
        if !to_rc[i] {
            continue;
        }
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        data[s..e].reverse();
    }
}

/// Reverse AND complement bytes within each masked row via `COMP`.
pub fn rc_flat_rows_inplace(
    data: &mut [u8],
    offsets: ArrayView1<i64>,
    to_rc: ArrayView1<bool>,
) {
    for i in 0..to_rc.len() {
        if !to_rc[i] {
            continue;
        }
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        let row = &mut data[s..e];
        row.reverse();
        for b in row.iter_mut() {
            *b = COMP[*b as usize];
        }
    }
}
```

Add `mod reverse;` to `src/lib.rs` near the other `mod` declarations.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev cargo test --lib reverse`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/reverse.rs src/lib.rs
git commit -m "feat(rust): in-place reverse/reverse-complement primitives for read path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: thread `to_rc` into the reference kernel (`get_reference`)

**Files:**
- Modify: `src/reference/mod.rs` (core `get_reference`), `src/ffi/mod.rs:728` (pyfunction)
- Test: `src/reference/mod.rs` `#[cfg(test)]`

**Interfaces:**
- Consumes: `reverse::rc_flat_rows_inplace`, `COMP` from Task 1.
- Produces: `get_reference` (core + pyfunction) gains a trailing optional `to_rc: Option<ArrayView1<bool>>` (core) / `to_rc: Option<PyReadonlyArray1<bool>>` (pyfunction). When `Some`, after building the output buffer the core calls `rc_flat_rows_inplace(out, out_offsets, to_rc)`. `None` ⇒ unchanged behavior.

- [ ] **Step 1: Write the failing test (core)**

```rust
// in src/reference/mod.rs #[cfg(test)]
#[test]
fn get_reference_applies_rc_when_masked() {
    // contig "ACGTAA" at offset 0; one region [0,4) -> "ACGT"
    let reference = ndarray::array![b'A', b'C', b'G', b'T', b'A', b'A'];
    let ref_offsets = ndarray::array![0i64, 6];
    let regions = ndarray::array![[0i32, 0, 4]];
    let out_offsets = ndarray::array![0i64, 4];
    let to_rc = ndarray::array![true];
    let out = get_reference(
        regions.view(), out_offsets.view(), reference.view(),
        ref_offsets.view(), b'N', false, Some(to_rc.view()),
    );
    // forward "ACGT" -> revcomp "ACGT"; use a non-palindrome to be sure:
    // region [0,3) "ACG" -> revcomp "CGT"
    assert_eq!(out.to_vec(), b"ACGT".to_vec());
}
```

(Adjust the assertion region to a non-palindrome, e.g. `[0,3)` → expect `b"CGT"`, so the test is non-vacuous.)

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev cargo test --lib reference`
Expected: FAIL — `get_reference` arity mismatch (no `to_rc` param).

- [ ] **Step 3: Implement**

In `src/reference/mod.rs`, add the trailing param and apply after the buffer is built:

```rust
pub fn get_reference(
    regions: ArrayView2<i32>,
    out_offsets: ArrayView1<i64>,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
    parallel: bool,
    to_rc: Option<ArrayView1<bool>>,
) -> Array1<u8> {
    let mut out = /* ...existing buffer build... */;
    if let Some(to_rc) = to_rc {
        crate::reverse::rc_flat_rows_inplace(
            out.as_slice_mut().unwrap(),
            out_offsets,
            to_rc,
        );
    }
    out
}
```

In `src/ffi/mod.rs:728`, add `to_rc: Option<PyReadonlyArray1<bool>>` as the trailing param and forward `to_rc.as_ref().map(|a| a.as_array())`. Update the Python caller `python/genvarloader/_dataset/_reference.py:686-695` (`_get_reference_rust`) to accept and pass `to_rc=None` for now (no behavior change — real mask wired in Task 7).

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev cargo test --lib reference`
Expected: PASS.

- [ ] **Step 5: Build + smoke the Python boundary**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev python -c "import genvarloader"`
Expected: import OK (signature change accepted).

- [ ] **Step 6: Commit**

```bash
git add src/reference/mod.rs src/ffi/mod.rs python/genvarloader/_dataset/_reference.py
git commit -m "feat(rust): optional in-kernel RC for get_reference

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: thread `to_rc` into `reconstruct_haplotypes_fused`

**Files:**
- Modify: `src/ffi/mod.rs:393-500`
- Test: `src/ffi/mod.rs` or a reconstruct core test module

**Interfaces:**
- Consumes: `reverse::rc_flat_rows_inplace`.
- Produces: `reconstruct_haplotypes_fused` gains trailing `to_rc: Option<PyReadonlyArray1<bool>>` (one bool per `(query, hap)` work item, length `n_work`). Applied to `out_data` against `out_offsets_vec` after Step 4 (the reconstruct write), before `into_pyarray`.

- [ ] **Step 1: Write the failing test**

Add a Rust test that drives the **reconstruct core** directly (it is pure Rust): reconstruct a tiny haplotype with no variants so output equals the reference window, then apply `rc_flat_rows_inplace` and assert the bytes equal the hand-computed revcomp. (Tests the exact call the kernel will make.)

```rust
#[test]
fn haplotype_buffer_rc_is_revcomp_of_forward() {
    let mut out = b"ACGTA".to_vec(); // pretend reconstructed forward bytes
    let offsets = ndarray::array![0i64, 5];
    let to_rc = ndarray::array![true];
    crate::reverse::rc_flat_rows_inplace(&mut out, offsets.view(), to_rc.view());
    assert_eq!(&out, b"TACGT"); // revcomp(ACGTA)
}
```

- [ ] **Step 2: Run to verify it fails / compiles red**

Run: `pixi run -e dev cargo test --lib`
Expected: FAIL until the kernel param is added (and this guard test passes once `reverse` is wired — it already exists from Task 1, so this step mainly guards the kernel arity change; verify the kernel signature change makes Python smoke fail first).

- [ ] **Step 3: Implement**

In `reconstruct_haplotypes_fused`, add trailing `to_rc: Option<PyReadonlyArray1<bool>>`. After Step 4 (`reconstruct::reconstruct_haplotypes_from_sparse(...)`), before `into_pyarray`:

```rust
if let Some(to_rc) = to_rc.as_ref() {
    crate::reverse::rc_flat_rows_inplace(
        out_data.as_slice_mut().unwrap(),
        out_offsets_vec.view(),
        to_rc.as_array(),
    );
}
```

Update the Python caller `_haps.py:828` to pass `to_rc=None` for now.

- [ ] **Step 4: Run tests + build**

Run: `pixi run -e dev cargo test --lib && pixi run -e dev maturin develop --release && pixi run -e dev python -c "import genvarloader"`
Expected: PASS + import OK.

- [ ] **Step 5: Commit**

```bash
git add src/ffi/mod.rs python/genvarloader/_dataset/_haps.py
git commit -m "feat(rust): optional in-kernel RC for reconstruct_haplotypes_fused

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: thread `to_rc` into `intervals_and_realign_track_fused` (reverse-only f32)

**Files:**
- Modify: `src/ffi/mod.rs:848` (and the f32 out buffer handling)
- Test: `src/ffi/mod.rs` `#[cfg(test)]`

**Interfaces:**
- Consumes: `reverse::reverse_flat_rows_inplace::<f32>`.
- Produces: `intervals_and_realign_track_fused` gains trailing `to_rc: Option<PyReadonlyArray1<bool>>` (one bool per `(query, hap)` row, length matching `out_offsets`). **Reverse only, no complement** (tracks are numeric). The `out` buffer is an in/out `PyReadwriteArray1<f32>`; apply over its slice against `out_offsets` after the realign write.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn track_buffer_rc_is_reverse_only() {
    let mut out = vec![1.0f32, 2.0, 3.0];
    let offsets = ndarray::array![0i64, 3];
    let to_rc = ndarray::array![true];
    crate::reverse::reverse_flat_rows_inplace(&mut out, offsets.view(), to_rc.view());
    assert_eq!(out, vec![3.0, 2.0, 1.0]); // no value transform
}
```

- [ ] **Step 2: Run to verify red on kernel arity**

Run: `pixi run -e dev cargo test --lib` then `maturin develop` smoke.
Expected: Python smoke fails on arity until param added.

- [ ] **Step 3: Implement**

Add trailing `to_rc: Option<PyReadonlyArray1<bool>>`. After the realign write into `out`:

```rust
if let Some(to_rc) = to_rc.as_ref() {
    crate::reverse::reverse_flat_rows_inplace(
        out.as_slice_mut().unwrap(),
        out_offsets.as_array(),
        to_rc.as_array(),
    );
}
```

Update the Python caller `_reconstruct.py:227` to pass `to_rc=None` for now.

- [ ] **Step 4: Run tests + build**

Run: `pixi run -e dev cargo test --lib && pixi run -e dev maturin develop --release && pixi run -e dev python -c "import genvarloader"`
Expected: PASS + import OK.

- [ ] **Step 5: Commit**

```bash
git add src/ffi/mod.rs python/genvarloader/_dataset/_reconstruct.py
git commit -m "feat(rust): optional in-kernel reverse for track realign kernel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: thread `to_rc` into `reconstruct_annotated_haplotypes_fused` (3 buffers in lockstep)

**Files:**
- Modify: `src/ffi/mod.rs:604-723`
- Test: `src/ffi/mod.rs` `#[cfg(test)]`

**Interfaces:**
- Consumes: `reverse::rc_flat_rows_inplace` (bytes) + `reverse::reverse_flat_rows_inplace::<i32>` (annotation arrays).
- Produces: trailing `to_rc: Option<PyReadonlyArray1<bool>>` (length `n_work`). Applies, per masked row over the shared `out_offsets_vec`: `rc_flat_rows_inplace(out_data)` (reverse+complement), `reverse_flat_rows_inplace(annot_v)` (reverse only), `reverse_flat_rows_inplace(annot_pos)` (reverse only) — all using the same offsets so the three stay aligned, matching `_FlatAnnotatedHaps.reverse_masked` (bytes complemented; `var_idxs`/`ref_coords` reversed without complement).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn annotated_rc_complements_bytes_reverses_indices() {
    let mut bytes = b"ACG".to_vec();          // revcomp -> "CGT"
    let mut vidx = vec![5i32, 6, 7];          // reverse -> [7,6,5]
    let mut rpos = vec![100i32, 101, 102];    // reverse -> [102,101,100]
    let offsets = ndarray::array![0i64, 3];
    let m = ndarray::array![true];
    crate::reverse::rc_flat_rows_inplace(&mut bytes, offsets.view(), m.view());
    crate::reverse::reverse_flat_rows_inplace(&mut vidx, offsets.view(), m.view());
    crate::reverse::reverse_flat_rows_inplace(&mut rpos, offsets.view(), m.view());
    assert_eq!(&bytes, b"CGT");
    assert_eq!(vidx, vec![7, 6, 5]);
    assert_eq!(rpos, vec![102, 101, 100]);
}
```

- [ ] **Step 2: Run to verify red on kernel arity**

Run: `pixi run -e dev cargo test --lib` + `maturin develop` smoke.
Expected: arity failure until added.

- [ ] **Step 3: Implement**

Add trailing `to_rc`. After Step 4 (reconstruct with annotation buffers), before returning:

```rust
if let Some(to_rc) = to_rc.as_ref() {
    let m = to_rc.as_array();
    crate::reverse::rc_flat_rows_inplace(out_data.as_slice_mut().unwrap(), out_offsets_vec.view(), m);
    crate::reverse::reverse_flat_rows_inplace(annot_v.as_slice_mut().unwrap(), out_offsets_vec.view(), m);
    crate::reverse::reverse_flat_rows_inplace(annot_pos.as_slice_mut().unwrap(), out_offsets_vec.view(), m);
}
```

Update the Python caller `_haps.py:984` to pass `to_rc=None` for now.

- [ ] **Step 4: Run tests + build**

Run: `pixi run -e dev cargo test --lib && pixi run -e dev maturin develop --release && pixi run -e dev python -c "import genvarloader"`
Expected: PASS + import OK.

- [ ] **Step 5: Commit**

```bash
git add src/ffi/mod.rs python/genvarloader/_dataset/_haps.py
git commit -m "feat(rust): optional in-kernel RC for annotated haplotype kernel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: thread `to_rc` into `reconstruct_haplotypes_spliced_fused` (permuted per-element)

**Files:**
- Modify: `src/ffi/mod.rs:521-577`
- Test: `src/ffi/mod.rs` `#[cfg(test)]`

**Interfaces:**
- Consumes: `reverse::rc_flat_rows_inplace`.
- Produces: trailing `to_rc: Option<PyReadonlyArray1<bool>>` — **already permuted per spliced element** (length = number of permuted elements = `out_offsets.len() - 1`). Applied over `out_offsets_a` (the permuted per-element offsets) so each masked element is RC'd in its own byte range, matching today's `to_rc_per_elem`. Assert in the caller (Task 7) that `to_rc.len() == out_offsets.len() - 1`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn spliced_rc_applies_per_element_over_permuted_offsets() {
    // two permuted elements: "ACG" (rc) and "TTT" (not rc)
    let mut out = b"ACGTTT".to_vec();
    let offsets = ndarray::array![0i64, 3, 6];
    let to_rc = ndarray::array![true, false];
    crate::reverse::rc_flat_rows_inplace(&mut out, offsets.view(), to_rc.view());
    assert_eq!(&out[0..3], b"CGT"); // revcomp(ACG)
    assert_eq!(&out[3..6], b"TTT"); // untouched
}
```

- [ ] **Step 2: Run to verify red on kernel arity**

Run: `pixi run -e dev cargo test --lib` + smoke.
Expected: arity failure until added.

- [ ] **Step 3: Implement**

Add trailing `to_rc`. After `reconstruct_haplotypes_from_sparse(...)`, before `into_pyarray`:

```rust
if let Some(to_rc) = to_rc.as_ref() {
    crate::reverse::rc_flat_rows_inplace(
        out_data.as_slice_mut().unwrap(),
        out_offsets_a,
        to_rc.as_array(),
    );
}
```

Update the Python caller `_haps.py:894` to pass `to_rc=None` for now.

- [ ] **Step 4: Run tests + build**

Run: `pixi run -e dev cargo test --lib && pixi run -e dev maturin develop --release && pixi run -e dev python -c "import genvarloader"`
Expected: PASS + import OK.

- [ ] **Step 5: Commit**

```bash
git add src/ffi/mod.rs python/genvarloader/_dataset/_haps.py
git commit -m "feat(rust): optional in-kernel RC for spliced haplotype kernel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: strand=−1 parity fixtures + non-vacuity assertions (safety net BEFORE wiring)

**Files:**
- Modify: `tests/parity/test_dataset_parity.py`

**Interfaces:**
- Consumes: existing dataset parity harness + kernel-spy backstop.
- Produces: parameterized fixtures with a **mix of `+` and `−`** strand regions covering haplotypes, reference, tracks, annotated, and the spliced variant of each; plus a non-vacuity assertion. These must **pass on the current (pre-wiring) code** (rust == numba, both via the post-pass), establishing the regression net that Task 8 must keep green.

- [ ] **Step 1: Write the strand=−1 parity fixtures**

Add a fixture that builds a dataset whose `input_regions` BED includes negative-strand rows (strand column `-1`) interleaved with positive ones, `max_jitter=0`. Parameterize over kinds `["haplotypes", "reference", "tracks", "tracks-seqs", "annotated"]` and spliced/unspliced. Assert byte-identical output between the two backends using the existing harness, and add:

```python
def test_negative_strand_actually_reverse_complements(neg_strand_dataset):
    # Non-vacuity: a '-' region's bytes differ from the '+'-oriented bytes.
    ds = neg_strand_dataset
    out = ds[neg_region_idx, sample_idx]
    fwd = forward_oriented_reference(ds, neg_region_idx, sample_idx)  # helper
    assert out.tobytes() != fwd.tobytes()  # RC genuinely fired
    assert out.tobytes() == revcomp(fwd).tobytes()  # and is the exact RC
```

(Use the spy backstop to assert the kernel ran on the live `__getitem__` path.)

- [ ] **Step 2: Run on current code, both backends**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/parity/test_dataset_parity.py -q --basetemp=$(pwd)/.pytest_tmp
GVL_BACKEND=numba pixi run -e dev pytest tests/parity/test_dataset_parity.py -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS on both (net established; the wiring isn't done yet, so both paths still use the post-pass).

- [ ] **Step 3: Commit**

```bash
git add tests/parity/test_dataset_parity.py
git commit -m "test(parity): strand=-1 fixtures + non-vacuity RC assertions

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Python wiring — thread real `to_rc`, make post-pass backend-and-kind-conditional

**Files:**
- Modify: `python/genvarloader/_dataset/_query.py` (`_getitem_unspliced` ~`:188`, `_getitem_spliced` ~`:259`), `_protocol.py`, `_reconstruct.py` (`SeqsTracks`/`HapsTracks`/`Tracks.__call__` + track kernel call), `_haps.py` (three kernel calls), `_reference.py` (`_get_reference_rust`, `_fetch_spliced_ref`, standalone RefDataset RC `:438`), `_ref.py` (`Ref.__call__` get_reference call).
- Test: `tests/parity/test_dataset_parity.py` (Task 7 fixtures stay green).

**Interfaces:**
- Consumes: every kernel's `to_rc` param (Tasks 2-6); Task 7 fixtures.
- Produces:
  - A helper `_active_backend() -> str` (returns `os.environ.get("GVL_BACKEND", "rust")`) so `_query.py`'s guard matches what the recon methods used. Place it next to the recon dispatch (e.g. `_reconstruct.py` or `_query.py`).
  - `to_rc` flows: `_query.py` computes the mask → `view.recon(..., to_rc=...)` → reconstructors forward it to the rust fused kernels (numba branch ignores it).
  - Post-pass becomes: numba ⇒ RC all kinds (unchanged); rust ⇒ RC only `RaggedVariants`.

- [ ] **Step 1: Add `to_rc` to the Reconstructor protocol + all `__call__`s**

In `_protocol.py`, add `to_rc: NDArray[np.bool_] | None = None` to `Reconstructor.__call__`. Mirror the param (trailing, default `None`) in `SeqsTracks.__call__`, `HapsTracks.__call__`, `Tracks.__call__`, `Ref.__call__`, `Haps.__call__`, and any kind variants. Each forwards `to_rc` to the fused kernel call on the rust branch only; the numba branch leaves it unused. For composite reconstructors (`SeqsTracks`, `HapsTracks`) forward the same `to_rc` to each sub-call.

- [ ] **Step 2: Pass `to_rc` into the rust kernels**

Replace the `to_rc=None` placeholders added in Tasks 2-6 with the forwarded `to_rc` (converted to a contiguous bool array on the rust branch: `None if to_rc is None else np.ascontiguousarray(to_rc, np.bool_)`). For tracks, the mask is per `(query, hap)` row — replicate the per-query mask across ploidy the same way `out_offsets` is laid out (mirror the existing `reverse_masked` broadcast: `np.repeat`/broadcast in C order to match `out_offsets` rows).

- [ ] **Step 3: Rewire `_query.py` post-pass (the core change)**

In `_getitem_unspliced`:

```python
to_rc = view.full_regions[r_idx, 3] == -1 if view.rc_neg else None
recon = view.recon(..., to_rc=to_rc)
if not isinstance(recon, tuple):
    recon = (recon,)
if view.rc_neg:
    if _active_backend() == "numba":
        recon = tuple(reverse_complement_ragged(r, to_rc) for r in recon)
    else:
        # rust folded flat-seq kinds in-kernel; only the deferred RaggedVariants
        # (Target 7) still needs the Python pass.
        recon = tuple(
            reverse_complement_ragged(r, to_rc) if isinstance(r, RaggedVariants) else r
            for r in recon
        )
```

In `_getitem_spliced`: keep the existing `to_rc_per_elem` computation, pass it into `view.recon(..., to_rc=to_rc_per_elem)`, and apply the identical numba-vs-rust guard. (Spliced output is never `RaggedVariants`, so the rust branch is a no-op there.)

- [ ] **Step 4: Rewire reference RC sites**

In `_reference.py`: thread `to_rc` into `_get_reference_rust`/`get_reference`. For the standalone RefDataset spliced path (`:438-444`), apply the same backend guard — on rust pass `to_rc_perm` into `_fetch_spliced_ref`→`get_reference` and skip `per_elem.reverse_masked`; on numba keep `per_elem.reverse_masked(to_rc_perm, comp=_COMP)`. In `_ref.py`, pass `to_rc` into the unspliced `get_reference` call on the rust branch.

- [ ] **Step 5: Confirm no other callers regressed**

Run: `grep -rn "reverse_complement_ragged\|reverse_masked" python/`
Expected: callers are only the numba-guarded post-pass + the RaggedVariants rust branch + the numba RefDataset branch. No stray unconditional RC remains on the rust path.

- [ ] **Step 6: Run the parity net + cargo, both backends**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev cargo test --lib
pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp
GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp
```
Expected: PASS on both backends (Task 7 fixtures now exercise rust in-kernel RC vs numba post-pass and stay byte-identical).

- [ ] **Step 7: Full tree, both backends**

Run:
```bash
pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/ && pixi run -e dev typecheck
```
Expected: PASS / clean.

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/
git commit -m "feat: fold strand RC into rust kernels; numba post-pass retained as oracle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: perf re-measure + roadmap update

**Files:**
- Modify: `docs/roadmaps/rust-migration.md`

**Interfaces:**
- Consumes: the de-noised `tests/benchmarks/test_e2e.py` harness + `tests/benchmarks/profiling/profile.py`.

- [ ] **Step 1: Re-measure rust÷numba ratios**

Run (release build already done):
```bash
pixi run -e dev pytest tests/benchmarks/test_e2e.py -q --basetemp=$(pwd)/.pytest_tmp
```
Compare the **min** per-batch for `haplotypes`, `tracks-only`, `tracks-seqs`, `annotated` against the starting points (haplotypes 0.94×, tracks-only 0.63×, etc.).

- [ ] **Step 2: Confirm RC self-time is gone from the rust profile**

Run:
```bash
NUMBA_NUM_THREADS=1 perf record -F 999 -o p.data -- .pixi/envs/dev/bin/python \
    tests/benchmarks/profiling/profile.py --mode haplotypes --n-batches 12000
perf report --stdio --no-children -i p.data | head -40
```
Expected: no `reverse_complement_*` / seqpro RC frame in the rust flat profile.

- [ ] **Step 3: Update the roadmap**

In `docs/roadmaps/rust-migration.md` round-2 block: tick Target 6, record the re-measured ratios under the Phase 5 checkpoint, set the PR link, and set/confirm the marker that **Target 6 must merge before rayon**.

- [ ] **Step 4: Commit**

```bash
git add docs/roadmaps/rust-migration.md
git commit -m "docs(roadmap): record Target 6 RC fold results; gate rayon on 5+6+7

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Two primitives + `_COMP` LUT → Task 1. ✓
- Five flat kinds in-kernel RC → Tasks 2 (reference), 3 (haplotypes), 4 (tracks, reverse-only), 5 (annotated, 3 buffers), 6 (splice, permuted). ✓
- Mask computed in Python, threaded as `Option<bool>`; `None` fast path → Task 8 steps 1-2 + each kernel's `Option`. ✓
- Insertion/trailing-fill ordering preserved (RC after forward write) → enforced by applying the primitive after the reconstruct core in every kernel task. ✓
- Backend-conditional post-pass; numba oracle unchanged; `reverse_complement_ragged` retained → Task 8 step 3 (corrects the spec's "delete" wording per the approved decision). ✓
- Third RC site `_reference.py:438` → Task 8 step 4. ✓
- `RaggedVariants` deferred to Target 7; still post-passed on both backends → Task 8 step 3 (rust branch RaggedVariants-only). ✓
- Vacuous-pass guard: strand=−1 fixtures + non-vacuity assertion → Task 7. ✓
- Parity both backends + full tree + lint/typecheck → Task 8 steps 6-7. ✓
- Perf re-measure + roadmap → Task 9. ✓
- Scale guard not regressed: no `ascontiguousarray` added on memmaps (only on small mask/region arrays) → respected in Task 8 step 2. ✓

**Type consistency:** `to_rc` is `Option<PyReadonlyArray1<bool>>` (pyfunction) / `Option<ArrayView1<bool>>` (core) / `NDArray[np.bool_] | None` (Python) throughout. Primitives named `reverse_flat_rows_inplace` / `rc_flat_rows_inplace` consistently. `_active_backend()` defined once (Task 8) and referenced in `_query.py`/`_reference.py`.

**Note on numba kernel test red/green:** the per-kernel cargo tests (Tasks 2-6) validate the primitive call against hand-computed revcomp on synthetic buffers; the kernel-arity change is smoke-checked via `maturin develop` + import. End-to-end RC correctness is gated by the Task 7 fixtures across the Task 8 flip. If a reconstruct core is not directly callable in a pure-Rust test for a given kernel, rely on the primitive's Task-1 unit tests + the Task 7 parity net (documented per task).
