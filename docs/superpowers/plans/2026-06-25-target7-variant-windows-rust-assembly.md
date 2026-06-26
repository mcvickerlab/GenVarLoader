# Target 7 — variant-windows/variants assembly in one Rust call — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the per-batch object/numpy-temporary churn on the `variants` + `variant-windows` flat-output read path into one flag-driven Rust call that owns the reference fetch + LUT tokenize + flank/window assembly and returns flat `(data, offsets)` buffers, so Python builds the wrapper objects once.

**Architecture:** A new Rust module `src/variants/windows.rs` holds small pure cores (`tokenize`, `slice_flanks`, `assemble_alt_window`, `fetch_windows`) and two mode orchestrators (`assemble_variants_mode`, `assemble_windows_mode`) generic over the token type. Two FFI pyfunctions (`assemble_variant_buffers_u8`, `assemble_variant_buffers_i32`) monomorphize the token type and return a `dict[str, (data, seq_offsets)]`. Python keeps the cheap, dtype-polymorphic front-end (v_idxs gather / AF filter / scalar-field gather) and the `fill_empty_groups` post-pass; only the ragged byte/token assembly tail moves to Rust, behind the dispatch registry with the existing Python/numba helpers retained as the parity oracle.

**Tech Stack:** Rust (`ndarray`, `numpy`/PyO3), Python (numpy, numba oracle), `pixi` for env/build/test, `maturin` for the Rust↔Python build, hypothesis + pytest parity harness.

## Global Constraints

- Branch `opt/target-7-windows-rust-assembly` off `zero-copy-scale-safe-readpath` (do NOT branch off `master`/`rust-migration`).
- Byte-identical parity is the landing gate: the Rust output must equal the existing Python/numba assembly (dtype, shape, values) for both `variants` and `variant-windows`, across the full `ref`/`alt` ∈ {window, allele} mode matrix, empty groups, and the `flank_tokens` ride-along.
- Front edge is **assembly tail only**: the v_idxs gather / AF filter / compaction / scalar-field gather stay in Python; the issue-#231 custom-FORMAT dtype-polymorphic numba fallback must remain intact (never route a custom-dtype field through the new typed Rust call).
- `fill_empty_groups` stays a separate Python post-pass over the existing `fill_empty_seq/scalar/fixed` Rust cores — do NOT fold it into the new call.
- Do NOT delete the numba/numpy assembly helpers (`compute_windows`, `compute_ref_window`, `compute_alt_window`, `tokenize_alleles`, `compute_flank_tokens`); they become the registered parity oracle.
- Do NOT reintroduce per-batch `np.ascontiguousarray` on sample-scale memmaps (keep `tests/integration/test_scale_guard.py` green). The mega-call's globals come from `Haps.ffi_static` (sub-linear, already cached) + the variant `ref`-allele bytes.
- Build after every Rust change: `pixi run -e dev maturin develop --release`. Rust unit tests: `pixi run -e dev cargo-test`. Python tests need `--basetemp=$(pwd)/.pytest_tmp` (HPC cross-device `os.link` Errno 18 guard).
- `test_e2e_variants` is a **pre-existing xfail** (`_FlatVariants.to_fixed` missing) — confirm it xfails identically at base; not a regression introduced here.
- Conventional commits; commit at the end of every task. End commit messages with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.

---

## File Structure

- **Create** `src/variants/windows.rs` — pure cores (`tokenize`, `slice_flanks`, `assemble_alt_window`, `fetch_windows`) + mode orchestrators (`assemble_variants_mode`, `assemble_windows_mode`) + the `VariantBufs<Tok>` return struct + Rust unit tests.
- **Modify** `src/variants/mod.rs` — add `pub mod windows;` and re-export nothing else (cores stay in the submodule).
- **Modify** `src/ffi/mod.rs` — two pyfunctions `assemble_variant_buffers_u8` / `assemble_variant_buffers_i32` returning a `PyDict`.
- **Modify** `src/lib.rs` — `add_function` for both pyfunctions.
- **Modify** `python/genvarloader/_dataset/_flat_flanks.py` — add `_assemble_variant_buffers_numba` (the oracle that composes existing helpers into the dict contract) — keeps all current helpers.
- **Modify** `python/genvarloader/_dataset/_flat_variants.py` — register `assemble_variant_buffers`, add the Rust shim that selects the u8/i32 monomorphization, and rewrite the `get_variants_flat` assembly tail to call `get("assemble_variant_buffers")` and wrap the returned dict once.
- **Modify** `tests/parity/_harness.py` — add `assert_kernel_parity_dict`.
- **Create** `tests/parity/test_assemble_variant_buffers_parity.py` — mode-matrix + empty + flank parity.
- **Modify** `tests/parity/test_dataset_parity.py` — spy that the kernel runs on the live windows/variants `__getitem__` path.
- **Modify** `docs/roadmaps/rust-migration.md` — tick target 7, record re-measured ratios, set PR link.

---

### Task 1: Rust pure cores — `tokenize`, `slice_flanks`, `assemble_alt_window`

**Files:**
- Create: `src/variants/windows.rs`
- Modify: `src/variants/mod.rs:1` (add `pub mod windows;`)
- Test: cargo unit tests inside `src/variants/windows.rs`

**Interfaces:**
- Produces:
  - `pub fn tokenize<Tok: Copy>(bytes: ArrayView1<u8>, lut: ArrayView1<Tok>) -> Array1<Tok>`
  - `pub fn slice_flanks(data: ArrayView1<u8>, rw_off: ArrayView1<i64>, flank_len: usize) -> (Array1<u8>, Array1<u8>)` — each `(n*flank_len,)`, variant-major: `f5[i*L+k] = data[rw_off[i]+k]`, `f3[i*L+k] = data[rw_off[i+1]-L+k]`
  - `pub fn assemble_alt_window(f5: ArrayView1<u8>, f3: ArrayView1<u8>, alt_data: ArrayView1<u8>, alt_seq_off: ArrayView1<i64>, flank_len: usize) -> (Array1<u8>, Array1<i64>)`

- [ ] **Step 1: Create the module file with the three cores**

Create `src/variants/windows.rs`:

```rust
//! Variant-windows / variants flat-buffer assembly cores (pure ndarray).
//! PyO3 lives in `crate::ffi`. Mirrors the Python helpers in
//! `_dataset/_flat_flanks.py` (`tokenize_alleles`, `_slice_flanks`,
//! `_assemble_alt_windows`, `compute_*`) — byte-identical by construction.
use ndarray::{Array1, ArrayView1};

/// Apply a 256-entry byte->token lookup table. `out[i] = lut[bytes[i]]`.
/// Mirrors numpy `lut[bytes]`. `Tok` is the token dtype (u8 or i32).
pub fn tokenize<Tok: Copy>(bytes: ArrayView1<u8>, lut: ArrayView1<Tok>) -> Array1<Tok> {
    let n = bytes.len();
    let mut out: Vec<Tok> = Vec::with_capacity(n);
    for i in 0..n {
        out.push(lut[bytes[i] as usize]);
    }
    Array1::from_vec(out)
}

/// Derive per-variant (f5, f3) fixed-`flank_len` flanks from a contiguous
/// per-variant window read `[start-L, end+L)`. `f5` = first `L` bytes of each
/// row, `f3` = last `L`. Both returned flat `(n*L,)`, variant-major. Mirrors
/// `_slice_flanks` (`f5 = data[rw_off[:-1,None]+cols]`,
/// `f3 = data[rw_off[1:,None]-L+cols]`).
pub fn slice_flanks(
    data: ArrayView1<u8>,
    rw_off: ArrayView1<i64>,
    flank_len: usize,
) -> (Array1<u8>, Array1<u8>) {
    let n = rw_off.len() - 1;
    let mut f5: Vec<u8> = Vec::with_capacity(n * flank_len);
    let mut f3: Vec<u8> = Vec::with_capacity(n * flank_len);
    for i in 0..n {
        let s = rw_off[i] as usize;
        let e = rw_off[i + 1] as usize;
        for k in 0..flank_len {
            f5.push(data[s + k]);
        }
        for k in 0..flank_len {
            f3.push(data[e - flank_len + k]);
        }
    }
    (Array1::from_vec(f5), Array1::from_vec(f3))
}

/// Concatenate `flank5 . alt . flank3` per variant into a flat byte buffer.
/// `f5`/`f3` are `(n*flank_len,)` variant-major. Mirrors numba
/// `_assemble_alt_windows`. Returns `(out_bytes, out_offsets)`.
pub fn assemble_alt_window(
    f5: ArrayView1<u8>,
    f3: ArrayView1<u8>,
    alt_data: ArrayView1<u8>,
    alt_seq_off: ArrayView1<i64>,
    flank_len: usize,
) -> (Array1<u8>, Array1<i64>) {
    let n = alt_seq_off.len() - 1;
    let mut out_off = Array1::<i64>::zeros(n + 1);
    for i in 0..n {
        let alt_len = alt_seq_off[i + 1] - alt_seq_off[i];
        out_off[i + 1] = out_off[i] + 2 * flank_len as i64 + alt_len;
    }
    let total = out_off[n] as usize;
    let mut out: Vec<u8> = Vec::with_capacity(total);
    for i in 0..n {
        for k in 0..flank_len {
            out.push(f5[i * flank_len + k]);
        }
        for k in alt_seq_off[i] as usize..alt_seq_off[i + 1] as usize {
            out.push(alt_data[k]);
        }
        for k in 0..flank_len {
            out.push(f3[i * flank_len + k]);
        }
    }
    (Array1::from_vec(out), out_off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_tokenize_u8() {
        // lut maps byte 65('A')->0, 67('C')->1, everything else->9 (unknown).
        let mut lut = vec![9u8; 256];
        lut[65] = 0;
        lut[67] = 1;
        let lut = Array1::from_vec(lut);
        let bytes = arr1(&[65u8, 67, 78]); // A, C, N(unknown)
        let out = tokenize(bytes.view(), lut.view());
        assert_eq!(out.to_vec(), vec![0u8, 1, 9]);
    }

    #[test]
    fn test_tokenize_i32() {
        // i32 tokens (alphabet larger than 255 forces i32 in Python).
        let mut lut = vec![999i32; 256];
        lut[71] = 300; // 'G' -> 300
        let lut = Array1::from_vec(lut);
        let bytes = arr1(&[71u8, 84]); // G, T(unknown)
        let out = tokenize(bytes.view(), lut.view());
        assert_eq!(out.to_vec(), vec![300i32, 999]);
    }

    #[test]
    fn test_slice_flanks() {
        // 2 variants, L=2. var0 window=[1,2,3,4,5] (len 5), var1=[6,7,8,9] (len 4).
        // rw_off = [0, 5, 9].
        let data = arr1(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9]);
        let rw_off = arr1(&[0i64, 5, 9]);
        let (f5, f3) = slice_flanks(data.view(), rw_off.view(), 2);
        // f5: first 2 of each = [1,2 | 6,7]; f3: last 2 of each = [4,5 | 8,9]
        assert_eq!(f5.to_vec(), vec![1u8, 2, 6, 7]);
        assert_eq!(f3.to_vec(), vec![4u8, 5, 8, 9]);
    }

    #[test]
    fn test_assemble_alt_window() {
        // L=1. f5=[10|20], f3=[11|21]. alt: var0="A"(65), var1="CG"(67,71).
        let f5 = arr1(&[10u8, 20]);
        let f3 = arr1(&[11u8, 21]);
        let alt_data = arr1(&[65u8, 67, 71]);
        let alt_seq_off = arr1(&[0i64, 1, 3]);
        let (out, off) = assemble_alt_window(
            f5.view(),
            f3.view(),
            alt_data.view(),
            alt_seq_off.view(),
            1,
        );
        // var0: 10, 65, 11  (2*1 + 1 = 3 bytes)
        // var1: 20, 67,71, 21  (2*1 + 2 = 4 bytes)
        assert_eq!(out.to_vec(), vec![10u8, 65, 11, 20, 67, 71, 21]);
        assert_eq!(off.to_vec(), vec![0i64, 3, 7]);
    }
}
```

- [ ] **Step 2: Wire the module in**

Add to `src/variants/mod.rs` as the first line after the module doc comment (line 1):

```rust
pub mod windows;
```

- [ ] **Step 3: Run the cores' unit tests to verify they pass**

Run: `pixi run -e dev cargo-test 2>&1 | rtk err`
Expected: the four new `windows::tests::*` tests PASS; existing tests still pass.

- [ ] **Step 4: Commit**

```bash
rtk git add src/variants/windows.rs src/variants/mod.rs
rtk git commit -m "feat(variants): add tokenize/slice_flanks/assemble_alt_window cores

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Rust `fetch_windows` helper (reference window reads)

**Files:**
- Modify: `src/variants/windows.rs`
- Test: cargo unit test inside `src/variants/windows.rs`

**Interfaces:**
- Consumes: `crate::reference::get_reference(regions: ArrayView2<i32>, out_offsets: ArrayView1<i64>, reference: ArrayView1<u8>, ref_offsets: ArrayView1<i64>, pad_char: u8, parallel: bool) -> Array1<u8>`
- Produces: `pub fn fetch_windows(v_contigs: ArrayView1<i32>, starts_v: ArrayView1<i32>, ilens_v: ArrayView1<i32>, flank_len: i64, reference: ArrayView1<u8>, ref_offsets: ArrayView1<i64>, pad_char: u8) -> (Array1<u8>, Array1<i64>)` — the per-variant `[start-L, end+L)` read flat buffer + its per-variant offsets (`rw_off`, len `n+1`). `ends = starts - min(ilen,0) + 1`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/variants/windows.rs`:

```rust
    #[test]
    fn test_fetch_windows() {
        use ndarray::Array1 as A1;
        // Single contig reference: bytes 0..20.
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        // 1 variant, contig 0, start=5, ilen=0 (SNP) → end = 5 - 0 + 1 = 6.
        // L=2 → read [start-L, end+L) = [3, 8) → bytes [3,4,5,6,7].
        let v_contigs = arr1(&[0i32]);
        let starts = arr1(&[5i32]);
        let ilens = arr1(&[0i32]);
        let (data, rw_off) = fetch_windows(
            v_contigs.view(),
            starts.view(),
            ilens.view(),
            2,
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        assert_eq!(data.to_vec(), vec![3u8, 4, 5, 6, 7]);
        assert_eq!(rw_off.to_vec(), vec![0i64, 5]);
    }

    #[test]
    fn test_fetch_windows_deletion_widens() {
        use ndarray::Array1 as A1;
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        // ilen=-2 (2bp deletion) → end = start - (-2) + 1 = start + 3.
        // start=5, L=1 → read [4, 9) → bytes [4,5,6,7,8] (len 5).
        let v_contigs = arr1(&[0i32]);
        let starts = arr1(&[5i32]);
        let ilens = arr1(&[-2i32]);
        let (data, rw_off) = fetch_windows(
            v_contigs.view(),
            starts.view(),
            ilens.view(),
            1,
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        assert_eq!(data.to_vec(), vec![4u8, 5, 6, 7, 8]);
        assert_eq!(rw_off.to_vec(), vec![0i64, 5]);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev cargo-test 2>&1 | rtk err`
Expected: FAIL — `cannot find function fetch_windows in this scope`.

- [ ] **Step 3: Implement `fetch_windows`**

Add to `src/variants/windows.rs` (above the `#[cfg(test)]` module). Note the `use` additions at the top of the file — change the import line to:

```rust
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
```

Then add:

```rust
/// Fetch the per-variant reference window `[start-L, end+L)` into one flat
/// buffer, with `ends = starts - min(ilen, 0) + 1`. Returns `(data, rw_off)`
/// where `rw_off` are per-variant byte boundaries (len `n+1`). Reuses
/// `reference::get_reference`'s padded core (absolute-coordinate OOB padding).
/// Mirrors `reference.fetch(v_contigs, starts-L, ends+L)`.
pub fn fetch_windows(
    v_contigs: ArrayView1<i32>,
    starts_v: ArrayView1<i32>,
    ilens_v: ArrayView1<i32>,
    flank_len: i64,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
) -> (Array1<u8>, Array1<i64>) {
    let n = starts_v.len();
    let mut regions = Array2::<i32>::zeros((n, 3));
    let mut rw_off = Array1::<i64>::zeros(n + 1);
    for i in 0..n {
        let start = starts_v[i] as i64;
        let ilen = ilens_v[i] as i64;
        let end = start - ilen.min(0) + 1;
        let rstart = start - flank_len;
        let rend = end + flank_len;
        regions[[i, 0]] = v_contigs[i];
        regions[[i, 1]] = rstart as i32;
        regions[[i, 2]] = rend as i32;
        rw_off[i + 1] = rw_off[i] + (rend - rstart);
    }
    let data = crate::reference::get_reference(
        regions.view(),
        rw_off.view(),
        reference,
        ref_offsets,
        pad_char,
        false, // serial: disjoint output already; this is per-variant fanout
    );
    (data, rw_off)
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev cargo-test 2>&1 | rtk err`
Expected: `windows::tests::test_fetch_windows` and `..._deletion_widens` PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/variants/windows.rs
rtk git commit -m "feat(variants): add fetch_windows reference-read helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Rust `assemble_variants_mode` orchestrator (byte alleles + flank_tokens)

**Files:**
- Modify: `src/variants/windows.rs`
- Test: cargo unit test inside `src/variants/windows.rs`

**Interfaces:**
- Consumes: `crate::variants::gather_alleles(v_idxs, allele_bytes, allele_offsets) -> (Array1<u8>, Array1<i64>)`; Task 1/2 cores.
- Produces:
  - `pub struct VariantBufs<Tok> { pub byte_bufs: Vec<(&'static str, Array1<u8>, Array1<i64>)>, pub tok_bufs: Vec<(&'static str, Array1<Tok>, Array1<i64>)> }`
  - `pub fn assemble_variants_mode<Tok: Copy>(...) -> VariantBufs<Tok>` (signature in Step 3)

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/variants/windows.rs`:

```rust
    #[test]
    fn test_assemble_variants_mode_alt_and_flank() {
        use ndarray::Array1 as A1;
        // Global alleles: v0="A"(65), v1="CG"(67,71). offsets [0,1,3].
        let alt_global = arr1(&[65u8, 67, 71]);
        let alt_off = arr1(&[0i64, 1, 3]);
        // Select v_idxs [1, 0] in one row.
        let v_idxs = arr1(&[1i32, 0]);
        let row_offsets = arr1(&[0i64, 2]);
        // Reference 0..20, single contig. v_starts/ilens are GLOBAL (indexed by v_idx).
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        let v_starts = arr1(&[5i32, 8]); // global per-variant
        let ilens = arr1(&[0i32, 0]);
        let v_contigs = arr1(&[0i32, 0]); // per-selected-variant contig
        // L=1, token LUT: identity-ish u8 (byte value -> itself for the test).
        let lut: A1<u8> = A1::from_vec((0u8..=255).collect());

        let bufs = assemble_variants_mode::<u8>(
            v_idxs.view(),
            row_offsets.view(),
            alt_global.view(),
            alt_off.view(),
            None, // no ref alleles
            None,
            true, // want_flank
            1,    // flank_len
            Some(lut.view()),
            v_contigs.view(),
            v_starts.view(),
            ilens.view(),
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        // byte_bufs: only "alt". v_idxs [1,0] → "CG" then "A" → [67,71,65], off [0,2,3].
        assert_eq!(bufs.byte_bufs.len(), 1);
        let (name, data, off) = &bufs.byte_bufs[0];
        assert_eq!(*name, "alt");
        assert_eq!(data.to_vec(), vec![67u8, 71, 65]);
        assert_eq!(off.to_vec(), vec![0i64, 2, 3]);
        // tok_bufs: only "flank_tokens". Each variant: [f5(1) | f3(1)] = 2 tokens.
        // var0 = v_idx 1: start=8, ilen=0 → end=9, read [7,10) = [7,8,9]; f5=[7], f3=[9].
        // var1 = v_idx 0: start=5, ilen=0 → end=6, read [4,7) = [4,5,6]; f5=[4], f3=[6].
        // tokens (identity lut) = [7,9, 4,6]; offsets = row_offsets [0,2].
        assert_eq!(bufs.tok_bufs.len(), 1);
        let (tname, tdata, toff) = &bufs.tok_bufs[0];
        assert_eq!(*tname, "flank_tokens");
        assert_eq!(tdata.to_vec(), vec![7u8, 9, 4, 6]);
        assert_eq!(toff.to_vec(), vec![0i64, 2]);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev cargo-test 2>&1 | rtk err`
Expected: FAIL — `cannot find function assemble_variants_mode` / `cannot find struct VariantBufs`.

- [ ] **Step 3: Implement the struct + orchestrator**

Add to `src/variants/windows.rs` (above the `#[cfg(test)]` module):

```rust
/// Assembled flat buffers returned by the mode orchestrators. `byte_bufs` carry
/// raw allele bytes (u8); `tok_bufs` carry LUT-applied tokens (`Tok`). Each
/// tuple is `(field_name, data, seq_offsets)`.
pub struct VariantBufs<Tok> {
    pub byte_bufs: Vec<(&'static str, Array1<u8>, Array1<i64>)>,
    pub tok_bufs: Vec<(&'static str, Array1<Tok>, Array1<i64>)>,
}

/// Gather per-selected-variant `start`/`ilen` from the GLOBAL arrays via `v_idxs`.
fn gather_starts_ilens(
    v_idxs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
) -> (Array1<i32>, Array1<i32>) {
    let n = v_idxs.len();
    let mut s = Array1::<i32>::zeros(n);
    let mut il = Array1::<i32>::zeros(n);
    for i in 0..n {
        let v = v_idxs[i] as usize;
        s[i] = v_starts[v];
        il[i] = ilens[v];
    }
    (s, il)
}

/// Plain-`variants` assembly tail: raw alt bytes (always), raw ref bytes
/// (optional), `flank_tokens` ride-along (optional). Mirrors the variants tail
/// of `get_variants_flat` (gather_alleles + compute_flank_tokens).
#[allow(clippy::too_many_arguments)]
pub fn assemble_variants_mode<Tok: Copy>(
    v_idxs: ArrayView1<i32>,
    row_offsets: ArrayView1<i64>,
    alt_global: ArrayView1<u8>,
    alt_off_global: ArrayView1<i64>,
    ref_global: Option<ArrayView1<u8>>,
    ref_off_global: Option<ArrayView1<i64>>,
    want_flank: bool,
    flank_len: i64,
    lut: Option<ArrayView1<Tok>>,
    v_contigs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
) -> VariantBufs<Tok> {
    let mut byte_bufs = Vec::new();
    let mut tok_bufs = Vec::new();

    let (alt_data, alt_seq_off) =
        crate::variants::gather_alleles(v_idxs, alt_global, alt_off_global);
    byte_bufs.push(("alt", alt_data, alt_seq_off));

    if let (Some(rg), Some(ro)) = (ref_global, ref_off_global) {
        let (ref_data, ref_seq_off) = crate::variants::gather_alleles(v_idxs, rg, ro);
        byte_bufs.push(("ref", ref_data, ref_seq_off));
    }

    if want_flank {
        let lut = lut.expect("flank tokens requested but no token LUT supplied");
        let (starts_v, ilens_v) = gather_starts_ilens(v_idxs, v_starts, ilens);
        let (rw_data, rw_off) = fetch_windows(
            v_contigs, starts_v.view(), ilens_v.view(), flank_len, reference, ref_offsets,
            pad_char,
        );
        let l = flank_len as usize;
        let (f5, f3) = slice_flanks(rw_data.view(), rw_off.view(), l);
        // Concatenate [f5 | f3] per variant (2L tokens, variant-major), tokenize.
        let n = f5.len() / l;
        let mut flank_bytes: Vec<u8> = Vec::with_capacity(n * 2 * l);
        for i in 0..n {
            for k in 0..l {
                flank_bytes.push(f5[i * l + k]);
            }
            for k in 0..l {
                flank_bytes.push(f3[i * l + k]);
            }
        }
        let fb = Array1::from_vec(flank_bytes);
        let tok = tokenize(fb.view(), lut);
        // flank_tokens offsets are the variant-level row_offsets (fixed 2L inner
        // axis carried separately Python-side as a trailing regular dim).
        tok_bufs.push(("flank_tokens", tok, row_offsets.to_owned()));
    }

    VariantBufs { byte_bufs, tok_bufs }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev cargo-test 2>&1 | rtk err`
Expected: `test_assemble_variants_mode_alt_and_flank` PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/variants/windows.rs
rtk git commit -m "feat(variants): assemble_variants_mode (alt/ref bytes + flank tokens)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Rust `assemble_windows_mode` orchestrator (token windows)

**Files:**
- Modify: `src/variants/windows.rs`
- Test: cargo unit test inside `src/variants/windows.rs`

**Interfaces:**
- Consumes: Task 1/2/3 cores + `gather_alleles`.
- Produces: `pub fn assemble_windows_mode<Tok: Copy>(...) -> VariantBufs<Tok>` (signature in Step 3). `ref_mode`/`alt_mode`: `1` = window (flanked, tokenized), `2` = allele (bare tokenized). Field names: `ref_window`/`alt_window` for mode 1, `ref`/`alt` for mode 2.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/variants/windows.rs`:

```rust
    #[test]
    fn test_assemble_windows_mode_both_windows() {
        use ndarray::Array1 as A1;
        // Global alt alleles: v0="A"(65). offsets [0,1].
        let alt_global = arr1(&[65u8]);
        let alt_off = arr1(&[0i64, 1]);
        let v_idxs = arr1(&[0i32]);
        let row_offsets = arr1(&[0i64, 1]);
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        let v_starts = arr1(&[5i32]);
        let ilens = arr1(&[0i32]);
        let v_contigs = arr1(&[0i32]);
        let lut: A1<u8> = A1::from_vec((0u8..=255).collect()); // identity

        let bufs = assemble_windows_mode::<u8>(
            v_idxs.view(),
            row_offsets.view(),
            1, // ref_mode = window
            1, // alt_mode = window
            alt_global.view(),
            alt_off.view(),
            None,
            None,
            1, // flank_len
            lut.view(),
            v_contigs.view(),
            v_starts.view(),
            ilens.view(),
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        // SNP start=5 ilen=0 → end=6; read [4,7) = [4,5,6]. L=1.
        // ref_window tokens (identity) = [4,5,6], off [0,3].
        // alt_window = f5[4] . alt[65] . f3[6] = [4,65,6], off [0,3].
        assert_eq!(bufs.byte_bufs.len(), 0);
        let names: Vec<&str> = bufs.tok_bufs.iter().map(|t| t.0).collect();
        assert_eq!(names, vec!["ref_window", "alt_window"]);
        assert_eq!(bufs.tok_bufs[0].1.to_vec(), vec![4u8, 5, 6]);
        assert_eq!(bufs.tok_bufs[0].2.to_vec(), vec![0i64, 3]);
        assert_eq!(bufs.tok_bufs[1].1.to_vec(), vec![4u8, 65, 6]);
        assert_eq!(bufs.tok_bufs[1].2.to_vec(), vec![0i64, 3]);
    }

    #[test]
    fn test_assemble_windows_mode_bare_alleles() {
        use ndarray::Array1 as A1;
        // alt v0="AC"(65,67); ref v0="G"(71).
        let alt_global = arr1(&[65u8, 67]);
        let alt_off = arr1(&[0i64, 2]);
        let ref_global = arr1(&[71u8]);
        let ref_off = arr1(&[0i64, 1]);
        let v_idxs = arr1(&[0i32]);
        let row_offsets = arr1(&[0i64, 1]);
        let reference: A1<u8> = A1::from_vec((0u8..20).collect());
        let ref_offsets = arr1(&[0i64, 20]);
        let v_starts = arr1(&[5i32]);
        let ilens = arr1(&[0i32]);
        let v_contigs = arr1(&[0i32]);
        let lut: A1<u8> = A1::from_vec((0u8..=255).collect());

        let bufs = assemble_windows_mode::<u8>(
            v_idxs.view(),
            row_offsets.view(),
            2, // ref_mode = allele (bare)
            2, // alt_mode = allele (bare)
            alt_global.view(),
            alt_off.view(),
            Some(ref_global.view()),
            Some(ref_off.view()),
            1,
            lut.view(),
            v_contigs.view(),
            v_starts.view(),
            ilens.view(),
            reference.view(),
            ref_offsets.view(),
            b'N',
        );
        let names: Vec<&str> = bufs.tok_bufs.iter().map(|t| t.0).collect();
        assert_eq!(names, vec!["ref", "alt"]);
        // bare ref tokens = [71], off [0,1]; bare alt tokens = [65,67], off [0,2].
        assert_eq!(bufs.tok_bufs[0].1.to_vec(), vec![71u8]);
        assert_eq!(bufs.tok_bufs[0].2.to_vec(), vec![0i64, 1]);
        assert_eq!(bufs.tok_bufs[1].1.to_vec(), vec![65u8, 67]);
        assert_eq!(bufs.tok_bufs[1].2.to_vec(), vec![0i64, 2]);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev cargo-test 2>&1 | rtk err`
Expected: FAIL — `cannot find function assemble_windows_mode`.

- [ ] **Step 3: Implement `assemble_windows_mode`**

Add to `src/variants/windows.rs` (above the `#[cfg(test)]` module):

```rust
/// `variant-windows` assembly tail. `ref_mode`/`alt_mode`: 1 = flanked window
/// (`[start-L,end+L)` for ref; `flank5.alt.flank3` for alt), 2 = bare tokenized
/// allele. Produces only token buffers (scalar fields are handled Python-side).
/// Mirrors the windows branch of `get_variants_flat` (incl. the single fused
/// fetch shared by ref_window + alt_window).
#[allow(clippy::too_many_arguments)]
pub fn assemble_windows_mode<Tok: Copy>(
    v_idxs: ArrayView1<i32>,
    _row_offsets: ArrayView1<i64>,
    ref_mode: i64,
    alt_mode: i64,
    alt_global: ArrayView1<u8>,
    alt_off_global: ArrayView1<i64>,
    ref_global: Option<ArrayView1<u8>>,
    ref_off_global: Option<ArrayView1<i64>>,
    flank_len: i64,
    lut: ArrayView1<Tok>,
    v_contigs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
) -> VariantBufs<Tok> {
    let mut tok_bufs = Vec::new();
    let l = flank_len as usize;

    // alt alleles are always gathered (needed for alt window or bare alt).
    let (alt_data, alt_seq_off) =
        crate::variants::gather_alleles(v_idxs, alt_global, alt_off_global);

    // One fused fetch if either side needs a window read.
    let need_fetch = ref_mode == 1 || alt_mode == 1;
    let fetched = if need_fetch {
        let (starts_v, ilens_v) = gather_starts_ilens(v_idxs, v_starts, ilens);
        Some(fetch_windows(
            v_contigs, starts_v.view(), ilens_v.view(), flank_len, reference, ref_offsets,
            pad_char,
        ))
    } else {
        None
    };

    // ref side (ordered first to match Python field insertion order).
    if ref_mode == 1 {
        let (rw_data, rw_off) = fetched.as_ref().expect("ref window needs a fetch");
        let tok = tokenize(rw_data.view(), lut);
        tok_bufs.push(("ref_window", tok, rw_off.clone()));
    } else if ref_mode == 2 {
        let rg = ref_global.expect("bare ref allele needs ref byte buffer");
        let ro = ref_off_global.expect("bare ref allele needs ref offsets");
        let (ref_data, ref_seq_off) = crate::variants::gather_alleles(v_idxs, rg, ro);
        let tok = tokenize(ref_data.view(), lut);
        tok_bufs.push(("ref", tok, ref_seq_off));
    }

    // alt side.
    if alt_mode == 1 {
        let (rw_data, rw_off) = fetched.as_ref().expect("alt window needs a fetch");
        let (f5, f3) = slice_flanks(rw_data.view(), rw_off.view(), l);
        let (alt_bytes, alt_off) = assemble_alt_window(
            f5.view(),
            f3.view(),
            alt_data.view(),
            alt_seq_off.view(),
            l,
        );
        let tok = tokenize(alt_bytes.view(), lut);
        tok_bufs.push(("alt_window", tok, alt_off));
    } else if alt_mode == 2 {
        let tok = tokenize(alt_data.view(), lut);
        tok_bufs.push(("alt", tok, alt_seq_off));
    }

    VariantBufs { byte_bufs: Vec::new(), tok_bufs }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev cargo-test 2>&1 | rtk err`
Expected: both `test_assemble_windows_mode_*` PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/variants/windows.rs
rtk git commit -m "feat(variants): assemble_windows_mode (token windows + bare alleles)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: FFI pyfunctions + registration

**Files:**
- Modify: `src/ffi/mod.rs`
- Modify: `src/lib.rs:36` (after the last `add_function` for variants)
- Test: Python smoke import (Step 5)

**Interfaces:**
- Produces two Python-callable functions, importable as
  `from genvarloader.genvarloader import assemble_variant_buffers_u8, assemble_variant_buffers_i32`.
- Signature (identical for both; the suffix names the token dtype `Tok`):
  ```
  assemble_variant_buffers_<tok>(
      mode: int,                # 0 = variants, 1 = windows
      v_idxs: i32[n],
      row_offsets: i64[b*p+1],
      alt_global: u8[],
      alt_off_global: i64[],
      ref_global: Optional[u8[]],
      ref_off_global: Optional[i64[]],
      want_ref_bytes: bool,     # variants mode: emit raw "ref" bytes
      want_flank: bool,         # variants mode: emit "flank_tokens"
      ref_mode: int,            # windows mode: 1 window / 2 allele
      alt_mode: int,            # windows mode: 1 window / 2 allele
      flank_len: int,
      lut: Optional[<tok>[256]],
      v_contigs: i32[n],
      v_starts: i32[],          # global per-variant
      ilens: i32[],             # global per-variant
      reference: u8[],
      ref_offsets: i64[],       # contig offsets
      pad_char: int,
  ) -> dict[str, tuple[np.ndarray, np.ndarray]]   # name -> (data, seq_offsets)
  ```

- [ ] **Step 1: Add the shared dict-builder + two pyfunctions**

Add to the top imports of `src/ffi/mod.rs` (extend the existing `use` lines):

```rust
use numpy::PyArrayMethods;
use pyo3::types::PyDict;
use crate::variants::windows::{assemble_variants_mode, assemble_windows_mode, VariantBufs};
```

Add these functions to `src/ffi/mod.rs` (near the other variants pyfunctions):

```rust
/// Build the `{name: (data, seq_offsets)}` dict from assembled buffers.
fn bufs_to_pydict<'py, Tok: numpy::Element + Copy>(
    py: Python<'py>,
    bufs: VariantBufs<Tok>,
) -> Bound<'py, PyDict> {
    let d = PyDict::new(py);
    for (name, data, off) in bufs.byte_bufs {
        d.set_item(name, (data.into_pyarray(py), off.into_pyarray(py)))
            .unwrap();
    }
    for (name, data, off) in bufs.tok_bufs {
        d.set_item(name, (data.into_pyarray(py), off.into_pyarray(py)))
            .unwrap();
    }
    d
}

/// Monomorphized assembly entry. `Tok` is the token dtype; `mode` selects
/// variants (0) vs windows (1). See module docs in `variants::windows`.
#[allow(clippy::too_many_arguments)]
fn assemble_variant_buffers_impl<'py, Tok: numpy::Element + Copy>(
    py: Python<'py>,
    mode: i64,
    v_idxs: PyReadonlyArray1<i32>,
    row_offsets: PyReadonlyArray1<i64>,
    alt_global: PyReadonlyArray1<u8>,
    alt_off_global: PyReadonlyArray1<i64>,
    ref_global: Option<PyReadonlyArray1<u8>>,
    ref_off_global: Option<PyReadonlyArray1<i64>>,
    want_ref_bytes: bool,
    want_flank: bool,
    ref_mode: i64,
    alt_mode: i64,
    flank_len: i64,
    lut: Option<PyReadonlyArray1<Tok>>,
    v_contigs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    reference: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
) -> Bound<'py, PyDict> {
    let rg = ref_global.as_ref().map(|a| a.as_array());
    let ro = ref_off_global.as_ref().map(|a| a.as_array());
    let lut_v = lut.as_ref().map(|a| a.as_array());
    let bufs = if mode == 0 {
        assemble_variants_mode::<Tok>(
            v_idxs.as_array(),
            row_offsets.as_array(),
            alt_global.as_array(),
            alt_off_global.as_array(),
            if want_ref_bytes { rg } else { None },
            if want_ref_bytes { ro } else { None },
            want_flank,
            flank_len,
            lut_v,
            v_contigs.as_array(),
            v_starts.as_array(),
            ilens.as_array(),
            reference.as_array(),
            ref_offsets.as_array(),
            pad_char,
        )
    } else {
        assemble_windows_mode::<Tok>(
            v_idxs.as_array(),
            row_offsets.as_array(),
            ref_mode,
            alt_mode,
            alt_global.as_array(),
            alt_off_global.as_array(),
            rg,
            ro,
            flank_len,
            lut_v.expect("windows mode requires a token LUT"),
            v_contigs.as_array(),
            v_starts.as_array(),
            ilens.as_array(),
            reference.as_array(),
            ref_offsets.as_array(),
            pad_char,
        )
    };
    bufs_to_pydict(py, bufs)
}

/// u8-token assembly (token_dtype == uint8). See `assemble_variant_buffers_impl`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn assemble_variant_buffers_u8<'py>(
    py: Python<'py>,
    mode: i64,
    v_idxs: PyReadonlyArray1<i32>,
    row_offsets: PyReadonlyArray1<i64>,
    alt_global: PyReadonlyArray1<u8>,
    alt_off_global: PyReadonlyArray1<i64>,
    ref_global: Option<PyReadonlyArray1<u8>>,
    ref_off_global: Option<PyReadonlyArray1<i64>>,
    want_ref_bytes: bool,
    want_flank: bool,
    ref_mode: i64,
    alt_mode: i64,
    flank_len: i64,
    lut: Option<PyReadonlyArray1<u8>>,
    v_contigs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    reference: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
) -> Bound<'py, PyDict> {
    assemble_variant_buffers_impl::<u8>(
        py, mode, v_idxs, row_offsets, alt_global, alt_off_global, ref_global,
        ref_off_global, want_ref_bytes, want_flank, ref_mode, alt_mode, flank_len,
        lut, v_contigs, v_starts, ilens, reference, ref_offsets, pad_char,
    )
}

/// i32-token assembly (token_dtype == int32). See `assemble_variant_buffers_impl`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn assemble_variant_buffers_i32<'py>(
    py: Python<'py>,
    mode: i64,
    v_idxs: PyReadonlyArray1<i32>,
    row_offsets: PyReadonlyArray1<i64>,
    alt_global: PyReadonlyArray1<u8>,
    alt_off_global: PyReadonlyArray1<i64>,
    ref_global: Option<PyReadonlyArray1<u8>>,
    ref_off_global: Option<PyReadonlyArray1<i64>>,
    want_ref_bytes: bool,
    want_flank: bool,
    ref_mode: i64,
    alt_mode: i64,
    flank_len: i64,
    lut: Option<PyReadonlyArray1<i32>>,
    v_contigs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    reference: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
) -> Bound<'py, PyDict> {
    assemble_variant_buffers_impl::<i32>(
        py, mode, v_idxs, row_offsets, alt_global, alt_off_global, ref_global,
        ref_off_global, want_ref_bytes, want_flank, ref_mode, alt_mode, flank_len,
        lut, v_contigs, v_starts, ilens, reference, ref_offsets, pad_char,
    )
}
```

- [ ] **Step 2: Register both in `src/lib.rs`**

After the line `m.add_function(wrap_pyfunction!(ffi::fill_empty_seq_i32, m)?)?;` (currently `src/lib.rs:35`), add:

```rust
    m.add_function(wrap_pyfunction!(ffi::assemble_variant_buffers_u8, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::assemble_variant_buffers_i32, m)?)?;
```

- [ ] **Step 3: Build the extension**

Run: `pixi run -e dev maturin develop --release 2>&1 | rtk err`
Expected: builds clean (no errors). Warnings about `too_many_arguments` are suppressed by the `allow` attributes.

- [ ] **Step 4: Run the Rust unit tests again (regression)**

Run: `pixi run -e dev cargo-test 2>&1 | rtk err`
Expected: all `windows::tests::*` plus existing tests PASS.

- [ ] **Step 5: Smoke-test the import**

Run:
```bash
pixi run -e dev python -c "from genvarloader.genvarloader import assemble_variant_buffers_u8, assemble_variant_buffers_i32; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
rtk git add src/ffi/mod.rs src/lib.rs
rtk git commit -m "feat(ffi): assemble_variant_buffers_{u8,i32} pyfunctions

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Python numba oracle + dispatch registration + dict parity harness

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_flanks.py`
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (imports + register block)
- Modify: `tests/parity/_harness.py`
- Test: `tests/parity/test_assemble_variant_buffers_parity.py` (created in Task 8; harness verified here via a tiny inline check)

**Interfaces:**
- Produces:
  - `_flat_flanks._assemble_variant_buffers_numba(mode, v_idxs, row_offsets, alt_global, alt_off_global, ref_global, ref_off_global, want_ref_bytes, want_flank, ref_mode, alt_mode, flank_len, lut, v_contigs, v_starts, ilens, reference, ref_offsets, pad_char) -> dict[str, tuple[np.ndarray, np.ndarray]]` — same contract as the Rust pyfunctions, composed from the existing helpers.
  - `_flat_variants._assemble_variant_buffers_rust(...same args...)` — the dtype-selecting shim.
  - dispatch key `"assemble_variant_buffers"` (default `"rust"`).
  - `tests.parity._harness.assert_kernel_parity_dict(name, *inputs)`.

- [ ] **Step 1: Write the numba oracle composing existing helpers**

Add to `python/genvarloader/_dataset/_flat_flanks.py` (after the existing imports and `from ._flat_variants import _FlatWindow`):

```python
from ._flat_variants import _gather_alleles  # noqa: E402  (numba/rust dispatch gather)


def _assemble_variant_buffers_numba(
    mode,
    v_idxs,
    row_offsets,
    alt_global,
    alt_off_global,
    ref_global,
    ref_off_global,
    want_ref_bytes,
    want_flank,
    ref_mode,
    alt_mode,
    flank_len,
    lut,
    v_contigs,
    v_starts,
    ilens,
    reference,
    ref_offsets,
    pad_char,
):
    """Parity oracle: compose the existing numpy/numba assembly helpers into the
    same ``{name: (data, seq_offsets)}`` dict the Rust mega-call returns.

    ``reference``/``ref_offsets``/``pad_char`` are the raw reference-genome
    arrays; this oracle wraps them in a lightweight fetch shim so it can reuse
    ``compute_*`` unchanged."""
    from numpy.typing import NDArray  # noqa: F401

    out: dict = {}
    v_idxs = np.ascontiguousarray(v_idxs, np.int32)
    row_offsets = np.ascontiguousarray(row_offsets, np.int64)

    # per-selected-variant start/ilen (global arrays indexed by v_idxs)
    starts_v = np.asarray(v_starts, np.int32)[v_idxs]
    ilens_v = np.asarray(ilens, np.int32)[v_idxs]
    v_contigs = np.ascontiguousarray(v_contigs, np.int32)

    class _RefShim:
        """Minimal reference.fetch() over raw arrays, matching Reference.fetch."""

        def fetch(self, contigs, starts, ends):
            from .._ragged import Ragged
            from ..genvarloader import get_reference

            lengths = np.asarray(ends) - np.asarray(starts)
            from .._utils import lengths_to_offsets

            offs = lengths_to_offsets(lengths)
            regions = np.stack(
                [
                    np.asarray(contigs, np.int32),
                    np.asarray(starts, np.int32),
                    np.asarray(ends, np.int32),
                ],
                axis=1,
            )
            seqs = get_reference(
                regions,
                offs,
                np.asarray(reference, np.uint8),
                np.asarray(ref_offsets, np.int64),
                int(pad_char),
                False,
            )
            return Ragged.from_offsets(seqs.view("S1"), (len(contigs), None), offs)

    ref_shim = _RefShim()
    lut_arr = None if lut is None else np.asarray(lut)

    if mode == 0:
        alt_data, alt_seq_off = _gather_alleles(v_idxs, alt_global, alt_off_global)
        out["alt"] = (np.ascontiguousarray(alt_data, np.uint8), alt_seq_off)
        if want_ref_bytes:
            ref_data, ref_seq_off = _gather_alleles(v_idxs, ref_global, ref_off_global)
            out["ref"] = (np.ascontiguousarray(ref_data, np.uint8), ref_seq_off)
        if want_flank:
            tok, off = compute_flank_tokens(
                ref_shim, v_contigs, starts_v, ilens_v, flank_len, lut_arr, row_offsets
            )
            out["flank_tokens"] = (tok, np.asarray(off, np.int64))
    else:
        alt_data, alt_seq_off = _gather_alleles(v_idxs, alt_global, alt_off_global)
        if ref_mode == 1:
            rw = compute_ref_window(
                ref_shim, v_contigs, starts_v, ilens_v, flank_len, lut_arr, row_offsets
            )
            out["ref_window"] = (rw.data, rw.seq_offsets)
        elif ref_mode == 2:
            ref_data, ref_seq_off = _gather_alleles(v_idxs, ref_global, ref_off_global)
            rw = tokenize_alleles(ref_data, ref_seq_off, lut_arr, row_offsets)
            out["ref"] = (rw.data, rw.seq_offsets)
        if alt_mode == 1:
            aw = compute_alt_window(
                ref_shim, v_contigs, starts_v, ilens_v, alt_data, alt_seq_off,
                flank_len, lut_arr, row_offsets,
            )
            out["alt_window"] = (aw.data, aw.seq_offsets)
        elif alt_mode == 2:
            aw = tokenize_alleles(alt_data, alt_seq_off, lut_arr, row_offsets)
            out["alt"] = (aw.data, aw.seq_offsets)
    return out
```

> Note: confirm the import paths `from .._ragged import Ragged`, `from .._utils import lengths_to_offsets`, and `from ..genvarloader import get_reference` resolve in this package (grep them: `rtk grep "def lengths_to_offsets" python/genvarloader/_utils.py` and `rtk grep "get_reference" python/genvarloader/__init__.py` / the compiled module). If `get_reference` is not yet exported from the Python package, import it from `..genvarloader` (the compiled extension) — it is already used by `_reference.py:143`, so mirror that exact import.

- [ ] **Step 2: Add the Rust dtype-selecting shim + register the kernel**

In `python/genvarloader/_dataset/_flat_variants.py`, add to the rust imports block (near the other `from ..genvarloader import ... as ..._rust`):

```python
from ..genvarloader import assemble_variant_buffers_i32 as _assemble_i32_rust
from ..genvarloader import assemble_variant_buffers_u8 as _assemble_u8_rust
```

Then add the shim + registration (place it after the existing `register(...)` blocks, e.g. after the `fill_empty_seq` registrations):

```python
def _assemble_variant_buffers_rust(
    mode,
    v_idxs,
    row_offsets,
    alt_global,
    alt_off_global,
    ref_global,
    ref_off_global,
    want_ref_bytes,
    want_flank,
    ref_mode,
    alt_mode,
    flank_len,
    lut,
    v_contigs,
    v_starts,
    ilens,
    reference,
    ref_offsets,
    pad_char,
):
    """Select the u8/i32 monomorphization by token dtype. ``lut`` is None only
    when no tokenized output is requested (plain variants, no flank); then the
    u8 entry is used and ``lut`` stays None."""
    fn = _assemble_u8_rust
    if lut is not None and np.asarray(lut).dtype == np.int32:
        fn = _assemble_i32_rust
    return fn(
        int(mode),
        np.ascontiguousarray(v_idxs, np.int32),
        np.ascontiguousarray(row_offsets, np.int64),
        np.ascontiguousarray(alt_global, np.uint8),
        np.ascontiguousarray(alt_off_global, np.int64),
        None if ref_global is None else np.ascontiguousarray(ref_global, np.uint8),
        None if ref_off_global is None else np.ascontiguousarray(ref_off_global, np.int64),
        bool(want_ref_bytes),
        bool(want_flank),
        int(ref_mode),
        int(alt_mode),
        int(flank_len),
        None if lut is None else np.ascontiguousarray(lut),
        np.ascontiguousarray(v_contigs, np.int32),
        np.ascontiguousarray(v_starts, np.int32),
        np.ascontiguousarray(ilens, np.int32),
        np.ascontiguousarray(reference, np.uint8),
        np.ascontiguousarray(ref_offsets, np.int64),
        int(pad_char),
    )


def _assemble_variant_buffers_numba_entry(*args):
    from ._flat_flanks import _assemble_variant_buffers_numba

    return _assemble_variant_buffers_numba(*args)


register(
    "assemble_variant_buffers",
    numba=_assemble_variant_buffers_numba_entry,
    rust=_assemble_variant_buffers_rust,
    default="rust",
)
```

> The numba entry is a thin lazy wrapper to avoid a circular import (`_flat_flanks` imports from `_flat_variants`).

- [ ] **Step 3: Add the dict parity assertion to the harness**

Add to `tests/parity/_harness.py`:

```python
def assert_kernel_parity_dict(name: str, *inputs) -> None:
    """Parity for kernels that RETURN a dict[str, tuple[ndarray, ...]].

    Asserts identical key sets and byte-identical values per key (dtype, shape,
    values) between the numba and rust backends.
    """
    numba_fn, rust_fn = _dispatch.backends(name)
    got_numba = numba_fn(*inputs)
    got_rust = rust_fn(*inputs)
    assert set(got_numba) == set(got_rust), (
        f"{name}: keys {sorted(got_numba)} != {sorted(got_rust)}"
    )
    for key in got_numba:
        nt = got_numba[key]
        rt = got_rust[key]
        assert len(nt) == len(rt), f"{name}[{key}]: tuple len {len(nt)} != {len(rt)}"
        for i, (a, b) in enumerate(zip(nt, rt)):
            a = np.asarray(a)
            b = np.asarray(b)
            assert a.dtype == b.dtype, f"{name}[{key}][{i}]: dtype {a.dtype} != {b.dtype}"
            assert a.shape == b.shape, f"{name}[{key}][{i}]: shape {a.shape} != {b.shape}"
            np.testing.assert_array_equal(a, b)
```

- [ ] **Step 4: Build + verify the registration imports cleanly**

Run:
```bash
pixi run -e dev maturin develop --release 2>&1 | rtk err
pixi run -e dev python -c "import genvarloader._dataset._flat_variants as m; from genvarloader._dispatch import backends; print(backends('assemble_variant_buffers'))"
```
Expected: prints the `(numba_entry, rust_shim)` callables tuple — confirms the key registered.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_flanks.py python/genvarloader/_dataset/_flat_variants.py tests/parity/_harness.py
rtk git commit -m "feat(variants): register assemble_variant_buffers (rust default, numba oracle)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Rewrite `get_variants_flat` assembly tail to call the dispatched kernel

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py:974-1083` (the windows branch + flank ride-along + the alt/ref allele gather in the scalar-field block)
- Test: covered by Task 8 parity + the existing `tests/parity/test_variants_dataset_parity.py`

**Interfaces:**
- Consumes: `get("assemble_variant_buffers")(...)` from Task 6 returning `dict[str, (data, seq_off)]`.
- Produces: unchanged public return types `_FlatVariants` / `_FlatVariantWindows` (callers see no change).

- [ ] **Step 1: Replace the alt/ref allele gather + windows branch + flank ride-along**

In `get_variants_flat`, the current flow gathers `alt` (and optional `ref`) alleles inline (lines ~927-942), then later builds windows (lines ~974-1055) and the flank ride-along (lines ~1057-1077). Replace those three regions so the **ragged** buffers come from one dispatched call, while **scalar** fields stay inline.

Concretely, after the scalar/dosage/custom fields are built into `fields` (keep all of that), compute the shared inputs and call the kernel:

```python
    from .._haps import _HapsFfiStatic  # noqa: F401  (type only)

    stat = haps.ffi_static
    # v_contigs: per-selected-variant contig id (only needed when fetching).
    needs_fetch = (
        regions is not None
        and haps.token_lut is not None
        and (
            (issubclass(haps.kind, _FlatVariantWindows) and opt is not None)
            or bool(haps.flank_length)
        )
    )
    if needs_fetch:
        regions_arr = np.asarray(regions)
        group_contigs = np.repeat(regions_arr[:, 0], eff_ploidy)
        v_contigs = np.repeat(group_contigs, np.diff(row_offsets)).astype(np.int32)
    else:
        v_contigs = np.zeros(len(v_idxs), np.int32)

    ref_present = "ref" in haps.var_fields and haps.variants.ref is not None
    ref_global = ref_off_global = None
    if ref_present or (
        issubclass(haps.kind, _FlatVariantWindows)
        and opt is not None
        and (opt.ref == "allele")
    ):
        ref_global = np.asarray(haps.variants.ref.data).view(np.uint8)
        ref_off_global = np.asarray(haps.variants.ref.offsets, np.int64)
```

- [ ] **Step 2: Build the windows-mode result from the dict**

Replace the windows branch (`if regions is not None and issubclass(haps.kind, _FlatVariantWindows) and opt is not None:` ... `return win`) with:

```python
    opt = haps.window_opt
    if (
        regions is not None
        and issubclass(haps.kind, _FlatVariantWindows)
        and opt is not None
    ):
        L = opt.flank_length
        ref_mode = 1 if opt.ref == "window" else 2
        alt_mode = 1 if opt.alt == "window" else 2
        bufs = get("assemble_variant_buffers")(
            1,  # windows mode
            v_idxs,
            row_offsets,
            stat.alt_alleles,
            stat.alt_offsets,
            ref_global,
            ref_off_global,
            False,  # want_ref_bytes (windows mode emits tokens, not raw bytes)
            False,  # want_flank
            ref_mode,
            alt_mode,
            L,
            haps.token_lut,
            v_contigs,
            stat.v_starts,
            stat.ilens,
            stat.ref,        # reference genome buffer
            stat.ref_offsets,  # contig offsets
            haps.reference.pad_char,
        )
        wshape = (b, eff_ploidy, None, None)
        wfields = {k: v for k, v in fields.items() if k not in ("alt", "ref")}
        win = _FlatVariantWindows(wfields)
        for name, (data, seq_off) in bufs.items():
            fw = _FlatWindow(data, np.asarray(seq_off, np.int64), row_offsets, wshape)
            setattr(win, name, fw)
        if haps.dummy_variant is not None:
            win = win.fill_empty_groups(
                haps.dummy_variant, unk=haps.unknown_token, flank_length=L
            )
        return win
```

- [ ] **Step 3: Build the plain-variants alt/ref + flank result from the dict**

Replace the inline alt/ref allele gather and the flank ride-along so the plain-variants path also goes through the kernel. Where the code currently does `fields["alt"] = _FlatAlleles(...)` and `fields["ref"] = _FlatAlleles(...)`, and the later `if haps.flank_length and ...: compute_flank_tokens(...)` block, replace with a single call after the scalar fields are assembled:

```python
    want_flank = bool(
        haps.flank_length and haps.token_lut is not None and regions is not None
    )
    L = haps.flank_length or 0
    bufs = get("assemble_variant_buffers")(
        0,  # variants mode
        v_idxs,
        row_offsets,
        stat.alt_alleles,
        stat.alt_offsets,
        ref_global,
        ref_off_global,
        ref_present,  # want_ref_bytes
        want_flank,
        0,  # ref_mode (unused in variants mode)
        0,  # alt_mode (unused)
        L,
        haps.token_lut,
        v_contigs,
        stat.v_starts,
        stat.ilens,
        stat.ref if stat.ref is not None else np.zeros(0, np.uint8),
        stat.ref_offsets if stat.ref_offsets is not None else np.zeros(1, np.int64),
        haps.reference.pad_char if haps.reference is not None else 0,
    )
    alt_data, alt_seq_off = bufs["alt"]
    fields["alt"] = _FlatAlleles(
        np.asarray(alt_data, np.uint8), np.asarray(alt_seq_off, np.int64), row_offsets, shape
    )
    if "ref" in bufs:
        ref_data, ref_seq_off = bufs["ref"]
        fields["ref"] = _FlatAlleles(
            np.asarray(ref_data, np.uint8), np.asarray(ref_seq_off, np.int64), row_offsets, shape
        )
    flat = _FlatVariants(fields)
    if "flank_tokens" in bufs:
        from .._flat import _Flat

        tok, off = bufs["flank_tokens"]
        flat.flank_tokens = _Flat.from_offsets(
            tok, (b, eff_ploidy, None, 2 * L), np.asarray(off, np.int64)
        )

    if haps.dummy_variant is not None:
        flat = flat.fill_empty_groups(haps.dummy_variant, unk=haps.unknown_token)

    return flat
```

> IMPORTANT ordering: the `fields` dict insertion order determines downstream wrapping; today `alt` is inserted before `start`/`ref`/etc. Preserve the existing field order — build `fields["alt"]` placeholder position by keeping the scalar block as-is and only swapping the alt/ref *values* to come from `bufs`. If the original code inserted `alt` first, keep `alt` first (move the `bufs["alt"]` assignment up to where `fields["alt"]` was originally set, not appended at the end). Verify with `RaggedVariants` field order in a parity run (Task 8).

- [ ] **Step 4: Remove the now-dead inline assembly**

Delete the now-unreachable inline `compute_windows`/`compute_ref_window`/`compute_alt_window`/`tokenize_alleles`/`compute_flank_tokens` call sites in `get_variants_flat` (the helper *functions* stay in `_flat_flanks.py` as the oracle). Confirm no other caller depends on them on the hot path: `rtk grep "compute_windows\|compute_ref_window\|compute_alt_window\|compute_flank_tokens\|tokenize_alleles" python/genvarloader/_dataset/_flat_variants.py` should now only show imports used by the oracle, not the hot path.

- [ ] **Step 5: Build + smoke-run one windows query**

Run:
```bash
pixi run -e dev maturin develop --release 2>&1 | rtk err
pixi run -e dev pytest tests/parity/test_variants_dataset_parity.py -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | rtk err
```
Expected: existing variants dataset parity PASSES on the default (rust) backend.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py
rtk git commit -m "perf(variants): route windows/variants assembly through one rust call

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Parity fixtures + dataset backstop spy + both-backend gate

**Files:**
- Create: `tests/parity/test_assemble_variant_buffers_parity.py`
- Modify: `tests/parity/test_dataset_parity.py` (add a kernel-spy that proves the call runs on the live windows/variants `__getitem__` path)

**Interfaces:**
- Consumes: `assert_kernel_parity_dict` (Task 6), the registered `assemble_variant_buffers` kernel.

- [ ] **Step 1: Write the kernel-level mode-matrix parity test**

Create `tests/parity/test_assemble_variant_buffers_parity.py`:

```python
"""Parity: the new assemble_variant_buffers mega-call (rust) must be
byte-identical to the composed numba oracle for variants + variant-windows,
across the ref/alt mode matrix, the flank ride-along, and empty selections."""

import numpy as np
import pytest

import genvarloader._dataset._flat_variants  # noqa: F401  (triggers register())
from tests.parity._harness import assert_kernel_parity_dict

pytestmark = pytest.mark.parity


def _reference():
    # single contig of 40 bytes, ASCII A/C/G/T cycling.
    bases = np.frombuffer(b"ACGT", np.uint8)
    ref = np.tile(bases, 10).astype(np.uint8)
    ref_offsets = np.array([0, ref.size], np.int64)
    return ref, ref_offsets


def _lut(dtype):
    # A->0 C->1 G->2 T->3, everything else (incl. N) -> 4 (unknown).
    lut = np.full(256, 4, dtype)
    for i, b in enumerate(b"ACGT"):
        lut[b] = i
    return lut


def _globals():
    # 3 global variants: alt "A","CG","T"; ref "C","G","AA".
    alt = np.frombuffer(b"ACGT", np.uint8)  # placeholder; rebuild explicitly below
    alt_bytes = np.frombuffer(b"ACGT", np.uint8)
    # alt alleles: v0="A", v1="CG", v2="T"
    alt_data = np.frombuffer(b"ACGT", np.uint8)
    alt_data = np.frombuffer(b"A" b"CG" b"T", np.uint8)
    alt_off = np.array([0, 1, 3, 4], np.int64)
    ref_data = np.frombuffer(b"C" b"G" b"AA", np.uint8)
    ref_off = np.array([0, 1, 2, 4], np.int64)
    v_starts = np.array([5, 12, 20], np.int32)
    ilens = np.array([0, -1, 1], np.int32)  # SNP, 1bp del, 1bp ins
    return alt_data, alt_off, ref_data, ref_off, v_starts, ilens


@pytest.mark.parametrize("tok_dtype", [np.uint8, np.int32])
@pytest.mark.parametrize("ref_mode,alt_mode", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_windows_mode_matrix(tok_dtype, ref_mode, alt_mode):
    ref, ref_offsets = _reference()
    alt_data, alt_off, ref_data, ref_off, v_starts, ilens = _globals()
    lut = _lut(tok_dtype)
    # one row selecting all 3 variants
    v_idxs = np.array([0, 1, 2], np.int32)
    row_offsets = np.array([0, 3], np.int64)
    v_contigs = np.zeros(3, np.int32)
    assert_kernel_parity_dict(
        "assemble_variant_buffers",
        1,  # windows
        v_idxs, row_offsets, alt_data, alt_off, ref_data, ref_off,
        False, False, ref_mode, alt_mode, 2, lut, v_contigs, v_starts, ilens,
        ref, ref_offsets, ord("N"),
    )


@pytest.mark.parametrize("tok_dtype", [np.uint8, np.int32])
@pytest.mark.parametrize("want_ref,want_flank", [(False, False), (True, False), (False, True), (True, True)])
def test_variants_mode_matrix(tok_dtype, want_ref, want_flank):
    ref, ref_offsets = _reference()
    alt_data, alt_off, ref_data, ref_off, v_starts, ilens = _globals()
    lut = _lut(tok_dtype) if want_flank else None
    v_idxs = np.array([2, 0, 1], np.int32)
    row_offsets = np.array([0, 1, 3], np.int64)  # 2 rows
    v_contigs = np.zeros(3, np.int32)
    assert_kernel_parity_dict(
        "assemble_variant_buffers",
        0,  # variants
        v_idxs, row_offsets, alt_data, alt_off, ref_data, ref_off,
        want_ref, want_flank, 0, 0, 2, lut, v_contigs, v_starts, ilens,
        ref, ref_offsets, ord("N"),
    )


@pytest.mark.parametrize("mode,ref_mode,alt_mode", [(0, 0, 0), (1, 1, 1)])
def test_empty_selection(mode, ref_mode, alt_mode):
    """A row that selects zero variants must round-trip identically."""
    ref, ref_offsets = _reference()
    alt_data, alt_off, ref_data, ref_off, v_starts, ilens = _globals()
    lut = _lut(np.uint8)
    v_idxs = np.array([], np.int32)
    row_offsets = np.array([0, 0], np.int64)  # 1 empty row
    v_contigs = np.array([], np.int32)
    assert_kernel_parity_dict(
        "assemble_variant_buffers",
        mode,
        v_idxs, row_offsets, alt_data, alt_off, ref_data, ref_off,
        False, (mode == 0), ref_mode, alt_mode, 2, lut, v_contigs, v_starts, ilens,
        ref, ref_offsets, ord("N"),
    )
```

> Clean up the placeholder lines in `_globals` (the first two `alt`/`alt_bytes`/`alt_data` reassignments are scratch — keep only the final explicit `alt_data = np.frombuffer(b"A" b"CG" b"T", np.uint8)`). Verify the test file has no unused locals via `ruff check`.

- [ ] **Step 2: Run the kernel parity on both backends**

Run:
```bash
pixi run -e dev pytest tests/parity/test_assemble_variant_buffers_parity.py -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | rtk err
GVL_BACKEND=numba pixi run -e dev pytest tests/parity/test_assemble_variant_buffers_parity.py -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | rtk err
```
Expected: all PASS on both backends. (The dict harness compares numba vs rust internally regardless of `GVL_BACKEND`, but running both confirms registration import paths are env-independent.)

- [ ] **Step 3: Add a live-path kernel spy to the dataset backstop**

In `tests/parity/test_dataset_parity.py`, add a test that monkeypatches the registry's rust entry for `assemble_variant_buffers` with a counting wrapper, opens a small variant-windows dataset, indexes one batch, and asserts the wrapper was called (proves the kernel runs on the live `__getitem__`, guarding against a vacuous parity pass). Mirror the existing spy pattern in that file. Skeleton:

```python
def test_assemble_variant_buffers_runs_on_live_windows_path(tmp_path):
    """The rust mega-call must actually fire on the windows __getitem__ path."""
    from genvarloader import _dispatch

    entry = _dispatch._REGISTRY["assemble_variant_buffers"]
    calls = {"n": 0}
    real = entry["rust"]

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    entry["rust"] = spy
    try:
        ds = _open_variant_windows_dataset(tmp_path)  # reuse this file's helper
        _ = ds[0, 0]
    finally:
        entry["rust"] = real
    assert calls["n"] > 0, "assemble_variant_buffers never ran on the live path"
```

> Use the existing dataset-construction helper in `test_dataset_parity.py` (grep for how the file builds a windows/variants dataset: `rtk grep "variant.windows\|VarWindowOpt\|with_seqs" tests/parity/test_dataset_parity.py`). If no windows helper exists, build a minimal one with `gvl.write` + `Dataset.open(...).with_seqs("variant-windows", VarWindowOpt(...))`, matching the corpus the other dataset-parity tests use.

- [ ] **Step 4: Run the dataset backstop + the variants/windows dataset parity, both backends**

Run:
```bash
pixi run -e dev pytest tests/parity/test_dataset_parity.py tests/parity/test_variants_dataset_parity.py -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | rtk err
GVL_BACKEND=numba pixi run -e dev pytest tests/parity/test_dataset_parity.py tests/parity/test_variants_dataset_parity.py -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | rtk err
```
Expected: all PASS on both backends.

- [ ] **Step 5: Full tree, both backends, + lint/format/typecheck**

Run:
```bash
pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | rtk err
GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | rtk err
pixi run -e dev cargo-test 2>&1 | rtk err
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
```
Expected: full tree PASSES on both backends (except the pre-existing `test_e2e_variants` xfail, which must xfail identically — confirm it is xfail, not fail). Rust tests pass; lint/format/typecheck clean.

- [ ] **Step 6: Commit**

```bash
rtk git add tests/parity/test_assemble_variant_buffers_parity.py tests/parity/test_dataset_parity.py
rtk git commit -m "test(parity): assemble_variant_buffers mode matrix + live-path spy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Perf re-measure + roadmap update

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (round-2 target 7 entry + re-measurement block + Phase-5 marker/PR link)

**Interfaces:** none (documentation + measurement).

- [ ] **Step 1: Confirm the pre-existing xfail is unchanged at this branch**

Run: `pixi run -e dev pytest tests/benchmarks/test_e2e.py::test_e2e_variants -q --basetemp=$(pwd)/.pytest_tmp 2>&1 | rtk err`
Expected: `xfailed` (NOT failed, NOT passed). Record that it matches base behavior.

- [ ] **Step 2: Re-measure variant-windows and variants (rust vs numba, min of pedantic)**

Run (build release first if not already):
```bash
pixi run -e dev maturin develop --release 2>&1 | rtk err
pixi run -e dev pytest tests/benchmarks/test_e2e.py -k "variant" --benchmark-only -q --basetemp=$(pwd)/.pytest_tmp
```
Also capture the `perf` flat self-time to confirm the GC/eval share dropped:
```bash
NUMBA_NUM_THREADS=1 perf record -F 999 -o p.data -- .pixi/envs/dev/bin/python \
    tests/benchmarks/profiling/profile.py --mode variant-windows --n-batches 12000
perf report --stdio --no-children -i p.data | head -40
```
Expected: GC (`gc_collect_main`/`deduce_unreachable`/`visit_reachable`/`dict_traverse`) self-time share is materially lower than the ~14% baseline; record the new variant-windows and variants min-ms ratios.

- [ ] **Step 3: Update the roadmap**

In `docs/roadmaps/rust-migration.md`, change target 7's marker from ⬜ to ✅ (or 🚧 with the PR link if not yet merged), append the re-measured variant-windows/variants ratios to the round-2 re-measurement block, and set the PR link. Keep the wording consistent with how targets 1–4 record their results (status marker + branch/PR + before→after numbers).

- [ ] **Step 4: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): target 7 done — variant-windows rust assembly, re-measured

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Final push gate (per CLAUDE.md)**

Confirm the full tree is green on both backends (Task 8 Step 5) and the branch is ready for PR. Open the PR against `zero-copy-scale-safe-readpath` (the base branch), not `master`.

---

## Self-Review

**Spec coverage:**
- Scope = all variants + windows → Tasks 3 (variants mode) + 4 (windows mode), routed in Task 7. ✓
- Rust owns the fetch → Task 2 `fetch_windows` reusing `reference::get_reference`. ✓
- One mega-call → single FFI entry per token dtype (Task 5), one dispatch key (Task 6). ✓
- Front edge = assembly tail only → front-end + scalar gather untouched in Task 7; #231 dtype-polymorphic fields never routed through the typed call. ✓
- fill_empty stays separate → Task 7 keeps `fill_empty_groups` post-pass. ✓
- Parity via registry with numba oracle → Task 6 oracle + Task 8 mode-matrix + live-path spy. ✓
- Perf gate + roadmap → Task 9. ✓
- Pre-existing xfail handling → Task 9 Step 1 + Task 8 Step 5 note. ✓
- Scale-guard not regressed → globals sourced from `ffi_static` (sub-linear), no new `ascontiguousarray` on sample-scale memmaps. ✓

**Placeholder scan:** Two intentional verification-and-adjust notes remain (Task 6 Step 1 import-path confirmation; Task 7 Step 3 field-order preservation; Task 8 Step 3 dataset-helper reuse). These are explicit "grep-then-confirm" instructions with the exact command and fallback, not vague TODOs — acceptable because the exact existing symbol/helper must be confirmed against the live tree rather than guessed.

**Type consistency:** `VariantBufs<Tok>` (Task 3) is consumed unchanged in Tasks 4–5. Field names (`alt`, `ref`, `ref_window`, `alt_window`, `flank_tokens`) are identical across the Rust orchestrators (Tasks 3–4), the numba oracle (Task 6), the Python wrapping (Task 7), and the parity test (Task 8). The mega-call argument order is identical across the Rust pyfunctions (Task 5), the rust shim + numba oracle (Task 6), and both call sites (Task 7) and the parity tests (Task 8).

---

## Risks & watch-points (for the implementer)

- **Field insertion order** (`_FlatVariants.fields`) feeds `RaggedVariants` construction order downstream. Task 7 Step 3 must preserve today's order (`alt` first where it was first); the dataset parity in Task 8 Step 4 is the gate that catches a reordering.
- **`reference is None`** path: variants mode with no reference + no flank must still emit `alt` (and `ref`) bytes. Task 7 passes zero-length reference placeholders in that case; the empty-selection parity (Task 8 `test_empty_selection`) and the no-reference dataset parity cover it.
- **Token dtype selection**: `_assemble_variant_buffers_rust` picks i32 only when `lut.dtype == int32`; otherwise u8. When `lut is None` (plain variants, no flank), u8 entry with `lut=None` — the orchestrator never touches the LUT on that path.
- **`unphased_union`**: `row_offsets` is already folded to `eff_ploidy=1` before the kernel call (front-end, unchanged). `v_contigs` is built with `eff_ploidy`, so it stays consistent. Add an `unphased_union=True` windows fixture to the dataset parity if the existing corpus lacks one.
