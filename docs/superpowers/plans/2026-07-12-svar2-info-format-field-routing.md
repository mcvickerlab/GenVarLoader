# SVAR2 INFO/FORMAT Field Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route arbitrary scalar-numeric SVAR2 INFO/FORMAT fields through gvl's read-bound decode kernel into `RaggedVariants` and variant-windows (`_FlatVariantWindows`) outputs.

**Architecture:** The seam is Rust-side: gvl's `decode_variants_from_svar2_readbound` kernel gains provenance tracking (via genoray's `gather_haps_readbound_src` + `unpack_vk_src`/`dense_abs_row`) and gathers per-variant field bytes via genoray's exported `FieldView`. Field byte-buffers ride the existing `pos`-parallel offset machinery, so multi-contig reorder and `unphased_union` need no special handling. Python discovers the store's field manifest and wraps the returned buffers into the output types.

**Tech Stack:** Rust (PyO3, `genoray_core` crate path-dep, `ndarray`/`numpy`), Python (numpy, `seqpro.rag.Ragged`), pixi, genoray SVAR2 field-read API (unreleased, from genoray main).

## Global Constraints

- **genoray dependency:** build a wheel from genoray **main** (`/carter/users/dlaub/projects/genoray`, HEAD ~`acc59cb`, which has the field-read API — it is 87 commits past the `3.0.0` tag and UNRELEASED). Rust links it via the existing `Cargo.toml` path-dep. Do **not** modify genoray. Do **not** hand-bump genoray's version.
- **Drop gvl's genoray version pin** (`pyproject.toml` `"genoray>=3,<4"` → `"genoray"`); re-pin at genoray release. The wheel will report version `2.15.0` — acceptable once the pin is dropped.
- **Field types supported:** scalar-numeric only — INFO/FORMAT `Type=Integer/Float` (+ `Flag` for INFO), `Number` `1` or `A`. Non-scalar fields are rejected at genoray's write boundary and never reach gvl.
- **Dtype fidelity:** field values are passed as raw little-endian bytes + an itemsize from Rust; Python `.view(dtype)`s them. No widening/conversion. Missing entries carry genoray's stored `default`/sentinel verbatim.
- **Zero-overhead no-field path:** when no fields are requested, the kernel must behave byte-identically to today (call `gather_haps_readbound`, no provenance, no `FieldView`).
- **NFS build gotchas (from genoray dev notes):** export `CARGO_TARGET_DIR=/tmp/gvl-target` before `cargo`/`maturin` on this NFS checkout to avoid linker bus errors; genoray's own Rust tests need `--no-default-features`.
- **Commit convention:** Conventional Commits (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`/`chore:`). All commits to branch `svar2-m6b-kernel`. Never touch `main`.
- **Pre-commit hook caveat:** the `pyrefly-check` pre-commit hook fails to solve the env until Phase 0 Task 2 completes. Until then, commit with `git commit --no-verify`. **After** Task 2, drop `--no-verify` and let hooks run.

## File Structure

**Rust (crate root `/carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel`):**
- `src/ffi/mod.rs` — FFI entry `decode_variants_from_svar2_readbound` (`:1330`); gains a `fields` param + field-buffer outputs. Also the tuple→Range migration at all `HapRanges::new` call sites (`:963,1116,1227,1372`).
- `src/svar2/mod.rs` — `decode_variants_from_split` (`:~282`) gains optional provenance capture + field gather; `split_to_flat` (`:~158`) tuple→Range fix; new `FieldGather` struct; tests.
- `src/svar2/store.rs` — `Svar2Store` retains `store_path` + accessor (needed to rebuild `ContigPaths` for `FieldView`).
- `Cargo.toml` — no code change (already path-deps genoray main); may need `use` of new genoray symbols in the `.rs` files.

**Python:**
- `python/genvarloader/_dataset/_svar2_haps.py` — field discovery in `from_path`/`__post_init__`; field pass-through + wrapping in `_reconstruct_variants` and `_reconstruct_variant_windows`.
- `python/genvarloader/_dataset/_impl.py` — `var_fields` request block (`:338-369`): add an early SVAR2 branch that skips SVAR1 lazy-loading.

**Config / docs:**
- `pyproject.toml` (`:14`), `pixi.toml` (`:110`) — dependency.
- `tests/dataset/test_svar2_fields_read.py` (new) — integration oracle test.
- `skills/genvarloader/SKILL.md`, `CHANGELOG.md` — docs.

---

# Phase 0 — Baseline: migrate gvl to genoray main (green build, no feature yet)

> gvl's Rust was written against genoray 2.15.0, where `BatchResultSplit`/`HapRanges` range fields were `(usize,usize)` tuples. genoray's `505a37f` refactor changed them to `Range<usize>` (present since the `3.0.0` tag). The "adopt genoray 3.0.0 API" commit touched **only Python** (0 `.rs` files), so the Rust does not compile against current genoray. Phase 0 makes it compile and pass the existing test suite against genoray main, establishing the baseline the field feature builds on.

### Task 0.1: Build a genoray wheel from main

**Files:**
- None in gvl (produces a wheel on disk).

- [ ] **Step 1: Build + repair the wheel**

```bash
cd /carter/users/dlaub/projects/genoray
export CARGO_TARGET_DIR=/tmp/genoray-wheel-target
pixi run --manifest-path ci/wheel/pixi.toml build      # -> wheelhouse/*.whl (abi3 cp310)
pixi run --manifest-path ci/wheel/pixi.toml repair      # auditwheel -> dist/*.whl (manylinux)
ls -la dist/*.whl
```

Expected: a `genoray-2.15.0-cp310-abi3-manylinux_*_x86_64.whl` in `dist/`.

- [ ] **Step 2: Sanity-check the wheel has the field-read API**

```bash
cd /carter/users/dlaub/projects/genoray
python -c "import zipfile,glob; z=zipfile.ZipFile(glob.glob('dist/genoray-*-abi3-manylinux*.whl')[0]); src=z.read('genoray/_svar2.py').decode(); print('available_fields' in src, 'from_pgen' in src)"
```

Expected: `True True`.

- [ ] **Step 3: Commit** — nothing to commit (artifact only). Record the wheel path for Task 0.2.

### Task 0.2: Point gvl at the wheel + drop the version pin

**Files:**
- Modify: `pyproject.toml:14`
- Modify: `pixi.toml:110`

**Interfaces:**
- Produces: a solvable pixi env with genoray main (field-read API) importable.

- [ ] **Step 1: Drop the version constraint**

In `pyproject.toml`, change the dependency line:

```toml
    "genoray",
```

(from `"genoray>=3,<4",`).

- [ ] **Step 2: Point the pixi pin at the new wheel**

In `pixi.toml`, replace the `genoray = { path = ... }` line (`:110`) with the manylinux wheel built in Task 0.1 (use the actual filename from `dist/`):

```toml
genoray = { path = "/carter/users/dlaub/projects/genoray/dist/genoray-2.15.0-cp310-abi3-manylinux_2_28_x86_64.whl" }
```

- [ ] **Step 3: Re-solve + verify import**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
pixi install
pixi run python -c "from genoray import SparseVar2; print(hasattr(SparseVar2,'available_fields'), hasattr(SparseVar2,'with_fields'))"
```

Expected: solve succeeds; prints `True True`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.toml pixi.lock
git commit --no-verify -m "chore(svar2): pin genoray main wheel, drop version constraint for dev"
```

### Task 0.3: Migrate gvl Rust range types tuple→Range

**Files:**
- Modify: `src/ffi/mod.rs` (the `to_pairs` closure + all `HapRanges::new` sites `:963,1116,1227,1372`; the `decode_variants_from_svar2_readbound` closure `:~1356-1370`)
- Modify: `src/svar2/mod.rs` (`split_to_flat` `:~158`, `decode_variants_from_split` `:~300`, and the `#[cfg(test)]` `BatchResultSplit` literals)

**Interfaces:**
- Consumes: `genoray_core::query::HapRanges::new(&[u32], &[usize], &[Range<usize>], &[Range<usize>], &[Range<usize>], &[Range<usize>], usize)`; `BatchResultSplit { dense_snp_range: Vec<Range<usize>>, dense_indel_range: Vec<Range<usize>>, vk_src: Vec<u32>, dense_src: Vec<(bool,u32)>, .. }` (derives `Default`).
- Produces: a gvl crate that compiles against genoray main; behavior byte-identical (no feature).

- [ ] **Step 1: Convert the FFI range marshaling to `Range<usize>`**

In `src/ffi/mod.rs`, replace the `to_pairs` closure (in `decode_variants_from_svar2_readbound`, ~`:1356`) and the analogous marshaling in the reconstruct/diffs/tracks FFI fns with a `to_ranges` helper. Add near the top of the module:

```rust
use std::ops::Range;

fn arr2_to_ranges(a: numpy::ndarray::ArrayView2<i64>) -> Vec<Range<usize>> {
    a.rows()
        .into_iter()
        .map(|r| (r[0] as usize)..(r[1] as usize))
        .collect()
}
```

Then at each site that built `Vec<(usize,usize)>` for a `HapRanges::new` arg (search `to_pairs` and inline `.map(|r| (r[0] as usize, r[1] as usize))`), replace with `arr2_to_ranges(<array>.as_array())`. The `HapRanges::new(...)` calls themselves are unchanged (they now receive `&[Range<usize>]`).

- [ ] **Step 2: Convert `BatchResultSplit` range consumption in `src/svar2/mod.rs`**

In `split_to_flat` and `decode_variants_from_split`, every `let (ss, se) = br.dense_snp_range[q];` / `let (is_, ie) = br.dense_indel_range[q];` becomes a `Range` read. Use `.clone()`-free field access:

```rust
let Range { start: ss, end: se } = br.dense_snp_range[q];
let Range { start: is_, end: ie } = br.dense_indel_range[q];
```

(add `use std::ops::Range;` to `src/svar2/mod.rs`). All downstream uses of `ss/se/is_/ie` are unchanged (they're `usize`).

- [ ] **Step 3: Fix the `#[cfg(test)]` `BatchResultSplit` literals**

In `src/svar2/mod.rs` tests, every `dense_snp_range: vec![(0, 1)]` etc. becomes `dense_snp_range: vec![0..1]`, `vec![(0, 2), (2, 4)]` becomes `vec![0..2, 2..4]`, etc. `BatchResultSplit` derives `Default`, so add `..Default::default()` at the end of each struct literal to cover the new `vk_src`/`dense_src` fields:

```rust
let br = BatchResultSplit {
    n_regions: 1,
    n_samples: 1,
    ploidy: 2,
    vk: vec![/* ... */],
    vk_off: vec![/* ... */],
    dense_snp: vec![/* ... */],
    dense_snp_range: vec![0..1],
    // ... other existing fields ...
    ..Default::default()
};
```

- [ ] **Step 4: Compile to green**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run cargo check 2>&1 | tail -30
```

Expected: `Finished`. If additional tuple→Range or new-field errors surface (compiler-driven; the refactor is mechanical), fix each at the reported site exactly as in Steps 1–3, then re-run until green.

- [ ] **Step 5: Run the Rust unit tests**

```bash
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run cargo test --no-default-features 2>&1 | tail -20
```

Expected: all existing svar2 Rust tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ffi/mod.rs src/svar2/mod.rs
git commit --no-verify -m "refactor(svar2): migrate gvl read-bound Rust to genoray Range<usize> API"
```

### Task 0.4: Build the extension + green existing test suite (baseline gate)

**Files:**
- None (build + test).

- [ ] **Step 1: Build the editable extension**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run maturin develop 2>&1 | tail -15
```

Expected: `Built ... Installed genvarloader`.

- [ ] **Step 2: Run the existing SVAR2 test suite**

```bash
pixi run pytest tests/dataset/test_svar2_readbound_variants.py tests/dataset/test_svar2_readbound_haps.py -x -q 2>&1 | tail -25
```

Expected: all PASS (baseline against genoray main; no feature yet). This gate proves the migration is correct before adding fields.

- [ ] **Step 3: Confirm the pre-commit hook now solves**

```bash
git commit --allow-empty -m "chore: verify hooks solve" && git reset --soft HEAD~1
```

Expected: no pyrefly solve error. From here on, commit **without** `--no-verify`.

---

# Phase 1 — Rust: provenance + field gather

### Task 1.1: `Svar2Store` retains `store_path`

**Files:**
- Modify: `src/svar2/store.rs`

**Interfaces:**
- Produces: `Svar2Store::store_path(&self) -> &str` — the store base dir, for rebuilding `ContigPaths`.

- [ ] **Step 1: Add the field + accessor**

In `src/svar2/store.rs`, add `store_path: String` to the struct, set it in `new`, and expose it:

```rust
#[pyclass]
pub struct Svar2Store {
    readers: HashMap<String, ContigReader>,
    store_path: String,
}

impl Svar2Store {
    pub fn reader(&self, contig: &str) -> Option<&ContigReader> {
        self.readers.get(contig)
    }
    pub fn store_path(&self) -> &str {
        &self.store_path
    }
}
```

In `new`, capture `store_path` before the loop consumes `contigs` (note: `store_path: &str` param):

```rust
        let store_path = store_path.to_string();
        // ... existing loop populating `readers` ...
        Ok(Self { readers, store_path })
```

- [ ] **Step 2: Compile**

```bash
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run cargo check 2>&1 | tail -10
```

Expected: `Finished`.

- [ ] **Step 3: Commit**

```bash
git add src/svar2/store.rs
git commit -m "feat(svar2): Svar2Store retains store_path for field-sidecar paths"
```

### Task 1.2: `FieldGather` + provenance capture in `decode_variants_from_split`

**Files:**
- Modify: `src/svar2/mod.rs`
- Test: `src/svar2/mod.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `genoray_core::query::{unpack_vk_src, dense_abs_row, FieldView}`; `BatchResultSplit.vk_src` (populated only by `gather_haps_readbound_src`).
- Produces:
  ```rust
  pub struct FieldGather<'a> {
      pub views: [genoray_core::query::FieldView; 4], // FieldSub order: VkSnp,VkIndel,DenseSnp,DenseIndel
      pub is_format: bool,
      pub width: usize,                                 // dtype.width_bytes()
      pub cohort_n_samples: usize,
      pub _marker: std::marker::PhantomData<&'a ()>,
  }
  pub fn decode_variants_from_split(
      br: &BatchResultSplit,
      lut_bytes: &[u8],
      lut_off: &[i64],
      fields: &[FieldGather<'_>],
      on_disk_snp: &[std::ops::Range<usize>],   // HapRanges dense_snp_range; empty if no fields
      on_disk_indel: &[std::ops::Range<usize>],
      orig_samples: &[usize],
  ) -> (VariantsSoa, Vec<Vec<u8>>)               // second: one byte-buffer per field, parallel to pos
  ```
  When `fields.is_empty()`, the second return is `Vec::new()` and no provenance work runs (byte-identical to today).

- [ ] **Step 1: Write the failing Rust test (provenance identity)**

Add to `src/svar2/mod.rs` `#[cfg(test)] mod tests`. This test builds a split with one var_key entry and one dense-snp entry and an INFO field whose stored value at each source row equals that row index, then asserts the decoded field values match the merge order:

```rust
#[test]
fn test_decode_fields_provenance_identity() {
    use genoray_core::query::{pack_vk_src, KeyRef};
    // One query, ploidy 1. var_key call idx 5 at pos 10; dense-snp row (abs) 3 at pos 20.
    let br = BatchResultSplit {
        n_regions: 1,
        n_samples: 1,
        ploidy: 1,
        vk: vec![KeyRef { position: 10, key: svar2_codec::encode_pure_del(-1) }],
        vk_off: vec![0, 1],
        vk_src: vec![pack_vk_src(false, 5)], // var_key/snp, call idx 5
        dense_snp: vec![KeyRef { position: 20, key: svar2_codec::encode_pure_del(-1) }],
        dense_snp_range: vec![0..1],
        dense_snp_present: vec![0b1],
        dense_snp_present_off: vec![0, 1],
        dense_indel: vec![],
        dense_indel_range: vec![0..0],
        dense_indel_present: vec![],
        dense_indel_present_off: vec![0, 0],
    };
    // INFO field: value_at(i) == i (an i32 store big enough for idx 5 and dense row 3).
    // Build FieldView test doubles via a helper that fills values.bin so value_at(i)=i.
    let fields = make_identity_i32_fields(); // see Step 3 helper
    let (soa, bufs) = decode_variants_from_split(
        &br, &[], &[0i64], &fields, &[0..1], &[0..0], &[0],
    );
    assert_eq!(soa.pos, vec![10, 20]); // var_key before dense on distinct positions
    // Field buffer holds i32 LE bytes: [5, 3] (vk call idx 5 first, dense abs row 3 second).
    let vals: Vec<i32> = bufs[0]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals, vec![5, 3]);
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run cargo test --no-default-features test_decode_fields_provenance_identity 2>&1 | tail -15
```

Expected: FAIL — `decode_variants_from_split` arity mismatch / `make_identity_i32_fields` undefined.

- [ ] **Step 3: Implement `FieldGather`, the test helper, and the gather**

Add the struct + `use` lines at the top of `src/svar2/mod.rs`:

```rust
use std::ops::Range;
use genoray_core::query::{dense_abs_row, unpack_vk_src, FieldView};

pub struct FieldGather {
    pub views: [FieldView; 4], // FieldSub::all() order: VkSnp, VkIndel, DenseSnp, DenseIndel
    pub is_format: bool,
    pub width: usize,
    pub cohort_n_samples: usize,
}
```

Change `decode_variants_from_split`'s signature to the one in **Interfaces** above. Inside the merge loop, at the point where `(p, key)` is selected, ALSO record which channel + index it came from, then (only if `!fields.is_empty()`) resolve provenance and append field bytes. Concretely, replace the selection block:

```rust
                let (p, key, chan, cidx) = if has_vk && p_vk <= p_sn && p_vk <= p_in {
                    let e = &br.vk[i_vk];
                    let out = (e.position, e.key, 0u8, i_vk);
                    i_vk += 1;
                    out
                } else if has_sn && p_sn <= p_in {
                    let e = &br.dense_snp[i_sn];
                    let out = (e.position, e.key, 1u8, i_sn);
                    i_sn += 1;
                    out
                } else {
                    let e = &br.dense_indel[i_in];
                    let out = (e.position, e.key, 2u8, i_in);
                    i_in += 1;
                    out
                };
                let (il, alt) = decode_alt(key, lut_bytes, lut_off);
                pos.push(p as i32);
                ilen.push(il as i32);
                alt_bytes.extend_from_slice(&alt);
                str_off.push(alt_bytes.len() as i64);

                if !fields.is_empty() {
                    // Resolve (is_dense, is_indel, elem_row) for this emitted variant.
                    let (is_dense, is_indel, row) = match chan {
                        0 => {
                            let (is_indel, call_idx) = unpack_vk_src(br.vk_src[cidx]);
                            (false, is_indel, call_idx)
                        }
                        1 => (true, false, dense_abs_row(&on_disk_snp[q], &(ss..se), cidx)),
                        _ => (true, true, dense_abs_row(&on_disk_indel[q], &(is_..ie), cidx)),
                    };
                    let sub_ix = match (is_dense, is_indel) {
                        (false, false) => 0,
                        (false, true) => 1,
                        (true, false) => 2,
                        (true, true) => 3,
                    };
                    for (fi, f) in fields.iter().enumerate() {
                        let view = &f.views[sub_ix];
                        let elem = if is_dense && f.is_format {
                            row * f.cohort_n_samples + orig_samples[q]
                        } else {
                            row
                        };
                        field_bufs[fi].extend_from_slice(view.bytes_at(elem));
                    }
                }
```

Note: the var_key `sub_ix` uses `is_indel` from `unpack_vk_src` (a var_key entry may be snp OR indel), matching `FieldSub` order. Declare `field_bufs` before the loop:

```rust
    let mut field_bufs: Vec<Vec<u8>> = (0..fields.len()).map(|_| Vec::new()).collect();
```

and return `(VariantsSoa { .. }, field_bufs)`.

`ss/se/is_/ie` are already in scope per query (from Task 0.3 Step 2). Update the three existing callers of `decode_variants_from_split` in `src/ffi/mod.rs` (variants, and the two reconstruct/windows paths that call it — grep `decode_variants_from_split`) to pass `&[]`, `&[]`, `&[]`, `&[]` for the no-field case for now (Task 1.3 wires the real args for the variants FFI).

Add the test helper `make_identity_i32_fields()` in `#[cfg(test)]`. Because `FieldView::open` reads real sidecar files, the helper writes a tiny temp store dir with `values.bin` files whose i32 elements equal their index (0,1,2,...) for each of the 4 subs, then opens 4 `FieldView`s via `genoray_core::query::FieldView::open(&ContigPaths::new(tmp, "chr1"), "info", "X", sub, StorageDtype::from_meta_str("i32"), 1)` for `sub in FieldSub::all()`. (Use `tempfile::tempdir()`; add `tempfile` as a dev-dependency in `Cargo.toml` if not present.)

- [ ] **Step 4: Run the test to verify it passes**

```bash
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run cargo test --no-default-features test_decode_fields_provenance_identity 2>&1 | tail -15
```

Expected: PASS. Also re-run the full Rust suite to confirm the no-field callers still pass:

```bash
pixi run cargo test --no-default-features 2>&1 | tail -15
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/svar2/mod.rs Cargo.toml
git commit -m "feat(svar2): field-byte gather with vk_src/dense provenance in split decode"
```

### Task 1.3: FFI — `decode_variants_from_svar2_readbound` gains fields

**Files:**
- Modify: `src/ffi/mod.rs` (`:1330`)

**Interfaces:**
- Consumes: `genoray_core::query::{gather_haps_readbound_src, FieldView}`, `genoray_core::layout::{ContigPaths, FieldSub}`, `genoray_core::field::StorageDtype`, `Svar2Store::store_path`, `ContigReader::n_samples`.
- Produces: Python signature
  ```
  decode_variants_from_svar2_readbound(
      store, contig, region_starts, orig_samples,
      vk_snp_range, vk_indel_range, dense_snp_range, dense_indel_range,
      ploidy,
      fields: list[tuple[str, str, str]],   # (category, name, dtype_str); may be []
  ) -> (pos, ilen, alt_bytes, str_off, var_off,
        field_bufs: list[np.ndarray[u8]], field_itemsizes: list[int])
  ```

- [ ] **Step 1: Add the `fields` param and open `FieldView`s**

Add `fields: Vec<(String, String, String)>` as the last param. After resolving `reader`, and only when `!fields.is_empty()`, build a `Vec<FieldGather>`:

```rust
    use genoray_core::field::StorageDtype;
    use genoray_core::layout::{ContigPaths, FieldSub};

    let n_samples = reader.n_samples();
    let paths = ContigPaths::new(store.store_path(), contig);
    let gathers: Vec<crate::svar2::FieldGather> = fields
        .iter()
        .map(|(cat, name, dtype_str)| {
            let dtype = StorageDtype::from_meta_str(dtype_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let width = dtype.width_bytes().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("field {name}: unresolved dtype"))
            })?;
            let subs = FieldSub::all();
            let views = [
                FieldView::open(&paths, cat, name, subs[0], dtype, n_samples),
                FieldView::open(&paths, cat, name, subs[1], dtype, n_samples),
                FieldView::open(&paths, cat, name, subs[2], dtype, n_samples),
                FieldView::open(&paths, cat, name, subs[3], dtype, n_samples),
            ];
            let mut opened = Vec::with_capacity(4);
            for v in views {
                opened.push(v.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?);
            }
            Ok(crate::svar2::FieldGather {
                views: [opened.remove(0), opened.remove(0), opened.remove(0), opened.remove(0)],
                is_format: cat == "format",
                width,
                cohort_n_samples: n_samples,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
```

(Confirm `StorageDtype::from_meta_str` returns `Result`; if it returns `Option`, adapt the `.map_err`. `FieldSub::all()` returns `[FieldSub; 4]` in order VkSnp,VkIndel,DenseSnp,DenseIndel.)

- [ ] **Step 2: Choose the gather + call the decode with field args**

Replace the `py.detach` body so it uses `gather_haps_readbound_src` when fields are requested, and passes the on-disk dense ranges + `orig_samples`:

```rust
    let has_fields = !gathers.is_empty();
    let (soa, field_bufs) = py.detach(move || {
        let rb = genoray_core::query::HapRanges::new(
            &region_starts_v, &orig_samples_v,
            &vk_snp_range_v, &vk_indel_range_v,
            &dense_snp_range_v, &dense_indel_range_v, ploidy,
        );
        let br = if has_fields {
            genoray_core::query::gather_haps_readbound_src(reader, &rb)
        } else {
            genoray_core::query::gather_haps_readbound(reader, &rb)
        };
        let (lut_bytes, lut_off_u64) = reader.lut_arrays();
        let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();
        crate::svar2::decode_variants_from_split(
            &br, &lut_bytes, &lut_off,
            &gathers, &dense_snp_range_v, &dense_indel_range_v, &orig_samples_v,
        )
    });
```

(Requires `dense_snp_range_v`/`dense_indel_range_v`/`orig_samples_v` to be `move`d in; they already are `Vec`s from Task 0.3. `gathers` is `move`d in.)

- [ ] **Step 3: Return the field buffers + itemsizes**

Change the return type to append `Vec<Bound<PyArray1<u8>>>` + `Vec<usize>`, and build them:

```rust
    let field_out: Vec<Bound<'py, PyArray1<u8>>> =
        field_bufs.into_iter().map(|b| Array1::from_vec(b).into_pyarray(py)).collect();
    let itemsizes: Vec<usize> = gathers.iter().map(|g| g.width).collect();
    Ok((
        Array1::from_vec(soa.pos).into_pyarray(py),
        Array1::from_vec(soa.ilen).into_pyarray(py),
        Array1::from_vec(soa.alt_bytes).into_pyarray(py),
        Array1::from_vec(soa.str_off).into_pyarray(py),
        Array1::from_vec(soa.var_off).into_pyarray(py),
        field_out,
        itemsizes,
    ))
```

Update the return type in the signature to `PyResult<( ..5 existing.., Vec<Bound<'py, PyArray1<u8>>>, Vec<usize> )>`. **Move `let itemsizes` computation before the `py.detach` closure consumes `gathers`** (compute widths into a plain `Vec<usize>` first, then move `gathers` into the closure).

- [ ] **Step 4: Compile**

```bash
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run cargo check 2>&1 | tail -20
```

Expected: `Finished`. Fix borrow/move errors by hoisting `itemsizes` before the closure (Step 3 note).

- [ ] **Step 5: Rebuild the extension**

```bash
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run maturin develop 2>&1 | tail -8
```

Expected: `Installed genvarloader`.

- [ ] **Step 6: Commit**

```bash
git add src/ffi/mod.rs
git commit -m "feat(svar2): decode_variants_from_svar2_readbound returns INFO/FORMAT field buffers"
```

---

# Phase 2 — Python wiring

### Task 2.1: Field discovery in `Svar2Haps`

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py` (`__post_init__` `:169`, `from_path` `:178`, add dataclass fields)

**Interfaces:**
- Consumes: `genoray.SparseVar2(path).available_fields -> dict[str, StoredField]` where `StoredField.{name, category('info'|'format'), dtype(np.dtype), key}`; `genoray._svar2_fields._META_DTYPE: dict[np.dtype, str]`.
- Produces: `Svar2Haps.available_var_fields` includes store field keys; `Svar2Haps._store_fields: dict[str, tuple[str,str,str,np.dtype]]` keyed by field key → `(category, name, dtype_str, np_dtype)`.

- [ ] **Step 1: Add dataclass fields + populate in `from_path`**

Add two `field()`-defaulted slots to `Svar2Haps` (they follow base defaults):

```python
    store_field_keys: list[str] = field(default_factory=list)
    store_fields: dict[str, tuple[str, str, str, "np.dtype"]] = field(default_factory=dict)
    """key -> (category, name, dtype_str, np_dtype) for store INFO/FORMAT fields."""
```

In `from_path`, after `sv = SparseVar2(str(svar2_path))`:

```python
        from genoray._svar2_fields import _META_DTYPE

        store_field_keys = list(sv.available_fields.keys())
        store_fields = {
            sf.key: (sf.category, sf.name, _META_DTYPE[sf.dtype], sf.dtype)
            for sf in sv.available_fields.values()
        }
```

and pass `store_field_keys=store_field_keys, store_fields=store_fields` into the `cls(...)` call.

- [ ] **Step 2: Extend `available_var_fields` in `__post_init__`**

Replace the hard-coded line (`:174`):

```python
        self.available_var_fields = ["alt", "ilen", "start"] + [
            k for k in self.store_field_keys
            if k not in {"alt", "ilen", "start", "ref", "dosage"}
        ]
```

- [ ] **Step 3: Rebuild + smoke-check discovery**

Requires a fixture store with fields (created in Task 3.1). For now compile-check the import:

```bash
pixi run python -c "import genvarloader._dataset._svar2_haps"
```

Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add python/genvarloader/_dataset/_svar2_haps.py
git commit -m "feat(svar2): Svar2Haps discovers store INFO/FORMAT fields into available_var_fields"
```

### Task 2.2: `_impl.py` — SVAR2 branch skips SVAR1 lazy-loading

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (`:338-369`)

**Interfaces:**
- Consumes: `Svar2Haps` (import), `self.available_var_fields`.
- Produces: for `Svar2Haps`, `var_fields` is validated + set via `replace(...)` with no SVAR1 lazy-loading.

- [ ] **Step 1: Add the early SVAR2 branch**

At the top of the `if var_fields is not None:` block (`:338`), before the `custom_fmt`/`load_info` logic, insert:

```python
        if var_fields is not None:
            missing = list(set(var_fields) - set(self.available_var_fields))
            if missing or not isinstance(self._seqs, Haps):
                raise ValueError(f"Missing variant fields: {missing}")
            from ._svar2_haps import Svar2Haps

            if isinstance(self._seqs, Svar2Haps):
                # SVAR2 fields are read on demand by the decode kernel; no lazy
                # INFO/dosage/custom-FORMAT loading (that is SVAR1-only).
                haps = replace(to_evolve.get("_seqs", self._seqs), var_fields=var_fields)
                to_evolve["_seqs"] = haps
            else:
                # ... existing SVAR1 lazy-loading block (custom_fmt, load_info, ...) ...
                haps = to_evolve.get("_seqs", self._seqs)
                # (unchanged existing lines through)
                haps = replace(haps, var_fields=var_fields)
                to_evolve["_seqs"] = haps
```

(Keep the existing SVAR1 body verbatim inside the `else`. The `missing` check stays shared at the top.)

- [ ] **Step 2: Compile-check**

```bash
pixi run python -c "import genvarloader._dataset._impl"
```

Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "feat(svar2): skip SVAR1 field lazy-loading for Svar2Haps var_fields"
```

### Task 2.3: Route fields into `RaggedVariants`

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py` (`_reconstruct_variants` `:558`)

**Interfaces:**
- Consumes: the extended FFI (Task 1.3) returning `(pos, ilen, alt_bytes, str_off, var_off, field_bufs, field_itemsizes)`.
- Produces: `RaggedVariants(**fields)` including each requested store field, sharing `alt`'s offsets.

- [ ] **Step 1: Compute the requested field triples + call with fields**

At the top of `_reconstruct_variants`, compute the requested extra fields (order stable):

```python
        builtin = {"alt", "start", "ref", "ilen", "dosage"}
        req_keys = [f for f in self.var_fields if f not in builtin]
        field_specs = [
            (self.store_fields[k][0], self.store_fields[k][1], self.store_fields[k][2])
            for k in req_keys
        ]
        field_dtypes = [self.store_fields[k][3] for k in req_keys]
```

Change the kernel call to pass `field_specs`, and unpack the two new returns per contig group:

```python
            pos, ilen, alt_bytes, str_off, var_off, field_bufs, field_isizes = (
                decode_variants_from_svar2_readbound(
                    self.store, self.ds_contigs[ci],
                    gi[0], gi[1], gi[2], gi[3], gi[4], gi[5], P, field_specs,
                )
            )
```

Accumulate per-field buffers per group into `cat_fields: list[list[np.ndarray]]` (one inner list per requested field), asserting `field_isizes[j] == field_dtypes[j].itemsize`.

- [ ] **Step 2: Single-contig fast path — wrap fields**

In the `len(cat_pos) == 1` branch, build a `fields` dict parallel to the existing `alt`/`start`/`ilen`, then splat into `RaggedVariants`:

```python
            extra = {
                req_keys[j]: Ragged.from_offsets(
                    cat_fields[j][0].view(field_dtypes[j]), shape, var_off_g
                )
                for j in range(len(req_keys))
            }
            return RaggedVariants(
                alt=Ragged.from_offsets(cat_alt[0].view("S1"), shape, var_off_g, str_offsets=str_off_g),
                start=Ragged.from_offsets(cat_pos[0], shape, var_off_g),
                ilen=Ragged.from_offsets(cat_ilen[0], shape, var_off_g),
                **extra,
            )
```

- [ ] **Step 3: Multi-contig path — reorder fields by the same `src`**

In the general path, after computing `src, var_off_g = _ragged_arange_src(grouped_var_off, perm)`, each field is per-variant so it reorders exactly like `pos`:

```python
            extra = {}
            for j in range(len(req_keys)):
                fc = np.concatenate([g[j] for g in per_group_fields]) if per_group_fields else np.zeros(0, np.uint8)
                fc_typed = fc.view(field_dtypes[j])
                fg = fc_typed[:0].copy() if src.size == 0 else fc_typed[src]
                extra[req_keys[j]] = Ragged.from_offsets(fg, shape, var_off_g)
            return RaggedVariants(alt=alt_r, start=pos_r, ilen=ilen_r, **extra)
```

(`per_group_fields[g][j]` is group `g`'s buffer for field `j`, already `.view(dtype)`-ed to length `n_var_group`. Reuse the same `src` computed for `pos_g`/`ilen_g`.)

- [ ] **Step 4: Run the integration test (written in Task 3.1)**

```bash
pixi run pytest tests/dataset/test_svar2_fields_read.py -x -q 2>&1 | tail -25
```

Expected: `RaggedVariants` value assertions PASS (windows test may still fail until Task 2.4).

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_svar2_haps.py
git commit -m "feat(svar2): route INFO/FORMAT fields into RaggedVariants"
```

### Task 2.4: Route fields into variant-windows (`_FlatVariantWindows`)

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py` (`_reconstruct_variant_windows` `:653`)

**Interfaces:**
- Consumes: the extended FFI; `_FlatVariantWindows.fields: dict[str, _Flat]`; `_Flat.from_offsets`.
- Produces: variant-windows output whose `.fields` includes each requested store field (like `start`/`ilen`).

- [ ] **Step 1: Pass field_specs + collect per group**

Mirror Task 2.3 Step 1 in `_reconstruct_variant_windows`: compute `req_keys`/`field_specs`/`field_dtypes` (identical code), pass `field_specs` to `decode_variants_from_svar2_readbound`, unpack `field_bufs, field_isizes`, and collect per-group typed buffers.

- [ ] **Step 2: Add fields to the `fields` dict (single + multi-contig)**

In the `len(cat_pos) == 1` branch, after `fields = {"start": ...}` (and optional `ilen`), add:

```python
            for j, k in enumerate(req_keys):
                fields[k] = _Flat.from_offsets(
                    cat_fields[j][0].view(field_dtypes[j]), shape, row_off
                )
```

In the multi-contig `else`, after `pos_g`/`ilen_g`, reorder each field by the same `src` and add:

```python
            for j, k in enumerate(req_keys):
                fc = np.concatenate([g[j] for g in per_group_fields]).view(field_dtypes[j])
                fg = fc[:0].copy() if src.size == 0 else fc[src]
                fields[k] = _Flat.from_offsets(fg, shape, row_off_g)
```

- [ ] **Step 3: Extend `DummyVariant` fill for empty groups**

`fill_empty_groups` needs a fill value per field. Set the dummy's per-field fill to the store default (or the dtype's sentinel: `NaN` for float, `iinfo.min` for int, `False` for bool) in `DummyVariant.info` keyed by field key when the dummy is constructed for svar2 (locate where `self.dummy_variant` is set for `Svar2Haps`; if none is set for windows, construct one carrying the field fills). Assert in the test that empty groups carry the fill.

- [ ] **Step 4: Run the windows test**

```bash
pixi run pytest tests/dataset/test_svar2_fields_read.py -x -q 2>&1 | tail -25
```

Expected: all PASS (RaggedVariants + windows).

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_svar2_haps.py
git commit -m "feat(svar2): route INFO/FORMAT fields into variant-windows output"
```

---

# Phase 3 — Integration test + docs

### Task 3.1: Integration oracle test (write this BEFORE Task 2.3 — it is the Phase-2 gate)

**Files:**
- Create: `tests/dataset/test_svar2_fields_read.py`

**Interfaces:**
- Consumes: `genoray.SparseVar2.from_vcf(info_fields=, format_fields=)`, `gvl.write`, `gvl.Dataset(...).with_seqs("variants", var_fields=[...])`.

- [ ] **Step 1: Build the fixture VCF + convert with fields**

Write a helper that emits a small bgzipped VCF over TWO contigs with variants that route to BOTH var_key and dense channels (a common variant → dense, a rare per-sample variant → var_key), carrying INFO `AF` (Float), INFO `NS` (Integer), FORMAT `DP` (Integer). Convert:

```python
from genoray import SparseVar2
from genoray._svar2_fields import InfoField, FormatField

SparseVar2.from_vcf(
    out=store_dir, source=vcf_gz, reference=fasta,
    info_fields=[InfoField("AF"), InfoField("NS")],
    format_fields=[FormatField("DP")],
)
```

- [ ] **Step 2: Write the ground-truth oracle**

Parse the VCF with `cyvcf2` into a dict `{(contig, pos): {"AF": ..., "NS": ..., "DP": {sample: ...}}}`. This is the expected-value source.

- [ ] **Step 3: Write the failing test (RaggedVariants values)**

```python
import numpy as np
import pytest

@pytest.mark.parametrize("union", [False, True])
def test_svar2_ragged_variants_fields(tmp_path, union):
    ds_path = _write_dataset(tmp_path)  # gvl.write over the svar2 source
    import genvarloader as gvl
    ds = gvl.Dataset.open(ds_path, reference=FASTA).with_seqs(
        "variants", var_fields=["alt", "start", "ilen", "AF", "NS", "DP"],
    )
    if union:
        ds = ds.with_settings(unphased_union=True)  # use the real gvl API name
    rv = ds[0]  # RaggedVariants
    # For each (batch,ploid) group, walk variants and compare rv.AF / rv.NS / rv.DP
    # to the oracle by (contig, start). Assert exact dtype + value, incl. NaN for
    # VCF-missing (use np.testing.assert_array_equal with equal_nan for floats).
    _assert_fields_match(rv, oracle)  # helper asserts every decoded field value
```

Add a second test hitting a multi-contig batch (indices spanning both contigs) and a third asserting the FORMAT `DP` value equals the oracle's per-sample value for the queried sample.

- [ ] **Step 4: Run to verify it fails (before Task 2.3)**

```bash
pixi run pytest tests/dataset/test_svar2_fields_read.py -x -q 2>&1 | tail -20
```

Expected: FAIL — `AF` not in `available_var_fields` / not present on `rv`. (This test is the gate for Tasks 2.3/2.4; implement those to make it pass.)

- [ ] **Step 5: Add the variant-windows value test**

Add `test_svar2_variant_windows_fields` that requests variant-windows output with the same `var_fields` and asserts `win.fields["AF"]` etc. match the oracle, plus an empty-group case asserting the dummy fill.

- [ ] **Step 6: Commit (test only; may be red until Phase 2 done)**

```bash
git add tests/dataset/test_svar2_fields_read.py
git commit -m "test(svar2): oracle test for INFO/FORMAT field routing (var_key+dense, multi-contig, union)"
```

### Task 3.2: Full suite + docs

**Files:**
- Modify: `skills/genvarloader/SKILL.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run the full svar2 + variants suite**

```bash
export CARGO_TARGET_DIR=/tmp/gvl-target
pixi run maturin develop 2>&1 | tail -5
pixi run pytest tests/dataset -k svar2 -q 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 2: Update SKILL.md**

Document that svar2 `variants`/variant-windows outputs now honor arbitrary store INFO/FORMAT fields via `var_fields`, that available fields come from the store's manifest (`available_var_fields`), and the dtype/missing-value semantics (stored dtype preserved; missing = default/sentinel/NaN).

- [ ] **Step 3: Add CHANGELOG entry**

Under `## Unreleased`:

```markdown
### Added
- SVAR2 datasets now route arbitrary scalar-numeric INFO/FORMAT fields (stored via
  `genoray.SparseVar2.from_vcf(info_fields=, format_fields=)`) into `variants`
  (`RaggedVariants`) and variant-windows outputs, selectable via `var_fields`.
```

- [ ] **Step 4: Commit**

```bash
git add skills/genvarloader/SKILL.md CHANGELOG.md
git commit -m "docs(svar2): document INFO/FORMAT field routing in variants outputs"
```

- [ ] **Step 5: Push + update the draft PR**

```bash
git push origin svar2-m6b-kernel
```

The draft PR #266 updates automatically. Note in the PR description that the genoray Python pin is a local dev wheel pending a genoray svar-2 release (the version constraint was dropped for development).

---

## Self-Review

**Spec coverage:**
- Rust seam (provenance + FieldView) → Tasks 1.2, 1.3. ✓
- gvl→genoray-main migration (discovered; not in spec but a hard prerequisite) → Phase 0. ✓ (larger than the spec's "prerequisite" section anticipated — see handoff.)
- Field discovery (`available_fields`) → Task 2.1. ✓
- `RaggedVariants` routing → Task 2.3; variant-windows → Task 2.4. ✓
- Dense-FORMAT `orig_samples[q]` stride subtlety → Task 1.2 Step 3 (`row * cohort_n_samples + orig_samples[q]`). ✓
- `unphased_union` / multi-contig need no special handling → Tasks 2.3/2.4 use the same `src`/`var_off` machinery. ✓ (tested in 3.1 with `union` param.)
- Provenance-on-cursor invariant + identity test → Task 1.2 Steps 1–4. ✓
- Testing (var_key/dense/multi-contig/union/missing) → Task 3.1. ✓
- Docs (SKILL.md, CHANGELOG) → Task 3.2. ✓
- `_impl.py` SVAR1-lazy-load hazard (discovered) → Task 2.2. ✓

**Placeholder scan:** Phase 0 Task 0.3 Step 4 is intentionally compiler-driven ("fix each reported site as in Steps 1–3") — legitimate for a mechanical type migration, with the known edits shown concretely. All feature code steps show full code.

**Type consistency:** `decode_variants_from_split` new signature (Task 1.2 Interfaces) matches its call in Task 1.3 Step 2 and the no-field callers updated in Task 1.2 Step 3. FFI return tuple (Task 1.3) matches the Python unpack in Tasks 2.3/2.4. `store_fields` value shape `(category, name, dtype_str, np_dtype)` set in Task 2.1 matches use in 2.3/2.4. `FieldSub::all()` order asserted consistently (VkSnp,VkIndel,DenseSnp,DenseIndel) in Task 1.2 and 1.3.

## Known risks / verify-during-execution

- **`StorageDtype::from_meta_str` / `FieldView::open` / `FieldSub::all` exact signatures** — verify against genoray `src/field.rs`, `src/query/field.rs`, `src/layout.rs` at build time; the `Result` vs `Option` shape may need a `.map_err`/`.ok_or_else` tweak (Task 1.3 Step 1 notes this).
- **`ContigReader` has no `paths()` accessor** — we rebuild `ContigPaths::new(store.store_path(), contig)` (Task 1.1 + 1.3). Confirm `ContigPaths::new(base_out_dir, chrom)` is the correct constructor and that its `field_values(cat,name,sub)` layout matches what the writer produced.
- **`unphased_union` API name** — Task 3.1 uses `with_settings(unphased_union=True)`; confirm the actual gvl setter.
- **`DummyVariant` construction site for svar2 windows** — Task 2.4 Step 3 must locate where (if anywhere) `self.dummy_variant` is set for `Svar2Haps`; if unset, construct one carrying per-field fills.
