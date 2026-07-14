# genoray Read-Bound Per-Class Gather + Query-Only Build — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a htslib-free query-only build of genoray plus a read-bound, per-class dense gather that reconstructs the same `BatchResult` payload as today's union path **without** building the contig-wide `DenseUnion` (eliminating the O(N_contig) per-read residual), so GenVarLoader can link `genoray_core` as a query-only path-dep and reconstruct SVAR2 entirely in Rust.

**Architecture:** Two additive changes to the shipped `svar-2` search/gather split. (1) A `conversion` cargo feature (default-on) gates every htslib-touching module so `--no-default-features` compiles the read/query core alone. (2) `find_ranges` additionally emits per-class `dense_snp_range` / `dense_indel_range` (computed by a per-class `SearchTree` at search time), and a new `gather_ranges_readbound` slices each on-disk dense class window directly into a new split-dense `BatchResultSplit` — never calling `dense_union()`. The shipped `find_ranges` / `gather_ranges` / `read_ranges` / `overlap_batch` stay byte-unchanged as the parity oracle.

**Tech Stack:** Rust 2024, PyO3 0.29, `numpy` 0.29, `svar2-codec` (workspace member), `rust-htslib` (made optional), `cargo test` with `proptest` + `tempfile`.

**Repo:** `/carter/users/dlaub/projects/genoray` — branch `svar-2` (HEAD `7099f16`). Lib crate name is `genoray_core`. This is the **absolute** path GenVarLoader path-deps (there is no `../genoray` sibling checkout; the spec's `../genoray` is wrong).

## Global Constraints

- **Byte-identical parity contract.** For any `contig, regions, samples`, the read-bound path reconstructs the *same variants per hap* as the shipped union path and the `decode_hap` oracle — field-for-field. The split-dense `BatchResultSplit` merged with var_key equals `overlap_batch`'s union merged with var_key, per hap.
- **Additive.** `overlap_batch`, `find_ranges`, `gather_ranges`, `read_ranges`, `BatchResult`, `RangesBundle.dense_range`, and every existing Python dict key stay **byte-unchanged**. New code is new structs/functions/fields only. The full existing test suite (`tests/test_ranges_split.rs`, `tests/test_batch.rs`, `tests/test_decode_mat.rs`, `svar2-codec` proptests) stays green.
- **Query core is htslib-free.** After Task 1, `cargo build --no-default-features` and `cargo test --no-default-features` compile and pass without linking `rust-htslib`. The default (wheel) build is behavior-unchanged.
- **`rust-htslib` reach = `vcf_reader.rs` + `lib.rs:40-52`.** Only `vcf_reader.rs` uses htslib *types*; `lib.rs`'s `index_bcf_csi`/`index_vcf` are a second direct call site. Both must be gated, or `--no-default-features` fails to compile `lib.rs`.
- **`DenseView` + `carried()` live in `query.rs:120-136`**, not `dense.rs`. `dense.rs` holds only `DENSE_REGISTRY`/`DenseClass`/`DenseSpec`/`DenseMap`.
- **`decode_key` is `svar2_codec::decode_key`**, re-exported verbatim as `rvk::decode_key` (`rvk.rs:14`). Codec primitives used here: `rvk::snp_code_to_key`, `rvk::unpack_snp_key_at`, `rvk::deletion_len`.
- **Row/hap index conventions (unchanged):** `RangesBundle` per-hap row = `r * (n_samples*ploidy) + si*ploidy + p` where `si` is the *selected* sample slot and `sample_cols[si]` is the original sample index. `BatchResult` hap index = `(r*n_samples + s)*ploidy + p`, region-major.
- **Every Rust step:** `cargo test` compiles from source (no separate rebuild needed). Run `cargo test -p genoray_core` for the query core. Run `cargo fmt` + `cargo clippy --all-targets` before each commit; both must be clean.
- **Local-only.** No crates.io / PyPI publish in this plan. Task 6 builds a **local wheel** and confirms the crate builds for the downstream gvl path-dep; that wheel and path-dep MUST be the same commit.

---

## File Structure

- `Cargo.toml` — make `rust-htslib` optional; add `conversion` feature (default-on). *(Task 1)*
- `src/lib.rs` — `#[cfg(feature = "conversion")]` gates on htslib-touching modules + `index_bcf_csi`/`index_vcf`/`run_conversion_pipeline` + their `#[pymodule]` registrations. *(Task 1)*
- `src/query.rs` — add `dense_snp_overlap` / `dense_indel_overlap` methods, two new `RangesBundle` fields, `BatchResultSplit` struct, `gather_ranges_readbound` fn. *(Tasks 2, 3)*
- `src/py_query_ranges.rs` — add the two new range keys to `bundle_to_dict` / `bundle_from_dict`. *(Task 5)*
- `tests/test_readbound_gather.rs` — new parity + zero-union + per-class test file. *(Task 4)*
- `tests/common/mod.rs` — reused as-is (no change).
- `docs/roadmaps/*` (genoray-side roadmap) — mark the read-bound gather + conversion feature. *(Task 7)*

---

## Task 1: `conversion` cargo feature (htslib-free query core)

**Files:**
- Modify: `Cargo.toml` (`[dependencies]` `rust-htslib` line; `[features]` block)
- Modify: `src/lib.rs:5-32` (module decls), `src/lib.rs:40-52` (`index_bcf_csi`/`index_vcf`), `src/lib.rs:164-170` (`#[pymodule]`)

**Interfaces:**
- Produces: a `conversion` feature such that `default = ["conversion", "extension-module"]`; `cargo build --no-default-features` compiles the query core (`query`, `search`, `spine`, `bits`, `nrvk`, `rvk`, `layout`, `dense`, `types`, `error`, `cost_model`, `py_query*`) without `rust-htslib`.

- [ ] **Step 1: Write the failing build check**

Add this test to a new file `tests/test_query_only_build.rs`:

```rust
//! Compile-guard: the query core must build & link without the `conversion`
//! feature (no rust-htslib). If this file compiles under
//! `--no-default-features`, the gate is correct.
#[test]
fn query_core_symbols_are_reachable_without_conversion() {
    // Referencing these paths forces the query core to be part of the
    // no-default-features build graph.
    use genoray_core::query::{ContigReader, find_ranges, gather_ranges};
    let _ = ContigReader::open;
    let _ = find_ranges;
    let _ = gather_ranges;
}
```

- [ ] **Step 2: Run it to verify current state**

Run: `cargo test --no-default-features --test test_query_only_build 2>&1 | tail -30`
Expected: **FAIL to compile** — `rust-htslib` symbols in `vcf_reader.rs` / `lib.rs` are unconditionally in the build graph, and `pyo3/auto-initialize` (dev-dep) needs libpython. If it fails only on libpython linkage, add `--features pyo3/auto-initialize` is NOT wanted; instead confirm the failure mentions htslib/`vcf_reader`. Record the actual first error.

- [ ] **Step 3: Make `rust-htslib` optional + add the feature**

In `Cargo.toml`, change the `rust-htslib` dependency line to optional and add the feature. Replace:

```toml
rust-htslib = { version = "1.0", default-features = false }
```

with:

```toml
rust-htslib = { version = "1.0", default-features = false, optional = true }
```

and replace the `[features]` block:

```toml
[features]
default = ["extension-module"]
extension-module = ["pyo3/extension-module"]
```

with:

```toml
[features]
# `conversion` pulls in rust-htslib and gates the VCF→svar2 write/convert
# pipeline. Off => query-only core (what gvl links as a path-dep).
default = ["conversion", "extension-module"]
conversion = ["dep:rust-htslib"]
extension-module = ["pyo3/extension-module"]
```

- [ ] **Step 4: Gate the htslib-touching modules in `src/lib.rs`**

At `src/lib.rs:5-32`, add `#[cfg(feature = "conversion")]` above each of these `pub mod` lines (leave the query-core modules ungated): `vcf_reader`, `writer`, `orchestrator`, `normalize`, `budget`, `executor`, `monitor`, `streams`, `merge`, `max_del`, `dense_merge`, `meta`, `py_convert`. Also gate `pub use orchestrator::process_chromosome;`. Example:

```rust
#[cfg(feature = "conversion")]
pub mod vcf_reader;
#[cfg(feature = "conversion")]
pub mod writer;
// ... (repeat for the list above)
#[cfg(feature = "conversion")]
pub use orchestrator::process_chromosome;
```

Gate the two direct htslib call sites and the conversion pyfunctions at `src/lib.rs:40-52`:

```rust
#[cfg(feature = "conversion")]
fn index_bcf_csi(/* ...existing signature... */) { /* ...unchanged body... */ }

#[cfg(feature = "conversion")]
#[pyfunction]
fn index_vcf(/* ...existing... */) -> PyResult<()> { /* ...unchanged... */ }

#[cfg(feature = "conversion")]
#[pyfunction]
fn run_conversion_pipeline(/* ...existing... */) -> PyResult<()> { /* ...unchanged... */ }
```

And gate their registrations in the `#[pymodule]` at `src/lib.rs:164-170`:

```rust
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(run_conversion_pipeline, m)?)?;
    #[cfg(feature = "conversion")]
    m.add_function(wrap_pyfunction!(index_vcf, m)?)?;
    m.add_class::<crate::py_query::PyContigReader>()?;
    Ok(())
}
```

> If the compiler reports another module transitively pulling htslib (e.g. a query-core module that `use`s a now-gated module), gate the *offending `use`*, not the query-core module — the query core must stay ungated. Record any module you had to additionally gate in the commit message.

- [ ] **Step 5: Verify the query-only build compiles and passes**

Run: `cargo build --no-default-features 2>&1 | tail -20`
Expected: compiles clean (no `rust-htslib`).
Run: `cargo test --no-default-features --test test_query_only_build 2>&1 | tail -20`
Expected: **PASS**.

- [ ] **Step 6: Verify the default (wheel) build is unchanged**

Run: `cargo build 2>&1 | tail -5 && cargo test 2>&1 | tail -30`
Expected: full suite green (default features include `conversion`, so nothing else changed).

- [ ] **Step 7: fmt + clippy + commit**

Run: `cargo fmt && cargo clippy --all-targets --no-default-features 2>&1 | tail -20 && cargo clippy --all-targets 2>&1 | tail -20`
Expected: no warnings.

```bash
cd /carter/users/dlaub/projects/genoray
git add Cargo.toml src/lib.rs tests/test_query_only_build.rs
git commit -m "feat(query): conversion feature gates htslib; query core builds --no-default-features

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Per-class dense overlap + `find_ranges` emits `dense_snp_range` / `dense_indel_range`

**Files:**
- Modify: `src/query.rs` — add `dense_snp_overlap` / `dense_indel_overlap` methods to `impl ContigReader` (near `vk_snp_overlap` at `:608-651`); add two fields to `RangesBundle` (`:590-606`); populate them in `find_ranges` (`:657-701`).
- Test: `tests/test_readbound_gather.rs` (new; created here, extended in Task 4).

**Interfaces:**
- Consumes: `DenseView::positions()` and `DenseView::keys` on `reader.dense_snp` / `reader.dense_indel` (both `Option<DenseView>`); `reader.dense_indel_max_del: u32`; `rvk::deletion_len`, `rvk::unpack_snp_key_at`; `SearchTree::new`, `overlap_range` (already imported in `query.rs`).
- Produces:
  - `impl ContigReader { fn dense_snp_overlap(&self, q_start: u32, q_end: u32) -> (usize, usize); fn dense_indel_overlap(&self, q_start: u32, q_end: u32) -> (usize, usize); }` — absolute `[s,e)` into that class's on-disk dense positions/keys table; `(0,0)` when the class table is absent/empty.
  - `RangesBundle` gains `pub dense_snp_range: Vec<(usize, usize)>` and `pub dense_indel_range: Vec<(usize, usize)>`, each length `n_regions` (per-region, sample-independent — dense is cohort-shared).

- [ ] **Step 1: Write the failing test**

Create `tests/test_readbound_gather.rs` with the synth harness copied from `test_ranges_split.rs:55-79` and this first test:

```rust
//! Read-bound per-class gather: find_ranges emits per-class dense ranges and
//! gather_ranges_readbound replays them into BatchResultSplit without building
//! the contig-wide DenseUnion.
mod common;

use common::{SynthRecord, build_contig};
use genoray_core::query::{
    ContigReader, find_ranges, gather_ranges, gather_ranges_readbound, overlap_batch,
};
use genoray_core::search;
use tempfile::tempdir;

fn synth_reader(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord { pos: 100, ref_allele: b"A", alts: vec![&b"C"[..]], gt: vec![1, 0, 0, 0] },
        SynthRecord { pos: 200, ref_allele: b"A", alts: vec![&b"AT"[..]], gt: vec![0, 1, 1, 1] },
        SynthRecord { pos: 300, ref_allele: b"AT", alts: vec![&b"A"[..]], gt: vec![1, 1, 0, 1] },
    ];
    build_contig(out, "chr1", &samples, 2, &records);
    ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap()
}

#[test]
fn test_find_ranges_emits_per_class_dense_ranges() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let rb = find_ranges(&reader, &regions, None);
    // Both per-class range vectors are per-region (dense is cohort-shared).
    assert_eq!(rb.dense_snp_range.len(), regions.len());
    assert_eq!(rb.dense_indel_range.len(), regions.len());
    // Each per-class window is a subset of that class's table; ranges are valid.
    for &(s, e) in rb.dense_snp_range.iter().chain(rb.dense_indel_range.iter()) {
        assert!(s <= e);
    }
    // Region 0 spans the whole contig: it must see the one dense SNP (pos 100 is
    // var_key here, but the SNP class table is nonempty iff any SNP is dense) and
    // the dense indels. The union window must equal snp∪indel counts.
    let (us0, ue0) = rb.dense_range[0];
    let snp0 = rb.dense_snp_range[0].1 - rb.dense_snp_range[0].0;
    let indel0 = rb.dense_indel_range[0].1 - rb.dense_indel_range[0].0;
    assert_eq!(ue0 - us0, snp0 + indel0,
        "union window size must equal sum of per-class window sizes");
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --test test_readbound_gather test_find_ranges_emits_per_class_dense_ranges 2>&1 | tail -20`
Expected: **FAIL to compile** — `gather_ranges_readbound` unresolved (added Task 3) and `rb.dense_snp_range` unknown field.

> To iterate on Task 2 alone before Task 3 exists, temporarily comment the `gather_ranges_readbound` import; restore it in Task 3.

- [ ] **Step 3: Add the two `RangesBundle` fields**

In `src/query.rs`, extend the `RangesBundle` struct (`:590-606`) — append after `vk_indel_range`:

```rust
    /// `[s, e)` into `dense/snp`'s on-disk positions/keys, per region (dense is
    /// cohort-shared, so one window per region, not per hap). Read-bound path.
    pub dense_snp_range: Vec<(usize, usize)>,
    /// `[s, e)` into `dense/indel`'s on-disk positions/keys, per region.
    pub dense_indel_range: Vec<(usize, usize)>,
```

- [ ] **Step 4: Add the per-class overlap methods**

In `src/query.rs`, inside `impl ContigReader` (right after `vk_indel_overlap` ends at `:651`), add:

```rust
    /// Absolute `[s, e)` into `dense/snp`'s positions/keys for one region.
    /// SNP v_end = pos + 1 (max_region_length = 0). `(0, 0)` if no snp table.
    fn dense_snp_overlap(&self, q_start: u32, q_end: u32) -> (usize, usize) {
        let d = match &self.dense_snp {
            Some(d) => d,
            None => return (0, 0),
        };
        let positions = d.positions();
        if positions.is_empty() {
            return (0, 0);
        }
        let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        let tree = SearchTree::new(positions);
        overlap_range(&tree, &v_ends, 0, q_start, q_end)
    }

    /// Absolute `[s, e)` into `dense/indel`'s positions/keys for one region.
    /// Indel v_end = pos + 1 + deletion_len(key); per-contig dense max_del bound.
    fn dense_indel_overlap(&self, q_start: u32, q_end: u32) -> (usize, usize) {
        let d = match &self.dense_indel {
            Some(d) => d,
            None => return (0, 0),
        };
        let positions = d.positions();
        if positions.is_empty() {
            return (0, 0);
        }
        let keys = as_u32(&d.keys);
        debug_assert_eq!(positions.len(), keys.len());
        let v_ends: Vec<u32> = positions
            .iter()
            .zip(keys.iter())
            .map(|(&pos, &key)| pos + 1 + rvk::deletion_len(key))
            .collect();
        let tree = SearchTree::new(positions);
        overlap_range(&tree, &v_ends, self.dense_indel_max_del, q_start, q_end)
    }
```

- [ ] **Step 5: Populate the fields in `find_ranges`**

In `find_ranges` (`:657-701`), after the existing `dense_range` / `region_starts` computation (`:672-677`), add:

```rust
    let dense_snp_range: Vec<(usize, usize)> = regions
        .iter()
        .map(|&(qs, qe)| reader.dense_snp_overlap(qs, qe))
        .collect();
    let dense_indel_range: Vec<(usize, usize)> = regions
        .iter()
        .map(|&(qs, qe)| reader.dense_indel_overlap(qs, qe))
        .collect();
```

and add the two fields to the returned `RangesBundle { ... }` literal (`:691-700`):

```rust
        dense_snp_range,
        dense_indel_range,
```

- [ ] **Step 6: Fix the other `RangesBundle` construction site**

`find_ranges` is the only constructor, but `cargo build` will confirm. Run: `cargo build 2>&1 | tail -20`. If any other `RangesBundle { ... }` literal errors on the missing fields, add the two fields there too.

- [ ] **Step 7: Run the per-class range test (isolate it)**

Temporarily comment the `gather_ranges_readbound` import + the not-yet-written test bodies so only `test_find_ranges_emits_per_class_dense_ranges` compiles. Run:
`cargo test --test test_readbound_gather test_find_ranges_emits_per_class_dense_ranges 2>&1 | tail -20`
Expected: **PASS**.

- [ ] **Step 8: Confirm the shipped path is byte-unchanged**

Run: `cargo test --test test_ranges_split 2>&1 | tail -20`
Expected: all existing split tests green (we only *added* fields + methods).

- [ ] **Step 9: fmt + clippy + commit**

```bash
cargo fmt && cargo clippy --all-targets 2>&1 | tail -20
git add src/query.rs tests/test_readbound_gather.rs
git commit -m "feat(query): find_ranges emits per-class dense_snp_range/dense_indel_range

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `BatchResultSplit` + `gather_ranges_readbound` (no `dense_union`)

**Files:**
- Modify: `src/query.rs` — add `BatchResultSplit` struct (near `BatchResult` at `:488-504`); add `pub fn gather_ranges_readbound` (after `gather_ranges` at `:711-811`).
- Test: `tests/test_readbound_gather.rs` (Task 4 asserts parity).

**Interfaces:**
- Consumes: `RangesBundle` (now with `dense_snp_range` / `dense_indel_range` from Task 2); `reader.vk_snp` / `reader.vk_indel` packed positions/keys; `reader.dense_snp` / `reader.dense_indel` (`DenseView`, with `.positions()`, `.keys`, `.carried(hap, col)`); `spine::merge_keys`; `bits::set_bit`; `rvk::{snp_code_to_key, unpack_snp_key_at, deletion_len}`; `as_bytes`, `as_u32`, `KeyRef`.
- Produces:
  - `pub fn gather_ranges_readbound(reader: &ContigReader, rb: &RangesBundle) -> BatchResultSplit` — cartesian R×S'; the parity-test vehicle. Builds **zero** `SearchTree`, never calls `reader.dense_union()`.
  - `pub fn gather_haps_readbound(reader, region_starts, orig_samples, vk_snp_range, vk_indel_range, dense_snp_range, dense_indel_range, ploidy) -> BatchResultSplit` — **flat per-query** (one `(region, sample)` per query row); the primitive gvl links and calls (Plan 2, Task 4). `n_samples = 1`, hap index `q*ploidy + p`.
- `BatchResultSplit` fields (var_key merged per hap, dense split by class):
  ```rust
  pub struct BatchResultSplit {
      pub n_regions: usize, pub n_samples: usize, pub ploidy: usize,
      pub vk: Vec<KeyRef>, pub vk_off: Vec<usize>,
      pub dense_snp: Vec<KeyRef>, pub dense_snp_range: Vec<(usize, usize)>,
      pub dense_snp_present: Vec<u8>, pub dense_snp_present_off: Vec<usize>,
      pub dense_indel: Vec<KeyRef>, pub dense_indel_range: Vec<(usize, usize)>,
      pub dense_indel_present: Vec<u8>, pub dense_indel_present_off: Vec<usize>,
  }
  ```
  Per-hap presence bitmask is over that class's per-region window `[ds..de)`, LSB-first; `*_present_off` (len H+1) holds **bit** offsets. `H = n_regions * n_samples * ploidy`, hap index `(r*n_samples+s)*ploidy+p` over the *selected* samples.

- [ ] **Step 1: Add the `BatchResultSplit` struct**

In `src/query.rs`, after the `BatchResult` struct (`:504`), add:

```rust
/// Read-bound analog of `BatchResult`: the var_key channel merged per hap (as
/// today), but the dense channel **split per class** so no contig-wide
/// `DenseUnion` is built. gvl merges `var_key ⋈ dense_snp ⋈ dense_indel` by
/// position downstream. `H = n_regions * n_samples * ploidy`, hap index
/// `(r*n_samples + s)*ploidy + p` over the *selected* samples.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchResultSplit {
    pub n_regions: usize,
    pub n_samples: usize,
    pub ploidy: usize,
    /// Flat merged var_key channel (snp+indel per hap); `vk_off` (len H+1) slices it.
    pub vk: Vec<KeyRef>,
    pub vk_off: Vec<usize>,
    /// Per-region `dense/snp` windows (uniform keys), concatenated.
    pub dense_snp: Vec<KeyRef>,
    /// `[s, e)` into `dense_snp` per region (len n_regions).
    pub dense_snp_range: Vec<(usize, usize)>,
    /// Per-hap presence bitmask over that region's `dense_snp[s..e]`, LSB-first;
    /// `dense_snp_present_off` (len H+1) holds BIT offsets.
    pub dense_snp_present: Vec<u8>,
    pub dense_snp_present_off: Vec<usize>,
    /// Per-region `dense/indel` windows (uniform u32 keys), concatenated.
    pub dense_indel: Vec<KeyRef>,
    pub dense_indel_range: Vec<(usize, usize)>,
    pub dense_indel_present: Vec<u8>,
    pub dense_indel_present_off: Vec<usize>,
}
```

- [ ] **Step 2: Write `gather_ranges_readbound`**

In `src/query.rs`, after `gather_ranges` (`:811`), add. This mirrors `gather_ranges`'s var_key gather verbatim, and replaces the single union presence loop with two per-class window loops that read positions/keys straight from each on-disk dense table (no `dense_union()`):

```rust
/// Tree-free, union-free gather: replay a `RangesBundle` into a split-dense
/// `BatchResultSplit`. Builds NO `SearchTree` and never calls `dense_union()` —
/// each region's dense windows come from the per-class `dense_snp_range` /
/// `dense_indel_range` computed in `find_ranges`. The var_key channel is
/// identical to `gather_ranges`; only the dense side is split per class.
pub fn gather_ranges_readbound(reader: &ContigReader, rb: &RangesBundle) -> BatchResultSplit {
    let ploidy = rb.ploidy;
    let n_samples = rb.n_samples;
    let n_regions = rb.n_regions;
    let hpr = n_samples * ploidy;

    let snp_positions = reader.vk_snp.positions();
    let snp_keys = as_bytes(&reader.vk_snp.keys);
    let indel_positions = reader.vk_indel.positions();
    let indel_keys = as_u32(&reader.vk_indel.keys);

    // Dense class tables (may be absent).
    let d_snp = reader.dense_snp.as_ref();
    let d_indel = reader.dense_indel.as_ref();
    let d_snp_pos: &[u32] = d_snp.map(|d| d.positions()).unwrap_or(&[]);
    let d_indel_pos: &[u32] = d_indel.map(|d| d.positions()).unwrap_or(&[]);

    // --- dense channel windows (per region), decoded to uniform keys once ---
    let mut dense_snp: Vec<KeyRef> = Vec::new();
    let mut dense_snp_range: Vec<(usize, usize)> = Vec::with_capacity(n_regions);
    let mut dense_indel: Vec<KeyRef> = Vec::new();
    let mut dense_indel_range: Vec<(usize, usize)> = Vec::with_capacity(n_regions);
    for r in 0..n_regions {
        let (ss, se) = rb.dense_snp_range[r];
        let base = dense_snp.len();
        if let Some(d) = d_snp {
            let keys = as_bytes(&d.keys);
            for j in ss..se {
                dense_snp.push(KeyRef {
                    position: d_snp_pos[j],
                    key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, j)),
                });
            }
        }
        dense_snp_range.push((base, dense_snp.len()));

        let (is_, ie_) = rb.dense_indel_range[r];
        let base = dense_indel.len();
        if let Some(d) = d_indel {
            let keys = as_u32(&d.keys);
            for j in is_..ie_ {
                dense_indel.push(KeyRef {
                    position: d_indel_pos[j],
                    key: keys[j],
                });
            }
        }
        dense_indel_range.push((base, dense_indel.len()));
    }

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut dense_snp_present: Vec<u8> = Vec::new();
    let mut dense_snp_present_off: Vec<usize> = vec![0];
    let mut dense_indel_present: Vec<u8> = Vec::new();
    let mut dense_indel_present_off: Vec<usize> = vec![0];

    for r in 0..n_regions {
        let qs = rb.region_starts[r];
        let (ss, se) = rb.dense_snp_range[r];
        let (is_r, ie_r) = rb.dense_indel_range[r];
        for si in 0..n_samples {
            let orig_s = rb.sample_cols[si];
            for p in 0..ploidy {
                let col = orig_s * ploidy + p;
                let hap = col;
                let row = r * hpr + si * ploidy + p;

                // --- var_key gather (identical to gather_ranges) ---
                let (vs, ve) = rb.vk_snp_range[row];
                let mut snp_run: Vec<KeyRef> = Vec::new();
                for (j, &pos) in snp_positions.iter().enumerate().take(ve).skip(vs) {
                    if qs < pos + 1 {
                        snp_run.push(KeyRef {
                            position: pos,
                            key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                        });
                    }
                }
                let (vis, vie) = rb.vk_indel_range[row];
                let mut indel_run: Vec<KeyRef> = Vec::new();
                for j in vis..vie {
                    let pos = indel_positions[j];
                    let v_end = pos + 1 + rvk::deletion_len(indel_keys[j]);
                    if qs < v_end {
                        indel_run.push(KeyRef { position: pos, key: indel_keys[j] });
                    }
                }
                let merged = spine::merge_keys(vec![snp_run, indel_run]);
                vk.extend_from_slice(&merged);
                vk_off.push(vk.len());

                // --- dense/snp presence bits over [ss..se) ---
                let nbits = se - ss;
                let bit_base = *dense_snp_present_off.last().unwrap();
                let need = (bit_base + nbits).div_ceil(8);
                if dense_snp_present.len() < need {
                    dense_snp_present.resize(need, 0);
                }
                if let Some(d) = d_snp {
                    for (k, j) in (ss..se).enumerate() {
                        // snp v_end = pos + 1; left-overlap re-check qs < v_end.
                        if d.carried(hap, j) && qs < d_snp_pos[j] + 1 {
                            bits::set_bit(&mut dense_snp_present, bit_base + k);
                        }
                    }
                }
                dense_snp_present_off.push(bit_base + nbits);

                // --- dense/indel presence bits over [is_r..ie_r) ---
                let nbits = ie_r - is_r;
                let bit_base = *dense_indel_present_off.last().unwrap();
                let need = (bit_base + nbits).div_ceil(8);
                if dense_indel_present.len() < need {
                    dense_indel_present.resize(need, 0);
                }
                if let Some(d) = d_indel {
                    let keys = as_u32(&d.keys);
                    for (k, j) in (is_r..ie_r).enumerate() {
                        let v_end = d_indel_pos[j] + 1 + rvk::deletion_len(keys[j]);
                        if d.carried(hap, j) && qs < v_end {
                            bits::set_bit(&mut dense_indel_present, bit_base + k);
                        }
                    }
                }
                dense_indel_present_off.push(bit_base + nbits);
            }
        }
    }

    BatchResultSplit {
        n_regions,
        n_samples,
        ploidy,
        vk,
        vk_off,
        dense_snp,
        dense_snp_range,
        dense_snp_present,
        dense_snp_present_off,
        dense_indel,
        dense_indel_range,
        dense_indel_present,
        dense_indel_present_off,
    }
}
```

> **Presence-bit indexing note:** `d.carried(hap, col)` addresses the *global* per-class dense column, and here `j` (the absolute on-disk row inside `[ss..se)`) **is** that global column, because `dense_snp_range` / `dense_indel_range` are absolute indices into the class table. This is the read-bound simplification: the union path had to remap through `dense.src[j] = (is_indel, dcol)`; per-class, `j` is already `dcol`.

- [ ] **Step 3: Add the flat per-query gather `gather_haps_readbound` (gvl's read primitive)**

`gather_ranges_readbound` is cartesian R×S' and is the parity-test vehicle (easy to compare to `overlap_batch`). But gvl reads an **arbitrary set of `(region, sample)` pairs** — one query row each, exactly like SVAR1's `geno_offset_idx` — so it needs a *flat per-query* primitive where each query carries its own original sample (dense carriage needs `hap = orig_sample*ploidy + p`). Add, after `gather_ranges_readbound`:

```rust
/// Flat per-query read-bound gather for gvl's arbitrary-(region,sample) reads.
/// Each of `n_q = region_starts.len()` queries is one (region, sample) pair
/// reconstructing `ploidy` haps. Range arrays are per-query (`dense_*_range`,
/// length n_q) or per-(query,ploid) (`vk_*_range`, length n_q*ploidy, row =
/// q*ploidy + p). Builds zero SearchTrees and never calls `dense_union()`.
/// Returns a `BatchResultSplit` with `n_samples = 1`, hap index `q*ploidy + p`.
pub fn gather_haps_readbound(
    reader: &ContigReader,
    region_starts: &[u32],
    orig_samples: &[usize],
    vk_snp_range: &[(usize, usize)],
    vk_indel_range: &[(usize, usize)],
    dense_snp_range: &[(usize, usize)],
    dense_indel_range: &[(usize, usize)],
    ploidy: usize,
) -> BatchResultSplit {
    let n_q = region_starts.len();
    assert_eq!(orig_samples.len(), n_q);
    assert_eq!(dense_snp_range.len(), n_q);
    assert_eq!(dense_indel_range.len(), n_q);
    assert_eq!(vk_snp_range.len(), n_q * ploidy);
    assert_eq!(vk_indel_range.len(), n_q * ploidy);

    let snp_positions = reader.vk_snp.positions();
    let snp_keys = as_bytes(&reader.vk_snp.keys);
    let indel_positions = reader.vk_indel.positions();
    let indel_keys = as_u32(&reader.vk_indel.keys);
    let d_snp = reader.dense_snp.as_ref();
    let d_indel = reader.dense_indel.as_ref();
    let d_snp_pos: &[u32] = d_snp.map(|d| d.positions()).unwrap_or(&[]);
    let d_indel_pos: &[u32] = d_indel.map(|d| d.positions()).unwrap_or(&[]);

    // Dense windows per query (uniform keys), decoded once.
    let mut dense_snp: Vec<KeyRef> = Vec::new();
    let mut dense_snp_range_out: Vec<(usize, usize)> = Vec::with_capacity(n_q);
    let mut dense_indel: Vec<KeyRef> = Vec::new();
    let mut dense_indel_range_out: Vec<(usize, usize)> = Vec::with_capacity(n_q);
    for q in 0..n_q {
        let (ss, se) = dense_snp_range[q];
        let base = dense_snp.len();
        if let Some(d) = d_snp {
            let keys = as_bytes(&d.keys);
            for j in ss..se {
                dense_snp.push(KeyRef {
                    position: d_snp_pos[j],
                    key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, j)),
                });
            }
        }
        dense_snp_range_out.push((base, dense_snp.len()));
        let (is_, ie_) = dense_indel_range[q];
        let base = dense_indel.len();
        if let Some(d) = d_indel {
            let keys = as_u32(&d.keys);
            for j in is_..ie_ {
                dense_indel.push(KeyRef { position: d_indel_pos[j], key: keys[j] });
            }
        }
        dense_indel_range_out.push((base, dense_indel.len()));
    }

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut dense_snp_present: Vec<u8> = Vec::new();
    let mut dense_snp_present_off: Vec<usize> = vec![0];
    let mut dense_indel_present: Vec<u8> = Vec::new();
    let mut dense_indel_present_off: Vec<usize> = vec![0];

    for q in 0..n_q {
        let qs = region_starts[q];
        let orig_s = orig_samples[q];
        let (ss, se) = dense_snp_range[q];
        let (is_r, ie_r) = dense_indel_range[q];
        for p in 0..ploidy {
            let hap = orig_s * ploidy + p;
            let row = q * ploidy + p;

            // var_key gather.
            let (vs, ve) = vk_snp_range[row];
            let mut snp_run: Vec<KeyRef> = Vec::new();
            for (j, &pos) in snp_positions.iter().enumerate().take(ve).skip(vs) {
                if qs < pos + 1 {
                    snp_run.push(KeyRef {
                        position: pos,
                        key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(snp_keys, j)),
                    });
                }
            }
            let (vis, vie) = vk_indel_range[row];
            let mut indel_run: Vec<KeyRef> = Vec::new();
            for j in vis..vie {
                let pos = indel_positions[j];
                let v_end = pos + 1 + rvk::deletion_len(indel_keys[j]);
                if qs < v_end {
                    indel_run.push(KeyRef { position: pos, key: indel_keys[j] });
                }
            }
            vk.extend_from_slice(&spine::merge_keys(vec![snp_run, indel_run]));
            vk_off.push(vk.len());

            // dense/snp presence over [ss..se).
            let nbits = se - ss;
            let bit_base = *dense_snp_present_off.last().unwrap();
            let need = (bit_base + nbits).div_ceil(8);
            if dense_snp_present.len() < need { dense_snp_present.resize(need, 0); }
            if let Some(d) = d_snp {
                for (k, j) in (ss..se).enumerate() {
                    if d.carried(hap, j) && qs < d_snp_pos[j] + 1 {
                        bits::set_bit(&mut dense_snp_present, bit_base + k);
                    }
                }
            }
            dense_snp_present_off.push(bit_base + nbits);

            // dense/indel presence over [is_r..ie_r).
            let nbits = ie_r - is_r;
            let bit_base = *dense_indel_present_off.last().unwrap();
            let need = (bit_base + nbits).div_ceil(8);
            if dense_indel_present.len() < need { dense_indel_present.resize(need, 0); }
            if let Some(d) = d_indel {
                let keys = as_u32(&d.keys);
                for (k, j) in (is_r..ie_r).enumerate() {
                    let v_end = d_indel_pos[j] + 1 + rvk::deletion_len(keys[j]);
                    if d.carried(hap, j) && qs < v_end {
                        bits::set_bit(&mut dense_indel_present, bit_base + k);
                    }
                }
            }
            dense_indel_present_off.push(bit_base + nbits);
        }
    }

    BatchResultSplit {
        n_regions: n_q,
        n_samples: 1,
        ploidy,
        vk,
        vk_off,
        dense_snp,
        dense_snp_range: dense_snp_range_out,
        dense_snp_present,
        dense_snp_present_off,
        dense_indel,
        dense_indel_range: dense_indel_range_out,
        dense_indel_present,
        dense_indel_present_off,
    }
}
```

- [ ] **Step 4: Build**

Run: `cargo build 2>&1 | tail -20`
Expected: compiles (unused-warnings fine until Task 4 references both functions).

- [ ] **Step 5: fmt + clippy + commit**

```bash
cargo fmt && cargo clippy --all-targets 2>&1 | tail -20
git add src/query.rs
git commit -m "feat(query): gather_ranges_readbound + gather_haps_readbound + BatchResultSplit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Parity + zero-union + per-class tests

**Files:**
- Modify: `tests/test_readbound_gather.rs` (add the parity/zero-union tests).

**Interfaces:**
- Consumes: `gather_ranges_readbound`, `find_ranges`, `overlap_batch`, `read_ranges`, `BatchResult::decode_hap`, `search::search_tree_build_count`.

**Parity strategy (byte-identical contract).** `BatchResultSplit` has a different *shape* than `BatchResult` (dense split, not unioned), so we assert parity at the **decoded-variants** level: for every `(r, s, p)`, the set of merged `(position, key)` from the read-bound result equals the set from `overlap_batch`'s union path. We build a local `readbound_decode_hap` helper that merges `vk ⋈ dense_snp ⋈ dense_indel` (mirroring how gvl will), then compare against `BatchResult::decode_hap` (the shipped oracle).

- [ ] **Step 1: Add the read-bound decode helper + parity test**

Append to `tests/test_readbound_gather.rs`:

```rust
use genoray_core::query::{BatchResultSplit, HapCalls, decode_keyref_pub}; // see note below

/// Merge vk ⋈ dense_snp ⋈ dense_indel for one hap and decode — the gvl-side
/// reconstruction, expressed as a test oracle.
fn readbound_decode_hap(
    br: &BatchResultSplit,
    reader: &ContigReader,
    r: usize,
    s: usize,
    p: usize,
) -> Vec<(u32, i32)> {
    use genoray_core::query::KeyRef;
    let h = (r * br.n_samples + s) * br.ploidy + p;
    let mut merged: Vec<KeyRef> = br.vk[br.vk_off[h]..br.vk_off[h + 1]].to_vec();

    let (ss, se) = br.dense_snp_range[r];
    let bit0 = br.dense_snp_present_off[h];
    for (k, j) in (ss..se).enumerate() {
        if genoray_core::bits_get_bit(&br.dense_snp_present, bit0 + k) {
            merged.push(br.dense_snp[j]);
        }
    }
    let (is_, ie_) = br.dense_indel_range[r];
    let bit0 = br.dense_indel_present_off[h];
    for (k, j) in (is_..ie_).enumerate() {
        if genoray_core::bits_get_bit(&br.dense_indel_present, bit0 + k) {
            merged.push(br.dense_indel[j]);
        }
    }
    // Stable position sort (var_key already ahead of dense within its own run).
    merged.sort_by_key(|kr| kr.position);
    merged.into_iter().map(|kr| (kr.position, kr.key as i32)).collect()
}

#[test]
fn test_readbound_reconstructs_union_per_hap() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32), (150u32, 250u32)];

    let oracle = overlap_batch(&reader, &regions);
    let rb = find_ranges(&reader, &regions, None);
    let got = gather_ranges_readbound(&reader, &rb);

    assert_eq!(got.n_regions, oracle.n_regions);
    assert_eq!(got.n_samples, oracle.n_samples);
    assert_eq!(got.ploidy, oracle.ploidy);

    for r in 0..oracle.n_regions {
        for s in 0..oracle.n_samples {
            for p in 0..oracle.ploidy {
                // Oracle: decode via the shipped union decode_hap, keep (pos, key).
                let hc: HapCalls = oracle.decode_hap(&reader, r, s, p);
                // decode_hap returns decoded alts, not raw keys — compare on the
                // (position, ilen) projection that survives decode instead.
                let want: Vec<(u32, i32)> =
                    hc.positions.iter().zip(hc.ilens.iter()).map(|(&a, &b)| (a, b)).collect();
                let got_keys = readbound_decode_hap(&got, &reader, r, s, p);
                // Decode the read-bound raw keys the same way to get ilens.
                let got_dec: Vec<(u32, i32)> = got_keys
                    .iter()
                    .map(|&(pos, key)| (pos, decode_keyref_pub(pos, key as u32, &reader)))
                    .collect();
                assert_eq!(got_dec, want, "hap (r={r}, s={s}, p={p})");
            }
        }
    }
}
```

> **Helper exports needed.** This test references three items that must be `pub` in genoray. In `src/query.rs`: make `decode_keyref` reachable via a thin public wrapper `pub fn decode_keyref_pub(position: u32, key: u32, reader: &ContigReader) -> i32` that builds a `KeyRef { position, key }`, calls the existing `decode_keyref(kr, reader.lut.as_ref())`, and returns `.ilen`. In `src/lib.rs` add `pub fn bits_get_bit(bytes: &[u8], i: usize) -> bool { bits::get_bit(bytes, i) }` (a re-export shim; `bits` is already `pub mod`). Also `pub use query::{BatchResultSplit, KeyRef, HapCalls};` if not already public — `HapCalls` (`:435`) and `KeyRef` are already `pub`. Add these shims in the same Task 3/Task 4 commit.

- [ ] **Step 2: Run the parity test**

Run: `cargo test --test test_readbound_gather test_readbound_reconstructs_union_per_hap 2>&1 | tail -30`
Expected: **PASS**. If a hap mismatches, the failure prints `(r, s, p)` — debug the per-class window vs. union window for that region (usually a `qs < v_end` re-check discrepancy).

- [ ] **Step 3: Add the zero-union / zero-tree assertion**

Append:

```rust
#[test]
fn test_readbound_gather_builds_no_search_tree() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let rb = find_ranges(&reader, &regions, None);
    let before = search::search_tree_build_count();
    let _ = gather_ranges_readbound(&reader, &rb);
    assert_eq!(
        search::search_tree_build_count(),
        before,
        "gather_ranges_readbound must build zero SearchTrees (no dense_union)"
    );
}
```

- [ ] **Step 4: Add a sample-subset parity test (mirrors the `read_ranges` subset oracle)**

Append:

```rust
#[test]
fn test_readbound_subset_matches_full_selected_haps() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out); // 2 samples, ploidy 2
    let regions = vec![(0u32, 1_000_000u32)];

    let full = gather_ranges_readbound(&reader, &find_ranges(&reader, &regions, None));
    // Select only sample 1.
    let sub = gather_ranges_readbound(&reader, &find_ranges(&reader, &regions, Some(&[1])));
    assert_eq!(sub.n_samples, 1);
    for p in 0..reader_ploidy(&reader) {
        let a = readbound_decode_hap(&sub, &reader, 0, 0, p);   // selected slot 0 == orig sample 1
        let b = readbound_decode_hap(&full, &reader, 0, 1, p);  // orig sample 1
        assert_eq!(a, b, "subset ploid {p}");
    }
}

fn reader_ploidy(_r: &ContigReader) -> usize { 2 }
```

- [ ] **Step 5: Add the flat-gather parity test (flat ≡ cartesian for full cohort)**

Append — proves `gather_haps_readbound` (gvl's primitive) agrees with the cartesian `gather_ranges_readbound` when the flat queries enumerate the full R×S' cohort:

```rust
#[test]
fn test_flat_gather_matches_cartesian_full_cohort() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out); // 2 samples, ploidy 2
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];
    let ploidy = 2usize;

    let rb = find_ranges(&reader, &regions, None);
    let cart = gather_ranges_readbound(&reader, &rb);

    // Enumerate flat queries in the SAME order cart lays out haps:
    // region-major, samples 0..S, so query q = r*S + s, orig sample = s.
    let s_n = rb.n_samples;
    let mut region_starts = Vec::new();
    let mut orig_samples = Vec::new();
    let mut vk_snp_range = Vec::new();
    let mut vk_indel_range = Vec::new();
    let mut dsr = Vec::new();
    let mut dir_ = Vec::new();
    for r in 0..regions.len() {
        for s in 0..s_n {
            region_starts.push(rb.region_starts[r]);
            orig_samples.push(rb.sample_cols[s]);
            dsr.push(rb.dense_snp_range[r]);
            dir_.push(rb.dense_indel_range[r]);
            for p in 0..ploidy {
                let row = r * (s_n * ploidy) + s * ploidy + p;
                vk_snp_range.push(rb.vk_snp_range[row]);
                vk_indel_range.push(rb.vk_indel_range[row]);
            }
        }
    }
    let flat = gather_haps_readbound(
        &reader, &region_starts, &orig_samples,
        &vk_snp_range, &vk_indel_range, &dsr, &dir_, ploidy,
    );

    // Compare decoded per-hap. cart hap (r,s,p) == flat query q=r*S+s, ploid p.
    for r in 0..regions.len() {
        for s in 0..s_n {
            for p in 0..ploidy {
                let a = readbound_decode_hap(&cart, &reader, r, s, p);
                let b = readbound_decode_hap(&flat, &reader, r * s_n + s, 0, p);
                assert_eq!(a, b, "flat vs cartesian (r={r}, s={s}, p={p})");
            }
        }
    }
}
```

- [ ] **Step 6: Run the full new test file**

Run: `cargo test --test test_readbound_gather 2>&1 | tail -30`
Expected: all five tests PASS.

- [ ] **Step 7: Run the full suite (additive guarantee)**

Run: `cargo test 2>&1 | tail -40`
Expected: entire genoray suite green — the shipped union path and every existing test unchanged.

- [ ] **Step 8: fmt + clippy + commit**

```bash
cargo fmt && cargo clippy --all-targets 2>&1 | tail -20
git add src/query.rs src/lib.rs tests/test_readbound_gather.rs
git commit -m "test(query): read-bound gather parity vs union+decode, zero-tree, subset

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Python surface — `find_ranges` dict exposes the per-class ranges

**Why:** gvl's **write** path calls genoray's Python `find_ranges(...)` and streams the resulting arrays into the cache (Plan 2, Task 2). The read path is pure-Rust and needs no Python surface, but the write cache needs `dense_snp_range` / `dense_indel_range` in the `find_ranges` dict.

**Files:**
- Modify: `src/py_query_ranges.rs` — `bundle_to_dict` (`:~28`) and `bundle_from_dict` (`:~120`).

**Interfaces:**
- Produces: the Python `find_ranges(...)` / `read_ranges(...)` dict gains keys `dense_snp_range` and `dense_indel_range`, each an `(R, 2)` `int64` numpy array. Existing keys byte-unchanged.

- [ ] **Step 1: Write the failing Python parity test**

Create `tests/test_py_ranges_readbound.py` (run via the built wheel — this is a Python test, deferred to after Task 6's wheel build; write it now, run it in Task 6):

```python
import numpy as np
# genoray._core.PyContigReader is constructed the same way the existing
# py_query_ranges tests do; reuse that harness path if one exists in genoray's
# python test suite. Placeholder assertion of the new keys:
def _assert_keys(d):
    assert "dense_snp_range" in d and "dense_indel_range" in d
    for k in ("dense_snp_range", "dense_indel_range"):
        a = np.asarray(d[k])
        assert a.ndim == 2 and a.shape[1] == 2 and a.dtype == np.int64
```

- [ ] **Step 2: Add the keys to `bundle_to_dict`**

In `src/py_query_ranges.rs`, in `bundle_to_dict`, next to where `dense_range` is inserted, add (mirror the exact `(R,2)` i64 conversion used for `dense_range`):

```rust
    let snp = PyArray2::from_vec2(
        py,
        &rb.dense_snp_range.iter().map(|&(s, e)| vec![s as i64, e as i64]).collect::<Vec<_>>(),
    )?;
    dict.set_item("dense_snp_range", snp)?;
    let indel = PyArray2::from_vec2(
        py,
        &rb.dense_indel_range.iter().map(|&(s, e)| vec![s as i64, e as i64]).collect::<Vec<_>>(),
    )?;
    dict.set_item("dense_indel_range", indel)?;
```

> Match the *existing* `dense_range` serialization idiom in this file exactly (it may use `PyArray2::from_owned_array` over an `Array2`); if so, build `Array2::from_shape_vec((R, 2), flat)` the same way rather than `from_vec2`. Read the `dense_range` block first and copy its shape/dtype path.

- [ ] **Step 3: Add the keys to `bundle_from_dict`**

If `bundle_from_dict` round-trips (used by the Rust dict-parity test), parse the two new keys back into `Vec<(usize,usize)>` mirroring `dense_range`'s parse. If `bundle_from_dict` is only used for the union oracle and ignores unknown keys, guard the parse with `if let Some(...) = dict.get_item(...)` so old dicts still load.

- [ ] **Step 4: Build + confirm the Rust dict-parity test still passes**

Run: `cargo test --test test_ranges_split 2>&1 | tail -20`
Expected: `assert_payload_dicts_eq` still green (it checks only the original key set; new keys are additive).

- [ ] **Step 5: fmt + clippy + commit**

```bash
cargo fmt && cargo clippy --all-targets 2>&1 | tail -20
git add src/py_query_ranges.rs tests/test_py_ranges_readbound.py
git commit -m "feat(py): find_ranges dict exposes dense_snp_range/dense_indel_range

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Build the local wheel + confirm downstream crate build

**Files:** none (build/verification only).

- [ ] **Step 1: Build the wheel with default features (conversion on)**

Run (in the genoray pixi/venv used to build the wheel):
`cd /carter/users/dlaub/projects/genoray && pixi run -e dev maturin develop --release 2>&1 | tail -20` *(use genoray's actual build task; if it uses `maturin build`, produce the wheel and note its path)*
Expected: wheel builds; `python -c "import genoray"` works.

- [ ] **Step 2: Run the Python range test from Task 5**

Run: `pytest tests/test_py_ranges_readbound.py -q 2>&1 | tail -20`
Expected: PASS (new dict keys present, correct shape/dtype).

- [ ] **Step 3: Confirm the query-only crate builds for the downstream path-dep**

Run: `cargo build --no-default-features --release 2>&1 | tail -10`
Expected: clean — this is exactly what gvl's `genoray_core = { path = ..., default-features = false }` compiles (Plan 2, Task 3).

- [ ] **Step 4: Record the exact HEAD commit for the sync contract**

Run: `git rev-parse HEAD`
Record the commit hash in the PR description and in Plan 2's Task 3 (the gvl path-dep and this wheel MUST be this commit).

- [ ] **Step 5: Commit any build-config changes** (only if `pixi.toml` / CI touched; otherwise skip).

---

## Task 7: genoray docs / roadmap

**Files:**
- Modify: genoray's migration/roadmap doc (search `docs/` for the search/gather-split roadmap entry).

- [ ] **Step 1: Mark the read-bound gather + conversion feature**

Add a roadmap entry: `conversion` query-only build ✅; per-class `find_ranges` ranges + `gather_ranges_readbound` + `BatchResultSplit` ✅ (parity vs union & `decode_hap`, zero-tree control); note it is additive to the shipped split. Link this plan.

- [ ] **Step 2: Commit**

```bash
git add docs/
git commit -m "docs: read-bound per-class gather + conversion feature roadmap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes (traceability to the spec)

- **Spec A1 (conversion feature)** → Task 1. Correction applied: gate `lib.rs:40-52` (`index_bcf_csi`/`index_vcf`) too, not only `vcf_reader.rs`.
- **Spec A2 (per-class ranges + read-bound gather + split-dense BatchResult)** → Tasks 2 (ranges), 3 (`gather_ranges_readbound` + `BatchResultSplit`), 5 (Python surface for the write cache).
- **Spec "parity vs union & decode"** → Task 4 (decoded-per-hap parity, zero-tree control, subset oracle).
- **Spec "shipped union path byte-unchanged"** → every task re-runs `test_ranges_split` / full suite; new code is additive structs/fields/functions only.
- **Spec "build local wheel + crate"** → Task 6.
- **Open question (channel factoring)** → resolved as: var_key merged per hap (unchanged) + dense split per class in `BatchResultSplit`. gvl consumes this exact shape (Plan 2, Task 4).
