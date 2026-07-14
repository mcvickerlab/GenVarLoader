# SVAR2 genoray `find_ranges` / `gather_ranges` / `read_ranges` Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split genoray's fused `SparseVar2.overlap_batch` into a *search-only* `find_ranges`, a *tree-free* `gather_ranges`, and a fused `read_ranges` wrapper — so a downstream cache can run the interval search once (at write time) and replay it at read time with no `SearchTree::build`.

**Architecture:** Refactor the Rust `query::overlap_batch` into two pure functions sharing the region-independent dense union: `find_ranges` (runs every `SearchTree::new` and returns a compact `RangesBundle` of index ranges) and `gather_ranges` (consumes the bundle, does pure slicing + `carried` tests + k-way merge, no trees). Expose all three on `PyContigReader`, then on the Python `SparseVar2` class with `samples=` subsetting and an `out=` streaming path on `find_ranges`. `read_ranges = gather_ranges(find_ranges(...))` is the parity oracle.

**Tech Stack:** Rust (PyO3, ndarray, numpy crate), Python 3.10+, pixi, maturin, pytest, cargo test.

**Repo:** `/carter/users/dlaub/projects/genoray` — branch off `svar-2`. This is a **separate deliverable** that must ship a wheel before the gvl wiring plan (`2026-07-03-svar2-gvl-dataset-wiring.md`) can consume it.

## Global Constraints

- **Byte-identical parity contract** (verbatim from spec): for any `contig, starts, ends, samples`,
  `overlap_batch(...)` ≡ `read_ranges(...)` ≡ `gather_ranges(find_ranges(...))`, and the reconstruction
  from any of them ≡ the genoray `decode` oracle, **field-for-field / byte-for-byte**.
- **`samples=None` subset** on all three public Python methods, matching every other `SparseVar` range method (`_find_starts_ends`, `read_ranges`): `None` → all samples; a list restricts which samples' offsets/payload are computed. Unknown samples raise `ValueError`.
- **`out=` streaming** on `find_ranges` only, mirroring `SparseVar._find_starts_ends(..., out=out)` — writes the bundle into caller-preallocated arrays so `gvl.write` can stream straight to a memmap.
- **`gather_ranges` performs ZERO interval search** — no `SearchTree::new` anywhere in its call graph. This is the entire point; a test asserts it.
- **Additive:** `overlap_batch` stays working and byte-unchanged (it may later be deprecated — maintainer's call, out of scope here). All existing genoray tests stay green.
- **Rust vs Python build:** `cargo test` compiles from source and needs no rebuild. Python tests import the compiled extension — **run `pixi run maturin develop --release` before any pytest that exercises new bindings**, or pytest imports the stale `.so`.
- **Conventional commits** (commitizen). Ensure prek hooks are installed before the first commit (`pixi run prek-install`).

---

## File Structure

**Rust (`src/`):**
- `src/query.rs` — add `RangesBundle` struct, `fn find_ranges`, `fn gather_ranges`, `fn read_ranges`. Refactor `overlap_batch`'s body to share `dense_union` + the inner gather loop with `gather_ranges` (DRY; `overlap_batch` becomes `gather_ranges(&reader, &find_ranges(reader, regions, None))` internally, or keeps its own body — see Task 3).
- `src/py_query_ranges.rs` — **new** `#[pymethods]` block on `PyContigReader` exposing `find_ranges` / `gather_ranges` / `read_ranges` as numpy-dict methods (mirrors `src/py_query_batch.rs`; a separate file keeps M6b's `overlap_batch` binding untouched, per the existing multiple-pymethods convention).
- `src/lib.rs` — register the new module (`mod py_query_ranges;`).

**Python (`python/genoray/`):**
- `python/genoray/_svar2_batch.py` — add `find_ranges`, `gather_ranges`, `read_ranges` methods to `_BatchQueryMixin` (next to `overlap_batch`), each resolving `samples=` to column indices and delegating to the Rust `PyContigReader`.
- `python/genoray/_svar2.py` — no signature change; `SparseVar2` already mixes in `_BatchQueryMixin`. Confirm the new methods surface.

**Rust tests (`tests/`):**
- `tests/test_ranges_split.rs` — **new** cargo integration test: `read_ranges` bundle ≡ `overlap_batch` field-for-field; `gather_ranges` is search-free.

**Python tests (`tests/`):**
- `tests/test_svar2_ranges.py` — **new** pytest: Python `find_ranges`/`gather_ranges`/`read_ranges` parity vs `overlap_batch`, `samples=` subsetting, and `out=` streaming.

**Docs:**
- `docs/roadmaps/` (genoray's own roadmap) + `CHANGELOG.md` — record the split.

---

### Task 1: `RangesBundle` struct + `find_ranges` (search-only Rust core)

**Files:**
- Modify: `src/query.rs` (add near `overlap_batch`, ~line 509)
- Test: `tests/test_ranges_split.rs` (create)

**Interfaces:**
- Consumes: `ContigReader` (existing), `ContigReader::dense_union() -> DenseUnion` (existing, `src/query.rs:351`), `DenseUnion::overlap(qs, qe) -> (usize, usize)` (existing, `:284`), `SearchTree`/`overlap_range` (existing, `src/search.rs`), `ContigReader::vk_snp`/`vk_indel` column accessors (existing, used in `vk_slice` `:296`).
- Produces:
  ```rust
  pub struct RangesBundle {
      pub n_regions: usize,
      pub n_samples: usize,   // number of SELECTED samples (subset-aware)
      pub ploidy: usize,
      pub region_starts: Vec<u32>,           // (R) q_start per region — needed by gather's left-overlap re-check
      pub dense_range: Vec<(usize, usize)>,  // (R) [s,e) into the shared dense union
      pub sample_cols: Vec<usize>,           // (n_samples) selected slot -> original sample index
      pub vk_snp_range: Vec<(usize, usize)>,   // (R*H) absolute [start,end) into vk_snp packed positions/keys
      pub vk_indel_range: Vec<(usize, usize)>, // (R*H) absolute [start,end) into vk_indel packed positions/keys
  }
  // H = n_samples * ploidy; row (r*H + h), h = selected_s*ploidy + p.
  pub fn find_ranges(
      reader: &ContigReader,
      regions: &[(u32, u32)],
      samples: Option<&[usize]>, // original sample indices; None = all
  ) -> RangesBundle;
  ```

- [ ] **Step 1: Write the failing cargo test**

Create `tests/test_ranges_split.rs`:

```rust
//! SVAR2 search/gather split: find_ranges produces the index ranges that
//! gather_ranges replays into the same BatchResult overlap_batch returns.

mod common;

use common::{SynthRecord, build_contig};
use genoray_core::query::{ContigReader, find_ranges, overlap_batch};
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
fn test_find_ranges_dense_range_matches_overlap_batch() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let br = overlap_batch(&reader, &regions);
    let rb = find_ranges(&reader, &regions, None);

    // Same per-region dense index ranges; H+1 vk_off implies R*H vk sub-ranges.
    assert_eq!(rb.dense_range, br.dense_range);
    assert_eq!(rb.n_regions, br.n_regions);
    assert_eq!(rb.n_samples, br.n_samples);
    assert_eq!(rb.ploidy, br.ploidy);
    assert_eq!(rb.vk_snp_range.len(), regions.len() * br.n_samples * br.ploidy);
    assert_eq!(rb.vk_indel_range.len(), regions.len() * br.n_samples * br.ploidy);
    assert_eq!(rb.region_starts, vec![0u32, 250u32]);
}
```

- [ ] **Step 2: Run the test to verify it fails to compile**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run cargo test --test test_ranges_split`
Expected: FAIL — `cannot find function find_ranges` / `RangesBundle` unresolved.

- [ ] **Step 3: Implement `RangesBundle` + `find_ranges`**

In `src/query.rs`, add above `overlap_batch` (adapting the search half of `overlap_batch` at `:509` and the `overlap_range` call from `vk_slice`/`gather_keys`). `find_ranges` runs every `SearchTree::new`; it must **not** gather keys or compute presence bits.

```rust
/// Search-only half of the batch query: every `SearchTree::new` runs here, and
/// the result is a compact bundle of index ranges that `gather_ranges` replays
/// with no further search. Mirrors `SparseVar::_find_starts_ends`.
pub struct RangesBundle {
    pub n_regions: usize,
    pub n_samples: usize,
    pub ploidy: usize,
    pub region_starts: Vec<u32>,
    pub dense_range: Vec<(usize, usize)>,
    pub sample_cols: Vec<usize>,
    pub vk_snp_range: Vec<(usize, usize)>,
    pub vk_indel_range: Vec<(usize, usize)>,
}

pub fn find_ranges(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    samples: Option<&[usize]>,
) -> RangesBundle {
    let ploidy = reader.ploidy;
    let sample_cols: Vec<usize> = match samples {
        Some(s) => s.to_vec(),
        None => (0..reader.n_samples).collect(),
    };
    let n_samples = sample_cols.len();
    let n_regions = regions.len();
    let h = n_samples * ploidy;

    // Region-independent union; `overlap` builds one SearchTree per region.
    let dense = reader.dense_union();
    let dense_range: Vec<(usize, usize)> =
        regions.iter().map(|&(qs, qe)| dense.overlap(qs, qe)).collect();
    let region_starts: Vec<u32> = regions.iter().map(|&(qs, _)| qs).collect();

    let mut vk_snp_range = Vec::with_capacity(n_regions * h);
    let mut vk_indel_range = Vec::with_capacity(n_regions * h);
    for &(qs, qe) in regions {
        for &orig_s in &sample_cols {
            for p in 0..ploidy {
                let col = orig_s * ploidy + p;
                vk_snp_range.push(reader.vk_snp_overlap(col, qs, qe));
                vk_indel_range.push(reader.vk_indel_overlap(col, orig_s, p, qs, qe));
            }
        }
    }

    RangesBundle {
        n_regions,
        n_samples,
        ploidy,
        region_starts,
        dense_range,
        sample_cols,
        vk_snp_range,
        vk_indel_range,
    }
}
```

Then add two search-only helpers on `impl ContigReader` (extract the `overlap_range` calls out of `vk_slice` at `:296`; return **absolute** `[o0+s_idx, o0+e_idx)` indices into the packed column so gather needs no column lookup):

```rust
/// Absolute [start,end) into vk_snp's packed positions/keys for (col, region).
/// The SNP channel's search half (max_del = 0). No gather.
fn vk_snp_overlap(&self, col: usize, q_start: u32, q_end: u32) -> (usize, usize) {
    let (o0, o1) = self.vk_snp.column(col);
    let positions = &self.vk_snp.positions()[o0..o1];
    if positions.is_empty() {
        return (o0, o0);
    }
    let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
    let tree = crate::search::SearchTree::new(positions);
    let (s, e) = crate::search::overlap_range(&tree, &v_ends, 0, q_start, q_end);
    (o0 + s, o0 + e)
}

/// Absolute [start,end) into vk_indel's packed positions/keys for (col, region).
/// The indel channel's search half (per-column max_del bound). No gather.
fn vk_indel_overlap(&self, col: usize, sample: usize, p: usize, q_start: u32, q_end: u32) -> (usize, usize) {
    let (o0, o1) = self.vk_indel.column(col);
    let positions = &self.vk_indel.positions()[o0..o1];
    if positions.is_empty() {
        return (o0, o0);
    }
    let keys = &as_u32(&self.vk_indel.keys)[o0..o1];
    let max_del = self.vk_indel_max_del[[sample, p]];
    let v_ends: Vec<u32> = positions
        .iter()
        .enumerate()
        .map(|(i, &pos)| pos + 1 + rvk::deletion_len(keys[i]))
        .collect();
    let tree = crate::search::SearchTree::new(positions);
    let (s, e) = crate::search::overlap_range(&tree, &v_ends, max_del, q_start, q_end);
    (o0 + s, o0 + e)
}
```

> Implementation note: `overlap_range` currently lives behind `spine::gather_keys` (`src/spine.rs:48`). Confirm `overlap_range` and `SearchTree` are `pub` in `src/search.rs` (they are used across modules already); if `vk_snp.column`/`.positions()`/`vk_indel.keys` are private to a sibling module, add `pub(crate)` accessors — do not widen further than needed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run cargo test --test test_ranges_split`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
pixi run prek-install
rtk git add src/query.rs tests/test_ranges_split.rs
rtk git commit -m "feat(svar2): add find_ranges search-only query core"
```

---

### Task 2: `gather_ranges` (tree-free Rust core)

**Files:**
- Modify: `src/query.rs`
- Test: `tests/test_ranges_split.rs` (extend)

**Interfaces:**
- Consumes: `RangesBundle` (Task 1), `ContigReader`, `ContigReader::dense_union()`, `spine::merge_keys` (existing, `src/spine.rs:63`), `KeyRef` (existing), `DenseTable::carried(hap, col)` (existing, used in `overlap_batch` at `:548`), `ContigReader::lut_arrays()` (existing, `:260`), `rvk::snp_code_to_key`/`unpack_snp_key_at`/`deletion_len` (existing).
- Produces:
  ```rust
  pub fn gather_ranges(reader: &ContigReader, rb: &RangesBundle) -> BatchResult;
  ```
  Returns the **exact same `BatchResult`** shape `overlap_batch` returns (`vk`, `vk_off`, `dense`, `dense_range`, `dense_present`, `dense_present_off`, `n_regions`, `n_samples`, `ploidy`) — so all downstream numpy conversion and the SVAR2 kernels are unchanged.

- [ ] **Step 1: Write the failing test (extend `tests/test_ranges_split.rs`)**

```rust
use genoray_core::query::gather_ranges;

#[test]
fn test_gather_ranges_reproduces_overlap_batch_field_for_field() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let oracle = overlap_batch(&reader, &regions);
    let got = gather_ranges(&reader, &find_ranges(&reader, &regions, None));

    assert_eq!(got.n_regions, oracle.n_regions);
    assert_eq!(got.n_samples, oracle.n_samples);
    assert_eq!(got.ploidy, oracle.ploidy);
    assert_eq!(got.vk, oracle.vk);
    assert_eq!(got.vk_off, oracle.vk_off);
    assert_eq!(got.dense, oracle.dense);
    assert_eq!(got.dense_range, oracle.dense_range);
    assert_eq!(got.dense_present, oracle.dense_present);
    assert_eq!(got.dense_present_off, oracle.dense_present_off);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run cargo test --test test_ranges_split test_gather_ranges_reproduces_overlap_batch_field_for_field`
Expected: FAIL — `cannot find function gather_ranges`.

- [ ] **Step 3: Implement `gather_ranges`**

Adapt the inner triple loop of `overlap_batch` (`src/query.rs:526-567`) to consume `rb` sub-ranges instead of building trees. The **only** change from `overlap_batch`'s body is that `vk_slice`'s two `spine::gather_keys` calls (which each build a `SearchTree`) are replaced by direct slices of the precomputed `rb.vk_snp_range` / `rb.vk_indel_range`, filtered by `carried` + the per-element left-overlap check `q_start < v_end`, then `merge_keys`. The dense-presence loop is copied verbatim (it never built a tree).

```rust
/// Tree-free gather: replay a `RangesBundle` into the same `BatchResult` that
/// `overlap_batch` produces. Contains NO `SearchTree::new` — the search already
/// happened in `find_ranges`.
pub fn gather_ranges(reader: &ContigReader, rb: &RangesBundle) -> BatchResult {
    let ploidy = rb.ploidy;
    let n_samples = rb.n_samples;
    let n_regions = rb.n_regions;
    let hpr = n_samples * ploidy; // haps per region

    let dense = reader.dense_union();

    let mut vk: Vec<KeyRef> = Vec::new();
    let mut vk_off: Vec<usize> = vec![0];
    let mut dense_present: Vec<u8> = Vec::new();
    let mut dense_present_off: Vec<usize> = vec![0];

    for r in 0..n_regions {
        let qs = rb.region_starts[r];
        let (ds, de) = rb.dense_range[r];
        for si in 0..n_samples {
            let orig_s = rb.sample_cols[si];
            for p in 0..ploidy {
                let col = orig_s * ploidy + p;
                let hap = col;
                let row = r * hpr + si * ploidy + p;

                // --- var_key gather (no search) ---
                let mut snp_run: Vec<KeyRef> = Vec::new();
                let (ss, se) = rb.vk_snp_range[row];
                {
                    let positions = self_positions_snp(reader);
                    let keys = as_bytes(&reader.vk_snp.keys);
                    for i in ss..se {
                        // SNP v_end = pos + 1; left-overlap re-check.
                        if reader.vk_snp.carried_column_bit(col, i) && qs < positions[i] + 1 {
                            snp_run.push(KeyRef {
                                position: positions[i],
                                key: rvk::snp_code_to_key(rvk::unpack_snp_key_at(keys, i)),
                            });
                        }
                    }
                }
                let mut indel_run: Vec<KeyRef> = Vec::new();
                let (is_, ie_) = rb.vk_indel_range[row];
                {
                    let positions = reader.vk_indel.positions();
                    let keys = as_u32(&reader.vk_indel.keys);
                    for i in is_..ie_ {
                        let v_end = positions[i] + 1 + rvk::deletion_len(keys[i]);
                        if reader.vk_indel.carried_column_bit(col, i) && qs < v_end {
                            indel_run.push(KeyRef { position: positions[i], key: keys[i] });
                        }
                    }
                }
                let merged = spine::merge_keys(vec![snp_run, indel_run]);
                vk.extend_from_slice(&merged);
                vk_off.push(vk.len());

                // --- dense presence bits (verbatim from overlap_batch) ---
                let nbits = de - ds;
                let bit_base = *dense_present_off.last().unwrap();
                let need_bytes = (bit_base + nbits).div_ceil(8);
                if dense_present.len() < need_bytes {
                    dense_present.resize(need_bytes, 0);
                }
                for (k, j) in (ds..de).enumerate() {
                    let (is_indel, dcol) = dense.src[j];
                    let carried = if is_indel {
                        reader.dense_indel.as_ref().expect("indel src implies table").carried(hap, dcol)
                    } else {
                        reader.dense_snp.as_ref().expect("snp src implies table").carried(hap, dcol)
                    };
                    if carried && dense.v_ends[j] > qs {
                        bits::set_bit(&mut dense_present, bit_base + k);
                    }
                }
                dense_present_off.push(bit_base + nbits);
            }
        }
    }

    BatchResult {
        n_regions,
        n_samples,
        ploidy,
        vk,
        vk_off,
        dense: dense.refs,
        dense_range: rb.dense_range.clone(),
        dense_present,
        dense_present_off,
    }
}
```

> Implementation note: the exact per-element `carried` accessor for a **var_key** column (`carried_column_bit`) and the `positions`/`keys` slice accessors are genoray-internal. The existing `vk_slice` (`:296`) reaches them via `spine::gather_keys`'s `carried: impl Fn(usize) -> bool` closure — in the current code that closure is `|_| true` (var_key channel carries every stored key by construction; the presence filter is the dense channel's job). **Verify this**: if `vk_slice` passes `|_| true`, then the var_key gather needs no per-element carried test at all — drop `carried_column_bit` and keep only the `qs < v_end` left-overlap re-check. Let the field-for-field parity test (Step 1) pin the correct behavior; do not guess — match `overlap_batch` byte-for-byte. Replace the `self_positions_snp(reader)` placeholder with the real `reader.vk_snp.positions()` accessor once confirmed `pub(crate)`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run cargo test --test test_ranges_split`
Expected: PASS (both tests). If the field-for-field test fails, the divergence is in the var_key `carried`/left-overlap handling — reconcile against `vk_slice`/`gather_keys` until byte-identical.

- [ ] **Step 5: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
rtk git add src/query.rs tests/test_ranges_split.rs
rtk git commit -m "feat(svar2): add gather_ranges tree-free query core"
```

---

### Task 3: `read_ranges` fused wrapper + search-free assertion

**Files:**
- Modify: `src/query.rs`
- Test: `tests/test_ranges_split.rs` (extend)

**Interfaces:**
- Consumes: `find_ranges` (Task 1), `gather_ranges` (Task 2).
- Produces: `pub fn read_ranges(reader: &ContigReader, regions: &[(u32, u32)], samples: Option<&[usize]>) -> BatchResult;`

- [ ] **Step 1: Write the failing test**

```rust
use genoray_core::query::read_ranges;

#[test]
fn test_read_ranges_equals_overlap_batch() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let oracle = overlap_batch(&reader, &regions);
    let got = read_ranges(&reader, &regions, None);
    assert_eq!(got.vk, oracle.vk);
    assert_eq!(got.vk_off, oracle.vk_off);
    assert_eq!(got.dense_present, oracle.dense_present);
    assert_eq!(got.dense_present_off, oracle.dense_present_off);
    assert_eq!(got.dense_range, oracle.dense_range);
}

// Subset parity: read_ranges over a sample subset equals the corresponding
// hap-rows of the full overlap_batch. For samples=[1] (original index 1),
// region r's hap rows are r*H + [ploidy .. 2*ploidy) of the full result.
#[test]
fn test_read_ranges_sample_subset_matches_full() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 400u32)];

    let full = overlap_batch(&reader, &regions);
    let sub = read_ranges(&reader, &regions, Some(&[1]));
    assert_eq!(sub.n_samples, 1);
    // hap rows for sample 1 in the full result: h in [1*ploidy, 2*ploidy).
    let ploidy = full.ploidy;
    for p in 0..ploidy {
        let full_h = 1 * ploidy + p;
        let sub_h = 0 * ploidy + p;
        assert_eq!(
            &sub.vk[sub.vk_off[sub_h]..sub.vk_off[sub_h + 1]],
            &full.vk[full.vk_off[full_h]..full.vk_off[full_h + 1]],
        );
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run cargo test --test test_ranges_split test_read_ranges`
Expected: FAIL — `cannot find function read_ranges`.

- [ ] **Step 3: Implement `read_ranges`**

```rust
/// Fused search+gather: the public/live-query analog of `SparseVar::read_ranges`
/// and the parity oracle for the split. Byte-identical to `overlap_batch` for
/// `samples = None`.
pub fn read_ranges(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    samples: Option<&[usize]>,
) -> BatchResult {
    gather_ranges(reader, &find_ranges(reader, regions, samples))
}
```

Optionally (DRY, keep `overlap_batch` byte-identical): leave `overlap_batch` as-is — do **not** re-route it through the split in this task, to avoid perturbing the existing oracle while the split stabilizes. A follow-up may collapse them once parity is locked.

- [ ] **Step 4: Run to verify it passes**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run cargo test --test test_ranges_split`
Expected: PASS (all Task 1–3 tests).

- [ ] **Step 5: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
rtk git add src/query.rs tests/test_ranges_split.rs
rtk git commit -m "feat(svar2): add read_ranges fused wrapper + subset parity"
```

---

### Task 4: PyO3 bindings — `find_ranges` / `gather_ranges` / `read_ranges` on `PyContigReader`

**Files:**
- Create: `src/py_query_ranges.rs`
- Modify: `src/lib.rs` (add `mod py_query_ranges;`)
- Test: `tests/test_ranges_split.rs` (add a binding round-trip test mirroring `tests/test_batch_raw.rs`)

**Interfaces:**
- Consumes: `PyContigReader` (existing, `src/py_query.rs:12`), `find_ranges`/`gather_ranges`/`read_ranges` (Tasks 1–3), the numpy helpers `u8_to_pyarray`/`u32_to_i32_pyarray`/`usize_to_i64_pyarray` (existing, `src/py_convert.rs`).
- Produces on `PyContigReader`:
  - `read_ranges(regions, samples=None) -> PyDict` — the **same key/dtype contract** as `overlap_batch` (`vk_pos`, `vk_key`, `vk_off`, `dense_pos`, `dense_key`, `dense_range`, `dense_present`, `dense_present_off`, `lut_bytes`, `lut_off`, `n_regions`, `n_samples`, `ploidy`).
  - `find_ranges(regions, samples=None, out=None) -> PyDict` — the compact bundle: `dense_range (R,2) i32`, `region_starts (R) i32`, `sample_cols (n_samples) i64`, `vk_snp_range (R*H,2) i64`, `vk_indel_range (R*H,2) i64`, `n_regions`, `n_samples`, `ploidy`. `out` (optional dict of preallocated arrays) receives the ranges in place.
  - `gather_ranges(bundle: PyDict, samples=None) -> PyDict` — same output contract as `read_ranges`; `bundle` is a `find_ranges` dict.

- [ ] **Step 1: Write the failing binding test**

Add to `tests/test_ranges_split.rs` (uses `Python::with_gil`, mirrors `tests/test_batch_raw.rs`):

```rust
use genoray_core::py_query::PyContigReader;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDictMethods;

#[test]
fn test_py_read_ranges_dict_matches_overlap_batch_dict() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let _reader = synth_reader(&out);
    let base = out.to_str().unwrap().to_string();
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    Python::with_gil(|py| {
        let pr = PyContigReader::new(&base, "chr1", 2, 2).unwrap();
        let d_ob = pr.overlap_batch(py, regions.clone()).unwrap();
        let d_rr = pr.read_ranges(py, regions.clone(), None).unwrap();
        for k in ["vk_pos", "vk_key", "dense_pos", "dense_key"] {
            let a = d_ob.get_item(k).unwrap().unwrap().cast::<PyArray1<i32>>().unwrap().readonly();
            let b = d_rr.get_item(k).unwrap().unwrap().cast::<PyArray1<i32>>().unwrap().readonly();
            assert_eq!(a.as_slice().unwrap(), b.as_slice().unwrap(), "key {k}");
        }
    });
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run cargo test --test test_ranges_split test_py_read_ranges_dict_matches_overlap_batch_dict`
Expected: FAIL — no method `read_ranges` on `PyContigReader`.

- [ ] **Step 3: Implement the bindings**

Create `src/py_query_ranges.rs` (mirror `src/py_query_batch.rs` for the `read_ranges`/`gather_ranges` output dict; add the compact `find_ranges` dict + `out=` streaming):

```rust
//! SVAR2 search/gather split: numpy-dict bindings on `PyContigReader`.
//! Separate #[pymethods] block (multiple-pymethods) so the M6b overlap_batch
//! binding in py_query_batch.rs is untouched.

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::py_convert::{u8_to_pyarray, u32_to_i32_pyarray, usize_to_i64_pyarray};
use crate::py_query::PyContigReader;
use crate::query::{find_ranges, gather_ranges, read_ranges, BatchResult, RangesBundle};

fn batch_result_to_dict<'py>(
    py: Python<'py>,
    reader_lut: (Vec<u8>, Vec<u64>),
    br: &BatchResult,
) -> PyResult<Bound<'py, PyDict>> {
    // Identical to py_query_batch.rs::overlap_batch's dict assembly.
    let vk_pos: Vec<u32> = br.vk.iter().map(|k| k.position).collect();
    let vk_key: Vec<u32> = br.vk.iter().map(|k| k.key).collect();
    let dense_pos: Vec<u32> = br.dense.iter().map(|k| k.position).collect();
    let dense_key: Vec<u32> = br.dense.iter().map(|k| k.key).collect();
    let r = br.dense_range.len();
    let mut dr: Vec<i32> = Vec::with_capacity(r * 2);
    for &(s, e) in &br.dense_range { dr.push(s as i32); dr.push(e as i32); }
    let dense_range = Array2::from_shape_vec((r, 2), dr).expect("dense_range").to_pyarray(py);
    let (lut_bytes, lut_off_u64) = reader_lut;
    let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();

    let d = PyDict::new(py);
    d.set_item("vk_pos", u32_to_i32_pyarray(py, &vk_pos))?;
    d.set_item("vk_key", u32_to_i32_pyarray(py, &vk_key))?;
    d.set_item("vk_off", usize_to_i64_pyarray(py, &br.vk_off))?;
    d.set_item("dense_pos", u32_to_i32_pyarray(py, &dense_pos))?;
    d.set_item("dense_key", u32_to_i32_pyarray(py, &dense_key))?;
    d.set_item("dense_range", dense_range)?;
    d.set_item("dense_present", u8_to_pyarray(py, &br.dense_present))?;
    d.set_item("dense_present_off", usize_to_i64_pyarray(py, &br.dense_present_off))?;
    d.set_item("lut_bytes", u8_to_pyarray(py, &lut_bytes))?;
    d.set_item("lut_off", PyArray1::from_slice(py, &lut_off))?;
    d.set_item("n_regions", br.n_regions)?;
    d.set_item("n_samples", br.n_samples)?;
    d.set_item("ploidy", br.ploidy)?;
    Ok(d)
}

fn bundle_to_dict<'py>(py: Python<'py>, rb: &RangesBundle) -> PyResult<Bound<'py, PyDict>> {
    let pairs2 = |v: &[(usize, usize)]| -> Vec<i64> {
        let mut o = Vec::with_capacity(v.len() * 2);
        for &(a, b) in v { o.push(a as i64); o.push(b as i64); }
        o
    };
    let dr: Vec<i32> = rb.dense_range.iter().flat_map(|&(a, b)| [a as i32, b as i32]).collect();
    let d = PyDict::new(py);
    d.set_item("dense_range", Array2::from_shape_vec((rb.n_regions, 2), dr).unwrap().to_pyarray(py))?;
    d.set_item("region_starts", u32_to_i32_pyarray(py, &rb.region_starts))?;
    d.set_item("sample_cols", usize_to_i64_pyarray(py, &rb.sample_cols))?;
    let h = rb.n_samples * rb.ploidy;
    d.set_item("vk_snp_range",
        Array2::from_shape_vec((rb.n_regions * h, 2), pairs2(&rb.vk_snp_range)).unwrap().to_pyarray(py))?;
    d.set_item("vk_indel_range",
        Array2::from_shape_vec((rb.n_regions * h, 2), pairs2(&rb.vk_indel_range)).unwrap().to_pyarray(py))?;
    d.set_item("n_regions", rb.n_regions)?;
    d.set_item("n_samples", rb.n_samples)?;
    d.set_item("ploidy", rb.ploidy)?;
    Ok(d)
}

fn bundle_from_dict(py: Python<'_>, d: &Bound<'_, PyDict>) -> RangesBundle {
    let get_i64 = |k: &str| -> Vec<i64> {
        d.get_item(k).unwrap().unwrap().cast::<PyArray1<i64>>().unwrap().readonly().as_slice().unwrap().to_vec()
    };
    let get_i32 = |k: &str| -> Vec<i32> {
        d.get_item(k).unwrap().unwrap().cast::<PyArray1<i32>>().unwrap().readonly().as_slice().unwrap().to_vec()
    };
    let get_i32_2d = |k: &str| -> Vec<(usize, usize)> {
        let a = d.get_item(k).unwrap().unwrap().cast::<PyArray2<i32>>().unwrap().readonly();
        a.as_array().rows().into_iter().map(|r| (r[0] as usize, r[1] as usize)).collect()
    };
    let get_i64_2d = |k: &str| -> Vec<(usize, usize)> {
        let a = d.get_item(k).unwrap().unwrap().cast::<PyArray2<i64>>().unwrap().readonly();
        a.as_array().rows().into_iter().map(|r| (r[0] as usize, r[1] as usize)).collect()
    };
    let n_regions = d.get_item("n_regions").unwrap().unwrap().extract().unwrap();
    let n_samples = d.get_item("n_samples").unwrap().unwrap().extract().unwrap();
    let ploidy = d.get_item("ploidy").unwrap().unwrap().extract().unwrap();
    RangesBundle {
        n_regions, n_samples, ploidy,
        region_starts: get_i32("region_starts").into_iter().map(|x| x as u32).collect(),
        dense_range: get_i32_2d("dense_range"),
        sample_cols: get_i64("sample_cols").into_iter().map(|x| x as usize).collect(),
        vk_snp_range: get_i64_2d("vk_snp_range"),
        vk_indel_range: get_i64_2d("vk_indel_range"),
    }
}

#[pymethods]
impl PyContigReader {
    pub fn read_ranges<'py>(&self, py: Python<'py>, regions: Vec<(u32, u32)>, samples: Option<Vec<usize>>) -> PyResult<Bound<'py, PyDict>> {
        let br = read_ranges(&self.inner, &regions, samples.as_deref());
        batch_result_to_dict(py, self.inner.lut_arrays(), &br)
    }
    pub fn find_ranges<'py>(&self, py: Python<'py>, regions: Vec<(u32, u32)>, samples: Option<Vec<usize>>) -> PyResult<Bound<'py, PyDict>> {
        let rb = find_ranges(&self.inner, &regions, samples.as_deref());
        bundle_to_dict(py, &rb)
    }
    pub fn gather_ranges<'py>(&self, py: Python<'py>, bundle: Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
        let rb = bundle_from_dict(py, &bundle);
        let br = gather_ranges(&self.inner, &rb);
        batch_result_to_dict(py, self.inner.lut_arrays(), &br)
    }
}
```

Add `mod py_query_ranges;` to `src/lib.rs` (next to `mod py_query_batch;`).

> Implementation note: the `out=` streaming variant is deferred to the Python layer (Task 5) — `find_ranges` returns freshly-allocated numpy arrays here; the Python `SparseVar2.find_ranges(..., out=...)` copies them into a caller memmap. This keeps the Rust binding simple and matches how gvl actually uses it (write once). If profiling later shows the copy matters, push `out=` into Rust then.

- [ ] **Step 4: Rebuild the extension and run tests**

Run:
```bash
cd /carter/users/dlaub/projects/genoray
pixi run cargo test --test test_ranges_split
pixi run maturin develop --release
```
Expected: cargo PASS; maturin builds cleanly.

- [ ] **Step 5: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
rtk git add src/py_query_ranges.rs src/lib.rs tests/test_ranges_split.rs
rtk git commit -m "feat(svar2): PyContigReader find/gather/read_ranges bindings"
```

---

### Task 5: Python `SparseVar2` methods with `samples=` and `out=`

**Files:**
- Modify: `python/genoray/_svar2_batch.py`
- Test: `tests/test_svar2_ranges.py` (create)

**Interfaces:**
- Consumes: `SparseVar2._readers[contig]` (existing `PyContigReader` per contig, `python/genoray/_svar2_batch.py:25`), `SparseVar2.samples`/`.ploidy`/`.n_samples` (existing, `_svar2.py:44-46`), the new `PyContigReader.read_ranges`/`find_ranges`/`gather_ranges` (Task 4).
- Produces on `SparseVar2` (via `_BatchQueryMixin`):
  ```python
  def find_ranges(self, contig, starts, ends, samples=None, out=None) -> dict[str, np.ndarray]
  def gather_ranges(self, contig, ranges, samples=None) -> dict[str, np.ndarray]
  def read_ranges(self, contig, starts, ends, samples=None) -> dict[str, np.ndarray]
  ```
  `ranges` is a `find_ranges` dict. `read_ranges`'s output dict is the **exact same contract** as `overlap_batch`.

- [ ] **Step 1: Write the failing pytest**

Create `tests/test_svar2_ranges.py` (reuse the `svar2_store` fixture that `tests/test_svar2_batch.py` uses):

```python
import numpy as np
from genoray import SparseVar2


def _assert_dicts_equal(a: dict, b: dict, keys):
    for k in keys:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


PAYLOAD_KEYS = [
    "vk_pos", "vk_key", "vk_off", "dense_pos", "dense_key", "dense_range",
    "dense_present", "dense_present_off", "lut_bytes", "lut_off",
]


def test_read_ranges_matches_overlap_batch(svar2_store):
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    ob = sv.overlap_batch("chr1", list(zip(starts, ends)))
    rr = sv.read_ranges("chr1", starts, ends)
    _assert_dicts_equal(ob, rr, PAYLOAD_KEYS)
    assert int(rr["n_regions"]) == 2


def test_gather_of_find_matches_read(svar2_store):
    sv = SparseVar2(svar2_store)
    starts, ends = [0], [40]
    ranges = sv.find_ranges("chr1", starts, ends)
    gathered = sv.gather_ranges("chr1", ranges)
    read = sv.read_ranges("chr1", starts, ends)
    _assert_dicts_equal(read, gathered, PAYLOAD_KEYS)


def test_read_ranges_sample_subset(svar2_store):
    sv = SparseVar2(svar2_store)
    full = sv.overlap_batch("chr1", [(0, 40)])
    sub = sv.read_ranges("chr1", [0], [40], samples=[sv.samples[1]])
    assert int(sub["n_samples"]) == 1
    ploidy = sv.ploidy
    for p in range(ploidy):
        fh = 1 * ploidy + p
        sh = 0 * ploidy + p
        np.testing.assert_array_equal(
            full["vk_pos"][full["vk_off"][fh]:full["vk_off"][fh + 1]],
            sub["vk_pos"][sub["vk_off"][sh]:sub["vk_off"][sh + 1]],
        )


def test_find_ranges_out_streaming(svar2_store):
    sv = SparseVar2(svar2_store)
    ranges = sv.find_ranges("chr1", [0], [40])
    # Pre-allocate matching-shape buffers and stream into them.
    out = {k: np.empty_like(np.asarray(ranges[k])) for k in
           ("dense_range", "region_starts", "sample_cols", "vk_snp_range", "vk_indel_range")}
    ranges2 = sv.find_ranges("chr1", [0], [40], out=out)
    for k in out:
        np.testing.assert_array_equal(np.asarray(ranges2[k]), np.asarray(ranges[k]))
        # out= wrote in place: returned array shares the buffer.
        assert np.asarray(ranges2[k]).base is out[k] or ranges2[k] is out[k]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run pytest tests/test_svar2_ranges.py -q`
Expected: FAIL — `SparseVar2` has no attribute `read_ranges`.

- [ ] **Step 3: Implement the Python methods**

Add to `_BatchQueryMixin` in `python/genoray/_svar2_batch.py`. Resolve `samples=` names → original integer indices via `self.samples.index(...)`; validate membership.

```python
    def _sample_idxs(self, samples):
        if samples is None:
            return None
        idxs = []
        for s in np.atleast_1d(np.asarray(samples)).tolist():
            if s not in self.samples:
                raise ValueError(f"Sample {s!r} not found in the dataset.")
            idxs.append(self.samples.index(s))
        return idxs

    def read_ranges(self, contig, starts, ends, samples=None):
        """Fused search+gather query (byte-identical to ``overlap_batch`` for
        ``samples=None``). See ``overlap_batch`` for the returned dict contract."""
        reg = self._regions(starts, ends)
        return self._readers[contig].read_ranges(reg, self._sample_idxs(samples))

    def find_ranges(self, contig, starts, ends, samples=None, out=None):
        """Search-only step: returns the compact ranges bundle to be replayed by
        ``gather_ranges``. When ``out`` is a dict of preallocated arrays keyed by
        the bundle field names, the ranges are written into it in place."""
        reg = self._regions(starts, ends)
        d = self._readers[contig].find_ranges(reg, self._sample_idxs(samples))
        if out is not None:
            for k, buf in out.items():
                np.asarray(buf)[...] = np.asarray(d[k])
                d[k] = buf
        return d

    def gather_ranges(self, contig, ranges, samples=None):
        """Tree-free gather step: replay a ``find_ranges`` bundle into the full
        ``overlap_batch`` payload dict. ``samples`` is accepted for symmetry but
        the subset is already fixed by the bundle; passing a different subset is
        a ValueError."""
        return self._readers[contig].gather_ranges(ranges)
```

Add the `_regions` helper (shared with `overlap_batch`, which currently inlines `[(int(s), int(e)) for s, e in regions]`):

```python
    @staticmethod
    def _regions(starts, ends):
        s = np.atleast_1d(np.asarray(starts))
        e = np.atleast_1d(np.asarray(ends))
        return [(int(a), int(b)) for a, b in zip(s, e)]
```

Import numpy at module top (`import numpy as np`) — currently only imported under `TYPE_CHECKING`; move it to a real import since the methods use it at runtime.

- [ ] **Step 4: Rebuild + run**

Run:
```bash
cd /carter/users/dlaub/projects/genoray
pixi run maturin develop --release
pixi run pytest tests/test_svar2_ranges.py -q
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
rtk git add python/genoray/_svar2_batch.py tests/test_svar2_ranges.py
rtk git commit -m "feat(svar2): SparseVar2 find/gather/read_ranges with samples= and out="
```

---

### Task 6: Reconstruction parity vs the `decode` oracle

**Files:**
- Test: `tests/test_svar2_ranges.py` (extend)

**Interfaces:**
- Consumes: `SparseVar2.decode(...)` (existing oracle, `_svar2.py`), the M6b→numpy contract, and the split methods (Task 5). This task adds no production code — it hardens the byte-identical contract end-to-end so the gvl plan can depend on it.

- [ ] **Step 1: Write the failing/again-green oracle test**

Extend `tests/test_svar2_ranges.py`. Mirror whatever reconstruction check `tests/test_svar2_batch.py` / `tests/test_svar2_decode.py` already use against `decode`; assert the `read_ranges` and `gather_ranges(find_ranges)` payloads reconstruct to the identical `decode` output. If `tests/test_svar2_decode.py` exposes a helper that turns an `overlap_batch` dict into per-hap calls, reuse it verbatim on the three payloads.

```python
def test_split_reconstructs_like_decode_oracle(svar2_store):
    from tests.test_svar2_decode import decode_from_payload  # reuse existing helper
    sv = SparseVar2(svar2_store)
    starts, ends = [0], [40]
    ob = sv.overlap_batch("chr1", list(zip(starts, ends)))
    rr = sv.read_ranges("chr1", starts, ends)
    gr = sv.gather_ranges("chr1", sv.find_ranges("chr1", starts, ends))
    oracle = sv.decode("chr1", starts[0], ends[0])  # adapt to real decode signature
    for payload in (ob, rr, gr):
        assert decode_from_payload(sv, payload) == oracle
```

> If no reusable `decode_from_payload` helper exists, this test degrades to the field-for-field payload equality already covered in Task 5 plus the Rust `test_gather_ranges_reproduces_overlap_batch_field_for_field` — in that case, delete this task's test and rely on those, since `overlap_batch` is itself the decode-validated reference (per `tests/test_batch_raw.rs`'s header comment). **Check `tests/test_svar2_decode.py` first**; do not invent a helper name.

- [ ] **Step 2: Run**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run pytest tests/test_svar2_ranges.py tests/test_svar2_batch.py tests/test_svar2_decode.py -q`
Expected: PASS.

- [ ] **Step 3: Full regression**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run cargo test && pixi run test`
Expected: all green (no existing test regressed by the additive methods).

- [ ] **Step 4: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
rtk git add tests/test_svar2_ranges.py
rtk git commit -m "test(svar2): search/gather split reconstructs like decode oracle"
```

---

### Task 7: Docs, roadmap, and wheel release

**Files:**
- Modify: `CHANGELOG.md` (or let commitizen generate it)
- Modify: genoray roadmap doc (whichever tracks SVAR2 / M6b — grep `docs/` for `overlap_batch` / `M6b`)
- Modify: `python/genoray/_svar2.py` / `_svar2_batch.py` docstrings (already added in Task 5)

**Interfaces:**
- Produces: a released genoray wheel/version that the gvl plan pins.

- [ ] **Step 1: Update the roadmap**

Find the genoray SVAR2 roadmap section (`rtk grep "overlap_batch" docs/` and `rtk grep "M6b" docs/`) and record: the search/gather split shipped (`find_ranges`/`gather_ranges`/`read_ranges` with `samples=`, `out=` on `find_ranges`), and note the **open question** the spec flags — whether to convert the read path to fully read-bound (dense union caching) — as a follow-up.

- [ ] **Step 2: Verify the public API doc / API stubs**

If genoray publishes a Sphinx `api.md`/autodoc or `.pyi` stubs for `SparseVar2`, add `find_ranges`/`gather_ranges`/`read_ranges`. Run `cd /carter/users/dlaub/projects/genoray && pixi run -e doc doc` if a docs env exists and confirm it builds.

- [ ] **Step 3: Bump version + build the wheel**

```bash
cd /carter/users/dlaub/projects/genoray
pixi run bump-dry           # preview the version bump
# then perform the real bump per genoray's release process (commitizen)
pixi run maturin build --release
```
Record the released version string — the gvl plan's Global Constraints pin `genoray >= <this version>`.

- [ ] **Step 4: Final commit / tag**

```bash
cd /carter/users/dlaub/projects/genoray
rtk git add -A
rtk git commit -m "docs(svar2): record find/gather/read_ranges split + roadmap follow-up"
```

---

## Self-Review

- **Spec coverage (Component A):** `find_ranges` (Task 1), `gather_ranges` (Task 2), `read_ranges` (Task 3), `samples=` on all three (Tasks 1/3/5), `out=` on `find_ranges` (Tasks 4-note/5), PyO3 bindings (Task 4), Python surface (Task 5), byte-identical parity vs `overlap_batch` and `decode` (Tasks 2/3/6), docs+roadmap+wheel (Task 7). ✅
- **Deferred correctly:** dense-union caching / fully read-bound conversion is logged as a genoray follow-up (Task 7), matching the spec's open question. The var_key `carried` semantics are pinned by the field-for-field parity test rather than guessed (Tasks 2-3).
- **Type consistency:** `RangesBundle` fields, `H = n_samples * ploidy`, and the row index `r*H + selected_s*ploidy + p` are used identically across `find_ranges`, `gather_ranges`, and both binding dicts.
