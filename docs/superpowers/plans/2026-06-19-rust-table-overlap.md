# Rust Table Overlap (COITrees) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `gvl.Table`'s polars-bio overlap backend with a self-contained COITrees-based Rust module, fixing `max_mem` blow-up during `gvl.write()`/`update()`, removing the segfault source, and promoting `Table` from `experimental` to the public API.

**Architecture:** A new Rust module `src/tables.rs` (mirroring the single-file `src/bigwig.rs` pattern) owns an immutable interval store grouped by `(chrom_code, sample_code)`, a COITrees-backed overlap engine, and a streaming writer. A PyO3 `#[pyclass] RustTable` is built once from the canonical columns; Python (`_table.py`) shrinks to a polars-based constructor/validator that factor-encodes the frame and delegates all overlap to `RustTable`. Trees are built lazily one contig at a time and dropped on contig change.

**Tech Stack:** Rust (PyO3 0.28 abi3-py310, ndarray 0.17, coitrees), Python (polars, numpy, seqpro Ragged), genoray `ContigNormalizer`.

## Global Constraints

- **abi3 wheels must keep building** across py310–313 × linux/macOS (standing roadmap invariant).
- **On-disk format is headerless raw little-endian bytes** (NOT real `.npy`): `intervals.npy` = rows of `INTERVAL_DTYPE` = `[i32 start, i32 end, f32 value]` (aligned, 12 bytes/row); `offsets.npy` = `i64` little-endian, length `n_cells + 1`, region-major sample-minor, starting at 0. Read back via `np.memmap`. Match this exactly — see `python/genvarloader/_ragged.py:28`.
- **Overlap semantics:** half-open `[start, end)`, zero-based. Two intervals overlap iff `a.start < b.end and a.end > b.start` (matches `_brute_count` in `tests/unit/test_table.py` and the polars-bio options set today). **Assumption:** all stored intervals and query regions have positive width (`end > start`); genomic tracks never contain zero-width intervals. Generators must respect this.
- **Ordering contract:** output intervals are region-major, sample-minor; within each `(region, sample)` cell, sorted by `start` (stable in stored order for equal starts). The Table is pre-sorted by `(chrom, sample_id, start)` in Python, so emitting overlapping indices in ascending index order yields ascending `start` order.
- **Contig normalization:** use `genoray._utils.ContigNormalizer` and its vectorized `c_idxs(contigs) -> NDArray[int]` at the Python boundary; never per-row `normalize_contig_name`.
- **Dtypes:** `start`/`end` int32 on disk and in `RaggedIntervals`, `value` float32. Query `starts`/`ends` passed to Rust as int32. `sample_codes`/`chrom_codes` int32. offsets int64.
- **PR strategy:** one bundled PR on branch `feat/rust-table-overlap` (already created). `main` stays shippable.
- **Commits:** end every commit message with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Use `rtk git ...` per repo convention.
- **Test command:** `pixi run -e dev pytest <path> -q`; Rust: `pixi run -e dev cargo test --release` (or `cargo test` for speed during dev). Rebuild the extension before pytest with `pixi run -e dev maturin develop` (or rely on the pixi test task which compiles).

---

## File Structure

- **Create** `src/tables.rs` — Rust store + COITrees engine + count + intervals + streaming write + `#[cfg(test)]` cargo tests.
- **Modify** `src/lib.rs` — `pub mod tables;`, register `RustTable` pyclass.
- **Modify** `Cargo.toml` — add `coitrees`.
- **Rewrite** `python/genvarloader/_table.py` — `Table` becomes a `RustTable`-backed shim; `annot_overlap` rewritten; delete polars-bio code + `ExperimentalWarning`.
- **Modify** `python/genvarloader/_dataset/_write.py` — `_write_track` Table branch; `_annot_intervals` Rust path; `_write_track_rust` contig loop → `ContigNormalizer.c_idxs`.
- **Modify** `python/genvarloader/__init__.py` — export `Table` in `__all__`.
- **Delete** `python/genvarloader/experimental/` subpackage.
- **Modify/un-gate tests:** `tests/unit/test_table.py`, `tests/unit/dataset/test_write_tracks.py`, `tests/unit/test_write_annot.py`, `tests/integration/dataset/test_write_tracks_e2e.py`, `tests/dataset/test_with_methods.py` (comment).
- **Modify** `pyproject.toml` — delete `table` extra.
- **Modify** `docs/source/api.md` — `genvarloader.experimental.Table` → `genvarloader.Table`.
- **Modify** `skills/genvarloader/SKILL.md`, `docs/roadmaps/rust-migration.md`.

---

## Task 1: Cargo dep + Rust store + `RustTable::new`

**Files:**
- Modify: `Cargo.toml`
- Create: `src/tables.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: Rust `RustTable` with fields and a constructor `RustTable::build(sample_codes, chrom_codes, starts, ends, values, n_samples, n_contigs) -> RustTable`; internal `ContigStore`/`SampleIntervals` grouping. Stored columns are assumed pre-sorted by `(chrom_code, sample_code, start)`.

- [ ] **Step 1: Add the coitrees dependency**

In `Cargo.toml`, under `[dependencies]`, add:

```toml
coitrees = "0.4"
```

- [ ] **Step 2: Write the failing cargo test for store grouping**

Create `src/tables.rs` with only the test (so it fails to compile = fails):

```rust
use anyhow::Result;
use ndarray::prelude::*;

// (implementation added in later steps)

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Columns pre-sorted by (chrom_code, sample_code, start), as Python guarantees.
    fn toy() -> RustTable {
        // sample 0: chr0 [0,10),[50,60); sample 1: chr0 [10,20); sample 0: chr1 [0,5)
        let sample_codes = array![0i32, 0, 1, 0];
        let chrom_codes = array![0i32, 0, 0, 1];
        let starts = array![0i32, 50, 10, 0];
        let ends = array![10i32, 60, 20, 5];
        let values = array![1.0f32, 2.0, 3.0, 4.0];
        RustTable::build(
            sample_codes.view(),
            chrom_codes.view(),
            starts.view(),
            ends.view(),
            values.view(),
            2,
            2,
        )
    }

    #[test]
    fn store_groups_by_contig_then_sample() {
        let t = toy();
        // chr0 sample0 has 2 intervals; chr0 sample1 has 1; chr1 sample0 has 1; chr1 sample1 has 0
        assert_eq!(t.n_in_cell(0, 0), 2);
        assert_eq!(t.n_in_cell(0, 1), 1);
        assert_eq!(t.n_in_cell(1, 0), 1);
        assert_eq!(t.n_in_cell(1, 1), 0);
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pixi run -e dev cargo test --lib tables::tests::store_groups_by_contig_then_sample`
Expected: compile error (`RustTable` not found).

- [ ] **Step 4: Implement the store + constructor**

Prepend to `src/tables.rs` (above the test module):

```rust
use anyhow::Result;
use ndarray::prelude::*;
use pyo3::prelude::*;

/// One sample's intervals on one contig, sorted by start.
#[derive(Default, Clone)]
struct SampleIntervals {
    starts: Vec<i32>,
    ends: Vec<i32>,
    values: Vec<f32>,
}

/// All samples' intervals on one contig.
#[derive(Clone)]
struct ContigStore {
    samples: Vec<SampleIntervals>, // indexed by sample_code
}

#[pyclass]
pub struct RustTable {
    n_samples: usize,
    store: Vec<ContigStore>, // indexed by chrom_code (0..n_contigs)
}

impl RustTable {
    pub fn build(
        sample_codes: ArrayView1<i32>,
        chrom_codes: ArrayView1<i32>,
        starts: ArrayView1<i32>,
        ends: ArrayView1<i32>,
        values: ArrayView1<f32>,
        n_samples: usize,
        n_contigs: usize,
    ) -> RustTable {
        let mut store: Vec<ContigStore> = (0..n_contigs)
            .map(|_| ContigStore {
                samples: vec![SampleIntervals::default(); n_samples],
            })
            .collect();
        for i in 0..sample_codes.len() {
            let c = chrom_codes[i] as usize;
            let s = sample_codes[i] as usize;
            let cell = &mut store[c].samples[s];
            cell.starts.push(starts[i]);
            cell.ends.push(ends[i]);
            cell.values.push(values[i]);
        }
        RustTable { n_samples, store }
    }

    #[cfg(test)]
    fn n_in_cell(&self, chrom: usize, sample: usize) -> usize {
        self.store[chrom].samples[sample].starts.len()
    }
}
```

- [ ] **Step 5: Register the module in lib.rs**

In `src/lib.rs`, add after `pub mod bigwig;`:

```rust
pub mod tables;
```

And inside the `genvarloader` pymodule function, after the existing `add_function` lines, register the class:

```rust
    m.add_class::<tables::RustTable>()?;
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pixi run -e dev cargo test --lib tables::tests::store_groups_by_contig_then_sample`
Expected: PASS. (A warning about unused `n_samples`/`Result` is fine for now.)

- [ ] **Step 7: Commit**

```bash
rtk git add Cargo.toml Cargo.lock src/tables.rs src/lib.rs
rtk git commit -m "feat(rust): table interval store + RustTable::build

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: COITrees overlap count

**Files:**
- Modify: `src/tables.rs`

**Interfaces:**
- Consumes: `RustTable` from Task 1.
- Produces: per-contig lazy tree builder `RustTable::build_trees(chrom: usize) -> Vec<coitrees::BasicCOITree<u32, u32>>`; `RustTable::count(chrom_code: i32, q_starts, q_ends, sel_samples: &[i32]) -> Array2<i32>` returning shape `(n_regions, n_sel_samples)`; `chrom_code < 0` returns zeros.

- [ ] **Step 1: Write the failing cargo test (count vs brute force)**

Add to the `tests` module in `src/tables.rs`:

```rust
    fn brute_count(t: &RustTable, chrom: usize, qs: &[i32], qe: &[i32], sel: &[i32]) -> Array2<i32> {
        let mut out = Array2::<i32>::zeros((qs.len(), sel.len()));
        for (sj, &s) in sel.iter().enumerate() {
            let cell = &t.store[chrom].samples[s as usize];
            for (ri, (&rs, &re)) in qs.iter().zip(qe).enumerate() {
                let mut n = 0;
                for k in 0..cell.starts.len() {
                    if cell.starts[k] < re && cell.ends[k] > rs {
                        n += 1;
                    }
                }
                out[[ri, sj]] = n;
            }
        }
        out
    }

    #[test]
    fn count_matches_brute_force() {
        let t = toy();
        let qs = [0i32, 55, 5];
        let qe = [15i32, 65, 55];
        let sel = [0i32, 1];
        let got = t.count(0, &qs, &qe, &sel);
        let exp = brute_count(&t, 0, &qs, &qe, &sel);
        assert_eq!(got, exp);
    }

    #[test]
    fn count_unknown_contig_is_zeros() {
        let t = toy();
        let got = t.count(-1, &[0i32], &[10i32], &[0i32]);
        assert_eq!(got, Array2::<i32>::zeros((1, 1)));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev cargo test --lib tables::tests::count_matches_brute_force`
Expected: compile error (`count` not found).

- [ ] **Step 3: Implement lazy tree builder + count**

Add to the `impl RustTable` block. **NOTE:** confirm the coitrees 0.4 API names against `cargo doc -p coitrees` while implementing — the query callback yields an interval whose metadata accessor may be `.metadata` (field) or `.metadata()`. The cargo test in Step 4 is the gate.

```rust
use coitrees::{BasicCOITree, Interval, IntervalTree};

impl RustTable {
    /// Build one COITree per sample for `chrom`. Intervals are stored half-open
    /// [start, end); coitrees is inclusive, so we store [start, end-1] and query
    /// [qs, qe-1]. Metadata = index into the sample's sorted arrays.
    fn build_trees(&self, chrom: usize) -> Vec<BasicCOITree<u32, u32>> {
        self.store[chrom]
            .samples
            .iter()
            .map(|cell| {
                let ivs: Vec<Interval<u32>> = (0..cell.starts.len())
                    .map(|k| Interval::new(cell.starts[k], cell.ends[k] - 1, k as u32))
                    .collect();
                BasicCOITree::new(&ivs)
            })
            .collect()
    }

    pub fn count(
        &self,
        chrom_code: i32,
        q_starts: &[i32],
        q_ends: &[i32],
        sel_samples: &[i32],
    ) -> Array2<i32> {
        let n_regions = q_starts.len();
        let n_sel = sel_samples.len();
        let mut out = Array2::<i32>::zeros((n_regions, n_sel));
        if chrom_code < 0 {
            return out;
        }
        let trees = self.build_trees(chrom_code as usize);
        for (sj, &s) in sel_samples.iter().enumerate() {
            let tree = &trees[s as usize];
            for ri in 0..n_regions {
                let c = tree.query_count(q_starts[ri], q_ends[ri] - 1) as i32;
                out[[ri, sj]] = c;
            }
        }
        out
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev cargo test --lib tables::tests::count`
Expected: both `count_matches_brute_force` and `count_unknown_contig_is_zeros` PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/tables.rs Cargo.lock
rtk git commit -m "feat(rust): COITrees overlap count for RustTable

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Interval materialization from offsets

**Files:**
- Modify: `src/tables.rs`

**Interfaces:**
- Consumes: `RustTable`, `build_trees`, `count` from Tasks 1–2.
- Produces: `RustTable::intervals_from_offsets(chrom_code, q_starts, q_ends, sel_samples, offsets: &[i64]) -> (Array2<i32>, Array1<f32>)` where the first array is `(n_intervals, 2)` of `[start, end]` and the second is values, laid out region-major sample-minor, within each cell sorted by ascending start (= ascending stored index). `offsets` has length `n_regions*n_sel + 1`.

- [ ] **Step 1: Write the failing cargo test (round-trip vs count)**

Add to the `tests` module:

```rust
    fn offsets_from_count(counts: &Array2<i32>) -> Vec<i64> {
        let mut v = vec![0i64];
        let mut acc = 0i64;
        for r in 0..counts.nrows() {
            for s in 0..counts.ncols() {
                acc += counts[[r, s]] as i64;
                v.push(acc);
            }
        }
        v
    }

    #[test]
    fn intervals_from_offsets_ordered_and_correct() {
        let t = toy();
        let qs = [0i32, 5];
        let qe = [60i32, 55];
        let sel = [0i32, 1];
        let counts = t.count(0, &qs, &qe, &sel);
        let offsets = offsets_from_count(&counts);
        let (coords, vals) = t.intervals_from_offsets(0, &qs, &qe, &sel, &offsets);

        // cell (region0, sample0): intervals [0,10) v1 and [50,60) v2, sorted by start
        assert_eq!(coords[[0, 0]], 0);
        assert_eq!(coords[[0, 1]], 10);
        assert_eq!(vals[0], 1.0);
        assert_eq!(coords[[1, 0]], 50);
        assert_eq!(coords[[1, 1]], 60);
        assert_eq!(vals[1], 2.0);
        // total length equals sum of counts
        assert_eq!(vals.len(), counts.sum() as usize);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev cargo test --lib tables::tests::intervals_from_offsets_ordered_and_correct`
Expected: compile error (`intervals_from_offsets` not found).

- [ ] **Step 3: Implement materialization**

Add to `impl RustTable`:

```rust
    pub fn intervals_from_offsets(
        &self,
        chrom_code: i32,
        q_starts: &[i32],
        q_ends: &[i32],
        sel_samples: &[i32],
        offsets: &[i64],
    ) -> (Array2<i32>, Array1<f32>) {
        let n_total = *offsets.last().unwrap() as usize;
        let mut coords = Array2::<i32>::zeros((n_total, 2));
        let mut values = Array1::<f32>::zeros(n_total);
        if chrom_code < 0 || n_total == 0 {
            return (coords, values);
        }
        let chrom = chrom_code as usize;
        let trees = self.build_trees(chrom);
        let n_sel = sel_samples.len();
        for ri in 0..q_starts.len() {
            for (sj, &s) in sel_samples.iter().enumerate() {
                let cell_idx = ri * n_sel + sj;
                let base = offsets[cell_idx] as usize;
                let tree = &trees[s as usize];
                let mut idxs: Vec<u32> = Vec::new();
                tree.query(q_starts[ri], q_ends[ri] - 1, |iv| idxs.push(iv.metadata));
                // ascending stored index == ascending start (Table pre-sorted by start)
                idxs.sort_unstable();
                let cell = &self.store[chrom].samples[s as usize];
                for (j, &k) in idxs.iter().enumerate() {
                    let k = k as usize;
                    coords[[base + j, 0]] = cell.starts[k];
                    coords[[base + j, 1]] = cell.ends[k];
                    values[base + j] = cell.values[k];
                }
            }
        }
        (coords, values)
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev cargo test --lib tables::tests::intervals_from_offsets_ordered_and_correct`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/tables.rs
rtk git commit -m "feat(rust): materialize ordered intervals from offsets

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Streaming writer + PyO3 methods

**Files:**
- Modify: `src/tables.rs`

**Interfaces:**
- Consumes: all of Tasks 1–3.
- Produces (PyO3 methods on `RustTable`, callable from Python):
  - `RustTable.__new__(sample_codes, chrom_codes, starts, ends, values, n_samples, n_contigs)` (numpy int32/int32/int32/int32/float32 arrays + two ints).
  - `count(chrom_code: i32, starts: i32[], ends: i32[], sel_samples: i32[]) -> PyArray2<i32>`.
  - `intervals(chrom_code, starts, ends, sel_samples, offsets: i64[]) -> (PyArray2<i32>, PyArray1<f32>)`.
  - `write_track(out_dir: str, chrom_codes: i32[], starts: i32[], ends: i32[], sel_samples: i32[], max_mem: usize)` — region-major streaming write of `intervals.npy` + `offsets.npy`, building trees lazily per contig (rebuild on `chrom_code` change), raising on a single `(region)` whose materialized bytes exceed `max_mem`.

- [ ] **Step 1: Write the failing cargo test (write_track byte-identical to count+intervals oracle)**

Add to the `tests` module:

```rust
    #[test]
    fn write_track_matches_oracle_bytes() {
        let t = toy();
        // two regions on chr0, one on chr1, in contig-grouped order
        let chrom_codes = [0i32, 0, 1];
        let qs = [0i32, 5, 0];
        let qe = [60i32, 55, 5];
        let sel = [0i32, 1];
        let tmp = std::env::temp_dir().join("gvl_table_write_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        t.write_track_impl(&tmp, &chrom_codes, &qs, &qe, &sel, 1 << 30)
            .unwrap();

        // Oracle: per-contig count -> offsets -> intervals, concatenated in region order.
        let mut exp_itv: Vec<u8> = Vec::new();
        let mut exp_off: Vec<u8> = Vec::new();
        let mut acc = 0i64;
        exp_off.extend_from_slice(&acc.to_le_bytes());
        // group regions by contig preserving order
        let mut ri = 0usize;
        while ri < chrom_codes.len() {
            let c = chrom_codes[ri];
            let mut rj = ri;
            while rj < chrom_codes.len() && chrom_codes[rj] == c {
                rj += 1;
            }
            let cs = &qs[ri..rj];
            let ce = &qe[ri..rj];
            let counts = t.count(c, cs, ce, &sel);
            let offsets = offsets_from_count(&counts);
            let (coords, vals) = t.intervals_from_offsets(c, cs, ce, &sel, &offsets);
            for i in 0..vals.len() {
                exp_itv.extend_from_slice(&coords[[i, 0]].to_le_bytes());
                exp_itv.extend_from_slice(&coords[[i, 1]].to_le_bytes());
                exp_itv.extend_from_slice(&vals[i].to_le_bytes());
            }
            for k in 0..counts.len() {
                acc += counts.as_slice().unwrap()[k] as i64;
                exp_off.extend_from_slice(&acc.to_le_bytes());
            }
            ri = rj;
        }
        let got_itv = std::fs::read(tmp.join("intervals.npy")).unwrap();
        let got_off = std::fs::read(tmp.join("offsets.npy")).unwrap();
        assert_eq!(got_itv, exp_itv, "intervals bytes mismatch");
        assert_eq!(got_off, exp_off, "offsets bytes mismatch");
    }

    #[test]
    fn write_track_errors_when_region_exceeds_max_mem() {
        let t = toy();
        let tmp = std::env::temp_dir().join("gvl_table_write_oom");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // region [0,60) on chr0 with both samples has >=1 interval -> exceeds 1 byte
        let res = t.write_track_impl(&tmp, &[0i32], &[0i32], &[60i32], &[0i32, 1], 1);
        assert!(res.is_err());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev cargo test --lib tables::tests::write_track`
Expected: compile error (`write_track_impl` not found).

- [ ] **Step 3: Implement the streaming writer (pure-Rust core)**

Add to `impl RustTable`. The struct value bytes use 12 bytes/row matching `INTERVAL_DTYPE` (i32,i32,f32, align=True → exactly 12, no padding).

```rust
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

impl RustTable {
    #[allow(clippy::too_many_arguments)]
    pub fn write_track_impl(
        &self,
        out_dir: &Path,
        chrom_codes: &[i32],
        q_starts: &[i32],
        q_ends: &[i32],
        sel_samples: &[i32],
        max_mem: usize,
    ) -> Result<()> {
        std::fs::create_dir_all(out_dir)?;
        let mut itv_w = BufWriter::new(File::create(out_dir.join("intervals.npy"))?);
        let mut off_w = BufWriter::new(File::create(out_dir.join("offsets.npy"))?);

        let n_regions = chrom_codes.len();
        let mut acc: i64 = 0;
        off_w.write_all(&acc.to_le_bytes())?; // leading 0

        // Lazy per-contig trees, rebuilt when chrom_code changes (bed is contig-grouped).
        let mut cur_chrom: i32 = -2;
        let mut trees: Vec<BasicCOITree<u32, u32>> = Vec::new();

        for ri in 0..n_regions {
            let c = chrom_codes[ri];
            if c != cur_chrom {
                trees = if c < 0 {
                    Vec::new()
                } else {
                    self.build_trees(c as usize)
                };
                cur_chrom = c;
            }
            // gather this region's overlaps for all selected samples
            let mut region_rows: Vec<(i32, i32, f32)> = Vec::new();
            let mut per_cell_counts: Vec<i64> = Vec::with_capacity(sel_samples.len());
            for &s in sel_samples {
                let mut start_count = 0i64;
                if c >= 0 {
                    let cell = &self.store[c as usize].samples[s as usize];
                    let tree = &trees[s as usize];
                    let mut idxs: Vec<u32> = Vec::new();
                    tree.query(q_starts[ri], q_ends[ri] - 1, |iv| idxs.push(iv.metadata));
                    idxs.sort_unstable();
                    start_count = idxs.len() as i64;
                    for &k in &idxs {
                        let k = k as usize;
                        region_rows.push((cell.starts[k], cell.ends[k], cell.values[k]));
                    }
                }
                per_cell_counts.push(start_count);
            }
            // max_mem guard: one region's total materialized bytes
            let region_bytes = region_rows.len() * 12;
            if region_bytes > max_mem {
                anyhow::bail!(
                    "Memory usage per region exceeds max_mem ({} > {}).",
                    region_bytes,
                    max_mem
                );
            }
            // write region rows (already in cell-major, start-sorted order)
            for (s, e, v) in &region_rows {
                itv_w.write_all(&s.to_le_bytes())?;
                itv_w.write_all(&e.to_le_bytes())?;
                itv_w.write_all(&v.to_le_bytes())?;
            }
            // write per-cell offsets
            for n in per_cell_counts {
                acc += n;
                off_w.write_all(&acc.to_le_bytes())?;
            }
        }
        itv_w.flush()?;
        off_w.flush()?;
        Ok(())
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev cargo test --lib tables::tests::write_track`
Expected: both write-track tests PASS.

- [ ] **Step 5: Add the PyO3 method wrappers**

Add a `#[pymethods] impl RustTable` block (separate from the plain `impl`). Put it at the end of the non-test code in `src/tables.rs`:

```rust
use numpy::{prelude::*, PyArray1, PyArray2, PyReadonlyArray1};

#[pymethods]
impl RustTable {
    #[new]
    fn py_new(
        sample_codes: PyReadonlyArray1<i32>,
        chrom_codes: PyReadonlyArray1<i32>,
        starts: PyReadonlyArray1<i32>,
        ends: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        n_samples: usize,
        n_contigs: usize,
    ) -> RustTable {
        RustTable::build(
            sample_codes.as_array(),
            chrom_codes.as_array(),
            starts.as_array(),
            ends.as_array(),
            values.as_array(),
            n_samples,
            n_contigs,
        )
    }

    #[pyo3(name = "count")]
    fn py_count<'py>(
        &self,
        py: Python<'py>,
        chrom_code: i32,
        starts: PyReadonlyArray1<i32>,
        ends: PyReadonlyArray1<i32>,
        sel_samples: PyReadonlyArray1<i32>,
    ) -> Bound<'py, PyArray2<i32>> {
        let out = self.count(
            chrom_code,
            starts.as_array().as_slice().unwrap(),
            ends.as_array().as_slice().unwrap(),
            sel_samples.as_array().as_slice().unwrap(),
        );
        out.into_pyarray(py)
    }

    #[pyo3(name = "intervals")]
    fn py_intervals<'py>(
        &self,
        py: Python<'py>,
        chrom_code: i32,
        starts: PyReadonlyArray1<i32>,
        ends: PyReadonlyArray1<i32>,
        sel_samples: PyReadonlyArray1<i32>,
        offsets: PyReadonlyArray1<i64>,
    ) -> (Bound<'py, PyArray2<i32>>, Bound<'py, PyArray1<f32>>) {
        let (coords, vals) = self.intervals_from_offsets(
            chrom_code,
            starts.as_array().as_slice().unwrap(),
            ends.as_array().as_slice().unwrap(),
            sel_samples.as_array().as_slice().unwrap(),
            offsets.as_array().as_slice().unwrap(),
        );
        (coords.into_pyarray(py), vals.into_pyarray(py))
    }

    #[pyo3(name = "write_track")]
    #[allow(clippy::too_many_arguments)]
    fn py_write_track(
        &self,
        out_dir: std::path::PathBuf,
        chrom_codes: PyReadonlyArray1<i32>,
        starts: PyReadonlyArray1<i32>,
        ends: PyReadonlyArray1<i32>,
        sel_samples: PyReadonlyArray1<i32>,
        max_mem: usize,
    ) -> PyResult<()> {
        self.write_track_impl(
            &out_dir,
            chrom_codes.as_array().as_slice().unwrap(),
            starts.as_array().as_slice().unwrap(),
            ends.as_array().as_slice().unwrap(),
            sel_samples.as_array().as_slice().unwrap(),
            max_mem,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
```

- [ ] **Step 6: Build the extension and verify it imports**

Run: `pixi run -e dev maturin develop`
Then: `pixi run -e dev python -c "from genvarloader.genvarloader import RustTable; print(RustTable)"`
Expected: prints the class, no error.

- [ ] **Step 7: Commit**

```bash
rtk git add src/tables.rs
rtk git commit -m "feat(rust): streaming Table writer + PyO3 RustTable methods

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Python `Table` on `RustTable` + un-gate unit tests

**Files:**
- Rewrite: `python/genvarloader/_table.py`
- Modify: `tests/unit/test_table.py`

**Interfaces:**
- Consumes: `genvarloader.genvarloader.RustTable` (Task 4).
- Produces: `Table` class with the same public surface (`name`, `samples`, `contigs`, `__init__`, `from_path`, `count_intervals`, `_intervals_from_offsets`) backed by `RustTable`; module-level `annot_overlap(regions, annot) -> RaggedIntervals`. No polars-bio, no `ExperimentalWarning`.

- [ ] **Step 1: Un-gate and retarget the unit test (write the failing test)**

In `tests/unit/test_table.py`, replace the header (lines 1–27) with:

```python
import numpy as np
import polars as pl
import pytest
from genvarloader import Table
from genvarloader._utils import lengths_to_offsets
```

(Delete the `os` import, the `GVL_TEST_EXPERIMENTAL` skip block, and the `pytestmark` filterwarnings line — `Table` no longer warns.) Leave the rest of the file unchanged.

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_table.py -q`
Expected: FAIL — `ImportError: cannot import name 'Table' from 'genvarloader'` (not yet exported) or constructor still warns/uses polars-bio.

- [ ] **Step 3: Rewrite `_table.py`**

Replace the entire contents of `python/genvarloader/_table.py` with:

```python
"""Tabular interval track source for :func:`gvl.write()`.

Mirrors the :class:`BigWigs` reader API surface so that
:func:`genvarloader._dataset._write._write_track` can dispatch to either.
Overlap queries are served by the Rust ``RustTable`` (COITrees) backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from genoray._utils import ContigNormalizer

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._ragged import RaggedIntervals


CANONICAL_COLS = ("sample_id", "chrom", "start", "end", "value")


class Table:
    """Long-form interval track keyed by ``(sample_id, chrom, start, end, value)``.

    Overlap queries are served by a Rust COITrees backend. Coordinates are
    zero-based, half-open ``[start, end)``.
    """

    name: str
    samples: list[str]
    contigs: Mapping[str, int]

    def __init__(
        self,
        name: str,
        data: pl.DataFrame | Mapping[str, pl.DataFrame],
        column_map: Mapping[str, str] | None = None,
    ) -> None:
        from .genvarloader import RustTable

        self.name = name
        df = self._normalize_input(data, column_map)
        df = df.cast(
            {
                "sample_id": pl.Utf8,
                "chrom": pl.Utf8,
                "start": pl.Int32,
                "end": pl.Int32,
                "value": pl.Float32,
            }
        ).sort("chrom", "sample_id", "start")
        self._df = df
        self.samples = sorted(df["sample_id"].unique().to_list())
        self.contigs = {
            row["chrom"]: int(row["max_end"])
            for row in df.group_by("chrom")
            .agg(pl.col("end").max().alias("max_end"))
            .iter_rows(named=True)
        }

        # Factor-encode for the Rust store. Contig order is the dict insertion
        # order of `self.contigs`; sample order is `self.samples`.
        self._contig_list = list(self.contigs.keys())
        self._cnorm = ContigNormalizer(self._contig_list)
        sample_to_code = {s: i for i, s in enumerate(self.samples)}
        sample_codes = (
            df.select(
                pl.col("sample_id").replace_strict(
                    sample_to_code, return_dtype=pl.Int32
                )
            )
            .to_series()
            .to_numpy()
        )
        chrom_to_code = {c: i for i, c in enumerate(self._contig_list)}
        chrom_codes = (
            df.select(
                pl.col("chrom").replace_strict(chrom_to_code, return_dtype=pl.Int32)
            )
            .to_series()
            .to_numpy()
        )
        self._rust = RustTable(
            np.ascontiguousarray(sample_codes, dtype=np.int32),
            np.ascontiguousarray(chrom_codes, dtype=np.int32),
            np.ascontiguousarray(df["start"].to_numpy(), dtype=np.int32),
            np.ascontiguousarray(df["end"].to_numpy(), dtype=np.int32),
            np.ascontiguousarray(df["value"].to_numpy(), dtype=np.float32),
            len(self.samples),
            len(self._contig_list),
        )

    @classmethod
    def from_path(
        cls,
        name: str,
        path: str | Path | Mapping[str, str | Path],
        column_map: Mapping[str, str] | None = None,
    ) -> Table:
        if isinstance(path, Mapping):
            data: dict[str, pl.DataFrame] = {
                sid: cls._read_path(Path(p)) for sid, p in path.items()
            }
            return cls(name, data, column_map)
        return cls(name, cls._read_path(Path(path)), column_map)

    @staticmethod
    def _read_path(p: Path) -> pl.DataFrame:
        suf = p.suffix.lower()
        if suf == ".csv":
            return pl.read_csv(p)
        if suf in (".tsv", ".txt"):
            return pl.read_csv(p, separator="\t")
        if suf == ".parquet":
            return pl.read_parquet(p)
        if suf in (".arrow", ".ipc"):
            return pl.read_ipc(p)
        raise ValueError(
            f"Unsupported file extension {suf!r}. "
            "Expected one of .csv, .tsv, .txt, .parquet, .arrow, .ipc."
        )

    @staticmethod
    def _normalize_input(
        data: pl.DataFrame | Mapping[str, pl.DataFrame],
        column_map: Mapping[str, str] | None,
    ) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            df = Table._apply_column_map(data, column_map, expect_sample_id=True)
        else:
            frames: list[pl.DataFrame] = []
            for sid, sub in data.items():
                renamed = Table._apply_column_map(
                    sub, column_map, expect_sample_id=False
                )
                frames.append(renamed.with_columns(sample_id=pl.lit(sid)))
            if not frames:
                raise ValueError("Empty mapping passed to Table.")
            df = pl.concat(frames, how="vertical_relaxed")
        missing = [c for c in CANONICAL_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s) {missing}. "
                f"Use `column_map` to rename if your columns differ from {CANONICAL_COLS}."
            )
        return df.select(*CANONICAL_COLS)

    @staticmethod
    def _apply_column_map(
        df: pl.DataFrame,
        column_map: Mapping[str, str] | None,
        expect_sample_id: bool,
    ) -> pl.DataFrame:
        if not column_map:
            return df
        rename = {
            actual: canonical
            for canonical, actual in column_map.items()
            if actual in df.columns
        }
        if not expect_sample_id:
            rename.pop("sample_id", None)
        return df.rename(rename)

    def _resolve_samples(self, sample: str | list[str] | None) -> list[str]:
        if sample is None:
            return list(self.samples)
        if isinstance(sample, str):
            samples = [sample]
        else:
            samples = list(sample)
        if missing := set(samples) - set(self.samples):
            raise ValueError(f"Sample(s) {missing} not found in Table.")
        return samples

    def _chrom_code(self, contig: str) -> int:
        """Resolve `contig` to its code in this Table, or -1 if absent."""
        norm = self._cnorm.norm(contig)
        if norm is None:
            return -1
        return self._contig_list.index(norm)

    def _sample_codes(self, samples: list[str]) -> NDArray[np.int32]:
        s2c = {s: i for i, s in enumerate(self.samples)}
        return np.array([s2c[s] for s in samples], dtype=np.int32)

    def count_intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> NDArray[np.int32]:
        samples = self._resolve_samples(sample)
        starts_arr = np.ascontiguousarray(np.atleast_1d(starts), dtype=np.int32)
        ends_arr = np.ascontiguousarray(np.atleast_1d(ends), dtype=np.int32)
        return self._rust.count(
            self._chrom_code(contig),
            starts_arr,
            ends_arr,
            self._sample_codes(samples),
        )

    def _intervals_from_offsets(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> RaggedIntervals:
        from seqpro.rag import Ragged

        from ._ragged import RaggedIntervals

        samples = self._resolve_samples(sample)
        starts_arr = np.ascontiguousarray(np.atleast_1d(starts), dtype=np.int32)
        ends_arr = np.ascontiguousarray(np.atleast_1d(ends), dtype=np.int32)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        n_regions = len(starts_arr)
        n_samples = len(samples)
        shape = (n_regions, n_samples, None)

        coords, values = self._rust.intervals(
            self._chrom_code(contig),
            starts_arr,
            ends_arr,
            self._sample_codes(samples),
            offsets,
        )
        flat_starts = np.ascontiguousarray(coords[:, 0], dtype=np.int32)
        flat_ends = np.ascontiguousarray(coords[:, 1], dtype=np.int32)
        return RaggedIntervals(
            Ragged.from_offsets(flat_starts, shape, offsets),
            Ragged.from_offsets(flat_ends, shape, offsets),
            Ragged.from_offsets(values, shape, offsets),
        )


def annot_overlap(regions: pl.DataFrame, annot: pl.DataFrame) -> "RaggedIntervals":
    """Sample-less interval overlap; real implementation added in Step 4."""
    raise NotImplementedError
```

> **NOTE:** `annot_overlap` is stubbed here so the file imports but `test_write_annot.py`
> still fails (forcing the gate). Step 4 replaces the stub with the real implementation.
> If you are reading tasks out of order, skip straight to the Step 4 version.

- [ ] **Step 4: Replace `annot_overlap` with the clean implementation**

Replace the entire `annot_overlap` function with:

```python
def annot_overlap(regions: pl.DataFrame, annot: pl.DataFrame) -> "RaggedIntervals":
    """Sample-less interval overlap of `regions` (chrom/chromStart/chromEnd) against a
    BED-like `annot` (chrom/chromStart/chromEnd/score). Returns a RaggedIntervals of
    shape (n_regions, None) ordered by (region, start), via the Rust COITrees backend."""
    from seqpro.rag import Ragged

    from ._ragged import RaggedIntervals
    from ._utils import lengths_to_offsets

    annot_long = annot.select(
        pl.lit("__annot__").alias("sample_id"),
        "chrom",
        pl.col("chromStart").alias("start"),
        pl.col("chromEnd").alias("end"),
        pl.col("score").alias("value"),
    )
    table = Table("__annot__", annot_long)

    n_regions = regions.height
    # Per-region interval arrays indexed by input region position.
    per_start: list[NDArray[np.int32]] = [np.empty(0, np.int32)] * n_regions
    per_end: list[NDArray[np.int32]] = [np.empty(0, np.int32)] * n_regions
    per_val: list[NDArray[np.float32]] = [np.empty(0, np.float32)] * n_regions

    reg = regions.with_row_index("_q")
    for (contig,), part in reg.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        contig = str(contig)
        q_idx = part["_q"].to_numpy()
        starts = part["chromStart"].to_numpy()
        ends = part["chromEnd"].to_numpy()
        counts = table.count_intervals(contig, starts, ends, sample=["__annot__"])
        offsets = lengths_to_offsets(counts.ravel())
        itvs = table._intervals_from_offsets(
            contig, starts, ends, offsets, sample=["__annot__"]
        ).squeeze(1)
        for j, qi in enumerate(q_idx):
            per_start[qi] = np.asarray(itvs.starts[j], dtype=np.int32)
            per_end[qi] = np.asarray(itvs.ends[j], dtype=np.int32)
            per_val[qi] = np.asarray(itvs.values[j], dtype=np.float32)

    lengths = np.array([len(s) for s in per_start], dtype=np.int32)
    offsets = lengths_to_offsets(lengths)
    flat_starts = np.concatenate(per_start) if n_regions else np.empty(0, np.int32)
    flat_ends = np.concatenate(per_end) if n_regions else np.empty(0, np.int32)
    flat_values = np.concatenate(per_val) if n_regions else np.empty(0, np.float32)
    shape = (n_regions, None)
    return RaggedIntervals(
        Ragged.from_offsets(flat_starts, shape, offsets),
        Ragged.from_offsets(flat_ends, shape, offsets),
        Ragged.from_offsets(flat_values, shape, offsets),
    )
```

Also delete the now-unused placeholder `_region_order` reference and any leftover lines from Step 3's incomplete tail.

- [ ] **Step 5: Export `Table` from the package**

In `python/genvarloader/__init__.py`, add `Table` to the imports and `__all__`. Find the import block and `__all__` list and add:

```python
from ._table import Table
```

and add `"Table"` to `__all__`.

- [ ] **Step 6: Run the unit tests to verify they pass**

Run: `pixi run -e dev maturin develop && pixi run -e dev pytest tests/unit/test_table.py -q`
Expected: all tests PASS (including `test_table_count_intervals_matches_brute_force`, `_normalizes_contig_names`, `_intervals_from_offsets_roundtrip`).

- [ ] **Step 7: Commit**

```bash
rtk git add python/genvarloader/_table.py python/genvarloader/__init__.py tests/unit/test_table.py
rtk git commit -m "feat: back gvl.Table with Rust COITrees engine; drop polars-bio

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Wire write path + annot path + `c_idxs`; un-gate write/annot tests

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py`
- Modify: `tests/unit/dataset/test_write_tracks.py`
- Modify: `tests/unit/test_write_annot.py`
- Modify: `tests/integration/dataset/test_write_tracks_e2e.py`

**Interfaces:**
- Consumes: `Table` (Task 5) with `_rust.write_track`, `_chrom_code`, `_sample_codes`.
- Produces: `_write_track` routes `Table → _write_track_table`; `_write_track_table(out_dir, bed, track, samples, max_mem)`; `_annot_intervals` keeps calling `annot_overlap` (now Rust-backed); `_write_track_rust` uses `ContigNormalizer.c_idxs`.

- [ ] **Step 1: Un-gate the write-tracks + annot tests (write failing tests)**

- In `tests/unit/dataset/test_write_tracks.py`: delete the `GVL_TEST_EXPERIMENTAL` skip block and change `from genvarloader.experimental import Table` to `from genvarloader import Table`. Remove any `ExperimentalWarning` filter.
- In `tests/integration/dataset/test_write_tracks_e2e.py`: same two edits.
- In `tests/unit/test_write_annot.py`: delete the skip block (lines ~7–11) and change `from genvarloader._table import annot_overlap` to keep working (it stays in `_table`).

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run -e dev pytest tests/unit/dataset/test_write_tracks.py tests/unit/test_write_annot.py -q`
Expected: FAIL — Table writes still go through `_write_track_legacy` (polars-bio path) which now errors (polars-bio import removed) or produces different bytes.

- [ ] **Step 3: Add `_write_track_table` and route to it**

In `python/genvarloader/_dataset/_write.py`, modify `_write_track` (around line 1386) to add a Table branch:

```python
def _write_track(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "IntervalTrack",
    samples: list[str] | None,
    max_mem: int,
):
    from .._bigwig import BigWigs
    from .._table import Table

    if isinstance(track, BigWigs):
        _samples = samples if samples is not None else track.samples
        if missing := (set(_samples) - set(track.samples)):
            raise ValueError(f"Samples {missing} not found in track.")
        return _write_track_rust(out_dir, bed, track, _samples, max_mem)
    if isinstance(track, Table):
        _samples = samples if samples is not None else track.samples
        if missing := (set(_samples) - set(track.samples)):
            raise ValueError(f"Samples {missing} not found in track.")
        return _write_track_table(out_dir, bed, track, _samples, max_mem)
    return _write_track_legacy(out_dir, bed, track, samples, max_mem)
```

Add the new function next to `_write_track_rust`:

```python
def _write_track_table(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "Table",
    samples: list[str],
    max_mem: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # bed is contig-grouped (sp.bed.sort). Map per-region chrom -> Table contig code.
    chrom_codes = track._cnorm.c_idxs(bed["chrom"].to_numpy())
    # c_idxs maps via auto-normalization; regions whose contig is absent from the
    # Table must map to -1 (zero intervals). Detect via norm().
    norm = track._cnorm.norm(bed["chrom"].to_list())
    chrom_codes = np.where(
        np.array([n is None for n in norm]), -1, chrom_codes
    ).astype(np.int32)
    starts = np.ascontiguousarray(bed["chromStart"].to_numpy(), dtype=np.int32)
    ends = np.ascontiguousarray(bed["chromEnd"].to_numpy(), dtype=np.int32)
    track._rust.write_track(
        str(out_dir),
        np.ascontiguousarray(chrom_codes, dtype=np.int32),
        starts,
        ends,
        track._sample_codes(samples),
        int(max_mem),
    )
```

> **NOTE on `c_idxs` + `ContigNormalizer`:** `track._cnorm` is built over `track._contig_list` (the Table's own contigs, same order as the Rust store), so `c_idxs` returns codes that index the Rust store correctly. For region contigs absent from the Table, `norm()` returns `None`; we force those to `-1`.

- [ ] **Step 4: Switch `_write_track_rust` contig loop to `c_idxs`**

In `_write_track_rust` (around line 1360–1377), the per-row `normalize_contig_name` loop builds `contigs`, `starts_l`, `ends_l`. Replace it with vectorized normalization. Note `_write_track_rust` passes contig *names* (strings) to `bigwig_write_track`, so map codes back to the track's contig names:

```python
def _write_track_rust(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "BigWigs",
    samples: list[str],
    max_mem: int,
) -> None:
    from ..genvarloader import bigwig_write_track

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [track.paths[s] for s in samples]

    track_contigs = list(track.contigs)
    cnorm = ContigNormalizer(track_contigs)
    norm = cnorm.norm(bed["chrom"].to_list())
    if any(n is None for n, c in zip(norm, bed["chrom"].to_list())):
        bad = next(c for n, c in zip(norm, bed["chrom"].to_list()) if n is None)
        raise ValueError(f"Contig {bad!r} not found in bigWig track {track.name!r}.")
    contigs = [str(n) for n in norm]
    starts = np.ascontiguousarray(bed["chromStart"].to_numpy(), dtype=np.int32)
    ends = np.ascontiguousarray(bed["chromEnd"].to_numpy(), dtype=np.int32)
    bigwig_write_track(
        paths,
        contigs,
        starts,
        ends,
        int(max_mem),
        str(out_dir),
        False,
    )
```

(Confirm `ContigNormalizer` is imported at the top of `_write.py` — it is, at line 24. Keep the trailing positional args to `bigwig_write_track` matching the existing signature: `(paths, contigs, starts, ends, max_mem, out_dir, sample_less)`.)

- [ ] **Step 5: Remove the polars-bio doc note in `_annot_intervals`**

In `_annot_intervals` (around line 1108–1112) update the docstring line that says "polars-bio overlap (experimental, requires the `table` extra)" to "Rust COITrees overlap". No logic change — it already calls `annot_overlap`.

- [ ] **Step 6: Run the write/annot tests to verify they pass**

Run: `pixi run -e dev maturin develop && pixi run -e dev pytest tests/unit/dataset/test_write_tracks.py tests/unit/test_write_annot.py tests/integration/dataset/test_write_tracks_e2e.py -q`
Expected: PASS.

- [ ] **Step 7: Add a memory-regression test**

Create `tests/unit/dataset/test_table_max_mem.py`:

```python
"""max_mem is respected by the Rust Table writer (the polars-bio blow-up regression)."""

import numpy as np
import polars as pl
import pytest
from genvarloader import Table
from genvarloader._dataset._write import _write_track_table


def _dense_table(n_intervals: int) -> Table:
    starts = np.arange(0, n_intervals * 10, 10, dtype=np.int64)
    return Table(
        "signal",
        pl.DataFrame(
            {
                "sample_id": ["s0"] * n_intervals,
                "chrom": ["chr1"] * n_intervals,
                "start": starts,
                "end": starts + 5,
                "value": np.ones(n_intervals, np.float32),
            }
        ),
    )


def test_write_track_table_raises_when_region_exceeds_max_mem(tmp_path):
    t = _dense_table(1000)
    bed = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [10_000]}
    )
    # One region overlaps ~1000 intervals = ~12 KB; cap at 12 bytes -> must raise.
    with pytest.raises(RuntimeError, match="max_mem"):
        _write_track_table(tmp_path, bed, t, ["s0"], max_mem=12)


def test_write_track_table_succeeds_within_budget(tmp_path):
    t = _dense_table(1000)
    bed = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [10_000]}
    )
    _write_track_table(tmp_path, bed, t, ["s0"], max_mem=1 << 20)
    assert (tmp_path / "intervals.npy").exists()
    assert (tmp_path / "offsets.npy").exists()
```

Run: `pixi run -e dev pytest tests/unit/dataset/test_table_max_mem.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/unit/dataset/test_write_tracks.py tests/unit/test_write_annot.py tests/integration/dataset/test_write_tracks_e2e.py tests/unit/dataset/test_table_max_mem.py
rtk git commit -m "feat: route Table/annot writes through Rust; vectorize contig norm

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Hard-move Table to public API; delete experimental subpackage

**Files:**
- Delete: `python/genvarloader/experimental/__init__.py` (and the `experimental/` dir)
- Modify: `docs/source/api.md`
- Modify: `tests/dataset/test_with_methods.py` (stale comment only)

**Interfaces:**
- Consumes: `genvarloader.Table` (exported in Task 5).
- Produces: no `genvarloader.experimental` module; `genvarloader.Table` is the only public path.

- [ ] **Step 1: Find every remaining reference to the experimental Table**

Run:

```bash
rtk grep -rn "genvarloader.experimental\|from .experimental\|import experimental\|ExperimentalWarning" python/ tests/ docs/source/
```

Expected matches: `docs/source/api.md:20`, and any test still importing from `experimental` (should be none after Tasks 5–6). Fix each:
- `docs/source/api.md`: change `.. autoclass:: genvarloader.experimental.Table` to `.. autoclass:: genvarloader.Table`.
- Any test import → `from genvarloader import Table`.

- [ ] **Step 2: Delete the experimental subpackage**

```bash
rtk git rm -r python/genvarloader/experimental
```

- [ ] **Step 3: Fix the stale comment in `test_with_methods.py`**

`tests/dataset/test_with_methods.py:~32` references "the polars_bio C extension in a bad state, which segfaulted". Update the comment to reflect that polars-bio is gone (Table now uses the Rust backend). Comment-only change; no logic.

- [ ] **Step 4: Verify nothing imports the deleted module**

Run: `pixi run -e dev python -c "import genvarloader; print(genvarloader.Table)"`
Expected: prints `<class 'genvarloader._table.Table'>`, no error.

Run: `rtk grep -rn "experimental" python/genvarloader/`
Expected: no matches (or only unrelated strings).

- [ ] **Step 5: Run the full unit + dataset suites**

Run: `pixi run -e dev pytest tests/unit tests/dataset -q`
Expected: PASS (no collection errors from missing `experimental`).

- [ ] **Step 6: Commit**

```bash
rtk git add -A
rtk git commit -m "feat!: promote gvl.Table to public API; remove experimental subpackage

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Drop the `table` extra; docs, skill, roadmap, memory

**Files:**
- Modify: `pyproject.toml`
- Modify: `skills/genvarloader/SKILL.md`
- Modify: `docs/roadmaps/rust-migration.md`
- Modify (memory): `/Users/david/.claude/projects/-Users-david-projects-GenVarLoader/memory/project_polars_bio_segfault.md` + `MEMORY.md`

**Interfaces:**
- Consumes: completed Tasks 1–7.
- Produces: no `[table]` extra; docs/skill/roadmap reflect Rust-backed, non-experimental Table.

- [ ] **Step 1: Remove the `table` extra from pyproject**

In `pyproject.toml`, delete the comment block (lines ~42–44) and the `table = ['polars-bio']` line (line ~45) under `[project.optional-dependencies]`. If `table` is referenced in any `all`/aggregate extra, remove it there too — check with:

```bash
rtk grep -n "table" pyproject.toml
```

- [ ] **Step 2: Verify the package still resolves**

Run: `pixi run -e dev python -c "import genvarloader; import polars_bio" 2>&1 | head -1`
Expected: `genvarloader` imports fine; `polars_bio` ImportError is acceptable/expected (it's no longer a dependency). The key check: `python -c "import genvarloader"` succeeds.

- [ ] **Step 3: Update the SKILL.md**

In `skills/genvarloader/SKILL.md`, find the `Table` section and the "Common gotchas" / pointer table. Remove: the `[table]` extra install instruction, the polars-bio mention, and the "experimental" framing. State that `gvl.Table` is a core interval-track source backed by a Rust COITrees overlap engine, zero-based half-open coordinates, usable directly as a `tracks=`/`annot_tracks=` source in `gvl.write`. Keep the constructor/`from_path` usage examples (they are unchanged).

- [ ] **Step 4: Update the roadmap**

In `docs/roadmaps/rust-migration.md`:
- Under **Phase 4**, add a checked sub-task under the write-pipeline item:
  `- [x] Table + annot overlap: COITrees Rust engine replaces polars-bio (this PR)`
- Add a dated entry to the **Notes & decisions log**:

```markdown
- 2026-06-19: Ported gvl.Table + annot_overlap off polars-bio onto a COITrees Rust
  engine (`src/tables.rs`, `RustTable` PyO3 class). Fixes max_mem disrespect during
  write/update (counting is exact, streaming writer bounds the working set to one
  region's overlaps + one contig's trees), removes the non-deterministic polars-bio
  segfault (#395), drops the `[table]` extra, and promotes Table from
  `genvarloader.experimental` to the public API (now CI-covered via a brute-force
  numpy oracle + property tests). Coordinates half-open/zero-based; positive-width
  intervals assumed.
```

- [ ] **Step 5: Update memory**

Edit `/Users/david/.claude/projects/-Users-david-projects-GenVarLoader/memory/project_polars_bio_segfault.md`: note that as of 2026-06-19 (PR `feat/rust-table-overlap`) gvl.Table + annot_overlap no longer use polars-bio (Rust COITrees backend); the `[table]` extra is removed and Table is a public, CI-covered API. Keep the historical context. Update the one-line pointer in `MEMORY.md` accordingly.

- [ ] **Step 6: Commit**

```bash
rtk git add pyproject.toml skills/genvarloader/SKILL.md docs/roadmaps/rust-migration.md
rtk git commit -m "docs: drop table extra; de-experimentalize Table; update roadmap+skill

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Property-based parity + full verification

**Files:**
- Create: `tests/unit/test_table_parity.py`

**Interfaces:**
- Consumes: `genvarloader.Table`, `annot_overlap`.
- Produces: hypothesis property tests asserting byte-identical parity vs a brute-force numpy oracle across `count_intervals`, `_intervals_from_offsets`, and `annot_overlap`.

- [ ] **Step 1: Write the property test**

Create `tests/unit/test_table_parity.py`:

```python
"""Property-based parity: Rust Table vs a brute-force numpy oracle."""

import numpy as np
import polars as pl
from genvarloader import Table
from genvarloader._table import annot_overlap
from genvarloader._utils import lengths_to_offsets
from hypothesis import given, settings
from hypothesis import strategies as st


def _rand_table(rng, n_samples, n_contigs, n_intervals):
    rows = []
    for _ in range(n_intervals):
        s = int(rng.integers(0, n_samples))
        c = int(rng.integers(0, n_contigs))
        start = int(rng.integers(0, 500))
        width = int(rng.integers(1, 50))  # positive width only
        rows.append((f"s{s}", f"chr{c}", start, start + width, float(rng.random())))
    df = pl.DataFrame(
        rows, schema=["sample_id", "chrom", "start", "end", "value"], orient="row"
    )
    return df


def _brute(df, contig, starts, ends, samples):
    counts = np.zeros((len(starts), len(samples)), np.int32)
    cells = {}
    for si, s in enumerate(samples):
        sub = df.filter((pl.col("sample_id") == s) & (pl.col("chrom") == contig)).sort(
            "start"
        )
        ts = sub["start"].to_numpy()
        te = sub["end"].to_numpy()
        tv = sub["value"].to_numpy()
        for ri, (rs, re_) in enumerate(zip(starts, ends)):
            mask = (ts < re_) & (te > rs)
            counts[ri, si] = int(mask.sum())
            cells[(ri, si)] = (
                ts[mask].astype(np.int32),
                te[mask].astype(np.int32),
                tv[mask].astype(np.float32),
            )
    return counts, cells


@settings(max_examples=100, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n_samples=st.integers(1, 3),
    n_contigs=st.integers(1, 3),
    n_intervals=st.integers(0, 40),
    n_regions=st.integers(1, 6),
)
def test_count_and_intervals_match_oracle(
    seed, n_samples, n_contigs, n_intervals, n_regions
):
    rng = np.random.default_rng(seed)
    df = _rand_table(rng, n_samples, n_contigs, n_intervals)
    if df.height == 0:
        return
    t = Table("sig", df)
    samples = [f"s{i}" for i in range(n_samples)]
    for c in range(n_contigs):
        contig = f"chr{c}"
        starts = rng.integers(0, 500, n_regions).astype(np.int32)
        ends = (starts + rng.integers(1, 100, n_regions)).astype(np.int32)
        present = [s for s in samples if s in t.samples]
        if not present:
            continue
        counts = t.count_intervals(contig, starts, ends, sample=present)
        exp_counts, cells = _brute(df, contig, starts, ends, present)
        np.testing.assert_array_equal(counts, exp_counts)

        offsets = lengths_to_offsets(counts.ravel())
        itvs = t._intervals_from_offsets(
            contig, starts, ends, offsets, sample=present
        )
        n_sel = len(present)
        for ri in range(n_regions):
            for sj in range(n_sel):
                cell = ri * n_sel + sj
                lo, hi = int(offsets[cell]), int(offsets[cell + 1])
                exp_s, exp_e, exp_v = cells[(ri, sj)]
                np.testing.assert_array_equal(itvs.starts.data[lo:hi], exp_s)
                np.testing.assert_array_equal(itvs.ends.data[lo:hi], exp_e)
                np.testing.assert_array_equal(itvs.values.data[lo:hi], exp_v)
```

- [ ] **Step 2: Run the property test**

Run: `pixi run -e dev pytest tests/unit/test_table_parity.py -q`
Expected: PASS (100 examples).

- [ ] **Step 3: Run the full Rust + Python suite (per CLAUDE.md, full tree)**

Run:

```bash
pixi run -e dev cargo test --release
pixi run -e dev pytest tests -q
```

Expected: all PASS. (If any pre-existing slow/torch tiers are environment-gated, they behave as before — this change doesn't touch them.)

- [ ] **Step 4: Lint + format + typecheck**

Run:

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
```

Expected: clean. Fix any issues (e.g. unused imports left from the `_table.py` rewrite). Re-run until clean. The pre-push hook enforces `ruff format`.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/unit/test_table_parity.py
rtk git commit -m "test: property-based parity for Rust Table vs brute-force oracle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** Tasks 1–4 = Rust engine/store/writer (spec §Architecture, §Overlap engine). Task 5 = Python shim + delete polars-bio (spec §Python shim). Task 6 = write-path wiring + `c_idxs` + memory regression (spec §Python shim, §max_mem fix, risk #2). Task 7 = hard move + delete experimental (spec §API promotion). Task 8 = extra removal, skill, roadmap, memory (spec §Roadmap, risk #4). Task 9 = brute-force oracle parity (spec §Testing).
- **coitrees API risk:** the exact callback/metadata accessor (`iv.metadata`) and `BasicCOITree::new(&ivs)` / `Interval::new` signatures must be confirmed against the pinned `coitrees` version via `cargo doc -p coitrees`. The Task 2–4 cargo tests are the gate; adjust accessors if the version differs, keeping behavior identical.
- **Ordering:** within-cell order is guaranteed by `idxs.sort_unstable()` over stored indices, which are ascending-start because Python pre-sorts by `(chrom, sample_id, start)`. Do not change the Python sort.
- **Zero-width assumption:** generators use positive width; the half-open→inclusive `end-1` mapping is only correct for `end > start`. If real data ever has zero-width intervals, revisit `build_trees`.
