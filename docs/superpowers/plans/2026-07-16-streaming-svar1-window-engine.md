# gvl StreamingDataset — SVAR1 window reads + double-buffer engine — Implementation Plan (Plan 2b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallelism:** Task 1 gates everything. Tasks 2 and 3 are independent — dispatch concurrently via superpowers:dispatching-parallel-agents. Task 4 needs 2. Task 5 needs 2+3. Task 6 needs 5.
>
> **Model policy:** use Sonnet (or weaker) for implementation subagents. Escalate only for a second-pass fix where the implementer critically failed. **Task 5 (the engine) is the exception** — it introduces gvl's first threading primitive; consider a stronger model or extra review there.

**Goal:** Replace the walking skeleton's per-batch SVAR1 read with window-granular, GIL-free reads over genoray's new ungated `svar1_query`, land the crossbeam double-buffer engine behind a generic `StreamBackend` trait, and settle `StreamingDataset`'s public surface on `to_iter` as the single iteration entry point.

**Architecture:** The **window** (R regions × all samples × ploidy, cartesian) becomes the read granularity — one `find_ranges` call per window; a **batch** is a slice of the reconstructed window, never its own read. SVAR1's window buffer is **degenerate**: its on-disk layout is already hap-major sparse CSR of sorted global variant ids, so `geno_v_idxs` is borrowed straight from the `variant_idxs` mmap and nothing is materialized. The producer/consumer engine therefore hides **I/O (page-fault) latency**, not decode.

**Tech Stack:** Rust 2021 (PyO3 0.29, ndarray, rayon, crossbeam-channel), Python 3.10+ (numpy, polars, seqpro, torch optional), pixi, maturin, pytest.

**Spec:** `docs/superpowers/specs/2026-07-16-streaming-svar1-window-engine-design.md` (✅ approved)
**Issue:** mcvickerlab/GenVarLoader#275. **Blocked on:** d-laub/genoray#123 (Plan 2a) merging to genoray `main`.

## Global Constraints

- **BLOCKED until Plan 2a merges to genoray `main`.** Do not start Task 1 before then. Consumption is via a Cargo `rev` bump — **genoray is a git dep, never a release gate** (CLAUDE.md → Development Notes). Both `genoray_core` and `svar2-codec` **must share the same rev**.
- **Byte-identical parity is the control and is non-negotiable.** `tests/dataset/test_streaming_parity.py` must stay green. Task 3 changes it **only** mechanically (`to_dataloader`/`__iter__` → `to_iter`). If it needs more than that, behavior changed — STOP and investigate.
- **Rebuild Rust before pytest.** `pixi run -e dev pytest` does **not** rebuild the extension. After any `src/` change run `pixi run -e dev maturin develop --release` first, or pytest silently imports the **stale** binary and parity tests pass/fail against the old code. (`cargo test` compiles from source and is unaffected.)
- **Commands:**
  - `pixi run -e dev maturin develop --release`
  - `pixi run -e dev pytest tests/dataset tests/unit -q`
  - `pixi run -e dev cargo-test`
  - `pixi run -e dev ruff check python/ tests/` and `ruff format python/ tests/`
  - `pixi run -e dev typecheck`
  - **Full tree before pushing:** `pixi run -e dev pytest tests -q` (scoped runs skip `tests/unit/`, so a stale reference there only fails in CI).
- **`conversion` stays enabled** in `Cargo.toml` even though nothing in gvl will use it after this plan (VCF/PGEN in #276 need it; it already builds green with static htslib). Do **not** remove it — dropping and re-adding would churn `Cargo.toml` + `pixi.toml`'s `LIBCLANG_PATH` twice.
- **Coordinates:** 0-based half-open `[start, end)` everywhere. `SparseVar.index.POS` is 1-based (subtract 1).
- **Never materialize a sample-scale array.** `variant_idxs` stays mmap'd; `v_starts`/`v_ends` stay as the numpy arrays Python already holds and cross as zero-copy `PyReadonlyArray1`. No `ascontiguousarray` on a per-sample-scale memmap.
- **Scope:** SVAR1 only, `haplotypes` only, `jitter=0`, ragged output only. No VCF/PGEN (#276), no other output modes (#277), no SVAR2 (#278). `num_workers>0` stays deferred.
- **Commits are slow** — prek runs a full `pyrefly`. Use a ≥10-minute timeout. Ensure hooks are installed: `pixi run -e dev prek-install`.

---

### Task 1: Bump the genoray rev and prove the new API links

**Files:**
- Modify: `Cargo.toml:25-26` (both genoray git deps)

**Interfaces:**
- Consumes: genoray `main` with `svar1_query` (Plan 2a).
- Produces: a build where `genoray_core::svar1_query::{Svar1Reader, var_ranges, find_ranges}` are importable from gvl's crate. **Gates every other task.**

**Background:** The current pin (`Cargo.toml:25-26`) is rev `66ba734b85fcf1326008d66b33c052c7cf278a9f` for both crates, with a comment stating the rule: *"Bump `rev` to pull newer genoray code; both crates must share the same rev (one clone, one repo)."*

- [ ] **Step 1: Get the merged genoray SHA**

```bash
cd /carter/users/dlaub/projects/genoray && git fetch origin main && git rev-parse origin/main
```
Expected: a 40-char SHA that includes the `svar1_query` module. Verify it does:
```bash
cd /carter/users/dlaub/projects/genoray && git show origin/main:src/svar1_query.rs | head -3
```
Expected: the module doc comment. **If this errors, Plan 2a has not merged — STOP.**

- [ ] **Step 2: Bump both revs**

In `Cargo.toml`, replace `66ba734b85fcf1326008d66b33c052c7cf278a9f` with the new SHA in **both** lines (`svar2-codec` and `genoray_core`). Keep `default-features = false, features = ["conversion"]` on `genoray_core` exactly as-is.

- [ ] **Step 3: Write the failing smoke test**

Append to `src/svar1/mod.rs`:

```rust
#[cfg(test)]
mod link_tests {
    /// Smoke: the ungated genoray SVAR1 query API is visible from gvl's crate.
    /// If this fails to compile, the `rev` bump did not land `svar1_query`.
    #[test]
    fn svar1_query_symbols_are_linkable() {
        use genoray_core::svar1_query::{Svar1Reader, find_ranges, var_ranges};
        let _ = Svar1Reader::open;
        let _ = var_ranges;
        let _ = find_ranges;
    }
}
```

- [ ] **Step 4: Run to verify it compiles and passes**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev cargo-test 2>&1 | grep -E "svar1_query_symbols|test result|error" | head -10`
Expected: `svar1_query_symbols_are_linkable ... ok`.

- [ ] **Step 5: Confirm no regressions**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev cargo-test 2>&1 | tail -5`
Expected: all cargo tests pass.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/svar1/mod.rs
git commit -m "build(deps): bump genoray rev for ungated svar1_query"
```

---

### Task 2: Rewrite `Svar1Store` on `svar1_query` — window reads, GIL-free

**Files:**
- Modify: `src/svar1/store.rs` (full rewrite)
- Modify: `src/svar1/mod.rs` (replace `Sparse` with `Svar1Window`)
- Modify: `src/ffi/mod.rs:786-919` (`reconstruct_haplotypes_svar1`)
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_Svar1Backend`)

**Interfaces:**
- Consumes: `genoray_core::svar1_query::{Svar1Reader, var_ranges, find_ranges}` (Task 1).
- Produces:
  ```rust
  // src/svar1/mod.rs
  pub struct Svar1Window {
      pub o_starts: Vec<i64>,          // len n_regions*n_samples*ploidy
      pub o_stops: Vec<i64>,
      pub geno_offset_idx: ndarray::Array2<i64>,  // (n_regions*n_samples, ploidy)
  }
  // src/svar1/store.rs
  impl Svar1Store {
      pub fn ploidy(&self) -> usize;
      pub fn reader(&self) -> &genoray_core::svar1_query::Svar1Reader;
      pub fn meta(&self, contig: &str) -> Option<&ContigMeta>;
      pub fn read_window(&self, contig: &str, v_starts_c: &[u32], v_ends_c: &[u32],
                         regions: &[(u32, u32)], samples: &[usize])
          -> anyhow::Result<super::Svar1Window>;
  }
  ```
  Python: `Svar1Store(store_path, contigs, n_samples, ploidy)` + `set_contig_meta(contig, contig_start, n_local, max_v_len)`.
  FFI: `reconstruct_haplotypes_svar1(store, contig, v_starts_c, v_ends_c, region_bounds, sample_idx, v_starts, ilens, alt_alleles, alt_offsets, ref_, ref_offsets, pad_char, parallel)`.

**Background the implementer needs — read all of this before writing code:**

**1. What is being deleted and why.** The current `read_window` (`src/svar1/store.rs:77-171`) constructs a fresh `Svar1RecordSource` **per call** and clones the whole contig table into it. `Svar1RecordSource::new` is **O(all CSR entries)** — it eagerly inverts the contig's entire hap-major CSR (`build_variant_major`). It is also forward-only and takes its vectors **by value** (hence the clones). The per-contig table exists **only** to feed that constructor, which is why `set_contig_table` + `.tolist()` exist at all. Delete the record source and **all four vanish**. Do not "fix" them with `PyReadonlyArray1`/`Arc` — that would optimize a structure that should not exist.

**2. The GIL bug is one lifetime parameter.** Compare:
- SVAR1 (`src/ffi/mod.rs:802`): `store: PyRef<'_, ...>` → `read_window` must run **before** `py.detach` (`:826`), holding the GIL for the whole read.
- SVAR2 (`src/ffi/mod.rs:1088`): `store: PyRef<'py, ...>` → `reader` borrows for `'py` and is **moved into** `py.detach(move || …)` (`:1157`), so the gather runs GIL-free.

Use `PyRef<'py, ...>` and move the reader borrow into the detach closure. `Svar1Reader` is `Sync` by auto-derive (`Mmap`, `Vec<i64>`, `usize` all are), exactly like `ContigReader`.

**3. The window buffer is degenerate.** `reconstruct_haplotypes_from_sparse` wants `geno_v_idxs: ArrayView1<i32>` plus `geno_o_starts`/`geno_o_stops: ArrayView1<i64>`. `find_ranges` returns **absolute** index pairs into `variant_idxs`, and `Svar1Reader::variant_idxs()` returns `&[i32]`. So the kernel input **is** the query output — pass the mmap slice directly. Nothing is copied or materialized.

**4. Cartesian, not pairwise.** The window is `regions × samples × ploidy`. `find_ranges` emits C-order `(range, sample, ploid)`, so batch row `bi = ri * n_samples + si` and the CSR row index is `bi * ploidy + p` — i.e. `geno_offset_idx[[bi, p]] = bi * ploidy + p`, an identity mapping. The kernel's `regions` array must be expanded to one row per `(region, sample)` pair (each region repeated `n_samples` times).

**5. `v_ends` convention.** `v_end = POS_0based - min(ILEN, 0)`. gvl already computes this at `_write.py:1004-1006` as `POS - ILEN.list.first().clip(upper_bound=0)`. Note the kernel's own overlap test uses `v_end = v_start - min(ilen,0) + 1` — that `+1` is *inside* `get_diffs_sparse` and is a different convention; do **not** add it when building `v_ends` for `var_ranges`, which follows genoray's `_var_end_expr()`.

**6. `max_v_len`** is Python's convention: `max(v_ends - v_starts)` per contig, computed once at construction. See Plan 2a Task 1 — it is an over-estimate and provably overshoot-safe.

- [ ] **Step 1: Write the failing Rust test**

Replace the `#[cfg(test)] mod tests` block at the bottom of `src/svar1/store.rs` with:

```rust
#[cfg(test)]
mod tests {
    use std::io::Write;

    fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(bytemuck::cast_slice(data)).unwrap();
    }

    #[test]
    fn open_missing_store_is_err() {
        let err = super::Svar1Store::open_meta("/no/such/svar", 2, 2);
        assert!(err.is_err());
    }

    #[test]
    fn read_window_is_cartesian_and_borrows_the_mmap() {
        // 2 samples x ploidy 2 = 4 haps. Global ids 0..2 on one contig at
        // contig_start 0. Per-hap sorted global ids:
        //   hap0: [0]   hap1: []   hap2: [0, 1]   hap3: [1]
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 0, 1, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 1, 3, 4]);

        let mut store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        // var0 SNP@10 (v_end 11), var1 SNP@20 (v_end 21); max_v_len = 1
        store.set_contig_meta_rs("chr1", 0, 2, 1);

        let v_starts_c: Vec<u32> = vec![10, 20];
        let v_ends_c: Vec<u32> = vec![11, 21];

        // One region [0, 30) covering both variants, both samples.
        let w = store
            .read_window("chr1", &v_starts_c, &v_ends_c, &[(0, 30)], &[0, 1])
            .unwrap();

        // batch = 1 region * 2 samples = 2 rows; 2 rows * ploidy 2 = 4 CSR rows.
        assert_eq!(w.geno_offset_idx.shape(), &[2, 2]);
        assert_eq!(w.o_starts.len(), 4);
        // geno_offset_idx is the identity bi*ploidy + p
        assert_eq!(w.geno_offset_idx[[0, 0]], 0);
        assert_eq!(w.geno_offset_idx[[1, 1]], 3);
        // hap0 -> [0,1); hap1 -> [1,1) empty; hap2 -> [1,3); hap3 -> [3,4)
        assert_eq!(w.o_starts, vec![0, 1, 1, 3]);
        assert_eq!(w.o_stops, vec![1, 1, 3, 4]);
    }

    #[test]
    fn read_window_unknown_contig_is_err() {
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 1, 1, 1]);
        let store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        assert!(store.read_window("nope", &[], &[], &[(0, 10)], &[0]).is_err());
    }

    #[test]
    fn read_window_empty_contig_yields_all_empty_rows() {
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[] as &[i32]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 0, 0, 0, 0]);
        let mut store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        store.set_contig_meta_rs("chr1", 0, 0, 0);
        let w = store.read_window("chr1", &[], &[], &[(0, 30)], &[0, 1]).unwrap();
        for (s, e) in w.o_starts.iter().zip(&w.o_stops) {
            assert_eq!(s, e, "empty contig must give in-bounds zero-length rows");
        }
    }
}
```

Add `bytemuck` to `[dev-dependencies]` in `Cargo.toml` if `cargo-test` reports it missing (it is a transitive dep of genoray, not a direct gvl dep):

```toml
bytemuck = "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev cargo-test 2>&1 | tail -20`
Expected: FAIL — `no function or associated item named 'set_contig_meta_rs'` / `open_meta` arity mismatch.

- [ ] **Step 3: Rewrite `src/svar1/mod.rs`**

Replace the whole file:

```rust
pub mod store;

/// One window's sparse-genotype CSR geometry, produced by `Svar1Store::read_window`.
///
/// **This is the degenerate case of the SVAR1-style window buffer: it holds only
/// offsets.** SVAR1's on-disk layout is already hap-major sparse CSR of sorted global
/// variant ids, so there is nothing to decode and no table to materialize —
/// `geno_v_idxs` is borrowed straight from the `variant_idxs` mmap
/// (`Svar1Reader::variant_idxs`) and handed to the kernel as-is. VCF/PGEN backends
/// (#276) DO materialize an owned buffer; do not take this as their template.
///
/// A window is CARTESIAN: `n_regions x n_samples x ploidy`. `o_starts`/`o_stops` are
/// `n_regions * n_samples * ploidy` long in C-order `(region, sample, ploid)` —
/// absolute indices into `variant_idxs`. `geno_offset_idx` is
/// `(n_regions * n_samples, ploidy)` and maps batch row `bi = ri * n_samples + si`
/// and hap `p` to CSR row `bi * ploidy + p`.
pub struct Svar1Window {
    pub o_starts: Vec<i64>,
    pub o_stops: Vec<i64>,
    pub geno_offset_idx: ndarray::Array2<i64>,
}

#[cfg(test)]
mod link_tests {
    /// Smoke: the ungated genoray SVAR1 query API is visible from gvl's crate.
    #[test]
    fn svar1_query_symbols_are_linkable() {
        use genoray_core::svar1_query::{Svar1Reader, find_ranges, var_ranges};
        let _ = Svar1Reader::open;
        let _ = var_ranges;
        let _ = find_ranges;
    }
}
```

- [ ] **Step 4: Rewrite `src/svar1/store.rs`**

Replace the whole file:

```rust
use std::collections::HashMap;

use genoray_core::svar1_query::{Svar1Reader, find_ranges, var_ranges};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

/// Per-contig scalars. Three numbers — the big `v_starts`/`v_ends` arrays stay on the
/// Python side and cross per call as zero-copy `PyReadonlyArray1` borrows, so nothing
/// sample- or variant-scale is duplicated into Rust residency.
pub struct ContigMeta {
    /// This contig's first variant's GLOBAL id (contigs are contiguous in id space).
    pub contig_start: u32,
    pub n_local: usize,
    /// `max(v_ends - v_starts)` over the contig — genoray `var_ranges`'s convention.
    /// An over-estimate of `overlap_range`'s `>=` bound, which is overshoot-safe.
    pub max_v_len: u32,
}

/// Opened once; holds ONE `Svar1Reader` for the store's lifetime (an SVAR1 store is a
/// single flat directory — no per-contig readers, unlike `Svar2Store`) plus per-contig
/// scalars. Converges on the `Svar2Store` shape and ends up smaller.
#[pyclass]
pub struct Svar1Store {
    reader: Svar1Reader,
    contigs: HashMap<String, ContigMeta>,
}

impl Svar1Store {
    /// Opens the store's mmap'd CSR. Used by tests + `#[new]`.
    pub fn open_meta(store_path: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        let reader = Svar1Reader::open(store_path, n_samples, ploidy)
            .map_err(|e| PyIOError::new_err(format!("open svar store {store_path}: {e}")))?;
        Ok(Self {
            reader,
            contigs: HashMap::new(),
        })
    }

    pub fn reader(&self) -> &Svar1Reader {
        &self.reader
    }

    pub fn ploidy(&self) -> usize {
        self.reader.ploidy()
    }

    pub fn meta(&self, contig: &str) -> Option<&ContigMeta> {
        self.contigs.get(contig)
    }

    /// Rust-side setter (the `#[pymethods]` one delegates here) so unit tests can
    /// register a contig without a Python interpreter.
    pub fn set_contig_meta_rs(
        &mut self,
        contig: &str,
        contig_start: u32,
        n_local: usize,
        max_v_len: u32,
    ) {
        self.contigs.insert(
            contig.to_string(),
            ContigMeta {
                contig_start,
                n_local,
                max_v_len,
            },
        );
    }

    /// Read ONE window: `regions x samples x ploidy`, cartesian.
    ///
    /// `v_starts_c`/`v_ends_c` are this contig's LOCAL 0-based starts (ascending) and
    /// exclusive ends (`v_end = POS - min(ILEN, 0)`), borrowed from the caller's numpy
    /// arrays. `regions` are 0-based half-open on `contig`; `samples` are absolute
    /// sample indices.
    ///
    /// Two binary-search stages, no walk: `var_ranges` (POS -> global variant ids, one
    /// search tree for the whole window) then `find_ranges` (ids -> absolute CSR index
    /// pairs, two `partition_point`s per hap). Returns offsets only —
    /// `geno_v_idxs` is `self.reader().variant_idxs()`, borrowed by the caller.
    pub fn read_window(
        &self,
        contig: &str,
        v_starts_c: &[u32],
        v_ends_c: &[u32],
        regions: &[(u32, u32)],
        samples: &[usize],
    ) -> anyhow::Result<super::Svar1Window> {
        let m = self
            .contigs
            .get(contig)
            .ok_or_else(|| anyhow::anyhow!("no contig metadata registered for {contig}"))?;

        if v_starts_c.len() != m.n_local || v_ends_c.len() != m.n_local {
            anyhow::bail!(
                "read_window: contig {contig} has n_local={} but got v_starts={} v_ends={}",
                m.n_local,
                v_starts_c.len(),
                v_ends_c.len()
            );
        }

        let ranges = var_ranges(v_starts_c, v_ends_c, m.max_v_len, m.contig_start, regions);
        let b = find_ranges(&self.reader, &ranges, Some(samples));

        // `find_ranges` emits C-order (region, sample, ploid), so batch row
        // bi = ri * n_samples + si and CSR row = bi * ploidy + p — an identity map.
        let ploidy = b.ploidy;
        let batch = b.n_ranges * b.n_samples;
        let mut geno_offset_idx = ndarray::Array2::<i64>::zeros((batch, ploidy));
        for bi in 0..batch {
            for p in 0..ploidy {
                geno_offset_idx[[bi, p]] = (bi * ploidy + p) as i64;
            }
        }

        Ok(super::Svar1Window {
            o_starts: b.starts,
            o_stops: b.stops,
            geno_offset_idx,
        })
    }
}

#[pymethods]
impl Svar1Store {
    /// Open the SVAR1 store at `store_path`. `n_samples`/`ploidy` must match the
    /// store's `offsets.npy` length (`n_samples * ploidy + 1`) — a mismatch errors here
    /// rather than indexing out of bounds later.
    #[new]
    fn new(store_path: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        Self::open_meta(store_path, n_samples, ploidy)
    }

    fn n_samples(&self) -> usize {
        self.reader.n_samples()
    }

    #[pyo3(name = "ploidy")]
    fn ploidy_py(&self) -> usize {
        self.reader.ploidy()
    }

    /// Register a contig's scalars: its first variant's GLOBAL id, its variant count,
    /// and `max(v_ends - v_starts)`. Three numbers — no arrays cross here.
    #[pyo3(name = "set_contig_meta")]
    fn set_contig_meta(&mut self, contig: &str, contig_start: u32, n_local: usize, max_v_len: u32) {
        self.set_contig_meta_rs(contig, contig_start, n_local, max_v_len);
    }
}
```

- [ ] **Step 5: Run the Rust tests**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev cargo-test 2>&1 | grep -E "svar1|test result|error" | head -20`
Expected: the 4 `svar1::store::tests::*` pass. `src/ffi/mod.rs` will still fail to compile — that is Step 6.

- [ ] **Step 6: Rewrite the FFI**

In `src/ffi/mod.rs`, replace the entire `reconstruct_haplotypes_svar1` function (currently lines ~786-919, from its `///` doc comment through its closing brace) with:

```rust
/// Streaming SVAR1 WINDOW reconstruction: read one cartesian window
/// (`regions x samples x ploidy`) directly from a live `.svar` store via genoray's
/// ungated `svar1_query` (two binary-search stages, no record walk), then reconstruct
/// via the same `reconstruct_haplotypes_from_sparse` core as
/// `reconstruct_haplotypes_fused`.
///
/// The whole read runs INSIDE `py.detach` — `store` is `PyRef<'py, _>` so the reader
/// borrow survives into the closure, exactly like
/// `reconstruct_haplotypes_from_svar2_readbound`. (The pre-window skeleton used
/// `PyRef<'_, _>` and had to read under the GIL.)
///
/// `geno_v_idxs` is the `variant_idxs` mmap slice itself — zero copy, no materialized
/// buffer. Ragged output only (no fixed-length, keep/exonic, or to_rc). `regions[:, 0]`
/// is hardcoded to 0: this is a single-contig call and `ref_`/`ref_offsets` are expected
/// pre-sliced to `[0, contig_len]`.
///
/// `v_starts_c`/`v_ends_c` are CONTIG-LOCAL (u32) and feed the range search;
/// `v_starts`/`ilens` are the GLOBAL static table the kernel indexes with the returned
/// ids. Both are borrowed, never copied.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_svar1<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar1::store::Svar1Store>,
    contig: &str,
    v_starts_c: PyReadonlyArray1<u32>, // contig-local 0-based starts, ascending
    v_ends_c: PyReadonlyArray1<u32>,   // contig-local exclusive ends
    region_bounds: PyReadonlyArray2<i32>, // (r, 2) = (start, end), 0-based half-open
    sample_idx: PyReadonlyArray1<i64>,    // (s,) absolute sample indices
    v_starts: PyReadonlyArray1<i32>,      // GLOBAL static table (from SparseVar.index)
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)> {
    use crate::genotypes;
    use crate::reconstruct;

    // `ref_` is sliced then `.as_slice()`'d inside the reconstruct core; a strided view
    // would panic there as an uncatchable PanicException. `ref_offsets` is only indexed
    // directly (stride-safe) and must NOT be gated.
    require_contiguous_1d(&ref_, "ref_")?;

    let rb = region_bounds.as_array();
    let n_regions = rb.nrows();
    let regions_v: Vec<(u32, u32)> = (0..n_regions)
        .map(|i| (rb[[i, 0]].max(0) as u32, rb[[i, 1]].max(0) as u32))
        .collect();
    let samples_v: Vec<usize> = sample_idx.as_array().iter().map(|&s| s as usize).collect();
    let n_samples = samples_v.len();
    let ploidy = store.ploidy();
    let batch = n_regions * n_samples;
    let n_work = batch * ploidy;

    // Expand the cartesian product into one kernel `regions` row per (region, sample),
    // C-order (region, sample) so it lines up with find_ranges' output.
    let mut regions_arr = Array2::<i32>::zeros((batch, 3));
    for ri in 0..n_regions {
        for si in 0..n_samples {
            let bi = ri * n_samples + si;
            regions_arr[[bi, 1]] = rb[[ri, 0]];
            regions_arr[[bi, 2]] = rb[[ri, 1]];
        }
    }
    let shifts_arr = Array2::<i32>::zeros((batch, ploidy)); // jitter=0 in this plan

    let v_starts_c_v: Vec<u32> = v_starts_c.as_array().to_vec();
    let v_ends_c_v: Vec<u32> = v_ends_c.as_array().to_vec();

    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let alt_alleles_a = alt_alleles.as_array();
    let alt_offsets_a = alt_offsets.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();

    // `store` is PyRef<'py>, so this borrow outlives the detach.
    let store_ref: &crate::svar1::store::Svar1Store = &store;

    let result = py.detach(move || -> anyhow::Result<(Array1<u8>, Array1<i64>)> {
        let w = store_ref.read_window(
            contig,
            &v_starts_c_v,
            &v_ends_c_v,
            &regions_v,
            &samples_v,
        )?;

        // ZERO COPY: the kernel's sparse index input IS the store's mmap.
        let geno_v_idxs = store_ref.reader().variant_idxs();
        let geno_v_idxs_view = numpy::ndarray::ArrayView1::from(geno_v_idxs);

        let o_starts_arr = Array1::from_vec(w.o_starts);
        let o_stops_arr = Array1::from_vec(w.o_stops);
        let geno_offset_idx = w.geno_offset_idx;

        let q_starts_owned: Array1<i32> = regions_arr.column(1).to_owned();
        let q_ends_owned: Array1<i32> = regions_arr.column(2).to_owned();
        let diffs = genotypes::get_diffs_sparse(
            geno_offset_idx.view(),
            geno_v_idxs_view,
            o_starts_arr.view(),
            o_stops_arr.view(),
            ilens_a,
            None,
            None,
            Some(q_starts_owned.view()),
            Some(q_ends_owned.view()),
            Some(v_starts_a),
            parallel,
        );

        // out_offsets prefix-sum — ragged output only.
        let mut out_offsets_vec: Array1<i64> = Array1::zeros(n_work + 1);
        {
            let mut acc: i64 = 0;
            for k in 0..n_work {
                let query = k / ploidy;
                let hap = k % ploidy;
                let ref_len = (regions_arr[[query, 2]] - regions_arr[[query, 1]]) as i64;
                let diff = diffs[[query, hap]] as i64;
                acc += (ref_len + diff).max(0);
                out_offsets_vec[k + 1] = acc;
            }
        }

        let total = out_offsets_vec[n_work] as usize;
        let mut out_data: Array1<u8> = uninit_output(total);

        reconstruct::reconstruct_haplotypes_from_sparse(
            out_data.view_mut(),
            out_offsets_vec.view(),
            regions_arr.view(),
            shifts_arr.view(),
            geno_offset_idx.view(),
            o_starts_arr.view(),
            o_stops_arr.view(),
            geno_v_idxs_view,
            v_starts_a,
            ilens_a,
            alt_alleles_a,
            alt_offsets_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            None, // keep
            None, // keep_offsets
            None, // annot_v_idxs — not supported in the streaming path
            None, // annot_ref_pos — not supported in the streaming path
            parallel,
        );

        Ok((out_data, out_offsets_vec))
    });

    let (out_data, out_offsets_vec) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py)))
}
```

- [ ] **Step 7: Update `_Svar1Backend` (Python)**

In `python/genvarloader/_dataset/_streaming.py`, replace `_Svar1Backend.__init__`'s store-construction block (the `self._store = Svar1Store(...)` line through the end of the `for c in self._contigs:` loop) with:

```python
        self._store = Svar1Store(str(svar_path), self.n_samples, self.ploidy)

        # Per contig: register three scalars and cache the contig-local u32 arrays the
        # range search borrows. The arrays stay HERE (numpy) and cross per call as
        # zero-copy PyReadonlyArray1 -- nothing variant-scale is duplicated into Rust.
        # (The old skeleton pushed the whole POS/REF/ALT table across as Python lists
        # via .tolist() -- ~10M int objects for a human chr1 -- purely to feed
        # Svar1RecordSource's constructor. No record source, no table.)
        chrom = idx["CHROM"].cast(pl.Utf8).to_numpy()
        # v_end = POS_0based - min(ILEN, 0); genoray's `_var_end_expr()` convention.
        # NOT the kernel's `v_start - min(ilen,0) + 1` -- that +1 lives inside
        # get_diffs_sparse and is a different convention.
        v_ends_all = (v_starts - np.minimum(ilens, 0)).astype(np.uint32)
        self._contig_arrays: dict[str, tuple[NDArray[np.uint32], NDArray[np.uint32]]] = {}

        for c in self._contigs:
            mask = chrom == c
            n_local = int(mask.sum())
            if n_local == 0:
                self._store.set_contig_meta(c, 0, 0, 0)
                self._contig_arrays[c] = (
                    np.empty(0, np.uint32),
                    np.empty(0, np.uint32),
                )
                continue

            first = int(np.argmax(mask))
            # The per-contig slices below assume this contig's rows are one CONTIGUOUS
            # block starting at `first`. True for a SparseVar built from a
            # position-sorted VCF; if violated the failure mode is a silently WRONG
            # per-contig table -- parity breaks with no error. Fail fast instead.
            if not mask[first : first + n_local].all():
                raise ValueError(
                    f"SVAR index rows for contig {c!r} are not contiguous; "
                    "the streaming SVAR1 backend requires a position-sorted store."
                )

            vs_c = np.ascontiguousarray(v_starts[first : first + n_local], np.uint32)
            ve_c = np.ascontiguousarray(v_ends_all[first : first + n_local], np.uint32)
            # Python's var_ranges convention: max(v_ends - v_starts). Exactly 1 larger
            # than search::overlap_range's `>=` bound -- an OVER-estimate, which only
            # widens the candidate window and is provably overshoot-safe. Do not
            # subtract 1; UNDER-estimating would be a correctness bug.
            max_v_len = int(
                (ve_c.astype(np.int64) - vs_c.astype(np.int64)).max()
            )
            contig_start = int(idx["index"][first])

            self._store.set_contig_meta(c, contig_start, n_local, max_v_len)
            self._contig_arrays[c] = (vs_c, ve_c)
```

Then replace `reconstruct_window` with:

```python
    def reconstruct_window(
        self, r_idx: NDArray[np.intp], s_idx: NDArray[np.intp]
    ) -> Ragged:
        """Reconstruct one CARTESIAN window: every region in `r_idx` x every sample in
        `s_idx`, single-contig. Returns a `Ragged[np.bytes_]` (S1) of shape
        `(len(r_idx) * len(s_idx), ploidy, ~length)`, C-order (region, sample).
        """
        from ..genvarloader import reconstruct_haplotypes_svar1

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)

        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError(
                "_Svar1Backend.reconstruct_window: window spans multiple contigs; "
                "every Rust call must be single-contig (the scheduler groups by contig)."
            )
        contig_name = self._contigs[contig_idx]
        vs_c, ve_c = self._contig_arrays[contig_name]

        region_bounds = np.ascontiguousarray(self._regions[r_idx, 1:3], np.int32)

        ref_c_idx = self._ref.c_map.contigs.index(contig_name)
        c_s = int(self._ref.offsets[ref_c_idx])
        c_e = int(self._ref.offsets[ref_c_idx + 1])
        ref_bytes = np.ascontiguousarray(self._ref.reference[c_s:c_e], np.uint8)
        ref_offsets = np.array([0, c_e - c_s], dtype=np.int64)

        data, offsets = reconstruct_haplotypes_svar1(
            self._store,
            contig_name,
            vs_c,
            ve_c,
            region_bounds,
            np.ascontiguousarray(s_idx, np.int64),
            self._v_starts,
            self._ilens,
            self._alt_alleles,
            self._alt_offsets,
            ref_bytes,
            ref_offsets,
            self._ref.pad_char,
            True,
        )
        batch = len(r_idx) * len(s_idx)
        return Ragged.from_offsets(
            data.view("S1"), (batch, self.ploidy, None), np.asarray(offsets, np.int64)
        )
```

Also delete the now-unused `_reject_unsupported_variants` justification comment's reference to REF/ALT-derived ILEN if it no longer applies — **but keep the `_reject_unsupported_variants` call itself**: gvl still cannot reconstruct multi-allelic/symbolic/breakend variants, and `ILEN` from the index is now what drives `v_ends`.

- [ ] **Step 8: Adapt `_plan`/`__iter__` to cartesian windows**

`_plan` currently yields **pairwise** `(r_idx, s_idx)` via `unravel_index`. With cartesian windows it must yield a region set + the full sample set. Replace `_plan` and `__iter__`:

```python
    def _plan(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield one WINDOW per step: (region_idxs, sample_idxs), cartesian.

        Region-major, single-contig per window (`self._regions` is sorted by
        (contig_idx, start), so each contig's regions are one contiguous run). Every
        sample is read per window -- variant stores return all samples per range read
        essentially for free, so the effective item order is region-major,
        sample-inner. NOT pairwise: `StreamingDataset` has no `__getitem__`, so the
        traversal is a fixed cartesian sweep and the window is the read granularity.
        """
        n_regions, n_samples = self.shape
        if n_regions == 0:
            return
        all_samples = np.arange(n_samples, dtype=np.intp)
        contig_idxs = self._regions[:, 0]
        run_bounds = np.flatnonzero(np.diff(contig_idxs)) + 1
        run_starts = np.concatenate(([0], run_bounds))
        run_ends = np.concatenate((run_bounds, [n_regions]))
        for r_lo, r_hi in zip(run_starts, run_ends):
            for w_lo in range(int(r_lo), int(r_hi), self._window_regions):
                w_hi = min(w_lo + self._window_regions, int(r_hi))
                yield np.arange(w_lo, w_hi, dtype=np.intp), all_samples

    def _iter_batches(self, batch_size: int) -> Iterator[tuple]:
        """Drive the plan and slice each reconstructed window into batches.

        The window is the READ granularity; a batch is a slice of it. Batches may span
        window boundaries only in the sense that a window's trailing partial batch is
        emitted as-is -- windows never split a (region, sample) cell.
        """
        for r_idx, s_idx in self._plan():
            data = self._reconstruct_window(r_idx, s_idx)
            # Window rows are C-order (region, sample): row bi = ri*n_samples + si.
            n_s = len(s_idx)
            flat_r = np.repeat(self._sort_order[r_idx], n_s)
            flat_s = np.tile(s_idx, len(r_idx))
            n_rows = len(flat_r)
            for lo in range(0, n_rows, batch_size):
                hi = min(lo + batch_size, n_rows)
                yield data[lo:hi], flat_r[lo:hi], flat_s[lo:hi]
```

Add a `_window_regions` field to the dataclass (default `64`) and set it in `__init__` via `object.__setattr__(self, "_window_regions", 64)`. Add it to the field declarations:

```python
    _window_regions: int = 64
```

with a comment:

```python
    # Regions per read window. Window >> batch is the whole point: one Rust call per
    # window amortizes the search + page faults across many batches. 64 is a
    # placeholder default -- Task 4 measures and replaces it.
```

- [ ] **Step 9: Rebuild and run parity**

Run:
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine
pixi run -e dev maturin develop --release 2>&1 | tail -3
pixi run -e dev pytest tests/dataset/test_streaming_parity.py tests/dataset/test_svar1_window.py -q 2>&1 | tail -15
```
Expected: **`test_streaming_matches_written_all_cells` PASSES** — this is the control. `test_svar1_window.py` and `test_dataloader_len_matches_batches_yielded` will need mechanical updates (they call `_with_batch_size`/`list(sds)`); fix them to use `_iter_batches`. **If `test_streaming_matches_written_all_cells` fails, STOP** — the rewrite changed behavior.

- [ ] **Step 10: Full suite + lint**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine
pixi run -e dev pytest tests/dataset tests/unit -q 2>&1 | tail -5
pixi run -e dev cargo-test 2>&1 | tail -3
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck 2>&1 | tail -3
```
Expected: all green.

- [ ] **Step 11: Commit**

```bash
git add src/svar1/ src/ffi/mod.rs python/genvarloader/_dataset/_streaming.py tests/ Cargo.toml
git commit -m "perf(streaming): window-granular GIL-free SVAR1 reads via genoray svar1_query

Deletes Svar1RecordSource, ContigTable, set_contig_table, every .tolist(), and
every per-batch clone -- all artifacts of using a conversion record producer for
a query. read_window is now two binary-search stages inside py.detach, and
geno_v_idxs is the variant_idxs mmap slice itself (zero copy)."
```

---

### Task 3: `to_iter` — one iteration entry point

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py`
- Modify: `tests/dataset/test_streaming_parity.py`
- Modify: `docs/source/dataset.md:174-200`
- Modify: `docs/source/api.md:91-99`
- Modify: `docs/source/faq.md:110-125` ("Can I use gvl without writing a dataset first?")
- Modify: `skills/genvarloader/SKILL.md:369-395` (+ the gotcha at `:488`)

**Interfaces:**
- Consumes: `_iter_batches` (Task 2 Step 8). **Independent of Task 2's Rust work — can run concurrently.**
- Produces:
  ```python
  StreamingDataset.to_iter(batch_size: int = 1, return_indices: bool = True) -> Iterator[tuple]
  StreamingDataset.to_torch_dataset(batch_size: int = 1, return_indices: bool = True) -> td.IterableDataset
  StreamingDataset.to_dataloader(batch_size=1, num_workers=0, return_indices=True, **dl_kwargs) -> td.DataLoader
  ```
  `__iter__` is **removed**. `__getitem__` still raises `TypeError`.

**Background:** Today `StreamingDataset` **is** a torch `IterableDataset` and `to_torch_dataset()` raises `TypeError`. That inverts: `StreamingDataset` becomes a plain frozen dataclass owning the traversal, `to_iter()` is the real work, and the torch handle is built on demand. `to_torch_dataset` matches `gvl.Dataset`'s existing name for the same concept — one name, one meaning. Torch stays an optional import behind `@requires_torch`.

- [ ] **Step 1: Write the failing test**

Add to `tests/dataset/test_streaming_parity.py`:

```python
def test_to_iter_is_the_one_entry_point(svar1_multicontig_fixture):
    """`to_iter` is the single iteration API. `__iter__` is REMOVED -- one and only
    one obvious way to do it; `to_torch_dataset`/`to_dataloader` wrap `to_iter`."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")

    assert not hasattr(sds, "__iter__"), "__iter__ must be removed; use to_iter()"
    with pytest.raises(TypeError):
        iter(sds)

    batches = list(sds.to_iter(batch_size=4))
    assert len(batches) > 0
    data, r_idx, s_idx = batches[0]
    assert len(r_idx) == len(s_idx)


def test_to_iter_return_indices_false_yields_data_only(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    first = next(iter(sds.to_iter(batch_size=2, return_indices=False)))
    assert not isinstance(first, tuple), "return_indices=False must yield data alone"


def test_to_torch_dataset_wraps_to_iter(svar1_multicontig_fixture):
    """`to_torch_dataset` now RETURNS an IterableDataset (it used to raise TypeError,
    because StreamingDataset itself was one). Same name as Dataset.to_torch_dataset."""
    import torch.utils.data as td

    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    tds = sds.to_torch_dataset(batch_size=4)
    assert isinstance(tds, td.IterableDataset)
    assert len(list(tds)) == len(list(sds.to_iter(batch_size=4)))


def test_to_iter_covers_every_cell_exactly_once(svar1_multicontig_fixture):
    """Window/batch separation must not drop or duplicate cells."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    seen = []
    for _data, r_idx, s_idx in sds.to_iter(batch_size=3):
        seen.extend(zip(r_idx.tolist(), s_idx.tolist()))
    n_regions, n_samples = sds.shape
    assert sorted(seen) == sorted(
        (r, s) for r in range(n_regions) for s in range(n_samples)
    )
```

Update `test_no_map_style_access` — `to_torch_dataset` no longer raises:

```python
def test_no_map_style_access(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(f.bed, reference=f.reference_path, variants=f.svar_path)

    # to_torch_dataset() no longer raises -- it returns an IterableDataset wrapping
    # to_iter(). Only random access is refused.
    with pytest.raises(TypeError):
        _ = sds[0, 0]
```

And update `test_streaming_matches_written_all_cells` + `test_dataloader_len_matches_batches_yielded` to use `to_dataloader` (unchanged API) — they should need no edit beyond confirming they still pass.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev pytest tests/dataset/test_streaming_parity.py -q 2>&1 | tail -10`
Expected: FAIL — `to_iter` doesn't exist; `hasattr(sds, "__iter__")` is True.

- [ ] **Step 3: Implement the surface**

In `_streaming.py`, delete `__iter__` and the module-level `_make_streaming_torch_dataset`, and replace `to_torch_dataset`/`to_dataloader` with:

```python
    def to_iter(
        self, batch_size: int = 1, return_indices: bool = True
    ) -> Iterator[tuple]:
        """Iterate haplotype batches. **This is the one iteration entry point** —
        :meth:`to_torch_dataset` and :meth:`to_dataloader` are thin wrappers over it,
        and there is no ``__iter__`` (one and only one obvious way).

        Iteration is a fixed cartesian sweep of BED regions x samples in a
        data-layout-optimal order (region-major for variants). There is no random
        access and no ad-hoc query: ``sds[r, s]`` raises :class:`TypeError`.

        Parameters
        ----------
        batch_size
            Number of ``(region, sample)`` cells per yielded batch. Batches are slices
            of a much larger read *window*; ``batch_size`` does not affect I/O
            granularity.
        return_indices
            If ``True`` (the default), yield ``(data, region_idxs, sample_idxs)``;
            if ``False``, yield ``data`` alone. Indices are in the caller's **original
            BED-row order** (not sorted-storage order), matching ``gvl.Dataset[r, s]``.
        """
        for data, r_idx, s_idx in self._iter_batches(batch_size):
            if return_indices:
                yield data, r_idx, s_idx
            else:
                yield data

    def n_batches(self, batch_size: int) -> int:
        """Number of batches :meth:`to_iter` will yield at ``batch_size``.

        NOT ``ceil(len(self) / batch_size)``: the plan batches *within* each window,
        so every window's last batch may be partial. Counting the plan is cheap (it
        only materializes small index arrays).
        """
        return sum(1 for _ in self._iter_batch_spans(batch_size))

    def _iter_batch_spans(self, batch_size: int) -> Iterator[int]:
        """Batch sizes the plan will yield, without reconstructing anything."""
        for r_idx, s_idx in self._plan():
            n_rows = len(r_idx) * len(s_idx)
            for lo in range(0, n_rows, batch_size):
                yield min(lo + batch_size, n_rows) - lo

    def __getitem__(self, idx) -> None:
        raise TypeError(
            "StreamingDataset is iterable-only; use to_iter() instead of map-style "
            "indexing. Iteration order is fixed by the data layout, so there is no "
            "random access."
        )

    @requires_torch
    def to_torch_dataset(
        self, batch_size: int = 1, return_indices: bool = True
    ) -> "td.IterableDataset":
        """Wrap :meth:`to_iter` in a torch :class:`IterableDataset`. Thin wrapper —
        all the work is in ``to_iter``. Named to match
        :meth:`Dataset.to_torch_dataset` (same concept, same name)."""
        import torch.utils.data as td

        sds = self

        class _StreamingTorchDataset(td.IterableDataset):
            def __iter__(self):
                return sds.to_iter(batch_size, return_indices)

            def __len__(self) -> int:
                return sds.n_batches(batch_size)

        return _StreamingTorchDataset()

    @requires_torch
    def to_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        return_indices: bool = True,
        *,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Callable | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> "td.DataLoader":
        """Wrap :meth:`to_torch_dataset` in a torch
        :class:`DataLoader <torch.utils.data.DataLoader>`. Thin wrapper.

        Parameters
        ----------
        num_workers
            Must be 0. ``StreamingDataset``'s own engine IS the concurrency strategy
            (mirrors :meth:`Dataset.to_dataloader`'s ``buffered``/``double_buffered``
            restriction); worker-process sharding of the window plan is a later plan.
        """
        if num_workers > 0:
            raise ValueError(
                "StreamingDataset.to_dataloader: num_workers>0 is not implemented "
                "yet; the streaming engine IS the concurrency strategy for "
                "StreamingDataset (mirrors gvl.Dataset.to_dataloader's "
                "buffered/double_buffered modes, which impose the same restriction). "
                "Use num_workers=0."
            )

        import torch.utils.data as td

        return td.DataLoader(
            self.to_torch_dataset(batch_size, return_indices),
            batch_size=None,  # the dataset yields pre-assembled batches
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
```

Delete `_with_batch_size` and the `_batch_size` field — `batch_size` is now a `to_iter` argument, not instance state (one way to do it). Update `test_svar1_window.py`'s helper, which used `_with_batch_size`, to call `to_iter(batch_size=...)` instead.

- [ ] **Step 4: Run tests**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev pytest tests/dataset/test_streaming_parity.py tests/dataset/test_svar1_window.py -q 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Update the docs (the docs gate FIRES — public API changed)**

`docs/source/dataset.md` — replace the code block and the limitations paragraph:

````markdown
```python
sds = gvl.StreamingDataset(
    "rois.bed", reference="ref.fa", variants="normed.svar"
).with_seqs("haplotypes")

for data, region_idxs, sample_idxs in sds.to_iter(batch_size=32):
    ...  # data: Ragged[S1], shape (batch, ploidy, ~length)
```

`to_iter` is the one iteration entry point; `to_torch_dataset()` and `to_dataloader()`
are thin wrappers over it. There is no `__iter__` and no random access — `sds[r, s]`
raises `TypeError`, because iteration order is fixed by the data layout.
````

Replace the "currently more limited" paragraph's `**iterable-only** — `sds[r, s]` raises `TypeError`; use `sds.to_dataloader(...)`.` with:

````markdown
and **iterable-only** — `sds[r, s]` raises `TypeError`; use `sds.to_iter(...)` (or
`to_dataloader(...)` for torch). See the `genvarloader` skill for the full list of
what's not yet wired.
````

`docs/source/api.md` — update the autoclass members:

````markdown
.. autoclass:: StreamingDataset
    :members: __init__, shape, with_seqs, to_iter, to_torch_dataset, to_dataloader, n_batches
    :exclude-members: __new__
````

`docs/source/faq.md` — the "Can I use gvl without writing a dataset first?" answer (`:110-125`) contains a claim this change makes **FALSE**: *"`sds[r, s]` and `sds.to_torch_dataset()` raise `TypeError`"*. `to_torch_dataset()` no longer raises — it returns an `IterableDataset`. Update the code block:

````markdown
```python
sds = gvl.StreamingDataset(
    "rois.bed", reference="ref.fa", variants="normed.svar"
).with_seqs("haplotypes")

for data, region_idxs, sample_idxs in sds.to_iter(batch_size=32):
    ...
```
````

and replace the final paragraph's iterable-only clause with:

````markdown
`StreamingDataset` is currently narrower than `Dataset`: `.svar` variant sources only (VCF, PGEN, and `.svar2` raise `NotImplementedError`), `with_seqs("haplotypes")` only, `jitter=0` only, ragged output only, and it's **iterable-only** — `sds[r, s]` raises `TypeError` because iteration order is fixed by the data layout. `sds.to_iter(...)` is the one entry point; `to_torch_dataset()` and `to_dataloader()` are thin wrappers over it. Iteration is region-major, read one window at a time so each read stays within one contig; `to_iter(..., return_indices=True)` (the default) rides along `(region_idxs, sample_idxs)` in your original BED-row order. See the `genvarloader` skill for the full scope.
````

Also fix the "re-reads the live store on every window" sentence at `:123` if it now overstates the cost — it is accurate (each window does read the store), so leave it.

`skills/genvarloader/SKILL.md` — update the example to `to_iter`, and replace the iterable-only bullet:

````markdown
- **Iterable-only, no random access** — `sds[r, s]` raises `TypeError`. `to_iter(batch_size=...)` is the one entry point; `to_torch_dataset()` and `to_dataloader()` are thin wrappers over it. There is no `__iter__`.
````

and the gotcha at `SKILL.md:488`:

````markdown
- **`gvl.StreamingDataset` is iterable-only and `.svar`-only.** `sds[r, s]` raises `TypeError` — use `sds.to_iter(...)` (or `to_dataloader(...)` for torch); there is no `__iter__`. VCF/PGEN/`.svar2` variant sources, non-`"haplotypes"` `with_seqs` kinds, `jitter != 0`, and `num_workers > 0` all raise `NotImplementedError`/`ValueError`. See "`gvl.StreamingDataset` — write-free streaming (SVAR1 only)" above.
````

- [ ] **Step 6: Verify the api.md/`__all__` sync gate**

Run:
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`.

- [ ] **Step 7: Lint + typecheck + commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck 2>&1 | tail -3
git add python/genvarloader/_dataset/_streaming.py tests/dataset/ docs/source/dataset.md docs/source/api.md skills/genvarloader/SKILL.md
git commit -m "feat(streaming)!: to_iter is the one iteration entry point

BREAKING: StreamingDataset.__iter__ removed; use to_iter(batch_size=...).
StreamingDataset is no longer a torch IterableDataset -- to_torch_dataset()
now returns one (it previously raised TypeError). to_dataloader is unchanged."
```

---

### Task 4: Scale fixture + the deterministic measurement

**Files:**
- Create: `tests/dataset/test_streaming_scale.py`
- Modify: `src/svar1/store.rs` (add a window-read counter)
- Modify: `src/ffi/mod.rs` (expose the counter)

**Interfaces:**
- Consumes: `Svar1Store::read_window` (Task 2).
- Produces: `genvarloader.svar1_csr_entries_touched() -> int` (a test/observability hook).

**Background:** The existing 40bp toy fixtures **cannot observe** an O(all-entries)-per-batch bug. And per the standing convention on this hardware, **do not gate on absolute wall-clock** — the node is too noisy. The gate is a **deterministic counter**: CSR entries touched per window, which goes from `O(all entries on the contig)` to `O(log n + variants in window)`. That is a flat-vs-linear curve and is noise-immune.

genoray has the same idiom already: `search::search_tree_build_count()` (`src/search.rs:47-57`), a `thread_local!` `Cell<usize>` marked `#[doc(hidden)]`. Copy that shape.

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_streaming_scale.py`:

```python
"""Scale guard + the #275 throughput gate.

The gate is a DETERMINISTIC COUNTER, not wall-clock: this node is too noisy for
absolute timings (see the project's perf-gate convention). The walking skeleton
re-opened `Svar1RecordSource` per batch, whose constructor is O(all CSR entries) --
so entries-touched grew with the STORE, not the window. After the rewrite it must
grow with the WINDOW only. That's a flat-vs-linear curve; noise can't fake it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl

# A store big enough that "touches the whole contig" and "touches the window" differ
# by orders of magnitude. 200 variants x 20 samples, one contig.
_N_VARIANTS = 200
_N_SAMPLES = 20
_CONTIG_LEN = 4000


def _make_vcf(path: Path) -> None:
    rng = np.random.default_rng(0)
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr1,length={_CONTIG_LEN}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"S{i}" for i in range(_N_SAMPLES)),
    ]
    positions = np.sort(rng.choice(np.arange(2, _CONTIG_LEN - 2), _N_VARIANTS, replace=False))
    for pos in positions:
        gts = "\t".join(
            f"{rng.integers(0, 2)}|{rng.integers(0, 2)}" for _ in range(_N_SAMPLES)
        )
        lines.append(f"chr1\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t{gts}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def scale_fixture(tmp_path_factory):
    from genoray import VCF, SparseVar

    d = tmp_path_factory.mktemp("svar1_scale")
    ref = d / "ref.fa"
    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list("ACGT"), _CONTIG_LEN))
    ref.write_text(f">chr1\n{seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    _make_vcf(vcf)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar = d / "store.svar"
    SparseVar.from_vcf(
        svar, VCF(bcf), max_mem="1g",
        samples=[f"S{i}" for i in range(_N_SAMPLES)], overwrite=True,
    )
    return svar, ref


def test_entries_touched_scales_with_window_not_store(scale_fixture):
    """THE #275 GATE. Entries touched per window must be ~flat as the store's variant
    count grows -- i.e. proportional to the window's variants, not the contig's.

    The skeleton's `Svar1RecordSource::new` inverted the whole contig CSR per call,
    so this ratio would be ~1.0 (touching everything). After the rewrite a narrow
    window must touch a small fraction.
    """
    from genvarloader.genvarloader import svar1_csr_entries_touched

    svar, ref = scale_fixture
    total_entries = int(np.asarray(__import__("genoray").SparseVar(svar).genos.data).size)

    # One narrow region: ~1/40th of the contig.
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs("haplotypes")

    before = svar1_csr_entries_touched()
    list(sds.to_iter(batch_size=8))
    touched = svar1_csr_entries_touched() - before

    assert touched > 0, "counter is not wired"
    assert touched < total_entries * 0.25, (
        f"window read touched {touched} of {total_entries} CSR entries -- that is "
        "whole-store behavior, i.e. the O(all entries) per-call path is back"
    )


def test_entries_touched_is_flat_across_batch_size(scale_fixture):
    """Batch size must NOT affect I/O: the window is the read granularity and a batch
    is only a slice of it. The skeleton did one read PER BATCH, so halving batch_size
    doubled the work. Now it must be identical."""
    from genvarloader.genvarloader import svar1_csr_entries_touched

    svar, ref = scale_fixture
    bed = pl.DataFrame(
        {"chrom": ["chr1"] * 4, "chromStart": [0, 100, 200, 300], "chromEnd": [100, 200, 300, 400]}
    )
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs("haplotypes")

    counts = []
    for bs in (1, 8, 64):
        before = svar1_csr_entries_touched()
        list(sds.to_iter(batch_size=bs))
        counts.append(svar1_csr_entries_touched() - before)

    assert counts[0] == counts[1] == counts[2], (
        f"entries touched varied with batch_size: {counts} -- the window is the read "
        "granularity, so batch_size must not change I/O at all"
    )


def test_no_per_window_materialization_of_sample_scale_arrays(scale_fixture):
    """SCALE GUARD (mirrors the rust-migration defense).

    `geno_v_idxs` must be the `variant_idxs` MMAP SLICE ITSELF -- borrowed, not copied.
    If a future change reintroduces an owned per-window buffer, RSS grows with the
    store on every window and this catches it. Peak RSS growth across many windows must
    stay far below the store's genotype-array size.
    """
    import resource

    import genoray

    svar, ref = scale_fixture
    store_bytes = np.asarray(genoray.SparseVar(svar).genos.data).nbytes

    bed = pl.DataFrame(
        {
            "chrom": ["chr1"] * 8,
            "chromStart": list(range(0, 800, 100)),
            "chromEnd": list(range(100, 900, 100)),
        }
    )
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs("haplotypes")

    # Warm up so first-touch allocations aren't attributed to the loop.
    list(sds.to_iter(batch_size=8))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(5):
        list(sds.to_iter(batch_size=8))
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    grew_bytes = (after - before) * 1024  # ru_maxrss is KiB on Linux
    assert grew_bytes < store_bytes, (
        f"peak RSS grew {grew_bytes} B across repeated sweeps (store genotypes are "
        f"{store_bytes} B) -- a per-window copy of the sparse index has been "
        "reintroduced; geno_v_idxs must borrow the mmap"
    )


def test_scale_parity_still_byte_identical(scale_fixture, tmp_path):
    """The scale fixture must ALSO satisfy the parity oracle -- a fast wrong answer
    is not progress."""
    from genoray import SparseVar

    svar, ref = scale_fixture
    bed = pl.DataFrame(
        {"chrom": ["chr1"] * 3, "chromStart": [0, 500, 1500], "chromEnd": [200, 700, 1700]}
    )
    out = tmp_path / "scale.gvl"
    gvl.write(out, bed, variants=SparseVar(svar), samples=None, overwrite=True)
    written = gvl.Dataset.open(out, reference=ref).with_seqs("haplotypes")

    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs("haplotypes")
    for data, r_idx, s_idx in sds.to_iter(batch_size=7):
        for i in range(len(r_idx)):
            expected = written[int(r_idx[i]), int(s_idx[i])]
            for h in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data[i][h]), np.asarray(expected[h])
                )
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev pytest tests/dataset/test_streaming_scale.py -q 2>&1 | tail -10`
Expected: FAIL — `cannot import name 'svar1_csr_entries_touched'`.

- [ ] **Step 3: Add the counter**

At the top of `src/svar1/store.rs` (after the imports):

```rust
thread_local! {
    static CSR_ENTRIES_TOUCHED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Test/observability hook: total CSR entries spanned by window reads on the current
/// thread. The #275 throughput gate asserts this scales with the WINDOW, not the
/// store — the pre-rewrite path inverted the whole contig CSR per batch. Mirrors
/// genoray's `search::search_tree_build_count`.
#[doc(hidden)]
pub fn csr_entries_touched() -> usize {
    CSR_ENTRIES_TOUCHED.with(|c| c.get())
}
```

In `read_window`, immediately after the `find_ranges` call:

```rust
        // Observability: entries this window actually spans. See `csr_entries_touched`.
        let spanned: usize = b
            .starts
            .iter()
            .zip(&b.stops)
            .map(|(s, e)| (e - s) as usize)
            .sum();
        CSR_ENTRIES_TOUCHED.with(|c| c.set(c.get() + spanned));
```

**Note the thread-local caveat:** `read_window` runs inside `py.detach` but on the *same* OS thread as the caller (pyo3 does not move the closure to another thread), so a thread-local read from Python sees the count. rayon inside the *kernel* does not touch this counter.

In `src/ffi/mod.rs`, add:

```rust
/// Test hook: CSR entries spanned by SVAR1 window reads on this thread. See
/// `crate::svar1::store::csr_entries_touched`.
#[pyfunction]
pub fn svar1_csr_entries_touched() -> usize {
    crate::svar1::store::csr_entries_touched()
}
```

In `src/lib.rs`'s `#[pymodule]`, next to the other svar1 function:

```rust
    m.add_function(wrap_pyfunction!(ffi::svar1_csr_entries_touched, m)?)?;
```

- [ ] **Step 4: Rebuild and run**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine
pixi run -e dev maturin develop --release 2>&1 | tail -3
pixi run -e dev pytest tests/dataset/test_streaming_scale.py -q 2>&1 | tail -10
```
Expected: all 3 tests PASS. **If `test_entries_touched_is_flat_across_batch_size` fails, the window/batch separation is wrong** — a batch is still driving a read.

- [ ] **Step 5: Measure and set the `window_regions` default**

Task 2 Step 8 set `_window_regions = 64` as an explicit **placeholder**. The spec requires a *measured* default, not a guess — replace it now.

Sweep the knob against the scale fixture and record entries-touched and wall-clock per sweep:

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev python - <<'PY'
import time
import numpy as np, polars as pl
import genvarloader as gvl
from genvarloader._dataset._streaming import StreamingDataset

# Point these at the scale fixture's paths (print them from a pytest run, or rebuild
# the fixture here with the same helper).
svar, ref = "<svar path>", "<ref path>"
bed = pl.DataFrame({
    "chrom": ["chr1"] * 20,
    "chromStart": list(range(0, 2000, 100)),
    "chromEnd": list(range(100, 2100, 100)),
})
for wr in (1, 4, 16, 64, 256):
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs("haplotypes")
    object.__setattr__(sds, "_window_regions", wr)
    t0 = time.perf_counter()
    n = sum(1 for _ in sds.to_iter(batch_size=8))
    dt = time.perf_counter() - t0
    print(f"window_regions={wr:4d}  batches={n:4d}  {dt:.3f}s")
PY
```

Pick the knee of the curve — the smallest `window_regions` past which wall-clock stops improving — and set that as the default in `_streaming.py`, replacing the placeholder comment with the measured justification:

```python
    # Regions per read window. Window >> batch is the point: one Rust call per window
    # amortizes the search + page faults across many batches. Default measured on the
    # scale fixture (see docs/roadmaps/streaming-dataset.md, Plan 2): throughput is
    # flat past this, so a larger window only costs memory.
    _window_regions: int = <measured value>
```

**Caveat to honor:** this node is noisy, so use the *shape* of the curve (where it flattens), not any single absolute number, and take the best of ≥3 runs per setting in one session. If the curve is flat everywhere, say so and keep a small default — do not invent a knee that the data does not show.

- [ ] **Step 6: Record the before/after in the roadmap**

Run the counter against the pre-rewrite skeleton for the record:
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && git stash list  # ensure clean; do NOT use bare git stash (shared stack)
```
Instead, capture the numbers from this session's test output and add a bullet under the Plan-2 task in `docs/roadmaps/streaming-dataset.md` recording: entries touched per window before (≈ all CSR entries) vs after (≈ window variants × haps), and the batch-size invariance result. **Do not report wall-clock as the gate** — report it as secondary color only, and say so.

- [ ] **Step 7: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine
git add tests/dataset/test_streaming_scale.py src/svar1/store.rs src/ffi/mod.rs src/lib.rs python/genvarloader/_dataset/_streaming.py docs/roadmaps/streaming-dataset.md
git commit -m "test(streaming): scale fixture + deterministic entries-touched gate (#275)"
```

---

### Task 5: The double-buffer engine

**Files:**
- Modify: `Cargo.toml` (add `crossbeam-channel`)
- Create: `src/stream/mod.rs` (the `StreamBackend` trait + engine)
- Modify: `src/lib.rs` (`pub mod stream;`)
- Modify: `python/genvarloader/_dataset/_streaming.py` (drive the engine from `_iter_batches`)

**Interfaces:**
- Consumes: `Svar1Store::read_window` (Task 2), `to_iter` (Task 3).
- Produces:
  ```rust
  pub trait StreamBackend: Sync {
      type Buffer: Send;
      fn fill(&self, window: &WindowSpec, slot: &mut Self::Buffer) -> anyhow::Result<()>;
  }
  pub struct WindowSpec { pub contig_idx: usize, pub r_lo: usize, pub r_hi: usize }
  ```

**Background the implementer needs — this is the highest-risk task:**

**This introduces gvl's first threading primitive.** There is currently **zero** `std::thread`, crossbeam, or `unsafe impl Send/Sync` in `src/`. Review accordingly.

**What is overlapped: I/O latency, not decode.** SVAR1 has no decode. The producer faults in `variant_idxs` mmap pages and runs binary searches for window N+1 while the consumer reconstructs from window N. The page cache does not prefetch on an application's access pattern; a producer thread on a known traversal does. Because the traversal is fixed and fully known (no `__getitem__`), the prefetch is speculation-free.

**Copy genoray's `orchestrator.rs` for the pattern — but NOT for slot recycling.** From `orchestrator.rs`, copy:
- `thread::Builder::new().name(...)` per stage (perf attribution).
- `crossbeam_channel::bounded(N)`; backpressure is the bounded capacity.
- **Shutdown by `Sender` drop** — and note the subtlety documented at `orchestrator.rs:184-186`: any extra `Sender` clone held for introspection must drop before a downstream `recv()` can observe close.
- **Join everything, THEN classify panics** (`orchestrator.rs:359-405`). Its comment is explicit that early-returning on a producer error leaves the consumer blocked on `recv()` forever. `Err(_)` from `join` → a `WorkerPanicked`-style error; `Ok(Err(e))` → propagate the real error.

**Do NOT copy slot recycling — genoray has none.** It allocates each chunk fresh, moves it through the channel, and drops it. Recycling is net-new design here. Per the spec: **start with 2 slots (ping-pong)**; promote to an N-slot ring only on profiling evidence. Single producer + rayon-within-fill first.

- [ ] **Step 1: Write the failing test**

Create `src/stream/mod.rs` with only a test module:

```rust
//! Generic streaming engine: a producer thread fills window N+1 while the consumer
//! reconstructs from window N.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingBackend {
        fills: AtomicUsize,
    }
    impl StreamBackend for CountingBackend {
        type Buffer = Vec<usize>;
        fn fill(&self, w: &WindowSpec, slot: &mut Self::Buffer) -> anyhow::Result<()> {
            slot.clear();
            slot.extend(w.r_lo..w.r_hi);
            self.fills.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct FailingBackend;
    impl StreamBackend for FailingBackend {
        type Buffer = Vec<usize>;
        fn fill(&self, _w: &WindowSpec, _slot: &mut Self::Buffer) -> anyhow::Result<()> {
            anyhow::bail!("boom")
        }
    }

    fn windows(n: usize) -> Vec<WindowSpec> {
        (0..n)
            .map(|i| WindowSpec { contig_idx: 0, r_lo: i * 10, r_hi: i * 10 + 10 })
            .collect()
    }

    #[test]
    fn engine_yields_every_window_in_plan_order() {
        let be = CountingBackend { fills: AtomicUsize::new(0) };
        let mut seen = Vec::new();
        run_windows(&be, &windows(5), 2, |slot| {
            seen.push(slot[0]);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![0, 10, 20, 30, 40], "windows must arrive in plan order");
        assert_eq!(be.fills.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn engine_recycles_slots_and_caps_live_buffers() {
        // Slot recycling is NET-NEW here (genoray's orchestrator does not recycle --
        // it allocates fresh and drops). With 2 slots, at most 2 buffers exist for
        // the whole run regardless of window count.
        let be = CountingBackend { fills: AtomicUsize::new(0) };
        let mut n = 0;
        run_windows(&be, &windows(50), 2, |_slot| {
            n += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(n, 50);
    }

    #[test]
    fn producer_error_propagates_and_does_not_hang() {
        // orchestrator.rs's hard-won lesson: early-returning on a producer error
        // leaves the consumer blocked on recv() forever. This must return an Err,
        // not deadlock.
        let be = FailingBackend;
        let r = run_windows(&be, &windows(3), 2, |_slot| Ok(()));
        assert!(r.is_err(), "producer error must surface");
        assert!(format!("{:?}", r.unwrap_err()).contains("boom"));
    }

    #[test]
    fn consumer_error_propagates_and_does_not_hang() {
        let be = CountingBackend { fills: AtomicUsize::new(0) };
        let r = run_windows(&be, &windows(10), 2, |_slot| anyhow::bail!("consumer boom"));
        assert!(r.is_err());
        assert!(format!("{:?}", r.unwrap_err()).contains("consumer boom"));
    }

    #[test]
    fn empty_plan_is_ok() {
        let be = CountingBackend { fills: AtomicUsize::new(0) };
        run_windows(&be, &[], 2, |_slot| Ok(())).unwrap();
        assert_eq!(be.fills.load(Ordering::Relaxed), 0);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Add `pub mod stream;` to `src/lib.rs` and `crossbeam-channel = "0.5.15"` to `Cargo.toml`'s `[dependencies]` (matching genoray's pin).
Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev cargo-test 2>&1 | tail -10`
Expected: FAIL — `cannot find trait 'StreamBackend'`.

- [ ] **Step 3: Implement the engine**

Prepend to `src/stream/mod.rs`:

```rust
//! Generic streaming engine: a producer thread fills window N+1 while the consumer
//! reconstructs from window N.
//!
//! **What is overlapped is I/O latency, not decode.** For SVAR1 there is nothing to
//! decode — the producer faults in `variant_idxs` mmap pages and runs binary searches
//! ahead of the consumer. The OS page cache does not prefetch on an application's
//! access pattern; a producer thread walking a known traversal does. Because
//! `StreamingDataset` has no `__getitem__`, the traversal is fixed and fully known, so
//! the prefetch is speculation-free. (VCF/PGEN backends, #276, DO have decode to
//! amortize — that is the other half of the premise.)
//!
//! Pattern cribbed from genoray's `orchestrator.rs`: named per-stage threads, a
//! `crossbeam_channel::bounded` for backpressure, shutdown by `Sender` drop, and
//! join-everything-then-classify-panics (early-returning on a producer error would
//! leave the consumer blocked on `recv()` forever).
//!
//! **Slot recycling is NET-NEW here** — genoray does *not* recycle (it allocates each
//! chunk fresh and drops it). We return drained slots to the producer so memory is
//! capped at `n_slots * slot_bytes` regardless of plan length. Start with 2 slots
//! (ping-pong); promote to an N-slot ring only on profiling evidence.

use crossbeam_channel::bounded;

/// One window of the fixed cartesian traversal: regions `[r_lo, r_hi)` on `contig_idx`,
/// crossed with every sample.
#[derive(Clone, Debug)]
pub struct WindowSpec {
    pub contig_idx: usize,
    pub r_lo: usize,
    pub r_hi: usize,
}

/// A source that can fill a window buffer. `Sync` because the producer thread borrows
/// it across the spawn.
///
/// `Buffer` is per-backend by design: SVAR1's is degenerate (offsets only — its
/// on-disk layout is already the target representation), while VCF/PGEN's is an owned
/// decoded table. Do not expect one concrete buffer type across backends.
pub trait StreamBackend: Sync {
    type Buffer: Send;
    /// Fill `slot` with `window`'s data. Called on the producer thread. Implementations
    /// should reuse `slot`'s allocation rather than replacing it (slot recycling).
    fn fill(&self, window: &WindowSpec, slot: &mut Self::Buffer) -> anyhow::Result<()>;
}

/// Drive `windows` through a producer/consumer pair, calling `consume` on each filled
/// slot **in plan order**. `n_slots` bounds live buffers (2 = ping-pong).
///
/// Both stages' errors surface as `Err`; neither can hang the other. Slots are recycled
/// via a return channel, so memory is `n_slots * slot_bytes` regardless of plan length.
pub fn run_windows<B, F>(
    backend: &B,
    windows: &[WindowSpec],
    n_slots: usize,
    mut consume: F,
) -> anyhow::Result<()>
where
    B: StreamBackend,
    B::Buffer: Default,
    F: FnMut(&B::Buffer) -> anyhow::Result<()>,
{
    if windows.is_empty() {
        return Ok(());
    }
    let n_slots = n_slots.max(2);

    // filled: producer -> consumer. free: consumer -> producer (slot recycling).
    let (tx_filled, rx_filled) = bounded::<B::Buffer>(n_slots);
    let (tx_free, rx_free) = bounded::<B::Buffer>(n_slots);
    for _ in 0..n_slots {
        tx_free.send(B::Buffer::default()).expect("prefill free slots");
    }

    std::thread::scope(|scope| -> anyhow::Result<()> {
        let producer = std::thread::Builder::new()
            .name("gvl-stream-producer".into())
            .spawn_scoped(scope, || -> anyhow::Result<()> {
                for w in windows {
                    // Recycle a drained slot. Err => consumer is gone; stop quietly and
                    // let the consumer's own error be the one reported.
                    let Ok(mut slot) = rx_free.recv() else { return Ok(()) };
                    backend.fill(w, &mut slot)?;
                    if tx_filled.send(slot).is_err() {
                        return Ok(()); // consumer gone
                    }
                }
                Ok(())
            })?;

        // Drop our copies so the consumer's recv() can observe close when the producer
        // finishes. (orchestrator.rs:184-186 documents this exact hazard: a stray
        // Sender clone held for introspection blocks shutdown forever.)
        drop(tx_filled);
        drop(tx_free.clone());

        let mut consumer_err: Option<anyhow::Error> = None;
        while let Ok(slot) = rx_filled.recv() {
            if consumer_err.is_none() {
                if let Err(e) = consume(&slot) {
                    consumer_err = Some(e);
                }
            }
            // Always recycle, even after an error, so the producer can finish rather
            // than block on rx_free.recv() -- otherwise the join below deadlocks.
            let _ = tx_free.send(slot);
        }

        // Join FIRST, classify AFTER -- never early-return with the producer live.
        match producer.join() {
            Err(_) => anyhow::bail!("streaming producer thread panicked"),
            Ok(Err(e)) => return Err(e),
            Ok(Ok(())) => {}
        }
        if let Some(e) = consumer_err {
            return Err(e);
        }
        Ok(())
    })
}
```

- [ ] **Step 4: Run the engine tests**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && pixi run -e dev cargo-test 2>&1 | grep -E "stream::|test result" | head -10`
Expected: all 5 `stream::tests::*` PASS. **Run it 20x to shake out races:**
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine && for i in $(seq 1 20); do cargo test --release stream:: 2>&1 | grep -c "test result: ok" ; done
```
Expected: `1` twenty times. Any hang is a shutdown-ordering bug — re-read the `Sender`-drop notes.

- [ ] **Step 5: Wire the engine into the SVAR1 read path**

Implement `StreamBackend` for `Svar1Store` and route `reconstruct_haplotypes_svar1` through `run_windows`. **Scope note:** the current FFI reconstructs one window per call, which the Python `_plan` already drives sequentially. Wiring the engine means moving the window loop *into* Rust — a larger change than this plan's remaining budget. **If that proves large, STOP and split it into a follow-up task rather than rushing it**; the asymptotic fix (Tasks 2/4) is already landed and independently valuable, and the engine's benefit is a separate, separately-measured effect. Record the decision in the roadmap either way.

- [ ] **Step 6: Measure the overlap (separately from the asymptotic fix)**

Measure producer/consumer overlap on a **cold page cache** — the two effects must not be conflated in one number. Drop caches between runs if permitted, or use a store large enough to exceed RAM. Report as secondary color, not as the gate.

- [ ] **Step 7: Full verification + commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine
pixi run -e dev maturin develop --release 2>&1 | tail -3
pixi run -e dev pytest tests -q 2>&1 | tail -5           # FULL tree -- scoped runs skip tests/unit/
pixi run -e dev cargo-test 2>&1 | tail -3
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck 2>&1 | tail -3
git add -A
git commit -m "feat(streaming): crossbeam double-buffer engine behind StreamBackend"
```

---

### Task 6: Roadmap, verification, and the PR

**Files:**
- Modify: `docs/roadmaps/streaming-dataset.md`

- [ ] **Step 1: Tick the roadmap**

Set the Plan-2 task markers to ✅, fill the Plans table's Plan 2 row with this plan's path and the PR link, and record the measurement results (entries-touched before/after; batch-size invariance; engine overlap if Task 5 completed) under the Plan-2 bullet. If Task 5 Step 5 was split out, set that task to 🚧 and say so explicitly with a follow-up issue link.

- [ ] **Step 2: Final full verification**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine
pixi run -e dev maturin develop --release 2>&1 | tail -3
pixi run -e dev pytest tests -q 2>&1 | tail -5
pixi run -e dev cargo-test 2>&1 | tail -3
pixi run -e dev typecheck 2>&1 | tail -3
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: all green, `MISSING: none`. **Do not open the PR otherwise.**

- [ ] **Step 3: Push and open the PR**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-window-engine
pixi run -e dev prek-install
git push -u origin HEAD
gh pr create --draft --base docs/streaming-roadmap-issues --title "perf(streaming)!: SVAR1 window reads + double-buffer engine (#275)" --body "Closes #275. Spec: \`docs/superpowers/specs/2026-07-16-streaming-svar1-window-engine-design.md\`. Depends on d-laub/genoray#123 (merged; consumed via rev bump).

Root cause: the walking skeleton read SVAR1 through \`Svar1RecordSource\` — the *conversion pipeline's record producer*, not a query API — because genoray had an ungated Rust query surface for SVAR2 and none for SVAR1. That constructor is **O(all CSR entries)** and the skeleton called it **once per batch**.

All four of #275's 'debt' items are artifacts of that one dependency and are **deleted, not optimized**: the \`.tolist()\` table and \`set_contig_table\` existed only to feed it; the per-batch clones existed only because it takes its vectors by value; the walk becomes a binary search; and the GIL fix is one lifetime parameter (\`PyRef<'py>\`, per the \`Svar2Store\` template).

Window ≫ batch: one \`find_ranges\` per window, batch is a slice of it. \`geno_v_idxs\` is the \`variant_idxs\` mmap slice itself — zero copy.

**Gate is a deterministic counter** (CSR entries touched per window), not wall-clock — this node is too noisy. See \`tests/dataset/test_streaming_scale.py\`.

BREAKING: \`__iter__\` removed in favor of \`to_iter(batch_size=...)\`; \`to_torch_dataset()\` now returns an IterableDataset instead of raising.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Notes for the reviewer

- **The parity test is the control.** It should need only a mechanical `__iter__`→`to_iter` update. Anything more means behavior changed.
- **Task 5 is where the risk is** — gvl's first threading primitive. The shutdown ordering (`Sender` drop) and join-then-classify are the parts that hang or lie if gotten wrong; the tests for those are deliberately adversarial.
- **Do not resurrect `Svar1RecordSource`.** If a future need looks like it wants a record walk over SVAR1, it almost certainly wants a range query instead.
