# genoray ungated `svar1_query` — Implementation Plan (Plan 2a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallelism:** Tasks 1 and 2 are independent — dispatch them concurrently via superpowers:dispatching-parallel-agents. Task 3 needs Task 2. Task 4 needs 1–3. Tasks 5 and 6 are independent of each other and both need Task 4.
>
> **Model policy:** use Sonnet (or weaker) for implementation subagents. Escalate to a stronger model only for a second-pass fix where the implementer critically failed.

**Goal:** Add an ungated Rust SVAR1 range-query API to genoray (`Svar1Reader` + `var_ranges` + cartesian `find_ranges`), so consumers can query SVAR1 by genomic range without htslib and without the conversion pipeline's record producer.

**Architecture:** Two independent stages mirroring the Python path. **Stage A** (`var_ranges`) maps POS ranges → global variant-id ranges; it is a thin wrapper over the **already-existing, already-ungated** `search::overlap_range` and touches no reader. **Stage B** (`find_ranges`) maps variant-id ranges → absolute CSR index pairs via two `partition_point`s per haplotype. Output stops at index pairs — SVAR1 needs no gather (unlike SVAR2, which merges two channels and decodes keys).

**Tech Stack:** Rust 2024, PyO3 0.29 (`multiple-pymethods`), memmap2, bytemuck, ndarray; pixi; pytest + cargo test.

**Repo:** `/carter/users/dlaub/projects/genoray` (clean, on `main`). **This plan is entirely in genoray — do not touch GenVarLoader.**

**Issue:** d-laub/genoray#123. **Consumer:** mcvickerlab/GenVarLoader#275 (Plan 2b), via a Cargo `rev` bump after this merges to `main`.

## Global Constraints

- **The new module MUST be ungated.** genoray's gating criterion is exactly "does it pull rust-htslib/zstd". `svar1_query` uses only memmap2 + bytemuck + `crate::search` — all ungated. It must **never** import from `svar1_reader`, `record_source`, or `svar2_view` (all `#[cfg(feature = "conversion")]`).
- **`cargo test` always runs with `conversion` ON** (`test-rust = "cargo test --no-default-features --features conversion"`). The gate is therefore enforced ONLY by `check-core` (`cargo check --no-default-features`) plus the compile-guard test. **Both must be run.**
- **Commands** (note the `lint` env, NOT `dev`):
  - `pixi run -e lint test-rust` — cargo tests
  - `pixi run -e lint check-core` — the ungated gate
  - `pixi run test` — pytest (depends-on `gen`)
  - `pixi run typecheck` — `pyrefly check python/genoray`
- **Crate name is `genoray_core`** (package is `genoray`). Integration tests in `tests/*.rs` are an external crate and `use genoray_core::…`.
- **SVAR1 files are headerless raw buffers despite the `.npy` extension.** `variant_idxs.npy` = raw `i32`; `offsets.npy` = raw `i64`, length `n_samples * ploidy + 1`. Use `bytemuck`, **never** `ndarray_npy::read_npy` (that is only correct for SVAR2's real `.npy` sidecars).
- **Empty ranges must be in-bounds zero-length (`start == stop`), never a sentinel.** An out-of-range offset is poison downstream: seqpro's `Ragged.to_packed` multiplies the offset by element size and overflows int64 even for an empty row (`python/genoray/_svar/_kernels.py:239-243`).
- **`SearchTree` reserves `u32::MAX`** as its padding sentinel (`src/search.rs:13`); stored positions must be `< u32::MAX`. Genomic positions always are.
- **Contigs are contiguous in global-id space** — this contig owns global ids `[contig_start, contig_start + n_local)`.
- **Hap column = `sample * ploidy + p`** (sample-major, ploidy-minor). Consistent with SVAR2.

---

### Task 1: `var_ranges` — Stage A (pure; no reader)

**Files:**
- Create: `src/svar1_query.rs`
- Modify: `src/lib.rs` (add `pub mod svar1_query;` with a NOTE, after `pub mod spine;`)

**Interfaces:**
- Consumes: `crate::search::{SearchTree, overlap_range}` (existing, ungated).
- Produces:
  ```rust
  pub fn var_ranges(
      v_starts: &[u32], v_ends: &[u32], max_v_len: u32,
      contig_start: u32, regions: &[(u32, u32)],
  ) -> Vec<std::ops::Range<u32>>
  ```
  Returns one **global** half-open variant-id range per region, in `regions` order. Zero-length (`start == end`) when nothing overlaps.

**Background the implementer needs:**

`search::overlap_range` already exists and already implements this algorithm — its doc says it "mirrors the SVAR 1.0 `var_ranges` shape". Its signature (`src/search.rs:161`):

```rust
pub fn overlap_range(
    tree: &SearchTree,
    v_ends: &[u32],
    max_region_length: u32,
    q_start: u32,
    q_end: u32,
) -> (usize, usize)
```

It returns **contig-local** indices with an **exclusive** end, and `(ub, ub)` when nothing overlaps. Your job is to build the tree once for the whole batch, call it per region, and shift local → global by `contig_start`.

**On `max_v_len` (read this — it is NOT a bug):** `overlap_range`'s contract is `max_region_length >= max_i (v_ends[i] - v_starts[i] - 1)`, i.e. a `>=` bound. Python computes `max_v_len = (v_ends - v_starts).max()`, which is exactly 1 larger. **That is an over-estimate, and over-estimating is provably safe** — it only widens `lb = tree.lower_bound(q_start.saturating_sub(max_region_length))`, so the forward sub-scan still finds the same first true overlap. genoray's own `tests/test_query.rs` says so: *"an over-estimate is provably overshoot-safe (see `search.rs`)"*. We take `max_v_len` as a parameter using **Python's convention** so callers can pass `(v_ends - v_starts).max()` directly. Under-estimating would be a real bug; do not "optimize" this by subtracting 1.

**`v_ends` convention:** `v_end = POS - min(ILEN, 0)` where POS is 0-based (`python/genoray/_var_ranges.py:21-27`). So a SNP at 0-based start `s` has `v_end == s + 1`.

- [ ] **Step 1: Write the failing test**

Create `src/svar1_query.rs` containing ONLY this test module (the impl comes in Step 3):

```rust
//! Ungated SVAR1 range-query core.

#[cfg(test)]
mod tests {
    use super::*;

    // Three variants on a contig whose global ids start at 100.
    // local 0: SNP  at 10 -> v_end 11
    // local 1: DEL  at 20, ILEN -3 -> v_end 23
    // local 2: SNP  at 30 -> v_end 31
    // max_v_len (Python convention) = max(v_ends - v_starts) = max(1, 3, 1) = 3
    fn fixture() -> (Vec<u32>, Vec<u32>, u32) {
        (vec![10, 20, 30], vec![11, 23, 31], 3)
    }

    #[test]
    fn var_ranges_maps_local_overlap_to_global_ids() {
        let (vs, ve, mvl) = fixture();
        // [10, 21) overlaps local 0 (SNP@10) and local 1 (DEL@20) -> global 100..102
        let got = var_ranges(&vs, &ve, mvl, 100, &[(10, 21)]);
        assert_eq!(got, vec![100..102]);
    }

    #[test]
    fn var_ranges_deletion_spanning_query_start_is_included() {
        // The whole point of the sub-scan: a DEL starting BEFORE the query still
        // deletes bases inside it. Query [21, 22) starts after the DEL's POS (20)
        // but before its end (23), so local 1 must be included.
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 100, &[(21, 22)]);
        assert_eq!(got, vec![101..102]);
    }

    #[test]
    fn var_ranges_no_overlap_is_zero_length_not_sentinel() {
        // A zero-length in-bounds range -- NEVER a sentinel like u32::MAX. An
        // out-of-range offset overflows int64 in seqpro's Ragged.to_packed.
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 100, &[(50, 60)]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].start, got[0].end, "no-overlap must be zero-length");
    }

    #[test]
    fn var_ranges_empty_contig_yields_zero_length_ranges() {
        // n_local == 0: must not panic (a .max() over an empty slice would).
        let got = var_ranges(&[], &[], 0, 42, &[(0, 100), (5, 6)]);
        assert_eq!(got, vec![42..42, 42..42]);
    }

    #[test]
    fn var_ranges_batches_regions_in_order() {
        let (vs, ve, mvl) = fixture();
        let got = var_ranges(&vs, &ve, mvl, 0, &[(30, 31), (10, 11)]);
        assert_eq!(got, vec![2..3, 0..1], "output must be in `regions` order");
    }
}
```

Add to `src/lib.rs`, immediately after the `pub mod spine;` line:

```rust
// NOTE: `svar1_query` is ungated even though `svar1_reader` (the conversion-side
// SVAR1 RecordSource) is gated: it reads the same raw mmap'd buffers but has zero
// htslib/zstd dependency (memmap2 + bytemuck + `search.rs` only), and gvl links it
// with `default-features = false`. Same reasoning as `query` above.
pub mod svar1_query;
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | tail -20`
Expected: FAIL — compile error, `cannot find function 'var_ranges' in this scope`.

- [ ] **Step 3: Write minimal implementation**

Prepend to `src/svar1_query.rs` (above the `#[cfg(test)] mod tests`), keeping the `//!` module doc at the very top of the file:

```rust
//! Ungated SVAR1 range-query core: the query counterpart to the conversion-gated
//! `svar1_reader::Svar1RecordSource`.
//!
//! Two independent stages, mirroring `python/genoray/_var_ranges.py` +
//! `python/genoray/_svar/_kernels.py::_find_starts_ends`:
//!
//! * [`var_ranges`] — POS ranges -> global variant-id ranges. Pure; a thin wrapper
//!   over `search::overlap_range`, which already ports the Python algorithm.
//! * [`find_ranges`] — variant-id ranges -> absolute CSR index pairs into the
//!   `variant_idxs` mmap, via two `partition_point`s per haplotype.
//!
//! There is deliberately **no `gather_ranges`**: SVAR2 needs one because it merges
//! two channels and decodes keys, but SVAR1's on-disk layout is already the target
//! representation, so consumers build a zero-copy view straight from the index pairs
//! (cf. `SparseVar.read_ranges` -> `Ragged.from_offsets`).

use std::ops::Range;

use crate::search::{SearchTree, overlap_range};

/// POS ranges -> **global** half-open variant-id ranges, one per region, in
/// `regions` order.
///
/// * `v_starts` / `v_ends` — this contig's LOCAL 0-based variant starts (ascending)
///   and exclusive ends (`v_end = POS - min(ILEN, 0)`; a SNP at `s` has `v_end == s+1`).
/// * `max_v_len` — `max(v_ends - v_starts)` over the contig, i.e. **Python's
///   `var_ranges` convention** (`_var_ranges.py:78`). `overlap_range` only requires a
///   `>=` bound on the deletion span, so this over-estimates by exactly 1 and is
///   provably overshoot-safe (it merely widens the candidate window). Do NOT subtract
///   1 to "tighten" it — under-estimating IS a correctness bug.
/// * `contig_start` — this contig's first variant's GLOBAL id. Contigs are contiguous
///   in global-id space.
///
/// Nothing overlapping yields an **in-bounds zero-length** range (`start == end`),
/// never a sentinel: an out-of-range offset is poison for downstream byte math
/// (seqpro `Ragged.to_packed` overflows int64 even for an empty row). This
/// deliberately differs from Python `var_ranges`, which returns `INT32_MAX`.
///
/// Only the endpoints are guaranteed to overlap — an interior id can be a
/// deletion-spanned non-overlap. Same contract as `search::overlap_range` and SVAR 1.0
/// `var_ranges`.
pub fn var_ranges(
    v_starts: &[u32],
    v_ends: &[u32],
    max_v_len: u32,
    contig_start: u32,
    regions: &[(u32, u32)],
) -> Vec<Range<u32>> {
    debug_assert_eq!(v_starts.len(), v_ends.len());
    // An empty contig has no tree to build and no ends to scan.
    if v_starts.is_empty() {
        return regions.iter().map(|_| contig_start..contig_start).collect();
    }
    // One tree for the whole batch: `overlap_range` is called per region but the
    // tree build is hoisted, mirroring the SVAR2 search/gather split's intent.
    let tree = SearchTree::new(v_starts);
    regions
        .iter()
        .map(|&(q_start, q_end)| {
            let (s, e) = overlap_range(&tree, v_ends, max_v_len, q_start, q_end);
            (contig_start + s as u32)..(contig_start + e as u32)
        })
        .collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | grep -E "svar1_query|test result" | head -20`
Expected: the 5 `svar1_query::tests::var_ranges_*` tests PASS.

- [ ] **Step 5: Verify the ungated gate still holds**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint check-core 2>&1 | tail -5`
Expected: `Finished` — no errors. (If this fails, you imported something gated.)

- [ ] **Step 6: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
git add src/svar1_query.rs src/lib.rs
git commit -m "feat(svar1): ungated var_ranges (Stage A) over search::overlap_range"
```

---

### Task 2: `Svar1Reader` — open + mmap + accessors

**Files:**
- Modify: `src/svar1_query.rs`

**Interfaces:**
- Consumes: nothing from Task 1 (independent — can run concurrently).
- Produces:
  ```rust
  pub struct Svar1Reader { /* private */ }
  impl Svar1Reader {
      pub fn open(svar1_dir: &str, n_samples: usize, ploidy: usize) -> std::io::Result<Self>;
      pub fn n_samples(&self) -> usize;
      pub fn ploidy(&self) -> usize;
      pub fn variant_idxs(&self) -> &[i32];
      pub fn offsets(&self) -> &[i64];
  }
  ```

**Background the implementer needs:**

An SVAR1 store is **one flat directory** (NOT per-contig like SVAR2's `{out}/{chrom}/`). The two files this reader needs:

| File | dtype | Format | Length |
|---|---|---|---|
| `variant_idxs.npy` | `i32` | **headerless raw** | one entry per non-ref call; each hap's slice holds **sorted global** variant ids |
| `offsets.npy` | `i64` | **headerless raw** | `n_samples * ploidy + 1` (CSR over haplotypes) |

The `.npy` extension is a **lie** — there is no npy header. Python opens them with `np.memmap` (a flat raw buffer); Rust reads them with `bytemuck` and **no header skip**. Do not use `ndarray_npy::read_npy` (that is for SVAR2's real `.npy` sidecars only).

**Residency policy:** `variant_idxs` must stay **mmap'd** (it can be huge). `offsets` is only `num_haps + 1` long, so loading it resident matches the SVAR2 precedent (`sidecar.rs:59`) and is cheap.

**Why a local mmap helper:** `query::sidecar::mmap_file` is `pub(crate)` inside a **private** `sidecar` module, so it is not reachable from this sibling module. Write a local helper — this also keeps the ungated module free of any dependency that might later drift behind the gate. Mirror `svar1_reader.rs:96-106`'s `mmap_ro` shape and its SAFETY note.

**Alignment:** `std::fs::read` returns a `Vec<u8>` with no alignment guarantee, so `bytemuck::cast_slice::<u8, i64>` would panic. Use `bytemuck::pod_collect_to_vec`, which handles unaligned input. mmap pages ARE page-aligned, so `cast_slice` is fine for `variant_idxs` (see `sidecar.rs:38`'s comment).

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `src/svar1_query.rs`:

```rust
    use std::io::Write;

    /// Write a HEADERLESS raw buffer. SVAR1's `*.npy` files have no npy header
    /// despite the extension -- Python np.memmaps them, Rust bytemucks them.
    /// Mirrors `svar1_reader.rs`'s test helper of the same name.
    fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(bytemuck::cast_slice(data)).unwrap();
    }

    /// 2 samples x ploidy 2 = 4 haps. Per-hap sorted global ids:
    ///   hap0: [0, 2, 4]   hap1: [3]   hap2: [2]   hap3: []
    fn write_store(dir: &std::path::Path) {
        write_raw::<i32>(dir, "variant_idxs.npy", &[0, 2, 4, 3, 2]);
        write_raw::<i64>(dir, "offsets.npy", &[0, 3, 4, 5, 5]);
    }

    #[test]
    fn reader_opens_and_exposes_raw_buffers() {
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        assert_eq!(r.n_samples(), 2);
        assert_eq!(r.ploidy(), 2);
        assert_eq!(r.variant_idxs(), &[0, 2, 4, 3, 2]);
        assert_eq!(r.offsets(), &[0, 3, 4, 5, 5]);
    }

    #[test]
    fn reader_missing_dir_is_err() {
        assert!(Svar1Reader::open("/no/such/svar1/store", 2, 2).is_err());
    }

    #[test]
    fn reader_rejects_offsets_of_wrong_length() {
        // offsets MUST be num_haps + 1. A mismatch means the caller's
        // n_samples/ploidy disagree with the store -- fail loudly rather than
        // index out of bounds later inside find_ranges.
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 2]); // len 3 => 2 haps
        let err = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2); // wants 5
        assert!(err.is_err(), "offsets length mismatch must be an error");
    }

    #[test]
    fn reader_empty_variant_idxs_is_ok() {
        // A store where no hap carries any non-ref call: variant_idxs is
        // zero-length. memmap2 rejects empty maps, so this must not blow up.
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[] as &[i32]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 0, 0, 0, 0]);
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        assert_eq!(r.variant_idxs(), &[] as &[i32]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | tail -20`
Expected: FAIL — `cannot find type 'Svar1Reader' in this scope`.

- [ ] **Step 3: Write minimal implementation**

Add to the imports at the top of `src/svar1_query.rs`:

```rust
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
```

Then add the reader, above the `#[cfg(test)] mod tests` block:

```rust
/// mmap a file read-only, returning `None` for a zero-length file (memmap2 rejects
/// empty maps; an SVAR1 store where no hap carries a call has an empty
/// `variant_idxs`). Local rather than reusing `query::sidecar::mmap_file`, which is
/// `pub(crate)` inside a private module — and keeping this module's dependencies
/// minimal is what keeps it ungated.
fn mmap_ro(path: &Path) -> std::io::Result<Option<Mmap>> {
    let f = File::open(path)?;
    if f.metadata()?.len() == 0 {
        return Ok(None);
    }
    // SAFETY: a finished, read-only store artifact; we never mutate the file while
    // it is mapped. Same contract as `query::sidecar::mmap_file`.
    Ok(Some(unsafe { Mmap::map(&f)? }))
}

/// An SVAR1 store opened for range queries. Holds `variant_idxs` mmap'd (it is one
/// entry per non-ref call — never materialize it) and the small CSR `offsets`
/// resident (`num_haps + 1`), mirroring the SVAR2 `SubStreamView` split.
///
/// Unlike SVAR2's `ContigReader`, this takes **no `chrom`**: an SVAR1 store is one
/// flat directory and contigs are contiguous in global-id space. Per-contig scoping
/// is the caller's job, via `var_ranges`'s `contig_start`.
pub struct Svar1Reader {
    n_samples: usize,
    ploidy: usize,
    variant_idxs: Option<Mmap>,
    offsets: Vec<i64>,
}

impl Svar1Reader {
    /// Open the SVAR1 store rooted at `svar1_dir` for a cohort of `n_samples` at
    /// `ploidy`.
    ///
    /// NOTE: `variant_idxs.npy` and `offsets.npy` are **headerless raw buffers**
    /// despite the extension (Python np.memmaps them). Do not reach for
    /// `ndarray_npy::read_npy` — that is only correct for SVAR2's real `.npy`
    /// sidecars.
    pub fn open(svar1_dir: &str, n_samples: usize, ploidy: usize) -> std::io::Result<Self> {
        let dir = Path::new(svar1_dir);
        let variant_idxs = mmap_ro(&dir.join("variant_idxs.npy"))?;

        // `fs::read` gives an unaligned Vec<u8>, so `cast_slice` would panic;
        // `pod_collect_to_vec` copies element-wise and is alignment-safe. `offsets`
        // is tiny (num_haps + 1), so the copy is free.
        let offsets_bytes = std::fs::read(dir.join("offsets.npy"))?;
        let offsets: Vec<i64> = bytemuck::pod_collect_to_vec(&offsets_bytes);

        let want = n_samples * ploidy + 1;
        if offsets.len() != want {
            return Err(std::io::Error::other(format!(
                "{}/offsets.npy has {} entries; expected n_samples*ploidy+1 = {} \
                 (n_samples={n_samples}, ploidy={ploidy})",
                svar1_dir,
                offsets.len(),
                want,
            )));
        }

        Ok(Self {
            n_samples,
            ploidy,
            variant_idxs,
            offsets,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn ploidy(&self) -> usize {
        self.ploidy
    }

    /// The flat `variant_idxs` buffer: each hap's `offsets[h]..offsets[h+1]` slice
    /// holds its sorted global non-ref variant ids. Exposed so consumers can hand it
    /// straight to a kernel as a zero-copy sparse-index input.
    ///
    /// mmap pages are page-aligned, so `bytemuck`'s alignment check always passes;
    /// a missing/empty map yields an empty slice.
    pub fn variant_idxs(&self) -> &[i32] {
        match &self.variant_idxs {
            Some(m) => bytemuck::cast_slice(&m[..]),
            None => &[],
        }
    }

    /// CSR offsets over haplotypes; `len() == n_samples * ploidy + 1`. Hap column is
    /// `sample * ploidy + p`.
    pub fn offsets(&self) -> &[i64] {
        &self.offsets
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | grep -E "svar1_query|test result" | head -20`
Expected: the 4 `reader_*` tests PASS (plus Task 1's, if merged).

- [ ] **Step 5: Verify the gate**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint check-core 2>&1 | tail -5`
Expected: `Finished`, no errors.

- [ ] **Step 6: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
git add src/svar1_query.rs
git commit -m "feat(svar1): ungated Svar1Reader (mmap variant_idxs, resident offsets)"
```

---

### Task 3: `find_ranges` — Stage B (cartesian)

**Files:**
- Modify: `src/svar1_query.rs`

**Interfaces:**
- Consumes: `Svar1Reader` (Task 2), `Range<u32>` from `var_ranges` (Task 1).
- Produces:
  ```rust
  pub struct Svar1RangesBundle {
      pub n_ranges: usize,
      pub n_samples: usize,
      pub ploidy: usize,
      pub sample_cols: Vec<usize>,
      pub starts: Vec<i64>,   // len n_ranges*n_samples*ploidy, C-order (range, sample, ploid)
      pub stops: Vec<i64>,    // same layout
  }
  pub fn find_ranges(
      reader: &Svar1Reader, ranges: &[Range<u32>], samples: Option<&[usize]>,
  ) -> Svar1RangesBundle
  ```

**Background the implementer needs:**

This is the Rust port of `_find_starts_ends` (`python/genoray/_svar/_kernels.py:192-248`). The whole algorithm is: **each hap's CSR slice holds sorted global variant ids, so a `[v_lo, v_hi)` id range maps to a sub-slice by two binary searches.** Python does it with `np.searchsorted(sp_genos, var_ranges).T + o_s`; Rust does it with `partition_point` — the exact idiom already at `svar1_reader.rs:30-31`:

```rust
let s = hap.partition_point(|&g| g < contig_start);
let e = hap.partition_point(|&g| g < contig_end);
```

**Output layout** matches Python's `(2, r, s, p)`: `starts` is `out[0].ravel()`, `stops` is `out[1].ravel()`, both C-order over `(range, sample, ploid)`. Indices are **absolute** into the flat `variant_idxs` buffer (Python adds `o_s`; so do we).

**Cartesian, not pairwise** — `ranges × samples × ploidy`. This matches `_find_starts_ends`'s existing contract, and the only consumer (`StreamingDataset`) traverses a cartesian product by construction.

**Do NOT port Python's `argsort`/`unsorter` round-trip** (`_kernels.py:227-228, 245-246`). Each binary search is independent, so it is not needed for correctness in `_find_starts_ends`; it exists for memory-access locality under numba's `prange`. (It IS load-bearing in `_find_starts_ends_with_length`, which is out of scope.)

**No rayon.** The SVAR2 query core is deliberately single-threaded (`gather.rs:181`: *"Single-threaded; `rayon` over the H hap-slices … are M6b concerns"*). Match that; the consumer parallelizes.

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `src/svar1_query.rs`:

```rust
    #[test]
    fn find_ranges_binary_searches_each_hap_csr() {
        // hap0: [0, 2, 4] @ entries 0..3   hap1: [3] @ entry 3
        // hap2: [2] @ entry 4              hap3: []  @ entry 5..5
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();

        // id range [2, 5): hap0 -> entries 1..3 ([2,4]); hap1 -> 3..4 ([3]);
        //                  hap2 -> 4..5 ([2]);          hap3 -> 5..5 (empty)
        let b = find_ranges(&r, &[2..5], None);
        assert_eq!(b.n_ranges, 1);
        assert_eq!(b.n_samples, 2);
        assert_eq!(b.ploidy, 2);
        assert_eq!(b.sample_cols, vec![0, 1]);
        // C-order (range, sample, ploid) -> hap0, hap1, hap2, hap3
        assert_eq!(b.starts, vec![1, 3, 4, 5]);
        assert_eq!(b.stops, vec![3, 4, 5, 5]);
    }

    #[test]
    fn find_ranges_empty_id_range_is_in_bounds_zero_length() {
        // A zero-length input range must produce start == stop, in bounds --
        // never a sentinel (poison for seqpro Ragged.to_packed's int64 math).
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        let b = find_ranges(&r, &[7..7], None);
        for (s, e) in b.starts.iter().zip(&b.stops) {
            assert_eq!(s, e, "empty range must be zero-length");
            assert!(*s >= 0 && *s <= 5, "offset {s} must be in bounds of variant_idxs");
        }
    }

    #[test]
    fn find_ranges_sample_subset_selects_and_reorders() {
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        // sample 1 only -> haps 2, 3
        let b = find_ranges(&r, &[2..5], Some(&[1]));
        assert_eq!(b.n_samples, 1);
        assert_eq!(b.sample_cols, vec![1]);
        assert_eq!(b.starts, vec![4, 5]);
        assert_eq!(b.stops, vec![5, 5]);

        // reordered subset -> hap order follows sample_cols, not store order
        let b = find_ranges(&r, &[2..5], Some(&[1, 0]));
        assert_eq!(b.starts, vec![4, 5, 1, 3]);
    }

    #[test]
    fn find_ranges_multiple_ranges_are_c_order() {
        let tmp = tempfile::tempdir().unwrap();
        write_store(tmp.path());
        let r = Svar1Reader::open(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        let b = find_ranges(&r, &[0..1, 2..5], None);
        assert_eq!(b.n_ranges, 2);
        // range 0 ([0,1)): hap0 -> 0..1, others empty
        // range 1 ([2,5)): as above
        assert_eq!(b.starts, vec![0, 3, 4, 5, /* range 1 */ 1, 3, 4, 5]);
        assert_eq!(b.stops, vec![1, 3, 4, 5, /* range 1 */ 3, 4, 5, 5]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'find_ranges' in this scope`.

- [ ] **Step 3: Write minimal implementation**

Add above the `#[cfg(test)] mod tests` block in `src/svar1_query.rs`:

```rust
/// Absolute CSR index pairs for a cartesian `(range, sample, ploid)` query.
///
/// `starts`/`stops` are each `n_ranges * n_samples * ploidy` long in C-order
/// `(range, sample, ploid)` — i.e. exactly Python `_find_starts_ends`'s `(2, r, s, p)`
/// output with the leading axis split into two vectors. Indices are absolute into
/// [`Svar1Reader::variant_idxs`].
pub struct Svar1RangesBundle {
    pub n_ranges: usize,
    pub n_samples: usize,
    pub ploidy: usize,
    /// Original sample indices, in output order (identity when `samples` was `None`).
    pub sample_cols: Vec<usize>,
    pub starts: Vec<i64>,
    pub stops: Vec<i64>,
}

/// Variant-id ranges -> absolute CSR index pairs. The Rust port of
/// `_find_starts_ends` (`python/genoray/_svar/_kernels.py`).
///
/// Each hap's CSR run holds **sorted** global variant ids, so a `[v_lo, v_hi)` id
/// range maps to a sub-slice by two `partition_point`s. `samples`, if given, selects
/// (and reorders) a sample subset by original index; `None` means all samples in
/// store order.
///
/// Empty results are **in-bounds zero-length** (`start == stop`), never a sentinel —
/// see [`var_ranges`].
///
/// Single-threaded by design, matching the SVAR2 query core (`query/gather.rs`); the
/// consumer owns parallelism.
pub fn find_ranges(
    reader: &Svar1Reader,
    ranges: &[Range<u32>],
    samples: Option<&[usize]>,
) -> Svar1RangesBundle {
    let ploidy = reader.ploidy();
    let sample_cols: Vec<usize> = match samples {
        Some(s) => s.to_vec(),
        None => (0..reader.n_samples()).collect(),
    };

    let vi = reader.variant_idxs();
    let offs = reader.offsets();
    let n_ranges = ranges.len();
    let n_samples = sample_cols.len();
    let n = n_ranges * n_samples * ploidy;

    let mut starts = Vec::with_capacity(n);
    let mut stops = Vec::with_capacity(n);

    for r in ranges {
        let (lo, hi) = (r.start as i32, r.end as i32);
        for &s in &sample_cols {
            for p in 0..ploidy {
                let h = s * ploidy + p; // sample-major, ploidy-minor
                let o_s = offs[h];
                let o_e = offs[h + 1];
                let hap = &vi[o_s as usize..o_e as usize];
                // + o_s makes the index absolute into the flat buffer (Python does
                // the same: `np.searchsorted(sp_genos, var_ranges).T + o_s`).
                starts.push(hap.partition_point(|&g| g < lo) as i64 + o_s);
                stops.push(hap.partition_point(|&g| g < hi) as i64 + o_s);
            }
        }
    }

    Svar1RangesBundle {
        n_ranges,
        n_samples,
        ploidy,
        sample_cols,
        starts,
        stops,
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | grep -E "svar1_query|test result" | head -20`
Expected: all `svar1_query::tests::*` PASS.

- [ ] **Step 5: Verify the gate**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint check-core 2>&1 | tail -5`
Expected: `Finished`, no errors.

- [ ] **Step 6: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
git add src/svar1_query.rs
git commit -m "feat(svar1): ungated cartesian find_ranges (Stage B) via partition_point"
```

---

### Task 4: PyO3 bindings — `PySvar1Reader`

**Files:**
- Create: `src/py_svar1_query.rs`
- Modify: `src/lib.rs` (add `pub mod py_svar1_query;` and one `add_class` line)

**Interfaces:**
- Consumes: `Svar1Reader`, `var_ranges`, `find_ranges`, `Svar1RangesBundle` (Tasks 1–3).
- Produces (Python):
  ```python
  from genoray._core import PySvar1Reader
  r = PySvar1Reader(svar1_dir: str, n_samples: int, ploidy: int)
  r.n_samples() -> int
  r.ploidy() -> int
  r.var_ranges(v_starts: np.uint32[:], v_ends: np.uint32[:], max_v_len: int,
               contig_start: int, regions: list[tuple[int, int]]) -> np.ndarray  # (r, 2) int64
  r.find_ranges(ranges: np.ndarray, samples: list[int] | None) -> dict  # (r,2) int64 in
  ```
  `find_ranges` returns a dict: `starts` / `stops` (each `(r*s*p,)` int64), `sample_cols` (`(s,)` int64), plus `n_ranges` / `n_samples` / `ploidy` scalars.

**Background the implementer needs:**

genoray's convention (`src/py_query.rs`) is a thin pyclass wrapping the pure-Rust type, with `#[new]` marked `pub` so integration tests can call it as a plain Rust constructor:

```rust
#[pyclass]
pub struct PyContigReader {
    pub(crate) inner: ContigReader,
}

#[pymethods]
impl PyContigReader {
    #[new]
    pub fn new(base_out_dir: &str, chrom: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        let inner = ContigReader::open(base_out_dir, chrom, n_samples, ploidy)?;
        Ok(Self { inner })
    }
}
```

`ContigReader::open` returns `std::io::Result`, and `?` converts it via pyo3's built-in `From<io::Error> for PyErr`. Do the same.

The data contract is **a `PyDict` of numpy arrays**, not a pyclass graph — see `py_query_ranges.rs::bundle_to_dict`. Follow that.

Registration is one ungated line in `#[pymodule] fn _core` (`src/lib.rs:1069`), next to `m.add_class::<crate::py_query::PyContigReader>()?;`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_py_svar1_query.rs`:

```rust
//! Boundary test for the SVAR1 query PyO3 seam: `PySvar1Reader` opens a raw
//! SVAR1 store (headerless buffers, no conversion pipeline needed) and the
//! numpy-dict contract round-trips.

use std::io::Write;

use genoray_core::py_svar1_query::PySvar1Reader;
use pyo3::Python;
use tempfile::tempdir;

fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
    let mut f = std::fs::File::create(dir.join(name)).unwrap();
    f.write_all(bytemuck::cast_slice(data)).unwrap();
}

#[test]
fn py_svar1_reader_opens_a_raw_store() {
    let tmp = tempdir().unwrap();
    write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 2, 4, 3, 2]);
    write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 3, 4, 5, 5]);

    Python::attach(|_py| {
        let r = PySvar1Reader::new(tmp.path().to_str().unwrap(), 2, 2);
        assert!(r.is_ok(), "PySvar1Reader should open a raw SVAR1 store");
        let r = r.unwrap();
        assert_eq!(r.n_samples(), 2);
        assert_eq!(r.ploidy(), 2);
    });
}

#[test]
fn py_svar1_reader_missing_store_is_err() {
    Python::attach(|_py| {
        assert!(PySvar1Reader::new("/no/such/svar1/store", 2, 2).is_err());
    });
}
```

Note: `bytemuck` is a normal dependency (`Cargo.toml`), so it is available to integration tests.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | tail -20`
Expected: FAIL — `unresolved import 'genoray_core::py_svar1_query'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/py_svar1_query.rs`:

```rust
//! Python-facing handle over an SVAR1 store's range-query core. Wraps the pure-Rust
//! `svar1_query::Svar1Reader`. Ungated, like the module it wraps.
//!
//! Mirrors `py_query.rs`/`py_query_ranges.rs`: a thin pyclass plus a numpy-dict data
//! contract (not a pyclass graph). There is no `gather_ranges` counterpart — SVAR1
//! stops at index pairs, which the caller turns into a zero-copy `Ragged` view.

use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::svar1_query::{Svar1RangesBundle, Svar1Reader, find_ranges, var_ranges};

/// An SVAR1 store opened for querying. Constructed from Python as
/// `PySvar1Reader(svar1_dir, n_samples, ploidy)`.
#[pyclass]
pub struct PySvar1Reader {
    pub(crate) inner: Svar1Reader,
}

/// `Svar1RangesBundle` -> numpy dict: `starts`/`stops` (each `(r*s*p,)` int64,
/// C-order `(range, sample, ploid)`), `sample_cols` `(s,)` int64, plus
/// `n_ranges`/`n_samples`/`ploidy` scalars.
fn bundle_to_dict<'py>(py: Python<'py>, b: &Svar1RangesBundle) -> PyResult<Bound<'py, PyDict>> {
    let sample_cols: Vec<i64> = b.sample_cols.iter().map(|&x| x as i64).collect();
    let d = PyDict::new(py);
    d.set_item("starts", PyArray1::from_slice(py, &b.starts))?;
    d.set_item("stops", PyArray1::from_slice(py, &b.stops))?;
    d.set_item("sample_cols", PyArray1::from_slice(py, &sample_cols))?;
    d.set_item("n_ranges", b.n_ranges)?;
    d.set_item("n_samples", b.n_samples)?;
    d.set_item("ploidy", b.ploidy)?;
    Ok(d)
}

#[pymethods]
impl PySvar1Reader {
    // `pub` so integration tests (an external crate) can call it directly as a plain
    // Rust constructor; pyo3 keeps `#[new]` methods callable from Rust.
    #[new]
    pub fn new(svar1_dir: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        let inner = Svar1Reader::open(svar1_dir, n_samples, ploidy)?;
        Ok(Self { inner })
    }

    pub fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    pub fn ploidy(&self) -> usize {
        self.inner.ploidy()
    }

    /// Stage A: POS ranges -> GLOBAL variant-id ranges, `(n_regions, 2)` int64.
    ///
    /// `max_v_len` uses Python `var_ranges`'s convention (`max(v_ends - v_starts)`).
    /// No-overlap yields an in-bounds zero-length row, NOT Python's `INT32_MAX`
    /// sentinel.
    pub fn var_ranges<'py>(
        &self,
        py: Python<'py>,
        v_starts: Vec<u32>,
        v_ends: Vec<u32>,
        max_v_len: u32,
        contig_start: u32,
        regions: Vec<(u32, u32)>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let rs = var_ranges(&v_starts, &v_ends, max_v_len, contig_start, &regions);
        let mut flat: Vec<i64> = Vec::with_capacity(rs.len() * 2);
        for r in &rs {
            flat.push(r.start as i64);
            flat.push(r.end as i64);
        }
        let arr = ndarray::Array2::from_shape_vec((rs.len(), 2), flat)
            .expect("var_ranges shape");
        Ok(arr.to_pyarray(py))
    }

    /// Stage B: `(n_ranges, 2)` GLOBAL variant-id ranges -> the index-pair dict.
    pub fn find_ranges<'py>(
        &self,
        py: Python<'py>,
        ranges: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rs: Vec<std::ops::Range<u32>> = ranges.iter().map(|&(s, e)| s..e).collect();
        let b = find_ranges(&self.inner, &rs, samples.as_deref());
        bundle_to_dict(py, &b)
    }
}
```

In `src/lib.rs`, add next to the other `py_*` modules (they are all ungated):

```rust
pub mod py_svar1_query;
```

and inside `#[pymodule] fn _core`, immediately after the `PyContigReader` line:

```rust
    m.add_class::<crate::py_svar1_query::PySvar1Reader>()?;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | grep -E "py_svar1|test result" | head -10`
Expected: both `py_svar1_reader_*` tests PASS.

- [ ] **Step 5: Verify the gate**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint check-core 2>&1 | tail -5`
Expected: `Finished`. (`py_svar1_query` must be ungated — `py_convert`/`py_query` already are.)

- [ ] **Step 6: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
git add src/py_svar1_query.rs src/lib.rs tests/test_py_svar1_query.rs
git commit -m "feat(svar1): PySvar1Reader bindings (ungated, numpy-dict contract)"
```

---

### Task 5: Python differential tests vs the numba path

**Files:**
- Create: `tests/test_svar1_query_parity.py`

**Interfaces:**
- Consumes: `PySvar1Reader` (Task 4).
- Produces: nothing (test-only).

**Background the implementer needs:**

This is the **highest-value test in the plan**: `search.rs` claims it "mirrors the SVAR 1.0 `var_ranges` shape", but **no Rust↔Python differential test exists**. This creates one.

Two convention gaps you MUST encode rather than "fix":

1. **No-overlap sentinel.** Python `var_ranges` returns `np.iinfo(V_IDX_TYPE).max` (`INT32_MAX`) in **both** columns (`_var_ranges.py:105`). Rust returns an **in-bounds zero-length** range. Both are correct for their consumers; assert the correspondence, don't force equality.
2. **`max_v_len`.** Python's is 1 larger than `overlap_range`'s minimum contract. It is an over-estimate and provably overshoot-safe. Pass Python's value straight through.

Build the store with the real `SparseVar.from_vcf` so the on-disk layout is genuine. genoray's pytest convention is `from genoray import SparseVar` with fixtures in `tests/conftest.py`; `pixi run test` depends on `gen`.

`SparseVar.var_ranges(contig, starts, ends)` is the Python entry point (`_core.py:249`), and `SparseVar._find_starts_ends(contig, starts, ends, samples=...)` returns `(2, r, s, p)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_svar1_query_parity.py`:

```python
"""Differential parity: the ungated Rust `svar1_query` core vs genoray's Python/numba
SVAR1 query path.

`search.rs` has always claimed to "mirror the SVAR 1.0 `var_ranges` shape", but until
now nothing tested that claim. These tests pin it.

Two DELIBERATE convention gaps (assert the correspondence; do not "fix" either side):

1. No-overlap sentinel. Python `var_ranges` returns INT32_MAX in both columns; Rust
   returns an in-bounds zero-length range. Rust's is required downstream -- an
   out-of-range offset overflows int64 in seqpro's `Ragged.to_packed` even for an
   empty row (`_svar/_kernels.py:239-243`).
2. `max_v_len`. Python's `(v_ends - v_starts).max()` is exactly 1 larger than
   `overlap_range`'s `>=` contract wants. It is an OVER-estimate, which only widens
   the candidate window -- provably overshoot-safe. Under-estimating would be a bug.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genoray import SparseVar
from genoray._core import PySvar1Reader
from genoray._types import V_IDX_TYPE
from genoray._var_ranges import _var_end_expr

# chr1: SNP@3, INS@7 (C>CAT), SNP@10, DEL@12 (GTA>G, ILEN -2)  [1-based POS]
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""

SENTINEL = np.iinfo(V_IDX_TYPE).max


@pytest.fixture(scope="module")
def svar1_store(tmp_path_factory) -> Path:
    from genoray import VCF

    d = tmp_path_factory.mktemp("svar1_query_parity")
    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store.svar"
    SparseVar.from_vcf(out, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True)
    return out


def _contig_arrays(sv: SparseVar, contig: str):
    """Contig-local v_starts/v_ends/max_v_len/contig_start, mirroring what
    `_var_ranges.var_ranges` derives internally."""
    idx = sv.index.sort("index").filter(pl.col("CHROM") == contig)
    v_starts = (idx["POS"].to_numpy() - 1).astype(np.uint32)  # 0-based
    v_ends = idx.select(_var_end_expr()).to_series().to_numpy().astype(np.uint32)
    max_v_len = int((v_ends.astype(np.int64) - v_starts.astype(np.int64)).max())
    contig_start = int(idx["index"][0])
    return v_starts, v_ends, max_v_len, contig_start


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 40),   # whole contig
        (0, 5),    # leading
        (3, 20),   # sub-contig, drops the first SNP
        (11, 12),  # inside the DEL's span but after its POS -- the sub-scan case
        (35, 40),  # trailing, no variants
        (20, 21),  # interior, no variants
    ],
)
def test_var_ranges_matches_python(svar1_store: Path, start: int, end: int):
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")

    py = sv.var_ranges("chr1", [start], [end])  # (1, 2), GLOBAL ids
    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs = r.var_ranges(v_starts, v_ends, max_v_len, contig_start, [(start, end)])

    if (py[0] == SENTINEL).all():
        # Convention gap 1: Python signals "no overlap" with a sentinel; Rust with
        # an in-bounds zero-length range.
        assert rs[0, 0] == rs[0, 1], f"rust must be zero-length where python is sentinel: {rs[0]}"
    else:
        np.testing.assert_array_equal(rs[0], py[0].astype(np.int64))


def test_var_ranges_batches_match_python_elementwise(svar1_store: Path):
    """The batched call must agree with Python for every region at once -- a
    per-region loop could hide an ordering bug in the batch path."""
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")
    starts = [0, 3, 11, 20, 35]
    ends = [40, 20, 12, 21, 40]

    py = sv.var_ranges("chr1", starts, ends)
    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs = r.var_ranges(v_starts, v_ends, max_v_len, contig_start, list(zip(starts, ends)))

    assert rs.shape == py.shape
    for i in range(len(starts)):
        if (py[i] == SENTINEL).all():
            assert rs[i, 0] == rs[i, 1]
        else:
            np.testing.assert_array_equal(rs[i], py[i].astype(np.int64))


def test_find_ranges_matches_python_find_starts_ends(svar1_store: Path):
    """Stage B vs `_find_starts_ends`. Both are cartesian (r, s, p), so the shapes
    line up directly: Python's (2, r, s, p) -> starts = out[0].ravel()."""
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")
    starts = [0, 3, 35]
    ends = [40, 20, 40]

    # (2, r, s, p)
    py = sv._find_starts_ends("chr1", starts, ends, samples=None)

    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs_ranges = r.var_ranges(v_starts, v_ends, max_v_len, contig_start, list(zip(starts, ends)))
    d = r.find_ranges([(int(a), int(b)) for a, b in rs_ranges], None)

    np.testing.assert_array_equal(d["starts"], py[0].ravel())
    np.testing.assert_array_equal(d["stops"], py[1].ravel())
    assert d["n_ranges"] == len(starts)
    assert d["n_samples"] == len(sv.available_samples)
    assert d["ploidy"] == sv.ploidy


def test_find_ranges_sample_subset_matches_python(svar1_store: Path):
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")
    py = sv._find_starts_ends("chr1", [0], [40], samples=["S1"])

    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs_ranges = r.var_ranges(v_starts, v_ends, max_v_len, contig_start, [(0, 40)])
    d = r.find_ranges([(int(rs_ranges[0, 0]), int(rs_ranges[0, 1]))], [1])

    np.testing.assert_array_equal(d["starts"], py[0].ravel())
    np.testing.assert_array_equal(d["stops"], py[1].ravel())


def test_offsets_are_never_out_of_bounds(svar1_store: Path):
    """Guards convention gap 1 at the point it actually matters: every emitted
    offset must index into `variant_idxs`, including for empty rows. An
    out-of-range value overflows int64 in seqpro's Ragged.to_packed."""
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")
    n_entries = len(sv.genos.data)

    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs_ranges = r.var_ranges(v_starts, v_ends, max_v_len, contig_start, [(35, 40), (0, 40)])
    d = r.find_ranges([(int(a), int(b)) for a, b in rs_ranges], None)

    assert d["starts"].min() >= 0
    assert d["stops"].max() <= n_entries
    assert (d["stops"] >= d["starts"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run test tests/test_svar1_query_parity.py 2>&1 | tail -20`
Expected: FAIL — `ImportError: cannot import name 'PySvar1Reader'` until the extension is rebuilt. Rebuild first if needed:
Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e default maturin develop --release 2>&1 | tail -3`
Then re-run; it should now pass (the impl already exists from Tasks 1–4). **If any parity assertion fails, STOP** — that is a real Rust↔Python divergence. Per project policy, determine which side is wrong before "fixing" either: if the numba path is the buggy one, that is a separate genoray issue + PR, not something to paper over in the Rust wrapper.

- [ ] **Step 3: Run the whole pytest suite for regressions**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run test 2>&1 | tail -10`
Expected: no new failures.

- [ ] **Step 4: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
git add tests/test_svar1_query_parity.py
git commit -m "test(svar1): differential parity vs numba var_ranges/_find_starts_ends"
```

---

### Task 6: Compile guard — prove the module is htslib-free

**Files:**
- Modify: `tests/test_query_only_build.rs`

**Interfaces:**
- Consumes: `svar1_query::{Svar1Reader, var_ranges, find_ranges}` (Tasks 1–3).
- Produces: nothing (test-only).

**Background the implementer needs:**

This is **not optional decoration**. `test-rust` runs `cargo test --no-default-features --features conversion` — i.e. **`conversion` is always ON during cargo tests**. So a stray `use crate::svar1_reader::…` in `svar1_query` would compile green in the test suite and only break downstream in gvl (which links `default-features = false`). The gate is enforced by `check-core` plus this file. Referencing the symbols here forces them into the no-default-features build graph.

The existing file (entire contents):

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

- [ ] **Step 1: Add the failing guard**

Append to `tests/test_query_only_build.rs`:

```rust
/// The SVAR1 query core must ALSO build without `conversion`. This is the whole
/// point of `svar1_query` existing separately from the conversion-gated
/// `svar1_reader`: gvl links `genoray_core` with `default-features = false` and
/// must be able to query SVAR1 with no htslib.
///
/// NOTE: `test-rust` always runs with `conversion` ON, so this test passing under
/// `cargo test` proves nothing by itself — the gate is enforced by
/// `pixi run -e lint check-core` (`cargo check --no-default-features`), which is
/// what actually compiles this file without the feature.
#[test]
fn svar1_query_symbols_are_reachable_without_conversion() {
    use genoray_core::svar1_query::{Svar1Reader, find_ranges, var_ranges};
    let _ = Svar1Reader::open;
    let _ = var_ranges;
    let _ = find_ranges;
}

/// The PyO3 seam for SVAR1 queries must be ungated too (`py_query` already is).
#[test]
fn py_svar1_query_symbols_are_reachable_without_conversion() {
    use genoray_core::py_svar1_query::PySvar1Reader;
    let _ = PySvar1Reader::new;
}
```

- [ ] **Step 2: Run the guard under cargo test**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint test-rust 2>&1 | grep -E "query_only_build|test result" | head -5`
Expected: all 3 tests in the file PASS.

- [ ] **Step 3: Run the REAL gate**

Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint check-core 2>&1 | tail -5`
Expected: `Finished` with no errors. **This is the step that actually proves the gate.**

- [ ] **Step 4: Prove the guard has teeth (temporary negative check)**

Temporarily add `use crate::svar2_view::OverlapMode;` to the top of `src/svar1_query.rs`, then:
Run: `cd /carter/users/dlaub/projects/genoray && pixi run -e lint check-core 2>&1 | tail -5`
Expected: **FAILS** with an unresolved-import error (`svar2_view` is conversion-gated). This confirms `check-core` catches gate violations. **Now remove the line** and re-run to confirm it passes again.

- [ ] **Step 5: Commit**

```bash
cd /carter/users/dlaub/projects/genoray
git add tests/test_query_only_build.rs
git commit -m "test(svar1): compile-guard svar1_query under --no-default-features"
```

---

### Task 7: Open the PR

**Files:** none (git/gh only).

- [ ] **Step 1: Run the full verification suite**

```bash
cd /carter/users/dlaub/projects/genoray
pixi run -e lint test-rust 2>&1 | tail -5
pixi run -e lint check-core 2>&1 | tail -3
pixi run test 2>&1 | tail -5
pixi run typecheck 2>&1 | tail -3
```
Expected: all green. Do not proceed otherwise.

- [ ] **Step 2: Ensure prek hooks are installed, then push**

```bash
cd /carter/users/dlaub/projects/genoray
pixi run prek-install
git push -u origin HEAD
```

- [ ] **Step 3: Open the PR**

```bash
cd /carter/users/dlaub/projects/genoray
gh pr create --repo d-laub/genoray --title "feat(svar1): ungated Rust range-query API (Svar1Reader + var_ranges + find_ranges)" --body "Closes #123.

Adds an ungated \`svar1_query\` module: the **query** counterpart to the conversion-gated \`svar1_reader::Svar1RecordSource\` (which is a *conversion-pipeline record producer* — forward-only, O(all CSR entries) at construction — and the wrong tool for range queries).

Mostly assembles parts that already existed:
- **Stage A** (\`var_ranges\`) is a thin wrapper over \`search::overlap_range\`, which already ported the Python algorithm; nobody had wired it to SVAR1.
- **Stage B** (\`find_ranges\`) is two \`partition_point\`s per hap — the idiom already at \`svar1_reader.rs:30-31\`.
- No \`gather_ranges\`: SVAR1's on-disk layout is already the target representation, so consumers build a zero-copy view straight from the index pairs.

**Ungated** (memmap2 + bytemuck + \`search.rs\`; zero htslib/zstd), so gvl can query SVAR1 with \`default-features = false\`. Enforced by \`check-core\` + a compile guard in \`tests/test_query_only_build.rs\` — necessary because \`test-rust\` always runs with \`conversion\` ON.

**New differential tests vs the numba path.** \`search.rs\` claimed to mirror \`var_ranges\` but nothing tested it; \`tests/test_svar1_query_parity.py\` now does. Two conventions are deliberately preserved rather than \"fixed\":
- Python signals no-overlap with an \`INT32_MAX\` sentinel; Rust returns an **in-bounds zero-length** range. Rust's is required — an out-of-range offset overflows int64 in seqpro's \`Ragged.to_packed\` even for an empty row.
- Python's \`max_v_len\` is 1 larger than \`overlap_range\`'s \`>=\` contract. That's an **over**-estimate, which is provably overshoot-safe.

**Consumer:** mcvickerlab/GenVarLoader#275 picks this up via a \`rev\` bump once merged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 4: Report the PR URL** so Plan 2b can be unblocked once it merges.

---

## Notes for the reviewer

- **`svar1_reader` is untouched.** The conversion pipeline keeps its record producer; this adds a parallel query path. No behavior change to existing code.
- **The only `lib.rs` edits** are two `pub mod` lines and one `add_class` line, all ungated.
- **Scope deliberately excluded:** `_find_starts_ends_with_length` (a much larger port — `_length_walk_n_keep`, biallelic-only), and migrating genoray's Python off numba. This makes the latter possible later; it does not attempt it.
