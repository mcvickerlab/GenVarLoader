# Streaming VCF + PGEN Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add write-free `StreamingDataset` VCF and PGEN variant backends that reach byte-identical haplotype parity with `gvl.write()` + `Dataset[r, s]`, by decoding through genoray's Rust `RecordSource → ChunkAssembler → DenseChunk` pipeline behind one shared producer/consumer engine.

**Architecture:** genoray_core's conversion-gated readers produce a normalized, atomized `DenseChunk` per window (the window-local static variant table + dense `BitGrid3` genotypes). A new gvl transpose turns that into the sparse `geno_v_idxs` + CSR the existing reconstruction kernel consumes. A new `RecordStreamEngine` pyclass — mirroring `Svar1StreamEngine`'s detached-producer/`next_batch`/`py.detach` shape, but carrying an owned decoded window (local table in the slot) — runs the decode on the producer thread so it overlaps reconstruction on the consumer. VCF and PGEN differ only in a `WindowFiller` implementation.

**Tech Stack:** Rust (PyO3, ndarray, crossbeam-channel, rayon), genoray_core (`conversion` feature: rust-htslib, zstd, pgenlib-via-PyO3), Python (numpy, seqpro Ragged), pytest, cargo test, vcfixture (Python oracle + Rust `bulk` CLI), plink2, pixi.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-17-streaming-vcf-pgen-backends-design.md` — this plan implements it. Read it first.
- **Branch:** all work targets the long-lived `streaming` integration branch, **not** `main` (CLAUDE.md → Streaming dataset work). This worktree is on `spec/276-vcf-pgen` (off `streaming`).
- **Scope:** haplotypes-only output; `jitter=0`; `num_workers <= 1`. Output-mode breadth (`reference`/`annotated`/`variants`, `with_len`, `with_settings`, `min_af`/`max_af`, `var_fields`, jitter) is **#277 — do not implement here**.
- **Parity is the oracle:** byte-identical vs `gvl.write()` + `Dataset.open()[r, s]`, `jitter=0`. The new risk is genoray's **Rust** decoder (`ChunkAssembler`) diverging from the write path's **Python** cyvcf2/pgenlib + `dense2sparse`; pin it at the variant-table layer *before* reconstruction (Task 8 / Task 14).
- **Rebuild Rust before Python tests:** `pixi run -e dev maturin develop --release` after any `src/` change, or pytest imports the stale extension (CLAUDE.md). `cargo test` compiles from source and is unaffected.
- **cargo test needs libpython on the load path:** `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib` (memory: cargo-test-libpython-ldpath).
- **conversion feature stays enabled** (`Cargo.toml:27`, already present); this plan makes it load-bearing. Do not remove it.
- **Commit style:** conventional commits (commitizen gate). End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **prek hooks** are installed in this worktree; they run ruff + pyrefly + commitizen on commit.
- **Do not squash** when landing (memory: no-squash-merges).
- **Full tree before pushing** a change that renames/removes a public symbol or touches shared code: `pixi run -e dev pytest tests -q` (scoped runs skip `tests/unit/`).

**Reference templates (read before implementing):**
- Engine to mirror: `src/ffi/stream_engine.rs` (`Svar1StreamEngine` — threading, shutdown/panic discipline, `next_batch_core`, `generate_from_current`).
- Reconstruction wrapper to generalize: `src/ffi/mod.rs:980-1081` (`generate_batch_core`).
- SVAR1 store shape (for the pyclass pattern): `src/svar1/store.rs`, `src/svar1/mod.rs`.
- Python backend to mirror: `python/genvarloader/_dataset/_streaming.py:566-896` (`_Svar1Backend`) and the classification ladder at `:177-206`.
- genoray_core (rev `e07477e`, in `~/.cargo/git/checkouts/genoray-26e7da4241f8ed6f/e07477e/`): `record_source.rs` (`RecordSource`, `RawRecord`), `vcf_reader.rs` (`VcfRecordSource`), `pgen_reader.rs`/`pvar.rs` (`PgenRecordSource`/`PvarReader`), `chunk_assembler.rs` (`ChunkAssembler`, `read_next_chunk`), `types.rs` (`DenseChunk`, `BitGrid3`), `rvk.rs` (the `dense2sparse_vk` transpose to adapt).

---

# PR 1 — Shared engine + transpose + VCF backend

Lands the reusable core (Tasks 1–3) then the VCF backend on top (Tasks 4–9). Tasks 1–3 are the critical path; 4–9 build on them.

---

## Task 1: Decouple `generate_batch_core` from `Svar1Store`

Make the reconstruction wrapper reusable by VCF/PGEN, whose sparse genotypes live in a window-local buffer, not a store mmap. Pure refactor — SVAR1 behavior and all existing tests unchanged.

**Files:**
- Modify: `src/ffi/mod.rs:980-1081` (`generate_batch_core` signature + body)
- Modify: `src/ffi/mod.rs:1129-1144` (`svar1_generate_batch` caller)
- Modify: `src/ffi/stream_engine.rs:426-439` (`generate_from_current` caller)

**Interfaces:**
- Produces: `pub(crate) fn generate_batch_core(geno_v_idxs: &[i32], ploidy: usize, o_starts_b: &[i64], o_stops_b: &[i64], region_bounds_b: ArrayView2<i32>, v_starts: ArrayView1<i32>, ilens: ArrayView1<i32>, alt_alleles: ArrayView1<u8>, alt_offsets: ArrayView1<i64>, ref_: ArrayView1<u8>, ref_offsets: ArrayView1<i64>, pad_char: u8, parallel: bool) -> (Array1<u8>, Array1<i64>)` — the `store: &Svar1Store` first param is replaced by `geno_v_idxs: &[i32]` + `ploidy: usize`.

- [ ] **Step 1: Change the signature.** In `src/ffi/mod.rs`, replace the first parameter of `generate_batch_core`:

```rust
pub(crate) fn generate_batch_core(
    geno_v_idxs: &[i32],
    ploidy: usize,
    o_starts_b: &[i64],
    o_stops_b: &[i64],
    region_bounds_b: ndarray::ArrayView2<i32>,
    // ... rest unchanged ...
```

- [ ] **Step 2: Update the body.** Delete the two lines that derived these from `store` and use the params directly:

```rust
    // was: let ploidy = store.ploidy();
    // (ploidy is now a parameter)
    let n_work = batch * ploidy;
    // ...
    // was: let geno_v_idxs = store.geno_v_idxs();
    let geno_v_idxs_view = numpy::ndarray::ArrayView1::from(geno_v_idxs);
```

Remove the now-unused `use crate::svar1::store::...` if the function no longer references `Svar1Store`.

- [ ] **Step 3: Update the FFI caller.** In `svar1_generate_batch` (`src/ffi/mod.rs:1129`), pass the store's mmap + ploidy:

```rust
    let geno_v_idxs_s = store_ref.geno_v_idxs();
    let ploidy = store_ref.ploidy();
    let (out_data, out_offsets_vec) = py.detach(move || {
        generate_batch_core(
            geno_v_idxs_s,
            ploidy,
            o_starts_s,
            o_stops_s,
            rb,
            // ... rest unchanged ...
        )
    });
```

- [ ] **Step 4: Update the engine caller.** In `src/ffi/stream_engine.rs:426`, replace `&self.store` with the mmap + ploidy (ploidy is already bound at `generate_from_current:387`):

```rust
        Ok(crate::ffi::generate_batch_core(
            self.store.geno_v_idxs(),
            ploidy,
            o_starts_b,
            o_stops_b,
            // ... rest unchanged ...
        ))
```

Also update the test helper callers in `stream_engine.rs` (`expected_window:677`, the `direct` closure `:1020`) and any `generate_batch_core` call in `src/svar1/` tests: replace `&store` / `&s` with `store.geno_v_idxs()` / `s.geno_v_idxs()` and add the `ploidy` arg (2 in those fixtures).

- [ ] **Step 5: Build and run the full Rust suite to prove the refactor is behavior-preserving.**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test`
Expected: PASS — every existing `stream_engine` and `svar1` test still green (byte-identical output; only the call shape changed).

- [ ] **Step 6: Commit.**

```bash
git add src/ffi/mod.rs src/ffi/stream_engine.rs
git commit -m "refactor(streaming): generate_batch_core takes geno_v_idxs + ploidy, not Svar1Store

Decouples the reconstruction wrapper from SVAR1's store mmap so VCF/PGEN
backends can pass a window-local sparse buffer. SVAR1 callers pass
store.geno_v_idxs()/ploidy(); behavior byte-identical.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `DecodedWindow` buffer + `DenseChunk → geno_v_idxs` transpose

The window-local slot for VCF/PGEN, and the one piece of genuinely new decode logic: turn genoray's `DenseChunk` (variant-major static table + dense `BitGrid3` genotypes) into the hap-major sparse `geno_v_idxs` + CSR offsets the kernel wants. Adapted from genoray's `rvk::dense2sparse_vk` transpose (`rvk.rs:340-400`), but emitting the **variant column index** instead of a self-describing key.

**Files:**
- Create: `src/record_stream/mod.rs` (module root; declare `pub mod transpose;`)
- Create: `src/record_stream/transpose.rs` (`DecodedWindow`, `fill_decoded_window`)
- Modify: `src/lib.rs` (add `mod record_stream;` near the other `mod` declarations, ~line 12-18 area)

**Interfaces:**
- Produces: `#[derive(Default)] pub struct DecodedWindow { pub v_starts: Vec<i32>, pub ilens: Vec<i32>, pub alt_alleles: Vec<u8>, pub alt_offsets: Vec<i64>, pub geno_v_idxs: Vec<i32>, pub geno_offsets: Vec<i64>, pub job_idx: usize }`
- Produces: `pub fn fill_decoded_window(chunk: &genoray_core::types::DenseChunk, n_samples: usize, ploidy: usize, slot: &mut DecodedWindow)` — clears and refills `slot`'s vecs from `chunk`. `geno_offsets` is CSR of length `n_samples * ploidy + 1`; `geno_v_idxs[geno_offsets[h]..geno_offsets[h+1]]` are the ascending variant column-indices (into `slot.v_starts` etc.) carried by hap `h` (C-order `hap = s*ploidy + p`).

- [ ] **Step 1: Write the failing test.** In `src/record_stream/transpose.rs`, a `#[cfg(test)]` module. Build a `DenseChunk` by hand (its fields are public in `genoray_core::types`) and assert the transpose. Fixture: 2 variants (SNP @10 alt 'A' ilen 0; SNP @20 alt 'C' ilen 0), 2 samples × ploidy 2 = 4 haps, presence grid: hap0→{v0}, hap1→{}, hap2→{v0,v1}, hap3→{v1}.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use genoray_core::types::{BitGrid3, DenseChunk};

    fn dense_fixture() -> DenseChunk {
        // 2 variants, 2 samples, ploidy 2. BitGrid3 dims (V, S, P).
        let mut genos = BitGrid3::zeros(2, 2, 2);
        genos.or_bit(0, 0, 0); // v0, sample0, ploid0  -> hap0
        genos.or_bit(0, 1, 0); // v0, sample1, ploid0  -> hap2
        genos.or_bit(1, 1, 0); // v1, sample1, ploid0  -> hap2
        genos.or_bit(1, 1, 1); // v1, sample1, ploid1  -> hap3
        DenseChunk {
            chunk_id: 0,
            pos: vec![10, 20],
            ilens: vec![0, 0],
            alt: vec![b'A', b'C'],
            alt_offsets: vec![0, 1, 2],
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
        }
    }

    #[test]
    fn transpose_emits_variant_indices_hap_major() {
        let chunk = dense_fixture();
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, &mut slot);

        assert_eq!(slot.v_starts, vec![10, 20]);
        assert_eq!(slot.ilens, vec![0, 0]);
        assert_eq!(slot.alt_alleles, vec![b'A', b'C']);
        assert_eq!(slot.alt_offsets, vec![0, 1, 2]);
        // CSR over 4 haps: hap0=[0], hap1=[], hap2=[0,1], hap3=[1]
        assert_eq!(slot.geno_offsets, vec![0, 1, 1, 3, 4]);
        assert_eq!(slot.geno_v_idxs, vec![0, 0, 1, 1]);
    }
}
```

Verify against genoray's `BitGrid3` API before running: confirm the constructor is `BitGrid3::zeros(v, s, p)` and the setter `or_bit(v, s, p)` (from `types.rs:36`); if the names differ, adjust the test to the real API (the transpose reads bits via whatever getter `dense2sparse_vk` uses at `rvk.rs:340-400`, e.g. `genos.get_bit(v, s, p)`).

- [ ] **Step 2: Run it to see it fail.**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test transpose_emits_variant_indices_hap_major`
Expected: FAIL — `DecodedWindow` / `fill_decoded_window` not found.

- [ ] **Step 3: Implement `DecodedWindow` + `fill_decoded_window`.** The transpose loop order is hap-major (outer `s`, `p`; inner `v`), copied from `rvk.rs:340-400` but pushing `v` instead of a key. Static-table fields copy straight from `DenseChunk` (cast `u32→i32` / `u32→i64` for offsets):

```rust
use genoray_core::types::DenseChunk;

#[derive(Default)]
pub struct DecodedWindow {
    pub v_starts: Vec<i32>,
    pub ilens: Vec<i32>,
    pub alt_alleles: Vec<u8>,
    pub alt_offsets: Vec<i64>,
    pub geno_v_idxs: Vec<i32>,
    pub geno_offsets: Vec<i64>,
    pub job_idx: usize,
}

/// Fill `slot` (reusing its allocations) from a window's `DenseChunk`. The static table
/// copies straight across; the genotype transpose walks haps in C-order and pushes each
/// carried variant's COLUMN INDEX (into the static table), building a per-hap CSR.
pub fn fill_decoded_window(
    chunk: &DenseChunk,
    n_samples: usize,
    ploidy: usize,
    slot: &mut DecodedWindow,
) {
    let n_var = chunk.pos.len();

    slot.v_starts.clear();
    slot.v_starts.extend(chunk.pos.iter().map(|&p| p as i32));
    slot.ilens.clear();
    slot.ilens.extend_from_slice(&chunk.ilens);
    slot.alt_alleles.clear();
    slot.alt_alleles.extend_from_slice(&chunk.alt);
    slot.alt_offsets.clear();
    slot.alt_offsets.extend(chunk.alt_offsets.iter().map(|&o| o as i64));

    slot.geno_v_idxs.clear();
    slot.geno_offsets.clear();
    slot.geno_offsets.push(0);
    // Hap-major: for each (sample, ploid) hap, scan variants in ascending column order
    // and push any present variant's index. Ascending v => geno_v_idxs stay sorted per
    // hap, matching the SVAR1 CSR contract the kernel expects.
    for s in 0..n_samples {
        for p in 0..ploidy {
            for v in 0..n_var {
                if chunk.genos.get_bit(v, s, p) {
                    slot.geno_v_idxs.push(v as i32);
                }
            }
            slot.geno_offsets.push(slot.geno_v_idxs.len() as i64);
        }
    }
}
```

Match the real `BitGrid3` bit-getter name/signature from `types.rs` / `rvk.rs`. Create `src/record_stream/mod.rs` with `pub mod transpose;` and re-export `pub use transpose::{DecodedWindow, fill_decoded_window};`. Add `mod record_stream;` to `src/lib.rs`.

- [ ] **Step 4: Run the test to see it pass.**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test transpose_emits_variant_indices_hap_major`
Expected: PASS.

- [ ] **Step 5: Add a degenerate-case test** (empty chunk → empty table, `geno_offsets = [0; n_haps+1]`) and a monomorphic-hap test (all bits set on one variant). Append to the same test module and re-run.

```rust
    #[test]
    fn transpose_empty_chunk_is_all_zero_csr() {
        let genos = BitGrid3::zeros(0, 2, 2);
        let chunk = DenseChunk { chunk_id: 0, pos: vec![], ilens: vec![], alt: vec![],
            alt_offsets: vec![0], genos, info_staged: vec![], format_staged: vec![] };
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, &mut slot);
        assert!(slot.geno_v_idxs.is_empty());
        assert_eq!(slot.geno_offsets, vec![0, 0, 0, 0, 0]); // 4 haps, all empty
    }
```

- [ ] **Step 6: Commit.**

```bash
git add src/record_stream/ src/lib.rs
git commit -m "feat(streaming): DecodedWindow buffer + DenseChunk->geno_v_idxs transpose

New window-local slot for VCF/PGEN and the hap-major transpose that turns
genoray's DenseChunk into the sparse variant-index CSR the reconstruction
kernel consumes. Adapted from genoray rvk::dense2sparse_vk, emitting the
variant column index instead of a self-describing key.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `RecordStreamEngine` pyclass (producer/consumer over a `WindowFiller`)

Mirror `Svar1StreamEngine` exactly for the threading/shutdown/`next_batch`/`py.detach` structure, but: the slot is a `DecodedWindow` (carries the local static table), the producer fills it via a `Box<dyn WindowFiller + Send>` (the decode), and `generate_from_current` reads the table from the slot and calls the generalized `generate_batch_core`. The reference is still per-contig.

**Files:**
- Create: `src/record_stream/engine.rs` (`RecordStreamEngine`, `WindowFiller` trait, `RecordJob`, `ContigRef`)
- Modify: `src/record_stream/mod.rs` (`pub mod engine;`)

**Interfaces:**
- Produces: `pub trait WindowFiller: Send { fn fill(&self, job: &RecordJob, contig: &ContigRef, slot: &mut DecodedWindow) -> anyhow::Result<()>; }` — decode a window into `slot` (fills the static table + `geno_v_idxs`/`geno_offsets`; leaves `job_idx` to the engine).
- Produces: `pub struct RecordJob { pub contig_idx: usize, pub regions: Vec<(u32, u32)>, pub s_lo: usize, pub s_hi: usize }` and `pub struct ContigRef { pub name: String, pub ref_bytes: Vec<u8> }`.
- Produces: `RecordStreamEngine::new_rs(filler: Box<dyn WindowFiller + Send>, contigs: Vec<ContigRef>, jobs: Vec<RecordJob>, n_samples: usize, ploidy: usize, pad_char: u8, parallel: bool, batch_size: usize) -> Self` (Rust-facing ctor for tests) and `fn next_batch_core(&self) -> Option<anyhow::Result<(Array1<u8>, Array1<i64>)>>`.
- Consumes: `crate::record_stream::{DecodedWindow, fill_decoded_window}` (Task 2), `crate::ffi::generate_batch_core` (Task 1).

- [ ] **Step 1: Write the failing test.** A `#[cfg(test)]` module with a hand-rolled `WindowFiller` that ignores the source and fills a fixed `DecodedWindow` per job (so the engine is tested in isolation from genoray). Assert a ≥2-job plan flows through, batches equal a direct `fill + generate_batch_core`, and exhaustion returns `None` cleanly. This is the structural analog of `svar1_stream_engine_yields_windows_in_plan_order` — copy that test's shape from `stream_engine.rs:715-757`, substituting the `DecodedWindow` filler. Include the empty-plan (`..._empty_plan_is_none`), producer-error (`Ok(Err)` not EOF), producer-panic (`Err` join branch), and drop-midstream tests, each adapted from `stream_engine.rs`.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::record_stream::DecodedWindow;
    use ndarray::{Array1, Array2};

    /// Filler that decodes each window to a fixed 2-variant table with a per-job
    /// genotype pattern, so the engine's plumbing (not genoray) is under test.
    struct StubFiller;
    impl WindowFiller for StubFiller {
        fn fill(&self, job: &RecordJob, _c: &ContigRef, slot: &mut DecodedWindow) -> anyhow::Result<()> {
            let n_haps = (job.s_hi - job.s_lo) * 2;
            slot.v_starts = vec![10, 20];
            slot.ilens = vec![0, 0];
            slot.alt_alleles = vec![b'A', b'C'];
            slot.alt_offsets = vec![0, 1, 2];
            // every hap carries v0 only
            slot.geno_v_idxs = vec![0; n_haps];
            slot.geno_offsets = (0..=n_haps as i64).collect();
            Ok(())
        }
    }
    // ... build engine with StubFiller, one ContigRef {name:"chr1", ref_bytes: vec![b'T';30]},
    //     jobs = [ {0, [(0,30)], 0, 2}, {0, [(0,15)], 0, 2} ], batch_size huge;
    //     drain next_batch_core(); assert 2 batches, each == direct fill+generate.
}
```

- [ ] **Step 2: Run it to see it fail.**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test record_stream_engine`
Expected: FAIL — `RecordStreamEngine` not found.

- [ ] **Step 3: Implement the engine.** Copy `src/ffi/stream_engine.rs` wholesale into `src/record_stream/engine.rs` and apply these substitutions (keep the module doc comment's threading/shutdown rationale — it is still exactly correct):
  - Slot type `FilledWindow` → `DecodedWindow` (Task 2); drop the `o_starts`/`o_stops`-only fields — `DecodedWindow` already has them plus the table.
  - `ContigData` → `ContigRef` (`name`, `ref_bytes` only — the per-contig `v_starts_c`/`v_ends_c`/`contig_start`/`n_local`/`max_v_len` were SVAR1 store-meta and are gone; the local table now comes from the decode).
  - `WindowJob` → `RecordJob` (same fields).
  - The producer loop body: replace `store.read_window(...)` + `prefetch_runs_core(...)` with:

```rust
    let c = &contigs[job.contig_idx];
    filler.fill(job, c, &mut slot)?;   // decode this window into the slot
    slot.job_idx = job_idx;
```

  - Engine fields: drop `store`, `v_starts`/`ilens`/`alt_alleles`/`alt_offsets` (now per-window in the slot), `phys_sample_idx` (sample selection is the filler's job — see Task 4/10). Keep `n_samples`, `ploidy`, `pad_char`, `parallel`, `batch_size`, `contigs`, `jobs`, `state`. Add `filler: Arc<Box<dyn WindowFiller + Send + Sync>>` — or, since only the producer thread uses it, move it into the thread at `ensure_started` (like `store`/`jobs`). Prefer moving a `Box<dyn WindowFiller + Send>` into the producer via an `Option` taken in `ensure_started`, avoiding the `Sync` bound.
  - `generate_from_current`: read the table from the current slot, and pass the slot's `geno_v_idxs` to the generalized `generate_batch_core`:

```rust
    let filled = &state.current.as_ref().unwrap().filled; // a DecodedWindow
    // ... compute row_lo/row_hi/rb as in the SVAR1 version; o_lo = row_lo*ploidy,
    //     o_hi = row_hi*ploidy (the hap range for this batch slice) ...
    // Build this batch's per-hap CSR start/stop from the window's geno_offsets:
    let o_starts_b: Vec<i64> = filled.geno_offsets[o_lo..o_hi].to_vec();
    let o_stops_b: Vec<i64> = filled.geno_offsets[o_lo + 1..=o_hi].to_vec();
    let ref_offsets = Array1::from(vec![0i64, c.ref_bytes.len() as i64]);
    Ok(crate::ffi::generate_batch_core(
        &filled.geno_v_idxs,            // window-local sparse (Task 1 signature)
        self.ploidy,
        &o_starts_b,
        &o_stops_b,
        // ...
        ndarray::ArrayView1::from(filled.v_starts.as_slice()),
        ndarray::ArrayView1::from(filled.ilens.as_slice()),
        ndarray::ArrayView1::from(filled.alt_alleles.as_slice()),
        ndarray::ArrayView1::from(filled.alt_offsets.as_slice()),
        ndarray::ArrayView1::from(c.ref_bytes.as_slice()),
        ref_offsets.view(),
        self.pad_char,
        self.parallel,
    ))
```

  **CSR-offset note:** SVAR1's `o_starts`/`o_stops` are absolute indices into the mmap, sliced `[o_lo..o_hi]`. For `DecodedWindow` the per-hap CSR is `geno_offsets` (length `n_haps+1`). The batch's `o_starts_b`/`o_stops_b` for haps `[o_lo..o_hi)` are `geno_offsets[o_lo..o_hi]` (starts) and `geno_offsets[o_lo+1..=o_hi]` (stops) — build these two `Vec<i64>` slices in `generate_from_current`. `geno_v_idxs` is the window-local variant-index array (not a global mmap), so it is the correct `geno_v_idxs` arg to `generate_batch_core` and the CSR offsets index into it directly.
  - Delete SVAR1-store-specific tests; keep the adapted structural/error/panic/drop tests from Step 1.

- [ ] **Step 4: Run the engine tests to see them pass.**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test record_stream_engine`
Expected: PASS — all structural, empty-plan, error, panic, and drop tests green.

- [ ] **Step 5: Commit.**

```bash
git add src/record_stream/engine.rs src/record_stream/mod.rs
git commit -m "feat(streaming): RecordStreamEngine — producer/consumer over a WindowFiller

Mirrors Svar1StreamEngine's detached-producer/next_batch/py.detach shape and
its exact shutdown/panic discipline, but carries an owned decoded window
(local static table in the slot) and fills it via a WindowFiller (the decode).
VCF and PGEN plug in as fillers. Reconstruction reuses generate_batch_core.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `VcfWindowFiller` — decode a VCF window into a `DecodedWindow`

Wire genoray's `VcfRecordSource → ChunkAssembler` for one window and run the Task 2 transpose. This is the VCF-specific decode.

**Files:**
- Create: `src/record_stream/vcf.rs` (`VcfWindowFiller`)
- Modify: `src/record_stream/mod.rs` (`pub mod vcf;`)

**Interfaces:**
- Produces: `pub struct VcfWindowFiller { /* vcf_path, sample names, ploidy, fields, overlap, per-contig fasta handle or ref cache */ }` implementing `WindowFiller`.
- Consumes: `genoray_core::{vcf_reader::VcfRecordSource, chunk_assembler::ChunkAssembler, normalize::CheckRef}`, `crate::record_stream::{DecodedWindow, fill_decoded_window, WindowFiller, RecordJob, ContigRef}`.

- [ ] **Step 1: Write the failing test.** Generate a tiny VCF with the Python `vcfixture` package (call it via a `std::process` or, simpler, commit a small fixture `.vcf.gz` + `.tbi` under `tests/data/streaming/`). Assert `VcfWindowFiller::fill` on a known window yields a `DecodedWindow` whose `v_starts`/`ilens`/`alt_alleles` match the fixture's variants and whose `geno_v_idxs` match the known genotypes. Use a fixture with 1 SNP + 1 deletion across 2 samples so ILEN and atomization are exercised.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // fixture path: tests/data/streaming/two_var_two_sample.vcf.gz (committed, with .tbi)
    #[test]
    fn vcf_filler_decodes_window_to_local_table() {
        let filler = VcfWindowFiller::new(
            "tests/data/streaming/two_var_two_sample.vcf.gz",
            &["s1", "s2"], 2, /*fasta*/ None,
        ).unwrap();
        let job = RecordJob { contig_idx: 0, regions: vec![(0, 100)], s_lo: 0, s_hi: 2 };
        let contig = ContigRef { name: "chr1".into(), ref_bytes: vec![b'A'; 100] };
        let mut slot = DecodedWindow::default();
        filler.fill(&job, &contig, &mut slot).unwrap();
        assert_eq!(slot.v_starts, vec![/* known 0-based POS after atomization */]);
        assert_eq!(slot.ilens, vec![/* 0 for SNP, negative for DEL */]);
        // ... geno_v_idxs / geno_offsets per the known genotypes ...
    }
}
```

Compute the exact expected values from the fixture once, then hardcode them.

- [ ] **Step 2: Run it to see it fail.**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test vcf_filler_decodes_window`
Expected: FAIL — `VcfWindowFiller` not found.

- [ ] **Step 3: Implement `VcfWindowFiller`.** `fill` constructs a `VcfRecordSource` for the job's contig+regions+samples, wraps it in a `ChunkAssembler`, reads all chunks for the window, and transposes each into the slot (accumulating variants across chunks, since a window may span multiple assembler chunks). Match the write path's normalization config exactly — pass the reference FASTA iff `gvl.write` did (so left-align matches), and the same `CheckRef` mode. Verify these against `_write.py:747-760` and genoray's `ChunkAssembler::new` defaults.

```rust
use genoray_core::chunk_assembler::ChunkAssembler;
use genoray_core::vcf_reader::VcfRecordSource;
use genoray_core::normalize::CheckRef;

impl WindowFiller for VcfWindowFiller {
    fn fill(&self, job: &RecordJob, contig: &ContigRef, slot: &mut DecodedWindow) -> anyhow::Result<()> {
        let sample_refs: Vec<&str> = self.samples.iter().map(String::as_str).collect();
        let source = VcfRecordSource::new(
            &self.vcf_path, &contig.name, &sample_refs,
            /*htslib_threads*/ 1, self.ploidy, &self.fields,
            job.regions.clone(), self.overlap,
        )?;
        let mut asm = ChunkAssembler::new(
            Box::new(source), self.samples.len(), self.ploidy,
            self.fasta_path.as_deref(), &contig.name,
            /*skip_out_of_scope*/ true, self.check_ref, &self.fields,
        )?;
        // Accumulate the whole window (all chunks) into one local table, then transpose.
        // If a window fits one chunk (chunk_size >= window variants), one read suffices.
        let mut acc: Option<genoray_core::types::DenseChunk> = None;
        while let Some(chunk) = asm.read_next_chunk(self.chunk_size, 0, None)? {
            acc = Some(match acc { None => chunk, Some(prev) => concat_dense(prev, chunk) });
        }
        match acc {
            Some(chunk) => fill_decoded_window(&chunk, self.samples.len(), self.ploidy, slot),
            None => fill_decoded_window(&empty_dense(self.ploidy, self.samples.len()), self.samples.len(), self.ploidy, slot),
        }
        Ok(())
    }
}
```

Implement `concat_dense` (append two `DenseChunk`s — concatenate `pos`/`ilens`/`alt`(+offsets) and stack the `BitGrid3` along V) and `empty_dense`. **Simplify if possible:** size `chunk_size` so a window is always one chunk (window variant count is bounded by `window_regions × density`); then `read_next_chunk` is called once and `concat_dense` is unnecessary. Prefer that — set `chunk_size` large and assert a single chunk, documenting the invariant. Confirm `VcfRecordSource::new` / `ChunkAssembler::new` / `read_next_chunk` argument order against `vcf_reader.rs:270`, `chunk_assembler.rs:371`, `:579`.

- [ ] **Step 4: Run the test to see it pass.**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test vcf_filler_decodes_window`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/record_stream/vcf.rs src/record_stream/mod.rs tests/data/streaming/
git commit -m "feat(streaming): VcfWindowFiller decodes a VCF window to a DecodedWindow

Drives genoray VcfRecordSource -> ChunkAssembler for one window and transposes
the DenseChunk into the local sparse buffer. Normalization config matches the
write path (left-align iff FASTA, same check-ref) for parity.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `RecordStreamEngine` VCF constructor + FFI registration

Expose a Python `#[new]` (or `from_vcf` classmethod) that builds the engine with a `VcfWindowFiller`, and register the pyclass.

**Files:**
- Modify: `src/record_stream/engine.rs` (add `#[pymethods]` block: `#[new]` for VCF + `next_batch`)
- Modify: `src/lib.rs` (`m.add_class::<record_stream::engine::RecordStreamEngine>()?;` near `:27`)

**Interfaces:**
- Produces (Python-visible): `RecordStreamEngine(source_kind: str, vcf_path: str, sample_names: list[str], ploidy: int, contig_names: list[str], contig_ref_bytes: list[bytes], job_contig_idx: list[int], job_region_starts: list[list[int]], job_region_ends: list[list[int]], job_s_lo: list[int], job_s_hi: list[int], fasta_path: str | None, pad_char: int, parallel: bool, batch_size: int)` and `.next_batch() -> tuple[np.ndarray, np.ndarray] | None`.
- Consumes: `VcfWindowFiller` (Task 4), the `new_rs` core (Task 3).

- [ ] **Step 1: Write the failing test (Python).** In `tests/dataset/test_streaming_vcf.py`, drive the engine directly for a committed fixture and assert `next_batch` yields batches until `None`. (End-to-end parity is Task 8; this checks the FFI seam.)

```python
def test_record_stream_engine_vcf_yields_then_none(streaming_vcf_fixture):
    from genvarloader._genvarloader import RecordStreamEngine
    eng = RecordStreamEngine(
        "vcf", str(streaming_vcf_fixture.vcf), ["s1", "s2"], 2,
        ["chr1"], [streaming_vcf_fixture.chr1_ref_bytes],
        [0], [[0]], [[100]], [0], [2],
        str(streaming_vcf_fixture.fasta), ord("N"), False, 32,
    )
    batches = []
    while (b := eng.next_batch()) is not None:
        batches.append(b)
    assert len(batches) >= 1
    data, offsets = batches[0]
    assert data.dtype == np.uint8 and offsets.dtype == np.int64
```

- [ ] **Step 2: Build and run to see it fail.**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_vcf.py::test_record_stream_engine_vcf_yields_then_none -v`
Expected: FAIL — `RecordStreamEngine` has no `#[new]` yet / import error.

- [ ] **Step 3: Implement the `#[pymethods]`.** Mirror `Svar1StreamEngine`'s `#[new]` validation + `next_batch` (`stream_engine.rs:443-599`). The constructor validates the parallel per-contig/per-job arrays (reuse the length-check pattern verbatim), builds `Vec<ContigRef>` + `Vec<RecordJob>`, constructs the `VcfWindowFiller`, and calls `new_rs`. `next_batch` is identical to SVAR1's (`py.detach(|| self.next_batch_core())` + `into_pyarray`). Branch on `source_kind == "vcf"`; `"pgen"` returns `PyNotImplementedError` (filled in Task 10).

- [ ] **Step 4: Rebuild and run to see it pass.**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_vcf.py::test_record_stream_engine_vcf_yields_then_none -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/record_stream/engine.rs src/lib.rs tests/dataset/test_streaming_vcf.py
git commit -m "feat(streaming): RecordStreamEngine Python constructor (VCF) + registration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `_VcfBackend` (Python) + classification-ladder wiring

The Python backend duck-typed to what `StreamingDataset._iter_batches` consumes (engine strategy only), wired into the `variants=` classification ladder.

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (add `_VcfBackend`; wire the VCF branch at `:197-201`; add `_backend: _Svar1Backend | _VcfBackend | None` to the field type at `:113`)

**Interfaces:**
- Consumes: `RecordStreamEngine` (Task 5); `Reference.from_path` (`_streaming.py:592`); `genoray.VCF` for header metadata (`n_samples`, `ploidy`, sample names, contigs).
- Produces: `_VcfBackend` exposing `n_samples: int`, `ploidy: int`, `_sample_names: list[str]`, `build_engine(jobs, batch_size) -> RecordStreamEngine`. (No `_store`/`read_window`/`generate_batch` — VCF supports only the `engine` prefetch strategy; the `readahead` path is SVAR1-only.)

- [ ] **Step 1: Write the failing test.** In `tests/dataset/test_streaming_vcf.py`, assert `StreamingDataset(regions, reference=fasta, variants="x.vcf.gz")` constructs (no longer `NotImplementedError`) and reports the right `n_samples`/`ploidy`.

```python
def test_streaming_dataset_accepts_vcf(streaming_vcf_fixture):
    import genvarloader as gvl
    sds = gvl.StreamingDataset(
        streaming_vcf_fixture.regions,
        reference=str(streaming_vcf_fixture.fasta),
        variants=str(streaming_vcf_fixture.vcf),
    ).with_seqs("haplotypes")
    assert sds.n_samples == 2
    assert sds.ploidy == 2
```

- [ ] **Step 2: Run to see it fail.**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_vcf.py::test_streaming_dataset_accepts_vcf -v`
Expected: FAIL — `NotImplementedError` (VCF branch at `_streaming.py:197-201`).

- [ ] **Step 3: Implement `_VcfBackend` and wire the ladder.** Model the `__init__` on `_Svar1Backend.__init__` (`:581-721`) but source metadata from `genoray.VCF(path)` (header: `n_samples`, `ploidy`, sample names) instead of `SparseVar.index`; load `Reference.from_path` the same way; store the VCF path + contigs. `build_engine` mirrors `_Svar1Backend.build_engine` (`:723-795`) but constructs `RecordStreamEngine("vcf", vcf_path, sample_names, ploidy, contig_names, contig_ref_bytes, job_contig_idx, job_region_starts, job_region_ends, job_s_lo, job_s_hi, fasta_path, pad_char, parallel, batch_size)` — no global variant table (per-window local). Replace the VCF `NotImplementedError` at `:197-201`:

```python
        elif path_is_vcf(p):
            backend = _VcfBackend(p, reference, contigs, regions)
            self._backend_obj = backend
            n_samples = backend.n_samples
            ploidy = backend.ploidy
            samples = backend._sample_names
            contigs = backend._contigs
```

Set `self._backend = backend` and force `self._prefetch_strategy = "engine"` for VCF (the readahead path is SVAR1-only). Update the `_backend` field annotation.

- [ ] **Step 4: Run to see it pass.**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_vcf.py::test_streaming_dataset_accepts_vcf -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_vcf.py
git commit -m "feat(streaming): _VcfBackend + wire VCF into the StreamingDataset ladder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: VCF variant-table differential test (the parity gate)

Pin the #1 risk — genoray's Rust decoder vs the write path's Python decoder — at the cheapest layer, before reconstruction. Assert the streamed window's local variant table equals the written dataset's stored variant table for the same VCF.

**Files:**
- Create: `tests/dataset/test_streaming_vcf_parity.py`
- Test data: a small VCF built via `vcfixture` (Python) covering SNP + insertion + deletion + a multiallelic site across ≥3 samples.

**Interfaces:**
- Consumes: `_VcfBackend`/`RecordStreamEngine` (streamed table via a debug accessor — see Step 3), `gvl.write` + `Dataset.open` (oracle table via `SparseVar(dataset_dir).index` or the stored `_HapsFfiStatic`).

- [ ] **Step 1: Write the failing test.** Build the VCF, `gvl.write` it to a dataset, and compare variant tables. Getting the streamed table out requires a debug hook — add a `RecordStreamEngine.debug_decode_window(contig_idx, region_starts, region_ends, s_lo, s_hi) -> (v_starts, ilens, alt_alleles, alt_offsets)` pymethod that runs one `VcfWindowFiller::fill` and returns the local static table (no reconstruction). This accessor is test-only but ships (documented as such).

```python
def test_vcf_streamed_variant_table_matches_written(vcf_snp_ins_del_multi, tmp_path):
    import genvarloader as gvl, numpy as np
    from genoray import SparseVar
    # 1. write oracle
    gvl.write(tmp_path / "ds", vcf_snp_ins_del_multi.regions, variants=str(vcf_snp_ins_del_multi.vcf),
              reference=str(vcf_snp_ins_del_multi.fasta))
    oracle = SparseVar(tmp_path / "ds")  # or the stored variant table
    # 2. stream the same region window's table
    eng = _make_vcf_engine(vcf_snp_ins_del_multi)  # helper building RecordStreamEngine
    vs, il, alt, alt_off = eng.debug_decode_window(0, [0], [10_000], 0, vcf_snp_ins_del_multi.n_samples)
    # 3. compare on the overlapping variant set
    np.testing.assert_array_equal(vs, oracle_v_starts_in_region)
    np.testing.assert_array_equal(il, oracle_ilens_in_region)
    # ALT bytes via offsets
    assert _alts(alt, alt_off) == _alts(oracle_alt, oracle_alt_off)
```

- [ ] **Step 2: Run to see it fail.**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_vcf_parity.py::test_vcf_streamed_variant_table_matches_written -v`
Expected: FAIL — `debug_decode_window` missing (implement it) or a table mismatch (a real decoder divergence to resolve).

- [ ] **Step 3: Implement `debug_decode_window`** on `RecordStreamEngine` (one `fill` into a scratch `DecodedWindow`, return the four table arrays as numpy). If the tables mismatch, this is the parity bug the gate exists to catch — reconcile the normalization config (left-align/check-ref/atomization) against the write path (`_write.py`) until the tables match. Record any genoray-side divergence as a genoray issue if the mismatch is a genoray bug (memory: numba-oracle-bug-policy analog).

- [ ] **Step 4: Run to see it pass.**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_vcf_parity.py::test_vcf_streamed_variant_table_matches_written -v`
Expected: PASS — streamed table byte-equals the written table.

- [ ] **Step 5: Commit.**

```bash
git add src/record_stream/engine.rs tests/dataset/test_streaming_vcf_parity.py tests/data/streaming/
git commit -m "test(streaming): VCF variant-table differential gate vs written dataset

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: VCF end-to-end haplotype parity

The full oracle: every streamed haplotype byte-equals `Dataset.open(...)[r, s]`, `jitter=0`.

**Files:**
- Modify: `tests/dataset/test_streaming_vcf_parity.py` (add the end-to-end test)

**Interfaces:**
- Consumes: `gvl.StreamingDataset(...).to_iter(...)` (VCF), `gvl.write` + `Dataset.open` (oracle). Mirror the existing SVAR1 parity test (`tests/dataset/test_streaming_parity.py` — find `test_streaming_matches_written_all_cells`).

- [ ] **Step 1: Write the failing test.** Reuse the SVAR1 parity harness shape: write a dataset from the VCF, open it, iterate the streaming dataset, and for each emitted `(data, r_idx, s_idx)` assert `data` equals `written[r_idx, s_idx]`. Cover multi-region, multi-contig, and a window/batch-boundary case (batch_size not dividing window rows).

```python
def test_vcf_streaming_matches_written_all_cells(vcf_snp_ins_del_multi, tmp_path):
    import genvarloader as gvl, numpy as np
    gvl.write(tmp_path / "ds", vcf_snp_ins_del_multi.regions, variants=str(vcf_snp_ins_del_multi.vcf),
              reference=str(vcf_snp_ins_del_multi.fasta))
    written = gvl.Dataset.open(tmp_path / "ds", reference=str(vcf_snp_ins_del_multi.fasta)).with_seqs("haplotypes")
    sds = gvl.StreamingDataset(vcf_snp_ins_del_multi.regions, reference=str(vcf_snp_ins_del_multi.fasta),
                               variants=str(vcf_snp_ins_del_multi.vcf)).with_seqs("haplotypes")
    for data, r_idx, s_idx in sds.to_iter(batch_size=3):
        for k in range(len(r_idx)):
            exp = written[int(r_idx[k]), int(s_idx[k])]
            np.testing.assert_array_equal(data[k], exp)
```

- [ ] **Step 2: Run to see it fail** (if any prior task is incomplete) or pass. Run: `pixi run -e dev pytest tests/dataset/test_streaming_vcf_parity.py -v`. Expected initially: FAIL if a decode/reconstruct edge (indels, multiallelic, missing) diverges.

- [ ] **Step 3: Resolve divergences** via systematic-debugging (the variant-table gate from Task 7 isolates decode; a table-match-but-hap-mismatch points at the transpose or reconstruction args). Fix until byte-identical.

- [ ] **Step 4: Run the full streaming + unit suite.**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tests/dataset/test_streaming_vcf_parity.py
git commit -m "test(streaming): VCF end-to-end haplotype parity vs written Dataset

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: PR 1 docs gate + open the PR

**Files:**
- Modify: `skills/genvarloader/SKILL.md` (streaming `variants=` now accepts `.vcf[.gz]`/`.bcf`)
- Modify: `docs/source/faq.md`, `docs/source/dataset.md` (streaming reads VCF directly; htslib is a hard runtime requirement)
- Modify: `README.md` (streaming VCF support; requirements)
- Modify: `docs/roadmaps/streaming-dataset.md` (mark #276 VCF 🚧→ started; link this plan/spec)

- [ ] **Step 1:** Update the four docs + roadmap per the CLAUDE.md docs gate. Run the api.md/`__all__` sync check: `python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"` — expect `none` (no new public symbol).
- [ ] **Step 2:** Full tree: `pixi run -e dev pytest tests -q` and `pixi run -e dev cargo test` (with `LD_LIBRARY_PATH`). Expected: PASS.
- [ ] **Step 3: Commit + push + open draft PR into `streaming`.**

```bash
git add skills/ docs/ README.md
git commit -m "docs(streaming): document VCF streaming backend (#276)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push -u origin spec/276-vcf-pgen
gh pr create --draft --base streaming --title "perf(streaming): VCF backend behind RecordStreamEngine (#276)" \
  --body "Shared RecordStreamEngine + DenseChunk transpose + VCF backend. Part of #276. Relates to the StreamingDataset project.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

# PR 2 — PGEN backend

Additive on PR 1's engine. A second `WindowFiller` + Python backend, plus the benchmark harness.

---

## Task 10: `PgenWindowFiller` — POS→var-range mapping + decode

PGEN is not region-seekable; map the window's regions to a global variant-index range via `PvarReader`, construct `PgenRecordSource`, decode, transpose.

**Files:**
- Create: `src/record_stream/pgen.rs` (`PgenWindowFiller`)
- Modify: `src/record_stream/mod.rs` (`pub mod pgen;`)
- Modify: `src/record_stream/engine.rs` (`source_kind == "pgen"` branch in `#[new]`)

**Interfaces:**
- Produces: `PgenWindowFiller` implementing `WindowFiller`; holds the pgenlib `Py<PyAny>` reader handle, `.pvar` path, sample permutation, ploidy=2.
- Consumes: `genoray_core::{pgen_reader::PgenRecordSource, pvar::PvarReader, chunk_assembler::ChunkAssembler}`.

- [ ] **Step 1: Write the failing test.** Build a small PGEN (convert the Task 4 VCF via `plink2 --make-pgen`, committed under `tests/data/streaming/`). Assert `PgenWindowFiller::fill` on a window yields a `DecodedWindow` equal to the VCF filler's output for the same variants (cross-backend consistency at the buffer level).

```rust
#[test]
fn pgen_filler_matches_vcf_filler_on_shared_variants() {
    // fill via VcfWindowFiller and PgenWindowFiller for the same window; the
    // DecodedWindow static table + geno_v_idxs must be identical.
}
```

- [ ] **Step 2: Run to see it fail.** Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test pgen_filler`. Expected: FAIL — `PgenWindowFiller` not found.
- [ ] **Step 3: Implement `PgenWindowFiller::fill`:** open `PvarReader` to map the window's `regions` → `[var_start, var_end)` (scan `.pvar` POS; the reader is monotonic), build the pgenlib reader + `sample_perm` in Python at construction (store the `Py<PyAny>`), construct `PgenRecordSource::new(...)` (`pgen_reader.rs:70`), feed `ChunkAssembler`, transpose. The producer thread will need the GIL to drive pgenlib — acquire it inside `fill` with `Python::with_gil` only around the pgenlib call (pgenlib's C read releases it internally). Confirm arg order against `pgen_reader.rs:70` and `pvar.rs:55,140`.
- [ ] **Step 4: Run to see it pass.** Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test pgen_filler`. Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/record_stream/pgen.rs src/record_stream/mod.rs src/record_stream/engine.rs tests/data/streaming/
git commit -m "feat(streaming): PgenWindowFiller — pvar POS->var-range + decode

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: `_PgenBackend` (Python) + ladder wiring + engine constructor

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_PgenBackend`; wire PGEN branch at `:192-196`)
- Modify: `src/record_stream/engine.rs` (`#[new]` accepts `source_kind="pgen"` with `pvar_path` + pgenlib construction args)

**Interfaces:**
- Consumes: `genoray.PGEN` for header metadata; `RecordStreamEngine("pgen", ...)`.
- Produces: `_PgenBackend` (same duck interface as `_VcfBackend`: `n_samples`/`ploidy`/`_sample_names`/`build_engine`).

- [ ] **Step 1: Write the failing test** — `StreamingDataset(..., variants="x.pgen")` constructs and reports metadata. Mirror `test_streaming_dataset_accepts_vcf`.
- [ ] **Step 2: Run to see it fail** (`NotImplementedError` at `_streaming.py:192-196`).
- [ ] **Step 3: Implement `_PgenBackend`** (metadata from `genoray.PGEN`) and wire the ladder branch; add the `"pgen"` arm to the engine `#[new]`.
- [ ] **Step 4: Run to see it pass.**
- [ ] **Step 5: Commit.**

```bash
git add python/genvarloader/_dataset/_streaming.py src/record_stream/engine.rs tests/dataset/test_streaming_pgen.py
git commit -m "feat(streaming): _PgenBackend + wire PGEN into the StreamingDataset ladder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: PGEN parity (variant-table differential + end-to-end)

**Files:**
- Create: `tests/dataset/test_streaming_pgen_parity.py`

- [ ] **Step 1: Write the failing tests** — PGEN variant-table differential (streamed table == written-from-PGEN dataset table) and end-to-end haplotype parity. Mirror Tasks 7–8, `variants=<pgen>`.
- [ ] **Step 2: Run to see them fail** (or surface a decode divergence).
- [ ] **Step 3: Resolve divergences** (systematic-debugging; the table gate isolates decode).
- [ ] **Step 4: Full tree.** Run: `pixi run -e dev pytest tests/dataset tests/unit -q`. Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add tests/dataset/test_streaming_pgen_parity.py
git commit -m "test(streaming): PGEN variant-table differential + end-to-end parity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 13: Benchmark harness — `vcfixture bulk` + cohort sweep

Per the perf skill: parameterized benchmark sweeping n_samples, comparing synchronous vs engine, gated on deterministic signals + cold-cache overlap, not absolute wall-clock.

**Files:**
- Create: `benchmarking/streaming/gen_fixtures.sh` (or a pixi task) — `vcfixture bulk` → BCF, `plink2` → PGEN
- Create: `benchmarking/streaming/bench_streaming.py` — the sweep
- Modify: `pixi.toml` — add `gen-bench-vcf` / `gen-bench-pgen` tasks
- Modify: `docs/roadmaps/streaming-dataset.md` — record baseline + methodology

**Interfaces:**
- Consumes: `vcfixture` Rust CLI (`cargo install vcfixture --features cli`, or `cargo run` against `/carter/users/dlaub/projects/vcfixture-rs`), `plink2`, `gvl.StreamingDataset`.

- [ ] **Step 1: Add pixi tasks.** In `pixi.toml`:

```toml
[tasks.gen-bench-vcf]
cmd = "vcfixture bulk --profile germline-1kgp --samples ${N} --contigs chr1,chr2,chr3 --target-size ${SIZE} --seed 42 -o benchmarking/streaming/bench_${N}.bcf"

[tasks.gen-bench-pgen]
cmd = "plink2 --bcf benchmarking/streaming/bench_${N}.bcf --make-pgen --out benchmarking/streaming/bench_${N}"
```

Document `cargo install vcfixture --features cli` as the prerequisite (the `bulk` bin is behind the `cli` feature). Note in the script that these fixtures are gitignored (large).

- [ ] **Step 2: Write the sweep** (`bench_streaming.py`): for `N in [1_000, 10_000, 50_000]`, generate (or reuse) fixtures, time `StreamingDataset(...).to_iter(batch_size=32)` end-to-end (windows/s + items/s), record peak RSS (`resource.getrusage`), and compare the engine strategy vs a forced-synchronous baseline. Report median/min over reps. Confirm the scaling curve is flat-or-better per sample (decode-overlap win), not just one N. Measure the engine overlap on a **cold page cache** separately (drop caches or first-touch a fresh file) so decode-amortization isn't conflated. Measure VCF and PGEN separately (PGEN's GIL characteristic).
- [ ] **Step 3: Run the smallest sweep point** to validate the harness works end-to-end (not a perf claim yet).

Run: `N=1000 SIZE=50MB pixi run gen-bench-vcf && pixi run -e dev python benchmarking/streaming/bench_streaming.py --n 1000`
Expected: prints windows/s + peak RSS for VCF; no crash.

- [ ] **Step 4: Record baseline + methodology** in `docs/roadmaps/streaming-dataset.md` (numbers are same-session before/after + deterministic counters; shared node too noisy for absolute wall-clock — memory: rust-perf-gate-shared-node-noise).
- [ ] **Step 5: Commit.**

```bash
git add benchmarking/streaming/ pixi.toml docs/roadmaps/streaming-dataset.md
git commit -m "bench(streaming): vcfixture-bulk cohort-size sweep for VCF/PGEN backends

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 14: PR 2 docs gate + open the PR

**Files:**
- Modify: `skills/genvarloader/SKILL.md`, `docs/source/faq.md`, `docs/source/dataset.md`, `README.md` (add PGEN streaming; plink2 preprocessing note)
- Modify: `docs/roadmaps/streaming-dataset.md` (#276 ✅; PR link)

- [ ] **Step 1:** Update docs + roadmap; re-run the api.md/`__all__` sync check (expect `none`).
- [ ] **Step 2:** Full tree + cargo: `pixi run -e dev pytest tests -q` and `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib pixi run -e dev cargo test`. Expected: PASS.
- [ ] **Step 3: Commit + push + open draft PR into `streaming`.**

```bash
git add skills/ docs/ README.md
git commit -m "docs(streaming): document PGEN streaming backend (#276)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
gh pr create --draft --base streaming --title "perf(streaming): PGEN backend + cohort benchmark (#276)" \
  --body "PgenWindowFiller + _PgenBackend on the shared RecordStreamEngine, plus the vcfixture-bulk cohort sweep. Closes #276. Add to the StreamingDataset project.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Parallelization notes (for subagent-driven execution)

- **Critical path (sequential):** Task 1 → Task 2 → Task 3 → Task 5. These land the reusable core; each depends on the prior.
- **Task 4 (VCF filler)** depends on Task 2 (transpose) but is independent of Task 3 (engine) until Task 5 wires them — Tasks 3 and 4 can be built in parallel by two subagents after Task 2.
- **Tasks 7 + 8** (VCF parity) are sequential after Task 6 but independent of PR 2.
- **PR 2 Task 10 (PGEN filler)** is independent of the VCF filler (Task 4) — both are `WindowFiller` impls; a subagent can start it as soon as Task 3 (engine) lands, in parallel with the VCF-side Tasks 5–8.
- **Task 13 (benchmark harness)** is independent of parity tasks — can be built in parallel once a backend constructs.
- Per user preference: dispatch implementers with **subagent-driven-development** using **Sonnet or weaker** models, reserving stronger models for second-pass fixes only.
