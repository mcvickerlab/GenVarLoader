# Design: `StreamingDataset` — Rust-native VCF + PGEN backends

**Date:** 2026-07-17
**Status:** design (pending spec review)
**Roadmap:** `docs/roadmaps/streaming-dataset.md`
**Issue:** [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) (Plan 3/4)
**Follows:** SVAR1 window engine [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275)
(PRs [#282](https://github.com/mcvickerlab/GenVarLoader/pull/282),
[#295](https://github.com/mcvickerlab/GenVarLoader/pull/295),
[#297](https://github.com/mcvickerlab/GenVarLoader/pull/297))
**Supersedes parts of:** `docs/superpowers/specs/2026-07-15-streaming-dataset-vcf-pgen-svar1-design.md`
("spec A") — the VCF/PGEN backend sketch, corrected below.

## Summary

Add **VCF and PGEN variant backends** to the write-free `StreamingDataset`, behind the streaming
framework the SVAR1 work established. Both read variants directly from source files in
region-major windows, reconstruct haplotypes on the fly with the existing Rust kernel, and reach
**byte-identical parity** with `gvl.write()` + `Dataset.open()[r, s]` (the test oracle).

The decisive architectural choice (deliberated during design): **decode lives in Rust**, on
genoray's `conversion`-gated reader pipeline (`RecordSource` → `ChunkAssembler` → `DenseChunk`),
not at the Python `genoray.VCF`/`genoray.PGEN` layer. This keeps the backend symmetric with the
`Svar1StreamEngine` (a detached producer thread whose whole job is to hide **decode CPU** — the
win `madvise`/read-through cannot give, and the reason the double buffer earns its keep for
VCF/PGEN where SVAR1 only overlapped I/O). The cost — a second, independent VCF decoder that
must stay byte-identical to the write path's cyvcf2 — is accepted and policed by an early
variant-table differential test (see [Parity](#parity--the-1-risk)).

### What ships

- `gvl.StreamingDataset(regions, reference=..., variants="x.vcf.gz" | "x.pgen")` reads variants
  directly and yields haplotype batches in region-major order, no `gvl.write()`, no disk cache.
- One shared **`RecordStreamEngine`** (Rust pyclass) generic over genoray's `RecordSource`; VCF
  and PGEN differ only in how the source is constructed.
- Byte-identical to a written dataset for the same inputs (haplotypes, `jitter=0`).

### Scope (inherits the current streaming constraints)

**In scope:** two variant backends (VCF, PGEN); **haplotypes-only** output; `jitter=0`;
`num_workers <= 1`; parity across both backends; a `vcfixture bulk` → (VCF, PGEN) benchmark
harness.

**Out of scope (deferred to [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277)):**
`reference`/`annotated`/`variants` output modes, `with_len`, `with_settings`, `min_af`/`max_af`,
`var_fields`, jitter windows, `num_workers > 1` sharding. These are orthogonal to the backend
seam and land as a later plan, exactly as they are today for SVAR1.

## Supersession of spec A

Spec A remains the framework record. Its VCF/PGEN sketch is corrected here:

| Spec A claim | Correction |
|---|---|
| Producers drive `VcfRecordSource`/`PgenRecordSource` and "sparse-encode into the buffer" as loose steps | The genoray pipeline is **two-stage and already exists**: `RecordSource` → `ChunkAssembler::read_next_chunk` → `DenseChunk` (normalized, atomized, ILEN computed, ALT CSR, dense `BitGrid3` genotypes). gvl adds only a variant-index **transpose**, not a from-scratch sparse encoder. |
| htslib enablement is chunk 1, "primary risk… prove the wheel matrix" | htslib is **already enabled** — `Cargo.toml` links `genoray_core` with `features = ["conversion"]` today (a vestige of the deleted walking-skeleton `Svar1RecordSource`). This spec makes it **load-bearing** rather than newly-added. The wheel matrix already builds green with it; re-confirm on the pinned rev, don't re-litigate. |
| Backend fits the generic `StreamBackend`/`run_windows` engine | It does **not**. `run_windows` uses `std::thread::scope`, whose producer cannot outlive the many separate `next_batch()` FFI calls the iterator makes. The backend mirrors `Svar1StreamEngine`'s **detached `std::thread::spawn`** instead — the established precedent. |
| `StreamingDataset` is a torch `IterableDataset` | It is a plain frozen dataclass; `to_iter` is the entry point (settled in #275). |
| "cannot ship until genoray publishes" | genoray is a git `rev` pin; bumping the rev is the mechanism (CLAUDE.md → Development Notes). No new genoray API is required for this spec — the `conversion` readers already exist at the pinned rev. |

Unchanged from spec A: region-major order for variants; reuse of the reconstruction kernel and
output types; byte-identical parity as the oracle; `std::thread` + `crossbeam_channel::bounded`
(not tokio).

## Architecture

The reader pipeline is genoray's, used as-is. gvl adds a transpose and the engine wiring.

```
VcfRecordSource | PgenRecordSource        (both impl genoray_core::RecordSource, conversion-gated)
   │  VCF: rust-htslib IndexedReader region fetch, region-major within a contig
   │  PGEN: pvar POS→global-var-range, then pgenlib sequential 32 MiB refill + region filter
   ▼
ChunkAssembler::read_next_chunk  →  DenseChunk        ← ALL normalization happens here
   │  atomize + biallelic-split; left-align iff a reference FASTA is supplied (L_MAX=1000);
   │  check-ref; phase discarded; ILEN computed; ALT stored as a CSR; genotypes as BitGrid3 (V,S,P)
   ▼
[NEW gvl] variant-index transpose  →  window-local  geno_v_idxs: i32  +  geno_offsets (CSR)
   │  adapted from genoray rvk.rs dense2sparse_vk transpose, emitting the variant COLUMN INDEX v
   │  (into the DenseChunk static table) instead of genoray's self-describing per-call keys
   ▼
reconstruct_haplotypes_from_sparse                    (existing kernel — untouched)
```

### Why `DenseChunk` is the right seam

`DenseChunk` (`genoray_core::types`) is, field for field, the **window-local static variant
table** the reconstruction kernel already consumes:

| kernel input | `DenseChunk` field | note |
|---|---|---|
| `v_starts: i32` | `pos: Vec<u32>` | 0-based; `u32 → i32` |
| `ilens: i32` | `ilens: Vec<i32>` | **already computed** as `alt.len() - ref.len()` during atomization — not derived |
| `alt_alleles: u8` + `alt_offsets` | `alt: Vec<u8>` + `alt_offsets` | atomized ALT (DEL anchor = 1 base; INS = anchor+inserted), CSR |
| contig reference bytes | — (from gvl's `Reference`, per contig) | same as SVAR1; per-variant REF is **not** needed — reconstruction applies ALT+ILEN against the contig sequence |

Normalization — the parity-critical logic — is genoray's (`normalize.rs`), shared with the rest
of the conversion pipeline. gvl does not reimplement atomization or left-alignment.

### The one thing gvl writes: the transpose

genoray's `dense2sparse_vk` emits a *self-describing* hap-major stream (each call = genomic POS +
a 32-bit key inlining ILEN and ALT). The reconstruction kernel instead wants **indices** into the
static table. So gvl copies the transpose loop (`genoray rvk.rs`, the ungated `dense2sparse_vk`
body — available even without `conversion`) and, per set genotype bit, pushes the **variant
column index `v`** (the loop variable, which *is* the static-table row) into `geno_v_idxs`,
reusing genoray's per-hap `sample_lengths` counting verbatim to build the CSR offsets. This is
the whole gvl-side "decode-to-kernel-input" bridge: ~60 lines, one struct.

### The shared engine

A single **`RecordStreamEngine`** pyclass, generic over `S: RecordSource` (VCF and PGEN are the
two `S`), mirroring `Svar1StreamEngine` (`src/ffi/stream_engine.rs`):

- **Detached `std::thread::spawn` producer** owning the source + assembler, outliving many
  `next_batch()` FFI calls. Not `run_windows` (see supersession table).
- Two `crossbeam_channel::bounded(2)` channels (filled / free) — ping-pong slot recycling, the
  same shape as `Svar1StreamEngine`.
- Producer loop per window: pull a recycled slot → drive `RecordSource → ChunkAssembler` for the
  window → transpose into the slot → send. **This decode is what the consumer's reconstruction
  overlaps.**
- `next_batch(py) -> Option<(data, offsets)>` runs the blocking recv + reconstruction under
  `py.detach` (GIL released); reacquires only to marshal numpy arrays — identical to
  `Svar1StreamEngine`.
- Reconstruction reuses `generate_batch_core`, **generalized** to take the window-local static
  table from the slot instead of only `Svar1Store`'s global pyclass fields. The kernel call
  (`reconstruct_haplotypes_from_sparse`) is unchanged; only the arg-assembly wrapper learns to
  source the table from either place. Output stays batch-bounded (#284).

### The owned window buffer (slot)

Unlike SVAR1's offsets-only slot (indices into a global mmap), VCF/PGEN materialize an **owned
decoded window**:

```rust
#[derive(Default)]
struct DecodedWindow {
    // window-local static variant table (from DenseChunk)
    v_starts: Vec<i32>,
    ilens: Vec<i32>,
    alt_alleles: Vec<u8>,
    alt_offsets: Vec<i64>,
    // window-local sparse genotypes (from the transpose)
    geno_v_idxs: Vec<i32>,
    geno_offsets: Vec<i64>,   // CSR, per-hap
    job_idx: usize,
}
```

Recycled ping-pong; `fill` reuses the `Vec` allocations. This is the "owned buffer" the SVAR1
`mod.rs` doc anticipated for #276 ("VCF/PGEN backends DO materialize an owned buffer; do not take
[SVAR1] as their template").

## Per-source specifics

### VCF

`VcfRecordSource::new(vcf_path, chrom, samples, htslib_threads, ploidy, fields=&[], regions,
overlap)` — rust-htslib `IndexedReader` with true index region fetch (needs `.tbi`/`.csi`).
Records are walked region-major within a contig (`advance_region` re-fetches each coalesced
interval). Genotypes decode straight from the raw BCF integer buffer, all samples, allele indices
with `-1` = missing; phase is dropped. Fully **GIL-free** (no Python in the loop) — the cleanest
double-buffer overlap.

### PGEN

Not region-seekable at the PGEN level. Per window the backend maps region POS → a **global
variant-index range** `[var_start, var_end)` via `PvarReader` (streaming `.pvar`/`.pvar.zst`
text; POS 1-based→0-based, REF/ALT uppercased, `.` ALT → monomorphic), then constructs
`PgenRecordSource::new(pgen_reader, pvar_path, var_start, var_end, num_samples, chunk_size,
regions, overlap, sample_perm)`. Reads are sequential 32 MiB refills, region-filtered *after*
decode by a monotonic cursor. Ploidy is hardwired to 2; pgenlib's `-9` missing → `-1`.

**Known characteristic — PGEN decode is not GIL-free.** genoray's `PgenRecordSource` holds a
`Py<PyAny>` pgenlib reader and calls `read_alleles_range` across the GIL. pgenlib's C read should
release the GIL internally, so producer/consumer overlap is *mostly* preserved, but PGEN's
overlap is **measured separately** (not assumed). If contention proves material, the fallback is
documented but not built here (out of scope): a longer-lived source or a Rust-side pgen reader.

## Python surface

New `_VcfBackend` and `_PgenBackend` classes, **duck-typed to `_Svar1Backend`** — the same
implicit interface `StreamingDataset._iter_batches` consumes:
`n_samples`, `ploidy`, `_sample_names`, `_store`, `build_engine(jobs, batch_size)`,
`read_window(r_idx, s_idx)`, `generate_batch(...)`. `build_engine` returns the shared
`RecordStreamEngine`.

They wire into the classification ladder at `python/genvarloader/_dataset/_streaming.py:192-201`,
replacing the two `NotImplementedError`s (PGEN, VCF) with `_backend_obj = _PgenBackend(...)` /
`_VcfBackend(...)` and setting `contigs`/`n_samples`/`ploidy`/`samples` from the backend, exactly
as the `.svar` branch does at lines 177-186. The `_prefetch_strategy == "engine"` path in
`_iter_batches` needs no change — it already drives any backend that exposes `build_engine`.

`with_seqs` stays haplotypes-only (its non-haplotype guard is #277). `jitter != 0` and
`num_workers > 0` guards stay as they are.

## Parity — the #1 risk

Streaming decodes via genoray's **Rust** `ChunkAssembler`; the write oracle decodes via genoray's
**Python** cyvcf2 (VCF) / pgenlib (PGEN) + `dense2sparse`. Byte-identical parity requires these two
independent decoders to agree on:

- suffix-trimming and atomization rules (`normalize.rs::atomize_biallelic`);
- the substituted-DEL-anchor split;
- left-alignment **only when a reference is present**, with `L_MAX = 1000` partial-align
  truncation (matching bcftools `--buffer-size`);
- check-ref mode;
- phase-discard (each ploidy column an independent haplotype regardless of `|` vs `/`);
- missing → REF collapse (missing is an unset bit in the dense grid; no separate missing mask).

**Mitigation — pin it early, at the cheapest layer.** Before wiring reconstruction, a
**variant-table-level differential test** asserts the streamed `DenseChunk`'s `pos`/`ilens`/`alt`
equals the written dataset's stored variant table for the same input. This catches a decoder
divergence at the table, not after it has propagated through reconstruction into an opaque byte
diff. It is the highest-value, cheapest test in the effort — it needs no genotypes.

Then the full oracle: byte-identical streamed item == `Dataset.open(...)[r, s]` (mapped back by
emitted indices), haplotypes, `jitter=0`, for both VCF and PGEN.

## Memory

The window decode materializes a dense `BitGrid3` (V_window × n_samples × ploidy **bits**) plus
the transposed sparse output — heavier than SVAR1's offsets-only slot, though the dense grid is
bits, not bytes. The existing `max_mem` → `window_regions`/`window_samples` sizing
(`_streaming.py`) must account for the dense grid + sparse output. Per-batch **reconstruction**
output stays #284-bounded (batch-slice generation), and the ping-pong doubles the resident
window count (current + prefetched), as noted for the engine generally.

## Benchmark & performance plan

Follows `skills/performant-py-rust`. The double buffer's justification for VCF/PGEN is **hiding
decode CPU** (unlike SVAR1's I/O-only overlap), so the benchmark must confirm that.

**Workload dimensions (dominating axis drives the design):**

| dimension | typical | max | grows? | notes |
|---|---|---|---|---|
| **n_samples** | 1k–10k | 100k+ | **grows (cohort)** | dominates genotype width + decode; the OOM axis (#284) |
| n_variants/window | region density | — | with `window_regions` | drives `DenseChunk` + transpose cost |
| n_regions/window | 64 (`region_target`) | — | tunable | window granularity |
| ploidy | 2 | 2 | fixed | |
| file size | 100 MB–GBs | — | grows | I/O + decompress |

**Bound:** producer = CPU-bound decode (htslib decompress + parse + genoray normalize/atomize +
gvl transpose) + I/O; consumer = CPU-bound reconstruct; the two overlap on separate threads.
**Dominating axis = n_samples** → the benchmark sweeps it.

**Oracle:** the parity oracle above (byte-identical vs `gvl.write`).

**Benchmark:** parameterized, **sweep n_samples** at fixed variant density; report windows/s (or
items/s) + peak RSS; compare **synchronous-readahead vs engine** (the A-vs-C shape the SVAR1 work
used). Confirm the scaling curve, not just a single-N speedup. **Gate on deterministic secondary
signals + same-session before/after**, not absolute wall-clock (shared node is too noisy — the
established SVAR1 discipline). Measure the engine overlap on a **cold page cache**, separately, so
the decode-amortization win isn't conflated with anything else. PGEN's overlap is measured
separately from VCF's (GIL characteristic above).

**Fixtures — `vcfixture-rs` for large VCFs:**

- **Large VCF/BCF (new):** the `vcfixture` Rust crate's `bulk` subcommand (behind its `cli`
  feature) streams large, realistic-enough BCF from a fitted profile:
  ```bash
  vcfixture bulk --profile germline-1kgp --samples <N> --contigs chr1,chr2,chr3 \
    --target-size <sz> --seed 42 -o bench.bcf
  ```
  Add it as a **bench-fixture tool**, not a linked Cargo dependency: a pixi task
  (`gen-bench-vcf`) wrapping `cargo install vcfixture --features cli` (or a pinned `cargo run`
  against the `d-laub/vcfixture-rs` checkout). Sweeping `--samples` produces the cohort-size
  sweep the benchmark needs.
- **Large PGEN (new):** a `gen-bench-pgen` task converts the generated BCF → PGEN via `plink2`,
  so both backends benchmark off the same variants.
- **Parity fixtures (unchanged):** the existing Python `vcfixture` (pixi `>=0.6.0,<0.7`)
  ground-truth oracle for small, exact fixtures.

## Testing strategy

- **Variant-table differential** (highest value, cheapest): streamed `DenseChunk` `pos`/`ilens`/
  `alt` == written dataset's stored variant table, per backend. Runs before reconstruction.
- **Parity** (primary): byte-identical streamed haplotypes vs `Dataset[r, s]`, VCF and PGEN,
  `jitter=0`, across multi-contig windows and window/batch boundaries.
- **Engine unit tests** (Rust `cargo test`): slot fill/recycle; producer/consumer ownership
  handoff; window vs batch boundary; the transpose (dense grid → `geno_v_idxs` + CSR) against a
  hand-checked small grid.
- **Scale guard:** assert no per-batch materialization of a sample-scale array beyond the decode
  window (mirrors the rust-migration scale-guard).
- **Fixtures:** `vcfixture` (Python) for VCF ground truth; a small PGEN built via plink2 from the
  same VCF; `vcfixture bulk` (Rust CLI) for the benchmark only.

## Sequencing (for planning)

Two PRs on the long-lived `streaming` integration branch, both under #276:

1. **Shared engine + VCF backend.** `RecordStreamEngine` (generic over `RecordSource`) + the
   `DenseChunk` → `geno_v_idxs` transpose + `generate_batch_core` generalization + `VcfStore`
   pyclass + `_VcfBackend` (Python) + classification-ladder wiring + the variant-table
   differential test + VCF parity (via `vcfixture` Python oracle). VCF first because it has a
   ground-truth oracle and a GIL-free producer.
2. **PGEN backend.** `PvarReader` POS→var-range mapping + `PgenRecordSource` construction +
   `PgenStore`/`_PgenBackend` reusing the engine + PGEN parity + PGEN-specific overlap
   measurement. Depends on PR 1's engine.

The engine + transpose + `generate_batch_core` generalization land once (PR 1) and both backends
share them; PR 2 is additive.

## Docs gate

Per CLAUDE.md, before the feature PR: update `skills/genvarloader/SKILL.md` (new `variants=`
accepted sources for streaming), `docs/source/{api,faq,dataset}.md`, and `README.md` (streaming
now reads VCF/PGEN directly; htslib is a hard runtime requirement). Keep `api.md` in sync with
`__all__` (no new public symbol here beyond the already-exported `StreamingDataset`, but the
accepted `variants=` values change).

## Deferred / open questions

- **Output modes / `with_len` / `with_settings` / jitter / AF filtering / `var_fields`** →
  [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277).
- **`num_workers > 1` sharding** — deferred (the skeleton's guard stays).
- **PGEN GIL-overlap** — characterized, not optimized here; a Rust-side pgen reader or
  longer-lived source is a future option only if contention is measured to matter.
- **`window_regions`/`max_mem` defaults for the dense decode grid** — measure on the `vcfixture
  bulk` corpus; ship a sensible default, don't hardcode a guess.
- **Dropping htslib entirely** — *not* pursued: Path A makes `conversion` load-bearing on
  purpose. Recorded so a future reader doesn't mistake the enabled feature for dead weight.
