# Roadmap: Write-free `StreamingDataset` (Rust streaming engine)

**Status legend:** ⬜ not started · 🚧 in progress · ✅ done · ⏸️ blocked

> **Living tracker — read this first, update it as you go.** Any work on the
> `StreamingDataset` effort (Python surface or Rust engine) must read this file before
> starting and update it as part of the change: tick tasks, set the status marker, and add
> spec/plan/PR pointers.
>
> **Keep it brief.** This file is a high-level summary + task list only. Push all detail
> (designs, measurements, rationale) into the linked specs/plans/PRs and point here.

## Goal

A write-free `Dataset` for inference: read variants (VCF/PGEN/SVAR1/SVAR2) and/or intervals
(BigWigs/Table) **directly from source files**, reconstruct haplotypes/tracks on the fly, and
iterate in a **fixed, layout-optimal order** — no `gvl.write()`, no on-disk cache. Slower per
epoch than a written `Dataset`, but zero preprocessing and zero disk footprint. **Iterable
only** (no map-style random access).

## Key design decisions

- **A shared framework, one backend per source family.** The Python `StreamingDataset` surface,
  the region-major scheduler, and the async producer/consumer engine are generic over a
  `StreamBackend` trait. Each variant format plugs in its own buffer + kernels.
- **Two buffer styles (do not cross-convert).**
  - VCF/PGEN/SVAR1 → **SVAR1-style window buffer** (local static variant table + per-hap sparse
    indices) → existing `reconstruct_haplotypes_from_sparse`.
  - SVAR2 → **SVAR2-style buffer** (flat `vk_*/dense_*/lut_*` channels, keys decoded inline) →
    existing `reconstruct_haplotypes_from_svar2_readbound`. No SVAR2→SVAR1 conversion.
- **Buffer variants, not haplotypes.** A fixed-size, allocated-once double (ping-pong) buffer
  holds a large window (≫ batch) of bulk-decoded variants; reconstruction consumes a *slice* per
  batch. Variants are tiny vs. haplotypes; each compressed block / mmap page is touched once.
- **Concurrency: `std::thread` + `crossbeam_channel::bounded`** (ping-pong / N-slot ring, slot
  recycling), rayon inside stages. **Not tokio** — mmap + CPU-bound decode has no async-I/O await
  points. Mirrors genoray's `orchestrator.rs` conversion pipeline. (tokio reserved for a future
  remote/object-store extension.)
- **Rust-first, bypass the genoray Python API.** Build on `genoray_core` + `svar2-codec`
  (+ `bigtools`). VCF/PGEN/SVAR1 require enabling `genoray_core`'s `conversion` feature
  (pulls rust-htslib/C htslib) — a first-class abi3-wheel-matrix risk. SVAR2 uses the
  already-linked `genoray_core::query` surface.
- **Reuse reconstruction kernels + output types**; the buffer duck-types the kernels' sparse
  inputs. **Byte-identical parity** with `gvl.write()` + `Dataset[r, s]` (modulo jitter/rng) is
  the correctness oracle.
- **Layout-optimal order.** Variants → region-major; BigWigs → sample-major; mixed → region-major
  (documented non-optimal) + `iteration_order` override for the hardware-dependent optimum.

## Specs

| Spec | Scope | Status |
|---|---|---|
| `docs/superpowers/specs/2026-07-15-streaming-dataset-vcf-pgen-svar1-design.md` | Shared framework + VCF/PGEN/SVAR1 backend | ✅ approved |
| _TBD_ — issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) | SVAR2 backend (SVAR2-style buffer + read-bound kernels) behind the framework | ⬜ |
| _TBD_ — issue [#279](https://github.com/mcvickerlab/GenVarLoader/issues/279) | Interval (BigWigs/Table) streaming + variant+interval mixed scheduler | ⬜ |

## Plans (spec A)

| Plan | Scope | Status |
|---|---|---|
| `docs/superpowers/plans/2026-07-15-streaming-dataset-svar1-walking-skeleton.md` | Walking skeleton: SVAR1 → haplotypes end-to-end, parity-verified (no double-buffer) | ✅ done — PR [#274](https://github.com/mcvickerlab/GenVarLoader/pull/274) |
| _TBD (Plan 2)_ — issue [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275) | Double-buffer engine (crossbeam producer/consumer, window sizing, `num_workers` shard) | ⬜ |
| _TBD (Plan 3/4)_ — issue [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) | VCF backend / PGEN backend | ⬜ |
| _TBD (Plan 5)_ — issue [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277) | Output-mode breadth (annotated/variants, `with_len`, `min_af`/`max_af`, `var_fields`, jitter) | ⬜ |

## Tasks (spec A — corrected ordering)

> **Correction:** SVAR1's reader is `conversion`-gated (htslib), so htslib enablement is a shared
> prerequisite and moves first. The first working slice is a synchronous walking skeleton; the
> crossbeam double-buffer is a separate throughput plan layered on top once parity is locked.

- ✅ **htslib / `conversion` enablement + wheel matrix** — `features=["conversion"]` enabled; abi3
  `cp310-abi3` wheel builds with htslib **statically** linked (no dynamic hts/libdeflate), so no
  optional-gating fallback was needed. Two unanticipated build fixes were required: **bigtools
  0.5.6→0.5.8** (hts-sys wants `libdeflate-sys ^1.21`, bigtools 0.5.6 pinned 0.13 — both declare
  `links = "libdeflate"`, a hard Cargo resolver conflict) and **`clangdev` 18 + `LIBCLANG_PATH`**
  in `pixi.toml` for hts-sys's mandatory bindgen (mirrors genoray's own pixi setup; 18 pinned
  because newer clang breaks bindgen 0.69 layouts). _Walking-skeleton Task 1_
- ✅ **Framework skeleton** — Python `StreamingDataset` (IterableDataset, region-major scheduler,
  index-carrying batches in the user's original bed-row order, `to_dataloader`). Batches are
  contig-grouped so every Rust call is single-contig. _Walking-skeleton Tasks 2, 5_
- ✅ **SVAR1 backend (synchronous)** — `Svar1Store` pyclass + `read_window` → existing
  `reconstruct_haplotypes_from_sparse` via a new `reconstruct_haplotypes_svar1` FFI.
  **Byte-identical parity** vs `gvl.write()`+`Dataset.open()[r,s]` across an unsorted,
  interleaved multi-contig bed (12 regions × 3 samples) through a real `DataLoader`.
  Public `gvl.StreamingDataset` + docs shipped. _Walking-skeleton Tasks 3–6_
- ⬜ **Double-buffer engine** — crossbeam producer/consumer, generic `StreamBackend`. _Plan 2_ —
  issue [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275)
  - ⚠️ **Inherited perf/scale debt from the walking skeleton — fold into Plan 2:** the per-contig
    static variant table crosses the FFI as Python **lists** (`.tolist()`, ~10M `int` objects for
    a human chr1) and `read_window` **clones the whole contig table on every batch**; it also
    re-opens `Svar1RecordSource` and walks the whole contig per batch (O(records × batch)), and
    holds the GIL during the record walk. Fix with `PyReadonlyArray1` + slices/`Arc`.
- ⬜ **VCF backend / PGEN backend** — _Plan 3/4_ —
  issue [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276)
- ⬜ **Output-mode breadth + docs** — _Plan 5; docs folded in_ —
  issue [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277)

## Sequencing

**#275 (double-buffer engine) is the keystone** — it defines the generic `StreamBackend` trait
that every other backend implements, and it rewrites the same `src/ffi/mod.rs` +
`_dataset/_streaming.py` surface the other tasks would build on. Land it first; work started
against the skeleton's shape gets rewritten.

| Wave | Work | Parallel? |
|---|---|---|
| **Now** | Spec B ([#278](https://github.com/mcvickerlab/GenVarLoader/issues/278)), spec C ([#279](https://github.com/mcvickerlab/GenVarLoader/issues/279)) — **writing only** | ✅ docs-only, no code conflict with #275 |
| **1** | [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275) double-buffer engine + skeleton perf debt | ⛔ serial — blocks everything below |
| **2** | [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) VCF/PGEN · [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) SVAR2 impl · [#279](https://github.com/mcvickerlab/GenVarLoader/issues/279) interval impl | ✅ one backend each, behind the trait |
| **3** | [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277) output-mode breadth | ✅ orthogonal to backends (kernel/output dispatch, not buffers) |

Notes:

- **#277 is the one judgement call.** It only needs the *synchronous* skeleton, so it could start
  today — but it edits the same two files #275 rewrites. Stack it on #275 rather than racing it.
- **#278 has no external blocker.** genoray is a **git dependency pinned to a `rev`**, not a
  crates.io release — an unpublished `genoray_core::query` API is reached by bumping the rev.
  #278 sequences on #275 like the other backends. See CLAUDE.md → Development Notes.
- **#276 and #279 are the widest parallel slot** — different source families (htslib vs bigtools),
  no shared kernels.

## Pointers

- **Parity contract & migration conventions:** `docs/archive/roadmaps/rust-migration.md`
  (completed; byte-identical parity, strangler-fig loop, differential-test harness).
- **SVAR2 read-bound precedent (the SVAR2-backend template):** rust-migration Phase 6a —
  `genoray_core::query` (`ContigReader`/`find_ranges`/`gather_haps_readbound`/`decode_hap`) +
  `reconstruct_haplotypes_from_svar2_readbound`. Reached by bumping the genoray git `rev` — no
  release needed (CLAUDE.md → Development Notes).
- **genoray Rust absorption:** rust-migration Phase 6 (⬜) — VCF/PGEN ingest into the Rust stack;
  enabling the `conversion` feature here overlaps that work (htslib producers).
- **Prefetching dataloader prior art:** the existing `buffered`/`double_buffered` torch
  transports in `python/genvarloader/_torch.py`; the `prefetching-dataloader` bench branch.
- **Concurrency template:** genoray's `orchestrator.rs`/`executor.rs` (`std::thread` +
  `crossbeam_channel::bounded` producer→executor→writer).
