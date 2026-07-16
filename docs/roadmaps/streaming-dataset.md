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
| _TBD_ | SVAR2 backend (SVAR2-style buffer + read-bound kernels) behind the framework | ⬜ |
| _TBD_ | Interval (BigWigs/Table) streaming + variant+interval mixed scheduler | ⬜ |

## Plans (spec A)

| Plan | Scope | Status |
|---|---|---|
| `docs/superpowers/plans/2026-07-15-streaming-dataset-svar1-walking-skeleton.md` | Walking skeleton: SVAR1 → haplotypes end-to-end, parity-verified (no double-buffer) | 🚧 ready to execute |
| _TBD (Plan 2)_ | Double-buffer engine (crossbeam producer/consumer, window sizing, `num_workers` shard) | ⬜ |
| _TBD (Plan 3/4)_ | VCF backend / PGEN backend | ⬜ |
| _TBD (Plan 5)_ | Output-mode breadth (annotated/variants, `with_len`, `min_af`/`max_af`, `var_fields`, jitter) | ⬜ |

## Tasks (spec A — corrected ordering)

> **Correction:** SVAR1's reader is `conversion`-gated (htslib), so htslib enablement is a shared
> prerequisite and moves first. The first working slice is a synchronous walking skeleton; the
> crossbeam double-buffer is a separate throughput plan layered on top once parity is locked.

- ⬜ **htslib / `conversion` enablement + wheel matrix** — `features=["conversion"]`; prove abi3
  wheels; decide default-vs-optional gating. Shared prerequisite (build-risk spike). _Walking-skeleton Task 1_
- ⬜ **Framework skeleton** — Python `StreamingDataset` (IterableDataset, region-major scheduler,
  index-carrying batches, `to_dataloader`); reuse `_buffered_loader.py` patterns. _Walking-skeleton Task 2_
- ⬜ **SVAR1 backend (synchronous)** — `Svar1Store` pyclass + window read → existing
  `reconstruct_haplotypes_from_sparse`; byte-identical parity. _Walking-skeleton Tasks 3–6_
- ⬜ **Double-buffer engine** — crossbeam producer/consumer, generic `StreamBackend`. _Plan 2_
- ⬜ **VCF backend / PGEN backend / output modes / docs** — _Plans 3–5; docs folded per plan._

## Pointers

- **Parity contract & migration conventions:** `docs/archive/roadmaps/rust-migration.md`
  (completed; byte-identical parity, strangler-fig loop, differential-test harness).
- **SVAR2 read-bound precedent (the SVAR2-backend template):** rust-migration Phase 6a —
  `genoray_core::query` (`ContigReader`/`find_ranges`/`gather_haps_readbound`/`decode_hap`) +
  `reconstruct_haplotypes_from_svar2_readbound`. ⛔ genoray release-gate applies (dev-wired,
  unpublished).
- **genoray Rust absorption:** rust-migration Phase 6 (⬜) — VCF/PGEN ingest into the Rust stack;
  enabling the `conversion` feature here overlaps that work (htslib producers).
- **Prefetching dataloader prior art:** the existing `buffered`/`double_buffered` torch
  transports in `python/genvarloader/_torch.py`; the `prefetching-dataloader` bench branch.
- **Concurrency template:** genoray's `orchestrator.rs`/`executor.rs` (`std::thread` +
  `crossbeam_channel::bounded` producer→executor→writer).
