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

- **Buffer variants, not haplotypes.** A fixed-size, allocated-once **double (ping-pong)
  buffer** holds a large window (≫ batch) of bulk-decoded variants in a compact SVAR1-style
  sparse representation. Reconstruction consumes a *slice* per batch. Variants are tiny vs.
  haplotypes and reconstruction is cheap vs. buffer-fill I/O.
- **Async producer/consumer (Rust/tokio).** Fill buffer B while the consumer reads buffer A so
  I/O + bgzip decode overlaps reconstruction; each compressed block is touched once, not
  re-decoded per small batch.
- **Rust-first, strong-typed.** Build on the `genoray_core` + `svar2-codec` Rust crates
  (+ `bigtools`); **bypass the genoray Python API**. Throughput-critical; correctness and
  memory-safety (buffer always valid, no torn/stale reads) enforced by the type system.
- **Reuse existing reconstruction kernels.** The double-buffer feeds the same
  `reconstruct_haplotypes_fused` / read-bound kernels the written `Dataset` uses.
- **Layout-optimal order.** Single variant source → region-major; BigWigs → sample-major;
  mixed → region-major (documented non-optimal), with a power-user override for the
  hardware-dependent optimum.
- **Byte-identical parity** with `gvl.write()` + `Dataset.open()[r, s]` (modulo jitter/rng) is
  the correctness oracle — same contract as the Rust migration.

## Tasks

- ⬜ **Design & spec** — architecture, buffer format, async engine, scheduler, Python API.
  _Spec: TBD (`docs/superpowers/specs/…`)_
- ⬜ **Variants-only streaming path** — per backend (VCF/PGEN/SVAR1/SVAR2) + optional reference,
  region-major. _Plan: TBD · PR: TBD_
- ⬜ **Async double-buffer engine (Rust)** — producer/consumer, ping-pong buffer, SVAR1-style
  window format. _Plan: TBD · PR: TBD_
- ⬜ **Intervals-only streaming path** — BigWigs/Table, sample-major. _Plan: TBD · PR: TBD_
- ⬜ **Mixed sources + scheduler** — region-major default + `iteration_order` override. _Plan: TBD · PR: TBD_
- ⬜ **Python API + torch `IterableDataset`** — `gvl.StreamingDataset`, `with_*` builders,
  `to_dataloader`, index-carrying batches. _Plan: TBD · PR: TBD_
- ⬜ **Docs / skill** — `__all__` + `api.md` + `SKILL.md` + prose docs. _PR: TBD_

## Pointers

- **Parity contract & migration conventions:** `docs/archive/roadmaps/rust-migration.md`
  (completed; byte-identical parity, strangler-fig loop, differential-test harness).
- **SVAR2 read-bound precedent (the streaming template):** rust-migration Phase 6a — a live
  per-batch read backend feeding the reconstruction kernels with no dense genotype matrix.
- **genoray Rust absorption:** rust-migration Phase 6 (⬜) — VCF/PGEN ingest into the Rust
  stack; this effort drives that work.
- **Prefetching dataloader prior art:** `prefetching-dataloader` branch / dataloader bench
  (see rust-migration baseline-metrics section); the existing `buffered`/`double_buffered`
  torch transports in `python/genvarloader/_torch.py`.
