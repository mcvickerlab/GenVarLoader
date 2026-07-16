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
| `docs/superpowers/specs/2026-07-15-streaming-dataset-vcf-pgen-svar1-design.md` | Shared framework + VCF/PGEN/SVAR1 backend | 🚧 design (pending review) |
| _TBD_ | SVAR2 backend (SVAR2-style buffer + read-bound kernels) behind the framework | ⬜ |
| _TBD_ | Interval (BigWigs/Table) streaming + variant+interval mixed scheduler | ⬜ |

## Tasks (spec A — framework + VCF/PGEN/SVAR1)

_Detail + parallelizable chunks in the spec above; plans/PRs TBD._

- ⬜ **Framework skeleton** — Python `StreamingDataset` (IterableDataset, `with_*`,
  `to_dataloader`, `len`, index-carrying batches) + region-major scheduler + generic
  `StreamBackend` crossbeam producer/consumer engine. _Plan: TBD · PR: TBD_
- ⬜ **SVAR1 backend** — `Svar1Window` buffer + `Svar1RecordSource` producer → sparse reconstruct
  (no htslib; first to parity). _Plan: TBD · PR: TBD_
- ⬜ **htslib enablement + wheel matrix** — turn on `conversion`; prove abi3 wheels; decide
  default-vs-optional gating. _Plan: TBD · PR: TBD_
- ⬜ **VCF backend** — `VcfRecordSource` producer + sparse-encode; `vcfixture` parity. _Plan: TBD · PR: TBD_
- ⬜ **PGEN backend** — `PgenRecordSource`/`PvarReader` producer + sparse-encode; parity. _Plan: TBD · PR: TBD_
- ⬜ **Output modes + settings** — annotated/variants, `with_len`, `min_af`/`max_af`,
  `var_fields`, `rc_neg`, jitter window; parity per mode. _Plan: TBD · PR: TBD_
- ⬜ **Docs / skill** — `__all__` + `api.md` + `SKILL.md` + prose. _PR: TBD_

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
