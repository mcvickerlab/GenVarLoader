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
- **Iterable-only, fixed cartesian traversal.** No `__getitem__`, no ad-hoc queries: a
  `StreamingDataset` is a fixed cartesian sweep of BED × samples in an order *we* choose from the
  data layout. `to_iter()` is the one entry point; `to_torch_dataset()`/`to_dataloader()` are thin
  wrappers. This is what makes the double buffer unconditionally safe — the next window is always
  known, so prefetch is never speculative. ("Pairwise" `(region, sample)` reads are a map-style
  artifact of `gvl.Dataset` and have no place here.)
- **Window ≫ batch; the window is the read granularity.** One backend read per *window*
  (R regions × all samples × ploidy, cartesian); a batch is a *slice* of the reconstructed window,
  never its own read.
- **Two buffer styles (do not cross-convert).**
  - VCF/PGEN → **SVAR1-style window buffer** (owned: decoded local static variant table + per-hap
    sparse indices) → existing `reconstruct_haplotypes_from_sparse`.
  - SVAR1 → the **degenerate** case of that buffer: offsets only. Its on-disk layout is *already*
    hap-major sparse CSR of sorted global variant ids, so `geno_v_idxs` is borrowed straight from
    the `variant_idxs` mmap — nothing is materialized and there is nothing to decode.
  - SVAR2 → **SVAR2-style buffer** (flat `vk_*/dense_*/lut_*` channels, keys decoded inline) →
    existing `reconstruct_haplotypes_from_svar2_readbound`. No SVAR2→SVAR1 conversion.
- **Buffer variants, not haplotypes.** A fixed-size, allocated-once double (ping-pong) buffer
  holds a large window (≫ batch); reconstruction consumes a *slice* per batch. Variants are tiny
  vs. haplotypes. For VCF/PGEN the win is amortizing **decode**; for SVAR1 there is no decode and
  the win is **hiding I/O (page-fault) latency** — the page cache does not prefetch on an
  application's access pattern, but a producer thread on a known traversal does.
- **Concurrency: `std::thread` + `crossbeam_channel::bounded`**, rayon inside stages. **Not
  tokio** — mmap + CPU-bound decode has no async-I/O await points. genoray's `orchestrator.rs` is
  the template for *named per-stage threads, bounded channels, close-by-`Sender`-drop shutdown,
  and join-everything-then-classify-panics* — but **not** for slot recycling, which it does not do
  (it allocates each chunk fresh and drops it). Recycling is net-new design here. (tokio reserved
  for a future remote/object-store extension.)
- **Rust-first, bypass the genoray Python API.** Build on `genoray_core` + `svar2-codec`
  (+ `bigtools`). VCF/PGEN require `genoray_core`'s `conversion` feature (rust-htslib/C htslib;
  already enabled and statically linked into the abi3 wheel). SVAR2 uses the already-linked
  `genoray_core::query` surface. **SVAR1 uses the ungated `svar1_query` surface** (a genoray
  prerequisite — see Tasks) and needs no htslib.
- **Reuse reconstruction kernels + output types**; the buffer duck-types the kernels' sparse
  inputs. **Byte-identical parity** with `gvl.write()` + `Dataset[r, s]` (modulo jitter/rng) is
  the correctness oracle.
- **Layout-optimal order.** Variants → region-major; BigWigs → sample-major; mixed → region-major
  (documented non-optimal) + `iteration_order` override for the hardware-dependent optimum.

## Specs

| Spec | Scope | Status |
|---|---|---|
| `docs/superpowers/specs/2026-07-15-streaming-dataset-vcf-pgen-svar1-design.md` | Shared framework + VCF/PGEN/SVAR1 backend | ✅ approved (⚠️ partly superseded — see below) |
| `docs/superpowers/specs/2026-07-16-streaming-svar1-window-engine-design.md` | SVAR1 window reads (ungated genoray `svar1_query`) + double-buffer engine + `to_iter` surface. **Supersedes spec A**'s SVAR1-producer, decode-amortization, slot-recycling, release-gate, and `IterableDataset` claims. | 🚧 pending review — issue [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275) |
| _TBD_ — issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) | SVAR2 backend (SVAR2-style buffer + read-bound kernels) behind the framework | ⬜ |
| _TBD_ — issue [#279](https://github.com/mcvickerlab/GenVarLoader/issues/279) | Interval (BigWigs/Table) streaming + variant+interval mixed scheduler | ⬜ |

## Plans (spec A)

| Plan | Scope | Status |
|---|---|---|
| `docs/superpowers/plans/2026-07-15-streaming-dataset-svar1-walking-skeleton.md` | Walking skeleton: SVAR1 → haplotypes end-to-end, parity-verified (no double-buffer) | ✅ done — PR [#274](https://github.com/mcvickerlab/GenVarLoader/pull/274) |
| _TBD (Plan 2)_ — issue [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275) | **Re-scoped:** genoray ungated `svar1_query` → gvl window-granular SVAR1 reads + double-buffer engine + `to_iter` surface. Spec: `2026-07-16-streaming-svar1-window-engine-design.md` | ⬜ |
| _TBD (Plan 3/4)_ — issue [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) | VCF backend / PGEN backend | ⬜ |
| _TBD (Plan 5)_ — issue [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277) | Output-mode breadth (annotated/variants, `with_len`, `min_af`/`max_af`, `var_fields`, jitter) | ⬜ |

## Tasks (spec A — corrected ordering)

> **Correction (2026-07-15):** htslib enablement moved first as a shared prerequisite. The first
> working slice is a synchronous walking skeleton; the crossbeam double-buffer is a separate
> throughput plan layered on top once parity is locked.
>
> **Correction (2026-07-16, Plan 2 design):** "SVAR1's reader is `conversion`-gated, so htslib is
> a prerequisite **for SVAR1**" was true only of the *conversion* reader. SVAR1 needs a **query**
> API, and its files are headerless raw mmap buffers — so the correct SVAR1 path is **ungated and
> htslib-free** (new genoray `svar1_query`). htslib remains a genuine prerequisite for VCF/PGEN
> (#276) only. `conversion` **stays enabled** in gvl regardless: it already builds green with
> static htslib and Plans 3/4 need it, so dropping and re-adding would churn `Cargo.toml` +
> `pixi.toml`'s `LIBCLANG_PATH` twice.

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
- ⬜ **genoray prerequisite: ungated `svar1_query`** — `Svar1Reader` + `var_ranges` +
  cartesian `find_ranges` in genoray, ungated (memmap2 + bytemuck + the existing `search.rs`; no
  htslib). **Root cause of the skeleton's SVAR1 debt:** genoray has an ungated Rust *query* API
  for SVAR2 (`genoray_core::query`) and **none for SVAR1**, so the skeleton reached for the
  conversion-gated `Svar1RecordSource` — a *conversion-pipeline record producer*, not a query API.
  Stage A is nearly free (`search::overlap_range` already ports Python's `var_ranges`; nobody
  wired it to SVAR1); Stage B is two `partition_point`s per hap. Consumed via a `rev` bump — **not
  a release gate** (see Development Notes in CLAUDE.md). _Plan 2, chunk 1_
- ⬜ **SVAR1 window reads + double-buffer engine + `to_iter` surface** — crossbeam
  producer/consumer, generic `StreamBackend`. _Plan 2_ —
  issue [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275)
  - ⚠️ **Inherited perf/scale debt from the walking skeleton — DELETED, not fixed, by Plan 2.**
    All of it is downstream of the one wrong dependency above, so the rewrite removes it rather
    than optimizing it. (a) `Svar1RecordSource::new` is **O(all CSR entries)** — it eagerly
    inverts the contig's whole hap-major CSR — and the skeleton calls it **per batch**; the real
    cost is far worse than the O(records × batch) walk alone. (b) The `.tolist()` per-contig table
    (~10M `int` objects for a human chr1) and `set_contig_table` exist **only** to feed that
    constructor → gone with it, not converted to `PyReadonlyArray1`. (c) The per-batch
    `t.pos.clone()` et al. exist only because the constructor takes its vectors **by value** →
    gone with it, not fixed with slices/`Arc`. (d) The GIL-held walk (`src/ffi/mod.rs:826`,
    *before* the `py.detach` at `:850`) is fixed by the `Svar2Store` template — borrow the reader
    across `py.detach`.
  - ⚠️ The skeleton **conflates window and batch** (one Rust call per batch). That conflation is
    what produced both the per-batch contig walk and the apparent need for pairwise reads.
- ⬜ **VCF backend / PGEN backend** — _Plan 3/4_ —
  issue [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276)
- ⬜ **Output-mode breadth + docs** — _Plan 5; docs folded in_ —
  issue [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277)

## Sequencing

**#275 is the keystone** — it defines the generic `StreamBackend` trait that every other backend
implements, and it rewrites the same `src/ffi/mod.rs` + `_dataset/_streaming.py` surface the other
tasks would build on. Land it first; work started against the skeleton's shape gets rewritten.

**#275 is itself gated on a genoray PR** (ungated `svar1_query`) — a cross-repo prerequisite, not
a release gate: merge it to genoray `main`, then bump gvl's `rev`. Start there.

| Wave | Work | Parallel? |
|---|---|---|
| **Now** | Spec B ([#278](https://github.com/mcvickerlab/GenVarLoader/issues/278)), spec C ([#279](https://github.com/mcvickerlab/GenVarLoader/issues/279)) — **writing only** | ✅ docs-only, no code conflict with #275 |
| **0** | **genoray: ungated `svar1_query`** (+ merge to genoray `main`, bump gvl's `rev`) | ⛔ cross-repo prerequisite — gates #275 |
| **1** | [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275) SVAR1 window reads + double-buffer engine + `to_iter` | ⛔ serial — blocks everything below |
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
- **SVAR1 query precedent (the SVAR1-backend template):** genoray's ungated `svar1_query` —
  `Svar1Reader` + `var_ranges` (a thin wrapper over the pre-existing `search::overlap_range`) +
  cartesian `find_ranges`. Mirrors the Python `SparseVar.var_ranges` → `_find_starts_ends` →
  `Ragged.from_offsets` path (`_var_ranges.py`, `_svar/_core.py`, `_svar/_kernels.py`), which is
  what `gvl.write()` itself already calls (`_write.py:1023-1027`). ⚠️ **Not**
  `svar1_reader::Svar1RecordSource` — that is the conversion pipeline's record producer
  (forward-only, O(all CSR entries) at construction) and is the wrong tool for a query.
- **genoray Rust absorption:** rust-migration Phase 6 (⬜) — VCF/PGEN ingest into the Rust stack;
  enabling the `conversion` feature here overlaps that work (htslib producers).
- **Prefetching dataloader prior art:** the existing `buffered`/`double_buffered` torch
  transports in `python/genvarloader/_torch.py`; the `prefetching-dataloader` bench branch.
- **Concurrency template:** genoray's `orchestrator.rs`/`executor.rs` (`std::thread` +
  `crossbeam_channel::bounded` producer→executor→writer).
