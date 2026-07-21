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
| `docs/superpowers/specs/2026-07-16-streaming-svar1-window-engine-design.md` | SVAR1 window reads (ungated genoray `svar1_query`) + double-buffer engine + `to_iter` surface. **Supersedes spec A**'s SVAR1-producer, decode-amortization, slot-recycling, release-gate, and `IterableDataset` claims. | ✅ approved — issue [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275), PR [#282](https://github.com/mcvickerlab/GenVarLoader/pull/282) |
| `docs/superpowers/specs/2026-07-17-streaming-vcf-pgen-backends-design.md` | Rust-native VCF/PGEN decode (genoray `RecordSource -> ChunkAssembler -> DenseChunk`) behind the shared `RecordStreamEngine`; a `geno_v_idxs`/CSR transpose is the only gvl-side addition. | ✅ approved — issue [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) |
| `docs/superpowers/specs/2026-07-17-streaming-svar2-backend-design.md` | SVAR2 backend Phase 1 (SVAR2-style buffer + read-bound kernels) behind the framework | ✅ approved — issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) |
| `docs/superpowers/specs/2026-07-18-streaming-svar2-phase2-design.md` | SVAR2 Phase 2 — **measurement-gated, gate RUN**: SVAR2 streaming is CPU-bound (not IO-bound like SVAR1) and per-batch `parallel=True` is harmful (rayon overhead on ~64 tiny haplotypes/call). Design leads with a fast synchronous path (`parallel=False` + GIL-free Rust `find_ranges`, no rev bump) then Rust-side many-core super-batch reconstruction; producer thread demoted to a gated PR. `num_workers` rejected; relaxed completion-order iteration. | ✅ approved — issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) |
| `docs/superpowers/specs/2026-07-19-streaming-svar2-phase2-pr3-design.md` | SVAR2 Phase 2 **PR 3** (read↔reconstruct pipeline engine) — refines the phase-2 design's Lever 3: a `Svar2StreamEngine` Rust producer thread mirroring SVAR1's `Svar1StreamEngine`, moving the serial GIL-held glue off the critical path and overlapping it with reconstruct. Deterministic order kept (deviation from the parent spec's relaxed-order sketch — a single ordered producer needs no reordering barrier). | ✅ approved + **landed, gated off by measurement** — issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) |
| _TBD_ — issue [#279](https://github.com/mcvickerlab/GenVarLoader/issues/279) | Interval (BigWigs/Table) streaming + variant+interval mixed scheduler | ⬜ |
| `docs/superpowers/specs/2026-07-18-streaming-vcf-pgen-optimization-design.md` | Optimization pass over the landed VCF/PGEN backends (#276): deterministic counters, `--compare-dataset` harness arm, profile-gated levers (PGEN `var_start` narrowing, transpose counting-sort, reader-reuse) | 🚧 Phase 1 (counters + baseline profile) done, see below |

## Plans (spec A)

| Plan | Scope | Status |
|---|---|---|
| `docs/superpowers/plans/2026-07-15-streaming-dataset-svar1-walking-skeleton.md` | Walking skeleton: SVAR1 → haplotypes end-to-end, parity-verified (no double-buffer) | ✅ done — PR [#274](https://github.com/mcvickerlab/GenVarLoader/pull/274) |
| `docs/superpowers/plans/2026-07-16-streaming-svar1-window-engine.md` | **Re-scoped:** genoray ungated `svar1_query` → gvl window-granular SVAR1 reads + double-buffer engine + `to_iter` surface. Issue [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275). Spec: `2026-07-16-streaming-svar1-window-engine-design.md` | 🚧 Tasks 2-4 done; Task 5's generic `StreamBackend`/`run_windows` engine done; SVAR1 wiring (issue [#283](https://github.com/mcvickerlab/GenVarLoader/issues/283)) done — 8a (Rust engine) + 8b (Python wiring, `to_iter()` now overlaps producer I/O with consumer generation) both landed; cold-cache A-vs-C measured (producer-thread engine wins 1.46×, ships as default); [#296](https://github.com/mcvickerlab/GenVarLoader/issues/296) throughput-gate observability gap fixed (`b2c5af90`) |
| `docs/superpowers/plans/2026-07-17-streaming-vcf-pgen-backends.md` — issue [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) | VCF backend / PGEN backend. Spec: `docs/superpowers/specs/2026-07-17-streaming-vcf-pgen-backends-design.md` | ✅ **done — VCF + PGEN backends both landed, single draft PR [#299](https://github.com/mcvickerlab/GenVarLoader/pull/299) into `streaming`** (all 14 tasks). Shared `RecordStreamEngine` (generic detached-producer/consumer engine core) + `DenseChunk`→`geno_v_idxs` transpose + `VcfWindowFiller`/`PgenWindowFiller`. `gvl.StreamingDataset(variants="x.vcf[.gz]"\|"x.bcf"\|"x.pgen")` reads directly and reaches byte-identical haplotype parity with `gvl.write()` + `Dataset.open()[r,s]` (haplotypes-only, `jitter=0`) for both backends; PGEN is biallelic-only (multiallelic rejected loudly at construction). Includes a `vcfixture-bulk` cohort-size benchmark harness (Task 13). Docs updated (Task 14). Still in draft pending a final whole-branch review. |
| `docs/superpowers/plans/2026-07-17-streaming-svar2-backend.md` | SVAR2 backend, Phase 1 (parity: synchronous `Svar2Store` read + `.svar2` dispatch, byte-identical vs `gvl.write()`+`Dataset[r,s]`). | ✅ Phase 1 done — issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) |
| `docs/superpowers/plans/2026-07-18-streaming-svar2-phase2-pr1.md` | SVAR2 Phase 2 **PR 1** (fast synchronous path): PR 1a `parallel=False` streaming reconstruct + PR 1b GIL-free Rust `svar2_read_window` (genoray `find_ranges`, no rev bump). Spec: `2026-07-18-streaming-svar2-phase2-design.md`. PR 2 (super-batch parallel reconstruct) + PR 3 (gated relaxed-order pipeline) get their own plans. | ✅ PR 1 implemented (byte-parity preserved; full `tests/dataset tests/unit` sweep green) — draft PRs [#301](https://github.com/mcvickerlab/GenVarLoader/pull/301) (1a) → [#302](https://github.com/mcvickerlab/GenVarLoader/pull/302) (1b, stacked) into `streaming`. Issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278). PR 2/3 next. |
| `docs/superpowers/plans/2026-07-19-streaming-svar2-phase2-pr2.md` | SVAR2 Phase 2 **PR 2** (super-batch parallel reconstruct): recycled `Svar2ReconBuf` + `svar2_reconstruct_super_batch` FFI, Python super-batch drive (fill one coarse super-batch across cores via GIL-free rayon, drain `batch_size` slices), `should_parallelize`-gated. Deterministic order preserved (relaxed order is PR 3). Spec: `2026-07-18-streaming-svar2-phase2-design.md`. | 🚧 implemented, byte-parity preserved; core-util + #284 super-batch-flatness gates green. **Measured (2000×20000, 64×1000bp, 8 cores, `benchmarking/streaming/svar2_superbatch_sweep.py`, reconstruct-only best-of-3):** the reconstruct kernel is **memory-bandwidth-bound — only ~1.1–1.3× on 8 cores** (best speedup 1.31× at sb=16384; tiny super-batches ≤256 rows go *slower* parallel from rayon fork/join churn); best absolute reconstruct wall is at sb≈1024–4096 (bigger buffers hurt cache locality → serial time rises), and **end-to-end `to_iter` wall (~0.42 s) is dominated by the serial `read_window`/drive, not reconstruct** (cpu/wall ≤1.24 end-to-end). Default `SUPERBATCH_TARGET_ROWS=4096` kept — sits at the `should_parallelize` byte-gate boundary and near the wall optimum; no swept value clearly wins. **Implication: PR 2's multi-core lever works but its wall payoff is Amdahl-bounded by the serial read; overlapping read↔reconstruct (PR 3, relaxed-order pipeline) is the actual throughput lever.** Issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278). |
| `docs/superpowers/plans/2026-07-19-streaming-svar2-phase2-pr3.md` | SVAR2 Phase 2 **PR 3** (read↔reconstruct pipeline engine): `Svar2StreamEngine` Rust producer thread (`src/ffi/svar2_stream_engine.rs`) running the whole fill chain (`find_ranges` → gather/decode → super-batch reconstruct) GIL-free, ping-ponging two `Svar2ReconBuf`s through `bounded(2)` channels (mirrors SVAR1's `Svar1StreamEngine`); selected via the existing `_prefetch_strategy` seam as `"svar2_engine"`. Deterministic order preserved (no relaxed-order doc change). Spec: `2026-07-19-streaming-svar2-phase2-pr3-design.md`. | ✅ **implemented, gated off by measurement — default stays `"sync"`.** Byte-identical parity (`test_svar2_engine_matches_written`, `test_svar2_engine_matches_sync_bytewise`) and the #284 cohort-independence gate (`test_svar2_engine_output_is_flat_in_cohort_size`) both green under the engine. **Cold-cache A/B** (`benchmarking/streaming/svar2_cold_cache.py`, vcfixture bulk store, fresh store + `posix_fadvise(DONTNEED)` per (rep, strategy), best-of-3, `batch_size=32`, shared 8-core dev node): at n_samples=500, `sync` runs `[0.148, 0.138, 0.119]` (range `[0.119, 0.148]`) vs `svar2_engine` runs `[0.123, 0.110, 0.107]` (range `[0.107, 0.123]`); at n_samples=2000, `sync` runs `[0.613, 0.539, 0.596]` (range `[0.539, 0.613]`) vs `svar2_engine` runs `[0.381, 0.611, 0.345]` (range `[0.345, 0.611]`). **Ranges overlap at both cohort sizes** (engine's best-of-3 edges into/past sync's range each time, and at n=2000 the engine's own worst rep, 0.611s, is slower than sync's worst) — the SVAR1 Task-9 non-overlapping-ranges bar is not cleared, and n=2000 in particular reads as node noise dominating any real effect, not a stable win. Per the ship rule (`docs/superpowers/specs/2026-07-19-streaming-svar2-phase2-pr3-design.md` → "Gating"), **kept `_Svar2Backend._default_strategy = "sync"`** — a marginal/noisy result does not meet the bar, and the rule is to bias toward the safe, already-parity-anchored default on any doubt. The engine ships as off-by-default infrastructure (selectable via `_prefetch_strategy="svar2_engine"`), matching how SVAR1's Design C shipped off-by-default when it lost its own A-vs-C call — it remains available for future re-measurement (larger cohorts, a quieter node, or once #276's VCF/PGEN decode cost gives the GIL-free producer more to hide). Draft PR [#306](https://github.com/mcvickerlab/GenVarLoader/pull/306) (stacked on `278-svar2-phase2-pr2`); deferred non-blocking follow-ups [#307](https://github.com/mcvickerlab/GenVarLoader/issues/307) — **addressed** in the stacked PR-3 follow-up: `build_engine` now materializes reference bytes only for contigs a job touches (SVAR1 + SVAR2, empty placeholder for the rest, keeps the cohort-independent memory story on the reference axis); the SVAR2 producer's `phys` alloc is deferred to the find-ranges cache-miss path; and `svar2_fill_super_batch` uses `require_contiguous_1d` for the friendly `ref_` message. Byte-identical parity preserved. Issue [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278). |
| `docs/superpowers/plans/2026-07-19-streaming-output-mode-breadth-wave-a.md` — issue [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277) | Output-mode breadth, Wave A: `with_len` (int/ragged, no "variable"), read-time jitter (`with_settings(jitter=, rng=, deterministic=)`), `with_seqs("annotated")`. Spec: `docs/superpowers/specs/2026-07-19-streaming-output-mode-breadth-wave-a-design.md` | ✅ **done (Tasks 1-6 all landed)** — PR pending |
| `docs/superpowers/plans/2026-07-18-streaming-vcf-pgen-optimization.md` — issue [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) | Optimization pass. Spec: `docs/superpowers/specs/2026-07-18-streaming-vcf-pgen-optimization-design.md` | ✅ **done (Tasks 1-7 all landed)** — counters (Task 1), `--compare-dataset` arm (Task 2), baseline + profile (Task 3, `docs/roadmaps/streaming-optimization-baseline.md`), PGEN `var_start`/`var_end` narrowing (Task 4), word-level transpose (Task 5), VCF sample-index reuse (Task 6), docs/gate (Task 7, see results below) |

### Optimization pass (#276) results — Task 7 (2026-07-19)

Final measured before/after for the optimization pass, N=10000, same fixture/region-plan as
the Task-3 baseline (`--n-regions 200 --region-len 200 --batch-size 32`, `n_windows=4`).
Before numbers from `docs/roadmaps/streaming-optimization-baseline.md` (captured after Tasks
1-2, before any optimization edit); after numbers captured this task on current HEAD (Tasks
1-6 landed), same session, `pixi run -e dev maturin develop --release` immediately before
capture, via a standalone script mirroring the harness's own fixture helpers (not committed —
tmp scratch, per the task's constraints).

- **Streaming vs written `Dataset` (Task 3 baseline, N=10000):** streaming engine sweep
  5.2-11.5s vs written-`Dataset` sweep 1.2-1.4s post-preprocessing; `Dataset` write
  (preprocessing) itself costs 5.4-8.8s at this sweep scale — the whole point of
  `StreamingDataset` is skipping that cost, at a per-epoch throughput trade. Peak RSS: engine
  ~1.07M-1.35M KB (VCF), ~1.91M-2.0M KB (PGEN); sync/engine ratio 1.25-1.77x (engine's
  producer/consumer overlap wins, colored per the noisy-shared-node convention, not a gate).
- **`pgen_variants_decoded` (per-sweep total, 4 windows):** before 400,032 (= 4 × 100,008
  pvar variants — every window re-decoded the whole contig prefix from record 0) → **after
  98,583** (≈1x total variant count — each window now decodes only its own
  `var_start`/`var_end` range). ~4.06x reduction.
- **`transpose_word_reads` (per-sweep total):** before 17,140,000 (per-cell `get_bit`, =
  V×S×ploidy) → **after 535,628** (word-level two-pass counting-sort transpose, = 2×n_words,
  same counter redefined to a comparable word-reads unit in Task 5). ~32x reduction.
- **`vcf_sample_resolutions` (per-sweep total, 4 windows):** before ~n_windows (one
  O(n_samples) `HeaderView::sample_id` string lookup per window) → **after 1** (resolved once
  per sweep, reused across windows via genoray `with_sample_indices`).

**Cross-repo dependency:** this pass pins genoray branch `feat/vcf-sample-indices` (commit
`adf7e22e2e706ddc6d429baef92a96aff9e852e2`, v3.2.1+) in `Cargo.toml` for `with_sample_indices`
(Task 6). PR [#299](https://github.com/mcvickerlab/GenVarLoader/pull/299) (the VCF/PGEN
backends draft PR into `streaming`) is gated on that genoray branch merging to genoray `main`
before it can leave draft — an unpublished/unmerged genoray branch is never itself a blocker
(bump the rev), but #299 specifically should not be finalized until this rev is stable.

See `docs/superpowers/plans/2026-07-18-streaming-vcf-pgen-optimization.md` (full task list)
and `docs/roadmaps/streaming-optimization-baseline.md` (baseline + profile) for detail.

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
- ✅ **genoray prerequisite: ungated `svar1_query`** — `Svar1Reader` + `var_ranges` +
  cartesian `find_ranges` in genoray, ungated (memmap2 + bytemuck + the existing `search.rs`; no
  htslib). **Root cause of the skeleton's SVAR1 debt:** genoray has an ungated Rust *query* API
  for SVAR2 (`genoray_core::query`) and **none for SVAR1**, so the skeleton reached for the
  conversion-gated `Svar1RecordSource` — a *conversion-pipeline record producer*, not a query API.
  Stage A is nearly free (`search::overlap_range` already ports Python's `var_ranges`; nobody
  wired it to SVAR1); Stage B is two `partition_point`s per hap. Consumed via a `rev` bump — **not
  a release gate** (see Development Notes in CLAUDE.md). Folded into Task 2 below (rev
  `e07477e687c913f9605fc79ea251f1bb3b177aa9`) after Task 1 came back blocked on the same drifted
  `Svar1RecordSource` call site Task 2 deletes. _Plan 2, chunk 1_ —
  genoray issue [#123](https://github.com/d-laub/genoray/issues/123)
- 🚧 **SVAR1 window reads + double-buffer engine + `to_iter` surface** — crossbeam
  producer/consumer, generic `StreamBackend`. _Plan 2_ —
  issue [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275)
  - ✅ **Task 2 (read path): `Svar1Store` rewritten on `svar1_query`.** `read_window` is now two
    binary-search stages (`var_ranges` + `find_ranges`) inside `py.detach`, GIL-free;
    `geno_v_idxs` is `Svar1Reader::variant_idxs()` itself (zero copy). `Svar1RecordSource`,
    `ContigTable`, `set_contig_table`, and every `.tolist()`/per-batch clone are deleted, not
    optimized. Window is now the read granularity (`regions x samples x ploidy`, cartesian);
    `StreamingDataset._plan`/`_iter_batches` slice batches out of a window instead of driving
    pairwise reads. Byte-identical parity verified
    (`tests/dataset/test_streaming_parity.py::test_streaming_matches_written_all_cells`).
  - ✅ **Task 3 (`to_iter` surface): one iteration entry point.** `StreamingDataset` is now a
    plain frozen dataclass (no `__iter__`); `to_iter(batch_size, return_indices)` is the sole
    traversal method, with `n_batches`/`_iter_batch_spans` counting it without reconstructing.
    `to_torch_dataset()` no longer raises -- it wraps `to_iter` in an `IterableDataset` (matching
    `Dataset.to_torch_dataset`'s name/concept); `to_dataloader()` wraps `to_torch_dataset`.
    `__getitem__` still raises `TypeError` (iterable-only). `_batch_size`/`_with_batch_size`
    deleted -- `batch_size` is a `to_iter` argument, not instance state.
  - ✅🚧 **Task 5 (double-buffer engine): generic engine landed; SVAR1 wiring split out.**
    New `src/stream/mod.rs`: `StreamBackend` trait (`fn fill(&self, &WindowSpec, &mut
    Self::Buffer) -> anyhow::Result<()>`) + `run_windows(backend, windows, n_slots,
    consume)` — gvl's **first threading primitive** (previously zero `std::thread`/
    crossbeam/`unsafe impl Send+Sync` in `src/`). Producer/consumer over
    `crossbeam_channel::bounded` (added `crossbeam-channel = "0.5.15"`, matching
    genoray's pin), 2-slot ping-pong recycling (net-new vs. genoray's orchestrator,
    which allocates fresh and drops — an N-slot ring is deferred to profiling
    evidence), shutdown by `Sender` drop, join-everything-then-classify-panics. 5
    adversarial unit tests (plan-order yield, slot-cap recycling, producer-error,
    consumer-error, empty-plan), each run 20x in `--release` with zero hangs/failures.
    - **Compile-fix vs. the spec's literal listing:** the spec's `run_windows` body
      borrowed `tx_filled`/`rx_free` into a non-`move` `spawn_scoped` closure and then
      tried to move/drop them afterward — doesn't borrow-check. Fixed by moving both
      entirely into the producer closure (`move ||`), making it their sole owner; no
      clone of either is ever held back in the outer scope. This is a *stronger*
      version of the shutdown-safety property the spec asked for (orchestrator.rs's
      hazard is a stray `Sender` clone kept around for introspection — this design has
      no extra clone to forget to drop, so the hazard can't occur by construction).
    - **Step 5 (wire `StreamBackend` for `Svar1Store`, route
      `reconstruct_haplotypes_svar1` through `run_windows`) — SPLIT into a follow-up
      task, per the plan's own pre-authorized escape hatch.** Reasoning: today
      `reconstruct_haplotypes_svar1` is one Rust call per window, and Python's
      `_plan()`/`_iter_batches()` (`_dataset/_streaming.py`) drives the window loop
      *synchronously* — each `to_iter()` step blocks on one FFI call, gets one
      window's data back, slices it into batches, and only then asks for the next
      window. For the double-buffer engine to produce any real overlap, window N+1's
      fetch has to happen *while Python is busy with window N* — which means the
      producer thread must outlive a single FFI call and persist across `to_iter()`
      steps, with some Python-visible handle (a new pyclass wrapping a receiver, most
      likely) that `_iter_batches` polls instead of calling
      `reconstruct_haplotypes_svar1` directly. That's a new ownership/lifetime model
      (a background thread tied to the `StreamingDataset`/iterator's lifetime, not a
      single call), plus GIL re-acquisition to build `PyArray`s from data produced on
      the producer thread, plus deciding what `Svar1Store: StreamBackend`'s `Buffer`
      actually is (offsets-only via `read_window`, or the fully-reconstructed
      `(data, offsets)` output — the two halves currently live in one `py.detach`
      block in `src/ffi/mod.rs`, and splitting them changes what work each stage
      overlaps). Restructuring the FFI's window loop, `Svar1Store`'s ownership of
      per-contig arrays, and `_streaming.py`'s iteration protocol together, in one
      task, risked rushing exactly the parity-critical surface
      (`test_streaming_parity.py`, `test_scale_parity_still_byte_identical`) this
      whole effort is gated on. Tasks 2 + 4 (window-granular reads + the deterministic
      entries-touched gate) are the asymptotic fix and are already landed,
      independently valuable, and unaffected by this split — the double-buffer
      engine's benefit (overlapping producer I/O with consumer reconstruct) is a
      separate, separately-measured effect that this split defers rather than drops.
      Step 6 (cold-page-cache overlap measurement) is skipped for the same reason —
      it only makes sense once Step 5's wiring exists; no overlap number is reported
      here to avoid fabricating one.
    - **Follow-up: issue [#283](https://github.com/mcvickerlab/GenVarLoader/issues/283).**
      Wire `Svar1Store` as a `StreamBackend` and move `StreamingDataset`'s window loop
      into Rust behind a Python-visible iterator/handle, so `to_iter()` actually
      overlaps producer I/O with consumer reconstruct. Should re-decide the `Buffer`
      split (offsets-only vs. fully-reconstructed) as its first step, since that
      decision shapes the GIL-crossing design. Cold-page-cache overlap measurement
      (the deferred Step 6) belongs to this follow-up, not to #275.
    - ✅ **Task 8a (Rust engine): `Svar1StreamEngine` pyclass** — `src/ffi/stream_engine.rs`,
      producer/consumer overlap copying `run_windows`'s discipline (2-slot ping-pong
      `crossbeam_channel`, shutdown-by-`Sender`-drop, join-then-classify-panics), plus a
      per-contig-reference deviation from the original design note (byte-parity
      requirement for multi-contig plans — see the design note and task-8-report.md for
      the reasoning). Commit `814df7db`.
    - ✅ **Task 8b (Python wiring): `_streaming.py` drives the engine** — `_Svar1Backend`
      now caches per-contig meta (`_contig_meta`) and the store path (`_svar_path`) at
      construction, and gained `build_engine(jobs, batch_size)`; `_iter_batches`'s real
      (`self._backend is not None`) branch materializes `plan = list(self._plan())`
      once, builds one engine job per window from it (translating `s_idx` to physical
      sample indices the same way `read_window` does), and drives `engine.next_batch()`
      in lockstep with the SAME plan so job order and per-window batching can't diverge.
      `n_slots` bumped 1→2 (ping-pong residency). Byte-identical parity holds
      (`test_streaming_matches_written_all_cells`, `test_scale_parity_still_byte_identical`
      both green). Design note: `.superpowers/sdd/task-8b-design-note.md`; full report:
      `.superpowers/sdd/task-8-report.md`.
    - ✅ **Final-review Finding 1 fixed in-branch — engine job residency is now genuinely
      cohort-independent.** The first cut materialized `list(self._plan())` and one
      `WindowJob { phys_samples: Vec<usize> }` per window: `O(n_windows × window_samples)`
      ≈ cohort × regions sample-index metadata (~1–2 GB at 50k regions × 100k samples),
      resident in three places and unbounded by `max_mem` — eroding #284's guarantee, and
      the "cohort-independent" claim in `stream_engine.rs` was false. Because `_plan`
      always yields a CONTIGUOUS `arange(s_lo, s_hi)` sample chunk, the engine now holds
      the public→physical map (`phys_sample_idx`) ONCE (`O(n_samples)`) and each job
      carries only `(contig_idx, regions, s_lo, s_hi)` (region-scale); the producer
      borrows `&phys_sample_idx[s_lo..s_hi]` per window (zero-copy). Both drives
      (`_iter_batches` engine + readahead) stopped materializing `list(self._plan())`,
      building one compact region-scale plan instead. Total residency
      `O(n_windows × window_regions) + O(n_samples)`, never `O(n_windows × n_samples)`.
      Parity byte-identical under both strategies; 20× race-stable. Fix note:
      `.superpowers/sdd/task-8-fix2-note.md`; report: `.superpowers/sdd/task-8-report.md`.
    - ✅ **[#296](https://github.com/mcvickerlab/GenVarLoader/issues/296) fixed
      (commit `b2c5af90`).** The `#275` `svar1_csr_entries_touched()` throughput gate's
      `thread_local!` counter was blind to the engine's background producer thread
      (`read_window` now runs there), which failed
      `test_entries_touched_scales_with_window_not_store` **and** silently vacuum-passed
      `test_entries_touched_is_flat_across_batch_size` (0 == 0 == 0). Made the counter a
      process-wide `AtomicUsize`; both gates are meaningful again (146/3978 entries;
      `[472, 472, 472]` flat across batch_size).
    - ✅ **Task 9 (Design C + cold-cache A-vs-C measurement): SHIP DESIGN A.** Built
      Design C (single-thread read-ahead-one-window read-through prefetch, `svar1_prefetch_runs`
      FFI) behind an internal `_prefetch_strategy` toggle (`"engine"` default | `"readahead"`),
      byte-identical parity under both. Cold-cache (fresh-store-per-run, no root) A-vs-C on
      the shared node, best-of-3: **engine 0.286s vs readahead 0.417s — a 1.46× win for the
      producer-thread engine**, with non-overlapping run ranges (engine [0.286–0.335],
      readahead [0.417–0.432]) so it is well outside node noise. The readahead overlap proxy
      shows why: the single-thread read-through call is ~0.001s and cannot hide I/O (the
      fault cost is paid lazily inside `generate`), whereas the producer thread genuinely
      overlaps window N+1's read with window N's generation. Per the spec's decision rule
      (*A ≫ C ⇒ the producer thread pays for itself on I/O alone*), **Design A ships as the
      SVAR1 default** (already wired). **#276 implication:** the thread is justified for SVAR1
      I/O overlap directly (not merely reserved for #276), and it will additionally hide
      #276's decode CPU, which `madvise`/read-through cannot. Design C + the harness
      (`benchmarking/streaming/cold_cache_overlap.py`) are kept as off-by-default internal
      experimental infrastructure for #276 re-measurement. Harness/design:
      `.superpowers/sdd/task-9-design-note.md`; report: `.superpowers/sdd/task-9-report.md`.
  - ⚠️ **Inherited perf/scale debt from the walking skeleton — DELETED, not fixed, by Task 2.**
    All of it was downstream of the one wrong dependency above, so the rewrite removed it rather
    than optimizing it. (a) `Svar1RecordSource::new` was **O(all CSR entries)** — it eagerly
    inverted the contig's whole hap-major CSR — and the skeleton called it **per batch**; the real
    cost was far worse than the O(records × batch) walk alone. (b) The `.tolist()` per-contig
    table (~10M `int` objects for a human chr1) and `set_contig_table` existed **only** to feed
    that constructor → gone with it, not converted to `PyReadonlyArray1`. (c) The per-batch
    `t.pos.clone()` et al. existed only because the constructor took its vectors **by value** →
    gone with it, not fixed with slices/`Arc`. (d) The GIL-held walk (old `src/ffi/mod.rs:826`,
    *before* the `py.detach`) is fixed by the `Svar2Store` template — the reader now borrows across
    `py.detach` (`store: PyRef<'py, _>`).
  - ⚠️ The skeleton **conflated window and batch** (one Rust call per batch). That conflation is
    what produced both the per-batch contig walk and the apparent need for pairwise reads. Task 2
    fixes this: the window (`_window_regions`, default 64) is now the read granularity and a batch
    is a slice of it.
  - ✅ **Task 4 (deterministic scale gate): `svar1_csr_entries_touched()` counter +
    `tests/dataset/test_streaming_scale.py`.** A `thread_local!` `Cell<usize>` in
    `src/svar1/store.rs` (mirrors genoray's `search::search_tree_build_count`) sums
    `stop - start` over `read_window`'s `find_ranges` output on every call; exposed via
    `svar1_csr_entries_touched()` (`src/ffi/mod.rs` + `src/lib.rs`). **The gate is this
    counter, not wall-clock** — this node is too noisy for absolute timings (standing
    perf-gate convention). New 200-variant x 20-sample, single-contig scale fixture
    (`scale_fixture`) makes "touches the whole contig" vs "touches the window" differ by
    orders of magnitude, which the 40bp toy fixtures could not observe.
    - **Before (skeleton) — a reasoned argument, not a measured number:** the counter
      didn't exist pre-rewrite, so there is no direct old-code measurement to quote.
      Reasoned from the already-landed Task 2 description of what was deleted:
      `Svar1RecordSource::new` inverted the whole contig CSR *per batch*, so
      entries-touched *would have been* ≈100% of the store regardless of window —
      presented here as the motivating argument for the gate, not as a measured
      before/after pair.
    - **After (Task 2 rewrite, measured directly):** total CSR entries in the
      200-variant/20-sample scale store = 3978; a single narrow 1-region/100bp window
      touches 146 (3.7%, gate asserts `< 25%`) — proportional to the window's
      variants, not the store's.
    - **Batch-size invariance confirmed:** `svar1_csr_entries_touched()` delta is
      bit-identical across `batch_size` in `{1, 8, 64}` for the same window plan — a
      batch is a slice of the window, never its own read.
    - **Scale guard is a Rust unit test, not Python RSS:** the original `ru_maxrss`
      guard was flake-prone *and* blind — a `.to_vec()` regression on `geno_v_idxs`
      allocates only a few KB, far below `ru_maxrss`'s page-granularity high-water
      mark, so it could never fail on the defect it named. Replaced with
      `geno_v_idxs_borrows_the_mmap_not_a_copy` in `src/svar1/store.rs`, asserting
      pointer identity between the slice `reconstruct_haplotypes_svar1` hands the
      kernel and `Svar1Reader::variant_idxs()` — deterministic, fails immediately on
      an owned copy.
    - **`_window_regions` default — re-measured, correcting an invalid sweep:** the
      first pass swept `{1, 4, 16, 64, 256}` against the 20-region pytest
      `scale_fixture` and claimed wr=64 as "the knee" from wr=256 matching it exactly
      — but the fixture is only 20 regions, so any `window_regions >= 20` collapses
      the whole bed into **one window**; wr=64 and wr=256 were byte-for-byte the same
      execution path, and the "no further improvement" observation was arithmetic,
      not measurement. Re-swept `{1, 4, 16, 64, 256, 1024}` against a purpose-built
      2000-region/400kb-contig fixture where wr=256 and wr=1024 are genuinely
      unsaturated (8 and 2 windows respectively), best-of-3 per setting across 3
      independent sessions. `entries_touched` was **exactly flat** (40453) in every
      session at every setting — I/O is windowing-invariant, as designed. Wall-clock
      (**secondary color only, never the gate**) dropped sharply and monotonically
      from wr=1 through wr=64 (session-average best ≈0.84s → ≈0.19s, ~4.5x); wr=256
      and wr=1024 kept improving but only another ~5–11%, on the same order as this
      shared node's run-to-run noise (a single setting's 3 repeats can span up to
      ~2x). Honest read: this fixture does not resolve a hard knee above wr=64 — the
      steep early elbow is real and lands at 64, and anything past it is
      noise-level on this node. Kept **`_window_regions = 64`**: it captures
      essentially all the measured gain, and since a window is regions × *all*
      samples, a larger `window_regions` grows the per-call working set with no
      measured compensating benefit here — a pragmatic small default, not an invented
      knee.
    - **Dataclass default was inert — fixed:** `__init__` unconditionally
      `object.__setattr__`'d `_window_regions` to a hardcoded `64`, so the field's
      default (and its justification comment) never reached a constructed instance —
      editing the field default silently did nothing. Fixed by pulling the default
      from `type(self).__dataclass_fields__["_window_regions"].default` in `__init__`
      instead of duplicating the literal, making the field declaration the single
      source of truth.
    - **Unrelated bug found and fixed en route — split into its own commit:**
      `_Svar1Backend` indexed samples by the `.svar` store's native (VCF column)
      order, but `gvl.write()` always lexicographically sorts sample names
      (`_write.py`'s unconditional `samples.sort()`). The existing toy fixtures (≤3
      single-digit sample names) never exposed this because sort order and native
      order coincide there; the new 20-sample `"S0".."S19"` scale fixture does not
      (`"S10" < "S2"` lexicographically) and `test_scale_parity_still_byte_identical`
      caught it immediately. Fixed by sorting sample names at `_Svar1Backend`
      construction and translating every public (sorted-order) `sample_idx` to the
      store's physical column index (`_phys_sample_idx`) before it crosses into Rust
      — output row order is unaffected, only which physical column each row reads.
      This is a real, user-visible correctness fix, so it landed as its own
      `fix(streaming):` commit rather than folded into the `test:` commit that found
      it, so it gets a changelog entry.
    - ✅ **Memory-scaling concern — issue
      [#284](https://github.com/mcvickerlab/GenVarLoader/issues/284) — RESOLVED.**
      Originally: a window is regions × **all samples**, so peak memory scaled with
      cohort size and `_window_regions` alone couldn't bound it (the sample
      dimension was never chunked). Fixed by a 5-task plan (this branch,
      `spec/streaming-svar1-engine-memory`):
      - **Read/generate split:** `_Svar1Backend.read_window` (offsets only) is
        separated from `_Svar1Backend.generate_batch` (haplotype output for a
        `[lo:hi)` row slice). `StreamingDataset._iter_batches` reads a window's
        offsets once, then calls `generate_batch` once per `batch_size` slice —
        output is never materialized for a whole window.
      - **`max_mem` byte budget:** `StreamingDataset(..., max_mem="512MB")` (int
        bytes or a size string) is a new public constructor arg. `__init__` derives
        `_window_samples`/`_window_regions`/`_max_mem_bytes` from it (chunking both
        the region and the sample axis of the read window so the offsets buffer —
        `window_regions * window_samples * ploidy * 16 B` — stays within budget
        regardless of cohort size), keeping the pragmatic `region_target = 64`
        read-amortization knee as a local constant when the budget allows it.
      - **Per-batch generation:** `generate_batch(r_idx, s_idx, o_starts, o_stops,
        lo, hi)` allocates output for exactly `hi - lo` rows — parity-verified
        byte-identical against `gvl.write()` + `Dataset.open()[r, s]`
        (`test_scale_parity_still_byte_identical`).
      - **Deterministic cohort-scale gate (not `ru_maxrss`):**
        `test_generate_batch_output_is_flat_in_cohort_size`
        (`tests/dataset/test_streaming_scale.py`) proves a fixed-`batch_size=4`
        call's output byte count is **identical** between a 50-sample and a
        400-sample cohort (both produce the same non-zero byte count), while also
        proving the read window covers the *whole* cohort each time
        (`len(s_idx) == n_samples`) — so the flat output is evidence of per-batch
        generation, not just a small window. An earlier `ru_maxrss`-based version of
        this gate was replaced: it measured 0B at every cohort size up to 20000
        samples because the tiny fixture's output never crossed a page boundary, so
        it could not have failed on the #284 defect it named.
      - **`region_target` re-confirmed (Task 5, this session):** swept
        `window_regions ∈ {1, 4, 16, 64, 256}` against the Task 4 200-variant/
        20-sample/4000bp scale fixture with a 20-region bed, best-of-3,
        `batch_size=8`. Session-best wall-clock: wr=1 → 0.0140s, wr=4 → 0.0108s,
        wr=16 → 0.0087s, wr=64 → 0.0090s, wr=256 → 0.0095s — a clear early elbow by
        wr=16, then flat within this shared node's run-to-run noise (differences
        among 16/64/256 are ≤10%, comparable to a single setting's rep-to-rep
        spread). This 20-region bed also can't discriminate `window_regions ≥ 20`
        (any value that large collapses to one window), so it's a confirmation, not
        a re-derivation, of the prior 2000-region sweep that established wr=64 (see
        Task 4 above). **Kept `region_target = 64`** — consistent with, not
        contradicted by, this sweep.
      - Follow-up #283 (engine wiring) still doubles the resident window count
        (ping-pong buffering) on top of this; that interaction is unaffected by this
        fix and remains next.
    - **Final whole-branch review fix wave, before opening the PR:** (1) `fix:`
      `_Svar1Backend.reconstruct_window` looked up `ref_c_idx` by re-searching
      `self._ref.c_map.contigs` for the STORE's contig name, but
      `Reference.from_path` normalizes `c_map` to the FASTA's own naming style
      (UCSC vs Ensembl) — a store/FASTA pair using different styles raised a bare
      `ValueError`. `contig_idx` already indexes `self._contigs` in the same order
      `Reference.from_path` builds `offsets`, so the lookup was both redundant and
      wrong; replaced with a new shared `Reference._contig_slice(contig_idx)` (also
      now used by `Svar2Haps._ref_for_contig`), and added a mixed-naming-style
      regression test (`test_streaming_handles_mixed_contig_naming_style`) verified
      to fail pre-fix. Pre-existing (from the walking skeleton), not introduced by
      this plan. (2) `feat:` added `StreamingDataset.samples` (sorted sample names,
      matching `Dataset.samples`) and documented the `sample_idx` sorted-order
      convention next to the existing region-order docs (`faq.md`, `dataset.md`,
      the skill) — previously the convention from the Task 5 sample-order fix above
      was correct internally but invisible to users, who had no way to look up a
      `sample_idx`'s name without reaching for the store's native order (wrong) or
      re-deriving the sort themselves. (3) minor cleanups: dead `_bed` field
      removed, `_window_regions`' comment trimmed to its conclusion (the
      2000-region sweep fixture it cited isn't committed), and stale
      "contig-grouped batches" test comments updated to say "window" (assertions
      unchanged). See PR [#282](https://github.com/mcvickerlab/GenVarLoader/pull/282).
- ✅ **VCF backend / PGEN backend** — _Plan 3/4_ —
  issue [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) — **COMPLETE, PR
  [#299](https://github.com/mcvickerlab/GenVarLoader/pull/299) (draft, into `streaming`)**
  - ✅ **VCF backend (PR 1, Tasks 1-9) done.** Shared `RecordStreamEngine` (generic
    detached-producer/consumer engine core) + `DenseChunk`→`geno_v_idxs` transpose + VCF
    streaming backend. `gvl.StreamingDataset(variants="x.vcf[.gz]"|"x.bcf")` reads VCF/BCF
    directly and reaches byte-identical haplotype parity with `gvl.write()` +
    `Dataset.open()[r,s]` (haplotypes-only, `jitter=0`). htslib (via genoray's `conversion`
    feature) is now a hard runtime requirement of the package.
  - ✅ **PGEN backend (PR 2, Tasks 10-14) done.** Additive on PR 1's engine — a second
    `WindowFiller` + Python backend, landed on the same `spec/276-vcf-pgen` branch,
    extending the same PR (not a second PR). Tasks 10-12 (PgenWindowFiller,
    `_PgenBackend` wiring, table + end-to-end haplotype parity) byte-identical
    to `gvl.write` first-try; `gvl.StreamingDataset(variants="x.pgen")` is now public
    (biallelic PGEN only — multiallelic rejected loudly at `PgenWindowFiller`
    construction, before any decode). Task 14 (docs gate + PR update) done.
    - ✅ **Task 13 (benchmark harness): `benchmarking/streaming/{gen_fixtures.sh,
      bench_streaming.py}` + `gen-bench-vcf`/`gen-bench-pgen` pixi tasks.** Infra
      only — no perf claim shipped from this task (shared-node noise; see the
      `gvl-rust-perf-gate-shared-node-noise` memory). Fixtures come from
      `vcfixture bulk` (`cargo install vcfixture --features cli`, or
      `VCFIXTURE_RS_DIR=/path/to/vcfixture-rs` to build+run from a checkout) →
      BCF, converted to PGEN via `plink2 --bcf ... --make-pgen --allow-extra-chr
      --output-chr chrM` (same convention `tests/dataset/conftest.py`'s PGEN
      fixtures use, to keep `chr1`-style contig names instead of plink2's
      default un-prefixed human coding). Generated fixtures are gitignored
      (`benchmarking/streaming/.gitignore`).
      - **Methodology (per the perf skill + this project's standing perf-gate
        convention — read before trusting any number this harness prints):**
        - **Not gated on absolute wall-clock.** `bench_streaming.py` prints
          windows/s and items/s as secondary color (best/median-of-N,
          `--repeats`) only; the load-bearing signals are DETERMINISTIC
          counters — `n_windows`, `n_batches`, `n_rows` (asserted equal to
          `shape[0]*shape[1]` for both strategies, not just "didn't crash"),
          `bytes_emitted`, and `peak_rss_kb` (`resource.getrusage`).
        - **Engine vs. a forced-synchronous baseline**, the same A-vs-C shape
          Task 9's SVAR1 cold-cache harness used (`cold_cache_overlap.py`,
          #283/#296) — but constructed differently, since VCF/PGEN have no
          `read_window`/`generate_batch` split to drive by hand the way
          SVAR1's Design C does. "Sync" rebuilds a single-window
          `RecordStreamEngine` per plan step and fully drains it before the
          next window's engine is built, so no window's decode can overlap
          the previous window's consumption — no new Rust toggle needed.
        - **VCF and PGEN measured in separate sweeps**, never averaged — PGEN's
          decode is not GIL-free and has a different overlap characteristic
          than VCF's.
        - **Cold-page-cache overlap measured separately (`--cold`)**, same
          unprivileged "never-faulted-by-this-process" caveat as
          `cold_cache_overlap.py` (no `drop_caches` without root): copies the
          run's fixture files into a fresh, never-read tmp dir before each
          timed run, so producer-thread I/O overlap isn't conflated with an
          already-warm page cache from a prior repeat.
      - **Known PGEN limitation this benchmark is designed to surface (first
        optimization target for a follow-up, not fixed here):**
        `PgenWindowFiller` (`src/record_stream/pgen.rs`, "Coarse `var_start`"
        doc section) always decodes a contig from its first variant and
        re-scans the `.pvar` from byte 0 every window (no seek) — O(prefix)
        decode + O(var_start) line-skip PER WINDOW, not amortized. Since
        instrumenting the Rust decoder's actual re-scanned-variant count is
        out of scope for a benchmark-harness task, `bench_streaming.py` prints
        the analytically exact multiplier instead: `n_windows` IS the number
        of times the whole contig prefix is re-decoded, and `pvar_variants` is
        that prefix's size — their product is the repeated work VCF (tabix/
        CSI-seekable) does not pay. Follow-up: narrow `var_start` with
        `max_v_len` padding (or add `PvarReader` seek support in genoray), per
        the module doc.
      - **Smoke-validated**, not benchmarked at scale (this node is shared;
        large sweeps are out of scope for this task) — see task-13-report.md
        for the validation run and its printed counters.
- ✅ **Output-mode breadth Wave A + docs** — `with_len`, jitter, `with_seqs("annotated")`; docs
  folded in — issue [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277), plan
  `docs/superpowers/plans/2026-07-19-streaming-output-mode-breadth-wave-a.md`. `"variants"`/
  `"variant-windows"`/`min_af`/`max_af`/`var_fields` are **Wave B**, tracked separately as
  issue [#304](https://github.com/mcvickerlab/GenVarLoader/issues/304).
- ✅ **Per-variant global variant ids (Phase 2: PGEN + SVAR1) — issue
  [#305](https://github.com/mcvickerlab/GenVarLoader/issues/305).** Spec:
  `docs/superpowers/specs/2026-07-20-streaming-variants-output-wave-b-design.md`; plan:
  `docs/superpowers/plans/2026-07-20-streaming-global-variant-ids.md`. The scalar `var_base`
  (`var_base + local`) could only express a contiguous offset, which is wrong across a
  region-overlap gap or an interior-excluded variant. Fixed by carrying a per-variant
  dataset-global id on `DenseChunk`/`DecodedWindow.global_v_idxs`, gathered onto
  `annot_v_idxs` by `remap_annot_local_to_global` (genoray bump to
  `d-laub/genoray` rev `3a44e14`, [genoray PR #134](https://github.com/d-laub/genoray/pull/134),
  this branch). SVAR1 was already global (no-op); **PGEN now correct** via the `.pvar` row
  index, including narrowed-window and interior-exclusion gaps (regression-locked). **VCF
  remains window-local** — genoray's VCF reader hard-codes `global_idx = -1` — real VCF
  global ids are **Phase 3**, not yet landed. `var_base` itself has been fully retired
  (Task 2.5 close-out).
- 🚧 **Variants-output surface, Wave B — issue
  [#304](https://github.com/mcvickerlab/GenVarLoader/issues/304).** Plan:
  `docs/superpowers/plans/2026-07-21-streaming-variants-output-b0-b1.md`; backed by the reviewed
  design at `docs/superpowers/specs/2026-07-20-streaming-variants-output-wave-b-design.md`.
  Task 1 (**PR-B0**, region-overlap clip in the *written* `with_seqs("variants")` path, #202)
  ✅ **done** — `get_variants_flat` folded a region-overlap keep mask into the existing AF
  `keep`/`_compact_keep` block (matches `src/reconstruct/mod.rs`'s inclusion extent). Serves
  as the byte-identical parity oracle for **PR-B1** (streaming `with_seqs("variants")`).
  Task 2 (**PR-B1**, `RecordBackend::generate_variants` Rust core) ✅ **done** — window
  CSR → flat `VariantsBatch` (`alt`/`start`/`ilen`/`row_offsets`) via the shared
  `assemble_variants_window` helper (`src/variants/mod.rs`), region-overlap-clipped
  per `(row, ploid)` identically to `generate`'s haplotype path. Task 3 (**PR-B1**,
  `next_batch_variants` FFI + Python wiring + VCF/PGEN parity) ✅ **done** —
  `EngineBackend::generate_variants` (default-unsupported, `RecordBackend` overrides it;
  the shared `advance`/`NextSlice` cursor extracted out of `next_batch_core` so
  `next_batch_variants_core` reuses it without duplicating the producer/consumer loop),
  `RecordStreamEngine.next_batch_variants` marshals to a `dict[str, ndarray]`
  (`alt`/`alt_offsets`/`start`/`ilen`/`offsets`), and `StreamingDataset.with_seqs("variants")`
  packs a `RaggedVariants` from it (mirrors `_FlatAlleles.to_ragged`/`_Flat.to_ragged`).
  `tests/dataset/test_streaming_variants_parity.py` gates VCF + PGEN byte-identical against
  the corrected written oracle. SVAR1 wiring is **Task 4**, next.
  **Verification caveat:** with the currently-pinned genoray rev, the described PGEN
  "contig-scoped" leak could not be reproduced end-to-end through `gvl.write()` +
  `Dataset.open()` for disjoint narrow regions (write-time `pgen.var_idxs`/`.chunk` are
  already position-scoped) — the added tests (`tests/dataset/test_variants_region_clip.py`)
  are an invariant/regression guard rather than a literal pre-fix repro; the fix is still
  correct defense-in-depth and a verified no-op on all currently-passing scenarios. Follow-up
  filed for the still-unclipped `get_variants_flat(self, idx)` no-region call site
  (`_haps.py:676`, reached via `with_seqs("variants").with_tracks(...)`) — issue
  [#314](https://github.com/mcvickerlab/GenVarLoader/issues/314).

## Sequencing

**#275 is the keystone** — it defines the generic `StreamBackend` trait that every other backend
implements, and it rewrites the same `src/ffi/mod.rs` + `_dataset/_streaming.py` surface the other
tasks would build on. Land it first; work started against the skeleton's shape gets rewritten.

**#275 is itself gated on a genoray PR** (ungated `svar1_query`) — a cross-repo prerequisite, not
a release gate: merge it to genoray `main`, then bump gvl's `rev`. Start there.

| Wave | Work | Parallel? |
|---|---|---|
| **Now** | Spec B ([#278](https://github.com/mcvickerlab/GenVarLoader/issues/278)), spec C ([#279](https://github.com/mcvickerlab/GenVarLoader/issues/279)) — **writing only** | ✅ docs-only, no code conflict with #275 |
| **0** | **genoray: ungated `svar1_query`** ([genoray#123](https://github.com/d-laub/genoray/issues/123)) → merge to genoray `main` → bump gvl's `rev` | ⛔ cross-repo prerequisite — gates #275 |
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
