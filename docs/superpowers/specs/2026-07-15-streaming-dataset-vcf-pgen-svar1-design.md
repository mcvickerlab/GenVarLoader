# Design: write-free `StreamingDataset` — framework + VCF/PGEN/SVAR1 backend

**Date:** 2026-07-15
**Status:** design (approved to draft; pending spec review)
**Roadmap:** `docs/roadmaps/streaming-dataset.md`
**Effort:** first of a set — see [Scope & sibling specs](#scope--sibling-specs).

## Summary

Add `gvl.StreamingDataset`: a write-free, **iterable-only** dataset for inference. It reads
variants directly from source files, reconstructs haplotypes/variants on the fly using the
existing Rust reconstruction kernels, and iterates in a fixed, data-layout-optimal order — no
`gvl.write()`, no on-disk cache. It trades per-epoch throughput (generally slower than a written
`Dataset`) for zero preprocessing and zero disk footprint, which is ideal for one-shot inference.

This spec covers the **shared streaming framework** (Python surface, scheduler, the async
producer/consumer engine) and the **VCF/PGEN/SVAR1 variant backend**. The SVAR2 backend and
interval/track streaming are separate specs (below).

### Goals

- `gvl.StreamingDataset(regions, reference=None, variants=<vcf|pgen|svar>, ...)` reads variants
  directly and yields model-ready batches in layout-optimal (region-major) order.
- **Byte-identical** to `gvl.write()` + `Dataset.open()[r, s]` for the same inputs/settings
  (modulo jitter/rng) — reuse the existing reconstruction kernels; parity is the test oracle.
- Throughput-first Rust engine: a large, allocated-once double buffer decouples bulk variant
  decode/I/O from per-batch reconstruction, so each compressed block / mmap page is touched once.

### Non-goals

- **No map-style access.** No `__getitem__(r, s)`, no random access, no shuffling. Iteration
  order is forced by data layout.
- **No `gvl.write()` parity for `max_jitter` padding** — streaming reads exactly the requested
  window (+ jitter window when `jitter>0`); with `jitter=0` outputs match a written dataset
  exactly.
- Not a training-time augmentation surface (jitter is supported but off by default).

## Scope & sibling specs

The streaming effort is decomposed by data source so each piece is independently testable
against the parity oracle:

| Spec | Scope | Buffer / kernels |
|---|---|---|
| **This spec** | Shared framework + **VCF / PGEN / SVAR1** variant backend | SVAR1-style buffer → `reconstruct_haplotypes_from_sparse` |
| SVAR2 backend (separate) | SVAR2 variant backend behind the same framework | SVAR2-style buffer → `reconstruct_haplotypes_from_svar2_readbound` (flat `vk_*/dense_*/lut_*` channels) |
| Intervals + mixed (separate/later) | BigWigs/Table streaming (sample-major) + variant+interval mixed scheduler | live `count_intervals`/`_intervals_from_offsets` + realign kernels |

The framework (engine, scheduler, Python surface) is generic over a `StreamBackend` trait; the
SVAR2 spec adds a second backend implementation, not a second framework.

## Architecture

```
gvl.StreamingDataset  (torch IterableDataset, Python)
    │  with_seqs / with_tracks / with_len / with_settings  (shared with Dataset)
    │  __iter__  →  drives the Scheduler (region-major), yields index-carrying batches
    ▼
Rust streaming engine  (generic over StreamBackend)
    producer thread ──crossbeam bounded(N)──▶ consumer
      fill window buffer B                     reconstruct slice from buffer A
    (rayon inside stages: decode / gather / reconstruct)
    ▼
StreamBackend = Svar1WindowBackend  (this spec)
    producers (per source):
      VCF   → genoray_core VcfRecordSource   (htslib IndexedReader region fetch)  ┐
      PGEN  → genoray_core PgenRecordSource   (sequential batch refill)            ├─▶ sparse-encode
      SVAR1 → genoray_core Svar1RecordSource  (mmap, already variant-major sparse) ┘   into buffer
    buffer = SVAR1-style window (local static variant table + per-hap sparse idxs + CSR offsets)
    consumer → reconstruct::reconstruct_haplotypes_from_sparse  (existing kernel, unchanged)
```

Everything below the Python surface is new Rust **except** the reconstruction core
(`reconstruct_haplotypes_from_sparse`) and the reference reader (`get_reference`), which are
reused as-is.

## Python API

Constructor mirrors `gvl.write()`'s source parameters so the mental model transfers:

```python
sds = gvl.StreamingDataset(
    regions,              # BED path | polars.DataFrame | seqpro BED
    reference=None,       # path | gvl.Reference | genoray.Reference  (needed for haplotypes)
    variants=None,        # path: .vcf[.gz]/.bcf | .pgen | .svar   (this spec)
    jitter=0,             # default 0 (inference); >0 reads a jitter window
    deterministic=True, rc_neg=True, min_af=None, max_af=None, var_fields=None,
    *,
    window_regions=<auto>,   # regions per buffer window (engine tuning; auto from a byte budget)
    buffer_bytes=<default>,  # total buffer footprint (both ping-pong slots)
    iteration_order="auto",  # "auto" | "region"  (this spec is region-major only; see scheduler)
)
```

- At least one variant source is required in this spec (`tracks=` is the intervals spec).
- Output configured with the **same builder methods as `Dataset`**: `.with_seqs(kind)`
  (`reference`/`haplotypes`/`annotated`/`variants`), `.with_len(int)` (→ dense arrays via the
  existing padding path), `.with_settings(...)`. Each returns a new frozen `StreamingDataset`.
- `variants` with no `reference` → `variants` mode only (`RaggedVariants`), matching `Dataset`.
- Iteration: `for batch in sds:` or `sds.to_dataloader(batch_size=..., num_workers=0)`.
- `len(sds) == n_regions * n_samples` (progress bars / DataLoader length). **No `__getitem__`.**
- `to_torch_dataset()` (map-style) raises `TypeError` pointing to `to_dataloader`.

### Torch integration

`StreamingDataset` **is** a torch `IterableDataset` (contrast: `Dataset` wraps a map-style
`TorchDataset` in `_torch.py`). `to_dataloader` returns a `DataLoader(inner, batch_size=None)`;
the dataset yields pre-assembled batches (see [Yield contract](#yield-contract--indices)).
For `num_workers>1`, the scheduler shards its region-window plan into disjoint contiguous
blocks per worker (no duplication); v1 recommends `num_workers<=1` + internal rayon threading,
consistent with gvl's established stance.

## The SVAR1-style window buffer

A fixed-size, allocated-once buffer holding **one window of many adjacent regions'** worth of
variants in the layout `reconstruct_haplotypes_from_sparse` already consumes:

- **Local static variant table** (sub-linear in samples, grows with variants in the window):
  `v_starts: i32`, `ilens: i32`, `alt_alleles: u8 + alt_offsets: i64` (ragged ALT CSR),
  `ref_: u8 + ref_offsets: i64` (ragged REF CSR). This is the Rust analog of Python's
  `_HapsFfiStatic` (`_dataset/_haps.py`), but **window-local** rather than dataset-global.
- **Sparse genotype channel**: `geno_v_idxs: i32` (flat per-hap variant indices into the local
  table) + `geno_offsets` (CSR, `(2, ·)` starts/stops) + `geno_offset_idx` per `(row, ploid)`.

Reconstruction of a batch is a **slice** of this buffer handed to the existing kernel — no copy
of the store, no per-batch decode. The buffer is source-agnostic: each producer's only job is to
populate it for the next window.

**Sizing.** `window_regions` is chosen from `buffer_bytes` and a measured/estimated
bytes-per-region (variant density × samples × ploidy). Two slots (ping-pong); optionally an
N-slot ring if profiling shows a single producer can't stay ahead. Window ≫ batch so the
producer amortizes I/O across many consumer batches.

## The streaming engine (Rust)

A generic producer/consumer harness, generic over:

```rust
trait StreamBackend {
    type Buffer;                       // Svar1Window here
    fn fill(&mut self, window: &RegionWindow, slot: &mut Self::Buffer) -> Result<()>;
    fn reconstruct(&self, slot: &Self::Buffer, batch: &BatchSpec, out: &mut OutBuf) -> Result<()>;
}
```

- **Concurrency:** one `std::thread` **producer** + `crossbeam_channel::bounded(N)` handing
  filled buffer slots to the **consumer**; a return channel recycles drained slots (no
  per-window allocation). Producer fills slot B while the consumer reconstructs from slot A.
  `rayon` parallelizes within `fill` (decode/sparse-encode) and `reconstruct` (over
  `batch × ploidy`, reusing the kernel's existing `parallel` fan-out).
- **Backpressure:** the bounded channel + slot recycling caps memory at `N × slot_bytes`
  regardless of how far the producer would otherwise race ahead.
- **Why not tokio:** the workload is mmap + CPU-bound decode with no async-I/O await points;
  threads + a bounded channel is the correct primitive and mirrors genoray's existing
  `orchestrator.rs`/`executor.rs` conversion pipeline. tokio is reserved for a possible future
  remote/object-store streaming extension.
- **Safety:** strong typing makes buffer ownership explicit — a slot is either owned by the
  producer (filling) or the consumer (reconstructing), never both; the channel transfers
  ownership, so there are no torn/stale reads by construction.

## Per-source producers (VCF/PGEN/SVAR1)

Enable `genoray_core`'s `conversion` feature (currently off; gvl links
`default-features = false`). This pulls **rust-htslib** (C htslib) + zstd into gvl's build. Each
producer drives a `genoray_core` `RecordSource` over the current region window and sparse-encodes
into the buffer:

- **VCF/BCF** → `VcfRecordSource` (htslib `IndexedReader` region fetch, `next_record`). Region
  windows fetched in genomic order; sparse-encode all samples per record into the window buffer.
- **PGEN** → `PgenRecordSource` (sequential 32 MB batch refill) + `PvarReader` for records.
  Naturally sequential; region-major sweep aligns with its refill order.
- **SVAR1** → `Svar1RecordSource` (mmap, variant-major sparse). Cheapest — already the target
  representation; window population is a bounded slice + local-table rebuild.

A small amount of new Rust bridges `RawRecord`/`RecordSource` output into the `Svar1Window`
buffer (genotype → sparse per-hap indices; POS/REF/ALT/ilen → local static table). Variant
filtering (`min_af`/`max_af`, `var_fields`, exonic) is applied during fill to match the written
path.

**Precondition (unchanged from `gvl.write`):** variants must be normalized (left-aligned,
biallelic, atomized) and free of symbolic/breakend ALTs — a user precondition enforced by
validation, not fixed up here.

## Scheduler / iteration order

This spec is **region-major only** (variants). Regions are sorted `(contig, start)`; the
scheduler emits region windows in genomic order, each window sized to the buffer. Within a
window every sample is reconstructed (variant stores return all samples per range read for free),
so the effective item order is region-major, sample-inner. This is the single sequential pass
that makes a position-sorted store optimal.

`iteration_order` accepts `"auto"`/`"region"` here; `"sample"` and the mixed-source heuristics
arrive with the intervals spec. The parameter exists now so the hardware-dependent optimum has a
stable override point later.

## Yield contract & indices

Because plan order is not the natural `(region, sample)` enumeration, **every batch carries its
indices**:

- The dataset yields **pre-assembled batches** in plan order — `batch_size` consecutive items,
  last partial kept unless `drop_last`. Reconstruction stays vectorized (one kernel call per
  slice); batches may span window boundaries by accumulating items.
- Each batch is `(*data, region_idx, sample_idx)`; `return_indices` defaults to **True** (order
  is scrambled relative to input, so indices are essential to map outputs back).
- `data` is the same type a written `Dataset` returns for the configured mode
  (`Ragged`/`RaggedVariants`/`AnnotatedHaps`/dense via `with_len`).

## Reconstruction reuse & parity

- Consumer calls `reconstruct::reconstruct_haplotypes_from_sparse` (and the annotated/variants
  variants of it) unchanged; the reconstructor classes (`Ref`/`Haps`) and output types
  (`Ragged`/`RaggedVariants`/`AnnotatedHaps`/`_Flat*`) are reused. Reference bytes come from the
  existing `get_reference` kernel over a live `genoray.Reference` (mmap, per-contig cache).
- **Parity oracle:** build a small written dataset from the same VCF/PGEN/SVAR1 + reference;
  assert every streamed item equals `Dataset.open(...)[r, s]` (mapped back by emitted indices),
  across `reference`/`haplotypes`/`annotated`/`variants` modes and all three backends, with
  `jitter=0`. This is the same byte-identical contract the Rust migration used.

## Build / packaging & release sequencing (risks)

- **htslib in gvl wheels (primary risk).** Enabling `conversion` adds a C dependency (rust-htslib
  → htslib) that must keep the abi3 wheel matrix (py310–313 × linux/macOS, +windows if targeted)
  green. This is a first-class build-system task, not an afterthought: verify cross-compilation,
  static-vs-dynamic htslib linking, and wheel size before committing to the backend work. If the
  wheel matrix proves untenable, the fallback is to gate all three RecordSource backends behind an
  optional feature/extra. Note a truly htslib-free SVAR1 default is **not** free: genoray's SVAR1
  reader is itself `conversion`-gated, so it would require a bespoke gvl-side ungated mmap reader
  mirroring the SVAR2 `ContigReader` — extra work justified only if the wheel matrix forces it.
- **genoray release-gate.** The SVAR2 read-bound path (rust-migration Phase 6a, PR #266) is
  already dev-wired to a local genoray checkout; PyPI/crates.io genoray does not yet publish the
  query/conversion API this effort needs. This spec cannot ship until genoray publishes; track
  the same release-gate checklist (Cargo path-deps → published versions; `pixi.toml`/
  `pyproject.toml` genoray pins).

## Testing strategy

- **Parity** (primary): byte-identical vs. written `Dataset` across modes/backends (above), via a
  differential harness mirroring `tests/parity/`.
- **Engine unit tests** (Rust `cargo test`): buffer fill/recycle correctness; producer/consumer
  ownership handoff; window boundary vs. batch boundary; N-slot backpressure caps memory.
- **Scheduler tests**: region-major order + index correctness; multi-contig windows; worker
  sharding disjointness.
- **Fixtures**: `vcfixture` for VCF ground truth; small PGEN/SVAR1 built via genoray.
- **Scale guard**: assert no per-batch materialization of a sample-scale array (mirrors the
  rust-migration scale-guard defense).

## Deferred / open questions

- **Optimal `window_regions`/`buffer_bytes` defaults** — measure on a realistic corpus; expose
  the knobs, ship a sensible default, don't hardcode a guess.
- **N-slot ring vs. 2-slot ping-pong** — start with 2; promote to N only if a single producer
  can't stay ahead of the consumer in profiling.
- **Multi-threaded producer** (parallel window decode across contigs) — deferred; single producer
  + rayon-within-fill first.
- **Remote/object-store streaming** (S3/GCS) — the one place tokio/async would earn its keep;
  explicitly out of scope, noted as a future extension.

## Implementation chunks (for planning)

Parallelizable once the framework skeleton lands:

> **Correction (2026-07-15, from interface recon).** genoray's `Svar1RecordSource` and its
> `RecordSource`/`RawRecord` trait are `#[cfg(feature="conversion")]`-gated, and
> `conversion = ["dep:rust-htslib","dep:zstd"]`. gvl links `genoray_core` with
> `default-features=false`, so **none** of the three RecordSource readers compile in today.
> Enabling `conversion` (htslib) is therefore a **shared prerequisite for all three backends,
> including SVAR1** — it moves ahead of the backends. Reusing genoray's gated SVAR1 cursor (vs.
> writing a bespoke ungated gvl-side mmap reader) is the YAGNI choice since htslib is enabled
> anyway. `Svar1RecordSource` is a forward-only cursor (biallelic-collapsed `gt`), which suits a
> region-major sweep; `RawRecord` lacks ILEN (derive `alts[0].len()-ref.len()`). The pure-mmap
> `Svar2Store` pyclass is the wiring template. `IterableDataset` precedents already exist in
> `_buffered_loader.py`/`_double_buffered_loader.py` — reuse them.

1. **htslib / `conversion` enablement + wheel matrix** — add `features=["conversion"]` to the
   `genoray_core` dep; prove the abi3 wheel matrix (py310–313 × linux/macOS) still builds; decide
   default-vs-optional-feature gating. Shared prerequisite; front-loaded to retire the build risk.
2. **Framework skeleton** — Python `StreamingDataset` (IterableDataset, `with_*`, `to_dataloader`,
   `len`, index-carrying batches) + region-major scheduler + the generic `StreamBackend`
   producer/consumer engine (crossbeam, slot recycling). Reuses `_buffered_loader.py` patterns.
3. **SVAR1 backend** — `Svar1Window` buffer + `Svar1RecordSource` producer + consumer wiring to
   `reconstruct_haplotypes_from_sparse`. First backend to reach parity (simplest: mmap, already
   sparse, biallelic).
4. **VCF backend** — `VcfRecordSource` producer + sparse-encode; parity via `vcfixture`.
5. **PGEN backend** — `PgenRecordSource`/`PvarReader` producer + sparse-encode; parity.
6. **Output modes + settings** — annotated/variants modes, `with_len`, `min_af`/`max_af`,
   `var_fields`, `rc_neg`, jitter window; parity per mode.
7. **Docs/skill** — `__all__` + `api.md` entry + `SKILL.md` section + prose docs (dataset/faq),
   documenting the write-free workflow, ordering, and perf tradeoffs.

The first working vertical slice (walking skeleton) is chunks 1→2→3 restricted to haplotype
output over `.svar`; that is the subject of the first implementation plan.
