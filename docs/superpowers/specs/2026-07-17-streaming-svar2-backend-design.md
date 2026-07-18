# Streaming SVAR2 backend — design

**Issue:** [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) — spec B of the write-free
`StreamingDataset` effort. **Target branch:** `streaming`. **Roadmap:**
`docs/roadmaps/streaming-dataset.md` (fills the `_TBD_` SVAR2 row in the Specs table).

## Goal & scope

A second `StreamBackend` behind the now-landed streaming framework (#275 engine, #283 engine
wiring, #284 `max_mem`), reading `.svar2` stores **write-free** and reconstructing contiguous
haplotypes on the fly, iterating in a fixed cartesian BED × samples sweep.

**Correctness oracle:** byte-identical parity with `gvl.write()` + `Dataset.open()[r, s]` under
`.with_seqs("haplotypes")` (modulo jitter/rng, which is out of scope here).

**No genoray rev bump.** Unlike SVAR1 — which needed a new ungated `svar1_query` surface — the
SVAR2 read-bound query surface (`genoray_core::query::{ContigReader, HapRanges,
gather_haps_readbound}`) is *already* linked and used by the written-`Dataset` path at the current
pinned rev (`Cargo.toml` `e07477e6…`). This spec adds no external dependency.

### Non-goals (deferred, unchanged)

- **Splicing** (GTF-driven spliced reconstruction, scatter-write `..._readbound_into`,
  `build_splice_plan`/`SplicePlan`). SVAR2 gained *written-path* spliced support in PR #286/#289,
  but the SVAR1 streaming backend does not do splicing and this backend mirrors it. Deferred.
- **Other output modes:** annotated/variants, `with_len`, `min_af`/`max_af`, `var_fields`, jitter —
  all → issue [#277](https://github.com/mcvickerlab/GenVarLoader/issues/277) (output-mode breadth),
  which is orthogonal to backends (kernel/output dispatch, not buffers).
- **Interval/track streaming** → issue [#279](https://github.com/mcvickerlab/GenVarLoader/issues/279).

## Architecture — mirror SVAR1 one format down, reuse the kernel

The design is a structural clone of the landed SVAR1 backend, swapping the format-specific read
and window buffer while **reusing the existing SVAR2 read-bound reconstruction kernel untouched**.

| Layer | SVAR1 (exists) | SVAR2 (this spec) |
|---|---|---|
| Rust store | `Svar1Store.read_window` → offsets (`src/svar1/store.rs`) | `Svar2Store.read_window` → **range arrays computed live** (`src/svar2/store.rs`) |
| Rust engine | `Svar1StreamEngine` producer/consumer (`src/ffi/stream_engine.rs`) | `Svar2StreamEngine` — copy the discipline |
| Kernel | `generate_batch_core` → `reconstruct_haplotypes_from_sparse` | **existing** `reconstruct_haplotypes_from_svar2_readbound` + `split_to_flat` + `hap_diffs_svar2` (`src/ffi/mod.rs`, `src/svar2/mod.rs`, `src/reconstruct/mod.rs`) |
| Python backend | `_Svar1Backend` (`_dataset/_streaming.py`) | `_Svar2Backend` — mirror |
| Dispatch | `.svar` suffix | `.svar2` suffix (replace the `NotImplementedError`) |

**No SVAR2 → SVAR1 conversion** (roadmap invariant: two buffer styles, do not cross-convert).

### The one real gap to fill: live range computation

The *written* path slices a precomputed on-disk cache (`_Svar2Cache`, the memmapped
`genotypes/svar2_ranges/` int64 arrays) for the per-window range arrays
`vk_snp_range` / `vk_indel_range` `(n_hap, 2)` and `dense_snp_range` / `dense_indel_range`
`(n_region, 2)`. The streaming path has **no cache**, so `Svar2Store.read_window` must compute
those ranges **live** — the Rust analog of Python's `SparseVar2._find_ranges`, a query over the
store's index. Everything downstream of the range arrays
(`HapRanges::new` → `gather_haps_readbound` → `split_to_flat` → `hap_diffs_svar2` → the
`reconstruct_haplotypes_from_svar2` kernel) is **shared, unmodified code**.

## The window buffer and read/generate split (key design decision)

**Decision: gather on the producer thread into an owned, ping-pong-recycled window buffer that
holds the flat `vk_*/dense_*/lut_*` channels.** The consumer runs sizing (`hap_diffs_svar2`) + the
reconstruct kernel per `batch_size` **row slice**, so output stays per-batch bounded (#284).

Rationale:

- The SVAR2 I/O is `gather_haps_readbound` — it page-faults the var_key/dense bytes for the
  window's ranges. To overlap that I/O with reconstruction, the *gather* must run on the producer
  thread. Gathering into an owned window buffer is the way to do that using **only the existing
  query surface** (no new genoray API, no rev bump).
- This is literally the roadmap's "SVAR2-style buffer (flat `vk_*/dense_*/lut_*` channels, keys
  decoded inline)": the buffer holds `split_to_flat`'s output.
- **Contrast with SVAR1**, whose window buffer is just offsets and whose producer merely
  *prefetches* pages (`prefetch_runs_core`) so the consumer reads the mmap zero-copy. SVAR2 has no
  cheap byte-range prefetch helper on the query surface, so the SVAR1-shaped "ranges-only buffer +
  prefetch" is not available without a new genoray API. That variant is a possible later
  optimization if profiling shows the resident buffer matters; **out of scope now.**

**Memory (#284) holds:** the window-resident buffer is larger than SVAR1's (gathered channels, not
offsets), but it is still `O(window)` and bounded by `max_mem`'s sample-axis chunking. The consumer
allocates haplotype output for exactly `hi − lo` rows per batch, so output never materializes for a
whole window — the #284 guarantee is preserved.

### Buffer lifecycle (mirrors `Svar1StreamEngine`)

- Two-slot `crossbeam_channel::bounded(2)` ping-pong (`filled` / `free`), prefilled with two
  default buffers; the free-slot pool is the sole backpressure. Shutdown by `Sender` drop;
  join-producer-then-classify-panics. Producer thread named `gvl-svar2-stream-producer`.
- Producer per window: compute range arrays live (`read_window`) → `gather_haps_readbound` →
  `split_to_flat` into the recycled slot → send.
- Consumer per batch slice `[lo, hi)`: `hap_diffs_svar2` sizing → prefix-sum offsets →
  `reconstruct_haplotypes_from_svar2` into `(hi − lo)`-bounded output → wrap as
  `Ragged.from_offsets(..., "S1", ...)`.
- GIL crossing: the blocking recv/generate/join body runs inside `py.detach`; the GIL is
  reacquired only to build the output `PyArray`s (copying `Svar1StreamEngine::next_batch`).

## `max_mem` sizing for SVAR2

Reuse the existing region + sample chunking machinery (`StreamingDataset.__init__` derives
`_window_regions` / `_window_samples` / `_max_mem_bytes`). SVAR1 sizes the window on a fixed
`cell_bytes = ploidy * 16` (two int64 offsets). **SVAR2 sizes the window on the deterministic,
fixed-size range-array footprint** — `vk_snp_range` + `vk_indel_range` ≈ `ploidy * 32 B` per cell,
dense ranges per region — matching how the written path's `_Svar2Cache` is dimensioned.

`max_mem` is therefore **approximate** for SVAR2: the gathered flat channels add a bounded,
window-proportional amount on top of the range footprint that cannot be sized exactly a priori
(entries per hap are data-dependent). Document this. The region/sample chunk knobs, the
`region_target = 64` read-amortization default, and the two-slot ping-pong are unchanged from
SVAR1.

## Python wiring + a small boy-scout refactor

- **`StreamingDataset.__init__`**: the `.svar2` branch currently raises `NotImplementedError`
  (`_dataset/_streaming.py`); replace it with construction of `_Svar2Backend`.
- **`_Svar2Backend`** mirrors `_Svar1Backend`: opens `SparseVar2`; sorts sample names
  lexicographically and builds `_phys_sample_idx` (the same public→physical convention that keeps
  parity with `gvl.write()`'s unconditional sample sort); caches per-contig meta and the global
  static variant table; exposes `read_window` / `generate_batch` / `build_engine`. Sample-index
  translation to physical store columns happens Python-side before crossing into Rust, exactly as
  in `_Svar1Backend`.
- **Boy-scout refactor:** `_backend` is currently a bare `_Svar1Backend | None` and `_iter_batches`
  branches on `is not None`. Introduce a minimal Python `StreamBackend` `Protocol`
  (`read_window`, `generate_batch`, `build_engine`) so `_iter_batches` is polymorphic over both
  backends instead of type-branching. Small, in-scope, and the natural seam now that a second
  backend exists. `StreamingDataset._backend` is widened to `StreamBackend | None`.

## Parity & scale gates (tests)

Mirror the SVAR1 test pattern (`tests/dataset/test_streaming_parity.py`,
`tests/dataset/test_streaming_scale.py`) with SVAR2 fixtures.

- **Parity** (`test_streaming_parity_svar2.py`, or SVAR2 cases alongside the existing module): a
  `.svar2` store + a pre-written `gvl.Dataset` oracle; assert byte-identical
  `data[i][h] == written[r, s][h]` across an **unsorted, interleaved multi-contig bed** driven
  through a real `DataLoader`, parametrized over `prefetch_strategy` where applicable. Include the
  mixed-contig-naming-style and non-lexicographic-sample-order regressions that the SVAR1 suite
  already carries.
- **Scale gate** (SVAR2 analog of the #284 gate): a fixed-`batch_size` call's **output byte count
  is identical between a small and a large cohort** — evidence of per-batch generation, not
  whole-window materialization — while the read window still covers the whole cohort
  (`len(s_idx) == n_samples`). SVAR2 has no direct `csr_entries_touched` analog; this
  output-flatness gate is the meaningful deterministic one. *If* live `find_ranges` can expose a
  cheap touched-entries counter, add a #275-style windowing-invariance gate too; otherwise the
  output-flatness gate stands alone.
- New SVAR2 fixtures in `tests/dataset/conftest.py` mirroring `svar1_multicontig_fixture` /
  `svar1_mixed_naming_fixture` / `svar1_sample_order_fixture`, each carrying `.bed`,
  `.reference_path`, `.svar2_path`, and a pre-written `.dataset_path` oracle.

## Implementation sequencing (for the plan, not this spec)

Two phases within #278, **parity locked before threading**:

1. **Synchronous path:** `Svar2Store.read_window` (live ranges) → `gather` → kernel, wired through
   the existing synchronous test-seam / `readahead` iteration branch, until byte-identical parity
   holds across the multi-contig fixture.
2. **Engine:** wire `Svar2StreamEngine` (producer/consumer overlap) and re-verify parity + the
   scale gate.

Parity — the whole effort's gate — lands before the concurrency surface, exactly as the SVAR1 work
did (walking skeleton → engine).

## Deliverables / definition of done

1. This spec, approved, filling the `_TBD_` SVAR2 row in the roadmap Specs table (with status +
   PR pointers), and tracked on the StreamingDataset project board with a `streaming:`-prefixed
   issue reference.
2. `Svar2Store.read_window` (live ranges) + `Svar2StreamEngine` implemented behind the framework.
3. `_Svar2Backend` + `StreamBackend` Protocol + `.svar2` dispatch in `_streaming.py`.
4. Byte-identical parity vs `gvl.write()` + `Dataset[r, s]` and the #284-style scale gate, green.
5. Docs audit: `StreamingDataset`'s supported-format list (README, `docs/source/*.md`, the
   `genvarloader` skill) updated to include `.svar2`.

## Pointers

- **SVAR1 backend template:** `src/ffi/stream_engine.rs` (`Svar1StreamEngine`), `src/svar1/store.rs`
  (`Svar1Store.read_window`), `_dataset/_streaming.py` (`_Svar1Backend`, `StreamingDataset`).
- **SVAR2 read-bound kernel (reused):** `reconstruct_haplotypes_from_svar2_readbound`
  (`src/ffi/mod.rs`), `split_to_flat` / `FlatChannels` / `hap_diffs_svar2` (`src/svar2/mod.rs`),
  `reconstruct_haplotypes_from_svar2` (`src/reconstruct/mod.rs`).
- **Written-path SVAR2 buffer (the cache the streaming path replaces with live queries):**
  `Svar2Haps` / `_Svar2Cache` / `_gather_inputs` (`_dataset/_svar2_haps.py`).
- **Generic framework (future convergence target):** `StreamBackend` trait + `run_windows`
  (`src/stream/mod.rs`). Note the landed SVAR1 engine reimplements the protocol inline rather than
  using this trait; the SVAR2 engine mirrors the SVAR1 engine, so it too is inline for now.
- **Parity contract & migration conventions:** `docs/archive/roadmaps/rust-migration.md`
  (Phase 6a is the SVAR2 read-bound precedent).
