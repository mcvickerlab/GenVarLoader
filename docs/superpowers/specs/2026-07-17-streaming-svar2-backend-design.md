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

**Default design (to be confirmed by measurement, see "Performance" below): gather on the producer
thread into an owned, ping-pong-recycled window buffer that holds the flat `vk_*/dense_*/lut_*`
channels.** The consumer runs sizing (`hap_diffs_svar2`) + the reconstruct kernel per `batch_size`
**row slice**, so output stays per-batch bounded (#284).

Rationale:

- The SVAR2 window fill is `gather_haps_readbound` (page-faults the var_key/dense bytes for the
  window's ranges) followed by `split_to_flat` (key-decode + merge). To overlap that work with the
  consumer's reconstruction, the fill must run on the producer thread. Gathering into an owned
  window buffer does that using **only the existing query surface** (no new genoray API, no rev
  bump).
- This is literally the roadmap's "SVAR2-style buffer (flat `vk_*/dense_*/lut_*` channels, keys
  decoded inline)": the buffer holds `split_to_flat`'s output.
- **Contrast with SVAR1**, whose window buffer is just offsets and whose producer merely
  *prefetches* pages (`prefetch_runs_core`) so the consumer reads the mmap zero-copy. SVAR2 has no
  cheap byte-range prefetch helper on the query surface, so the SVAR1-shaped "ranges-only buffer +
  prefetch" is not available without a new genoray API. That variant is a possible later
  optimization if profiling shows the resident buffer matters; **out of scope now.**

**This is a hypothesis, not a settled fact — it must clear the measurement gate in the Performance
section before the engine ships.** The producer thread pays for itself only if the window fill
actually overlaps the consumer; whether it does depends on the IO-vs-CPU bound (below), which is
measured, not assumed. The synchronous path (phase 1 of the sequencing) is the fallback if the
gate does not clear.

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

The `ploidy * 32 B` per-cell figure is a **starting estimate, not a measured constant** — the plan
must validate it (measure actual resident bytes vs the derived window at a few cohort sizes) rather
than ship the guess. `region_target = 64` is inherited from SVAR1's sweep and must be re-confirmed
for SVAR2's heavier fill (its read-amortization knee may sit elsewhere), not assumed identical.

## Performance: characterization & measurement plan

Correctness parity is the hard gate; throughput is a secondary, **measured** dimension. Following
the performant-py-rust discipline — every optimization is a hypothesis, a benchmark is the only
thing that confirms it — the spec commits to the following rather than to asserted speedups.

**Phase 0 — target & evidence.** There is no fixed latency budget; the target is *relative*,
inherited from the roadmap: the streaming backend is expected to be slower per epoch than a written
`Dataset` but avoid all preprocessing. The one performance decision this spec must make with
evidence is **whether the producer-thread engine earns its complexity over the synchronous path**.
Decision rule (copied from SVAR1's #283 Design-A-vs-C gate): ship the engine only if it measurably
beats the synchronous/readahead path on a cold page cache, outside this shared node's run-to-run
noise. Otherwise ship the synchronous path and stop.

**Phase 1 — dimensions & bound.** Do not fabricate sizes; confirm the ranges against a real SVAR2
cohort before the sweep.

| dimension | typical | max | grows? | notes |
|---|---|---|---|---|
| n_samples (cohort) | 100s–1000s | 100k+ | **grows unbounded** | chunked by `max_mem` (sample axis); the memory-scaling axis |
| n_regions / window | 64 (`region_target`) | user bed size | fixed knob | read-amortization granularity |
| ploidy | 2 | small | fixed | multiplies hap count |
| variants per window | data-dependent | high in dense regions | grows with window | drives gather + `split_to_flat` cost and buffer size |
| var_key / dense entries per hap | data-dependent | — | — | the CPU cost of key-decode/merge |

**The bound is the open question that picks the lever, and it must be measured, not assumed.** The
window fill is a mix of IO (page-faulting var_key/dense bytes on a cold cache — `gather`) and CPU
(key-decode + merge — `split_to_flat`). SVAR2's fill is **CPU-heavier than SVAR1's**, which had no
decode at all. Consequences:

- If fill is **IO-bound**, the producer thread (concurrency, overlap of window N+1's faults with
  window N's reconstruct) is the right lever — the SVAR1 result carries over.
- If fill is **CPU-bound**, a single producer thread caps overlap at ~2× and the better lever may
  be **rayon *within* the gather/kernel** (data parallelism across samples — the reconstruct
  kernel already exposes a `parallel` flag). The two are not mutually exclusive, but which
  dominates decides where effort goes.

The plan resolves this by **profiling the synchronous fill first** (`pyinstrument` on the Python
driver / `samply` on the Rust fill) to split IO vs CPU time, *then* choosing/keeping the producer
thread on evidence.

**Phase 3/4 — harness (reuse, don't rebuild).** Reuse SVAR1's cold-cache overlap harness
(`benchmarking/streaming/cold_cache_overlap.py`) parameterized for `.svar2`; the correctness oracle
is the same byte-parity check as the functional tests (a faster variant that fails parity is a bug,
not a speedup). Sweep the **sample axis** (the dominating, unbounded dimension) to confirm the
memory model is flat (the #284 gate already does this deterministically) and that the engine's win,
if any, holds across cohort sizes rather than at one hand-picked size. Record the baseline
(synchronous path) number before wiring the engine.

**Do not pre-optimize the reused kernel.** `gather_haps_readbound`, `split_to_flat`, and
`reconstruct_haplotypes_from_svar2` are already-measured code from the rust migration (Phase 6a) —
profile before touching any of them. The one genuinely net-new hot path is `Svar2Store.read_window`
(live `find_ranges`); it is binary searches over the index and *should* be cheap, but that is a
hypothesis to confirm with a profile, not an assumption to ship on.

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
   holds across the multi-contig fixture. **This is also the perf baseline** — profile its fill to
   split IO vs CPU (Performance, Phase 1) and record its cold-cache number.
2. **Engine (gated on measurement):** wire `Svar2StreamEngine` (producer/consumer overlap) and
   re-verify parity + the scale gate. Ship it as the default **only if** it clears the Phase-0
   decision gate (cold-cache win over the synchronous path, outside node noise); otherwise keep it
   off-by-default / behind the `_prefetch_strategy` toggle and ship the synchronous path, recording
   the measured reason — mirroring how SVAR1 chose Design A over C on evidence.

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
