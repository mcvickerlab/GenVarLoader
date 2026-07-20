# Design: Optimization pass on `StreamingDataset` VCF + PGEN backends

**Date:** 2026-07-18
**Status:** design (pending spec review)
**Roadmap:** `docs/roadmaps/streaming-dataset.md`
**Issue:** [#276](https://github.com/mcvickerlab/GenVarLoader/issues/276) (optimization follow-up)
**PR:** [#299](https://github.com/mcvickerlab/GenVarLoader/pull/299) (draft, into `streaming`) — optimizations land in this PR
**Follows:** VCF/PGEN backends design `docs/superpowers/specs/2026-07-17-streaming-vcf-pgen-backends-design.md`
**Method:** `skills/performant-py-rust` (measurement-first; every optimization is a hypothesis a benchmark must confirm)

## Summary

PR #299 shipped correct, byte-identical VCF + PGEN streaming backends behind the shared
`RecordStreamEngine`. This is the **optimization pass**: reduce the per-epoch wall-clock of a
full `StreamingDataset.to_iter()` sweep for both backends at cohort scale, and **quantify the
streaming-vs-`gvl.Dataset` throughput gap** the roadmap asserts ("slower per epoch than a written
`Dataset`, but zero preprocessing") but has never measured.

The pass is **measurement-first**. Three bottleneck hypotheses are named below; none is treated as
confirmed until a profile at realistic scale says it dominates. The deliverable order is
harness → profile → optimize the confirmed hot spot → re-profile, not a fixed edit list. On this
shared, noisy node the **load-bearing gate is deterministic counters + same-session before/after**,
never absolute wall-clock (the standing perf-gate convention:
`gvl-rust-perf-gate-shared-node-noise`, `docs/roadmaps/streaming-dataset.md`).

### Decisions (settled during brainstorming)

- **Workload sweep:** `n_samples ∈ {1k, 10k, 50k}` at ~200 regions/contig (the design's dominating
  axis; runnable on this node at modest file sizes).
- **Where it lands:** in PR #299 directly. A genoray-side change is acceptable **as a separate
  genoray PR that #299 depends on via a `rev` bump** — Rust-side, so a merge (not a crates.io
  release) unblocks it. No release gate (CLAUDE.md → Development Notes).
- **Dataset comparison:** a **permanent** `--compare-dataset` arm in `bench_streaming.py` plus a
  documented number, not a throwaway measurement.

## Phase 0 — Target & warrant

**Target:** lower the full-sweep wall-clock of both backends at 1k→50k samples, with PGEN's
per-window contig-prefix re-decode as the headline (it is O(prefix) unamortized today), and produce
a documented streaming-vs-`Dataset` throughput ratio + peak-RSS comparison.

**Warrant (bottleneck evidence, all hypotheses until Phase 1c confirms):**

1. **PGEN coarse `var_start`** — documented in `src/record_stream/pgen.rs` ("Coarse `var_start`").
   `PgenWindowFiller::fill` always decodes a contig from variant 0 and `PgenRecordSource::new`
   re-opens the `.pvar` from byte 0 (no seek), text-skipping `var_start` lines. O(contig-prefix)
   pgenlib decode + O(var_start) `.pvar` line-skip **per window**, unamortized. The Task 13 harness
   already prints the analytic multiplier `n_windows × pvar_variants`.
2. **The transpose `fill_decoded_window`** (`src/record_stream/transpose.rs`) — `BitGrid3` is
   `(V,S,P)` C-order (`flat = v*S*P + s*P + p`), but the loop is hap-major (`s,p` outer, `v` inner),
   so the inner `v` loop **strides by `S*P` bits** and does a per-bit `get_bit`. At cohort scale
   (S=50k) that is O(V·S·P) cache-hostile bit-tests. Hits **both** backends (the transpose is
   source-agnostic). Not flagged in the backends design (which assumed decode dominates).
3. **Per-window reader re-open** — `VcfWindowFiller::fill` opens a fresh htslib `IndexedReader`
   every window; PGEN re-opens the `.pvar`. Per-window construction cost that a persisted reader
   would amortize.

If Phase 1c shows any of these is a negligible share at real scale, it is **not** optimized (YAGNI /
Amdahl). The profile decides.

## Phase 1 — Characterize (harness + counters land first, before any optimization)

### 1a. `--compare-dataset` arm (permanent)

Extend `benchmarking/streaming/bench_streaming.py` with a third driver alongside `engine`/`sync`:

- **Write once, timed separately.** `gvl.write()` a dataset from the *same* BCF/PGEN + reference +
  BED as the streaming run. Report `write_time_s` and on-disk `dataset_bytes` as **preprocessing
  cost**, printed on their own line — never folded into the throughput number.
- **Iterate identically.** `Dataset.open(path, reference).subset_to(samples=...)` then iterate the
  **same (region, sample) cartesian cells in the same region-major plan order**
  `StreamingDataset._plan()` yields, haplotypes-only, `jitter=0`, `batch_size` matched. Time the
  full sweep; report items/s + `peak_rss_kb`.
- **Built-in correctness oracle.** Streaming and written datasets are already byte-identical
  (existing parity tests), so the arm **asserts `bytes_emitted` is equal across engine / sync /
  dataset drivers**. A silent divergence in the comparison harness fails here, not in a hand-read
  number.
- Reuses the existing fixture plumbing (`_prepare_fixture`, `_build_reference`, `_make_bed`); the
  written dataset goes in a temp dir cleaned up per run, matching the `--cold` discipline.

### 1b. Deterministic decode counters (the load-bearing gate)

Mirror the SVAR1 `svar1_csr_entries_touched()` pattern (process-wide `AtomicUsize`, so the engine's
background producer thread is counted — the `#296` lesson):

- **`pgen_variants_decoded()`** — count variants pgenlib actually decodes across a sweep. Before the
  fix ≈ `contig-prefix × n_windows`; after ≈ `Σ window variants (+ padding)`. This is the **PGEN
  gate**, replacing the harness's current analytic-multiplier stand-in with a measured count.
- **`transpose_bits_scanned()`** — genotype bits examined in `fill_decoded_window`. Distinguishes
  "scan every bit" (`V·S·P`) from "skip empty words". This is the **transpose gate** on sparse data.

Both exposed via `src/lib.rs` like the SVAR1 counter, read by the harness and by dedicated Rust/py
scale tests.

### 1c. Profile

Profile each backend at **10k samples** (mid sweep point) with `perf` on the Python process
(no sudo, `paranoid=2`; per `gvl-profiling-perf-not-pyspy-native` — py-spy `--native` is ~10× too
slow), plus `perf stat` for instructions and LLC-load-misses. Fill the Phase-1 dimensions table
with **measured** per-stage shares:

| stage | thread | expected cost | measured share (10k) |
|---|---|---|---|
| genoray decode (htslib/pgenlib + normalize/atomize + dense grid fill) | producer | O(V·S·P) | _TBD_ |
| transpose `fill_decoded_window` | producer | O(V·S·P) | _TBD_ |
| reconstruct (`reconstruct_haplotypes_from_sparse`) | consumer | O(haps·seqlen + carried) | _TBD_ |
| reader open (`VcfRecordSource`/`PgenRecordSource::new`) | producer | O(1)–O(var_start) | _TBD_ |

**Only the stage(s) the profile says dominate get optimized.** VCF and PGEN are profiled
separately (PGEN's decode is not GIL-free; different overlap characteristic — never averaged).

## Phase 2 — Levers (ranked hypotheses, each gated by 1c + parity)

| # | Hypothesis | Lever | Deterministic gate |
|---|---|---|---|
| 1 | PGEN re-decodes the contig prefix every window | **Narrow `var_start`/`var_end`.** The construction-time `.pvar` scan (`contig_var_ranges`) already reads every row — also retain a per-contig `POS→vidx` array and `max_v_len` (max REF extent). Per window, binary-search `var_start = first vidx with POS ≥ window_start − max_v_len` (the pad keeps spanning upstream DELs visible, matching `OverlapMode::Variant`) and `var_end = last vidx with POS ≤ window_end`. This narrows the **pgenlib decode** with a gvl-only change. The residual O(var_start) `.pvar` **text re-skip** in `PgenRecordSource::new` needs a genoray `PvarReader` seek (separate genoray PR, rev bump) — only pursued if 1c shows it is material. | `pgen_variants_decoded()` drops from ~prefix to ~window; same-session wall-clock |
| 2 | Transpose is cache-hostile and tests every bit | **v-outer two-pass counting sort.** The `(V,S,P)` grid is contiguous in the `(S,P)` block for a fixed `v`. Pass 1: v-major sequential scan counting set bits per hap → prefix-sum → `geno_offsets`. Pass 2: v-major sequential scan filling `geno_v_idxs` at `cursor[hap]++` (ascending `v` preserved per hap — the CSR contract). Use **word-level popcount / `trailing_zeros`** on the `BitGrid3` backing words to skip empty stretches (genotypes are sparse). Worst case stays O(V·S·P) but sequential + sparse-skipping. Helps **both** backends. Must reproduce the exact hap-major CSR the current loop emits (unit-tested against the hand-checked grid in `transpose::tests`). | `transpose_bits_scanned()` ≪ V·S·P on sparse data; LLC-load-misses ↓; same-session wall-clock |
| 3 | Reader re-opened per window | **Persist the reader across `fill` calls.** VCF: keep one htslib `IndexedReader` on the producer thread, `advance_region`-style re-fetch per window instead of `VcfRecordSource::new` each time. PGEN: keep the `PvarReader`/pgen handle open (composes with lever 1). Requires confirming the persisted reader is confined to the single producer thread (`engine.rs` guarantees strictly-sequential `fill`). | open-count counter → 1 per sweep; same-session wall-clock |

**Correctness contract for every lever:** byte-identical parity is non-negotiable —
`tests/dataset/test_streaming_{vcf,pgen}_parity.py`, the Rust filler-equivalence `cargo test`s, and
the harness's cross-driver `bytes_emitted` assertion all must stay green. A lever that fails parity
is a bug, not a faster version; it is reverted.

## Phase 3 — Oracle + baseline

- **Oracle (exists):** `test_streaming_{vcf,pgen}_parity.py` (byte-identical streamed haplotypes vs
  `Dataset.open(...)[r,s]`) + Rust `pgen_filler_matches_vcf_filler_*` equivalence tests +
  `transpose::tests` hand-checked grid.
- **Baseline:** record the two counters + best/median-of-N wall-clock per backend at each sweep
  point, **same session**, before any optimization edit. Every subsequent change is measured against
  this same-session baseline, never against a number from another session/machine.
- **Rebuild rule (enforced):** `pixi run -e dev maturin develop --release` before every Python
  parity/bench run — pytest imports the stale `.so` otherwise (CLAUDE.md). `cargo test` compiles
  from source and is unaffected.

## Phase 4 — Optimize loop

Loop until diminishing returns: one hypothesis → one change → rebuild → re-run oracle + counters +
same-session bench → keep only if the gate moves **and** parity holds; revert otherwise. Re-profile
(the hot spot moves). For any Rust change claimed to vectorize / skip work, verify the mechanism
(`cargo-show-asm` or the `*_scanned` counter), don't assert it.

**Stop when:** the dominating stage is no longer dominating (Amdahl ceiling), or the remaining gain
is within this node's run-to-run noise, or complexity/maintenance cost outweighs the win. State
which stopped the loop.

## Deliverables (all in PR #299 / branch `spec/276-vcf-pgen`)

1. **Harness:** `--compare-dataset` arm + `pgen_variants_decoded()` / `transpose_bits_scanned()`
   counters exposed to Python, with scale tests gating them (deterministic, not RSS).
2. **The confirmed optimizations** — expected: lever 1 (PGEN `var_start` narrowing) + lever 2
   (transpose); lever 3 and the genoray seek only if 1c justifies them.
3. **Optional genoray PR** (`PvarReader` seek) consumed via `rev` bump, *if* the text re-skip proves
   material. Both genoray crates share one `rev` (CLAUDE.md).
4. **Docs:** `docs/roadmaps/streaming-dataset.md` (tick the optimization follow-up, record the
   measured PGEN before/after + streaming-vs-Dataset ratio), the `bench_streaming.py` module
   docstring (replace the analytic-multiplier caveat with the measured counter once the PGEN fix
   lands), and the `src/record_stream/pgen.rs` "Coarse `var_start`" doc section (mark resolved /
   describe the narrowing).

## Scope guard (YAGNI)

Out of scope, do **not** gold-plate: N-slot ring buffer (2-slot ping-pong stays until profiling
demands more), multiallelic PGEN, output-mode breadth / `with_len` / AF filtering / jitter (#277),
`num_workers > 1` sharding, `.pvar.zst` support. This pass is throughput of the existing
haplotypes-only, `jitter=0` path — nothing wider.

## Testing strategy

- **Parity (primary, exists):** byte-identical streamed haplotypes vs `Dataset[r,s]`, both backends,
  after every lever.
- **Deterministic scale gates (new):** `pgen_variants_decoded()` proportional to window variants
  (not contig prefix) after lever 1; `transpose_bits_scanned()` ≪ `V·S·P` on a sparse fixture after
  lever 2. Rust unit tests + a py scale test, mirroring `test_streaming_scale.py`.
- **Transpose equivalence (new):** the rewritten `fill_decoded_window` emits the byte-identical CSR
  of the current implementation across the existing hand-checked grids + a new larger sparse grid.
- **Harness correctness:** cross-driver `bytes_emitted` equality (engine == sync == dataset).
- **Wall-clock:** secondary color only, best/median-of-N, same-session — never a pass/fail gate.
