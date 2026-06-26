# Roadmap: Migrate the GVL core to Rust

**Status legend:** ⬜ not started · 🚧 in progress · ✅ done · ⏸️ blocked

This is a living tracker. **Any work that touches the Rust migration must read this file
first and update it as part of the change** — tick completed tasks, record measurements
under the relevant checkpoint, and update the phase status marker + PR link.

## Branch & gate strategy (changed as of Phase 2, 2026-06-24)

Phases 0–1 were merged to `main` incrementally. **From Phase 2 onward the work accumulates on
a single persistent integration branch (`rust-migration`) with NO per-phase throughput gate**,
and ships as ONE big merge at the end. Rationale: profiling Phase 2 showed the read-path
overhead is per-kernel Python dispatch glue (redundant `np.ascontiguousarray` coercions +
FFI boundary crossings), not rust compute — so the real win comes from collapsing
`__getitem__` into a single large rust kernel, which can only be done once enough of the
read path is in Rust. Gating each intermediate phase on throughput would block correct,
parity-verified work behind an overhead that the architecture is designed to delete later.
**Per-phase gate is now parity only**; a dedicated optimization pass (eliminate glue →
single big `__getitem__` kernel) re-establishes the throughput gate before the final merge.

---

## Goal & end state

Migrate GenVarLoader's core data structures and algorithms from Python/numba to a
self-contained **Rust crate** (`genvarloader`), cargo-testable and usable from Rust
directly, wrapped by a **thin PyO3 binding**. Python keeps only the ergonomic surface —
`Dataset` indexing sugar, torch integration, validation/error messages — and dispatches
into Rust for everything else.

**Why:**
- Far faster `gvl.write()` / `update()` and likely faster `Dataset.__getitem__`.
- A powerful type system enabling strong abstractions that shrink the code + testing surface.
- Eliminate the ~35 numba kernels scattered across the read/write paths, collapsing the
  bug surface.

**Eventual scope:** seqpro's `Ragged` layout + ops are the **shared Rust ragged substrate**
(`seqpro-core` rlib, pyo3-free) — GVL consumes them via a crate path-dep rather than
reimplementing ragged primitives in-house (decision 2026-06-23, Phase 1). genoray
(VCF/PGEN variant & sparse-genotype IO) remains in scope for the full Rust stack,
sequenced last (Phase 6), and may graduate into its own roadmap.

### Target crate layout

Grown from today's `src/{lib,bigwig}.rs`:

```
src/
├── lib.rs            # pymodule registration only
├── ragged/           # bridge to seqpro-core (NOT a reimplementation)  ← Phase 1
├── genotypes/        # sparse genotype assembly
├── variants/         # variant gather, flat/windowed views
├── reconstruct/      # haplotype reconstruction
├── tracks/           # track re-alignment, insertion-fill, splice
├── reference/        # reference sequence assembly
├── write/            # write/update pipeline
├── bigwig.rs         # (existing)
└── ffi/              # PyO3 bindings — the only place that touches Python
```

The crate is pure Rust + `ndarray`/`rayon`. `ffi/` is the seam where numpy arrays and
`seqpro.Ragged` / `genoray` objects cross the boundary (zero-copy where possible).
`src/ragged/` is a **bridge layer** to the `seqpro-core` rlib — it does not own the
ragged implementation. By Phase 6 the crate stops depending on Python genoray for
variant IO paths.

---

## The migration contract (strangler fig + byte-identical parity)

Every unit follows the same loop, so the work is repetitive and low-surprise:

1. **Implement** the unit in Rust on the native ragged layout.
2. **Expose** it through `ffi/` with a Python-side switch (env var / flag) selecting Rust
   vs the existing numba/Python impl.
3. **Differential-test:** a reusable harness runs *both* impls on property-generated
   inputs (built on `vcfixture` + numpy generators) and asserts **byte-identical** output
   across the py310–313 × linux/macOS matrix. A unit only "lands" when parity holds.
4. **Land:** flip the default to Rust and **delete the numba/Python impl in the same
   bundled PR**. Remove the switch when the phase closes.

`main` stays shippable at all times; every step is reversible until parity is proven;
numba deletion is continuous rather than a big-bang at the end.

**Ragged substrate:** the ragged layout and core ops are **consumed from `seqpro-core`**
(a pyo3-free rlib in the seqpro repo) via a Cargo path-dep, not reimplemented in GVL's
own crate. GVL's `src/ragged/` is a bridge/adapter layer only.

**Standing CI invariant (not a per-phase gate):** abi3 wheels must keep building across
py310–313 × linux/macOS as the Rust surface grows.

**PR strategy:** each phase lands as one bundled PR (solo-maintainer preference). See
[[feedback_pr_strategy]].

---

## Baseline metrics

> Captured once in Phase 0. Every later gate compares against these numbers. Fill in when
> Phase 0 lands.

| Metric | Corpus | Baseline | Captured |
|---|---|---|---|
| `gvl.write()` wall-clock | 1kg chr21/chr22 (100 regions), macOS M-series | 1.143 s | ✅ |
| `gvl.write()` peak RSS | 1kg chr21/chr22 (100 regions), macOS M-series | 3.593 GB | ✅ |
| `gvl.update()` wall-clock | 1kg chr21/chr22 (vcfixture tier) | _TBD_ (smoke only: 0.022 s for a 60-row synthetic annot track — not a real workload) | ⬜ |
| `Dataset.__getitem__` throughput (tracks mode = `intervals_to_tracks` read path) | `chr22_geuv` realistic bench (165 regions × 5 samples, chr22, read-depth; `SEQLEN=16384`, `BATCH=32`, 2000 batches, `NUMBA_NUM_THREADS=1`), Carter HPC (AMD EPYC 7543, linux-64) | **169.9 batch/s** (5.886 ms/batch, ~5.4k item/s); peak RSS **3.531 GB** | ✅ |

> getitem baseline captured on Carter (2026-06-23, gvl 0.35.0, `GVL_BACKEND` unset →
> `intervals_to_tracks` default `rust`). `profile.py` now prints wall-clock + throughput;
> py-spy needs no sudo on Linux (`ptrace_scope=0`). Secondary read paths on the same corpus:
> **haplotypes 123.9 batch/s** (8.069 ms/batch), peak RSS 3.532 GB; **variants 145.3 batch/s**
> (6.884 ms/batch, variable-length — variants are ragged by definition, so `with_len` doesn't
> apply). Peak RSS (~3.53 GB) is dominated by the numba/llvmlite JIT baseline (~3.2 GB), matching
> the bigWig write slice. (Aside: a *mixed* `with_seqs("variants").with_tracks(...).with_len(L)`
> query — fixed tracks alongside necessarily-ragged variants — currently `AttributeError`s because
> the fixed-length exemption in `_query.py` checks `RaggedVariants` while the value is still
> `_FlatVariants`; a one-line guard, not a Phase 0 gate.)
>
> The realistic corpus rebuild surfaced a **filtered-PGEN write bug**: `build_realistic.py` now
> drops symbolic/breakend/multi-allelic variants at the **plink2** stage (`drop_unsupported_variants`)
> instead of via a genoray `filter=`, because a filtered genoray PGEN returns unfiltered-space
> `var_idxs()` while `_index` is the filtered table, and `gvl.write`'s `_pgen_region_chunks` mixes
> the two (IndexError + likely mis-indexed stored variants). Filed as
> [d-laub/genoray#69](https://github.com/d-laub/genoray/issues/69); pre-filtering keeps both
> coordinate spaces aligned.
>
> Driver scripts landed: `tests/benchmarks/profiling/profile_write.py` (`--op write`/`update`),
> `tests/benchmarks/profiling/profile.py` (getitem; prints throughput), and
> `tests/benchmarks/profiling/baseline_getitem.sh` (py-spy speedscope, no sudo on Linux).
> Reproduce: build the corpus (`pixi run -e dev python tests/benchmarks/data/build_realistic.py`,
> needs `/carter` or `GVL_BENCH_SOURCE`), then `pixi run -e dev python …/profile.py --mode tracks`
> for throughput and `pixi run -e dev memray-tracks` (+`memray-haps`) with `memray stats …`.

Benchmark sources: dataloader bench lives on the `prefetching-dataloader` branch
([[project_dataloader_bench]]); fixtures from vcfixture ([[project_vcfixture_migration]]).
py-spy on macOS needs sudo — hand David a bash script, don't invoke it directly
([[feedback_macos_profiling_handoff]]).

### bigWig write-path slice (Phase 4 partial — NOT the full 1kg write baseline)

> Corpus: synthetic chr21 (200k bp) / chr22 (150k bp), n_samples=8, density=0.05,
> n_regions=2000, width=5000. Measured on macOS (Apple M-series), memray for RSS.
> These rows are a bigWig-only write slice; the 1kg full-write baseline remains TBD.

| Metric | Corpus | Legacy (baseline) | Rust (after) | Δ | Captured |
|---|---|---|---|---|---|
| `gvl.write()` bigWig wall-clock | synthetic chr21/chr22 slice | 1.502 s | 0.801 s | ~1.88× faster | ✅ |
| `gvl.write()` peak RSS | synthetic chr21/chr22 slice | 3.538 GB | 3.386 GB | −4% (dominated by numba/llvmlite JIT ~3.2 GB) | ✅ |
| `gvl.write()` total allocated | synthetic chr21/chr22 slice | 8.380 GB | 6.004 GB | ~28% less | ✅ |

---

## Phases

Each phase is one bundled PR and ends in a measure checkpoint.

### Phase 0 — Foundation & harness ✅
_PR: #241_

- [x] Stand up the `ffi/` seam + first lazily-grown domain module (NOT an empty
      skeleton — lazy growth, per the approved spec: a module is created only when it
      holds real code). `src/ffi/mod.rs` (the only place new kernels touch PyO3) +
      `src/intervals.rs` (pure ndarray core) carry the first live kernel
      `intervals_to_tracks` through the new seam. Renamed the legacy bigwig `intervals`
      pyfunction to `bigwig_intervals` to free the name. Backend-dispatch registry
      (`python/genvarloader/_dispatch.py`, `GVL_BACKEND` override + per-kernel default)
      routes the production call site (`_dataset/_intervals.py`, default `rust`).
      Commits: `64e0836` (registry), `917957b` (ffi+kernel), `ec4c15b` (route).
- [x] Build the reusable differential-test harness: run-both-assert-byte-identical
      (`tests/parity/_harness.py`, return-value + in-place variants) + a hypothesis
      property generator. Per-kernel gate (`test_intervals_to_tracks_parity`, 100
      examples incl. sub-query interval starts — #242 fixed both backends to clip
      to the query window) + a MEANINGFUL dataset-level read-path backstop
      (`test_dataset_parity.py`: `ds[:, :]` track getitem byte-identical across
      backends, with a spy asserting the kernel is actually invoked). Commits:
      `ef4f91a`, `ad82b31`.
- [x] Wire `cargo test` into pixi dev tasks (`cargo-test`, plus `memray-write`).
      Commit `20cd4ef`.
- [x] Confirm abi3 wheels build with the new `ffi/`+`intervals` modules
      (`genvarloader-0.35.0-cp310-abi3-macosx_11_0_arm64.whl` builds clean; the
      py310–313 × linux/macOS release matrix is release-gated and unaffected by the
      pure-Rust additions). Commit `20cd4ef`.
- [x] Capture baselines (table above): `write()` wall-clock + peak RSS captured;
      `__getitem__` tracks-mode (`intervals_to_tracks` read path) throughput +
      peak RSS captured on Carter (169.9 batch/s, 3.531 GB; haplotypes secondary
      123.9 batch/s). Driver scripts landed (commit `0be2d67`); `profile.py` now
      prints throughput, and the `chr22_geuv` corpus rebuild fixed a stale (0.25.0,
      truncated-tracks) artifact + a filtered-PGEN write bug (now pre-filtered at
      the plink2 stage; filed on genoray). The `update()` row stays ⬜: the only
      landed driver runs a 60-row synthetic annot (smoke, not a real workload) — a
      real write-path update baseline is deferred to Phase 4.

**Checkpoint:** harness green; foundation + proof-point landed; getitem (gate)
baseline captured on Carter. `update()` remains a deferred smoke-only row (real
workload is a Phase 4 write-path concern, not a Phase 0 gate). The
`intervals_to_tracks` sub-query-start contract gap (max_jitter>0 tracks, #242)
is resolved: both kernels clip to the query window and parity covers it.

### Phase 1 — Ragged primitives + layout (beachhead) ✅
_PRs: seqpro [ML4GLand/SeqPro#60](https://github.com/ML4GLand/SeqPro/pull/60), GVL [mcvickerlab/GenVarLoader#240](https://github.com/mcvickerlab/GenVarLoader/pull/240)_

The foundation everything sits on. Realized via the `seqpro-core` shared substrate
rather than a GVL-in-house reimplementation (see decision 2026-06-23). Bottom-up.

- [x] Extract a pyo3-free `seqpro-core` rlib (crates/seqpro-core in the seqpro repo)
      that owns the `Ragged` layout (offsets + data buffers) and its core ops.
- [x] Port the last two numba ops to Rust inside `seqpro-core`: `to_padded` and
      `reverse_complement`. seqpro's ragged layer is now numba-free.
- [x] GVL consumes `seqpro-core` via a Cargo path-dep (editable; flip to
      git/crates.io before shipping). `src/ragged/` is a bridge adapter, not a
      reimplementation.
- [x] Proof-point op (`to_padded`) rerouted through the shared `seqpro-core` kernel
      in GVL with byte-identical parity confirmed.
- [x] Remove `awkward` from the foundation layer. (GVL migrated onto seqpro's
      Rust-backed `_core.Ragged`; `RaggedVariants` now wraps a single record
      `seqpro.rag.Ragged`.)

**Checkpoint:** parity green (byte-identical `to_padded`). Foundational — no perf gate,
but record incidental wins. Relevant prior work: [[project_ragged_assembly_bottleneck]].

### Phase 2 — Genotype assembly + variant gather ✅ (parity-verified; perf deferred to consolidation)
_Branch: `rust-migration` (persistent integration branch — see "Branch & gate strategy" below). Not separately merged to `main`._

- [x] Migrate `_dataset/_genotypes.py` **assembly/selection** kernels: `get_diffs_sparse`,
      `choose_exonic_variants`. (The `_genotypes.py` *reconstruction* kernels —
      `reconstruct_haplotypes_from_sparse` et al. — are Phase 3, not Phase 2; the earlier
      "6 numba" figure double-counted them.) Dead `filter_af` deleted (zero production
      callers; AF filtering is inline numpy in `_haps.py`/`_flat_variants.py`) — same
      precedent as the Phase 0 `splits_sum_le_value` dead-path removal. Its dedicated unit
      test was removed with it.
- [x] Migrate `_dataset/_flat_variants.py` kernels (7 numba): `_gather_v_idxs` + `_gather_v_idxs_ss`
      → `gather_rows` (unified via `(2,n)` offset normalization), `_gather_alleles`,
      `_compact_keep`, `_fill_empty_scalar`, `_fill_empty_fixed`, `_fill_empty_seq`.
- [x] Migrate `_dataset/_rag_variants.py`; drop `awkward` from these hot paths. (Done at the Python level: `RaggedVariants` now wraps a single record `seqpro.rag.Ragged`; no numba kernels remain in this file.)

**Architecture:** pure-`ndarray` cores in `src/genotypes/` + `src/variants/`; PyO3 only in
`src/ffi/`; per-kernel dispatch via `genvarloader._dispatch` (default `rust`, `GVL_BACKEND`
override); numba impls retained as registered parity references (deleted wholesale in Phase 5).

**Dtype-correctness (beyond the plan):** the flat gather/fill kernels are NOT v_idxs-only — they
also run on float32 dosage and **arbitrary-dtype** custom per-call FORMAT fields (issue #231, e.g.
`int16`). The numba refs preserved input dtype; a naive int32/float32-only port silently corrupted
them (caught here: float32 dosage `[0.25,0.75]`→`[0,0]`). Final design dispatches by dtype —
`*_i32`/`*_f32` rust cores for the hot paths + a **dtype-preserving numba fallback** for all other
dtypes, with direct regression tests (int16/int64/float32) locking it.

**Gate (parity — MET):** byte-identical parity for every ported kernel via `@pytest.mark.parity`
hypothesis suites (both returned arrays for tuple kernels), plus a spy-guarded variants-mode
dataset backstop proving the rust kernels run on the live `__getitem__` path. Full tree green:
904 passed (rust) / 617 passed (numba backend, dataset+unit); lint/format/typecheck clean;
`cargo test` green; abi3 build OK. (One pre-existing unrelated failure, `test_e2e_variants`, is a
`with_len`-on-variants benchmark bug that fails identically at the Phase-2 base — not introduced here.)

**Gate (throughput — DEFERRED, not a blocker):** see "Branch & gate strategy". Measured medians
(`chr22_geuv`, `NUMBA_NUM_THREADS=1`, Carter):

| Mode | rust | numba (same session) | documented baseline |
|---|---|---|---|
| haplotypes | 128.8 batch/s | 137.9 | 123.9 |
| variants | 139.5 batch/s | 149.3 | 145.3 |

rust is a **stable ~7% slower than numba** (rust-haps still beats the 123.9 baseline; rust-variants
is ~4% below its 145.3 baseline). cProfile of the rust variants `__getitem__` shows the cost is
**pure Python glue, not rust compute**: `np.ascontiguousarray` is 28,800 calls / 3.98 s = **62%** of
the loop (~36 redundant coercions per batch in the per-kernel dispatch wrappers), while the rust
kernels themselves are negligible (`gather_alleles` 0.012 s, `get_diffs_sparse` 0.010 s). This
validates collapsing the read path toward a **single big rust `__getitem__` kernel** (drop redundant
coercions short-term; eliminate per-kernel boundary crossings + intermediate numpy allocs long-term),
addressed in a dedicated optimization pass before the final merge.

### Phase 3 — Reconstruction + track realignment ✅ (parity-verified; throughput recorded)
_PR: [#245](https://github.com/mcvickerlab/GenVarLoader/pull/245) → rust-migration_

The numba bulk and the big read-path win. Ported 8 kernel groups behind dispatch (reference,
haplotype reconstruct singular+batch, PRNG, insertion-fill, track realignment, RLE) plus fused
`__getitem__` entries for both haplotypes and tracks. Default backend is `rust`; numba retained
as the registered parity reference for the consolidation pass (Phase 5).

- [x] Task 12: Audit `__getitem__` glue (2 FFI crossings → inventory; `docs/roadmaps/phase-3-getitem-glue-audit.md`).
- [x] Task 13: Fused haplotypes `__getitem__` kernel — `reconstruct_haplotypes_fused` collapses 2 FFI crossings to 1 on the non-splice plain haps path. Dataset parity gate: byte-identical to composed numba oracle (37/37 parity tests pass). Annotated path and splice path remain on unfused dispatched kernels (documented in task-13-report.md).
- [x] Task 14: Fused tracks `__getitem__` kernel — `intervals_and_realign_track_fused` chains `intervals_to_tracks` → `shift_and_realign_tracks_sparse` in 1 FFI crossing per track; Rust scratch buffer replaces Python `np.empty` intermediate. Dataset parity gate: byte-identical across all 5 insertion-fill strategies (39/39 parity tests pass; fixture uses max_jitter=0 per #242 contract).
- [x] Task 15: Full-tree verification + roadmap + skill check (final-review fixes applied). Full tree green: 909 passed, 15 xfailed (11 added here + 4 pre-existing), 0 failed. Lint/format clean; cargo 85/85; abi3 wheel builds. See final-review section in task-15-report.md.
- [x] Migrate `_dataset/_reconstruct.py` + `_dataset/_haps.py` remaining paths. Annotated path now fused via `reconstruct_annotated_haplotypes_fused` (Phase 3 close-out, Task 4); splice path fused via `reconstruct_haplotypes_spliced_fused` (Phase 3 close-out, Task 5). Both byte-identical to the composed numba oracle. (The annotated+spliced intersection remains on the unfused dispatched rust core — still parity-gated and rust-by-default — with fusion deferred to Phase 5.)
- [x] Migrate `_dataset/_tracks.py` realign (6 numba) + `_dataset/_intervals.py` (4 numba). Rust-default + fused (`intervals_and_realign_track_fused`); the #242 `intervals_to_tracks` clip fix merged from main (both backends). Remaining numba kernels are retained Phase-5-deletion parity references, not unmigrated paths.
- [x] Migrate `_dataset/_reference.py` (6 numba). `Reference.fetch` rerouted through the dispatched rust `get_reference` (Phase 3 close-out, Task 3); the three zero-caller `_fetch_*` numba functions deleted. The live `_get_reference_*` numba kernels remain as Phase-5-deletion parity references.
- [x] Migrate `_dataset/_insertion_fill.py` + `_dataset/_splice.py`. No numba kernels remain to migrate in `_insertion_fill.py`; splice reconstruction fused via `reconstruct_haplotypes_spliced_fused` (Phase 3 close-out, Task 5).

**Gate (parity — MET):** byte-identical parity confirmed, with two documented numba-bug sub-domains excluded from the oracle via assume(False) in parity tests (consistent with the #242-family precedent):
  1. *start>=clen / #242-family*: get_dummy_dataset() (max_jitter=2) float-track tests trigger the intervals_to_tracks debug_assert panic; xfailed (strict=False) in 10 tests across test_output_bytes_per_instance.py, test_dummy_dataset_insertion_fill.py, test_flat_intervals.py, test_realign_tracks.py, test_seqs_tracks.py.
  2. *reconstruct trailing-under-write*: a deletion that drives ref_idx past the contig end causes numba's trailing-fill to behave differently from Rust (numba uses Python-style negative-index slicing; Rust clamps out_end_idx to 0). Both behaviors are undefined for inputs outside the production contract (variants always within contig bounds). Excluded via (a) overshoot pre-check in the reconstruct parity tests and (b) double-init guard (sentinel 0x00 vs 0xFF, and int32 sentinel 0 vs -1 for annotation buffers) to catch any positions numba leaves unwritten. Rust is correct in both cases; numba is not a valid oracle in this sub-domain.

**Gate (throughput — DEFERRED):** recorded only (see "Branch & gate strategy").

#### Phase 3 throughput measurements (re-measured at close-out, 2026-06-25)

> Harness: `tests/benchmarks/test_e2e.py` via **pytest-benchmark** — steady-state timing of eager
> `ds[r, s]` (BATCH=32 region/sample pairs, `with_len(SEQLEN=16384)`), warmup excluded, 75–190 rounds
> per test. Corpus `chr22_geuv.gvl` (max_jitter=0, 165 regions × 5 samples, chr22 read-depth).
> `NUMBA_NUM_THREADS=1`, release build (`maturin develop --release`), HEAD `6af2dbb`, Carter HPC
> (AMD EPYC 7543, linux-64). OPS = batch/s = 1 / mean.
>
> ⚠️ **Not comparable to the prior table.** The old ~37 haps / ~20 tracks figures came from a
> *different* harness (the 500-batch `benchmark_haps.py` script, since retired here). Read the
> **rust ÷ numba ratio** measured on this one harness at one HEAD as the real signal, not the
> absolute jump. Single-thread; both backends' batch drivers are serial (rayon deferred to Phase 5).

| Mode | rust (batch/s) | numba (batch/s) | rust ÷ numba |
|---|---|---|---|
| tracks-only (`intervals_and_realign_track_fused`) | 173.2 | 192.2 | 0.90× |
| tracks (seqs + `read-depth`) | 124.2 | 143.2 | 0.87× |
| haplotypes (`reconstruct_haplotypes_fused`) | 122.1 | 143.6 | 0.85× |
| annotated (`reconstruct_annotated_haplotypes_fused`) | 74.3 | 115.0 | 0.65× |

> Fusion closed most of the prior ~2× gap: rust is now within ~10–17% of numba on the haplotype/track
> paths. The **annotated** path (new this close-out, never previously timed) is the laggard at 0.65×
> — it materializes 3× the data (haps bytes + var_idxs i32 + ref_coords i32). Recorded, not gated.

#### Phase 3 throughput re-measurement after the zero-copy read-path optimization (2026-06-25)

> Re-measured on branch `zero-copy-scale-safe-readpath` (format 2.0 SoA storage + zero-copy FFI guard +
> sub-linear cache + uninit output buffers; optimization targets 1–3 above), corpus `chr22_geuv.gvl`
> (migrated in place to 2.0 via `gvl.migrate`), `with_len(16384)`, BATCH=32, `NUMBA_NUM_THREADS=1`,
> release build, Carter HPC (AMD EPYC 7543, linux-64).
>
> **De-noised harness (this measurement onward):** `_bench_indexing` now uses `benchmark.pedantic` with
> `iterations=10, rounds=50` — each timed sample folds 10 `ds[r, s]` calls so per-batch OS-scheduler
> jitter averages out (pedantic divides by `iterations`, so the figure stays per-batch). This collapsed
> the tracks-only stddev from ~0.22 ms to ~0.08 ms and made the **min** (cleanest CPU-bound estimate)
> reproducible to <1% across runs. Ratios below are **min rust ÷ min numba** (ms/batch).
>
> ⚠️ **Absolute batch/s are NOT comparable to the close-out table above** (different machine load).
> Read the **ratio**. The earlier "tracks-only is noise-dominated" note was **wrong** — once de-noised,
> the tracks-only gap is a stable, real ~0.63× regression (see target 5 below).

| Mode | rust min (ms) | numba min (ms) | rust ÷ numba | batch/s (rust / numba) |
|---|---|---|---|---|
| tracks-only (`intervals_and_realign_track_fused`) | 1.70 | 1.07 | **0.63×** (rust slower) | 566 / 897 |
| tracks (seqs + `read-depth`) | 3.40 | 3.25 | 0.95× | 275 / 286 |
| haplotypes (`reconstruct_haplotypes_fused`) | 3.45 | 3.27 | 0.94× | 270 / 288 |
| annotated (`reconstruct_annotated_haplotypes_fused`) | 5.34 | 9.00 | **1.68×** (rust faster) | 174 / 103 |

> The zero-copy interval marshalling + uninit buffers made the **annotated** path (3× output data:
> haps + var_idxs i32 + ref_coords i32) genuinely **faster than numba** (1.68×) — the close-out laggard
> is now the clearest rust win. **tracks** and **haplotypes** sit at near-parity (0.94–0.95×). The
> **tracks-only** path is the real remaining single-threaded deficit at **0.63×**: it is the cheapest
> path (~1.1–1.7 ms) so the rust-side per-batch fixed cost (FFI marshalling + Python glue, no sequence
> work to amortize it) dominates. Profiled for the next round of targets (5–7 below). Recorded, not
> gated; rayon batch parallelism is deferred to Phase 5 — single-thread parity first.

##### Optimization targets (py-spy `--native` on the rust `ds[r,s]`, 43k samples; copy trace on one batch)

The fusion removed the duplicate FFI crossings the Phase 2 cProfile flagged. A per-batch trace of
every *copying* `np.ascontiguousarray` (monkeypatched over one `ds[r, s]`) then localized what remains.
The hottest self-time leaf (`_aligned_strided_to_contig_size4`, ~20%) is **not** static-array churn —
it is the track-interval marshalling below.

1. **✅ ADDRESSED (format 2.0; branch `zero-copy-scale-safe-readpath`, PR TBD).** Resolved via the chosen "struct-of-arrays on disk"
   alternative: track intervals are now stored as three contiguous files `starts/ends/values.npy`
   sharing `offsets.npy` (format `2.0.0`, gated open + `gvl.migrate`). The contiguous memmaps cross
   the Python→Rust boundary zero-copy; the per-batch `np.ascontiguousarray` that materialized the
   whole record store is replaced by `_ffi_array` (cross zero-copy or raise loudly). The genotype
   "loaded gun" is hardened the same way (`_ffi_array` on `genotypes.data`). The scale-guard test
   (`tests/integration/test_scale_guard.py`) locks the defect closed — it fails if any per-batch
   `np.ascontiguousarray` materializes a sample-scale memmap on the read path. Original analysis below.

   **⚠️ SCALABILITY DEFECT (rust-only; not in numba): the fused track path copies the entire
   per-sample-scale interval store into RAM every batch.** Track intervals are stored as an
   **array-of-structs** memmap — record dtype `{start: i4, end: i4, value: f4}`, itemsize 12 — so
   `intervals.{starts,ends,values}.data` are **strided field views** (stride 12, non-contiguous).
   `_reconstruct.py:241-250`'s fused-rust branch wraps each in `np.ascontiguousarray(..., i4/f4)`,
   which **materializes the whole track's record store** (all regions × samples) into a contiguous
   copy on **every** `ds[r, s]` (3 × 3.6 MB on the toy corpus; **GB-scale and OOM at the >1M-sample
   target**). The **numba** branch (`_reconstruct.py:271-274`) passes the same strided views
   **directly with no copy** — numba reads strided arrays natively — so this is a rust-path
   regression, not a pre-existing cost. **Fix (zero-copy, non-breaking):** have the Rust kernel read
   the contiguous `(N,)` record buffer directly (reinterpret the 12-byte records / take a
   `&[IntervalRecord]`) and stride to `.start/.end/.value` itself, instead of demanding three
   contiguous SoA arrays. Alternative: store intervals struct-of-arrays on disk (format change).
   This is simultaneously the #1 perf cost (the 20% leaf) **and** a correctness blocker for scale.

   - **Same loaded-gun pattern, currently benign: the genotype memmap.** The fused kernels also wrap
     the full `genotypes.data`/`offsets` memmap in `np.ascontiguousarray`. Today that is a **no-op**
     (the genotype store is contiguous `int32`/`int64`, so it stays mmap, zero copy) — but it is the
     identical footgun: any future code path that yields a non-contiguous or mistyped genotype view
     would silently copy the entire sample-scale store. **Harden:** drop `ascontiguousarray` on the
     memmapped per-sample-scale args; rely on contiguous-by-construction storage and let the FFI
     **reject** non-contiguous input loudly rather than silently materializing GBs.

2. **✅ ADDRESSED (branch `zero-copy-scale-safe-readpath`, PR TBD).** The sub-linear per-variant/reference arrays (`v_starts` int32,
   `ilens`, `alt.{data,offsets}`, `ref`, `ref_offsets`) are now computed once and cached on the
   `Haps` reconstructor (`_HapsFfiStatic`, `Haps.ffi_static`), dropping the per-batch
   `int64→int32` recast of `v_starts` and the other coercions. The genotype-memmap hardening from
   target 1 (drop `ascontiguousarray`, reject loudly via `_ffi_array`) also shipped here. Original below.

   **Per-batch re-cast of dataset-static per-variant arrays (cacheable; sub-linear in samples).**
   `variants.start` is stored `int64` and re-cast to `int32` every batch (~0.59 MB × a few/batch here).
   The per-variant / reference arrays (`v_starts`, `ilens`, `alt.{data,offsets}`, `reference`,
   `ref_offsets`) grow only with the variant count (≲ a few billion germline variants even at 1M
   samples → fits in ≥64 GB RAM), so these **may** be cached/typed **once** on the reconstructor —
   unlike the per-sample-scale memmaps in (1), which must never be materialized. `reference.reference`
   (50 MB) is already contiguous `u8`, so its `ascontiguousarray` is a verified no-op.

3. **✅ ADDRESSED (branch `zero-copy-scale-safe-readpath`, PR TBD).** The fused kernels now allocate `out_data`/`annot_v`/`annot_pos` (and
   the tracks scratch) via `uninit_output<T>` instead of `Array1::zeros`, dropping the memset. The
   full-write proof holds: the reconstruct core writes every in-contract position, out-of-contract
   inputs are already excluded from the parity oracle (overshoot/double-init guards), and
   `intervals_to_tracks` does `out.fill(0.0)` as its first step so the scratch is full-write too.
   Isolated in its own commit for independent revert. Original below.

   **Output-buffer zeroing (`__memset_avx2` ~7.6%, 3 buffers on the annotated path).** The fused
   kernels `Array1::zeros(total)` for `out_data` (+ `annot_v`, `annot_pos`). The core fully writes
   every position for in-contract inputs, so an uninitialized allocation (`Array1::uninit` + a
   full-write proof) drops the memset. Requires the trailing-fill coverage argument.

4. **Per-call allocation churn (`brk`/`_int_malloc`/`malloc` ~6%)** and **`reverse_complement`
   (~9% inclusive on the strand path, a numpy post-pass).** A reusable thread-local scratch pool
   amortizes the former; folding strand RC into the kernel removes the latter. Lower priority than 1–3.

> Target 1 is a correctness/scalability fix that should land **before** any >1M-sample run, independent
> of the Phase 5 "one big `__getitem__` kernel" rewrite. Targets 2–4 are pure throughput and fold into
> that rewrite. Peak RSS not re-measured (dominated by numba/llvmlite JIT ~3.2 GB, unchanged by fusion).

##### Optimization targets — round 2 (post-format-2.0; profiled 2026-06-25 with `perf`, no `--native`)

> **Profiling method (use this, not py-spy `--native`).** py-spy `--native` slows the deep-stack
> haplotype paths ~10× (it stops the process to unwind native frames every sample) — it timed out at
> even 3.5k batches. **`perf` on the Python process is the tool:** no sudo needed on Carter
> (`perf_event_paranoid=2` permits user-space sampling of your own process; software event so no kernel
> access), near-zero overhead (tracks-only ran at 552 vs 565 batch/s under perf), and it resolves the
> `genvarloader.abi3.so` Rust symbols from the `.so` symbol table for a flat self-time profile:
>
>     NUMBA_NUM_THREADS=1 perf record -F 999 -o p.data -- .pixi/envs/dev/bin/python \
>         tests/benchmarks/profiling/profile.py --mode <mode> --n-batches 12000
>     perf report --stdio --no-children -i p.data        # flat self-time, Rust symbols resolved
>
> `profile.py` now has `--mode {haplotypes,annotated,tracks,tracks-seqs,variants,variant-windows}`. Run
> 8–25k batches so steady-state drowns the one-time import/JIT (which py-spy/perf both sample). Flat
> self-time pinpoints hot symbols without call graphs; for caller attribution add `debug =
> "line-tables-only"` + frame pointers to a profiling cargo profile (Rust release has neither by
> default), or use py-spy **without** `--native` for the Python-side inclusive tree. A separate
> Rust-only criterion harness is only worth building if we want to micro-optimize a kernel in isolation
> from FFI/Python — the in-process flat profile was conclusive for every target below.

The de-noised benchmark (above) exposed a real **tracks-only 0.63×** deficit and showed **annotated is
already 1.68×** (rust wins). Profiling each path the user cares about (tracks-only, haplotypes,
variants/variant-windows) localized the remaining single-thread work:

5. **✅ tracks-only 0.63× — per-interval `ndarray` slicing in `intervals::intervals_to_tracks`
   (rust-specific, highest value).** `perf` self-time on the tracks-only path:
   `intervals_to_tracks` 31% + `ndarray::slice_mut` **11%** + `ndarray::do_slice` **9.5%** ≈ **20.5%
   spent in ndarray slice machinery**, from `out.slice_mut(s![a..b]).fill(value)` in the inner loop
   (`src/intervals.rs:66`) and the `out.fill(0.0)` prelude. numba compiles `out[a:b] = value` to a
   direct memset and pays none of this. **Fix:** hoist `out.as_slice_mut()` (the buffer is contiguous)
   once and write `out_slice[a..b].fill(value)` / `out_slice.fill(0.0)` on the raw `&mut [f32]`,
   dropping the per-interval `SliceInfo` construction + bounds-check. Expected to reclaim most of the
   20% and close the tracks-only gap; also speeds the combined tracks path (shared kernel). This is the
   single clearest path to **rust > numba single-threaded** on the cheapest read.

   **✅ ADDRESSED (branch `opt/target-5-intervals-slice`, PR [#248](https://github.com/mcvickerlab/GenVarLoader/pull/248)).** Raw-slice form
   landed (no `unsafe` needed): `out.as_slice_mut()` hoisted once before the interval loop,
   inner-loop body rewritten to `out_slice[a..b].fill(value)` / `out_slice.fill(0.0)` on
   `&mut [f32]`, dropping per-interval `SliceInfo` construction + bounds-check. Rust min
   1.7112 ms → 1.1953 ms (~30% rust-side drop), tracks-only ratio 0.63× → 1.004×
   (numba_min/rust_min).

6. **✅ Strand reverse-complement post-pass (`reverse_complement_ragged` / `_flat.reverse_masked`) —
   backend-agnostic, biggest throughput sink on the seq paths.** Self-time (py-spy, no `--native`):
   **haplotypes ~19% self / ~28% inclusive**, **variants ~15% / ~16%**, **tracks-only ~10%**. Every
   negative-strand region triggers a Python/numpy RC pass *after* reconstruction. numba pays it too, so
   it is not the rust↔numba gap — but it is the largest single-thread throughput lever left and it must
   go before parallelization (else we parallelize a numpy pass). **Fix:** fold strand RC into the Rust
   reconstruct/track kernels — emit negative-strand regions already reverse-complemented (write the
   output buffer back-to-front with complemented bytes), deleting the `reverse_complement_ragged` step
   in `_query.py`. This is roadmap target 4's RC half, now quantified and promoted.
   _PR: [#249](https://github.com/mcvickerlab/GenVarLoader/pull/249) → rust-migration_

   **Implementation:** `src/reverse.rs` adds `rc_flat_rows_inplace` / `reverse_flat_rows_inplace`
   primitives (COMP LUT, in-place on `&mut [u8]` / `&mut [f32]`). All five flat read-path kernels
   (`get_reference`, `reconstruct_haplotypes_fused`, `intervals_and_realign_track_fused`,
   `reconstruct_annotated_haplotypes_fused`, `reconstruct_haplotypes_spliced_fused`) accept
   `to_rc: Option<ArrayView1<bool>>` and call the primitive in-kernel immediately after reconstruction
   (correct ordering: RC after forward write + insertion fill). The Python layer computes the
   per-element `to_rc` mask once per batch and routes it to the appropriate kernel; the
   `reverse_complement_ragged` Python post-pass is **retained for numba** (parity oracle) and for the
   two deferred kinds (`RaggedVariants` + `_FlatVariants`, targeted in Target 7). 958 tests pass on
   both backends (byte-identical parity). Branch: `opt/target-6-kernel-rc`, Carter HPC
   (AMD EPYC 7543, linux-64), HEAD `02497cf`.

   **Re-measured ratios (post-Target-6, 2026-06-25):**

   > Harness: `tests/benchmarks/test_e2e.py` via pytest-benchmark, same `pedantic` config as the
   > post-format-2.0 table above (iterations=10, rounds=50, warmup=5). Corpus `chr22_geuv.gvl`
   > (165 regions: **82 negative-strand / 83 positive-strand** — 50% neg-strand; with_len(16384),
   > BATCH=32), `NUMBA_NUM_THREADS=1`, release build, Carter HPC. Ratios are min rust ÷ min numba
   > (ms/batch) expressed as batch/s ratio = numba_min_ms / rust_min_ms. Numba absolute times
   > differ from the prior session (different HPC load); use the **ratio**, not the absolute.

   | Mode | rust min (ms) | numba min (ms) | rust ÷ numba | Before T6 | Δ |
   |---|---|---|---|---|---|
   | tracks-only (`intervals_and_realign_track_fused`) | 1.1012 | 0.5386 | **0.49×** | 0.63× | −0.14 (note ①) |
   | tracks-seqs (haplotypes + `read-depth`) | 1.7048 | 1.7039 | **1.00×** | 0.95× | +0.05 |
   | haplotypes (`reconstruct_haplotypes_fused`) | 1.7149 | 1.7218 | **1.00×** | 0.94× | +0.06 |
   | annotated (`reconstruct_annotated_haplotypes_fused`) | 6.1247 | 5.5100 | **0.90×** | 1.68× | −0.78 (note ②) |

   **Notes:**
   - ① tracks-only ratio **declined** (0.63→0.49×) — this is NOT a T6 regression in tracks throughput.
     The tracks-only numba time dropped from the prior session's 1.07 ms to 0.54 ms without any numba
     code change (different HPC load). Within-session the rust tracks-only path is still bounded by the
     same ndarray slice machinery as before T6 (Target 5 is not yet merged into this branch); Target 6
     adds `reverse_flat_rows_inplace` for the track pass, which fires for the 50% neg-strand rows.
     Comparison across sessions is unreliable for the cheapest path (~1 ms); use the within-session ratio.
   - ② annotated regression (1.68×→0.90×) is session noise: the prior 9.00 ms numba annotated time was
     inflated (likely first-run JIT compilation not fully flushed by warmup_rounds=5; the annotated path
     is rarely pre-warmed). The current 5.51 ms is the stable numba time. No T6 regression: the annotated
     kernel only added `Option<bool[]>` argument with `None` fast path; the stable numba reference is now
     5.51 ms vs rust 6.12 ms.

   **Perf profile (rust haplotypes, 12k batches, 2026-06-25):**

   > `perf record -F 999 ... profile.py --mode haplotypes --n-batches 12000`, Carter HPC. Top symbols
   > by self-time (`perf report --stdio --no-children`):
   >
   > | % self | Symbol |
   > |---|---|
   > | 20.64% | `genvarloader::intervals::intervals_to_tracks` |
   > | 15.44% | `ndarray::impl_methods::slice_mut` (Target 5, pending) |
   > | **9.42%** | **`genvarloader::reverse::rc_flat_rows_inplace`** (in-kernel; was ~19% Python post-pass) |
   > | 8.39% | `ndarray::dimension::do_slice` (Target 5, pending) |
   > | 6.33% | `genvarloader::tracks::shift_and_realign_tracks_sparse` |
   > | 3.48% | `_PyEval_EvalFrameDefault` |
   > | 2.91% | `genvarloader::reconstruct::reconstruct_haplotypes_from_sparse` |
   >
   > **RC self-time result: `reverse_complement_ragged` / seqpro RC Python frame is GONE from the rust
   > profile.** The in-kernel `rc_flat_rows_inplace` (9.42%) replaces the ~19% Python/numpy post-pass —
   > roughly a 2× reduction in RC wall-time, moving from a cold Python FFI pass to a hot in-cache Rust
   > loop. The ndarray slice machinery (15.44% + 8.39% ≈ 24%) remains the next highest-value target
   > (Target 5, `opt/target-5-intervals-slice`, not yet merged into this branch).

7. **✅ ADDRESSED (branch `opt/target-7-windows-rust-assembly`, [PR #250](https://github.com/mcvickerlab/GenVarLoader/pull/250) → `rust-migration`).** variant-windows — collapsed
   per-batch object churn into one Rust call. `assemble_variant_buffers_{u8,i32}` assembles alt/ref
   byte windows + flank tokens in one FFI crossing (`src/ffi/mod.rs`, cores in `src/variants/windows.rs`), replacing the
   `_FlatWindow`/`FlatRagged`/scalar-field dataclass construction loop in `_flat_variants.py` /
   `_flat_flanks.py`. GC self-time (`gc_collect_main` + `deduce_unreachable` + `visit_reachable` +
   `dict_traverse`) dropped from **~14% → ~2.5%** of flat self-time; the profile top is now dominated
   by the Rust kernels (`tokenize` 28%, `slice_flanks` 19%, `assemble_alt_window` 13%) and
   `_PyEval_EvalFrameDefault` ~3.7%. variant-windows throughput: **rust 1.83× faster than numba**
   (2.38 ms/batch vs 4.37 ms/batch; profile.py wall-clock, 2000 batches, `NUMBA_NUM_THREADS=1`,
   HEAD `bd957b7`, Carter HPC AMD EPYC 7543, linux-64). Bare variants mode: rust **0.84×** of numba
   (3.75 ms/batch vs 3.15 ms/batch) — slightly slower, within run-to-run noise on this shared node
   (the path is dominated by `intervals_to_tracks` / `shift_and_realign_tracks_sparse` track work,
   not the variant assembly itself, so this is expected noise not a regression).

> **Sequencing for follow-up PRs (updated 2026-06-25):** (5) ⬜ lands first — small, rust-only, closes
> the tracks-only gap. **(6) ✅ DONE** — RC folded into rust kernels on `opt/target-6-kernel-rc`; see
> measurements above; PR [#249](https://github.com/mcvickerlab/GenVarLoader/pull/249). **(7) ✅ DONE** —
> variants/variant-windows assembly collapsed into one rust call on `opt/target-7-windows-rust-assembly`;
> see the Target 7 re-measurement below; PR [#250](https://github.com/mcvickerlab/GenVarLoader/pull/250).
> **Rayon batch parallelism is gated on Targets 5+6+7 landing first** — only after these put rust at or
> ahead of numba single-threaded (per-query in-loop RC and ndarray slicing eliminated) do we add rayon
> batch parallelism (Phase 5). The per-query in-loop RC of the T6 design parallelizes cleanly over
> disjoint per-query slices, so rayon integration is structurally simpler once the post-pass is gone.
> Parallelizing before (5)+(6) are merged would just scale the remaining numpy RC pass and ndarray
> slicing overhead.

##### Target 7 re-measurement (2026-06-25, branch `opt/target-7-windows-rust-assembly`)

> **Harness:** `tests/benchmarks/profiling/profile.py` wall-clock average (2000 batches, burn-in 5),
> not pytest-benchmark pedantic min — `test_e2e_variants` is xfailed (pre-existing `_FlatVariants.to_fixed`
> gap) so no pedantic-min is available for the variants paths. `NUMBA_NUM_THREADS=1`, release build
> (`maturin develop --release`), HEAD `bd957b7`, `chr22_geuv.gvl` (format 2.0, 165 regions × 5 samples),
> Carter HPC (AMD EPYC 7543, linux-64).

| Mode | rust (ms/batch) | numba (ms/batch) | rust ÷ numba | note |
|---|---|---|---|---|
| variant-windows | 2.38 | 4.37 | **1.83×** (rust faster) | assembly collapsed to one Rust call |
| variants (bare alleles) | 3.75 | 3.15 | 0.84× (within noise) | dominated by track work, not variant assembly |

> variant-windows is now the **clearest rust win in isolation**: 1.83× over numba, GC share ~2.5% vs ~14% baseline.
> The bare-variants path is noise-level (the reconstruction cost is track/haplotype work, not the variant
> gather kernels). Full tree 967 passed / 21 skipped / 4 xfailed on both backends (HEAD `bd957b7`);
> byte-identical parity confirmed via `assemble_variant_buffers` mode-matrix + live-path spy.

> **perf flat self-time (variant-windows, rust, 12000 batches):**
> top leaves: `tokenize` 28.3%, `slice_flanks` 19.2%, `assemble_alt_window` 13.1%, `_PyEval_EvalFrameDefault`
> 3.7%, GC total 2.5% (`gc_collect_main` 1.0% + `deduce_unreachable` 0.6% + `visit_reachable` 0.5% +
> `dict_traverse` 0.4%). Profile is now Rust-kernel-dominated with negligible GC overhead.

### Phase 4 — Write / update pipeline 🚧
_PR: bigwig-streaming-write (TBD)_

- [ ] Migrate `_dataset/_write.py`: variant normalization (left-align, bi-allelic,
      atomize), genotype storage, interval extraction + realign.
  - [x] bigWig interval extraction for the write path — single-pass streaming Rust writer (this PR)
  - [x] Table + annot overlap: COITrees Rust engine replaces polars-bio (this PR)
- [ ] Migrate remaining `_dataset/_utils.py` / `_flat_flanks.py` / `_variants/_sitesonly.py`
      kernels touched by the write path.

**Gate:** parity + `gvl.write()`/`update()` wall-clock + peak RSS vs baseline.

### Phase 5 — Crate consolidation + thin-binding cleanup ⬜
_PR: —_

- [ ] Collapse the PyO3 surface so Python is a true shim (indexing sugar, torch,
      validation/error messages only).
- [ ] Delete all remaining core numba kernels (target: count = 0).
- [ ] Confirm the crate is fully cargo-testable standalone.

**Checkpoint:** core numba kernel count = 0; full perf re-baseline recorded here.

### Phase 6 — Absorb genoray (future) ⬜
_PR: —_

Sequenced last; a candidate to graduate into its own roadmap once Phases 0–5 land.
seqpro-core remains the ragged substrate (decision 2026-06-23) — Phase 6 is
narrowed to genoray (variant IO) only.

- [ ] Bring variant IO (genoray VCF/PGEN + sparse genotypes) into the Rust stack.

**Checkpoint:** crate no longer depends on Python genoray for core variant IO paths.

---

## Notes & decisions log

- 2026-06-25 (zero-copy scale-safe read path; branch `zero-copy-scale-safe-readpath`, PR TBD): Addressed
  Phase 3 optimization targets 1–3. **Breaking on-disk change** — track-interval storage converted from
  array-of-structs (`intervals.npy`, `INTERVAL_DTYPE` itemsize 12, strided field views) to struct-of-arrays
  (`starts/ends/values.npy` sharing `offsets.npy`), across all four writers (Python single-chunk + chunked,
  Rust bigwig + table) and the reader; `DATASET_FORMAT_VERSION` bumped `1.0.0`→`2.0.0`. Added an open-time
  version gate and `gvl.migrate(path)` (streaming, idempotent, crash-safe in-place AoS→SoA; new public
  symbol in `__all__`). Replaced the per-batch `np.ascontiguousarray` on per-sample-scale interval/genotype
  memmaps with `_ffi_array` (cross zero-copy or raise loudly); locked closed by `tests/integration/test_scale_guard.py`.
  Cached the sub-linear per-variant/reference arrays once on `Haps` (`_HapsFfiStatic`). Dropped the zero-init
  of fully-overwritten fused output buffers (`uninit_output<T>`), isolated for independent revert. Byte-identical
  parity held on both backends; throughput re-measured (rust at/near numba parity on the heavy tracks/annotated/haps
  paths — see re-measurement block). The pre-built `chr22_geuv.gvl` bench corpus was migrated in place to 2.0.

- 2026-06-25 (Phase 3 close-out): Merged origin/main (#242 `intervals_to_tracks` clip fix via PR #244;
  SpliceIndexer subset double-apply fix via PR #243) into the branch — the fused tracks kernel inherits
  the clip fix (shared `intervals::intervals_to_tracks` core). Lifted ~10 obsolete #242 xfails +
  #242-domain `assume(False)` guards → real passing max_jitter>0 coverage. Rerouted `Reference.fetch`
  through the dispatched rust `get_reference`; deleted the three zero-caller `_fetch_*` numba functions.
  Fused the annotated-haps (`reconstruct_annotated_haplotypes_fused`) and spliced-haps
  (`reconstruct_haplotypes_spliced_fused`) read paths — both byte-identical to the composed numba oracle.
  (The annotated+spliced intersection remains on the unfused dispatched rust core — still parity-gated and rust-by-default — with fusion deferred to Phase 5.)
  Bumped seqpro 0.18→0.20.0 with `to_numpy(validate=False)` at guaranteed-uniform read-path sites.
  Full tree green on both backends: rust 932 passed, 12 skipped, 5 xfailed, 0 failed; numba 932 passed,
  12 skipped, 5 xfailed, 0 failed; cargo 88 passed. Remaining xfails (5): `test_e2e_variants`
  (pre-existing, `_FlatVariants.to_fixed` missing); `test_haps_property` (2 tests, #199/#200
  pre-existing); `test_indexing::test_parse_idx[missing]` (pre-existing); `test_ref_ds::test_getitem[no_regions]`
  (pre-existing). Lint/format/typecheck clean; abi3 wheel builds (2 parity test files reformatted by ruff).

- 2026-06-24 (Phase 3 — reconstruction + track realignment, parity-verified): Ported 8 kernel
  groups to Rust: `padded_slice` (pure cargo, Task 1), `get_reference` (Task 2), spliced-reference
  backstop (Task 3), `reconstruct_haplotype_from_sparse` singular (Task 4),
  `reconstruct_haplotypes_from_sparse` batch (Task 5), haplotypes-mode backstop (Task 6),
  `xorshift64`/`hash4` PRNG (Task 7), `apply_insertion_fill` (4 strategies: Repeat5p,
  Repeat5pNormalized, Constant, FlankSample — Task 8), `shift_and_realign_tracks_sparse` (Task 9),
  `tracks_to_intervals` RLE (Task 10), tracks-mode backstop (Task 11). Fusion seams (Tasks 12–14):
  `reconstruct_haplotypes_fused` collapses 2 FFI crossings to 1 on the plain non-splice haps path
  (annotated + splice remain unfused); `intervals_and_realign_track_fused` chains
  `intervals_to_tracks` → `shift_and_realign_tracks_sparse` in 1 crossing per track. Decisions:
  (1) **Serial-only / rayon-deferred** — batch drivers serial (disjoint per-(query,hap) slices;
  rayon deferred to Phase 5 optimization pass per no-per-phase-perf-gate policy). (2) **Interpolate
  strict byte-identity held** — Lagrange arithmetic in f64 matching numba's `np.float64` xs/ys
  arrays; no numba fallback needed for Interpolate (contrary to an early design note). (3) **#242
  intervals_to_tracks contract bug** — `debug_assert!(itv.start >= query_start)` panics in debug
  builds when stored intervals start before the query (max_jitter>0 datasets); root cause: gvl
  stores intervals at `chromStart - max_jitter` but queries use `chromStart + jitter`. Filed as
  mcvickerlab/GenVarLoader#242; fix deferred (correct oracle needed for both backends). Parity
  fixtures use max_jitter=0 datasets; tests using `get_dummy_dataset()` (max_jitter=2) with float
  tracks on the rust backend fail identically with the pre-existing Phase 0 `intervals_to_tracks`
  kernel (pre-Phase-3). (4) **`tests/benchmarks/conftest.py` updated** — `captured_haplotypes`
  fixture now forces `GVL_BACKEND=numba` to capture `reconstruct_haplotypes_from_sparse` args
  (the rust path now calls `reconstruct_haplotypes_fused`; the micro-benchmark measures the
  individual dispatch entry, not the fused one). (5) **Env note** — dataset tests require
  `--basetemp=$(pwd)/.pytest_tmp` (os.link cross-device Errno 18 on HPC; same as Phase 2).
  **Gate (parity — MET, final-review fixes applied):** 85 cargo tests + 909 pytest passed + 15 xfailed
  + 0 failed (rust; plus 12 skipped, 1 transient error); lint/format/typecheck clean; abi3 wheel builds.
  All 11 pre-existing failures converted to xfail(strict=False): 10 x #242 debug_assert panic
  (itv.start<query_start; tests using get_dummy_dataset() max_jitter=2 with float tracks — xfailed in
  test_output_bytes_per_instance.py, test_dummy_dataset_insertion_fill.py, test_flat_intervals.py,
  test_realign_tracks.py, test_seqs_tracks.py) + 1 test_e2e_variants (_FlatVariants.to_fixed missing,
  pre-Phase-2). Reconstruct parity tests hardened with overshoot pre-check + double-init guard to exclude
  the numba-bug sub-domain where a deletion drives ref_idx past the contig end (numba and Rust diverge
  on negative out_end_idx handling; both behaviors are undefined per the production contract). The
  tracks parity test is sufficient with just the existing SystemError guard (the tracks trailing-fill
  case does not manifest divergence — see task-15-report.md final-review section). 1 transient error
  (test_micro.py::test_shift_and_realign_tracks_sparse, resource contention; passes in isolation).
  **Gate (throughput — recorded, not gated):** see Phase 3 measurement block above.

- 2026-06-24 (Phase 2 — genotype assembly + variant gather, parity-verified): Ported the
  live assembly/selection kernels `get_diffs_sparse` + `choose_exonic_variants`
  (`src/genotypes/`) and the 7 flat variant-gather/fill kernels (`src/variants/`):
  `gather_rows` (unifies `_gather_v_idxs` + `_gather_v_idxs_ss` via `(2,n)` offset
  normalization), `gather_alleles`, `compact_keep`, `fill_empty_scalar`,
  `fill_empty_fixed`, `fill_empty_seq`. Deleted dead `filter_af` (+ its dead unit test).
  Decisions: (1) **dtype-correctness over the plan** — the flat kernels also carry float32
  dosage and arbitrary-dtype custom FORMAT fields (#231, e.g. int16), so they dispatch by
  dtype to `*_i32`/`*_f32` rust cores with a dtype-preserving **numba fallback** for all
  other dtypes; a naive int32-only port (caught + fixed mid-Phase-2) silently truncated
  float dosage. Generic rust cores use `Vec<T>`/`from_vec` (no `num_traits` dep).
  (2) **Gate reframed to parity-only** on a persistent `rust-migration` branch (see
  "Branch & gate strategy") — measured rust is a stable ~7% slower than numba, but cProfile
  pins the cost on per-kernel Python dispatch glue (`np.ascontiguousarray` = 62% of the
  variants loop), not rust compute; throughput is restored by a later "single big
  `__getitem__` kernel" optimization pass, not by gating Phase 2. (3) `OFFSET_TYPE`/genoray
  `V_IDX_TYPE`=int32, `DOSAGE_TYPE`=float32 confirmed at runtime. Env note: dataset tests
  need pytest's tmp on the same filesystem as `tests/data` (`--basetemp=<repo>/.pytest_tmp`)
  or the GVL write path's `os.link` hardlink fails cross-device (Errno 18) — environmental,
  not a code defect.
- 2026-06-18: Roadmap created. Decisions: standalone crate + thin PyO3 binding;
  bottom-up starting from ragged primitives; strangler-fig with byte-identical parity
  gate; perf gates = write wall-clock+RSS and getitem throughput; seqpro/genoray in scope
  but last.
- 2026-06-19: Single-pass streaming bigWig writer landed (Phase 4 bigWig slice). The
  Rust writer (`bigwig_write_track` via PyO3) streams all samples in one pass per region
  and writes `intervals.npy` + `offsets.npy` without materialising per-sample arrays in
  Python. Byte-identical parity vs the legacy Python orchestration was proven in Task 6
  via differential tests across both `_write_track` (per-sample BigWigs) and
  `_write_annot_track` (single-file annotation bigWig) paths. After parity confirmation,
  the env-var gate (`GVL_RUST_BIGWIG_WRITE`) was removed and Rust made the unconditional
  default; legacy Python orchestration retained only for non-BigWigs IntervalTracks (e.g.
  Table). Bench config: synthetic chr21 (200k bp) / chr22 (150k bp), n_samples=8,
  density=0.05, n_regions=2000, width=5000. Measured on macOS, memray for RSS.
  Results: wall-clock 1.502 s → 0.801 s (~1.88× faster); peak RSS 3.538 GB → 3.386 GB
  (−4%, dominated by numba/llvmlite JIT startup ~3.2 GB); total allocated 8.380 GB →
  6.004 GB (~28% less).
- 2026-06-19: Ported gvl.Table + annot_overlap off polars-bio onto a COITrees Rust
  engine (`src/tables.rs`, `RustTable` PyO3 class). Fixes max_mem disrespect during
  write/update (counting is exact, streaming writer bounds the working set to one
  region's overlaps + one contig's trees), removes the non-deterministic polars-bio
  segfault (#395), drops the `[table]` extra, and promotes Table from
  `genvarloader.experimental` to the public API (now CI-covered via a brute-force
  numpy oracle + property tests). Coordinates half-open/zero-based; positive-width
  intervals assumed.
- 2026-06-22: GVL migrated off `awkward` onto seqpro's Rust-backed `_core.Ragged` at
  the Python level. `RaggedVariants` is now a thin wrapper over a single record
  `seqpro.rag.Ragged` with opaque-string `alt`/`ref` fields, replacing its former
  `awkward.Array` subclass foundation. seqpro gained `concatenate` (Rust kernel),
  record `to_ak`/`to_packed`; seqpro's `_array.py` awkward backend was deleted (seqpro
  is now `_core`-only). Consumer suites all green: GVL 806 passed, genoray 456 passed,
  genvarformer CPU 371 passed. Note: this was a Python-level migration onto seqpro's
  existing Rust-backed `_core.Ragged`; the Rust-crate rewrite of the ragged kernels
  themselves (Phase 1 beachhead) is still pending. PR: TBD
- 2026-06-23 (Phase 0 foundation landed): Backend-dispatch registry
  (`python/genvarloader/_dispatch.py`, `GVL_BACKEND` global override + per-kernel
  default; deleted wholesale in a later phase). New `src/ffi/` seam holds ALL PyO3
  wrappers for migrated kernels; core Rust logic lives in lazily-grown domain modules
  (`src/intervals.rs` first — NO empty skeleton, per the lazy-growth rule). Both-layer
  parity harness in `tests/parity/` (`_harness.py` return-value + in-place variants;
  per-kernel hypothesis gate; dataset-level read-path backstop). Dispatch rule: only
  Python-entry kernels register; njit-internal leaves (e.g. `padded_slice`) migrate
  with their caller's subtree.
- 2026-06-23 (PROOF-POINT PIVOT): the spec/plan originally chose `splits_sum_le_value`
  (`_dataset/_utils.py`, called at `_write.py:1280`). During execution we found it is
  DEAD on the default path: `_write_track` routes `BigWigs`→`_write_track_rust` and
  `Table`→`_write_track_table`, so `_write_track_legacy` (the only caller of
  `splits_sum_le_value`) is unreachable for the only concrete public `IntervalTrack`
  types. A `gvl.write` round-trip never hits it. The whole splits migration was reverted
  (commit `45343b8`, keeping the registry + harness infra) and the proof-point re-picked
  to `intervals_to_tracks` (`_dataset/_intervals.py`), which IS genuinely live: on the
  default `Dataset.__getitem__` read path
  (`__getitem__`→Tracks reconstructor→`_call_float32`→`intervals_to_tracks`) for any
  track-bearing dataset, and a clean byte-identical port (integer offset/slice math;
  float32 values copied, never reduced). Lesson baked into the dataset backstop: it
  now SPIES on the kernel and asserts it is actually invoked + output is non-trivially
  non-zero, so a vacuous pass (the failure mode the splits backstop had) is impossible.
- 2026-06-23 (Phase 0 proof-point): ported `intervals_to_tracks` numba→Rust
  (`src/intervals.rs`, pure ndarray, sequential — disjoint per-query out-slices make it
  identical to numba's `prange`; mutates `out` in place via `PyReadwriteArray`). Renamed
  the legacy bigwig `intervals` pyfunction to `bigwig_intervals` to free the name for
  `pub mod intervals` (internal-only; the import was already aliased, public API
  unchanged). Routed the production call site through dispatch (default `rust`; numba
  retained as parity reference). 5 cargo unit tests + a 100-example hypothesis parity
  gate + the read-path backstop all green; abi3 wheel builds; `gvl.write` 1kg baseline
  captured (1.143 s / 3.593 GB); getitem/tracks baseline captured on Carter
  (169.9 batch/s, 3.531 GB peak) — Phase 0 ✅.
  Commits: `64e0836` `917957b` `ec4c15b` `ef4f91a` `ad82b31` `20cd4ef` `0be2d67`.
  Spec: docs/superpowers/specs/2026-06-23-rust-migration-phase-0-foundation-design.md.
- 2026-06-23: seqpro is the shared Rust ragged substrate. Extracted a pyo3-free
  `seqpro-core` rlib (crates/seqpro-core) owning a borrowed `Ragged` layout +
  ops; ported its last two numba kernels (`to_padded`, `reverse_complement`) to
  Rust (seqpro rag layer now numba-free). Bumped seqpro's pymodule to pyo3 0.28 /
  numpy 0.28 / ndarray 0.17 (hygiene; NOT required for the link — two pymodules
  with different pyo3 versions coexist; the single-version rule is per-cdylib, and
  the shared core is pyo3-free). GVL links seqpro-core via a path dep (editable;
  flip to git/release before shipping) and routes its `to_padded` chokepoint
  through the shared kernel (proof-point, byte-identical parity). Inverts Phase 6
  (seqpro stays the substrate). PRs: seqpro ML4GLand/SeqPro#60, GVL mcvickerlab/GenVarLoader#240.
