# Roadmap: Migrate the GVL core to Rust

**Status legend:** ⬜ not started · 🚧 in progress · ✅ done · ⏸️ blocked

This is a living tracker. **Any work that touches the Rust migration must read this file
first and update it as part of the change** — tick completed tasks, record measurements
under the relevant checkpoint, and update the phase status marker + PR link.

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

### Phase 2 — Genotype assembly + variant gather ⬜
_PR: —_

- [ ] Migrate `_dataset/_genotypes.py` kernels (6 numba) onto the Rust layout.
- [ ] Migrate `_dataset/_flat_variants.py` kernels (7 numba).
- [x] Migrate `_dataset/_rag_variants.py`; drop `awkward` from these hot paths. (Done at the Python level: `RaggedVariants` now wraps a single record `seqpro.rag.Ragged`; no numba kernels remain in this file — any remaining numba rewrites are tracked in the unchecked items below.)

**Gate:** parity + `Dataset.__getitem__` throughput vs baseline (target speedup, no
regression).

### Phase 3 — Reconstruction + track realignment ⬜
_PR: —_

The numba bulk and the big read-path win.

- [ ] Migrate `_dataset/_reconstruct.py` + `_dataset/_haps.py`.
- [ ] Migrate `_dataset/_tracks.py` realign (6 numba) + `_dataset/_intervals.py` (4 numba).
- [ ] Migrate `_dataset/_reference.py` (6 numba).
- [ ] Migrate `_dataset/_insertion_fill.py` + `_dataset/_splice.py`.

**Gate:** parity + `Dataset.__getitem__` throughput vs baseline.

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
