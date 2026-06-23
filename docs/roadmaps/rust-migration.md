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
| `gvl.write()` wall-clock | 1kg chr21/chr22 (vcfixture tier) | _TBD_ | ⬜ |
| `gvl.write()` peak RSS | 1kg chr21/chr22 (vcfixture tier) | _TBD_ | ⬜ |
| `gvl.update()` wall-clock | 1kg chr21/chr22 (vcfixture tier) | _TBD_ | ⬜ |
| `Dataset.__getitem__` throughput | dataloader bench + py-spy A/B | _TBD_ | ⬜ |

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

### Phase 0 — Foundation & harness ⬜
_PR: —_

- [ ] Restructure `src/` into the target module skeleton (empty modules + `ffi/` seam).
- [ ] Build the reusable differential-test harness: run-both-assert-byte-identical +
      property generators on top of `vcfixture` + numpy.
- [ ] Wire `cargo test` into pixi dev tasks.
- [ ] Confirm abi3 wheels build across py310–313 × linux/macOS (standing invariant).
- [ ] Capture baselines (table above): `write()`/`update()` wall-clock + peak RSS,
      `__getitem__` throughput.

**Checkpoint:** harness green; baselines recorded in this file.

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
