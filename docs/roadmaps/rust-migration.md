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

**Eventual scope:** seqpro (the `Ragged` data structure + rag ops) and genoray (VCF/PGEN
variant & sparse-genotype IO) are sibling Python deps today. They are **in scope for the
full Rust stack**, but sequenced last (Phase 6) and may graduate into their own roadmap.

### Target crate layout

Grown from today's `src/{lib,bigwig}.rs`:

```
src/
├── lib.rs            # pymodule registration only
├── ragged/           # ragged layout (offsets+data) + ops  ← Phase 1 beachhead
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
`seqpro.Ragged` / `genoray` objects cross the boundary (zero-copy where possible). By
Phase 6 the crate stops depending on Python seqpro/genoray entirely.

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

### Phase 1 — Ragged primitives + layout (beachhead) ⬜
_PR: —_

The foundation everything sits on. Bottom-up.

- [ ] Define the native ragged layout in Rust (offsets + data buffers).
- [ ] Implement the ops gvl uses: lengths/offsets, slice, gather, `to_padded`,
      reverse-complement helpers.
- [ ] Zero-copy interop with `seqpro.Ragged` at the boundary (construct-from / view-as).
- [x] Remove `awkward` from the foundation layer. (Done at the Python level: GVL migrated onto seqpro's Rust-backed `_core.Ragged`; Rust-crate kernel rewrite is a separate pending step.)
- [ ] Differential parity vs `_ragged.py` / current seqpro paths.

**Checkpoint:** parity green. Foundational — no perf gate, but record incidental wins.
Relevant prior work: [[project_ragged_assembly_bottleneck]].

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

### Phase 6 — Absorb seqpro / genoray (future) ⬜
_PR: —_

Sequenced last; a candidate to graduate into its own roadmap once Phases 0–5 land.

- [ ] Bring ragged primitives fully in-house — drop the seqpro hot-path dependency.
- [ ] Bring variant IO (genoray VCF/PGEN + sparse genotypes) into the Rust stack.

**Checkpoint:** crate no longer depends on Python seqpro/genoray for core paths.

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
