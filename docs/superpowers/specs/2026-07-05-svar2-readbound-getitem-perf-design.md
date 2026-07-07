# Profile & optimize the SVAR2 read-bound `Dataset.__getitem__` path

> **Status:** design approved · **Date:** 2026-07-05 · **Repos/branches:**
> GenVarLoader `svar2-m6b-kernel` (PR #266, draft) + genoray `svar-2` (@ `aaf44fd`)
>
> Big picture: genoray `docs/roadmap/svar-2.md` (milestones M6b/M6d/M6e — the
> read-bound gather this work profiles).

## 1. Motivation

`Dataset.__getitem__` for a `.svar2`-backed dataset now dispatches through
`Svar2Haps` (`python/genvarloader/_dataset/_svar2_haps.py`) → the
`*_from_svar2_readbound` FFI kernels → genoray_core `gather_haps_readbound`. This
path builds **zero** interval-search trees and **zero** dense-union per read (the
structural win M6d/M6e were built for).

The only profiling artifacts on disk (`tmp/svar2_mvp/prof_out/e1`) measure the
**retired union oracle** — `SparseVar2Source.reconstruct` → genoray
`overlap_batch` → 68% `SearchTree::build`. That path is no longer on any live
read. **We are profiling the real read-bound path effectively from scratch.**

The PR bench also noted the small-workload latency is dominated by "per-read
Python/numpy cache-slicing overhead," never attributed to specific functions.
This effort attributes it and removes what is removable.

Goal: identify the hottest functions in the live read-bound `Dataset.__getitem__`
path across all three supported modes, then optimize — Python by static
analysis/inspection, hot Rust by inspecting `cargo asm` — with parity preserved.

## 2. Scope

**In scope** — the live read path for the three supported output modes:

- **haplotypes** — `Svar2Haps.get_haps_and_shifts` → `hap_diffs_from_svar2_readbound`
  + `reconstruct_haplotypes_from_svar2_readbound`.
- **variants** — `Svar2Haps._reconstruct_variants` → `decode_variants_from_svar2_readbound`.
- **tracks** — `Svar2Haps.realign_track_block` → `intervals_to_tracks` +
  `shift_and_realign_tracks_from_svar2_readbound`.

Shared Rust spine (candidates for `cargo asm`): gvl-side
`svar2::split_to_flat` / `hap_diffs_svar2` / `reconstruct_haplotypes_from_svar2`
(`src/svar2/mod.rs`, `src/reconstruct.rs`, `src/ffi/mod.rs`); genoray-side
`query::gather_haps_readbound` / `spine::merge_keys`
(`genoray/src/query.rs`, `genoray/src/spine.rs`).

**Out of scope:** the guarded-`NotImplementedError` modes (annotated, spliced,
`min_af`/`max_af`, in-kernel RC, `unphased_union`, variant-windows,
`max_jitter>0` variants — `unphased_union` and variant-windows have since been
*implemented*, see `2026-07-06-svar2-variant-windows-design.md`; the perf/fusion
work this doc scopes remains deferred for both); any on-disk **format** or
**public API** change; the union oracle (`SparseVar2Source`, `overlap_batch`)
except as the parity oracle; `gvl.write` (the write-time ranges cache producer).

## 3. Landing targets (two repos)

| Change site | Repo / branch | Rebuild to profile |
| --- | --- | --- |
| gvl Rust kernels (`src/svar2`, `src/reconstruct`, `src/ffi`) + Python (`_svar2_haps.py`) | GenVarLoader `svar2-m6b-kernel` (PR #266) | `maturin develop --release` |
| genoray Rust kernels (`gather_haps_readbound`, `merge_keys`) | genoray `svar-2` | gvl `maturin develop --release` rebuilds the `genoray_core` **crate path-dep** automatically |

**Wheel caveat:** the Python `genoray` package is a pre-built cp310 wheel
(`pixi.toml`). It only needs rebuilding if a change touches genoray's **Python
API** (e.g. a `find_ranges` dict key). Pure hot-path kernel edits consumed by gvl
through the `genoray_core` crate do **not** need a genoray wheel rebuild.

## 4. Tooling — forced substitution for py-spy

Empirically verified on the Carter node this runs on:

- **py-spy is unusable** — `ptrace_scope=2` + no sudo → "Permission Denied" for
  `dump`/`record`, `--native` or not.
- **Python is 3.10** in `-e dev` → no `perf` trampoline (`-Xperf` is 3.12+), so
  `perf` **cannot resolve Python frames** (Python stays a single opaque DSO).

We recover both halves py-spy would have given from two ptrace-free tools:

| Layer | Tool | Output |
| --- | --- | --- |
| Python fns ("python fns") | **cProfile** (stdlib) + **pyinstrument** (added to pixi deps for a low-overhead statistical wall-clock call-tree cross-check) | per-function cumulative/total time → the Python functions to inspect |
| Native/Rust ("total native %") | **perf** `record -g --call-graph fp -F 199`, run on `.pixi/envs/dev/bin/python` **directly** (not via `pixi run` — the launcher eats ~60% of samples), built with `-C force-frame-pointers=yes` | DSO split (native % = "total native %"), Rust symbol self-time, and the fp call-graph → the Rust fns to `cargo asm` |

`pyinstrument` is added to `pixi.toml` deps (gvl, and genoray if it profits).

## 5. Benchmark harness (rebuilt, real path)

The existing `tmp/svar2_mvp` drivers profile the **wrong** (union) path and
reference a **stale** store path (`repos/for_loukik/...`). Stores actually live at
`/carter/users/dlaub/projects/svar2_mvp/{germline,somatic}.{svar,svar2}`;
reference FASTA `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa` is present.

Add a driver under `tmp/svar2_mvp/` that exercises the **live**
`Dataset.open(path, reference=REF, svar2=<store>).with_seqs(mode)[regions, samples]`
warm loop for `mode ∈ {haplotypes, variants, tracks}`, over both cohorts
(germline 3202-smp, somatic 16007-smp), same fair workload as the PR bench
(fixed region set × all samples, warm caches, median of N). One code path per
capture (mirrors `prof_driver.py`) so cProfile/perf attribute cleanly. A tracks
run needs a BigWig or table track attached to the dataset; haplotypes/variants
reuse the PR bench setup.

## 6. Optimization method (measure → confirm → fix → re-measure)

Profile first; fix only what ranks hot. Static inspection already surfaces
concrete candidates to **confirm against the profile before touching**:

- **Redundant double gather (Python/FFI, high confidence):**
  `get_haps_and_shifts` calls `hap_diffs_from_svar2_readbound` *and then*
  `reconstruct_haplotypes_from_svar2_readbound` per contig group — **both** run
  the full `gather_haps_readbound` + `split_to_flat` internally, so genoray's
  gather + AoS→SoA marshalling executes **twice per haplotype/track read**.
  Candidate fix: gather once (fuse diffs into the reconstruct gather, or return
  diffs from a single kernel), so the read draws shifts and reconstructs off one
  gather.
- **Python FFI-shaping (`_svar2_haps.py`):** per-contig-group Python loops,
  repeated `ascontiguousarray` copies in `_gather_inputs`, the
  `_ragged_arange_gather` permutation passes — vectorize / drop copies where the
  profile justifies.
- **Rust hot fns via `cargo asm`:** likely `svar2::split_to_flat` (AoS→SoA copy),
  `svar2::merge_hap`/`decode_alt`, `gather_haps_readbound`, `merge_keys`,
  `hap_diffs_svar2`, and the `reconstruct_haplotypes_from_svar2` inner kernel.
  `cargo asm` targets: bounds-check elision, autovectorization of copy/merge
  loops, alloc churn (`SpecFromIter`/`_int_malloc` were hot even in the old run).
  Because these functions are independent, the `cargo asm` phase **fans out one
  subagent per hot function in parallel** (Sonnet implementers, each in its own
  git worktree so same-file edits don't clobber), and **each fix carries its own
  per-function parity test** (a focused `cargo test` asserting byte-identical
  output on representative inputs) alongside the instruction-count delta. The
  branch owner then merges the worktree fixes sequentially behind the full-tree
  parity gate, dropping any that regress or fail parity.

## 7. Measurement discipline & parity gate

Per hard-won project lessons (shared-node noise):

- **No cross-session absolute wall-clock claims.** Gate each optimization on
  **same-session before/after** plus a **deterministic `perf stat -e
  instructions,cycles`** delta (instruction count is noise-free). Record the
  instruction-count delta per change.
- **Parity is a hard gate.** The read-bound kernels are byte-identical to the
  union oracle. After every change: `pixi run -e dev pytest tests -q` (full tree
  — the svar2 suite is 31/31, plus the parity oracle), and for Rust changes
  `maturin develop --release` **first** (pytest imports the stale `.so`
  otherwise), plus `cargo test` on both repos. Any divergence blocks the change.
- The two documented intentional non-identities (pure-DEL ALT `b""`; SVAR1
  `max_ends` tie under-extension) are pre-existing and untouched.

## 8. Deliverables

1. A profiling report (cProfile + pyinstrument + perf DSO/symbol/callgraph
   tables, per mode × cohort) under `tmp/svar2_mvp/prof_out/`.
2. Confirmed optimizations implemented: gvl-side on `svar2-m6b-kernel` (#266),
   genoray-side on `svar-2` — each parity-gated, each with a recorded
   same-session instruction-count delta. The `cargo asm` phase is executed as a
   **parallel subagent fan-out** (one worktree-isolated Sonnet subagent per hot
   function, each with a per-function parity test), merged sequentially behind
   the full-tree parity gate.
3. `pyinstrument` added to `pixi.toml`.
4. No format/API/doc-surface changes (read-path internals only), so the
   skill/api.md/docs gates do not apply; the genoray roadmap needs no milestone
   flip (this is perf work under shipped M6b/M6d/M6e, noted in passing if a
   kernel signature changes).

## 9. Open questions

- Whether the double-gather fix needs a **new fused kernel signature** (a genoray
  `svar-2` API touch → genoray wheel rebuild) or can be done gvl-side by having
  the reconstruct kernel also return diffs (no genoray API change). Resolve after
  the profile confirms the gather is actually the dominant cost.
