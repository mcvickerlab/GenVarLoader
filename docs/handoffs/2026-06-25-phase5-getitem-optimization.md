# Handoff: Phase 5 — fully optimize `Dataset.__getitem__` (targets 5, 6, 7 + rayon)

**Date:** 2026-06-25
**Status:** Not started. Four parallel-ready workstreams.
**Audience:** GenVarLoader maintainers / per-workstream sessions.
**Roadmap:** `docs/roadmaps/rust-migration.md` — Phase 5 ⬜, "Optimization targets — round 2" (targets 5/6/7).
**Base branch:** `zero-copy-scale-safe-readpath` (format 2.0 SoA + zero-copy FFI + sub-linear cache + uninit buffers; PR TBD). All four workstreams branch from here.

## TL;DR

Phase 3 profiling (de-noised `test_e2e.py` benchmark + `perf` on the Python process) left three
single-thread deficits on the read path, then rayon batch parallelism as the capstone:

| # | Workstream | What | Kind | Parallel? |
|---|---|---|---|---|
| **5** | tracks-only ndarray slicing | hoist `out.as_slice_mut()` in `intervals_to_tracks`, drop per-interval `SliceInfo` | rust-only, **byte-identical** | now |
| **6** | strand reverse-complement | fold RC into **all** reconstruct/track kernels (incl. splice); delete `reverse_complement_ragged` | parity-gated (strand=-1) | now |
| **7** | variant-windows assembly | replace the per-batch `_FlatWindow`/`_FlatAlleles` object graph with **one Rust call** returning flat `(data, offsets)` | parity-gated | now |
| **rayon** | batch parallelism | `par_iter` over disjoint per-query slices in the fused kernels | parity-trivial (disjoint) | **after 5/6/7 merge** |

**Run 5, 6, 7 concurrently. Rayon is blocked until 5+6+7 land** — the roadmap is explicit that
parallelizing before the single-thread work just scales the numpy RC pass (6) and the ndarray
slicing (5). Each workstream is its own branch + its own parity-gated PR.

The measured starting point (branch `zero-copy-scale-safe-readpath`, `chr22_geuv.gvl`, `with_len(16384)`,
BATCH=32, `NUMBA_NUM_THREADS=1`, Carter EPYC 7543), **min rust ÷ min numba** ms/batch:

| Mode | rust ÷ numba | note |
|---|---|---|
| tracks-only | **0.63×** (rust slower) | target 5 fixes this |
| tracks (seqs + read-depth) | 0.95× | shares the target-5 kernel |
| haplotypes | 0.94× | target 6 is its biggest sink (~19% self / 28% incl RC) |
| annotated | **1.68×** (rust faster) | already a win post-format-2.0 |

---

## Shared context (every session reads this first)

### Where this sits

Phases 0–3 ported the read path to Rust behind a per-kernel dispatch registry
(`python/genvarloader/_dispatch.py`, default `rust`, `GVL_BACKEND=numba` override). The numba
kernels are **retained as registered parity oracles** (deleted wholesale later in Phase 5 — NOT in
these workstreams). The read path is fused: `__getitem__` → `QueryView.recon(...)` → one of the
fused FFI kernels in `src/ffi/mod.rs`.

### How to measure (use this, not py-spy `--native`)

py-spy `--native` slows the deep-stack haplotype paths ~10× and times out. Use `perf` on the Python
process — no sudo on Carter (`perf_event_paranoid=2`), near-zero overhead, resolves
`genvarloader.abi3.so` Rust symbols:

```bash
NUMBA_NUM_THREADS=1 perf record -F 999 -o p.data -- .pixi/envs/dev/bin/python \
    tests/benchmarks/profiling/profile.py --mode <mode> --n-batches 12000
perf report --stdio --no-children -i p.data        # flat self-time, Rust symbols resolved
```

`profile.py --mode {haplotypes,annotated,tracks,tracks-seqs,variants,variant-windows}`. Run 8–25k
batches so steady state drowns import/JIT. For the rust↔numba ratio use the de-noised
`pytest-benchmark` harness in `tests/benchmarks/test_e2e.py`: `_bench_indexing` uses
`benchmark.pedantic(iterations=10, rounds=50)` so per-batch OS jitter averages out — compare the
**min** (cleanest CPU-bound estimate), not the mean. Build release first:
`pixi run -e dev maturin develop --release`.

### Parity (the landing gate)

Every workstream lands only when output stays **byte-identical** to the numba oracle. The harness is
`tests/parity/` (`_harness.py` run-both-assert-byte-identical, return-value + in-place variants) plus
hypothesis property generators. The dataset-level backstop (`tests/parity/test_dataset_parity.py`)
spies on the kernel to prove it actually runs on the live `__getitem__` path (guards against vacuous
passes). Targets 5/7 are byte-identical by construction; target 6 is gated on **strand=-1** datasets
(see its section). Run both backends:

```bash
pixi run -e dev pytest tests/parity -q                      # rust default
GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q    # oracle
pixi run -e dev cargo-test                                  # rust unit tests
```

### Before pushing

Per `CLAUDE.md`: run the **full tree** on both backends before any push that touches shared code
(`pixi run -e dev pytest tests -q`, then `GVL_BACKEND=numba …`) — scoped runs skip `tests/unit/`.
Lint/format/typecheck: `pixi run -e dev ruff check python/ tests/ && ruff format … && typecheck`.
Update `docs/roadmaps/rust-migration.md` (tick the target, record the re-measured ratio, set the PR
link) as part of the work.

### Parallel-session coordination

- **One branch per workstream**, all off `zero-copy-scale-safe-readpath`. Use a git worktree per
  session to avoid stepping on each other's working tree.
- **File-overlap map** (plan rebases around these):
  - Target 5: `src/intervals.rs` only (+ its cargo tests). **No overlap** with 6/7.
  - Target 6: `src/intervals.rs` (track reverse), `src/ffi/mod.rs` + the reconstruct/track cores
    under `src/{reconstruct,tracks,intervals}/`, `python/genvarloader/_dataset/_query.py`,
    `_reconstruct.py`. **Overlaps target 5 in `intervals.rs`** and target 7 in `_query.py` — see below.
  - Target 7: `python/genvarloader/_dataset/_flat_variants.py`, `_flat_flanks.py`, new
    `src/variants/` code + `src/ffi/mod.rs`. **Overlaps target 6 in `src/ffi/mod.rs`** (additive — new
    pyfunctions, low conflict risk).
- **Merge order:** 5 first (smallest, rust-only), then 6 and 7 in either order; rebase the later ones.
  Rayon last, after all three are on the base branch.
- **HPC gotcha:** dataset tests need pytest's tmp on the same filesystem as `tests/data`
  (`--basetemp=$(pwd)/.pytest_tmp`) or the write path's `os.link` hardlink fails cross-device (Errno 18).

### Don't regress the format-2.0 read path

The base branch replaced per-batch `np.ascontiguousarray` on per-sample-scale memmaps with `_ffi_array`
(cross zero-copy or raise loudly) and caches sub-linear per-variant arrays on `Haps.ffi_static`
(`_HapsFfiStatic`). `tests/integration/test_scale_guard.py` fails if any per-batch
`np.ascontiguousarray` materializes a sample-scale memmap. Keep that test green — do **not** reintroduce
`ascontiguousarray` on `geno_v_idxs` / `itv_*` / genotype memmaps.

---

## Target 5 — tracks-only ndarray slicing (rust-only, byte-identical)

**Goal:** close the **0.63×** tracks-only deficit — the one read path where rust is clearly slower than
numba — and get rust ahead single-threaded on the cheapest read.

**Evidence (`perf` flat self-time, tracks-only path):** `intervals_to_tracks` 31% + `ndarray::slice_mut`
**11%** + `ndarray::do_slice` **9.5%** ≈ **20.5%** in ndarray slice machinery. Source: the per-interval
`out.slice_mut(s![a..b]).fill(value)` and the `out.fill(0.0)` prelude in
`src/intervals.rs:66` / `:27`. numba compiles `out[a:b] = value` to a direct memset and pays none of this.
tracks-only is the cheapest path (~1.1–1.7 ms) so this fixed per-interval cost dominates with no
sequence work to amortize it.

**Fix:** the `out` buffer is contiguous. Hoist `let out_slice = out.as_slice_mut().unwrap();` once at the
top, then write `out_slice[out_s + s as usize .. out_s + e as usize].fill(value)` and
`out_slice.fill(0.0)` on the raw `&mut [f32]` — dropping per-interval `SliceInfo` construction +
bounds-check. Keep the exact clamp/break semantics (start clamped ≥0, end ≤length, break on
`start >= length`, no-op when `e <= s`) — see the docstring at `src/intervals.rs:3-15`. This kernel is
shared by the combined **tracks** path too, so that improves with it.

**Files:** `src/intervals.rs` (`intervals_to_tracks` + its cargo tests). Nothing Python-side changes.

**Parity:** **byte-identical by construction** — same arithmetic, same write order, just a different way to
address the contiguous buffer. The 8 existing cargo unit tests (`src/intervals.rs:72+`) plus the
`intervals_to_tracks` hypothesis parity gate and the tracks dataset backstop must stay green. No oracle
change.

**Perf gate:** re-measure tracks-only via `test_e2e.py`; target rust ÷ numba ≥ 1.0 (was 0.63×). Record in
the roadmap's re-measurement block.

**Start your session here:**
1. Branch `opt/target-5-intervals-slice` off `zero-copy-scale-safe-readpath`.
2. Read `src/intervals.rs` end-to-end (it's ~220 lines).
3. TDD: the cargo tests already pin the contract — refactor under them, then add a profiling re-measure.
4. Gate: `cargo-test` + `pytest tests/parity -q` (both backends) + tracks-only `test_e2e` re-measure.

---

## Target 6 — fold strand reverse-complement into the kernels (delete the numpy post-pass)

**Goal:** delete the `reverse_complement_ragged` post-pass entirely (incl. the spliced per-element path)
by emitting negative-strand regions already reverse-complemented from the Rust kernels. This is the
**largest single-thread throughput lever** left and it is **backend-agnostic** (numba pays it too) — it
must go before rayon, else we parallelize a numpy pass.

**Evidence (py-spy, no `--native`, self-time):** RC post-pass is haplotypes **~19% self / ~28% inclusive**,
variants **~15% / ~16%**, tracks-only **~10%**. Every negative-strand region triggers a Python/numpy RC
pass *after* reconstruction.

**Current state:** `python/genvarloader/_dataset/_query.py`
- unspliced: `_getitem_unspliced` computes `to_rc = view.full_regions[r_idx, 3] == -1` and does
  `recon = tuple(reverse_complement_ragged(r, to_rc) for r in recon)` (~line 188–190).
- spliced: `_getitem_spliced` builds a **permuted per-element** mask `to_rc_per_elem` via
  `plan.permutation` (the spliced kernel writes pre-spliced bytes in permuted order) and applies the same
  call (~line 259–280).
- `reverse_complement_ragged` (~line 352–410) dispatches by output kind.

**RC semantics per output kind (the contract to reproduce in-kernel):**

| Output kind | Python today | In-kernel behavior |
|---|---|---|
| haplotypes `_Flat` (S1) | `reverse_masked(to_rc, comp=_COMP)` | reverse bytes **and** complement |
| reference `_Flat` (S1) | same | reverse + complement |
| annotated `_FlatAnnotatedHaps` | `reverse_masked(to_rc, _COMP)` | reverse+complement bytes **and reverse** the parallel `var_idxs`/`ref_coords` arrays (no complement on those — order only) |
| tracks `_Flat` (f32) | `reverse_masked(to_rc, comp=None)` | **reverse only**, no complement |
| variants `RaggedVariants` | `rc_(to_rc)` | reverse allele order within each row **and** complement allele bytes (ragged) |
| variant-windows | no-op (returns unchanged) | **skip** — reference-oriented |
| intervals | no-op | **skip** |

`_COMP` is the complement LUT (find it in `_query.py` / seqpro). Confirm exact mapping (incl. `N`,
IUPAC, lowercase if any) and reproduce it in Rust.

**Kernels to thread a per-query `to_rc: &[bool]` through** (`src/ffi/mod.rs`):
- `reconstruct_haplotypes_fused` (`:393`) — haplotypes
- `reconstruct_annotated_haplotypes_fused` (`:604`) — bytes + parallel arrays
- `reconstruct_haplotypes_spliced_fused` (`:521`) — **the hard one**, see below
- `intervals_and_realign_track_fused` (`:848`) — tracks (reverse only)
- `get_reference` (`:728`) — reference
- the variants allele-gather path (`gather_alleles` in `src/variants/`) — `RaggedVariants` RC

**Approach:** each kernel takes the per-query mask; when `to_rc[query]` is set, write that query's output
slice **back-to-front** with complemented bytes (seqs) or plain reversed values (tracks). For annotated,
reverse the parallel `var_idxs`/`ref_coords` slices in lockstep. Do the RC as the kernel writes (or as a
final in-place pass over each query's just-written slice — simpler to get byte-identical first, optimize
second). Mind the interaction with **insertion-fill** and **trailing-fill**: RC must apply to the final
post-fill bytes (same as today, where RC runs after reconstruction completes).

**The splice sub-case:** `reconstruct_haplotypes_spliced_fused` writes pre-spliced bytes in
**permuted** order (`plan.permutation`), and today RC is applied per spliced **element** with
`to_rc_per_elem`. In-kernel, pass the already-permuted per-element `to_rc` and reverse-complement each
spliced element's byte range as it is finalized. Verify the element boundaries you reverse match
`plan.group_offsets`. This is the part most likely to need careful TDD — start from the existing spliced
parity fixtures and add strand=-1 coverage.

**Delete after parity holds:** the `reverse_complement_ragged` calls in `_getitem_unspliced` /
`_getitem_spliced`, the function itself, and the now-dead `to_rc` plumbing in `_query.py`. Confirm no other
caller (`grep -rn reverse_complement_ragged python/`).

**Parity:** byte-identical vs the current post-pass. The default parity fixtures use `max_jitter=0` and may
be strand-agnostic — **add strand=-1 datasets** (mix of + and − regions) to the dataset parity backstop
for every output kind incl. annotated and spliced. Gate both backends. This is the workstream where a
vacuous pass is easiest, so assert the RC actually fires (regions with strand −1 produce RC'd bytes ≠ the
+ strand).

**Perf gate:** re-measure haplotypes/variants/tracks via `test_e2e`; expect the RC self-time gone and the
ratios up. Record in the roadmap.

**Start your session here:**
1. Branch `opt/target-6-kernel-rc` off `zero-copy-scale-safe-readpath`.
2. Read `_query.py:152-410` (both getitem paths + `reverse_complement_ragged` + the `_COMP` LUT), then the
   six kernels in `src/ffi/mod.rs` and their cores.
3. TDD order: reference (simplest, no fill) → haplotypes → tracks (reverse-only) → variants → annotated →
   **splice last**. Land each kind's in-kernel RC behind parity before deleting its post-pass branch.
4. Gate: `cargo-test` + `pytest tests/parity -q` (both backends, with new strand=-1 fixtures) + full tree.

---

## Target 7 — variant-windows assembly in one Rust call

**Goal:** kill the per-batch object churn on the `variant-windows` (and `variants`) flat-output path by
assembling the token/window buffers in **one Rust call returning flat arrays**, eliminating the per-batch
Python object graph. (This is the larger of the three; it effectively starts the windows half of the
deferred single-big-kernel rewrite.)

**Evidence (`perf` flat self-time, variant-windows):** no dominant Rust kernel — the cost is interpreter +
allocator: `_PyEval_EvalFrameDefault` ~8.5%, GC (`gc_collect_main` + `deduce_unreachable` +
`visit_reachable` + `dict_traverse`) **~14% combined**, dict/attr lookups, dynamic-symbol lookup
(ctypes/cffi binding) ~2.3%. The flat-windows assembly allocates many small objects per batch
(`_FlatWindow` / `_FlatVariants` / `_FlatAlleles` / scalar-field dataclasses).

**Current state:** trace `profile.py --mode variant-windows` and `--mode variants` into
`python/genvarloader/_dataset/_flat_variants.py` (`_FlatWindow` `:189`, `_FlatVariantWindows` `:270`,
`_FlatVariants` `:344`) and `_flat_flanks.py` (`_make_window` / ref+alt window builders `:116–220`). These
rebuild dicts of wrapper dataclasses, gather/fill via the `*_i32`/`*_f32` rust cores, and re-wrap, **every
batch**. The Phase-2 rust gather/fill kernels already exist (`src/variants/`,
`gather_rows`/`gather_alleles`/`compact_keep`/`fill_empty_*`) — the win here is collapsing the
**orchestration** that allocates Python objects around them.

**Approach:** add one (or a few) Rust pyfunction(s) in `src/ffi/mod.rs` that take the raw inputs the
windows path needs (gathered v_idxs / alleles / scalar fields + flank/tokenize/LUT params) and return the
final flat `(data, offsets)` token buffers directly — so the Python side constructs **one** `_Flat`/result
wrapper instead of a graph of `_FlatWindow`/`_FlatAlleles`. Reuse the existing `src/variants/` cores
internally. Inventory exactly which fields/windows the consumer actually reads downstream (in
`_query.py` reshape/pad and the flat-output assembly) so the Rust call returns precisely those, no more.

**Files:** new code in `src/variants/` + `src/ffi/mod.rs`; rewrite the assembly in
`_dataset/_flat_variants.py` / `_flat_flanks.py` to call it; keep the public output type
(`_FlatVariants` / `_FlatVariantWindows`) identical from the caller's view.

**Parity:** byte-identical token buffers + offsets vs the current Python assembly, for both `variants` and
`variant-windows`, incl. the flank-tokenize ride-along (`flank_tokens`), the empty-group fill
(`fill_empty_groups` / `DummyVariant`), and the unknown-token path. Note `test_e2e_variants` is a
**pre-existing xfail** (`_FlatVariants.to_fixed` missing) — don't conflate it with a regression; check it
xfails identically at the base before you start.

**Perf gate:** re-measure `variant-windows` and `variants` via `test_e2e`; expect the GC/eval self-time to
drop. Record in the roadmap.

**Start your session here:**
1. Branch `opt/target-7-windows-rust-assembly` off `zero-copy-scale-safe-readpath`.
2. `perf record` the `variant-windows` mode and read the assembly in `_flat_variants.py` / `_flat_flanks.py`
   top-to-bottom; map every per-batch allocation.
3. TDD: pin the current flat-buffer output (data+offsets) for `variants` and `variant-windows` as the
   oracle, then build the Rust call under it.
4. Gate: `cargo-test` + `pytest tests/parity tests/unit -q` (both backends) + `variant-windows` re-measure.

---

## Rayon — batch parallelism (BLOCKED: start only after 5/6/7 are merged)

**Goal:** parallelize the fused kernels' per-query loops with rayon, now that single-thread rust is ahead.

**Why blocked:** the roadmap is explicit — "Only after (5)+(6) put rust ahead single-threaded do we add
rayon batch parallelism — parallelizing first would just scale the numpy RC pass and the ndarray slicing."
Do not start until target 5, 6, and 7 are on the base branch.

**Approach:** the batch drivers are currently serial by deliberate design — per-`(query, hap)` output
slices are **disjoint**, which is exactly why they're embarrassingly parallel and why the serial result
already equals numba's `prange`. Convert the per-query loops in the fused kernels
(`reconstruct_haplotypes_fused`, `intervals_and_realign_track_fused`, the annotated/spliced variants) to
`rayon::par_iter` (or `par_chunks` over disjoint output slices — use `split_at_mut` / `ndarray`
`axis_chunks_iter_mut` to hand each thread a non-overlapping `&mut` slice). Expose a thread-count control
(env var or arg) so benchmarks can pin it; default to rayon's global pool.

**Parity:** **trivial** — disjoint slices, deterministic per-slice work, so output is identical regardless
of thread count. Run the existing parity suite at >1 thread.

**Perf gate:** throughput scaling vs thread count on `test_e2e`. **Re-baseline the whole read path here**
(the roadmap's Phase 5 checkpoint). Note the `NUMBA_NUM_THREADS=1` caveat — for an honest comparison, set
numba threads to match, or report both single- and multi-thread numbers explicitly.

**Start your session here (once unblocked):**
1. Branch off the merged base (with 5/6/7 in).
2. Confirm each fused kernel's per-query output slices are provably disjoint before parallelizing.
3. Gate: `cargo-test` + full parity suite at N>1 threads + a thread-scaling sweep recorded in the roadmap.

---

## Pointer table

| Need | Where |
|---|---|
| Roadmap + targets 5/6/7 detail | `docs/roadmaps/rust-migration.md` (round-2 optimization block) |
| Fused FFI kernels | `src/ffi/mod.rs` (`:66`, `:393`, `:521`, `:604`, `:728`, `:848`) |
| tracks slice kernel | `src/intervals.rs` |
| RC post-pass to delete | `python/genvarloader/_dataset/_query.py` (`reverse_complement_ragged`, getitem paths) |
| windows assembly | `python/genvarloader/_dataset/_flat_variants.py`, `_flat_flanks.py` |
| Phase-2 variant cores (reuse) | `src/variants/` |
| Dispatch registry | `python/genvarloader/_dispatch.py` (`GVL_BACKEND`) |
| Parity harness | `tests/parity/` |
| Perf benchmark | `tests/benchmarks/test_e2e.py`, `tests/benchmarks/profiling/profile.py` |
| Scale guard (don't regress) | `tests/integration/test_scale_guard.py` |
