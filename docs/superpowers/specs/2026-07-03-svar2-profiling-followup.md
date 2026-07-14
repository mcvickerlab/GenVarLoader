# SVAR2 Profiling Follow-up — Spec

> **Purpose:** turn the SVAR2 MVP benchmark's *unprofiled* latency observations into
> **concrete, attributed numbers** so we can decide (a) how much of the SVAR2 query-latency
> gap is Python-adapter overhead vs Rust hot paths vs memory layout, and (b) how to
> rebalance conversion threads for read-bound (few-contig) builds. **Measurement only — no
> optimization work is in scope here.**

**Date:** 2026-07-03 · **Depends on:** the MVP stores + scripts from
`2026-07-03-svar2-gvl-mvp-validate-benchmark` (plan) and its notes
[`../notes/2026-07-03-svar2-mvp-benchmark.md`](../notes/2026-07-03-svar2-mvp-benchmark.md),
[`../notes/2026-07-03-svar2-mvp-validation.md`](../notes/2026-07-03-svar2-mvp-validation.md).

## Why (background the next session needs)

The MVP benchmark measured SVAR1 (gvl `Dataset` over `.svar`) vs SVAR2 (`SparseVar2Source`
adapter over `.svar2`), all-samples × 3 chr21 regions, warm, median N=5:

| cohort | SVAR1 hap (s) | SVAR2 hap (s) | store SVAR1→SVAR2 |
| --- | --- | --- | --- |
| germline (3202 smp) | 0.0555 | 0.2834 | 5.67× smaller |
| somatic (16007 smp) | 0.3798 | 0.3797 | 1.46× smaller |

**Established (unconfounded):** SVAR2's on-disk store is 1.46–5.67× smaller.

**NOT established — the reasons this spec exists:**
1. The latency comparison is **not apples-to-apples**: the SVAR2 path is the *raw adapter*
   (new Python orchestration — per-call genoray `overlap_batch`, numpy conversions, the
   two-source Rust kernel), **not** the gvl `Dataset`. Prior gvl profiling (Dataset latency
   is Rust-bound, Python-orchestration negligible) does **not** transfer to this adapter.
2. The "SVAR2 scales better with cohort size" reading is **confounded** — germline vs
   somatic differ in AF/density/source, not just sample count.
3. The dense presence matrix is **hap-major** (`genoray:src/query.rs:131-134`, bit
   `hap * n_dense_variants + col`; refutes a `(V,S,P)` guess) — per-hap contiguous — **but**
   `n_dense_variants` is **contig-wide**, so a narrow-region × all-samples read scatters one
   cache line per hap, and presence is read **bit-by-bit** (`get_bit`), not word-parallel.
   Plausible but unmeasured.
4. Conversion is **read-bound**: single-contig builds show `1 concurrent chromosome | 3
   HTSlib decompression threads (7/8 active)`, VCF read/decompress dominating (chr21:
   germline ~11 min, somatic ~2 h). Thread rebalance is untested.

## Environment & tooling (read before profiling)

- **Env:** `pixi run -e default` (only env installed). **Python 3.10.20.**
- **CRITICAL perf/Python caveat:** on **Python < 3.12**, `perf` **cannot** symbolize Python
  stack frames — the `-X perf` / `PYTHONPERFSUPPORT` trampoline that emits `perf`-readable
  Python symbols only exists in **3.12+**
  (see <https://docs.python.org/3/howto/perf_profiling.html>). This env is 3.10, so `perf`
  alone shows resolved **Rust/native** symbols but **opaque** Python frames
  (`_PyEval_EvalFrameDefault` …). Therefore:
  - **py-spy** (`.pixi/envs/default/bin/py-spy`, confirmed present) — sampling profiler that
    resolves **Python** frames on any version and, with `--native`, also unwinds into the
    **Rust** extension. Use it for the **Python-vs-native split** (the "total Rust %") and
    Python-side hotspots.
  - **perf** (`/carter/users/dlaub/.pixi/bin/perf`, confirmed present) — use for
    **fine-grained Rust symbol-level** profiling once py-spy says Rust dominates.
- **Rust symbolization for perf/py-spy --native:** rebuild the genoray + gvl extensions
  with frame pointers + line-table debuginfo. In the relevant `Cargo.toml` profile (or via
  env): `RUSTFLAGS="-C force-frame-pointers=yes"` and `debug = "line-tables-only"` (or
  `debug = true`); then `pixi run -e default maturin develop` for gvl, and rebuild the
  genoray wheel (its build is `pixi run -e py310 maturin build --release` in the genoray
  repo — note genoray is a **frozen wheel** in gvl's env, so a debug-enabled rebuild + wheel
  swap is needed to symbolize genoray's `query.rs`/`svar2-codec` frames). Prefer
  `perf record -g --call-graph dwarf` if frame pointers are unavailable.
- **Noise control:** run on a dedicated compute node (`sbatch -p carter-compute
  --exclusive` or an interactive alloc), warm caches, median of N≥5, pin threads, record
  CPU governor / turbo state. Keep the MVP workload (same 3 regions, all-samples) for
  continuity.
- **Stores (already built):** `$W = /carter/users/dlaub/repos/for_loukik/svar2_mvp`:
  `{germline,somatic}.{svar,svar2,gvl}` + `{chr21,gdc.chr21}.norm.filt.bcf`. Reuse the MVP
  `benchmark.py` harness (`GenVarLoader worktree: tmp/svar2_mvp/benchmark.py`).

## Experiments

### E1 — Query-latency attribution (the primary experiment)

For each **(backend × cohort)** — SVAR2 adapter `reconstruct` and SVAR1 `ds_hap[:R,:S]`,
germline + somatic — on the MVP workload (3 regions, all samples):

1. **py-spy split.** `py-spy record --native --rate 500 -o <flame>.svg -- python <driver>`
   where `<driver>` loops the single path K times (warm first). Also capture
   `py-spy record --format speedscope`. Extract: **% wall-clock in Python vs % in native
   (Rust)**, and the top Python frames (numpy conversions, `overlap_batch` marshalling, FFI
   boundary). This directly answers *"is SVAR2 adapter latency Python overhead?"*.
2. **perf Rust detail** (only where py-spy shows native dominates): `perf record -g
   --call-graph dwarf -- python <driver>`; `perf report`. Read the top **Rust** symbols —
   e.g. `svar2_codec` decode, the dense presence `get_bit` gather, `overlap_range` search,
   memcpy/alloc. (Python frames will be opaque — that's expected; py-spy covered them.)

**Deliverable E1:** a table per (backend × cohort) of `{Python %, native %, wall-clock}` +
the top ~10 Rust symbols for the SVAR2 path. **Decision output:** classify the SVAR2 gap as
(A) Python-adapter overhead → Task B Dataset wiring likely erases it; (B) Rust hot-path /
layout → Task B must include a kernel/layout fix; or (C) mix (quantify).

### E2 — Same-cohort sample sweep (de-confound "scales with S")

Hold the **dataset fixed** and vary sample count S. Pick the **somatic** cohort (widest
range, up to 16007). Subsample the filtered BCF at **S ∈ {1000, 2000, 4000, 8000, 16007}**
(`bcftools view -S <list> --force-samples`), and for each S build **both** `.svar` and
`.svar2` (reuse `build_stores.py`; SLURM). Benchmark hap latency (warm, median N=5, same 3
regions, all S samples) for both backends at each S.

> Note the mild caveat: subsampling drops sites monomorphic in the subset, so variant count
> shrinks slightly with S; the AF *spectrum* is approximately preserved. Record per-S
> variant counts so the curves are interpretable.

**Deliverable E2:** hap-latency-vs-S curves for both backends on one cohort, with slopes.
**Decision output:** confirm/refute that SVAR2 latency is less S-sensitive than SVAR1 —
the claim the MVP notes explicitly declined to make.

### E3 — Dense-access layout probe (conditional on E1 showing dense gather hot)

If E1 attributes significant Rust time to the dense presence gather: measure SVAR2 hap
latency vs **(a) region width** and **(b) `n_dense_variants`** (contig-wide dense count).
Optionally a Rust microbench of the `get_bit`-per-column gather vs a word-parallel variant.
**Deliverable E3:** evidence for/against the "contig-wide stride + bit-by-bit read" cost,
and a rough size of the win from a region-local / word-parallel presence layout — input to
how much layout work Task B should carry.

### E4 — Conversion thread-allocation profiling (build side)

1. **Confirm the bottleneck.** Profile a single-contig `run_conversion_pipeline` on a
   compute node: py-spy `--native` (or perf on the Rust threads) to quantify time in
   **htslib read/decompress** vs **encode** vs **Phase-2 merge**. (genoray already exposes a
   sampler via `GENORAY_SAMPLE_INTERVAL` — capture its channel-fill / per-thread CPU%
   alongside.)
2. **Sweep the split.** For a fixed single-contig input, vary the thread policy — htslib
   decompression threads vs executor/writer threads (and total cores) — and measure build
   wall-clock. Find the split minimizing single-contig time.

**Deliverable E4:** build wall-clock vs thread-split table + a concrete recommended policy
for few-contig jobs, feeding
`genoray:docs/roadmap/architecture.md` → Open questions → *read-bound conversion / thread
allocation*.

## Deliverables & out of scope

- **Deliverable:** a notes file
  `docs/superpowers/notes/2026-07-XX-svar2-profiling-results.md` with the E1–E4 tables and
  two concrete recommendations: (1) where SVAR2 query latency actually goes (→ what Task B
  must include), and (2) the conversion thread-split policy. Profiling driver scripts under
  `tmp/svar2_mvp/`.
- **Out of scope (deliberately):** implementing Task B (Dataset wiring), any dense-layout
  change, and the conversion thread rebalance. This spec produces the numbers that decide
  *whether and how* to do those.

## Suggested execution

Small enough for a single focused session; a step-by-step plan can be derived from E1–E4 and
run via superpowers:subagent-driven-development. Start with **E1** (highest decision value:
it tells us if the whole latency story is just un-wired Python) and **E4** (independent,
build-side) — they need no new stores. E2 needs the SLURM sub-cohort builds; E3 is
conditional on E1.
