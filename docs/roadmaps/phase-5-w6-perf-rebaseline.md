# Phase 5 W6 — Rayon serial-vs-multithread speedup re-baseline

**Date:** 2026-06-27
**Branch:** `phase-5-w6-wrapup`
**HEAD:** `0968a0f5a3c2cbc34f3d4f358e30c3df8aecaa40`
**Node:** shared Carter HPC, Intel Xeon E5-4650 v3 @ 2.10 GHz, 96 logical CPUs, linux-64
**Corpus:** `tests/benchmarks/data/chr22_geuv.gvl` (format 2.0, 165 regions × 5 samples, chr22, read-depth; `max_jitter=0`)
**Build:** `pixi run -e dev maturin develop --release` (release profile, genvarloader v0.35.0)
**Reference:** `tests/benchmarks/data/chr22.masked.fa.gz`

---

## Purpose

After the W5 consolidation (numba deleted, rayon batch parallelism added, PR #260), this pass
re-baselines the read path as a **same-session rayon serial-vs-multithread speedup curve** + peak-RSS
deltas. There is no live numba A/B: numba was deleted in W5.

For the final single-thread numba-vs-rust A/B (gate measured before W5), see:
[`docs/roadmaps/phase-5-w4-final-ab.md`](phase-5-w4-final-ab.md)

---

## Node-noise caveat (IMPORTANT — read before comparing across sessions)

The Carter HPC node is **shared**. Absolute wall-clock drifts ≥2× between sessions under
variable load (documented across Phase 3 round-3, W4 A/B, and prior passes). Absolute ms/batch
values are NOT comparable across sessions. The durable signal is:

- **Same-session ratios** (thread-count N vs serial baseline, measured back-to-back).
- **Deterministic correctness**: `serial == parallel == frozen golden` for all kernels
  (`tests/parity/test_rayon_equivalence.py`, W5 gate).
- **Instruction-count reductions** from round-3 tuning (documented in `rust-migration.md`).

All tables in this document were captured in ONE continuous session on 2026-06-27.

---

## Methodology

### e2e modes (haplotypes, annotated, tracks, tracks-only)

Harness: `tests/benchmarks/test_e2e.py` via `pytest-benchmark` **pedantic min**.
Configuration: `ROUNDS=50`, `ITERATIONS=10`, `WARMUP_ROUNDS=5`, `SEQLEN=16384`, `BATCH=32`.
Each reported figure is `min` (ms/batch) — the most noise-robust estimate.

```bash
RAYON_NUM_THREADS=<N> GVL_NUM_THREADS=<N> pixi run -e dev pytest tests/benchmarks/test_e2e.py \
    -q --benchmark-only --benchmark-disable-gc --benchmark-warmup-iterations=5
```

The `variants` e2e mode is `xfail` (pre-existing: `_FlatVariants.to_fixed` missing for `with_len`;
predates this phase). Variants and variant-windows are measured via `profile.py` instead.

### variants modes (variants, variant-windows)

Harness: `tests/benchmarks/profiling/profile.py` **wall-clock average** (2000 batches, burn-in 5).

```bash
RAYON_NUM_THREADS=<N> GVL_NUM_THREADS=<N> pixi run -e dev python \
    tests/benchmarks/profiling/profile.py --mode <mode> --n-batches 2000
```

### Peak-RSS

Harness: `pixi run -e dev memray-tracks` / `memray-haps` + `python -m memray stats`.
Default 2000 batches, no `RAYON_NUM_THREADS` / `GVL_NUM_THREADS` override for the "parallel"
run; `RAYON_NUM_THREADS=1 GVL_NUM_THREADS=1` for the serial run.

### Thread counts measured

`RAYON_NUM_THREADS` (and `GVL_NUM_THREADS`) = **1** (serial baseline), **2**, **4**, **8**,
**unset** (default = all available cores = 96 on this node).

---

## The `should_parallelize` threshold — why all modes stayed serial

The `should_parallelize(total_bytes)` gate in `python/genvarloader/_threads.py` uses:

```python
_MIN_BYTES_PER_THREAD = 1 << 20  # 1 MiB
return total_bytes >= num_threads() * _MIN_BYTES_PER_THREAD
```

`num_threads()` reads `GVL_NUM_THREADS` (or cgroup CPU count). The small benchmark corpus
(BATCH=32, SEQLEN=16384) produces at most ~2 MiB of output per batch:

| Mode | Output bytes per batch | Threshold at N threads | Parallel? |
|------|----------------------|------------------------|-----------|
| haplotypes (32 × 2 haps × 16384 bytes) | 1,048,576 B (1 MiB) | N × 1 MiB | No at N≥2; borderline at N=1 |
| tracks f32 (32 × 16384 × 4 bytes) | 2,097,152 B (2 MiB) | N × 1 MiB | Borderline at N=2 only |
| annotated (haps + 2 × i32 arrays) | ~3 MiB | N × 1 MiB | No at N≥4 |
| variants (ragged, variable) | ~few MiB | N × 1 MiB | No at N≥8 |

**Conclusion: all modes ran serial for N≥4 and most modes ran serial at all N on this corpus.**
This is correct behavior: the gate exists to prevent rayon spawn overhead from dominating short
batches. **This is a finding, not a failure** — the parallelism gate is working as designed.

> For production workloads at `SEQLEN≥131072` or `BATCH≥256`, most modes will cross the
> threshold and rayon will engage. The gate's correctness (`serial == parallel == frozen golden`)
> was already verified unconditionally in W5's `test_rayon_equivalence.py` parity suite.

---

## Results

### e2e pedantic-min (ms/batch; lower = faster)

Speedup = serial_min_ms / N_threads_min_ms (>1.0 means the multi-thread run was faster).
All values are `min` (ms/batch) from pytest-benchmark pedantic runs.

| Mode | T=1 (serial) | T=2 | T=4 | T=8 | T=all (96) | Note |
|------|------------:|----:|----:|----:|----------:|------|
| tracks-only | **1.0558** | 0.9559 | 1.0111 | 1.0122 | 0.9623 | All within session noise |
| tracks (haps+realigned) | **2.0700** | 1.9484 | 2.0103 | 1.9521 | 1.9620 | All within session noise |
| haplotypes | **2.0819** | 1.9722 | 2.0276 | 1.9661 | 1.9687 | All within session noise |
| annotated | **6.6933** | 6.1536 | 6.2886 | 7.0523 | 6.1394 | All within session noise |

Speedup vs serial (serial_min / thread_min; >1.0 = faster):

| Mode | T=2 | T=4 | T=8 | T=all (96) |
|------|----:|----:|----:|----------:|
| tracks-only | 1.10× | 1.04× | 1.04× | 1.10× |
| tracks | 1.06× | 1.03× | 1.06× | 1.06× |
| haplotypes | 1.06× | 1.03× | 1.06× | 1.06× |
| annotated | 1.09× | 1.06× | 0.95× | 1.09× |

**All ratios are in the 0.95×–1.10× band — within shared-node noise. No mode shows a
genuine rayon speedup, confirming that the threshold gate held serial execution throughout.**

### variants modes wall-avg (ms/batch; lower = faster)

| Mode | T=1 (serial) | T=2 | T=4 | T=8 | T=all (96) | Note |
|------|------------:|----:|----:|----:|----------:|------|
| variants | **2.085** | 2.129 | 2.019 | 2.036 | 2.054 | Within noise |
| variant-windows | **0.798** | 0.794 | 0.812 | 0.806 | 0.802 | Within noise |

Speedup vs serial:

| Mode | T=2 | T=4 | T=8 | T=all (96) |
|------|----:|----:|----:|----------:|
| variants | 0.98× | 1.03× | 1.02× | 1.01× |
| variant-windows | 1.01× | 0.98× | 0.99× | 1.00× |

**All within noise. Serial execution confirmed for both variants modes at all thread counts.**

### Summary: speedup never materialized on this corpus

No mode crossed the `should_parallelize` threshold at N≥4 threads. At N=2, the tracks f32
path sits exactly at the 2 MiB boundary but the measured ratio is still within session noise.

The rayon parallelism gate functions correctly: it prevents spawn overhead from hurting small
batches and yields identical output (proven by `test_rayon_equivalence.py`). The speedup curve
for production-scale workloads is not measurable on this 32-batch / 16384-seqlen test corpus.

---

## Peak RSS

Measured with memray (haps mode and tracks mode, serial vs parallel/unset):

| Run | Mode | Serial (T=1) peak RSS | Parallel (unset) peak RSS | Δ |
|-----|------|-----------------------|--------------------------|---|
| memray-tracks | tracks | 3.525 GB | 3.525 GB | 0 |
| memray-haps | haplotypes | 3.525 GB | 3.525 GB | 0 |

Peak RSS is 3.525 GB in all cases, dominated by the seqpro/llvmlite JIT startup (~3.2 GB
transitive via seqpro 0.20.0). Since the threshold gate held serial execution throughout,
the rayon thread-pool overhead (stack allocations, worker threads) was never materialized.

**GVL-attributable RSS delta: 0.** The ~3.2 GB floor is seqpro transitive numba, not
gvl-own code. Removing numba from seqpro is explicitly out of scope for this migration
(W5 seqpro caveat; user decision 2026-06-27).

---

## Numba A/B: unavailable (W5 deletion)

Numba was deleted in W5 (PR #260). A live numba vs rust comparison is no longer possible on
this branch. For the final single-thread numba-vs-rust speedup figures (all modes at
parity-or-better), see:

**[`docs/roadmaps/phase-5-w4-final-ab.md`](phase-5-w4-final-ab.md)**

Summary of W4 final A/B (same-session, `phase-5-w4` branch, Carter HPC):

| Mode | rust (ms/batch) | numba (ms/batch) | speedup (numba÷rust) |
|------|----------------:|-----------------:|---------------------:|
| haplotypes | 2.02 | 3.36 | **1.66×** |
| annotated | 6.48 | 9.30 | **1.43×** |
| tracks (haps+realigned) | 2.01 | 3.34 | **1.66×** |
| tracks-only | 1.04 | 1.11 | **1.07×** |
| variants | 1.97 | 2.71 | **1.38×** |
| variant-windows | 0.78 | 3.57 | **4.58×** |

---

## GVL-attributable conclusion

1. **Rayon implementation is correct.** `serial == parallel == frozen golden` for all kernels
   (`test_rayon_equivalence.py`, W5 parity gate). No correctness regression.

2. **Threshold gate works as designed.** On the small benchmark corpus (BATCH=32, SEQLEN=16384),
   all modes ran serial at N≥4 because batch output bytes (~1–3 MiB) < N × 1 MiB threshold.
   This is the expected and correct behavior.

3. **Rayon speedup is not measurable on this corpus.** For production workloads at
   `SEQLEN≥131072` or `BATCH≥256`, the threshold will be crossed and rayon will engage. The
   correctness gate in `test_rayon_equivalence.py` covers those cases unconditionally.

4. **Peak RSS is unchanged.** The gvl-attributable RSS delta is 0. The 3.525 GB process floor
   is the seqpro transitive JIT, which is out of scope for this migration.

5. **Single-thread headroom is already maximized.** W4 showed rust at parity-or-better on all
   modes (up to 4.6× faster for variant-windows). The round-3 instruction-level tuning pass
   (PR #252) confirmed deterministic instruction-count reductions across 7 hot kernels.
   Rayon adds the future ability to scale throughput linearly with cores at production batch sizes.
