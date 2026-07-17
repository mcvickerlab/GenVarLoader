# Phase 5 W4 — Final single-thread numba-vs-rust `__getitem__` A/B

**Date:** 2026-06-26 · **Branch measured:** `phase-5-w4` (≡ `rust-migration` + W3 fusion `phase-5-w3`; W2 is test-only and perf-neutral) · **Node:** shared Carter HPC, single-thread (`NUMBA_NUM_THREADS=1`; rust serial — rayon is W5).

**Purpose:** the migration's final single-thread parity gate before the W5 consolidation (numba deletion + rayon). **Gate:** rust at parity-or-better single-thread across all `__getitem__` modes → proceed to consolidation. Benchmark-only; no code change.

## Methodology (and why)

The shared Carter node makes **absolute, cross-session wall-clock unreliable** — the same metric has drifted ≥2× between sessions minutes apart under variable load (round-3, PR #252). So this A/B follows the established rule: **measure rust AND numba in the SAME back-to-back session**, run twice to show within-session stability, and **pin the ratio direction explicitly** (here: `speedup = numba_ms / rust_ms`, higher ⇒ rust faster). The durable, trustworthy signal is **byte-identical numba/rust parity** (already gated across W1–W3 and the full parity suite) plus same-session improve-or-hold — not the absolute ms. The ms ratios below are reported as order-of-magnitude evidence, not precise constants.

Two independent tools, both single-thread, both backends, one session:
- `tests/benchmarks/test_e2e.py` — pytest-benchmark **pedantic min** (noise-robust per-call floor), seqlen 16384, batch 32, 50 rounds × 10 iterations, 5 warmup rounds.
- `tests/benchmarks/profiling/profile.py` — steady-state **mean wall-clock throughput**, 1500 batches after burn-in, two passes.

## Results

### `test_e2e.py` pedantic-min (ms/batch; lower = faster)

| Mode | rust min | numba min | speedup (numba÷rust) |
|------|---------:|----------:|---------:|
| haplotypes | 2.02 | 3.36 | **1.66×** |
| annotated | 6.48 | 9.30 | **1.43×** |
| tracks (haps+realigned tracks) | 2.01 | 3.34 | **1.66×** |
| tracks_only (pure track path) | 1.04 | 1.11 | **1.07×** |
| variants | — | — | xfail (pre-existing: `_FlatVariants.to_fixed` missing for `with_len`) |

### `profile.py` steady-state throughput (ms/batch; pass 1 / pass 2)

| Mode | rust | numba | speedup (pass1 / pass2) |
|------|-----:|------:|---------:|
| haplotypes | 2.27 / 2.02 | 3.63 / 3.34 | 1.60× / 1.65× |
| annotated | 6.92 / 6.41 | 9.05 / 8.93 | 1.31× / 1.39× |
| tracks (pure) | 1.08 / 1.08 | 1.13 / 1.12 | 1.05× / 1.04× |
| tracks-seqs | 2.03 / 2.03 | 3.34 / 3.34 | 1.65× / 1.65× |
| variants | 1.97 / 1.97 | 2.71 / 2.73 | 1.38× / 1.39× |
| variant-windows | 0.78 / 0.78 | 3.57 / 3.57 | 4.58× / 4.58× |

Both passes are tightly consistent (within-session stable), and the two tools agree.

## Conclusion — GATE PASSED

Rust is **parity-or-better single-thread on every mode**:
- The pure **tracks-only** path is the tightest at ~1.04–1.07× — effectively parity, rust marginally ahead. This path is dominated by per-batch fixed cost (region indexing + interval memmap IO), not kernel compute, so the backend choice barely moves it; rust is never behind.
- Every **compute-bound** path is clearly faster: haplotypes/tracks-seqs ~1.65×, annotated ~1.4×, variants ~1.4×, and **variant-windows ~4.6×** (fully rust-tokenized).

Combined with byte-identical parity (W1–W3 + the full parity suite, both backends), there is no single-thread regression risk in removing numba. **→ Proceed to W5 (consolidation: golden-snapshot the numba-oracle parity suites, delete numba, add rayon batch parallelism gated byte-identical to the serial golden result).**

Raw run logs: captured in-session (`profile.py` 6 modes × 2 backends × 2 passes; `test_e2e.py` 2 backends).
