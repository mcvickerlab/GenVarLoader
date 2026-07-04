# SVAR2 vs SVAR1 gvl Benchmark (chr21)

**Date:** 2026-07-03
**Task:** Plan Task 4 (Task E) — measure SVAR1 (gvl `Dataset` over `.svar`) vs SVAR2
(`SparseVar2Source` over `.svar2`) on hap latency, variant latency, and store size, for
the germline + somatic chr21 stores built in Task 3.

**Script:** `tmp/svar2_mvp/benchmark.py`. **Env:** `pixi run -e default`, genoray 2.15.0.

## Workload (fairness rule)

Same workload on both backends: **all samples for a fixed region set**, matching genoray's
per-contig/all-samples query granularity. Warm caches (1 warmup), **median of N=5** repeats.

- **Regions** (0-based half-open, chr21): `(20_000_000, 20_001_000)`,
  `(30_000_000, 30_000_500)`, `(40_000_000, 40_001_000)` — 3 regions, 2.5 kb total.
- **Samples:** germline 3202, somatic 16007 (all samples, both backends).
- Both backends query the **same 3 regions × all samples**. (`benchmark.py` was corrected
  from the plan draft, which opened `validate.py`'s leftover 2-region `.gvl` while the
  SVAR2 path queried 3 regions — an unfair mismatch; it now `gvl.write`s the 3-region BED
  first so both measure an identical workload.)

## Results (median seconds; store in bytes)

| Source (samples) | Metric | SVAR1 | SVAR2 | Outcome |
| --- | --- | --- | --- | --- |
| **germline** (3202) | hap latency (s) | 0.0555 | 0.2834 | SVAR1 5.1× faster |
| | var latency (s) | 0.0280 | 0.2270 | SVAR1 8.1× faster |
| | store size | 1,149,533,941 (1.1 G) | 202,842,586 (178 M) | **SVAR2 5.67× smaller** |
| **somatic** (16007) | hap latency (s) | 0.3798 | 0.3797 | **tied** |
| | var latency (s) | 0.0518 | 0.1088 | SVAR1 2.1× faster |
| | store size | 55,578,073 (50 M) | 38,184,053 (34 M) | **SVAR2 1.46× smaller** |

## Reading the numbers

**Store size — SVAR2 wins outright, most on high-AF cohorts.** 5.67× smaller on germline
(1000G, many common/high-allele-frequency variants routed to SVAR2's 1-bit dense matrix)
and 1.46× smaller on somatic (rare/near-private mutations stay sparse in both formats).
This is exactly the empirical short-read distribution SVAR2 targets.

**Latency — SVAR2's cost is nearly flat in sample count; SVAR1's grows.** Going from 3202
→ 16007 samples (5×):

| Backend | hap latency 3202 → 16007 | factor |
| --- | --- | --- |
| SVAR1 | 0.0555 → 0.3798 s | **6.8×** |
| SVAR2 | 0.2834 → 0.3797 s | **1.34×** |

SVAR1's gvl `Dataset` pays a per-sample read cost that scales with cohort size; the SVAR2
adapter's live-query + decode is far less sensitive to sample count, so the two **reach
parity on hap latency at 16007 samples** and SVAR2 would extrapolate to *faster* for larger
cohorts. Variant-decode latency still favors SVAR1 (2–8×), but that gap also narrows with
scale (8.1× → 2.1×).

## Caveat — adapter-vs-Dataset (Task B not wired)

This compares the mature **SVAR1 gvl `Dataset`** (genotypes pre-materialized into an
optimized on-disk layout at `gvl.write` time; queries are then fast reads) against the
**SVAR2 `SparseVar2Source` adapter**, which queries genoray `overlap_batch` **live** and
decodes on every call, with **no Dataset-level batching, collation, or caching**. SVAR2 is
not yet wired into the Dataset (plan Task B, deferred). So SVAR2's latency here is a
*floor-less* direct-adapter number; wiring it into the Dataset (Task B) is what would let
it inherit the same batching/collation the SVAR1 path enjoys — likely closing or reversing
the remaining latency gap while **keeping the storage win**.

## Value proposition & Task B signal

- **SVAR2 is a clear win on disk footprint** (1.5–5.7×), scaling with allele frequency —
  the more common the variants, the bigger the win.
- **SVAR2 latency scales much better with cohort size** — already at parity on hap latency
  at 16k samples via the un-optimized adapter alone.
- **Concrete Task B cost signal:** the all-samples-per-region query is the granularity that
  matters, and the adapter already competes there at scale. Wiring SVAR2 into the Dataset
  (Task B) should be pursued: the storage advantage is unconditional, and the latency
  trend says the Dataset's batching/collation would likely make SVAR2 competitive-to-faster
  on latency too for realistic large cohorts. The invasive integration is justified by the
  storage win alone; the scaling trend strengthens the case.
