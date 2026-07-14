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

## Results (median seconds; store size = apparent bytes, `du -sb`)

| Source (samples) | Metric | SVAR1 | SVAR2 | Outcome |
| --- | --- | --- | --- | --- |
| **germline** (3202) | hap latency (s) | 0.0555 | 0.2834 | SVAR1 5.1× faster |
| | var latency (s)† | 0.0280 | 0.2270 | — (different ops, see below) |
| | store size | 1,149,533,941 (1.15 GB) | 202,842,586 (203 MB) | **SVAR2 5.67× smaller** |
| **somatic** (16007) | hap latency (s) | 0.3798 | 0.3797 | **≈ parity** |
| | var latency (s)† | 0.0518 | 0.1088 | — (different ops, see below) |
| | store size | 55,578,073 (55.6 MB) | 38,184,053 (38.2 MB) | **SVAR2 1.46× smaller** |

Store-size labels are decimal (MB = 10⁶ B, GB = 10⁹ B) computed from the apparent byte
counts; all ratios below are computed from the byte counts, not the rounded labels.

**† The two "var latency" columns measure different operations and are NOT a like-for-like
decode comparison.** SVAR2 var = `sv2.decode(chrom, regions)` (genoray raw variant-record
decode → `Ragged`); SVAR1 var = `ds.with_seqs("variants")[:regions, :n_s]` (gvl
variant-**sequence** materialization). They are reported side by side only because each is
the natural "variants" call for its backend — do not read them as one backend decoding
"the same thing" faster.

## Reading the numbers

**Store size — SVAR2 wins outright, most on high-AF cohorts.** 5.67× smaller on germline
(1000G, many common/high-allele-frequency variants routed to SVAR2's 1-bit dense matrix)
and 1.46× smaller on somatic (rare/near-private mutations stay sparse in both formats).
This is exactly the empirical short-read distribution SVAR2 targets.

**Latency — SVAR1 leads on the small high-density cohort; the two are at parity on the
large one.** Between the two cohorts:

| Backend | hap latency germline (3202) → somatic (16007) |
| --- | --- |
| SVAR1 | 0.0555 → 0.3798 s |
| SVAR2 | 0.2834 → 0.3797 s |

⚠️ **This is NOT a controlled sample-count sweep — the two points are different datasets**
(1000G germline, high-AF/high-density, vs GDC somatic, rare/sparse), differing in allele
frequency, variant density, and source, not just in sample count. So the movement cannot be
attributed to cohort size alone: germline's high variant density is a large part of why
SVAR2's germline hap latency (0.283 s) is high at only 3202 samples. What the data *does*
show is a real, useful fact — **on the large 16007-sample somatic cohort the un-optimized
SVAR2 adapter already matches the mature SVAR1 Dataset on hap latency** (0.3797 vs 0.3798 s;
median of N=5, no dispersion/CI measured, so treat "parity" as "indistinguishable at this
resolution"). The *hypothesis* that SVAR2's latency is less sample-count-sensitive is
plausible and worth testing, but proving it needs a **same-cohort subsampling sweep**
(fix the dataset, vary S) — not this germline-vs-somatic pair. The var-latency columns are
different operations (†) and are not compared here.

## Caveat — this is two different code paths, NOT profiled

The latency columns compare the mature **SVAR1 gvl `Dataset`** path against the **SVAR2
`SparseVar2Source` adapter** — and the SVAR2 path here is **not the Dataset at all**. It is
a new Python adapter that calls genoray `overlap_batch` live and runs the two-source Rust
kernel per call, with no Dataset-level batching/collation/caching (Task B, deferred). So:

- We make **no claim about *why* the latencies differ.** We have not run `perf` / a Rust
  profiler on either path, so attributing the gap to "pre-materialization", "live decode",
  memory layout, or Python overhead would be speculation. Any "X is slow" statement is out
  of scope until profiled.
- **Prior gvl profiling does not transfer.** Earlier work found gvl `Dataset` latency was
  dominated by the Rust core with negligible Python-orchestration overhead — but that was
  measured on the `Dataset` path. The benchmarked SVAR2 path is the **raw adapter**, which
  is *new Python orchestration* (per-call genoray query + numpy conversions) that has never
  been profiled; its Python overhead on a small 3-region workload is unknown and could be a
  material fraction of the 0.28–0.38 s.
- **The size table compares variant *stores* (`.svar` vs `.svar2`), not query artifacts.**
  SVAR1's low latency is served from an additional pre-materialized `.gvl` Dataset that
  `gvl.write` produces (here only the 3 benchmark regions: germline `.gvl` 292 KB, somatic
  `.gvl` 1.6 MB — tiny because it holds just those regions, not the contig). The
  `.svar`-vs-`.svar2` size comparison is the correct *variant-source* comparison; just note
  SVAR1 additionally materializes a `.gvl` to reach its quoted latency, and a fully-wired
  SVAR2 (Task B) would likewise gain a Dataset artifact.

### What we *can* say about layout (checked in source, not profiled)

The user hypothesized the SVAR2 dense matrix is variant-major `(V, S, P)` — pessimal for
gvl's access pattern (contiguous variant slices for random `(sample, ploid)`). **Checked and
refuted:** the dense genotype/presence matrix is **hap-major**, variant *innermost* — bit
index `hap * n_dense_variants + col` (`genoray:src/query.rs:131-134`, matching
`data-model.md`'s "hap-major `(sample, ploid, variant)`, variant fastest-varying"). During
build the in-memory chunk is variant-major (`BitGrid3(v,s,p)`) and `dense_merge`
bit-**transposes** it to hap-major on disk (roadmap M4). So a *single* hap's dense variants
are contiguous — the good case for per-hap reconstruction.

A **related, real** concern remains (also unprofiled): `n_dense_variants` is the **whole
contig's** dense count, so a narrow-region × all-samples query reads one short contiguous
run per hap at stride `n_dense_variants` → a separate cache line per haplotype, scattered
across a contig-wide matrix; and presence is read **bit-by-bit** (`get_bit` per column), not
word-parallel. For germline (many common → dense variants → large `n_dense_variants`) that
scatter is worse, which is *directionally consistent* with germline's higher SVAR2 hap
latency — but this is a hypothesis for the profiler, not a conclusion, and it is
**contig-scoped**, so it ties into the split-by-contig question below.

## Limitations / open questions (before strong claims or Task B)

1. **Profile before concluding.** Run `perf` + a Rust profiler on both hap paths; separately
   measure the SVAR2 adapter's Python overhead vs its Rust kernel. Only then attribute the
   latency gap to a cause.
2. **Same-cohort sample sweep.** The germline↔somatic pair is confounded (different AF /
   density / source). To claim SVAR2 scales better with S, subsample one cohort and vary S
   with the dataset fixed.
3. **Single contig — split-by-contig layout unassessed.** All measurements are chr21 only.
   The contig-wide dense stride above and genoray's per-contig on-disk partition (M3) can
   only be evaluated multi-contig; defer that analysis.
4. **Build: reading dominates; single-contig underuses threads.** The conversion logs show
   the pipeline reserving `1 concurrent chromosome | 3 HTSlib decompression threads (7 of 8
   active, 1 idle)` — for a single-contig job most cores sit idle and VCF read/decompress is
   plainly the bottleneck (germline ~11 min, somatic ~2 h). Worth exploring a thread split
   that dedicates ~1 thread to the executor+writer and the rest to VCF decompress+read when
   few contigs are present (genoray conversion-pipeline tuning, not gvl).

## Value proposition & Task B signal

- **SVAR2 is a clear, unconfounded win on disk footprint** (1.46× somatic, 5.67× germline;
  larger where variants are common/high-AF). This alone is a strong reason to pursue Task B.
- **On the large 16k-sample cohort, the un-optimized SVAR2 adapter already matches SVAR1 on
  hap latency** — so wiring SVAR2 into the Dataset (Task B) does not start from a latency
  hole at scale. It is *not* established that SVAR2 "scales better with cohort size" (see
  Limitation 2).
- **Recommendation:** pursue Task B for the storage win, but gate any latency claims on the
  profiling in Limitation 1 first — a real chance the current SVAR2 adapter latency is
  Python-adapter overhead the Dataset path would erase, and an equally real chance the
  contig-wide dense access needs a layout fix. Profile, then decide how much layout work
  Task B should include.
