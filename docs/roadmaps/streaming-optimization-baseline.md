# Streaming VCF/PGEN optimization: baseline counters + profile (Task 3, #276)

Same-session "before" numbers for the streaming VCF/PGEN optimization pass
(`docs/superpowers/plans/2026-07-18-streaming-vcf-pgen-optimization.md`,
`docs/superpowers/specs/2026-07-18-streaming-vcf-pgen-optimization-design.md`). Captured on
HEAD after Tasks 1+2 landed (deterministic counters + `--compare-dataset` harness arm), before
any optimization edit. Every later task's gate is measured against these numbers, same
session, same node (`gvl-rust-perf-gate-shared-node-noise`: wall-clock is secondary color,
never pass/fail).

Environment: `pixi run -e dev maturin develop --release` immediately before capture (fresh
build, not stale `.so`). Counters read via
`genvarloader.genvarloader.{transpose_word_reads,transpose_word_reads_reset,
pgen_variants_decoded,pgen_variants_decoded_reset}` (module is `genvarloader.genvarloader`,
not `_core`).

## Fixture generation

`benchmarking/streaming/gen_fixtures.sh` (via the `gen-bench-vcf`/`gen-bench-pgen` pixi
tasks), `vcfixture bulk --profile germline-1kgp --seed 42 --contigs chr1 --target-size 50MB`,
then `plink2 --make-pgen`.

- **N=1000**: pre-existing fixtures, reused as-is.
- **N=10000**: generated fresh. `vcfixture bulk` took **~20 minutes** of wall time for
  N=10000 (vs. well under a minute for N=1000) — cohort simulation cost in the
  `germline-1kgp` profile scales materially with sample count independent of the
  `--target-size` cap (more per-sample work is paid to hit the same target file size, not
  less). One transient hiccup: an in-progress (not-yet-closed) BCF was briefly observed with
  `bcftools`/`plink2` reporting "No BGZF EOF marker" / "malformed record" because a stale
  file-existence check raced the still-running `vcfixture` process; waiting for the process
  to actually exit (not just for the file to appear) resolved it — the finished N=10000 BCF
  and PGEN both validate cleanly (`bcftools index`, `plink2 --make-pgen`) with no
  malformed-record errors.
- **N=50000**: **DEFERRED, best-effort, not generated.** Extrapolating from N=10000's ~20
  minute generation time (itself ~20-40x N=1000's), N=50000 would plausibly take well over an
  hour and risked blocking this measurement task without changing the Task-6 decision (which
  the 1k/10k profile already answers decisively — see below). Follow-up: generate N=50000
  out-of-band if a later task needs a third sweep point; the harness
  (`benchmarking/streaming/bench_streaming.py --n 50000 ...`) needs no changes, only the
  fixture.

## Counters + wall-clock (best/median of 3 reps, same session)

All rows: `--n-regions 200 --region-len 200 --batch-size 32`, `n_windows=4` for every
(backend, N) — the sweep's BED/region-plan is fixed across N. `bytes_emitted` is identical
across VCF, PGEN, and the written-Dataset driver at both N (the harness's cross-driver
`bytes_emitted` assertion passed — byte-identical parity confirmed for this baseline).

| backend | N | n_windows | pgen_variants_decoded | transpose_word_reads | engine best/median (s) | sync best/median (s) | sync/engine ratio | peak_rss_kb (engine) |
|---|---|---|---|---|---|---|---|---|
| vcf | 1000 | 4 | n/a | 1,850,000 | 0.716 / 0.729 | 1.267 / 1.322 | 1.77 | ~858k–883k |
| pgen | 1000 | 4 | 282,248 | 1,850,000 | 1.313 / 1.336 | 2.146 / 2.147 | 1.63 | ~1.05M |
| vcf | 10000 | 4 | n/a | 17,140,000 | 5.195 / 5.254 | 7.101 / 7.104 | 1.37 | ~1.07M–1.35M |
| pgen | 10000 | 4 | 400,032 | 17,140,000 | 11.366 / 11.466 | 14.164 / 14.755 | 1.25 | ~1.91M–2.0M |
| vcf | 50000 | — | — | — | DEFERRED (fixture not generated) | — | — | — |
| pgen | 50000 | — | — | — | DEFERRED (fixture not generated) | — | — | — |

`pgen_variants_decoded` == `n_windows * pvar_variants` exactly at both N (1000: 4×70,562 =
282,248; 10000: 4×100,008 = 400,032) — confirms the documented coarse-`var_start` behavior
(`src/record_stream/pgen.rs`'s "Coarse `var_start`" doc section): every window re-decodes the
**entire contig prefix from record 0**, not just the window's variants. This is Task 4's
target (narrow `var_start`/`var_end`), independent of the Task-6 question below.
`transpose_word_reads` is identical between VCF and PGEN at each N (same `(V,S,P)` grid shape
fed through the same `fill_decoded_window`), as expected.

`sync/engine` ratio (>1 = engine's cross-window producer/consumer overlap wins): 1.25-1.77x
across both backends and N, i.e. the shipped overlap is real and holds at this scale;
noted as color per the project's noisy-shared-node convention, not a gate.

## `--compare-dataset`: streaming vs. written Dataset (same run)

`bench_streaming.py --backend both --strategy both --repeats 3 --compare-dataset`, symbolic-SV-filtered
common-denominator variant set (`bench_<n>.filtered.{bcf,pgen}`), identical (region, sample)
plan cells, `bytes_emitted` identical across engine/sync/dataset drivers (assertion passed, no
divergence) at both N.

| N | dataset sweep (s) | write time (preprocessing, s) | on-disk dataset bytes |
|---|---|---|---|
| 1000 (vcf variant set) | 0.138 | 2.060 | 4,076,005 |
| 1000 (pgen variant set) | 0.103 | 1.347 | 8,678,203 |
| 10000 (vcf variant set) | 1.240 | 8.811 | 35,715,271 |
| 10000 (pgen variant set) | 1.426 | 5.408 | 42,245,587 |

The written-Dataset sweep itself is markedly faster than either streaming driver once
preprocessing is paid (as expected — it's reading a purpose-built on-disk layout), but the
`write=` preprocessing cost (1.3-8.8s at this small sweep scale) is the whole point of
`StreamingDataset`: zero preprocessing, at some per-epoch throughput cost. Not itself part of
the Task-6 gate; recorded here because Task 2 wired the comparison and this is its first
same-session baseline reading.

## 10k profile (`perf record -g --call-graph=dwarf`, `cycles:u`, engine strategy, 1 rep)

Producer-thread work lives entirely under `<VcfWindowFiller/PgenWindowFiller as WindowFiller>::fill`
on the `gvl-stream-prod` thread (perf resolves thread names — filtering `perf report -c
gvl-stream-prod` isolates it from the python main thread's consumer-side work and from the
rayon worker pool driving `reconstruct_haplotypes_from_sparse`, which run concurrently and
would otherwise swamp a whole-process percentage). Percentages below are `fill`'s own Children%
(i.e. normalized to 100% = one full `fill()` call), read directly off the call-graph subtree
under that node — not a flat, unfiltered process-wide percentage.

### VCF (`fill` = 8.92% of total process cycles; producer thread `gvl-stream-prod` = 9.83% of process total)

| stage | share of `fill()` | detail |
|---|---|---|
| **READER-OPEN** (`VcfRecordSource::new`) | **74.6%** (6.65 / 8.92) | Nearly all of this (6.49 of 6.65, i.e. ~97.6%) is `rust_htslib::bcf::header::HeaderView::sample_id` — an O(n_samples) **string** lookup (`__strlen_avx2` + `__memcmp_avx2_movbe`) mapping each of the 10,000 sample names to a header column index, redone from scratch on **every window** because `VcfRecordSource::new` is called fresh per window (no reader persistence yet). |
| DECODE (`next_record` → htslib `bgzf_read`/`libdeflate_deflate_decompress`) | 20.7% (1.85 / 8.92) | Genuine decode work — BGZF block decompression + record parsing. Expected O(window variants) cost, not itself a target. |
| TRANSPOSE (`fill_decoded_window`) | 4.7% (0.42 / 8.92) | Small relative to reader-open at this N. |

### PGEN (`fill` = 22.89% of total process cycles; producer thread `gvl-stream-prod` = 23.06% of process total)

| stage | share of `fill()` | detail |
|---|---|---|
| DECODE (`next_record` → pgenlib `read_alleles_range`/`plink2::GenoarrPhasedToAlleleCodes` etc.) | 98.5% (22.55 / 22.89) | This is the coarse-`var_start` over-decode volume (Task 4's target) landing as **decode-call cost**, not a separate reopen cost — `PgenRecordSource`/pgenlib handle is not the bottleneck; the *amount* of genotype data pulled per window is. |
| TRANSPOSE (`fill_decoded_window`) | 1.35% (0.31 / 22.89) | Small. |
| **READER-OPEN** (`PgenRecordSource::new`, `PgenWindowFiller::new`, `.pvar` text re-skip via `PvarReader`) | **<1%** (≤0.02% of total process cycles combined) | Negligible in this profile — the `.pvar` text scan cost the plan flagged as a Task-6 candidate for PGEN does **not** show up as material at N=10000; PGEN's per-window overhead is essentially all inside the pgenlib decode call itself. |

`perf stat -e instructions,LLC-load-misses` (same `--n 10000 --strategy engine --repeats 1`
command, whole process):

| backend | instructions | LLC-load-misses |
|---|---|---|
| vcf | 66,329,169,672 | 32,194,104 |
| pgen | 82,889,226,047 | 79,778,042 |

PGEN's ~25% higher instruction count and ~2.5x higher LLC-load-miss count vs. VCF at the same
N and window plan are consistent with the coarse-prefix over-decode (more data touched per
window, less locality) — a Task 4 signal, not a Task 6 one.

## Task 6 decision: **PROCEED**

Rule from the plan/spec: Task 6 (reader reuse — persist the VCF `IndexedReader`/PGEN
handle across `fill` calls instead of reopening per window, plus a genoray `PvarReader` seek
if the `.pvar` re-skip specifically is material) proceeds only if reader-open work is a
**material share (>10% of producer time at 10k)**.

- **VCF: reader-open (`VcfRecordSource::new`) is 74.6% of the producer thread's `fill()`
  time at N=10000** — dramatically over the 10% threshold, and almost entirely a single
  avoidable cost (`HeaderView::sample_id`'s O(n_samples) string lookup, repeated per window
  instead of once per sweep). This alone justifies Task 6.
- PGEN's specific `.pvar`-text-reskip / `PgenRecordSource::new` share is **not** material in
  this profile (<1%) — PGEN's dominant cost is the coarse-`var_start` over-decode *volume*
  (Task 4's target), not a reopen/reparse cost. Task 6's PGEN-side "persist the pgenlib
  handle" lever is still worth doing while touching this code path (cheap, composes with Task
  4), but the genoray `PvarReader`-seek follow-up the spec flagged as conditional on this
  profile is **not independently justified by this measurement** — defer it unless a
  post-Task-4 re-profile shows otherwise.

**Decision: Task 6 proceeds**, driven by the VCF reader-open finding (74.6% ≫ 10%).

## Rebuild rule reminder

Every number above followed `pixi run -e dev maturin develop --release` immediately
beforehand in the same session — pytest/bench scripts import the stale compiled extension
otherwise (`CLAUDE.md`, `gvl-rust-perf-gate-shared-node-noise`).

## Raw artifacts (not committed — local job scratch, referenced for reproducibility)

- `perf record` data: `perf_vcf.data` (1.39 GB, 173,094 samples), `perf_pgen.data` (1.93 GB,
  239,861 samples) at `/carter/users/dlaub/.claude/jobs/2b4cf8e8/tmp/`.
- Standalone counter-measurement script (constructs `StreamingDataset` via the harness's own
  `_contig_lengths`/`_build_reference`/`_make_bed`/`_count_pvar_variants` helpers, resets
  counters, drains one `to_iter()` sweep, prints them) at the same path
  (`measure_counters.py`) — not part of the harness, not committed; harness itself
  (`benchmarking/streaming/bench_streaming.py`) received no instrumentation edits for this
  task.
