# SVAR2 spliced-variant decode: benchmark + optimize (PR #286)

Date: 2026-07-17
Branch: `perf/svar2-spliced-variants-bench` (stacked on PR #286 head `deb76d6`)
Base for the eventual PR: same base as #286 (`main`).

## Problem

PR #286 adds spliced `with_seqs("variants")` output for SVAR2 and parallelizes
the decode/record-assembly with bounded Rayon chunks
(`src/svar2/mod.rs`), regrouping exon records into complete
`(transcript, sample, phase)` `RaggedVariants` cells in
`_fetch_spliced_variants` (`python/genvarloader/_dataset/_query.py`).

The PR ships **no throughput measurement**. The parallelization design
("~4 chunks per GVL worker", "indexed collection preserves ordering") is
asserted, not benchmarked. We do not know:

- whether the Rust parallel decode actually scales with cohort size, or
- whether the Python-side regroup dominates and starves it.

Prior findings on analogous SVAR2 read paths (recorded in agent memory:
`svar2-readpath-reorder-bottleneck`, `svar2-spliced-gather-bottleneck`) showed
the **Python regroup/gather dominated (~87–96%), not the Rust kernel**.
`_fetch_spliced_variants` has exactly that shape: `flat[plan.permutation]`
row-gather + per-field `to_packed()` + `Ragged.from_offsets`. It is the prime
suspect.

## Goal

1. Give the spliced-variant decode a reproducible throughput benchmark fed by
   realistic AoU-scale cohorts.
2. Run the `performant-py-rust` measure→optimize loop.
3. Land the winning change onto PR #286.

**Success:** either a meaningful reduction in median batch wall-time at the
largest cohort that fits on the bench node, or a measured demonstration that
the path already saturates (with the profile that proves it). Correctness must
be byte-identical to the current PR #286 output at every step.

## Phase 1 — workload characterization

CPU-bound. Dominating dimensions (confirmed with the maintainer):

| dimension            | typical            | max          | grows? | notes                                              |
|----------------------|--------------------|--------------|--------|----------------------------------------------------|
| `n_samples` (cohort) | 3,202 (1kGP)       | ~245k (AoU)  | grows  | per-sample decode; Rayon chunking bounded per-worker |
| `n_transcripts`/batch| 32–256             | study-dep    | grows  | fan-out the splice regroup runs over               |
| exon variants/query  | data-dependent     | —            | ~fixed | driven by cohort density × exon span               |
| ploidy               | 2                  | 2            | fixed  |                                                    |

Sweep `n_samples` and `n_transcripts`; hold density and ploidy fixed.

## Components (`benchmarking/svar2_spliced/`)

### 1. Cohort generator (`gen_cohort.py` / thin shell wrapper)
Shell out to the vcfixture-rs bulk CLI:

```
vcfixture bulk --profile germline-1kgp --samples S --contigs chr1 \
  --records N --seed 42 -o cohort_S.bcf
```

- `S` swept over a small set (e.g. 500 / 3_202 / 25_000 / and the largest that
  fits the node); `N` fixed so variant density per transcript is held constant
  across the sample sweep.
- The CLI binary is built once from the sibling checkout
  (`/carter/users/dlaub/projects/vcfixture-rs`, `--features cli`) or
  `cargo install`; the harness records the exact binary + version it used.
- vcfixture declares each `##contig` length as the **populated span**, not the
  real chromosome length — the fixture builder must read that span back and
  keep all synthetic exons inside it.

### 2. Fixture builder (`build_fixture.py`)
From one generated BCF:
- Read the populated contig span (from the BCF header / last record).
- Emit a synthetic reference FASTA covering that span.
- Emit a splice BED (`chrom,start,end,strand,transcript_id,exon_number`) whose
  exons tile inside the populated span, parameterized by `n_transcripts` and a
  fixed exon layout (a few exons/transcript, both strands represented).
- `gvl.write(...)` the BCF → SVAR2 `.gvl` store.

Fixtures are content-addressed by `(S, N, seed, layout)` and cached on disk so
the sweep does not re-write SVAR2 stores every run.

### 3. Bench driver (`bench_spliced.py`)
- `Dataset.open(...).with_seqs("variants")` + `splice_info=(transcript_id,
  exon_number)` + `var_filter="exonic"`.
- Build a fixed list of `T` transcript queries.
- Warmup, then best-of-k timing (median + min + spread) of the spliced
  `getitem` / `_fetch_spliced_variants` over `(S, T)` sweep.
- `GVL_FORCE_PARALLEL=1` and serial both measured, so we can see the parallel
  path's actual scaling vs. the serial baseline.
- Emit a small table (CSV + printed) so before/after is directly comparable.

### 4. Correctness oracle
- Freeze the current PR #286 output for each sweep point (pickle the
  `RaggedVariants` fields: `alt/start/ref/ilen/dosage/...` + offsets).
- Every candidate optimization must reproduce those arrays **exactly**
  (`np.array_equal` on data + offsets, per field), across the sweep and
  degenerate cases: 0 exonic variants in a transcript, single transcript,
  minus-strand transcript (RC path), single-group fast path.
- A candidate that fails the oracle is a bug, not a faster version.

### 5. Optimization loop (Phase 4)
- Profile: `pyinstrument` on the Python regroup; `samply`/`perf` on the Rust
  decode (perf on the Python process per memory `gvl-profiling-perf-not-pyspy`).
- One hypothesis → one change → re-run oracle **and** benchmark → keep only if
  it wins and stays correct.
- For any Rust change, confirm the mechanism (`cargo-show-asm`) rather than
  asserting it.
- Stop at the Phase-0 target, the Amdahl ceiling, or diminishing returns —
  and state which.

## Non-goals

- No LD/haplotype realism in the cohort (vcfixture draws i.i.d.; its own
  ablation shows LD is a ~0x lever on BCF parse-bound readers — irrelevant to
  decode throughput).
- No new public GVL API. The benchmark is dev tooling under `benchmarking/`.
- No changes to PR #286's *semantics* — only its performance, gated by the
  byte-identical oracle.

## vcfixture-rs edits (only if needed)

Consume the existing `bulk` CLI as-is. Edit vcfixture-rs only if the harness
needs a capability its bulk generator lacks (e.g. a payload/profile/sizing knob
required to hold density constant across the sample sweep). Any such edit is a
separate change in the sibling repo with its own commit + note here.

## Git / delivery

- Work in `.claude/worktrees/svar2-spliced-bench` on
  `perf/svar2-spliced-variants-bench` (branched from PR #286 head).
- Fresh pixi env for the worktree (do not share `.pixi`); rebuild the Rust
  extension with `maturin develop --release` after any `src/` change before
  running the Python benchmark.
- Push to an `origin` (mcvickerlab) branch; open a **draft** PR whose body
  carries the before/after sweep table, stacked so the numbers attach to #286.
```

