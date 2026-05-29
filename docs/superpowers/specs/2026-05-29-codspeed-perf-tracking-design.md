# Design: codspeed perf tracking + py-spy/memray profiling for the GVL 0.6.1 parity regression

**Date:** 2026-05-29
**Status:** approved design, pending implementation plan
**Related:** [`docs/superpowers/REGRESSIONS.md`](../REGRESSIONS.md)

## Goal

Two deliverables:

1. A **pytest-codspeed benchmark suite** that tracks the perf of the hot numba functions
   and the four generation pipelines (haplotype, variant, track, re-aligned track), so the
   ~18–20× regression documented in `REGRESSIONS.md` — and any future regression — is caught
   continuously.
2. A **py-spy + memray profiling pass** over the tracks, haplotype, and variant hot paths to
   identify where the regression lives, written up against the existing hypotheses in
   `REGRESSIONS.md`.

Decisions locked in during brainstorming:

- Benchmark granularity: **both layers** (isolated numba functions + end-to-end reconstructor).
- Measurement mode: **walltime + warmup** (works under both `pytest-benchmark` and `pytest --codspeed`).
- CI: **local-only** this round (no CodSpeed GitHub Action).
- Benchmark data: **realistic** — chr22 1kGP variants + GEUVADIS read-depth tracks, 5-for-5
  sample overlap, committed as a small self-contained `.gvl`.
- Micro-layer inputs: **sliced from the committed `.gvl`** (real genotype sparsity, interval
  distributions, indel sizes).

## Background: the regression

`REGRESSIONS.md` records that track dataloading is ~18–20× slower and uses dramatically more
RAM on 0.24.1 vs 0.6.1, confirmed by a controlled parity test. The fingered hot path is
tracks-only re-alignment: `with_seqs(None).with_tracks("read-depth").with_len(16384)`,
single numba thread. The suggested investigation order is: (1) profile the serial tracks-only
hot path, (4) find what now scales with seqlen at fixed nucleotides-per-batch. This design
operationalizes (1) for tracks, haplotypes, and variants, and adds continuous regression
tracking so any fix is measurable.

## Data: committed chr22 1kGP + GEUVADIS slice

### Sources (on `/carter`, not committed)

- Genotypes: `/carter/users/dlaub/data/1kGP/plink2/hg38.norm.{pgen,psam,pvar.zst}` (full 1kGP, hg38, normalized).
- Tracks: `/carter/users/dlaub/data/1kGP-rna-seq/bw_chr22/*.bw` (pre-extracted chr22 GEUVADIS read-depth BigWigs).
- Sample map: `/carter/users/dlaub/data/1kGP-rna-seq/sample_id_to_bigwig.csv` (452 GEUVADIS RNA-seq samples → BigWig path).
- Regions: `/carter/users/dlaub/data/1kGP-rna-seq/chr22_egenes.bed` (few hundred chr22 eGenes).
- Reference: hg38 from `/carter/users/dlaub/data/1kGP-rna-seq/Homo_sapiens_UCSC_hg38.tar.gz`.

### Sample-overlap finding

The pre-existing 5-sample thin slice (`/carter/users/dlaub/data/1kGP/plink2/gvl-tests/1kgp.thin`)
overlaps GEUVADIS in only **1 of 5** samples (HG00342), so it cannot be reused. The full chr22
1kGP set overlaps GEUVADIS in **451 samples**, so selecting 5 with both genotypes and a BigWig
is trivial.

### Build script — `tests/benchmarks/data/build_realistic.py` (committed; run once)

Regenerates the committed artifacts from the `/carter` sources. Steps:

1. **Sample selection** — pick 5 samples present in both `hg38.norm.psam` and
   `sample_id_to_bigwig.csv`, spanning a couple of populations (e.g. CEU + YRI). Write the
   chosen IDs to a committed `samples.txt` so the slice is deterministic.
2. **Variants** — `plink2 --pfile hg38.norm --chr 22 --keep <5 samples> --make-pgen` → a tiny
   chr22 PGEN (real indels, which drive re-alignment). The exact bcftools/plink2 preprocessing
   must follow gvl's documented variant-prep requirements (left-align, bi-allelic, atomized).
3. **Regions** — copy `chr22_egenes.bed` as the benchmark BED.
4. **Tracks** — the 5 matching `bw_chr22/*.bw` read-depth BigWigs.
5. **Reference** — extract chr22 from hg38, mask everything outside the benchmark regions to
   `N`, then bgzip → `chr22.masked.fa.gz`. A mostly-`N` chr22 compresses to a few MB while
   staying coordinate-correct for haplotype reconstruction over the benchmark windows.
6. **`gvl.write(...)`** the above → committed `chr22_geuv.gvl` (sparse genotypes + re-aligned
   interval tracks). 5 samples × few-hundred chr22 regions keeps it small.

### Committed layout — `tests/benchmarks/data/`

```
tests/benchmarks/data/
├── build_realistic.py     # regenerate from /carter (committed, not run in CI)
├── samples.txt            # the chosen 5 samples (committed)
├── chr22_egenes.bed       # benchmark regions (committed copy)
├── chr22.masked.fa.gz     # masked reference (committed, ~MBs)
└── chr22_geuv.gvl/        # built dataset (committed)
```

Benchmark fixtures load the committed dataset directly — no `/carter` dependency at bench time.
`build_realistic.py` is only for regeneration and requires `/carter`.

> **Size gate:** before committing `chr22_geuv.gvl` and `chr22.masked.fa.gz`, confirm the total
> committed footprint is acceptable (target: low tens of MB at most, consistent with existing
> committed test data under `data/`). If the dataset is larger than expected, reduce the region
> count (sample from `chr22_egenes.bed`) until it fits.

## Component 1 — benchmark suite (`tests/benchmarks/`)

```
tests/benchmarks/
├── data/                  # (above)
├── conftest.py            # session fixtures: open dataset; extract micro inputs once
├── test_micro.py          # isolated numba fns
├── test_e2e.py            # reconstructor __call__ paths
└── profiling/
    └── profile.py         # py-spy + memray driver (--mode tracks|haplotypes|variants)
```

### Fixtures — `conftest.py`

- Session-scoped fixture opens `chr22_geuv.gvl` (with `chr22.masked.fa.gz` as reference) once.
- A second session-scoped fixture slices out the real low-level arrays each micro-bench needs
  (sparse genotypes/offsets, ref-coords, interval values/offsets, indel diffs) — extracted once,
  reused across micro-benches. This gives realistic shapes and distributions without re-deriving
  per bench.

### Micro layer — `test_micro.py`

One bench per hot function, using the `benchmark` fixture (works under both `pytest-benchmark`
and `pytest --codspeed` walltime+warmup). Each bench:

- Performs an explicit warmup call before the measured region (numba `cache=True` is set, but
  the first in-process call still links).
- Asserts the output is non-degenerate once (the "honest" check) so a benchmark that silently
  degenerates to a no-op gets caught.

| Category | Function | Module |
|---|---|---|
| Haplotype | `reconstruct_haplotypes_from_sparse`, `get_diffs_sparse` | `_genotypes.py` |
| Track | `intervals_to_tracks` | `_intervals.py` |
| Re-aligned track | `shift_and_realign_tracks_sparse` | `_tracks.py` |
| Variant | `_infer_germline_ccfs` (+ `RaggedVariants` assembly) | `_rag_variants.py` |

### End-to-end layer — `test_e2e.py`

Reconstructor `__call__` over the opened dataset, at `with_len(16384)` (the regression seqlen):

- Haps: `haplotypes`, `annotated`, and `variants` kinds.
- Tracks.
- Tracks-only: `with_seqs(None).with_tracks("read-depth")` — the exact path `REGRESSIONS.md`
  fingered.

## Component 2 — pixi.toml

- Add `pytest-codspeed` to `[dependencies]`. (`pytest-benchmark`, `py-spy`, and `memray` are
  already present.)
- Tasks:
  - `bench` — `pytest tests/benchmarks --codspeed` (falls back to plain `pytest-benchmark` when
    `--codspeed` is unavailable).
  - `profile-tracks` / `profile-haps` / `profile-variants` — thin drivers around
    `profiling/profile.py --mode ...` under py-spy.
  - `memray-tracks` / `memray-haps` / `memray-variants` — the same modes under memray.
- No CodSpeed GitHub Action this round (local-only).

## Component 3 — profiling pass + write-up

`profiling/profile.py` takes `--mode {tracks,haplotypes,variants}`, runs single numba thread
(`NUMBA_NUM_THREADS=1`) over the committed chr22 dataset at `with_len(16384)`:

- **tracks** — `with_seqs(None).with_tracks("read-depth")` (the regression target).
- **haplotypes** — `with_seqs("haplotypes")` (reconstruction from real 1kGP indels + masked reference).
- **variants** — `with_seqs("variants")` (`RaggedVariants` assembly).

Each mode runs under both py-spy (sampling → speedscope/flamegraph) and memray (allocation →
peak-RSS attribution). After analysis, append a **"Profiling results (local, chr22 GEUVADIS
slice)"** section to `REGRESSIONS.md` mapping each pipeline's hot spots to the existing
hypotheses (serial bottleneck #1, memory-scales-with-seqlen #4), with the caveat that absolute
parity numbers still need the cluster-scale dataset.

## Out of scope

- CodSpeed CI / GitHub Action wiring (deferred; local-only this round).
- Synthetic input generators / scaling sweeps beyond the chr22 region set (micro inputs are
  sliced from the real `.gvl`).
- Root-causing or fixing the regression itself — this design produces the measurement and
  profiling infrastructure and the profiling write-up; fixes follow separately.
- Genome-wide slices (chr22-only).
- Restoring `tbb` / multi-thread scaling fixes (tracked separately in `REGRESSIONS.md`).

## Open risks

- **Committed dataset size** — gated above; reduce region count if needed.
- **Variant-generation bench** depends on `RaggedVariants` being meaningfully exercisable on the
  real phased 1kGP slice — it is, via the standard write path.
- **codspeed walltime variance** locally — mitigated by warmup + steady-state measurement; the
  micro layer is the low-noise regression signal, the e2e layer is the realistic-orchestration signal.
