# vcfixture test-data migration — design

**Date:** 2026-05-30
**Branch / worktree:** `worktree-test-vcfixture` (`.claude/worktrees/test-vcfixture`)
**Status:** approved design, pending implementation plan

## Problem

GenVarLoader's toy test fixtures depend on two heavy, fragile inputs:

- a **3 GB hg38 download** (via `pooch` in `tests/data/generate_ground_truth.py`), and
- a **hand-authored `tests/data/source.vcf`** whose variant positions are
  carefully hand-spaced so a polars `group_by`/`diff` heuristic can derive BED
  regions, with hardcoded "manual" cases (a spanning deletion at
  `chr19:1010696`, a no-variant region at `chr1:10e6`).

From these, the generator produces committed artifacts the integration suite
reads directly: filtered `vcf/`, `pgen/`, `filtered.svar/`, the
`phased_dataset.{vcf,pgen,svar}.gvl` datasets, and the bcftools-`consensus`
haplotype FASTAs in `consensus/` (the haplotype-reconstruction oracle).

We own [vcfixture](https://github.com/d-laub/vcfixture) (local checkout at
`/Users/david/projects/vcfixture`), a library that programmatically generates
small spec-correct VCFs **with decoded ground truth** (`GroundTruth`: numpy
genotypes, positions, ref/alts, variant class, phasing, INFO/FORMAT), a
`Reference` helper that draws spec-correct REF/ALT from a FASTA, and Hypothesis
`strategies`. It can eliminate the hand-authored VCF and, paired with a small
synthetic reference, the hg38 download — and turn example-based tests into
property-based ones.

## What vcfixture replaces, and the gap

- **Replaces well:** variant/genotype-level fixtures. gvl's loaded genotypes,
  allele frequencies, and variant records can be asserted directly against
  vcfixture's `GroundTruth`.
- **The gap:** gvl's deepest oracle is the *reconstructed haplotype sequence*
  (reference with variants applied). vcfixture's `GroundTruth` stops at decoded
  variant records; it does **not** reconstruct haplotype sequences. So the
  haplotype oracle stays on **`bcftools consensus`**, which we keep — but run
  against a tiny synthetic reference instead of hg38.
- **Strategy gap:** vcfixture's current `strategies.documents` draws REF bases
  at random, not tied to a reference. `bcftools norm` rejects REF that does not
  match the reference, so a *reference-consistent* document strategy must be
  added upstream before Phase 2.

## Goal

Kill the 3 GB hg38 download and the hand-authored `source.vcf` for the **toy**
datasets, replacing them with a small committed synthetic reference plus
programmatic vcfixture VCF generation; then progressively move to in-test
Hypothesis-driven generation and delete the committed toy fixtures entirely.

## Phasing

One branch, bundled into a single PR per maintainer preference.

### Phase 1 — replace generation, prove parity

Acceptance gate: **`pixi run -e dev gen` then `pixi run -e dev test` green with
no hg38 download.**

1. Add `vcfixture>=0.2.1` (bumped as needed, see Upstream) as a dev dependency
   in `pixi.toml`.
2. Rewrite `tests/data/generate_ground_truth.py`, keyed off a fixed seed for
   determinism:
   - **Synthetic reference.** Write a small bgzipped, `faidx`-indexed FASTA of
     random ACGT with arbitrary contigs (e.g. `chr1:100_000`, `chr2:20_000`).
     Small enough (~tens of KB) to **commit**, so toy tests need no download.
   - **Programmatic VCF** via `VcfBuilder`, REF drawn from the synthetic
     reference (`Reference.draw_ref_alt`), reproducing the scenario set the
     hand-authored file encoded: groups of nearby SNPs/indels within
     `SEQ_LEN//2`, a deliberate spanning deletion, a no-variant region,
     multiallelic + atomization coverage, sample names `s0,s1,s2`, diploid.
   - **Deterministic BED** constructed directly from known variant positions
     (replacing the polars `group_by`/`diff` heuristic), `max_jitter=2`.
   - **Unchanged tail:** bcftools `norm` (left-align) → atomize/split
     multiallelic → `consensus` per sample/hap (oracle), plink2 PGEN, genoray
     `SparseVar`, the three `gvl.write` calls.
3. Repoint `conftest.py` fixtures (`ref_fasta` → committed synthetic FASTA;
   `source_vcf` consumers) and run the audit sweep (below) until the suite is
   green against regenerated fixtures.

`tests/data/generate_1kg_ground_truth.py` is untouched and keeps its hg38
download.

### Phase 2 — property-based, delete committed files

1. Land the upstream **reference-consistent `documents` strategy** in vcfixture
   (optionally emitting `(reference, document, truth)` together) and release it.
2. Add a property-test module (e.g.
   `tests/integration/dataset/test_haps_property.py`) using Hypothesis:
   - per example: write synthetic reference + VCF, run bcftools norm +
     consensus → oracle haplotypes, `gvl.write`, then assert
     - gvl reconstructed haplotypes **==** consensus oracle, and
     - gvl genotypes / AF **==** vcfixture `GroundTruth`.
   - run across VCF/PGEN/SVAR sources so all three loaders are property-tested.
   - modest Hypothesis settings (small examples, capped count) to keep the
     suite fast; bcftools/plink2/samtools already in the dev env.
3. As coverage lands, delete `phased_dataset.*.gvl`, `consensus/`,
   `source.vcf`, the toy `fasta/` download, and now-dead generation branches;
   migrate remaining integration tests onto in-test or session-scoped
   generation fixtures.

## vcfixture upstream additions

Developed in `/Users/david/projects/vcfixture`, released, then consumed here by
version bump — no git/path pins to unreleased vcfixture.

- **Phase 1:** `vcfixture.reference.write_random_fasta(path, contigs, seed)` —
  generic random-FASTA generator.
- **Phase 2:** reference-consistent `documents` strategy (REF drawn from a
  supplied `Reference`), possibly paired `(reference, document, truth)` output.

## Parity decisions

- **Synthetic contig naming:** arbitrary names/sizes, breaking from the current
  hardcoded `chr1`/`chr19` real-hg38 positions.
- **Sample names & ploidy:** standardize the toy world on `s0,s1,s2`, diploid,
  and update the handful of hardcoded `NA00001–3` references rather than force
  vcfixture to emit `NA0000x`.

## Test parity & migration mechanics

The Phase 1 risk is hidden hardcoded assumptions. Steps:

1. **Audit sweep** — grep tests for `NA0000`, `chr19`, `1010696`,
   `10e6`/`10000000`, literal positions/sample names, and uses of the
   `ref_fasta`/`reference`/`source_vcf` fixtures; catalog required changes.
2. **Repoint fixtures** in `conftest.py`.
3. **Regenerate & run** (`gen` then `test`) until green — the Phase 1
   acceptance gate.
4. **Phase 2 deletions & migration** as above.
5. Verify `skills/genvarloader/SKILL.md` does not depend on toy fixtures (it
   tracks public API, so likely no change).

## Out of scope

- **1kg slow-tier** fixtures (real Zenodo data + bcftools consensus on hg38) —
  keep their own generation path and hg38 download.
- **bigwig / track** fixtures and the track re-alignment oracle (no bcftools
  equivalent) — keep existing tiny synthetic contigs and committed `.bw` files.
