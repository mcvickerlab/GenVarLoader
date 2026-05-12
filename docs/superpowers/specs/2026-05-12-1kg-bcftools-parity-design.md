# 1KG bcftools-consensus Parity Tests — Design

**Date:** 2026-05-12
**Status:** Draft

## Motivation

The existing haplotype reconstruction tests (`tests/dataset/test_ds_haps.py`)
verify parity with `bcftools consensus` using a small, hand-authored synthetic
VCF (`tests/data/source.vcf`, 20 bp regions). This catches gross bugs but does
not exercise:

- Dense, realistic variant distributions
- Long-range haplotypes (>20 bp) where many indels compound
- Real allele frequencies / heterozygosity patterns from 1000 Genomes

This spec adds a complementary real-data parity test using the chr21/chr22
subset of 1000 Genomes data for 5 individuals, hosted at
[Zenodo record 20132907](https://zenodo.org/records/20132907). The test
re-uses the exact same parity model as the synthetic test: for each
(region, sample, haplotype), `Dataset[region, sample][h]` must byte-equal
the output of `bcftools consensus -H {h+1} -s {sample}` on the same
normalized BCF.

## Scope

In scope:
- New data-preparation script that downloads the BCF, derives PGEN and SVAR,
  picks regions, writes three `.gvl` datasets, and emits per-call FASTAs.
- New test that asserts byte parity for all three backends.
- A pixi task to run the data prep on demand.

Out of scope:
- Track / interval parity (this is a haplotype-only test).
- Jitter, RC packing, splice — covered by other tests; this test uses
  `rc_neg=False` and no jitter so the comparison is a strict byte equality.
- Adding the generated `.gvl` directories or FASTAs to the repo. Data prep is
  invoked manually (or via the new pixi task) and writes under
  `tests/data/1kg/`.

## Files

- `tests/data/generate_1kg_ground_truth.py` — new data-prep script.
- `tests/dataset/test_ds_haps_1kg.py` — new parity test.
- `pixi.toml` — add a `gen-1kg` task. `test` is **not** modified; the new
  test is `@pytest.mark.slow` so it is excluded by the default
  pytest invocation.

## Data Layout

```
tests/data/
├── fasta/hg38.fa.bgz          # reused from existing generate_ground_truth.py
├── 1kg/
│   ├── source.bcf             # pooch-downloaded from Zenodo
│   ├── source.bcf.csi         # pooch-downloaded from Zenodo
│   ├── filtered.bcf           # normalized (left-align + atomize + split)
│   ├── filtered.bcf.csi
│   ├── filtered.pgen/.pvar/.psam
│   ├── filtered.svar/         # genoray SparseVar dir
│   ├── regions.bed            # 100 random regions
│   ├── phased_1kg.bcf.gvl/    # gvl.write output from BCF reader
│   ├── phased_1kg.pgen.gvl/
│   └── phased_1kg.svar.gvl/
└── 1kg_consensus/
    └── 1kg_{sample}_nr{row}_h{hap}.fa
```

## Data-Prep Pipeline

`tests/data/generate_1kg_ground_truth.py`. Implementation closely mirrors
`generate_ground_truth.py`; only the steps that differ are spelled out in
detail.

1. **Fetch BCF and CSI** via `pooch.retrieve` with a `known_hash`. The
   Zenodo record exposes both files as separate downloads; each is fetched
   independently to `tests/data/1kg/`. Hashes are filled in on first run
   (script logs the observed hash and exits if `known_hash` is the
   placeholder `None`, instructing the developer to update the file).
2. **Reference.** Assert `tests/data/fasta/hg38.fa.bgz` and its `.fai` /
   `.csi` exist. If not, instruct the user to run `pixi run -e dev gen`
   first. This avoids duplicating the 3 GB fetch.
3. **Normalize** with `bcftools norm -f <ref>` then
   `bcftools norm -a --atom-overlaps . -f <ref> -m -` (left-align, atomize,
   split multiallelics). Output bgzipped + indexed as `filtered.bcf`.
4. **PGEN**: `plink2 --bcf filtered.bcf --make-pgen --vcf-half-call r --out filtered`.
5. **SVAR**: `SparseVar.from_vcf(out, VCF(filtered.bcf), "50mb")` then
   `cache_afs()`.
6. **Pick regions** (seed = 0, N = 100, length = 10_000):
   - Read variant positions from the normalized BCF (chrom + pos only;
     polars from `bcftools view`).
   - Restrict to chr21 and chr22.
   - Uniformly sample 100 variants without replacement; for each, the
     region is `[pos - 5000, pos + 5000)` clipped to a reasonable margin
     (skip variants within 5 kb of contig start). With 10 kb windows on
     these chroms, each region will span many variants across the 5
     samples, exercising compound indel reconstruction.
   - Assign a random strand `+/-` (carried but unused — `rc_neg=False`).
   - Write `regions.bed` with columns
     `chrom, start, end, name=".", score=".", strand`.
7. **Write three datasets** via `gvl.write`:
   - `VCF(filtered.bcf)` → `phased_1kg.bcf.gvl`
   - `PGEN(filtered.pgen)` → `phased_1kg.pgen.gvl`
   - `SparseVar(filtered.svar)` → `phased_1kg.svar.gvl`

   All three with `max_jitter=0` (default) to keep the comparison strict.
8. **Generate ground-truth FASTAs.** For each `(row_nr, chrom, start, end)`
   in the BED, for each of the 5 samples, for `hap in (0, 1)`:

   ```
   samtools faidx <ref> {chrom}:{start+1}-{end} \
     | bcftools consensus -H {hap+1} -s {sample} -o <out.fa> filtered.bcf
   samtools faidx <out.fa>
   ```

   Output written to `tests/data/1kg_consensus/1kg_{sample}_nr{row}_h{hap}.fa`.
   Progress bar with total `n_regions * n_samples * 2` (= 1000).

The script is idempotent: re-running cleans the `1kg_consensus/`,
`phased_1kg.*.gvl/`, and `filtered.*` outputs before regenerating. The
pooched BCF/CSI are cached by pooch and not re-downloaded.

## Test

`tests/dataset/test_ds_haps_1kg.py`. Structure mirrors `test_ds_haps.py`
exactly:

```python
import pytest
from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import seqpro as sp
from genvarloader._ragged import RaggedSeqs
from pytest_cases import parametrize_with_cases

data_dir = Path(__file__).resolve().parents[1] / "data"
ref = data_dir / "fasta" / "hg38.fa.bgz"
cons_dir = data_dir / "1kg_consensus"

pytestmark = pytest.mark.slow


def dataset_bcf():
    return (
        gvl.Dataset.open(data_dir / "1kg" / "phased_1kg.bcf.gvl", ref, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )

# ... pgen, svar identical pattern ...


@parametrize_with_cases("dataset", cases=".", prefix="dataset_")
def test_ds_haps_1kg(dataset: gvl.RaggedDataset[RaggedSeqs, None]):
    for region, sample in product(range(dataset.n_regions), dataset.samples):
        c, s, e, rc = dataset.regions.select(
            "chrom", "chromStart", "chromEnd", "strand"
        ).row(region)
        haps = dataset[region, sample]
        for h in range(2):
            actual = sp.cast_seqs(haps[h])
            fpath = f"1kg_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(cons_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]).upper())
            np.testing.assert_equal(
                actual,
                desired,
                f"region: {region}, sample: {sample}, hap: {h}, "
                f"coords: {c}:{s + 1}-{e}",
            )
```

Each parametrize case is one backend. Total assertions: 3 backends × 100
regions × 5 samples × 2 haps = 3000.

## Pixi Tasks

Add a single new task; do not touch `test` or `gen`.

```toml
[tasks]
gen-1kg = { cmd = "python tests/data/generate_1kg_ground_truth.py", depends-on = ["gen"] }
```

`depends-on = ["gen"]` ensures the reference FASTA exists. The new test is
discovered by `pytest tests` but skipped unless `-m slow` is passed, which
is the project's existing convention.

## Failure Modes and Edge Cases

- **`known_hash` not yet populated.** First run prints the observed sha256
  of each downloaded file and exits with a clear message. Subsequent runs
  validate.
- **Reference missing.** Script exits early with the message "run
  `pixi run -e dev gen` first".
- **Variant near contig boundary.** Filter candidate variants to those
  with at least 5 kb of flanking sequence on the chromosome before
  sampling.
- **N/2 allele in haplotype.** `bcftools consensus -H {hap+1}` emits the
  observed allele; the gvl readers handle this via the normal variant
  application path. If parity fails on these, that is the bug the test
  is designed to catch.
- **Reference allele case.** `bcftools consensus` may emit lowercase
  reference bases (soft-masked regions in hg38). The test uses
  `.upper()` on the consensus FASTA before comparison, matching
  `test_ds_haps.py`.

## Non-Goals / YAGNI Notes

- No CI integration. The downloaded data is large and the test is slow;
  it is intended to be run manually or in a dedicated nightly job that
  the maintainer can wire up later.
- No support for re-running with different seeds or region counts as
  CLI flags initially. Constants at the top of the script suffice; we
  can promote them to `typer` options if a second use case appears.
- No parity test for tracks. This test is haplotype-only.
