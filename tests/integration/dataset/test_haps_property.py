"""Property-based coverage for gvl haplotype reconstruction and variant calls.

Each example draws a reference-consistent (ReferenceSpec, VcfDocument,
GroundTruth) from vcfixture, materializes it through the shared build_case
pipeline, and compares gvl output against two oracles:
  - Track 1a: gvl haplotypes == `bcftools consensus` (deepest oracle).
  - Track 1b: gvl applied-allele counts + AF == vcfixture GroundTruth.
  - Track 2:  gvl cleanly rejects non-canonical raw input.

Each example shells out to bcftools/plink2/samtools, so deadlines are disabled
and per-example isolation uses a TemporaryDirectory created inside the test body
(NOT a function-scoped fixture — that trips Hypothesis's health check).

Backend-specific filters
------------------------
All-REF records are dropped by build_case._normalize (``bcftools view --min-ac
1``) before gvl.write sees the VCF.  This enforces a gvl.write invariant and
removes the need to gate on _has_any_alt globally.

vcf/svar: broad draw — any GT class (phased, unphased, half-call, haploid,
  missing) is exercised.  Both backends preserve VCF allele order and therefore
  match the bcftools consensus oracle on all GT classes.

pgen: restricted to phased diploid GTs only.  plink2 (used to convert VCF →
  PGEN) diverges from the VCF oracle in two ways:
    (a) Unphased hets (e.g. 0/1 and 1/0) are both stored as (0,1,unphased) —
        allele order is canonicalized.
    (b) Haploid GTs (e.g. GT=1) are promoted to homozygous diploid (1/1).
  Phased diploid GTs, including phased half-calls (1|. and .|1 under
  --vcf-half-call r), are preserved faithfully.
"""
from __future__ import annotations

import sys
import tempfile
from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import pytest
import seqpro as sp
import vcfixture.strategies as st
from hypothesis import HealthCheck, assume, given, settings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_builders"))
from case import build_case  # noqa: E402

_SUPPRESS = [HealthCheck.too_slow, HealthCheck.filter_too_much]
_ALL_VIOLATIONS = frozenset({"multiallelic", "non_atomic", "non_left_aligned"})


def _has_any_alt(doc) -> bool:
    """Return True if any sample in any record carries at least one ALT allele."""
    for record in doc.records:
        for sample_fields in record.samples:
            gt = sample_fields.get("GT")
            if gt is not None and any(a != 0 for a in gt.alleles if a is not None):
                return True
    return False


def _all_gts_phased_diploid(doc) -> bool:
    """Return True if every sample genotype in every record is phased diploid.

    Used only for the pgen backend.  plink2 diverges from the VCF oracle in two
    empirically verified ways:

      (a) Unphased hets (0/1, 1/0) → both stored as (0,1,unphased); allele order
          is canonicalized, so pgen output differs from bcftools consensus.
      (b) Haploid GTs (GT=1) → promoted to homozygous diploid (1/1); both
          haplotypes get the ALT even when only one should.

    Phased diploid GTs are fully faithful, including phased half-calls (1|. and
    .|1 → (1,0,phased) and (0,1,phased) under --vcf-half-call r).

    vcf and svar preserve VCF allele order and do not need this filter.
    """
    for record in doc.records:
        for sample_fields in record.samples:
            gt = sample_fields.get("GT")
            if gt is None:
                continue
            if len(gt.alleles) != 2:
                return False
            if any(a is None for a in gt.alleles):
                return False
            # phased=(True,) means the genotype separator is '|' (phased)
            if not gt.phased or not gt.phased[0]:
                return False
    return True


@pytest.mark.parametrize("src", ["vcf", "pgen", "svar"])
@settings(max_examples=25, deadline=None, suppress_health_check=_SUPPRESS)
@given(case_inputs=st.reference_and_documents(
    violations=_ALL_VIOLATIONS, max_samples=2, max_records=4,
    max_contigs=2, max_contig_len=2000, max_repeats=3,
))
def test_haplotypes_match_consensus(src, case_inputs):
    spec, doc, _truth = case_inputs
    # Zero-record documents produce no BED regions and nothing to reconstruct;
    # skip them rather than treating them as failures.
    assume(len(doc.records) > 0)
    # STOPGAP (Change 3): gvl.write + plink2 both crash on a zero-variant VCF
    # (gvl.write: NoDataError on empty BED CSV; plink2: "No variants in --vcf
    # file" RuntimeError).  build_case._normalize now drops all-REF records via
    # `bcftools view --min-ac 1`, so a doc whose records are entirely all-REF
    # will produce a zero-variant VCF after prep.  Exclude such docs here until
    # gvl.write is hardened to accept zero-region input (tracked separately).
    assume(_has_any_alt(doc))
    # pgen only: plink2 canonicalizes unphased het allele order and promotes
    # haploid GTs to homozygous diploid, causing divergence from the bcftools
    # consensus oracle.  Phased diploid GTs (incl. phased half-calls) are
    # faithful.  vcf and svar preserve VCF order and don't need this filter.
    if src == "pgen":
        assume(_all_gts_phased_diploid(doc))
    with tempfile.TemporaryDirectory() as tmp:
        case = build_case(spec, doc, Path(tmp), sources=(src,), normalize=True)
        ds = (
            gvl.Dataset.open(case.gvl_path[src], case.ref_path, rc_neg=False)
            .with_len("ragged")
            .with_seqs("haplotypes")
            .with_tracks(False)
        )
        for region, sample in product(range(ds.n_regions), case.samples):
            haps = ds[region, sample]
            for h in range(2):
                actual = sp.cast_seqs(haps[h])
                fa_path = case.consensus_dir / f"source_{sample}_nr{region}_h{h}.fa"
                with pysam.FastaFile(str(fa_path)) as f:
                    desired = sp.cast_seqs(f.fetch(f.references[0]).upper())
                np.testing.assert_equal(
                    actual, desired,
                    f"src={src} region={region} sample={sample} hap={h}",
                )
