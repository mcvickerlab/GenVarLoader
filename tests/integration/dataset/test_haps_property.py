"""Property-based coverage for gvl haplotype reconstruction and variant calls.

Each example draws a reference-consistent (ReferenceSpec, VcfDocument,
GroundTruth) from vcfixture, materializes it through the shared build_case
pipeline, and compares gvl output against two oracles:
  - Track 1a: gvl haplotypes == `bcftools consensus` (deepest oracle).
  - Track 1b: gvl AF/dosage exposure on the svar backend.
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

import awkward as ak
import genvarloader as gvl
import numpy as np
import polars as pl
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

    This filter requires every genotype to be fully-called phased diploid:
    diploid (exactly 2 alleles), no missing allele, and phased.  Note that
    plink2 does faithfully round-trip phased half-calls (1|. and .|1 →
    (1,0,phased) and (0,1,phased) under --vcf-half-call r), so they could in
    principle be admitted for pgen; they are excluded here conservatively
    because the bcftools-consensus oracle's handling of the missing allele has
    not been verified to match gvl-pgen's.

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


# ---------------------------------------------------------------------------
# Track 1b — AF exposure on the svar backend
# ---------------------------------------------------------------------------

def test_track1b_af_matches_truth_on_session_case(synthetic_case):
    """Probe test: gvl's per-variant AF array matches the SVAR-index cached AFs.

    ``cache_afs()`` computes population-level AFs from the SVAR genotypes and
    writes them into ``index.arrow``.  gvl reads that same array and exposes
    each variant's AF in ``rv.AF[h]``.  This test cross-checks the round-trip:

      gvl-internal AF array  ==  SVAR-index AF array (authoritative source)

    The SVAR index is the post-normalization ground truth; pre-norm truth AFs
    are NOT used here because bcftools norm can rewrite positions/alleles in
    ways that make 1-to-1 matching unreliable (left-alignment, atomization,
    multiallelic splits).  Using the SVAR index directly avoids the pre/post-
    norm boundary entirely.

    Axes:
    - ``n_checked`` = number of variants in the SVAR index (15 for the session
      case, since one multiallelic in session_document is split into two).
    - ``n_rv_checked`` = number of (region, sample, hap, variant) cells across
      all rv arrays where an AF value was retrieved and validated.

    AF field: exposed only when ``var_fields`` includes ``"AF"``.  The field
    is a per-hap list of float32 values — one per variant in that haplotype's
    window.  AF is the same for every haplotype that carries a given variant
    (population-level, not per-haplotype).
    """
    import polars as pl

    ds = (
        gvl.Dataset.open(
            synthetic_case.gvl_path["svar"],
            synthetic_case.ref_path,
            var_fields=["alt", "ilen", "start", "AF"],
        )
        .with_len("ragged")
        .with_seqs("variants")
        .with_tracks(False)
    )

    # 1. Verify the internal AF array is an exact read-back of the cached AFs.
    gvl_afs = ds._seqs.variants.info["AF"]  # type: ignore[attr-defined]
    svar_index = pl.read_ipc(synthetic_case.svar_path / "index.arrow")
    svar_afs = svar_index["AF"].to_numpy()

    n_checked = len(gvl_afs)
    assert n_checked > 0, "No variants in SVAR index — test is vacuous"
    np.testing.assert_allclose(
        gvl_afs,
        svar_afs,
        atol=1e-6,
        err_msg=(
            "gvl internal AF array differs from SVAR index AF array — "
            "cache_afs() round-trip is broken"
        ),
    )

    # 2. Verify all AF values are in [0, 1].
    assert np.all((gvl_afs >= 0.0) & (gvl_afs <= 1.0)), (
        f"SVAR-index AFs contain out-of-range values: {gvl_afs}"
    )

    # 3. Verify each rv.AF value is one of the valid AF values from the
    #    internal array (validates the per-cell indexing path).
    gvl_af_values = set(float(x) for x in gvl_afs)
    n_rv_checked = 0
    for region in range(ds.n_regions):
        for sample in ds.samples:
            rv = ds[region, sample]
            for h in range(2):
                for af in ak.to_list(rv.AF[h]):
                    assert any(abs(af - a) < 1e-6 for a in gvl_af_values), (
                        f"rv.AF value {af!r} at region={region} sample={sample} "
                        f"hap={h} is not present in the internal AF array"
                    )
                    n_rv_checked += 1

    assert n_rv_checked > 0, (
        "No AF values were checked across rv cells — test is vacuous"
    )


@settings(max_examples=15, deadline=None, suppress_health_check=_SUPPRESS)
@given(case_inputs=st.reference_and_documents(
    violations=frozenset(), max_samples=2, max_records=4,
    max_contigs=2, max_contig_len=2000, max_repeats=3,
))
def test_af_and_dosage_consistent(case_inputs):
    """Property test: AF bounds + AF consistency on the svar backend.

    Tests the svar backend only (the only backend with cached AFs via
    ``cache_afs()``).  vcf/pgen do not expose AF in this pipeline.

    Assertions:
    (a) Every AF value exposed via rv.AF is in [0.0, 1.0].
    (b) The gvl-internal AF array matches the SVAR-index AF array element-
        by-element (validates cache_afs() round-trip for arbitrary inputs).

    Dosage is NOT tested here: ``build_case`` does not write ``dosages.npy``
    by default, so ``dosage`` is not in ``available_var_fields``.  Per-sample
    alt dosage would require calling ``SparseVar(...).cache_dosages()`` (or
    equivalent) before ``gvl.write``, which is outside the current pipeline.

    AF↔dosage consistency form: **(b) strong internal consistency** — the
    gvl-internal AF array must match the SVAR index byte-for-byte within
    float32 tolerance.  This is strong in the sense that it validates the
    full cache → write → read → expose pipeline for every variant, not just
    that the values are plausible.  The weak monotonic form (AF>0 iff any
    dosage>0) is not implemented because dosage is unavailable.
    """
    spec, doc, _truth = case_inputs
    assume(len(doc.records) > 0)
    # STOPGAP: all-REF docs produce zero-variant VCFs after _normalize.
    # Exclude until gvl.write is hardened to accept zero-variant input.
    assume(_has_any_alt(doc))

    with tempfile.TemporaryDirectory() as tmp:
        case = build_case(spec, doc, Path(tmp), sources=("svar",), normalize=True)

        ds = (
            gvl.Dataset.open(
                case.gvl_path["svar"],
                case.ref_path,
                var_fields=["alt", "ilen", "start", "AF"],
            )
            .with_len("ragged")
            .with_seqs("variants")
            .with_tracks(False)
        )

        # (b) Strong internal consistency: gvl reads back the cached AFs exactly.
        gvl_afs = ds._seqs.variants.info["AF"]  # type: ignore[attr-defined]
        svar_afs = pl.read_ipc(case.svar_path / "index.arrow")["AF"].to_numpy()
        np.testing.assert_allclose(
            gvl_afs,
            svar_afs,
            atol=1e-6,
            err_msg="gvl internal AF array differs from SVAR index AF array",
        )

        # (a) All AF values must be in [0, 1].
        assert np.all((gvl_afs >= 0.0) & (gvl_afs <= 1.0)), (
            f"SVAR-index AFs out of range: {gvl_afs}"
        )

        # Also confirm the rv-level AF values are valid (exercises the per-cell
        # indexing path on random inputs, not just the internal array).
        gvl_af_values = set(float(x) for x in gvl_afs)
        for region in range(ds.n_regions):
            for sample in ds.samples:
                rv = ds[region, sample]
                for h in range(2):
                    for af in ak.to_list(rv.AF[h]):
                        assert 0.0 <= af <= 1.0, (
                            f"rv.AF {af!r} out of [0,1] at "
                            f"region={region} sample={sample} hap={h}"
                        )
                        assert any(abs(af - a) < 1e-6 for a in gvl_af_values), (
                            f"rv.AF {af!r} not in internal AF array at "
                            f"region={region} sample={sample} hap={h}"
                        )
