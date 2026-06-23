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

All backends (Track 1a): restricted to diploid GTs.  The `bcftools consensus
  -H 2` oracle is undefined for haploid samples (there is no second haplotype),
  so gvl and bcftools diverge on haploid GTs in a platform-dependent way (linux
  bcftools keeps REF for the absent hap; macOS bcftools applies the allele).
  Diploid half-calls (1|., .|1) remain in scope; only true haploid GTs are
  excluded.

vcf/svar: otherwise a broad draw — any diploid GT class (phased, unphased,
  half-call, missing) is exercised.  Both backends preserve VCF allele order and
  match the bcftools consensus oracle on all diploid GT classes.

pgen: additionally restricted to phased diploid GTs only.  plink2 (used to
  convert VCF → PGEN) canonicalizes unphased het allele order (e.g. 0/1 and 1/0
  are both stored as (0,1,unphased)), diverging from the VCF oracle.  Phased
  diploid GTs, including phased half-calls (1|. and .|1 under --vcf-half-call
  r), are preserved faithfully.
"""

from __future__ import annotations

import sys
import tempfile
from itertools import product
from pathlib import Path

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


def _all_diploid(doc) -> bool:
    """Return True if every sample genotype in every record is diploid.

    The Track 1b AF oracle (``_oracle_afs_from_vcf``) uses the fixed-2N
    convention (alt count / ``2 * n_samples``), which is only well-defined when
    every genotype is diploid. Haploid genotypes (e.g. ``GT=1``) make the AF
    denominator ambiguous, so they are excluded from the AF property test.
    """
    for record in doc.records:
        for sample_fields in record.samples:
            gt = sample_fields.get("GT")
            if gt is not None and len(gt.alleles) != 2:
                return False
    return True


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
@given(
    case_inputs=st.reference_and_documents(
        violations=_ALL_VIOLATIONS,
        max_samples=2,
        max_records=4,
        max_contigs=2,
        max_contig_len=2000,
        max_repeats=3,
    )
)
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
    # gvl.write is hardened to accept zero-region input (gvl #201).
    assume(_has_any_alt(doc))
    # The `bcftools consensus -H 2` oracle is undefined for haploid samples:
    # there is no second haplotype, so gvl (which applies the single allele to
    # the requested hap) and bcftools diverge — and the divergence is
    # platform-dependent (linux bcftools keeps REF for the absent hap; macOS
    # bcftools applies the allele).  Restrict to diploid GTs for all backends,
    # mirroring the Track 1b AF oracle.  Diploid half-calls (1|., .|1) stay in
    # scope; only true haploid GTs (len(alleles) == 1) are excluded.
    assume(_all_diploid(doc))
    # pgen only: plink2 canonicalizes unphased het allele order, causing
    # divergence from the bcftools consensus oracle.  Phased diploid GTs (incl.
    # phased half-calls) are faithful.  vcf and svar preserve VCF order and
    # don't need this filter.
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
                    actual,
                    desired,
                    f"src={src} region={region} sample={sample} hap={h}",
                )


# ---------------------------------------------------------------------------
# Track 1b — AF exposure on the svar backend
# ---------------------------------------------------------------------------


def _oracle_afs_from_vcf(vcf_path) -> np.ndarray:
    """Compute per-variant AF from post-norm VCF genotypes using the fixed-2N convention.

    genoray/gvl uses ``alt_allele_count / (2 * n_samples)`` — missing alleles
    count in the denominator (fixed-2N), NOT called-AN (which would exclude
    missing alleles from the denominator).  This was confirmed empirically on
    the session case: variants at chr1:1210696 with ``1|.`` / ``.|1`` / ``0/1``
    GTs have AF=0.666667 and AF=0.166667 respectively, which matches fixed-2N
    but NOT called-AN (which gives 0.8 and 0.2).

    Records are returned in VCF iteration order, which matches the SVAR index
    order (both are sorted chrom/pos; both use 1-based coordinates).

    Only bi-allelic records (post-normalization) are expected; each record's
    ALT allele count is the number of allele values equal to 1 across all
    samples, treating None (missing) as 0.
    """
    oracle = []
    with pysam.VariantFile(str(vcf_path)) as vcf:
        samples = list(vcf.header.samples)
        n_samples = len(samples)
        fixed_denom = 2 * n_samples
        for rec in vcf.fetch():
            alt_count = 0
            for samp in samples:
                gt = rec.samples[samp]["GT"]
                for a in gt:
                    if a == 1:
                        alt_count += 1
            oracle.append(alt_count / fixed_denom)
    return np.array(oracle, dtype=np.float32)


def test_track1b_af_matches_vcf_oracle_on_session_case(synthetic_case):
    """gvl's per-variant AF array matches an independent post-norm-VCF oracle.

    Oracle: per-variant AF computed directly from ``case.vcf_path`` (the
    bgzipped, indexed, post-normalization VCF that the SVAR dataset is built
    from).  Convention: ``alt_allele_count / (2 * n_samples)`` (fixed-2N —
    missing alleles count in the denominator).

    This is a GENUINE cross-check: the oracle is derived independently of
    ``index.arrow`` (which is where gvl reads AF from).  A bug in genoray's
    ``cache_afs()`` AF computation would cause this test to fail, whereas the
    previous index.arrow comparison would have passed silently.

    Axes:
    - ``n_checked`` = number of variants in the SVAR index (17 for the session
      case: the chr2:1110696 multiallelic splits into 2 bi-allelic records, and
      the chr2:1234567 microsat A>GA,AC splits into 2 bi-allelic records after
      the stale INFO/AC strip that fixed the bcf_calc_ac retention bug).
    - ``n_rv_checked`` = number of (region, sample, hap, variant) cells across
      all rv arrays where an AF value was retrieved and validated.

    AF field: exposed only when ``var_fields`` includes ``"AF"``.  The field
    is a per-hap list of float32 values — one per variant in that haplotype's
    window.  AF is the same for every haplotype that carries a given variant
    (population-level, not per-haplotype).
    """
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

    # 1. Independent oracle from the post-norm VCF (not from index.arrow).
    gvl_afs = ds._seqs.variants.info["AF"]  # type: ignore[attr-defined]
    oracle_afs = _oracle_afs_from_vcf(synthetic_case.vcf_path)

    n_checked = len(gvl_afs)
    assert n_checked > 0, "No variants in SVAR index — test is vacuous"
    assert len(oracle_afs) == n_checked, (
        f"VCF oracle has {len(oracle_afs)} records but gvl has {n_checked} variants; "
        "ordering assumption violated"
    )
    np.testing.assert_allclose(
        gvl_afs,
        oracle_afs,
        atol=1e-6,
        err_msg=(
            "gvl AF array differs from independent VCF-derived oracle — "
            "cache_afs() AF computation is incorrect"
        ),
    )

    # 2. Verify all AF values are in [0, 1].
    assert np.all((gvl_afs >= 0.0) & (gvl_afs <= 1.0)), (
        f"gvl AFs contain out-of-range values: {gvl_afs}"
    )

    # 3. Verify each rv.AF value is one of the valid AF values from the oracle
    #    (validates the per-cell indexing path).
    oracle_af_set = set(float(x) for x in oracle_afs)
    n_rv_checked = 0
    for region in range(ds.n_regions):
        for sample in ds.samples:
            rv = ds[region, sample]
            for h in range(2):
                # rv.AF is a Ragged of shape (p, ~v) after scalar squeeze;
                # rv.AF[h] is a numpy array of AF values for haplotype h.
                for af in rv.AF[h].tolist():
                    assert any(abs(af - a) < 1e-6 for a in oracle_af_set), (
                        f"rv.AF value {af!r} at region={region} sample={sample} "
                        f"hap={h} is not present in the VCF oracle AF array"
                    )
                    n_rv_checked += 1

    assert n_rv_checked > 0, (
        "No AF values were checked across rv cells — test is vacuous"
    )


@settings(max_examples=15, deadline=None, suppress_health_check=_SUPPRESS)
@given(
    case_inputs=st.reference_and_documents(
        violations=frozenset(),
        max_samples=2,
        max_records=4,
        max_contigs=2,
        max_contig_len=2000,
        max_repeats=3,
    )
)
def test_af_and_dosage_consistent(case_inputs):
    """Property test: AF bounds + AF consistency on the svar backend.

    Tests the svar backend only (the only backend with cached AFs via
    ``cache_afs()``).  vcf/pgen do not expose AF in this pipeline.

    Assertions:
    (a) Every AF value exposed via rv.AF is in [0.0, 1.0].
    (b) The gvl AF array matches the independent post-norm-VCF oracle
        element-by-element (validates the full cache_afs() → gvl.write →
        read pipeline for arbitrary inputs).

    Oracle convention: ``alt_allele_count / (2 * n_samples)`` (fixed-2N —
    missing alleles count in the denominator).  This was confirmed on the
    session case; see ``_oracle_afs_from_vcf`` for details.

    Dosage is NOT tested here: ``build_case`` does not write ``dosages.npy``
    by default, so ``dosage`` is not in ``available_var_fields``.  Per-sample
    alt dosage would require calling ``SparseVar(...).cache_dosages()`` (or
    equivalent) before ``gvl.write``, which is outside the current pipeline.
    """
    spec, doc, _truth = case_inputs
    assume(len(doc.records) > 0)
    # STOPGAP: all-REF docs produce zero-variant VCFs after _normalize.
    # Exclude until gvl.write is hardened to accept zero-variant input (gvl #201).
    assume(_has_any_alt(doc))
    # The fixed-2N AF oracle is only well-defined for diploid genotypes; haploid
    # GTs (e.g. GT=1) make the AF denominator ambiguous. Exclude them.
    assume(_all_diploid(doc))

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

        # (b) Independent oracle: compute AF directly from the post-norm VCF.
        gvl_afs = ds._seqs.variants.info["AF"]  # type: ignore[attr-defined]
        oracle_afs = _oracle_afs_from_vcf(case.vcf_path)
        assert len(oracle_afs) == len(gvl_afs), (
            f"VCF oracle has {len(oracle_afs)} records but gvl has {len(gvl_afs)} variants"
        )
        np.testing.assert_allclose(
            gvl_afs,
            oracle_afs,
            atol=1e-6,
            err_msg="gvl AF array differs from independent VCF-derived oracle",
        )

        # (a) All AF values must be in [0, 1].
        assert np.all((gvl_afs >= 0.0) & (gvl_afs <= 1.0)), (
            f"gvl AFs out of range: {gvl_afs}"
        )

        # Also confirm the rv-level AF values are valid (exercises the per-cell
        # indexing path on random inputs, not just the internal array).
        oracle_af_set = set(float(x) for x in oracle_afs)
        for region in range(ds.n_regions):
            for sample in ds.samples:
                rv = ds[region, sample]
                for h in range(2):
                    # rv.AF is a Ragged of shape (p, ~v) after scalar squeeze;
                    # rv.AF[h] is a numpy array of AF values for haplotype h.
                    for af in rv.AF[h].tolist():
                        assert 0.0 <= af <= 1.0, (
                            f"rv.AF {af!r} out of [0,1] at "
                            f"region={region} sample={sample} hap={h}"
                        )
                        assert any(abs(af - a) < 1e-6 for a in oracle_af_set), (
                            f"rv.AF {af!r} not in VCF oracle AF array at "
                            f"region={region} sample={sample} hap={h}"
                        )


# ---------------------------------------------------------------------------
# Track 2 — clean rejection of non-canonical raw input
# ---------------------------------------------------------------------------

import hypothesis.strategies as hyp  # noqa: E402


def _raw_write_vcf(doc, tmp) -> None:
    """Render the RAW (un-normalized) doc and feed it directly to gvl.write.

    Deliberately skips the consensus/pgen/svar steps that full build_case runs,
    so gvl.write is the FIRST place any error can originate (bcftools consensus
    and plink2 would error first on raw violating input, masking the gvl.write
    ValueError we want to assert).

    Steps: render+bgzip+index the raw VCF -> derive BED -> VCF reader ->
    gvl.write.  No reference write, no normalization, no oracle, no pgen/svar.
    gvl.write does not require a reference FASTA (variants= path only).
    """
    import genvarloader as gvl
    from genoray import VCF
    from case import _bgzip_index, _derive_bed  # module-level helpers

    tmp = Path(tmp)
    raw_bytes = doc.render().encode()
    vcf_gz = _bgzip_index(raw_bytes, tmp / "raw.vcf.gz")
    bed = _derive_bed(vcf_gz, None)
    # If the violating doc produced no usable variant rows, _derive_bed returns
    # an empty DataFrame and gvl.write would fail with a no-data error unrelated
    # to the violation.  Skip such draws.
    assume(bed.height > 0)
    bed_path = tmp / "source.bed"
    bed.select(
        "chrom",
        "start",
        "end",
        pl.lit(".").alias("name"),
        pl.lit(".").alias("score"),
        "strand",
    ).write_csv(bed_path, include_header=False, separator="\t")
    reader = VCF(vcf_gz)
    if not reader._valid_index():
        reader._write_gvi_index()
    reader._load_index()
    gvl.write(path=tmp / "ds.gvl", bed=bed_path, variants=reader, max_jitter=2)


def _has_violation_label(doc, label: str) -> bool:
    """Return True if any record in doc carries the given violation label."""
    return any(label in record.labels for record in doc.records)


@hyp.composite
def _spec_and_violating_doc(draw, violation):
    spec = draw(st.references(max_contigs=2, max_contig_len=2000, max_repeats=3))
    doc = draw(
        st.documents(
            reference=spec,
            violations={violation},
            max_samples=2,
            max_records=4,
        )
    )
    return spec, doc


@settings(max_examples=10, deadline=None, suppress_health_check=_SUPPRESS)
@given(case_inputs=_spec_and_violating_doc("multiallelic"))
def test_multiallelic_raw_is_rejected(case_inputs):
    """gvl.write rejects raw multiallelic input (already enforced)."""
    _spec, doc = case_inputs
    # violations={"multiallelic"} enables but does not guarantee a multiallelic
    # record appears in every draw; skip draws that contain none.
    assume(_has_violation_label(doc, "multiallelic"))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="multi-allelic"):
            _raw_write_vcf(doc, tmp)


@pytest.mark.xfail(
    strict=False, reason="gvl #199: gvl.write does not validate atomization"
)
@settings(max_examples=10, deadline=None, suppress_health_check=_SUPPRESS)
@given(case_inputs=_spec_and_violating_doc("non_atomic"))
def test_non_atomic_raw_is_rejected(case_inputs):
    """gvl SHOULD reject raw non-atomic input; currently it does not (xfail #199)."""
    _spec, doc = case_inputs
    # violations={"non_atomic"} enables but does not guarantee a non-atomic
    # record appears in every draw; skip draws that contain none.
    assume(_has_violation_label(doc, "non_atomic"))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            _raw_write_vcf(doc, tmp)


@pytest.mark.xfail(
    strict=False, reason="gvl #200: gvl.write does not validate left-alignment"
)
@settings(max_examples=10, deadline=None, suppress_health_check=_SUPPRESS)
@given(case_inputs=_spec_and_violating_doc("non_left_aligned"))
def test_non_left_aligned_raw_is_rejected(case_inputs):
    """gvl SHOULD reject raw non-left-aligned input; currently it does not (xfail #200)."""
    _spec, doc = case_inputs
    # violations={"non_left_aligned"} labels records with "off_anchor"; skip draws
    # that contain no non-left-aligned record.
    assume(_has_violation_label(doc, "off_anchor"))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            _raw_write_vcf(doc, tmp)


# ---------------------------------------------------------------------------
# Track 2 (continued) — deterministic hand-crafted rejection tests
# ---------------------------------------------------------------------------
# vcfixture silently ignores violation labels it doesn't recognize — its
# records strategy only checks for "multiallelic", "non_atomic", and
# "non_left_aligned". Passing "symbolic"/"breakend" would produce clean VCFs
# with no such records, making hypothesis property tests vacuous, so the
# hand-crafted tests below are the coverage for those classes.

# A minimal VCF containing one clean SNP, one symbolic <DEL>, and one
# breakend ALT. Used to assert gvl.write rejects symbolic/breakend inputs.
_SYM_BND_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=2000>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">
##ALT=<ID=DEL,Description="Deletion">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1
chr1\t100\t.\tA\tT\t.\t.\t.\tGT\t0|1\t1|0
chr1\t200\t.\tG\t<DEL>\t.\t.\tSVTYPE=DEL;END=300;SVLEN=-100\tGT\t0|1\t0|0
chr1\t400\t.\tC\tC[chr1:600[\t.\t.\tSVTYPE=BND\tGT\t0|1\t0|0
"""


def _write_vcf_text(text: str, tmp) -> tuple[Path, Path]:
    """bgzip+index the given VCF text and derive a BED. Returns (vcf_gz, bed_path)."""
    from case import _bgzip_index, _derive_bed

    tmp = Path(tmp)
    vcf_gz = _bgzip_index(text.encode(), tmp / "raw.vcf.gz")
    bed = _derive_bed(vcf_gz, None)
    bed_path = tmp / "source.bed"
    bed.select(
        "chrom",
        "start",
        "end",
        pl.lit(".").alias("name"),
        pl.lit(".").alias("score"),
        "strand",
    ).write_csv(bed_path, include_header=False, separator="\t")
    return vcf_gz, bed_path


def test_symbolic_breakend_vcf_is_rejected():
    """gvl.write rejects a VCF containing symbolic and breakend ALTs."""
    import genvarloader as gvl
    from genoray import VCF

    with tempfile.TemporaryDirectory() as tmp:
        vcf_gz, bed_path = _write_vcf_text(_SYM_BND_VCF, tmp)
        reader = VCF(vcf_gz)
        if not reader._valid_index():
            reader._write_gvi_index()
        reader._load_index()
        with pytest.raises(ValueError, match="symbolic"):
            gvl.write(
                path=Path(tmp) / "ds.gvl",
                bed=bed_path,
                variants=reader,
                max_jitter=2,
            )


def test_symbolic_breakend_svar_is_rejected():
    """gvl.write rejects a .svar built (unfiltered) from symbolic/breakend input."""
    import genvarloader as gvl
    from genoray import SparseVar, VCF

    with tempfile.TemporaryDirectory() as tmp:
        vcf_gz, bed_path = _write_vcf_text(_SYM_BND_VCF, tmp)
        svar_path = Path(tmp) / "v.svar"
        # No filter: symbolic/breakend records are carried into the .svar index.
        SparseVar.from_vcf(svar_path, VCF(vcf_gz), max_mem="1g", overwrite=True)
        with pytest.raises(ValueError, match="symbolic"):
            gvl.write(
                path=Path(tmp) / "ds.gvl",
                bed=bed_path,
                variants=SparseVar(svar_path),
                max_jitter=2,
            )
