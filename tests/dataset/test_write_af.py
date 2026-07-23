"""Written path: `gvl.write` caches `AF` from a VCF's `INFO/AF` field into the
`.gvi` index (Wave B PR-B2, #317/#319, Task 5).

A VCF header declaring `INFO/AF` should end up with an `AF` column in the
written dataset's `variants.arrow`, mirroring the streaming SVAR1 path so a
written VCF-sourced `Dataset` can also `min_af`/`max_af`-filter. AF-less VCFs
must write unchanged (no spurious `AF` column).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader._dataset._haps import _Variants

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"  # chr1, 40bp
# idx2='A' (pos3), idx9='G' (pos10), idx15='T' (pos16)
assert len(_REF) == 40
assert _REF[2] == "A" and _REF[9] == "G" and _REF[15] == "T"

_VCF_WITH_AF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##INFO=<ID=AF,Number=1,Type=Float,Description="Allele frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\tAF=0.1\tGT\t1|0\t0|0
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t0|0
chr1\t16\t.\tT\tC\t.\t.\tAF=0.77\tGT\t1|1\t0|1
"""

# Realistic case: ``Number=A`` is the standard real-world INFO/AF declaration
# (one value per ALT allele). oxbow/genoray return this as a `List(Float32)`
# column, which must be coerced to scalar before it survives the
# `is_numeric()` filter in `_Variants.available_info_fields`/`from_table`.
# All records here are bi-allelic (one ALT each), so every list has length 1.
_VCF_WITH_AF_NUMBER_A = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\tAF=0.05\tGT\t1|0\t0|0
chr1\t10\t.\tG\tC\t.\t.\tAF=0.42\tGT\t1|1\t0|0
chr1\t16\t.\tT\tC\t.\t.\tAF=0.91\tGT\t1|1\t0|1
"""
# Expected AF values in ascending-POS order (3, 10, 16); a sorted single-contig
# VCF is read in POS order end-to-end (oxbow record order == file order), and
# `_attach_af_column` cross-checks POS alignment before attaching, so POS order
# is preserved all the way into `variants.arrow`.
_EXPECTED_AF_NUMBER_A = [0.05, 0.42, 0.91]


def _build_indexed_vcf(vcf_text: str, d: Path) -> Path:
    vcf = d / "in.vcf"
    vcf.write_text(vcf_text)
    vcf_gz = d / "in.vcf.gz"
    subprocess.run(["bcftools", "view", "-Oz", "-o", str(vcf_gz), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(vcf_gz)], check=True)
    return vcf_gz


@pytest.fixture(scope="module")
def vcf_with_af(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("write_af_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf_gz = _build_indexed_vcf(_VCF_WITH_AF, d)
    return vcf_gz, ref


@pytest.fixture(scope="module")
def vcf_with_af_number_a(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("write_af_number_a_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf_gz = _build_indexed_vcf(_VCF_WITH_AF_NUMBER_A, d)
    return vcf_gz, ref


@pytest.fixture(scope="module")
def regions() -> pl.DataFrame:
    return pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [len(_REF)]})


def test_write_caches_af_from_info(vcf_with_af, regions, tmp_path):
    vcf_gz, ref = vcf_with_af
    out = tmp_path / "ds"
    gvl.write(out, regions, variants=str(vcf_gz), overwrite=True)

    assert "AF" in _Variants.available_info_fields(out / "genotypes" / "variants.arrow")


def test_write_caches_af_number_a_values_round_trip(
    vcf_with_af_number_a, regions, tmp_path
):
    """``Number=A`` INFO/AF (the realistic real-world declaration) comes back from
    oxbow as `List(Float32)`; `_attach_af_column` must coerce it to scalar or the
    column is silently dropped by `is_numeric()` filtering downstream. This test
    verifies actual cached AF VALUES round-trip (not just column presence) --
    without the `List` -> scalar coercion in `_attach_af_column` this fails with
    `AF` absent from `v.info`.
    """
    vcf_gz, ref = vcf_with_af_number_a
    out = tmp_path / "ds"
    gvl.write(out, regions, variants=str(vcf_gz), overwrite=True)

    v = _Variants.from_table(out / "genotypes" / "variants.arrow")
    assert "AF" in v.info

    # POS-aligned comparison: the .gvi index and variants.arrow preserve VCF
    # (ascending-POS) record order end-to-end, so zipping POS with AF and
    # sorting by POS recovers per-record AF values unambiguously.
    order = np.argsort(v.start)
    np.testing.assert_allclose(
        np.asarray(v.info["AF"])[order], _EXPECTED_AF_NUMBER_A, rtol=1e-5
    )


def test_write_af_less_vcf_has_no_af_column(streaming_case, tmp_path):
    regions, _reference, vcf_no_af, _written = streaming_case("vcf")
    out = tmp_path / "ds"
    gvl.write(out, regions, variants=vcf_no_af, overwrite=True)

    assert "AF" not in _Variants.available_info_fields(
        out / "genotypes" / "variants.arrow"
    )


# A bi-allelic record whose `Number=.` INFO/AF still carries the full un-subset
# multiallelic AF list (`0.333,0.667`) after a `bcftools norm -m` split -- the
# ALT->AF mapping is ambiguous. `gvl.write()` must NOT raise (writing such a VCF
# is valid for non-AF use); it warns and declines to cache AF. Regression guard
# for a review-fix that originally raised here, breaking every unrelated test
# that writes this common fixture shape (test_output_format, test_unphased_union,
# test_flat_mode_equivalence).
_VCF_MULTIVALUE_AF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##INFO=<ID=AF,Number=.,Type=Float,Description="Allele frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\tAF=0.333,0.667\tGT\t1|0\t0|1
chr1\t16\t.\tT\tC\t.\t.\tAF=0.5\tGT\t1|1\t0|1
"""


def test_write_multivalue_af_writes_without_af_column(tmp_path):
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    vcf_gz = _build_indexed_vcf(_VCF_MULTIVALUE_AF, tmp_path)
    regions = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [len(_REF)]}
    )
    out = tmp_path / "ds"
    # Must NOT raise (the historic bug raised ValueError here).
    gvl.write(out, regions, variants=str(vcf_gz), overwrite=True)
    # AF is ambiguous -> not cached; write otherwise succeeded.
    assert "AF" not in _Variants.available_info_fields(
        out / "genotypes" / "variants.arrow"
    )
