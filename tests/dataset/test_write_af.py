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


@pytest.fixture(scope="module")
def vcf_with_af(tmp_path_factory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("write_af_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF_WITH_AF)
    vcf_gz = d / "in.vcf.gz"
    subprocess.run(["bcftools", "view", "-Oz", "-o", str(vcf_gz), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(vcf_gz)], check=True)
    return vcf_gz, ref


@pytest.fixture(scope="module")
def regions() -> pl.DataFrame:
    return pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [len(_REF)]})


def test_write_caches_af_from_info(vcf_with_af, regions, tmp_path):
    vcf_gz, ref = vcf_with_af
    out = tmp_path / "ds"
    gvl.write(out, regions, variants=str(vcf_gz), overwrite=True)

    assert "AF" in _Variants.available_info_fields(out / "genotypes" / "variants.arrow")


def test_write_af_less_vcf_has_no_af_column(streaming_case, tmp_path):
    regions, _reference, vcf_no_af, _written = streaming_case("vcf")
    out = tmp_path / "ds"
    gvl.write(out, regions, variants=vcf_no_af, overwrite=True)

    assert "AF" not in _Variants.available_info_fields(
        out / "genotypes" / "variants.arrow"
    )
