from pathlib import Path

from pytest_cases import fixture

import genvarloader as gvl


@fixture
def pgen_variants():
    pgen_path = Path(__file__).parent / "data" / "pgen" / "filtered_sample.pgen"
    sample_names = ["NA00001", "NA00002", "NA00003"]
    return gvl.variants.Variants.from_pgen(pgen_path, sample_names)


@fixture
def vcf_variants():
    vcf_path = Path(__file__).parent / "data" / "vcf" / "filtered_sample.vcf"
    return gvl.variants.Variants.from_vcf(vcf_path)


def test_vcf_to_tstore(vcf_variants):
    gvl.ZarrGenos.from_recs_genos(
        vcf_variants.records, vcf_variants.genotypesm, overwrite=True
    )
