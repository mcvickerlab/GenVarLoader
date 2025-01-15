from pathlib import Path

import genvarloader as gvl
from pytest_cases import fixture


@fixture
def pgen_variants():
    pgen_path = Path(__file__).parent / "data" / "pgen" / "filtered_sample.pgen"
    return gvl.Variants.from_file(pgen_path)


@fixture
def vcf_variants():
    vcf_path = Path(__file__).parent / "data" / "vcf" / "filtered_sample.vcf"
    return gvl.Variants.from_file(vcf_path)
