from pathlib import Path
from tempfile import NamedTemporaryFile

from genvarloader._variants._utils import path_is_pgen, path_is_vcf


def test_path_is_pgen():
    assert path_is_pgen(Path("test.pgen"))

    with NamedTemporaryFile(suffix=".pgen") as f:
        assert path_is_pgen(f.name[:-5])


def test_path_is_vcf():
    assert path_is_vcf(Path("test.vcf"))
    assert path_is_vcf(Path("test.vcf.gz"))
    assert path_is_vcf(Path("test.bcf"))
    assert path_is_vcf(Path("test.bcf.gz"))
