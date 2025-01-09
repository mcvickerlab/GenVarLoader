import re
from pathlib import Path
from typing import Union

VCF_SUFFIX = re.compile(r"\.[vb]cf(\.gz)?$")


def path_is_vcf(path: Union[str, Path]) -> bool:
    path = Path(path)
    return VCF_SUFFIX.search(path.name) is not None


def path_is_pgen(path: Union[str, Path]) -> bool:
    path = Path(path)
    return path.suffix == ".pgen" or Path(str(path) + ".pgen").exists()
