"""Unit tests for ``Svar2Link`` resolution + fingerprint integrity.

Mirrors ``test_svar_link_models.py`` but for the ``.svar2`` back-reference
(``_svar2_link.py``). Three pure/tmp_path tests exercise the override/no-op
error paths; one integration-flavored test builds a real ``.svar2`` store
(via genoray's conversion pipeline, same fixture recipe as
``tests/test_svar2_reconstruct.py``) to prove the fingerprint actually
detects a mutated store.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from genvarloader._dataset._svar2_link import (
    Svar2Fingerprint,
    Svar2Link,
    _resolve_svar2,
    _verify_svar2_fingerprint,
    make_svar2_link,
)

# Same tiny fixture recipe as tests/test_svar2_reconstruct.py::svar2_store.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


def test_resolve_prefers_override(tmp_path: Path):
    real = tmp_path / "cohort.svar2"
    real.mkdir()
    link = Svar2Link(
        relative_path="nope.svar2",
        absolute_path="/nope.svar2",
        fingerprint=Svar2Fingerprint(n_files=1, store_bytes=1),
    )
    assert _resolve_svar2(tmp_path, link, real) == real


def test_resolve_missing_override_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _resolve_svar2(tmp_path, None, tmp_path / "absent.svar2")


def test_verify_none_link_is_noop(tmp_path: Path):
    _verify_svar2_fingerprint(tmp_path, None)  # must not raise


@pytest.fixture(scope="module")
def svar2_store(tmp_path_factory) -> Path:
    from genoray import _core

    d = tmp_path_factory.mktemp("svar2_link")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store.svar2"
    _core.run_conversion_pipeline(
        str(bcf),
        str(ref),
        ["chr1"],
        str(out),
        ["S0", "S1"],
        25_000,
        2,
        1,
        8 * 1024 * 1024,
    )
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


def test_fingerprint_detects_mutated_store(svar2_store: Path, tmp_path: Path):
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()

    link = make_svar2_link(gvl_path, svar2_store)
    _verify_svar2_fingerprint(svar2_store, link)  # must not raise

    bin_files = sorted(svar2_store.rglob("*.bin"))
    assert bin_files, "expected at least one .bin file in a real .svar2 store"
    with open(bin_files[0], "ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError):
        _verify_svar2_fingerprint(svar2_store, link)
