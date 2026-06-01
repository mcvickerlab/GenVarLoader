"""Unit tests for the standardized session reference + source document."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pysam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_builders"))
from case import session_document, session_reference  # noqa: E402

# (contig, 1-based pos, expected REF) for every variant locus.
LOCI = [
    ("chr1", 111, "N"),
    ("chr1", 1010696, "GAGACGGGGCC"),
    ("chr1", 1110696, "A"),
    ("chr1", 1210696, "C"),
    ("chr1", 1210697, "T"),
    ("chr2", 14370, "N"),
    ("chr2", 17330, "N"),
    ("chr2", 1110696, "G"),
    ("chr2", 1234567, "A"),
]


def test_reference_has_expected_bases_at_loci(tmp_path: Path):
    ref = session_reference().write(tmp_path / "synthetic.fa.bgz")
    assert ref.exists()
    assert (ref.parent / (ref.name + ".fai")).exists()
    with pysam.FastaFile(str(ref)) as fa:
        for contig, pos, expected in LOCI:
            got = fa.fetch(contig, pos - 1, pos - 1 + len(expected)).upper()
            assert got == expected, f"{contig}:{pos} got {got!r}, want {expected!r}"


def test_reference_is_deterministic(tmp_path: Path):
    a = session_reference().write(tmp_path / "a.fa.bgz")
    b = session_reference().write(tmp_path / "b.fa.bgz")
    with pysam.FastaFile(str(a)) as fa, pysam.FastaFile(str(b)) as fb:
        assert fa.fetch("chr1", 0, 5000) == fb.fetch("chr1", 0, 5000)


def test_reference_n_masks_chr1_telomere(tmp_path: Path):
    ref = session_reference().write(tmp_path / "synthetic.fa.bgz")
    with pysam.FastaFile(str(ref)) as fa:
        assert fa.fetch("chr1", 0, 150).upper() == "N" * 150


def _norm(vcf_path: Path, ref_path: Path) -> str:
    out = subprocess.run(
        ["bcftools", "norm", "-f", str(ref_path), str(vcf_path)],
        check=True, capture_output=True,
    )
    return out.stdout.decode()


def test_source_vcf_samples_and_info(tmp_path: Path):
    spec = session_reference()
    spec.write(tmp_path / "synthetic.fa.bgz")
    text = session_document(spec).render()
    header_line = next(l for l in text.splitlines() if l.startswith("#CHROM"))
    assert header_line.endswith("s0\ts1\ts2")
    for fid in ("NS", "DP", "AF"):
        assert f"##INFO=<ID={fid}," in text


def test_source_vcf_passes_norm_and_preserves_coupled_positions(tmp_path: Path):
    spec = session_reference()
    ref = spec.write(tmp_path / "synthetic.fa.bgz")
    vcf = session_document(spec).write(tmp_path / "source.vcf", bgzip=False)
    normalized = _norm(vcf, ref)
    assert any(
        line.startswith("chr1\t1010696\t") and "GAGACGGGGCC" in line
        for line in normalized.splitlines()
    ), "chr1:1010696 10-bp deletion shifted or lost during norm"
    assert any(l.startswith("chr2\t1110696\t") for l in normalized.splitlines())
    assert any(l.startswith("chr2\t1234567\t") for l in normalized.splitlines())
