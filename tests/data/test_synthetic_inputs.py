"""Unit tests for the synthetic reference + source-VCF re-encoding helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pysam

from _synthetic import FLANK_GUARDS, build_source_vcf, write_synthetic_reference

# (contig, 1-based pos, expected REF) for every variant locus in the source VCF.
LOCI = [
    ("chr19", 111, "N"),
    ("chr19", 1010696, "GAGACGGGGCC"),
    ("chr19", 1110696, "A"),
    ("chr19", 1210696, "C"),
    ("chr19", 1210697, "T"),
    ("chr20", 14370, "N"),
    ("chr20", 17330, "N"),
    ("chr20", 1110696, "G"),
    ("chr20", 1234567, "A"),
]


def test_reference_has_expected_bases_at_loci(tmp_path: Path):
    ref = write_synthetic_reference(tmp_path / "synthetic.fa.bgz", seed=0)
    assert ref.exists()
    assert (ref.parent / (ref.name + ".fai")).exists()
    with pysam.FastaFile(str(ref)) as fa:
        for contig, pos, expected in LOCI:
            got = fa.fetch(contig, pos - 1, pos - 1 + len(expected)).upper()
            assert got == expected, f"{contig}:{pos} got {got!r}, want {expected!r}"


def test_reference_is_deterministic(tmp_path: Path):
    a = write_synthetic_reference(tmp_path / "a.fa.bgz", seed=0)
    b = write_synthetic_reference(tmp_path / "b.fa.bgz", seed=0)
    with pysam.FastaFile(str(a)) as fa, pysam.FastaFile(str(b)) as fb:
        assert fa.fetch("chr19", 0, 5000) == fb.fetch("chr19", 0, 5000)


def test_reference_has_flank_guards(tmp_path: Path):
    ref = write_synthetic_reference(tmp_path / "synthetic.fa.bgz", seed=0)
    with pysam.FastaFile(str(ref)) as fa:
        for contig, pos, guard in FLANK_GUARDS:
            got = fa.fetch(contig, pos - 2, pos - 1).upper()
            assert got == guard, f"{contig}:{pos} guard got {got!r}, want {guard!r}"


def _norm(vcf_path: Path, ref_path: Path) -> str:
    """Left-align with bcftools norm; return normalized VCF text."""
    out = subprocess.run(
        ["bcftools", "norm", "-f", str(ref_path), str(vcf_path)],
        check=True,
        capture_output=True,
    )
    return out.stdout.decode()


def test_source_vcf_re_encode_matches_header_and_samples(tmp_path: Path):
    ref = write_synthetic_reference(tmp_path / "synthetic.fa.bgz", seed=0)
    doc = build_source_vcf(ref)
    text = doc.render()
    # Samples preserved exactly.
    header_line = next(l for l in text.splitlines() if l.startswith("#CHROM"))
    assert header_line.endswith("NA00001\tNA00002\tNA00003")
    # INFO defs required by test_sitesonly are declared.
    for fid in ("NS", "DP", "AF"):
        assert f"##INFO=<ID={fid}," in text


def test_source_vcf_passes_norm_and_preserves_coupled_positions(tmp_path: Path):
    ref = write_synthetic_reference(tmp_path / "synthetic.fa.bgz", seed=0)
    doc = build_source_vcf(ref)
    vcf = doc.write(tmp_path / "source.vcf", bgzip=False)
    normalized = _norm(vcf, ref)  # raises if any REF mismatches the reference
    # The 10-bp deletion hardcoded by test_write_edge_cases must survive at pos.
    assert any(
        line.startswith("chr19\t1010696\t") and "GAGACGGGGCC" in line
        for line in normalized.splitlines()
    ), "chr19:1010696 10-bp deletion shifted or lost during norm"
    # chr20 multiallelic and microsat records present.
    assert any(l.startswith("chr20\t1110696\t") for l in normalized.splitlines())
    assert any(l.startswith("chr20\t1234567\t") for l in normalized.splitlines())
