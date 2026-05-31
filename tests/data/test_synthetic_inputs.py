"""Unit tests for the synthetic reference + source-VCF re-encoding helpers."""

from __future__ import annotations

from pathlib import Path

import pysam

from _synthetic import FLANK_GUARDS, write_synthetic_reference

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
