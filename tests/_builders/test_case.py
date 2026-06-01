"""Tests for the shared build_case fixture builder."""
from __future__ import annotations

from pathlib import Path

import genvarloader as gvl
import pysam
from vcfixture import ReferenceBuilder, VcfBuilder


def _tiny_spec_and_doc():
    """A minimal canonical, reference-consistent (spec, doc): one SNP, two samples."""
    ref = (
        ReferenceBuilder()
        .add_contig("chr1", 200)
        .set_base("chr1", 99, "A")  # 0-based; REF for the record below
        .build()
    )
    b = VcfBuilder(samples=["s0", "s1"], contigs=[("chr1", 200)], fileformat="VCFv4.0")
    b.fmt("GT")
    b.record("chr1", 100, ref="A", alt=["C"], gt=["0|1", "1|1"])  # 1-based pos
    return ref, b.build()


def test_build_case_produces_all_artifacts(tmp_path: Path):
    from tests._builders.case import build_case

    spec, doc = _tiny_spec_and_doc()
    case = build_case(spec, doc, tmp_path, sources=("vcf", "pgen", "svar"))

    # Reference is bgzipped + faidx'd and readable.
    assert case.ref_path.exists()
    assert (case.ref_path.parent / (case.ref_path.name + ".fai")).exists()
    with pysam.FastaFile(str(case.ref_path)) as fa:
        assert fa.fetch("chr1", 99, 100).upper() == "A"

    # One gvl dataset per requested source, each openable.
    assert set(case.gvl_path) == {"vcf", "pgen", "svar"}
    for src, path in case.gvl_path.items():
        ds = gvl.Dataset.open(path, case.ref_path)
        assert ds.n_regions >= 1, src

    # Consensus oracle FASTAs exist for every region/sample/hap.
    assert case.consensus_dir.is_dir()
    n_expected = case.regions.height * len(case.samples) * 2
    assert len(list(case.consensus_dir.glob("source_*.fa"))) == n_expected

    # Ground truth is exposed and shaped (n_variants, n_samples, ploidy).
    assert case.truth.genotypes.shape == (1, 2, 2)
    assert case.samples == ["s0", "s1"]


def test_session_document_shapes(tmp_path: Path):
    from tests._builders.case import session_document, session_reference

    spec = session_reference()
    ref = spec.write(tmp_path / "ref.fa.bgz")
    doc = session_document(spec)
    text = doc.render()

    # Standardized samples.
    header = next(l for l in text.splitlines() if l.startswith("#CHROM"))
    assert header.endswith("s0\ts1\ts2")

    # Standardized contigs only.
    assert "##contig=<ID=chr1," in text
    assert "##contig=<ID=chr2," in text
    assert "chr19" not in text and "chr20" not in text

    # The 10-bp deletion coupled to test_write_edge_cases is at chr1:1010696.
    assert any(
        line.startswith("chr1\t1010696\t") and "GAGACGGGGCC" in line
        for line in text.splitlines()
    )

    # Reference bases match the engineered REF alleles + N telomere.
    with pysam.FastaFile(str(ref)) as fa:
        assert fa.fetch("chr1", 0, 150).upper() == "N" * 150
        assert fa.fetch("chr1", 1010695, 1010706).upper() == "GAGACGGGGCC"
        assert fa.fetch("chr2", 14369, 14370).upper() == "N"
