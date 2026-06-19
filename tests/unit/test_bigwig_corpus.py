# tests/unit/test_bigwig_corpus.py
import numpy as np
import pyBigWig

from tests._bigwig_corpus import make_regions, make_synthetic_bigwigs


def test_make_synthetic_bigwigs_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    paths_a = make_synthetic_bigwigs(a, n_samples=2, seed=7)
    paths_b = make_synthetic_bigwigs(b, n_samples=2, seed=7)
    assert [p.name for p in paths_a] == ["sample_0.bw", "sample_1.bw"]
    # byte-identical given same seed
    assert paths_a[0].read_bytes() == paths_b[0].read_bytes()
    # has intervals on both contigs
    with pyBigWig.open(str(paths_a[0])) as bw:
        assert "chr21" in bw.chroms()
        assert len(bw.intervals("chr21")) > 0


def test_make_regions_grouped_in_contig_order(tmp_path):
    regions = make_regions({"chr21": 200_000, "chr22": 150_000}, n_per_contig=4, width=1000, seed=1)
    assert regions.columns == ["chrom", "chromStart", "chromEnd"]
    # contig-grouped in dict order (chr21 block then chr22 block)
    chroms = regions["chrom"].to_list()
    assert chroms == ["chr21"] * 4 + ["chr22"] * 4
    assert (regions["chromEnd"] - regions["chromStart"] == 1000).all()
