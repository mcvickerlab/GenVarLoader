from pathlib import Path

import genvarloader as gvl
import numpy as np
import pytest
from pytest import fixture


@fixture
def pgen():
    pgen_path = Path(__file__).parent / "data" / "pgen" / "sample.pgen"
    pgen = gvl.Variants.from_file(pgen_path)
    return pgen


@pytest.mark.skip
def test_snp(pgen: gvl.Variants):
    contig = "19"
    start = np.array([110])  # VCF is 1-based
    end = np.array([111])

    true_geno = np.array(
        [
            [[0], [0]],
            [[0], [0]],
            [[0], [1]],
        ],
        dtype=np.int8,
    )

    gvl_geno = pgen.read(contig, start, end)

    assert gvl_geno is not None
    np.testing.assert_equal(gvl_geno.genotypes, true_geno)


@pytest.mark.skip
def test_del(pgen: gvl.Variants):
    contig = "19"
    start = np.array([1010695])
    end = np.array([1010696])

    true_positions = np.array([1010694, 1010695, 1010695], np.int32)
    true_size_diffs = np.array([-6, -3, -10], np.int32)
    true_ref = np.frombuffer(b"CGAGACGGAGAGAGACGGGGCC", "S1")
    true_ref_offsets = np.array([0, 7, 11, 22], np.uint32)
    true_alt = np.frombuffer(b"CGG", "S1")
    true_alt_offsets = np.array([0, 1, 2, 3], np.uint32)
    true_geno = np.array(
        [
            [[0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [1, 0, 0]],
        ],
        dtype=np.int8,
    )
    true_offsets = np.array([0, 3], np.uint32)

    gvl_geno = pgen.read(contig, start, end)
    gvl_hap_geno, max_ends = pgen.read_for_haplotype_construction(contig, start, end)

    assert gvl_geno is not None
    np.testing.assert_equal(gvl_geno.positions, true_positions)
    np.testing.assert_equal(gvl_geno.size_diffs, true_size_diffs)
    np.testing.assert_equal(gvl_geno.ref.alleles, true_ref)
    np.testing.assert_equal(gvl_geno.ref.offsets, true_ref_offsets)
    np.testing.assert_equal(gvl_geno.alt.alleles, true_alt)
    np.testing.assert_equal(gvl_geno.alt.offsets, true_alt_offsets)
    np.testing.assert_equal(gvl_geno.genotypes, true_geno)
    np.testing.assert_equal(gvl_geno.offsets, true_offsets)

    assert gvl_hap_geno is not None
    np.testing.assert_equal(gvl_hap_geno.positions, true_positions)
    np.testing.assert_equal(gvl_hap_geno.size_diffs, true_size_diffs)
    np.testing.assert_equal(gvl_hap_geno.ref.alleles, true_ref)
    np.testing.assert_equal(gvl_hap_geno.ref.offsets, true_ref_offsets)
    np.testing.assert_equal(gvl_hap_geno.alt.alleles, true_alt)
    np.testing.assert_equal(gvl_hap_geno.alt.offsets, true_alt_offsets)
    np.testing.assert_equal(gvl_hap_geno.genotypes, true_geno)
    np.testing.assert_equal(gvl_hap_geno.offsets, true_offsets)
    np.testing.assert_equal(max_ends, np.array([1010696 + 10], np.int64))


@pytest.mark.skip
def test_ins(pgen: gvl.Variants):
    contig = "19"
    start = np.array([1110695])
    end = np.array([1110696])

    true_positions = np.array([1110695, 1110695], np.int32)
    true_size_diffs = np.array([75, 0], np.int32)
    true_ref = np.frombuffer(b"AA", "S1")
    true_ref_offsets = np.array([0, 1, 2], np.uint32)
    true_alt = np.frombuffer(
        b"AGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATAGATG",
        "S1",
    )
    true_alt_offsets = np.array([0, 76, 77], np.uint32)
    true_geno = np.array(
        [
            [[0, 0], [1, 0]],
            [[1, 0], [1, 0]],
            [[0, 0], [0, 1]],
        ],
        dtype=np.int8,
    )
    true_offsets = np.array([0, 2], np.uint32)

    gvl_geno = pgen.read(contig, start, end)
    gvl_hap_geno, max_ends = pgen.read_for_haplotype_construction(contig, start, end)

    assert gvl_geno is not None
    np.testing.assert_equal(gvl_geno.positions, true_positions)
    np.testing.assert_equal(gvl_geno.size_diffs, true_size_diffs)
    np.testing.assert_equal(gvl_geno.ref.alleles, true_ref)
    np.testing.assert_equal(gvl_geno.ref.offsets, true_ref_offsets)
    np.testing.assert_equal(gvl_geno.alt.alleles, true_alt)
    np.testing.assert_equal(gvl_geno.alt.offsets, true_alt_offsets)
    np.testing.assert_equal(gvl_geno.genotypes, true_geno)
    np.testing.assert_equal(gvl_geno.offsets, true_offsets)

    assert gvl_hap_geno is not None
    np.testing.assert_equal(gvl_hap_geno.positions, true_positions)
    np.testing.assert_equal(gvl_hap_geno.size_diffs, true_size_diffs)
    np.testing.assert_equal(gvl_hap_geno.ref.alleles, true_ref)
    np.testing.assert_equal(gvl_hap_geno.ref.offsets, true_ref_offsets)
    np.testing.assert_equal(gvl_hap_geno.alt.alleles, true_alt)
    np.testing.assert_equal(gvl_hap_geno.alt.offsets, true_alt_offsets)
    np.testing.assert_equal(gvl_hap_geno.genotypes, true_geno)
    np.testing.assert_equal(gvl_hap_geno.offsets, true_offsets)
    np.testing.assert_equal(max_ends, end)


@pytest.mark.skip
def test_split_snp(pgen: gvl.Variants):
    contig = "20"
    start = np.array([1110695])  # VCF is 1-based
    end = np.array([1110696])

    true_positions = np.array([1110695, 1110695], np.int32)
    true_size_diffs = np.array([0, 0], np.int32)
    true_ref = np.array([b"G", b"G"])
    true_alt = np.array([b"A", b"T"])
    true_geno = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[-9, 1], [-9, 1]],
        ],
        dtype=np.int8,
    )
    true_offsets = np.array([0, 2], np.uint32)

    gvl_geno = pgen.read(contig, start, end)
    gvl_hap_geno, max_ends = pgen.read_for_haplotype_construction(contig, start, end)

    assert gvl_geno is not None
    np.testing.assert_equal(gvl_geno.positions, true_positions)
    np.testing.assert_equal(gvl_geno.size_diffs, true_size_diffs)
    np.testing.assert_equal(gvl_geno.ref.alleles, true_ref)
    np.testing.assert_equal(gvl_geno.alt.alleles, true_alt)
    np.testing.assert_equal(gvl_geno.genotypes, true_geno)
    np.testing.assert_equal(gvl_geno.offsets, true_offsets)


if __name__ == "__main__":
    # test_del(pgen())
    test_split_snp(pgen())
