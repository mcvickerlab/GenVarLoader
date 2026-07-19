"""PR 1b: svar2_read_window Rust FFI — shape/dtype smoke + byte-equivalence vs the
Phase-1 name-based SparseVar2._find_ranges path."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def test_svar2_read_window_shapes(svar2_multicontig_fixture) -> None:
    from genoray import SparseVar2
    from genvarloader.genvarloader import Svar2Store, svar2_read_window

    fx = svar2_multicontig_fixture
    sv = SparseVar2(str(fx.svar2_path))
    ploidy = int(sv.ploidy)
    store = Svar2Store(str(fx.svar2_path), sv.contigs, sv.n_samples, ploidy)

    # One contig window: chr1 regions [0,20) and [4,24); all physical samples 0..n.
    contig = "chr1"
    starts = np.array([0, 4], np.uint32)
    ends = np.array([20, 24], np.uint32)
    phys = np.arange(sv.n_samples, dtype=np.int64)
    n_reg, n_s = len(starts), len(phys)

    vk_snp, vk_indel, dense_snp, dense_indel, sample_cols = svar2_read_window(
        store, contig, starts, ends, phys
    )
    for a in (vk_snp, vk_indel, dense_snp, dense_indel, sample_cols):
        assert np.asarray(a).dtype == np.int64
    assert np.asarray(vk_snp).size == n_reg * n_s * ploidy * 2
    assert np.asarray(vk_indel).size == n_reg * n_s * ploidy * 2
    assert np.asarray(dense_snp).size == n_reg * 2
    assert np.asarray(dense_indel).size == n_reg * 2
    assert np.asarray(sample_cols).size == n_s


def test_svar2_read_window_matches_find_ranges(svar2_multicontig_fixture) -> None:
    """The rewired Rust read_window is byte-identical to the Phase-1 name-based
    SparseVar2._find_ranges path for the same window."""
    import genvarloader as gvl

    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert backend is not None

    # Reference (old) implementation: name-based _find_ranges, reshaped as Phase 1 did.
    def old_read_window(r_idx, s_idx):
        r_idx = np.asarray(r_idx, np.intp)
        s_idx = np.asarray(s_idx, np.intp)
        contig_idx, contig = backend._contig_of(r_idx)
        rb = backend._regions[r_idx, 1:3]
        starts = np.ascontiguousarray(rb[:, 0])
        ends = np.ascontiguousarray(rb[:, 1])
        names = [backend._sample_names[i] for i in s_idx]
        d = backend._sv._find_ranges(contig, starts, ends, samples=names)
        n_reg, n_s, P = len(r_idx), len(s_idx), backend.ploidy
        return {
            "orig_samples": np.ascontiguousarray(d["sample_cols"], np.int64),
            "vk_snp": np.asarray(d["vk_snp_range"], np.int64).reshape(n_reg, n_s, P, 2),
            "vk_indel": np.asarray(d["vk_indel_range"], np.int64).reshape(
                n_reg, n_s, P, 2
            ),
            "dense_snp": np.asarray(d["dense_snp_range"], np.int64).reshape(n_reg, 2),
            "dense_indel": np.asarray(d["dense_indel_range"], np.int64).reshape(
                n_reg, 2
            ),
        }

    for r_idx, s_idx in sds._plan():
        new = backend.read_window(r_idx, s_idx)  # rewired (Rust) path
        old = old_read_window(r_idx, s_idx)
        for k in ("orig_samples", "vk_snp", "vk_indel", "dense_snp", "dense_indel"):
            np.testing.assert_array_equal(
                np.asarray(new[k]), old[k], err_msg=f"mismatch in {k}"
            )


# Regression fixture for the `_phys_sample_idx` mapping seam specifically:
# `svar2_multicontig_fixture` (tests/dataset/conftest.py) uses VCF header
# samples "S0 S1 S2", whose native (VCF-column) order already equals
# lexicographically-sorted order -- so `_phys_sample_idx` there is the
# identity permutation `[0, 1, 2]` and a broken sorted->physical mapping
# would silently pass `test_svar2_read_window_matches_find_ranges` above.
# Here the VCF header declares samples out of lex order ("S2 S0 S1"; sorted
# order is S0, S1, S2), so `_phys_sample_idx` is a genuine, non-identity
# permutation ([1, 2, 0]: sorted-position 0 is "S0" at native column 1, etc.).
# Two variants overlapping the query window give each sample a DIFFERENT
# genotype (no two samples share a genotype at either site), so a swapped
# sample column changes which vk_snp/vk_indel ranges (and orig_samples) end
# up at that sorted position -- a wrong mapping cannot pass by accident.
_SVAR2_SAMPLE_ORDER_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_SVAR2_SAMPLE_ORDER_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS2\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|1\t0|0\t0|1
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\t1|0
"""


@dataclass(slots=True)
class Svar2SampleOrderFixture:
    """Matched inputs to drive `_Svar2Backend` directly over a store whose VCF
    header sample order is NOT lexicographically sorted, so
    `_phys_sample_idx` is a real (non-identity) permutation."""

    svar2_path: Path
    reference_path: Path
    contigs: list[str]
    bed: pl.DataFrame


@pytest.fixture(scope="module")
def svar2_sample_order_fixture(tmp_path_factory) -> Svar2SampleOrderFixture:
    from genoray import SparseVar2

    d = tmp_path_factory.mktemp("svar2_sample_order_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR2_SAMPLE_ORDER_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR2_SAMPLE_ORDER_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar2_path = tmp_path_factory.mktemp("svar2_sample_order_store") / "store.svar2"
    SparseVar2.from_vcf(svar2_path, bcf, reference=str(ref), overwrite=True)

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})

    return Svar2SampleOrderFixture(
        svar2_path=svar2_path,
        reference_path=ref,
        contigs=["chr1"],
        bed=bed,
    )


def test_svar2_read_window_matches_find_ranges_unsorted_header(
    svar2_sample_order_fixture,
) -> None:
    """Same byte-equivalence oracle as
    `test_svar2_read_window_matches_find_ranges`, but over a store whose VCF
    header order is deliberately unsorted, so the sorted-name -> physical
    store column translation `_phys_sample_idx` performs in `read_window` is
    a genuine permutation (not the identity that `svar2_multicontig_fixture`
    happens to have). This is the seam the oracle exists to protect: a wrong
    `_phys_sample_idx` would read the wrong VCF column for at least one
    sample, and since every sample has a distinct genotype at both variants
    here, that would show up as a mismatch below."""
    import genvarloader as gvl

    fx = svar2_sample_order_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert backend is not None

    # Sanity-check the fixture actually exercises a non-identity permutation
    # -- otherwise this test would be just as vacuous as the one it hardens.
    assert backend._sample_names == ["S0", "S1", "S2"]
    assert not np.array_equal(
        backend._phys_sample_idx, np.arange(backend.n_samples, dtype=np.int64)
    ), "fixture's native VCF header order must differ from sorted order"

    # Reference (old) implementation: name-based _find_ranges, reshaped as
    # Phase 1 did. Identical to the helper in
    # test_svar2_read_window_matches_find_ranges above.
    def old_read_window(r_idx, s_idx):
        r_idx = np.asarray(r_idx, np.intp)
        s_idx = np.asarray(s_idx, np.intp)
        contig_idx, contig = backend._contig_of(r_idx)
        rb = backend._regions[r_idx, 1:3]
        starts = np.ascontiguousarray(rb[:, 0])
        ends = np.ascontiguousarray(rb[:, 1])
        names = [backend._sample_names[i] for i in s_idx]
        d = backend._sv._find_ranges(contig, starts, ends, samples=names)
        n_reg, n_s, P = len(r_idx), len(s_idx), backend.ploidy
        return {
            "orig_samples": np.ascontiguousarray(d["sample_cols"], np.int64),
            "vk_snp": np.asarray(d["vk_snp_range"], np.int64).reshape(n_reg, n_s, P, 2),
            "vk_indel": np.asarray(d["vk_indel_range"], np.int64).reshape(
                n_reg, n_s, P, 2
            ),
            "dense_snp": np.asarray(d["dense_snp_range"], np.int64).reshape(n_reg, 2),
            "dense_indel": np.asarray(d["dense_indel_range"], np.int64).reshape(
                n_reg, 2
            ),
        }

    for r_idx, s_idx in sds._plan():
        new = backend.read_window(r_idx, s_idx)  # rewired (Rust) path
        old = old_read_window(r_idx, s_idx)
        for k in ("orig_samples", "vk_snp", "vk_indel", "dense_snp", "dense_indel"):
            np.testing.assert_array_equal(
                np.asarray(new[k]), old[k], err_msg=f"mismatch in {k}"
            )
