"""Parity test for the read-bound SVAR2 variants decode kernel (Task 6).

Oracle: ``SparseVar2.decode`` (genoray's own record-``Ragged`` decode, no
overlap/clip filter — the gather already restricts to overlapping variants).
Under test: ``build_readbound_variants`` (genoray ``find_ranges`` + one Rust FFI
call via ``genoray_core::query::gather_haps_readbound`` -> per-hap ``merge_hap`` +
``decode_alt``, mirroring genoray's ``decode_hap``).

Both paths decode the SAME full cohort (``samples=None``), so the flat per-hap
(pos, ilen, alt) arrays and the shared variant-axis offsets must be identical.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

# 40 bp reference (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G), INS@6 (C>CAT),
# DEL@11 (GTA>G, ilen -2). Genotypes exercise both samples and both ploids.
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


@pytest.fixture(scope="module")
def svar2_store(tmp_path_factory) -> Path:
    from genoray import _core

    d = tmp_path_factory.mktemp("svar2_readbound_variants")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store"
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


def _assert_variants_match(oracle, rb) -> None:
    """Compare a genoray ``decode()`` record Ragged (pos/ilen/allele) against a
    ``RaggedVariants`` (alt/start/ilen) built by the read-bound kernel."""
    oracle_off = np.asarray(oracle.offsets)
    rb_off = np.asarray(rb.offsets)
    assert np.array_equal(oracle_off, rb_off), (
        f"variant-axis offsets mismatch: oracle={oracle_off.tolist()} "
        f"rb={rb_off.tolist()}"
    )

    pos_match = oracle["pos"].to_ak().to_list() == rb["start"].to_ak().to_list()
    ilen_match = oracle["ilen"].to_ak().to_list() == rb["ilen"].to_ak().to_list()
    allele_match = oracle["allele"].to_ak().to_list() == rb["alt"].to_ak().to_list()

    if pos_match and ilen_match and allele_match:
        return

    # Locate the first mismatching (hap, variant) for debuggability.
    n_hap = len(oracle_off) - 1
    o_pos, r_pos = oracle["pos"].to_ak().to_list(), rb["start"].to_ak().to_list()
    o_ilen, r_ilen = oracle["ilen"].to_ak().to_list(), rb["ilen"].to_ak().to_list()
    o_alt, r_alt = oracle["allele"].to_ak().to_list(), rb["alt"].to_ak().to_list()
    for h in range(n_hap):
        if (o_pos[h], o_ilen[h], o_alt[h]) != (r_pos[h], r_ilen[h], r_alt[h]):
            pytest.fail(
                f"mismatch at hap {h}: "
                f"oracle=(pos={o_pos[h]}, ilen={o_ilen[h]}, alt={o_alt[h]}) "
                f"rb=(pos={r_pos[h]}, ilen={r_ilen[h]}, alt={r_alt[h]})"
            )
    pytest.fail("mismatch but no single hap differed (offset/field bug?)")


@pytest.mark.parametrize(
    "regions",
    [
        [(0, 40)],  # whole contig: SNP + INS + DEL all in play
        [(0, 5), (5, 15), (15, 40)],  # split around the SNP/INS/DEL boundaries
        [(0, 40), (2, 2), (20, 25)],  # empty region + a variant-free window
    ],
)
def test_readbound_variants_match_decode_oracle(svar2_store, regions):
    import genoray

    from genvarloader._dataset._svar2_store_py import build_readbound_variants

    contig = "chr1"

    sv = genoray.SparseVar2(str(svar2_store))
    S, P = sv.n_samples, sv.ploidy
    assert (S, P) == (2, 2)

    oracle = sv.decode(contig, regions)
    rb = build_readbound_variants(sv, contig, regions)

    _assert_variants_match(oracle, rb)


# Fixture whose cost model routes a SNP into the DENSE/snp table (not var_key),
# so split_to_flat's snp-block concatenation + snp-before-indel window ordering
# are exercised with real data (see test_svar2_readbound_haps.py's identical
# fixture recipe for the routing rationale).
_VCF_DENSE_SNP = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


@pytest.fixture(scope="module")
def svar2_store_dense_snp(tmp_path_factory) -> Path:
    from genoray import _core

    d = tmp_path_factory.mktemp("svar2_readbound_variants_dense_snp")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF_DENSE_SNP)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store"
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


def test_readbound_variants_dense_snp_match_decode_oracle(svar2_store_dense_snp):
    """A SNP routed into dense/snp must decode identically to the oracle.

    Also sanity-checks (before asserting parity) that the SNP actually landed in
    dense/snp — i.e. ``find_ranges``' ``dense_snp_range`` is a non-empty window
    for a region covering it — so this test genuinely exercises split_to_flat's
    snp-block path rather than silently falling back to the var_key channel.
    """
    import genoray

    from genvarloader._dataset._svar2_store_py import build_readbound_variants

    contig = "chr1"

    sv = genoray.SparseVar2(str(svar2_store_dense_snp))
    assert (sv.n_samples, sv.ploidy) == (2, 2)

    # Routing sanity: the SNP@10 (0-based 9) must be in the dense/snp table, so a
    # region spanning it has a non-empty dense_snp window.
    d = sv.find_ranges(contig, [0], [40], samples=None)
    dense_snp_range = np.asarray(d["dense_snp_range"])  # (R, 2)
    dense_indel_range = np.asarray(d["dense_indel_range"])  # (R, 2)
    snp_win = int(dense_snp_range[0, 1] - dense_snp_range[0, 0])
    indel_win = int(dense_indel_range[0, 1] - dense_indel_range[0, 0])
    assert snp_win >= 1, (
        f"expected the SNP to route to dense/snp, but dense_snp_range is empty "
        f"({dense_snp_range.tolist()}); cost model did not dense-encode it"
    )
    # Non-triviality: dense/indel is also populated (INS@7 + DEL@12), so the
    # combined window mixes snp and indel entries (concatenation under test).
    assert indel_win >= 1, dense_indel_range.tolist()

    regions = [(0, 40), (0, 12), (9, 15), (8, 11)]
    oracle = sv.decode(contig, regions)
    rb = build_readbound_variants(sv, contig, regions)

    _assert_variants_match(oracle, rb)
