"""Parity test for the read-bound SVAR2 per-hap diffs kernel (Task 7a).

Oracle: the diffs implied by the read-bound HAPLOTYPE reconstruction
(``build_readbound_haps``) — per (region, hap), ``len(haplotype) - (region_end -
region_start)`` is exactly the ilen diff the reconstruct kernel computed internally
via ``svar2::hap_diffs_svar2`` before sizing/writing the output. Under test:
``build_readbound_diffs`` (same gather, but stops after ``hap_diffs_svar2`` and
returns just the diffs — no reconstruct pass).

This proves the newly-exposed diffs FFI matches what the reconstruct kernel uses
internally, without needing a second independent oracle.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

# Same fixtures as tests/dataset/test_svar2_readbound_haps.py: 40 bp reference
# (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G), INS@6 (C>CAT), DEL@11
# (GTA>G, ilen -2). Genotypes exercise both samples and both ploids.
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

    d = tmp_path_factory.mktemp("svar2_readbound_diffs")
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


# Fixture whose cost model routes a SNP into the DENSE/snp table (not var_key) —
# see test_svar2_readbound_haps.py for the routing rationale.
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

    d = tmp_path_factory.mktemp("svar2_readbound_diffs_dense_snp")
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


def _implied_diffs(regions, ref_arr, ref_offsets, sv, contig) -> np.ndarray:
    """Diffs implied by the read-bound haplotype reconstruction: per (region,
    hap), ``len(haplotype) - (region_end - region_start)``.

    Query order matches ``build_readbound_haps``/``build_readbound_diffs``:
    region-major, sample-minor (``q = r*S + s``), hap-minor within a query.
    """
    from genvarloader._dataset._svar2_store_py import build_readbound_haps

    S, P = sv.n_samples, sv.ploidy
    R = len(regions)

    rb = build_readbound_haps(
        sv,
        contig,
        regions,
        ref_arr,
        ref_offsets,
        pad_char=ord("N"),
        shifts=None,
        output_length=-1,
        parallel=False,
    )
    offsets = np.asarray(rb.offsets)  # (R*S*P + 1,)

    lengths = np.diff(offsets)  # (R*S*P,)
    ref_lens = np.repeat(
        np.asarray([e - s for s, e in regions], np.int64), S * P
    )  # (R*S*P,) region-major, sample-minor, hap-minor
    diffs = (lengths - ref_lens).astype(np.int32)
    return diffs.reshape(R * S, P)


@pytest.mark.parametrize(
    "regions",
    [
        [(0, 40)],  # whole contig: SNP + INS + DEL all in play
        [(0, 5), (5, 15), (15, 40)],  # split around the SNP/INS/DEL boundaries
        [(0, 40), (2, 2), (20, 25)],  # empty region + a variant-free window
    ],
)
def test_readbound_diffs_matches_implied_haps(svar2_store, regions):
    import genoray

    from genvarloader._dataset._svar2_store_py import build_readbound_diffs

    contig = "chr1"
    ref_bytes = _REF.encode()
    ref_arr = np.frombuffer(ref_bytes, np.uint8)
    ref_offsets = np.array([0, len(ref_bytes)], np.int64)

    sv = genoray.SparseVar2(str(svar2_store))
    assert (sv.n_samples, sv.ploidy) == (2, 2)

    implied = _implied_diffs(regions, ref_arr, ref_offsets, sv, contig)
    diffs = np.asarray(build_readbound_diffs(sv, contig, regions))

    assert diffs.shape == implied.shape
    assert np.array_equal(diffs, implied), (
        f"diffs mismatch: implied={implied.tolist()} diffs={diffs.tolist()}"
    )


def test_readbound_diffs_dense_snp_matches_implied_haps(svar2_store_dense_snp):
    """A SNP routed into dense/snp must diff-clip identically to what the
    reconstruct kernel implies."""
    import genoray

    from genvarloader._dataset._svar2_store_py import build_readbound_diffs

    contig = "chr1"
    ref_bytes = _REF.encode()
    ref_arr = np.frombuffer(ref_bytes, np.uint8)
    ref_offsets = np.array([0, len(ref_bytes)], np.int64)

    sv = genoray.SparseVar2(str(svar2_store_dense_snp))
    assert (sv.n_samples, sv.ploidy) == (2, 2)

    # Routing sanity: the SNP@10 (0-based 9) must be in the dense/snp table.
    d = sv.find_ranges(contig, [0], [40], samples=None)
    dense_snp_range = np.asarray(d["dense_snp_range"])  # (R, 2)
    snp_win = int(dense_snp_range[0, 1] - dense_snp_range[0, 0])
    assert snp_win >= 1, (
        f"expected the SNP to route to dense/snp, but dense_snp_range is empty "
        f"({dense_snp_range.tolist()}); cost model did not dense-encode it"
    )

    regions = [(0, 40), (0, 12), (9, 15), (8, 11)]
    implied = _implied_diffs(regions, ref_arr, ref_offsets, sv, contig)
    diffs = np.asarray(build_readbound_diffs(sv, contig, regions))

    assert diffs.shape == implied.shape
    assert np.array_equal(diffs, implied), (
        f"diffs mismatch: implied={implied.tolist()} diffs={diffs.tolist()}"
    )
