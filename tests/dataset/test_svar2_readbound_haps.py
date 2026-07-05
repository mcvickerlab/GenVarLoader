"""Parity test for the read-bound SVAR2 haplotype kernel (Task 4).

Oracle: ``SparseVar2Source.reconstruct`` (genoray ``overlap_batch``, eager dense-union
path). Under test: ``build_readbound_haps`` (genoray ``find_ranges`` + one Rust FFI call
via ``genoray_core::query::gather_haps_readbound`` -> ``svar2::split_to_flat`` ->
the SAME validated ``reconstruct_haplotypes_from_svar2`` kernel the oracle uses).

Both paths must be byte-identical: same offsets, same data.
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

    d = tmp_path_factory.mktemp("svar2_readbound")
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


@pytest.mark.parametrize(
    "regions",
    [
        [(0, 40)],  # whole contig: SNP + INS + DEL all in play
        [(0, 5), (5, 15), (15, 40)],  # split around the SNP/INS/DEL boundaries
        [(0, 40), (2, 2), (20, 25)],  # empty region + a variant-free window
    ],
)
def test_readbound_matches_union_oracle(svar2_store, regions):
    import genoray

    from genvarloader._dataset._svar2_source import SparseVar2Source
    from genvarloader._dataset._svar2_store_py import build_readbound_haps

    contig = "chr1"
    ref_bytes = _REF.encode()
    ref_arr = np.frombuffer(ref_bytes, np.uint8)
    ref_offsets = np.array([0, len(ref_bytes)], np.int64)

    sv = genoray.SparseVar2(str(svar2_store))
    S, P = sv.n_samples, sv.ploidy
    assert (S, P) == (2, 2)

    oracle = SparseVar2Source(sv).reconstruct(
        contig,
        regions,
        ref_arr,
        ref_offsets,
        pad_char=ord("N"),
        shifts=None,
        output_length=-1,
        parallel=False,
    )
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

    oracle_offsets = np.asarray(oracle.offsets)
    rb_offsets = np.asarray(rb.offsets)
    assert np.array_equal(oracle_offsets, rb_offsets), (
        f"offsets mismatch: oracle={oracle_offsets.tolist()} rb={rb_offsets.tolist()}"
    )

    oracle_data = np.asarray(oracle.data).view("u1")
    rb_data = np.asarray(rb.data).view("u1")
    if not np.array_equal(oracle_data, rb_data):
        # Locate the first mismatching (query, hap, byte) for debuggability.
        R = len(regions)
        H = P
        n_q = R * S
        for h in range(n_q * H):
            s0, e0 = int(oracle_offsets[h]), int(oracle_offsets[h + 1])
            s1, e1 = int(rb_offsets[h]), int(rb_offsets[h + 1])
            a = oracle_data[s0:e0]
            b = rb_data[s1:e1]
            if not np.array_equal(a, b):
                pytest.fail(
                    f"data mismatch at hap {h}: oracle={a.tobytes()!r} rb={b.tobytes()!r}"
                )
        pytest.fail("data mismatch but no single hap slice differed (offset bug?)")


def test_readbound_matches_union_oracle_with_shifts(svar2_store):
    """Non-trivial per-hap jitter shifts must also match byte-for-byte."""
    import genoray

    from genvarloader._dataset._svar2_source import SparseVar2Source
    from genvarloader._dataset._svar2_store_py import build_readbound_haps

    contig = "chr1"
    regions = [(0, 40), (5, 20)]
    ref_bytes = _REF.encode()
    ref_arr = np.frombuffer(ref_bytes, np.uint8)
    ref_offsets = np.array([0, len(ref_bytes)], np.int64)

    sv = genoray.SparseVar2(str(svar2_store))
    S, P = sv.n_samples, sv.ploidy
    n_q = len(regions) * S
    rng = np.random.default_rng(0)
    shifts = rng.integers(-2, 3, size=(n_q, P), dtype=np.int32)

    oracle = SparseVar2Source(sv).reconstruct(
        contig,
        regions,
        ref_arr,
        ref_offsets,
        pad_char=ord("N"),
        shifts=shifts,
        output_length=-1,
        parallel=False,
    )
    rb = build_readbound_haps(
        sv,
        contig,
        regions,
        ref_arr,
        ref_offsets,
        pad_char=ord("N"),
        shifts=shifts,
        output_length=-1,
        parallel=False,
    )

    assert np.array_equal(np.asarray(oracle.offsets), np.asarray(rb.offsets))
    assert np.array_equal(
        np.asarray(oracle.data).view("u1"), np.asarray(rb.data).view("u1")
    )
