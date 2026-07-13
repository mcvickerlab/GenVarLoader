"""Parity test for the read-bound SVAR2 haplotype kernel (Task 4).

Oracle: ``SparseVar2Source.reconstruct`` (genoray ``_overlap_batch``, eager dense-union
path). Under test: ``build_readbound_haps`` (genoray ``_find_ranges`` + one Rust FFI call
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


def test_readbound_haps_noncontiguous_ref_raises(svar2_store):
    """A non-C-contiguous ``ref_`` view must surface as ``ValueError``, not a Rust
    panic.

    ``build_readbound_haps`` (the Python oracle wrapper) defensively
    ``np.ascontiguousarray``s ``ref_`` before handing it to the FFI, so it can't be
    used to inject a strided array here -- this calls
    ``reconstruct_haplotypes_from_svar2_readbound`` directly, replaying the same
    ``_find_ranges`` marshalling ``build_readbound_haps`` does internally (see
    ``genvarloader/_dataset/_svar2_store_py.py::build_readbound_haps``), but with a
    genuinely non-contiguous ``ref_``.
    """
    import genoray

    from genvarloader.genvarloader import (
        Svar2Store,
        reconstruct_haplotypes_from_svar2_readbound,
    )

    contig = "chr1"
    regions = [(0, 40)]
    ref_bytes = _REF.encode()
    ref_offsets = np.array([0, len(ref_bytes)], np.int64)

    # A strided (non-contiguous) view carrying the same bytes as `_REF`: double up
    # each byte, then stride over every other one to recover the original values.
    doubled = np.repeat(np.frombuffer(ref_bytes, np.uint8), 2)
    ref_strided = doubled[::2]
    assert ref_strided.flags["C_CONTIGUOUS"] is False
    assert bytes(ref_strided) == ref_bytes

    sv = genoray.SparseVar2(str(svar2_store))
    S, P = sv.n_samples, sv.ploidy

    d = sv._find_ranges(
        contig, [s for s, _ in regions], [e for _, e in regions], samples=None
    )
    region_starts_r = np.asarray(d["region_starts"], np.int64)
    sample_cols = np.asarray(d["sample_cols"], np.int64)
    vk_snp_range = np.ascontiguousarray(d["vk_snp_range"], np.int64)
    vk_indel_range = np.ascontiguousarray(d["vk_indel_range"], np.int64)
    dense_snp_range_r = np.asarray(d["dense_snp_range"], np.int64)
    dense_indel_range_r = np.asarray(d["dense_indel_range"], np.int64)

    R = len(regions)
    n_q = R * S
    region_starts = np.repeat(region_starts_r, S).astype(np.uint32)
    orig_samples = np.tile(sample_cols, R)
    dense_snp_range = np.ascontiguousarray(
        np.repeat(dense_snp_range_r, S, axis=0), np.int64
    )
    dense_indel_range = np.ascontiguousarray(
        np.repeat(dense_indel_range_r, S, axis=0), np.int64
    )
    reg_arr = np.asarray(regions, np.int32).reshape(R, 2)
    region_bounds = np.ascontiguousarray(np.repeat(reg_arr, S, axis=0), np.int32)
    shifts_a = np.zeros((n_q, P), dtype=np.int32)

    store = Svar2Store(str(sv.path), sv.contigs, sv.n_samples, sv.ploidy)

    with pytest.raises(ValueError):
        reconstruct_haplotypes_from_svar2_readbound(
            store,
            contig,
            region_starts,
            orig_samples,
            vk_snp_range,
            vk_indel_range,
            dense_snp_range,
            dense_indel_range,
            region_bounds,
            shifts_a,
            ref_strided,
            ref_offsets,
            np.uint8(ord("N")),
            np.int64(-1),
            False,
        )


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


# Fixture whose cost model routes a SNP into the DENSE/snp table (not var_key),
# so split_to_flat's snp-block concatenation + snp-before-indel window ordering
# are exercised with real data. genoray dense-encodes a SNP when
# dense_bits (POS_BITS + 2 + n_samples*ploidy) < var_key_bits (34*x_calls); with
# 2 samples x ploidy 2 (np=4) that's 38 < 34*x, i.e. any SNP carried by >=2
# haplotypes goes dense. The SNP@10 below is carried by 3 haps -> dense/snp.
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

    d = tmp_path_factory.mktemp("svar2_readbound_dense_snp")
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


def test_readbound_dense_snp_matches_union_oracle(svar2_store_dense_snp):
    """A SNP routed into dense/snp must reconstruct byte-identically.

    Also sanity-checks (before asserting parity) that the SNP actually landed in
    dense/snp — i.e. ``_find_ranges``' ``dense_snp_range`` is a non-empty window
    for a region covering it — so this test genuinely exercises split_to_flat's
    snp-block path rather than silently falling back to the var_key channel.
    """
    import genoray

    from genvarloader._dataset._svar2_source import SparseVar2Source
    from genvarloader._dataset._svar2_store_py import build_readbound_haps

    contig = "chr1"
    ref_bytes = _REF.encode()
    ref_arr = np.frombuffer(ref_bytes, np.uint8)
    ref_offsets = np.array([0, len(ref_bytes)], np.int64)

    sv = genoray.SparseVar2(str(svar2_store_dense_snp))
    assert (sv.n_samples, sv.ploidy) == (2, 2)

    # Routing sanity: the SNP@10 (0-based 9) must be in the dense/snp table, so a
    # region spanning it has a non-empty dense_snp window.
    d = sv._find_ranges(contig, [0], [40], samples=None)
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

    assert np.array_equal(np.asarray(oracle.offsets), np.asarray(rb.offsets))
    assert np.array_equal(
        np.asarray(oracle.data).view("u1"), np.asarray(rb.data).view("u1")
    )


def _svar2_haps_dataset(tmp_path: Path, svar2_store: Path):
    """Build a full gvl Dataset over the ``svar2_store`` fixture and return its
    haplotypes view (Svar2Haps-backed).

    Lifted/adapted from ``test_svar2_dataset.py::_open_pair`` -- this file has no
    existing fixture that yields a live gvl.Dataset (only bare genoray stores),
    so this helper builds the minimal one needed to exercise Svar2Haps through
    the public Dataset API.
    """
    import polars as pl
    from genoray import SparseVar2

    import genvarloader as gvl

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})
    ref = svar2_store.parent / "ref.fa"
    d = tmp_path / "ds.gvl"
    gvl.write(d, bed, variants=SparseVar2(svar2_store), samples=None, overwrite=True)
    return gvl.Dataset.open(d, reference=ref).with_seqs("haplotypes")


def test_deterministic_haps_read_skips_pre_reconstruct_diffs(
    tmp_path: Path, svar2_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """A deterministic (shifts=0) haplotypes read must NOT call the separate
    hap_diffs readbound kernel -- reconstruct sizes itself internally. Guards the
    double-gather regression."""
    import genvarloader._dataset._svar2_haps as m

    calls = {"diffs": 0}
    real = m.hap_diffs_from_svar2_readbound

    def counting(*a, **k):
        calls["diffs"] += 1
        return real(*a, **k)

    monkeypatch.setattr(m, "hap_diffs_from_svar2_readbound", counting)

    ds2 = _svar2_haps_dataset(tmp_path, svar2_store)
    ds2[:, :]
    assert calls["diffs"] == 0
