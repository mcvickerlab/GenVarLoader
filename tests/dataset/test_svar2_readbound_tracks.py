"""Parity test for the read-bound SVAR2 track re-alignment kernel (Task 5).

Oracle: ``SparseVar2Source.realign_tracks`` (genoray ``overlap_batch``, eager
dense-union path). Under test: ``build_readbound_tracks`` (genoray
``find_ranges`` + one Rust FFI call via
``genoray_core::query::gather_haps_readbound`` -> ``svar2::split_to_flat`` ->
the SAME validated ``shift_and_realign_tracks_from_svar2`` kernel the oracle
uses).

Both paths must be byte-identical: same offsets, same (NaN-equal) data.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

# 40 bp reference (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G, low-carrier,
# routes to var_key), INS@6 (C>CAT), SNP@9 (G>C, carried by 3 haps -> dense/snp
# per the cost model used in test_svar2_readbound_haps.py), DEL@11 (GTA>G,
# ilen -2). Exercises both var_key and dense/snp + dense/indel channels.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
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
def svar2_store(tmp_path_factory) -> Path:
    from genoray import _core

    d = tmp_path_factory.mktemp("svar2_readbound_tracks")
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


def _synthetic_track_inputs(regions, seed=0):
    """Per-region flat f32 tracks + (R+1) offsets, plus a known-valid
    (params, strategy_id) combo mirroring tests/test_svar2_realign_tracks.py."""
    rng = np.random.default_rng(seed)
    lengths = [e - s for s, e in regions]
    toff = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tracks = rng.random(int(toff[-1])).astype(np.float32)
    strategy_id = 0  # irrelevant for insertion-fill in this test
    params = np.zeros(1, np.float64)
    base_seed = 0
    return tracks, toff, params, strategy_id, base_seed


@pytest.mark.parametrize(
    "regions",
    [
        [(0, 40)],  # whole contig: SNP + dense-SNP + INS + DEL all in play
        [(0, 5), (5, 15), (15, 40)],  # split around the variant boundaries
        [(0, 40), (2, 2), (20, 25)],  # empty region + a variant-free window
    ],
)
def test_readbound_tracks_match_union_oracle(svar2_store, regions):
    import genoray

    from genvarloader._dataset._svar2_source import SparseVar2Source
    from genvarloader._dataset._svar2_store_py import build_readbound_tracks

    contig = "chr1"
    sv = genoray.SparseVar2(str(svar2_store))
    S, P = sv.n_samples, sv.ploidy
    assert (S, P) == (2, 2)

    # Self-verify the fixture routes the >=2-carrier SNP@9 to dense/snp (mixed
    # with the dense indels) so this parity test genuinely exercises the dense
    # path for tracks. Without this, a future cost-model change could silently
    # demote the SNP to var_key and the test would still pass while covering less.
    d = sv.find_ranges(contig, [0], [40], samples=None)
    snp_win = int(
        np.asarray(d["dense_snp_range"])[0, 1] - np.asarray(d["dense_snp_range"])[0, 0]
    )
    indel_win = int(
        np.asarray(d["dense_indel_range"])[0, 1]
        - np.asarray(d["dense_indel_range"])[0, 0]
    )
    assert snp_win >= 1 and indel_win >= 1, (
        f"fixture must populate both dense channels; got dense_snp_range="
        f"{np.asarray(d['dense_snp_range']).tolist()}, dense_indel_range="
        f"{np.asarray(d['dense_indel_range']).tolist()}"
    )

    tracks, toff, params, strat, seed = _synthetic_track_inputs(regions)

    union = SparseVar2Source(sv).realign_tracks(
        contig, regions, tracks, toff, params, strat, seed, shifts=None, parallel=False
    )
    rb = build_readbound_tracks(
        sv,
        contig,
        regions,
        tracks,
        toff,
        params,
        strat,
        seed,
        shifts=None,
        parallel=False,
    )

    union_offsets = np.asarray(union.offsets)
    rb_offsets = np.asarray(rb.offsets)
    assert np.array_equal(union_offsets, rb_offsets), (
        f"offsets mismatch: union={union_offsets.tolist()} rb={rb_offsets.tolist()}"
    )

    union_data = np.asarray(union.data)
    rb_data = np.asarray(rb.data)
    if not np.allclose(union_data, rb_data, equal_nan=True):
        R = len(regions)
        n_q = R * S
        for h in range(n_q * P):
            s0, e0 = int(union_offsets[h]), int(union_offsets[h + 1])
            s1, e1 = int(rb_offsets[h]), int(rb_offsets[h + 1])
            a = union_data[s0:e0]
            b = rb_data[s1:e1]
            if not np.allclose(a, b, equal_nan=True):
                pytest.fail(f"data mismatch at hap {h}: union={a!r} rb={b!r}")
        pytest.fail("data mismatch but no single hap slice differed (offset bug?)")


def test_readbound_tracks_match_union_oracle_with_shifts(svar2_store):
    """Non-trivial per-hap jitter shifts must also match byte-for-byte."""
    import genoray

    from genvarloader._dataset._svar2_source import SparseVar2Source
    from genvarloader._dataset._svar2_store_py import build_readbound_tracks

    contig = "chr1"
    regions = [(0, 40), (5, 20)]
    sv = genoray.SparseVar2(str(svar2_store))
    S, P = sv.n_samples, sv.ploidy
    n_q = len(regions) * S

    tracks, toff, params, strat, seed = _synthetic_track_inputs(regions)

    rng = np.random.default_rng(1)
    shifts = rng.integers(-2, 3, size=(n_q, P), dtype=np.int32)

    union = SparseVar2Source(sv).realign_tracks(
        contig,
        regions,
        tracks,
        toff,
        params,
        strat,
        seed,
        shifts=shifts,
        parallel=False,
    )
    rb = build_readbound_tracks(
        sv,
        contig,
        regions,
        tracks,
        toff,
        params,
        strat,
        seed,
        shifts=shifts,
        parallel=False,
    )

    assert np.array_equal(np.asarray(union.offsets), np.asarray(rb.offsets))
    assert np.allclose(np.asarray(union.data), np.asarray(rb.data), equal_nan=True)
