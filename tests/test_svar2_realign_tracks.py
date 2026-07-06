"""End-to-end validation of the SVAR2 track-realign adapter path.

Builds a DEL-only SVAR2 store, realigns a reference track through gvl's SVAR2
path (SparseVar2Source.realign_tracks, the Rust two-source kernel), and compares
per-(region, sample, ploid) against gvl's INDEPENDENT pure-Python SVAR1 track
realign (shift_and_realign_track_sparse) fed genoray's materialized decode
records. Agreement proves the SVAR2 Rust track kernel matches the trusted SVAR1
realign semantics — including the DEL anchor.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

# 40 bp reference (chr1). Two pure DELs chosen to match the reference exactly:
#   POS 4  GTA>G  -> 0-based pos 3, ilen -2  (ref[3:6] == "GTA")
#   POS 10 GGG>G  -> 0-based pos 9, ilen -2  (ref[9:12] == "GGG")
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t4\t.\tGTA\tG\t.\t.\t.\tGT\t1|0\t1|1
chr1\t10\t.\tGGG\tG\t.\t.\t.\tGT\t0|1\t1|0
"""


@pytest.fixture(scope="module")
def svar2_del_store(tmp_path_factory) -> Path:
    from genoray import _core

    d = tmp_path_factory.mktemp("svar2_del")
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


def test_svar2_realign_tracks_matches_svar1_oracle(svar2_del_store):
    import genoray
    from genvarloader._dataset._svar2_source import SparseVar2Source
    from genvarloader._dataset._tracks import shift_and_realign_track_sparse

    contig = "chr1"
    q_start, q_end = 0, 40
    region_len = q_end - q_start
    regions = [(q_start, q_end)]

    sv = genoray.SparseVar2(str(svar2_del_store))
    S, P = sv.n_samples, sv.ploidy
    assert (S, P) == (2, 2)

    # A per-region reference track (f32). Random-but-fixed so a positional bug
    # can't hide behind a monotonic ramp.
    rng = np.random.default_rng(0)
    track = rng.random(region_len).astype(np.float32)

    strategy_id = 0  # irrelevant for DEL-only (insertion-fill unused)
    params = np.zeros(1, np.float64)
    base_seed = 0

    # --- SVAR2 path under test: one region, expanded internally to R*S*P haps ---
    src = SparseVar2Source(sv)
    out_rag = src.realign_tracks(
        contig,
        regions,
        track,  # flat per-region track buffer
        np.array([0, region_len], np.int64),  # (R+1) offsets
        params,
        strategy_id,
        base_seed,
        shifts=None,  # no jitter
        parallel=False,
    )

    # --- oracle: genoray decode records -> pure-Python SVAR1 realign, per hap ---
    raw = sv._readers[contig].decode_batch([(q_start, q_end)])
    R, So, Po = int(raw["n_regions"]), int(raw["n_samples"]), int(raw["ploidy"])
    assert (R, So, Po) == (1, S, P)
    off = np.asarray(raw["off"])  # (H+1,) per-hap variant offsets
    d_pos = np.asarray(raw["pos"])
    d_ilen = np.asarray(raw["ilen"])

    # Non-triviality: haps carry a varying number of DELs.
    per_hap_counts = (off[1:] - off[:-1]).tolist()
    assert per_hap_counts == [1, 1, 2, 1], per_hap_counts

    # `out_rag` is a `_Flat` (flat data/offsets buffer) cast to `Ragged` for typing;
    # `_Flat.__getitem__` only supports leading-axis slicing, so pull rows out via
    # its flat offsets directly — the same pattern used against this adapter's
    # sibling `_Flat` result in tests/test_svar2_reconstruct.py.
    out_data = np.asarray(out_rag.data)
    out_off = np.asarray(out_rag.offsets)

    for s in range(S):
        for p in range(P):
            h = (0 * S + s) * P + p  # region-major h=(r*S+s)*P+p
            gi0, gi1 = int(off[h]), int(off[h + 1])
            pos_h = np.ascontiguousarray(d_pos[gi0:gi1], np.int32)
            ilen_h = np.ascontiguousarray(d_ilen[gi0:gi1], np.int32)
            n_h = gi1 - gi0

            # Independently size the hap: region length + sum of (negative) ilens.
            exp_len = region_len + int(ilen_h.sum())

            got = out_data[int(out_off[h]) : int(out_off[h + 1])]
            assert got.shape[0] == exp_len, (
                f"(s={s},p={p}) SVAR2 len {got.shape[0]} != expected {exp_len} "
                f"(ilen={ilen_h.tolist()})"
            )

            # Synthetic single-hap SVAR1 layout: v_idxs 0..n_h, one group.
            geno_v_idxs = np.arange(n_h, dtype=np.int32)
            geno_offsets = np.array([0, n_h], np.int64)
            expected = np.empty(exp_len, np.float32)
            shift_and_realign_track_sparse(
                offset_idx=0,
                geno_v_idxs=geno_v_idxs,
                geno_offsets=geno_offsets,
                v_starts=pos_h,
                ilens=ilen_h,
                shift=0,
                track=track,
                query_start=q_start,
                out=expected,
                params=params,
                strategy_id=strategy_id,
                base_seed=base_seed,
                query=0,
                hap=h,
            )
            np.testing.assert_allclose(
                got,
                expected,
                rtol=0,
                atol=0,
                err_msg=f"(s={s},p={p}) SVAR2 track != SVAR1 oracle "
                f"(pos={pos_h.tolist()}, ilen={ilen_h.tolist()})",
            )
