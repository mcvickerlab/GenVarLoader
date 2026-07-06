"""Cross-repo end-to-end validation of the SVAR2 two-source reconstruction kernel.

Builds a known SVAR2 store from a VCF+FASTA fixture (genoray's conversion pipeline),
reconstructs haplotypes through gvl's two-source path (SparseVar2Source, which decodes
genoray's raw two-channel overlap_batch inline), and compares byte-for-byte against an
INDEPENDENT pure-Python consensus applied to genoray's materialized decode records
(the M6c oracle). Agreement proves: (a) gvl's var_key⋈dense merge+decode matches
genoray's decode, and (b) gvl's reconstruction loop matches an independent reference.
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

    d = tmp_path_factory.mktemp("svar2")
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


def _consensus(ref: bytes, pos, ilen, alleles, q_start: int, q_end: int) -> bytes:
    """Independent reference reconstruction: apply position-sorted (pos, ilen, allele)
    records to `ref[q_start:q_end]`. A pure DEL has an empty allele — the anchor base
    ref[pos] is retained and the following |ilen| bases are dropped (genoray's convention).
    """
    order = np.argsort(pos, kind="stable")
    out = bytearray()
    ref_idx = q_start
    for i in order:
        p = int(pos[i])
        il = int(ilen[i])
        al = bytes(alleles[i])
        v_end = p - min(0, il) + 1
        # DEL spanning the region start: advance ref past it, emit nothing.
        if il < 0 and p < q_start and v_end >= q_start:
            ref_idx = v_end
            continue
        if p < ref_idx:  # overlapping variant already consumed — first-one-wins
            continue
        if p >= q_end:
            break
        out += ref[ref_idx:p]
        seq = al if len(al) > 0 else ref[p : p + 1]
        out += seq
        ref_idx = v_end
    out += ref[ref_idx:q_end]
    return bytes(out)


def test_svar2_two_source_matches_decode_oracle(svar2_store):
    import genoray
    from genvarloader._dataset._svar2_source import SparseVar2Source

    contig = "chr1"
    q_start, q_end = 0, 40
    regions = [(q_start, q_end)]
    ref_bytes = _REF.encode()

    sv = genoray.SparseVar2(str(svar2_store))
    S, P = sv.n_samples, sv.ploidy
    assert (S, P) == (2, 2)

    # --- two-source reconstruction (the path under test) ---
    src = SparseVar2Source(sv)
    hap_rag = src.reconstruct(
        contig,
        regions,
        np.frombuffer(ref_bytes, np.uint8),
        np.array([0, len(ref_bytes)], np.int64),
        pad_char=ord("N"),
        shifts=None,  # no jitter
        output_length=-1,  # ragged
        parallel=False,
    )
    ts_data = np.asarray(hap_rag.data).view("S1").tobytes()
    ts_off = np.asarray(hap_rag.offsets)

    # --- oracle: genoray's materialized decode records (raw flat dict) ---
    raw = sv._readers[contig].decode_batch([(q_start, q_end)])
    R, So, Po = int(raw["n_regions"]), int(raw["n_samples"]), int(raw["ploidy"])
    assert (R, So, Po) == (1, S, P)
    H = R * So * Po
    off = np.asarray(raw["off"])  # (H+1,) per-hap variant offsets
    str_off = np.asarray(raw["str_off"])  # per-variant allele-byte offsets
    d_pos = np.asarray(raw["pos"])
    d_ilen = np.asarray(raw["ilen"])
    d_allele = np.asarray(raw["allele"]).tobytes()

    # Non-triviality: the fixture yields per-hap variant counts [2, 2, 1, 2]
    # (S0h0, S0h1, S1h0, S1h1) — SNP/INS/DEL spread across samples and ploids.
    per_hap_counts = (off[1:] - off[:-1]).tolist()
    assert per_hap_counts == [2, 2, 1, 2], per_hap_counts

    for h in range(H):
        gi0, gi1 = int(off[h]), int(off[h + 1])
        pos_h = d_pos[gi0:gi1]
        ilen_h = d_ilen[gi0:gi1]
        alleles_h = [
            d_allele[int(str_off[gi]) : int(str_off[gi + 1])] for gi in range(gi0, gi1)
        ]
        expected = _consensus(ref_bytes, pos_h, ilen_h, alleles_h, q_start, q_end)
        got = ts_data[int(ts_off[h]) : int(ts_off[h + 1])]
        assert got == expected, (
            f"hap {h}: two-source {got!r} != oracle {expected!r} "
            f"(pos={pos_h.tolist()}, ilen={ilen_h.tolist()})"
        )

    # Sensitivity anchor: a DEL-carrying hap must be shorter than the reference,
    # and an INS-only hap longer — proving indels actually change output length.
    hap_lens = (ts_off[1:] - ts_off[:-1]).tolist()
    assert min(hap_lens) < len(ref_bytes) < max(hap_lens), hap_lens
