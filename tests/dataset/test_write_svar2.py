"""Tests for the `.svar2` write path: `_write_from_svar2` + dispatch.

Builds a `.svar2` store (and a matched `.svar` store from the same VCF+FASTA)
using the same recipe as `tests/test_svar2_reconstruct.py`'s `svar2_store`
fixture, then exercises `gvl.write(..., variants=SparseVar2(...))`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader._dataset._svar2_link import Svar2Link

# 40 bp reference (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G), INS@6 (C>CAT),
# DEL@11 (GTA>G, ilen -2). Genotypes exercise both samples and both ploids.
# Mirrors tests/test_svar2_reconstruct.py's svar2_store fixture exactly, so the
# matched .svar (SVAR1) store built from the same VCF is a valid parity oracle.
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
def vcf_and_ref(tmp_path_factory) -> tuple[Path, Path]:
    """A bgzipped/indexed BCF + FASTA shared by the .svar2 and .svar fixtures."""
    d = tmp_path_factory.mktemp("svar2_write_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)
    return bcf, ref


@pytest.fixture(scope="module")
def svar2_store(vcf_and_ref, tmp_path_factory) -> Path:
    bcf, ref = vcf_and_ref
    from genoray import _core

    out = tmp_path_factory.mktemp("svar2_write") / "store.svar2"
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


@pytest.fixture(scope="module")
def svar1_store(vcf_and_ref, tmp_path_factory) -> Path:
    bcf, _ref = vcf_and_ref
    from genoray import VCF, SparseVar

    out = tmp_path_factory.mktemp("svar1_write") / "store.svar"
    SparseVar.from_vcf(out, VCF(bcf), max_mem="64m", overwrite=True)
    return out


def test_write_svar2_emits_cache(svar2_store: Path, tmp_path: Path):
    from genoray import SparseVar2

    svar2 = SparseVar2(svar2_store)
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [0, 5],
            "chromEnd": [20, 15],
        }
    )
    out = tmp_path / "ds.gvl"
    gvl.write(out, bed, variants=svar2, samples=None, overwrite=True)

    rd = out / "genotypes" / "svar2_ranges"
    meta = json.loads((rd / "svar2_meta.json").read_text())
    assert set(meta) >= {
        "vk_snp_range",
        "vk_indel_range",
        "dense_snp_range",
        "dense_indel_range",
        "sample_cols",
    }
    assert meta["ploidy"] == svar2.ploidy

    md = json.loads((out / "metadata.json").read_text())
    assert md["svar2_link"] is not None
    assert md["ploidy"] == svar2.ploidy
    Svar2Link.model_validate(md["svar2_link"])  # shape check

    # ---- FIX 1: verify cache CONTENTS (not just shapes/keys) against a direct
    # _find_ranges call over the same regions. gvl sorts the written samples, so
    # replay _find_ranges with the sorted sample list to match slot ordering.
    # This LOCKS the row-major (R, S, P) reshape and per-contig layout: a
    # scrambled / mis-transposed cache would fail loudly here.
    sorted_samples = sorted(
        svar2.available_samples
    )  # what gvl.write wrote (samples.sort())
    S, P = len(sorted_samples), svar2.ploidy

    def mm(name: str) -> np.ndarray:
        # raw memmaps are written as "<name>.npy" (no .npy header); the meta key
        # is the bare name. Read via np.memmap with the recorded shape/dtype.
        shape = tuple(meta[name]["shape"])
        return np.array(
            np.memmap(rd / f"{name}.npy", dtype=np.int64, mode="r", shape=shape)
        )

    vk_snp = mm("vk_snp_range")  # (R, S, P, 2)
    vk_indel = mm("vk_indel_range")  # (R, S, P, 2)
    dense_snp = mm("dense_snp_range")  # (R, 2)
    dense_indel = mm("dense_indel_range")  # (R, 2)

    # sample_cols is written with np.save (has a .npy header): read with np.load.
    sample_cols = np.load(rd / "sample_cols.npy")
    assert sample_cols.tolist() == [
        svar2.available_samples.index(s) for s in sorted_samples
    ]

    contig_offset = 0
    for (c,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        rc = df.height
        lo, hi = contig_offset, contig_offset + rc
        d = svar2._find_ranges(
            c,
            df["chromStart"].to_numpy(),
            df["chromEnd"].to_numpy(),
            samples=sorted_samples,
        )
        # vk ranges: reshape (rc, S, P, 2) -> (rc*S*P, 2) must equal _find_ranges'
        # row-major (R*S*P, 2). This pins the reshape done in _write_from_svar2.
        np.testing.assert_array_equal(
            vk_snp[lo:hi].reshape(rc * S * P, 2),
            np.asarray(d["vk_snp_range"], np.int64),
        )
        np.testing.assert_array_equal(
            vk_indel[lo:hi].reshape(rc * S * P, 2),
            np.asarray(d["vk_indel_range"], np.int64),
        )
        # dense ranges: per-region (rc, 2), upcast int32 -> int64.
        np.testing.assert_array_equal(
            dense_snp[lo:hi], np.asarray(d["dense_snp_range"], np.int64)
        )
        np.testing.assert_array_equal(
            dense_indel[lo:hi], np.asarray(d["dense_indel_range"], np.int64)
        )
        contig_offset += rc


def test_write_svar2_max_ends_matches_svar1(
    svar2_store: Path, svar1_store: Path, tmp_path: Path
):
    """SVAR1 parity gate: end-extension semantics must match exactly.

    Regions are chosen to overlap the DEL at (0-based) POS 11 with varying
    windows, so the extension is non-trivial and exercises the "no variants"
    (keep chromEnd) branch too.
    """
    from genoray import SparseVar, SparseVar2

    svar2 = SparseVar2(svar2_store)
    svar1 = SparseVar(svar1_store)

    bed = pl.DataFrame(
        {
            "chrom": ["chr1"] * 5,
            "chromStart": [0, 0, 5, 12, 20],
            "chromEnd": [15, 20, 10, 13, 30],
        }
    )

    out2 = tmp_path / "ds_svar2.gvl"
    gvl.write(out2, bed, variants=svar2, samples=None, overwrite=True)

    out1 = tmp_path / "ds_svar1.gvl"
    gvl.write(out1, bed, variants=svar1, samples=None, overwrite=True)

    regions2 = np.load(out2 / "regions.npy")
    regions1 = np.load(out1 / "regions.npy")

    # columns: chrom_idx, chromStart, chromEnd, strand
    chrom_end_2 = regions2[:, 2]
    chrom_end_1 = regions1[:, 2]

    assert chrom_end_2.tolist() == chrom_end_1.tolist(), (
        f"svar2 max_ends {chrom_end_2.tolist()} != svar1 max_ends {chrom_end_1.tolist()}"
    )


# Same-POS tie fixture (FIX 2): two records at POS 12 (0-based 11) with different
# ends -- a SNP (G>A, end=12) and a DEL (GTA>G, ILEN -2, end=14) -- placed on
# DIFFERENT haplotypes of S0 (SNP on hap0, DEL on hap1). A single haplotype
# cannot carry both an overlapping SNP and DEL, so putting them on the same hap
# would make the svar2 encoder drop one; different haps keeps both variants
# present and reachable. Ordering is the coordinator's exact example (SNP record
# first, DEL record second), so in store order the DEL gets the higher v_idx.
# SVAR1's max_ends picks the max-v_idx variant's end; svar2 picks the max-end
# variant on a POS tie -- here both rules select the DEL (end 14), so the paths
# agree. See the task-2 report for the reverse store order (DEL-first), where
# the two rules provably diverge.
_TIE_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t12\t.\tG\tA\t.\t.\t.\tGT\t1|0\t0|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t0|1\t0|0
"""


@pytest.fixture(scope="module")
def tie_stores(tmp_path_factory) -> tuple[Path, Path]:
    """Matched .svar2 and .svar stores from the same two-same-POS-records VCF."""
    from genoray import VCF, SparseVar, _core

    d = tmp_path_factory.mktemp("svar2_tie")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_TIE_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar2_out = d / "store.svar2"
    _core.run_conversion_pipeline(
        str(bcf),
        str(ref),
        ["chr1"],
        str(svar2_out),
        ["S0", "S1"],
        25_000,
        2,
        1,
        8 * 1024 * 1024,
    )
    assert (svar2_out / "meta.json").exists(), "svar2 conversion did not finish"

    svar1_out = d / "store.svar"
    SparseVar.from_vcf(
        svar1_out, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )
    return svar2_out, svar1_out


def test_write_svar2_max_ends_same_pos_tie(
    tie_stores: tuple[Path, Path], tmp_path: Path
):
    """SVAR1 parity on a same-POS tie: a SNP and a DEL at the same position.

    The bed region ends 1bp short of the DEL's footprint so the extension is
    variant-driven (not masked by the region's own chromEnd). Both paths must
    agree on the extended chromEnd.
    """
    from genoray import SparseVar, SparseVar2

    svar2_out, svar1_out = tie_stores
    svar2 = SparseVar2(svar2_out)
    svar1 = SparseVar(svar1_out)

    # POS 12 -> 0-based 11. Region [11, 13) overlaps it; region chromEnd 13 is
    # below the DEL end (14), so the max_ends extension is variant-driven.
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [11], "chromEnd": [13]})

    out2 = tmp_path / "tie_svar2.gvl"
    gvl.write(out2, bed, variants=svar2, samples=None, overwrite=True)
    out1 = tmp_path / "tie_svar1.gvl"
    gvl.write(out1, bed, variants=svar1, samples=None, overwrite=True)

    chrom_end_2 = np.load(out2 / "regions.npy")[:, 2]
    chrom_end_1 = np.load(out1 / "regions.npy")[:, 2]

    assert chrom_end_2.tolist() == chrom_end_1.tolist(), (
        f"same-POS tie: svar2 max_ends {chrom_end_2.tolist()} != "
        f"svar1 max_ends {chrom_end_1.tolist()}"
    )


def test_svar2_extend_to_length_false_raises(svar2_store: Path, tmp_path: Path):
    """extend_to_length=False is unsupported for a .svar2 source: it must raise
    NotImplementedError, not silently produce an extended dataset."""
    from genoray import SparseVar2

    svar2 = SparseVar2(svar2_store)
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [0, 5],
            "chromEnd": [20, 15],
        }
    )
    out = tmp_path / "ds.gvl"
    with pytest.raises(NotImplementedError, match="extend_to_length"):
        gvl.write(
            out,
            bed,
            variants=svar2,
            samples=None,
            extend_to_length=False,
            overwrite=True,
        )


def _reference_region_max_ends(svar2, contig, starts, ends, samples):
    """Byte-for-byte copy of the ORIGINAL _svar2_region_max_ends triple-loop,
    kept here as the oracle that pins the vectorized rewrite byte-identical."""
    import numpy as np

    R, S_all, P = len(starts), svar2.n_samples, svar2.ploidy
    sel = [svar2.available_samples.index(s) for s in samples]
    dec = svar2.decode(contig, list(zip(starts.tolist(), ends.tolist())))
    pos_arr = dec.data["pos"]
    ilen_arr = dec.data["ilen"]
    off = np.asarray(dec.offsets)
    out = np.asarray(ends, np.int64).copy()
    for r in range(R):
        best_pos, best_end = -1, -1
        for s in sel:
            for p in range(P):
                h = (r * S_all + s) * P + p
                a, b = int(off[h]), int(off[h + 1])
                if a == b:
                    continue
                seg_pos = pos_arr[a:b]
                seg_ilen = ilen_arr[a:b]
                j = int(np.argmax(seg_pos))
                p_pos = int(seg_pos[j])
                p_end = (p_pos + 1) - min(int(seg_ilen[j]), 0)
                if p_pos > best_pos or (p_pos == best_pos and p_end > best_end):
                    best_pos, best_end = p_pos, p_end
        if best_pos >= 0:
            out[r] = best_end
    return out.astype(np.int32)


def test_svar2_region_max_ends_matches_reference(svar2_store: Path):
    """Vectorized _svar2_region_max_ends must equal the original per-hap loop,
    including the pos-then-end tie-break and the empty-region default = chromEnd."""
    from genoray import SparseVar2

    from genvarloader._dataset._write import _svar2_region_max_ends

    svar2 = SparseVar2(svar2_store)
    # Overlaps the DEL at 0-based POS 11 with varying windows + a no-variant
    # region ([20,30]) so both the extension and keep-chromEnd branches run.
    starts = np.array([0, 0, 5, 12, 20], dtype=np.int64)
    ends = np.array([15, 20, 10, 13, 30], dtype=np.int64)
    samples = list(svar2.available_samples)

    got = _svar2_region_max_ends(svar2, "chr1", starts, ends, samples)
    ref = _reference_region_max_ends(svar2, "chr1", starts, ends, samples)
    np.testing.assert_array_equal(got, ref)

    # Anti-vacuity: at least one region must be EXTENDED past its chromEnd (the
    # DEL at POS 11 extends windows that overlap it), else the test only checks
    # the trivial default path.
    assert (got != ends.astype(np.int32)).any(), (
        f"test is vacuous: no region extended (got={got.tolist()}, ends={ends.tolist()})"
    )


def test_svar2_region_max_ends_large_positions():
    """Regression: the composite key must pack a BOUNDED tie-break, not the
    absolute end. A variant past ~2 Mb (real chromosomes are hundreds of Mb)
    must not overflow the packing / assert-fail. Uses a stub whose decode returns
    large positions so we can exercise realistic coordinates without a huge store.
    """
    from types import SimpleNamespace

    import numpy as np

    from genvarloader._dataset._write import _svar2_region_max_ends

    # 2 regions x 1 sample x ploidy 1 = 2 haps, 1 variant each:
    #   region 0: SNP  at pos 3_000_000 (ilen 0)  -> end 3_000_001
    #   region 1: DEL  at pos 5_000_000 (ilen -2) -> end 5_000_003
    class _StubSvar2:
        n_samples = 1
        ploidy = 1
        available_samples = ["S0"]

        def decode(self, contig, regions):
            return SimpleNamespace(
                data={
                    "pos": np.array([3_000_000, 5_000_000], np.int64),
                    "ilen": np.array([0, -2], np.int64),
                },
                offsets=np.array([0, 1, 2], np.int64),
            )

    svar2 = _StubSvar2()
    starts = np.array([0, 0], np.int64)
    ends = np.array([10, 10], np.int64)  # small chromEnd so both variants extend
    got = _svar2_region_max_ends(svar2, "chrBig", starts, ends, ["S0"])
    ref = _reference_region_max_ends(svar2, "chrBig", starts, ends, ["S0"])
    np.testing.assert_array_equal(got, ref)
    np.testing.assert_array_equal(got, np.array([3_000_001, 5_000_003], np.int32))
