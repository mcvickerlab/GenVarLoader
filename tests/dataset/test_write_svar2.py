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
        "region_starts",
        "sample_cols",
    }
    assert meta["ploidy"] == svar2.ploidy

    md = json.loads((out / "metadata.json").read_text())
    assert md["svar2_link"] is not None
    assert md["ploidy"] == svar2.ploidy
    Svar2Link.model_validate(md["svar2_link"])  # shape check

    region_starts_shape = tuple(meta["region_starts"]["shape"])
    region_starts = np.memmap(
        rd / "region_starts.npy", dtype=np.int64, mode="r", shape=region_starts_shape
    )
    assert region_starts.shape == (bed.height,)

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
    region_starts_full = mm("region_starts")  # (R,)

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
        # region_starts: exact per-contig match (upcast int32 -> int64).
        np.testing.assert_array_equal(
            region_starts_full[lo:hi], np.asarray(d["region_starts"], np.int64)
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
