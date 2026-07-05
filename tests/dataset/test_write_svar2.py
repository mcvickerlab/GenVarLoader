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
