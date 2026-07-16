"""Shared fixtures for tests/dataset/."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
import pyBigWig

import genvarloader as gvl

SEQLEN = 20

# 40 bp reference (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G), INS@6
# (C>CAT), dense SNP@9 (G>C, carried by 3 haps), DEL@11 (GTA>G, ilen -2).
# Mirrors the SVAR2-parity fixture in tests/dataset/test_svar2_dataset.py
# (`_src`/`svar_fixture`) -- same VCF, reused here for the streaming SVAR1
# window-read parity test (Task 4).
_SVAR1_STREAM_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_SVAR1_STREAM_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


@dataclass(slots=True)
class Svar1DatasetFixture:
    """Matched inputs for the streaming SVAR1 window-read parity test: the
    live `.svar` store + reference + 1-region bed to drive `_Svar1Backend`
    directly, and an independently-written+opened `gvl.Dataset` over the same
    store/bed/reference to compare against."""

    svar_path: Path
    reference_path: Path
    contigs: list[str]
    bed: pl.DataFrame
    dataset: gvl.Dataset


@pytest.fixture(scope="module")
def svar1_dataset_fixture(tmp_path_factory) -> Svar1DatasetFixture:
    from genoray import SparseVar, VCF

    d = tmp_path_factory.mktemp("svar1_stream_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR1_STREAM_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR1_STREAM_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar_path = tmp_path_factory.mktemp("svar1_stream_store") / "store.svar"
    SparseVar.from_vcf(
        svar_path, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})

    out = tmp_path_factory.mktemp("svar1_stream_ds") / "d1.gvl"
    gvl.write(out, bed, variants=SparseVar(svar_path), samples=None, overwrite=True)
    dataset = gvl.Dataset.open(out, reference=ref)

    return Svar1DatasetFixture(
        svar_path=svar_path,
        reference_path=ref,
        contigs=["chr1"],
        bed=bed,
        dataset=dataset,
    )


@pytest.fixture(scope="session")
def snap_dataset(source_bed, vcf_dir, reference, tmp_path_factory):
    """Phased VCF dataset with a "5ss" BigWig track, opened with a reference.

    Mirrors the ``base_ds`` fixture in ``tests/dataset/test_with_methods.py``.
    Opened with default settings (output_length="ragged", sequence_type="haplotypes",
    jitter=0, max_jitter=2, deterministic=True, rc_neg=True).
    """
    from genoray import VCF

    tmp_dir = tmp_path_factory.mktemp("snap_ds")
    out = tmp_dir / "snap.gvl"

    vcf_samples = ["s0", "s1", "s2"]
    # Header lengths are generous upper bounds for the regions in source.bed.
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(vcf_samples):
        bw_path = tmp_dir / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            # One short interval per contig region in source.bed; values differ
            # per sample. Mirrors base_ds in tests/dataset/test_with_methods.py.
            value = float(i + 1)
            bw.addEntries(
                ["chr1", "chr1", "chr2", "chr2"],
                [499_990, 1_010_686, 17_320, 1_234_560],
                ends=[500_030, 1_010_706, 17_340, 1_234_580],
                values=[value, value, value, value],
            )
        bw_paths[sample] = str(bw_path)

    bigwigs = gvl.BigWigs("5ss", bw_paths)
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(
        path=out,
        bed=source_bed,
        variants=vcf,
        tracks=bigwigs,
        max_jitter=2,
    )
    return gvl.Dataset.open(out, reference=reference)
