"""Shared fixtures for tests/dataset/."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import pyBigWig

import genvarloader as gvl

SEQLEN = 20

# 40 bp reference (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G), INS@6 (C>CAT),
# DEL@11 (GTA>G, ilen -2). Genotypes exercise both samples and both ploids.
# S2 is 0|0 everywhere: an ENTIRELY EMPTY sample column, so the sparse cache's
# "cell not present" branch is exercised. Variants stop at 0-based 13, so a
# region past that (see test_write_svar2_emits_cache) is an entirely empty ROW.
# Mirrors tests/test_svar2_reconstruct.py's svar2_store fixture (which keeps only
# S0/S1), so the matched .svar (SVAR1) store built from the same VCF is still a
# valid parity oracle.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\t0|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\t0|0
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
        ["S0", "S1", "S2"],
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
