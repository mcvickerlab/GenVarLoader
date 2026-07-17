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


# Two-contig reference + variants for the Task 5 multi-region/multi-contig
# streaming parity test. chr1 reuses `_SVAR1_STREAM_REF`/its 4 variants
# (SNP@3, INS@7, SNP@10, DEL@12); chr2 is a distinct 40bp reference with its
# own SNP + insertion, matching the `_REF2`/`_VCF2` pattern in
# `test_svar2_dataset.py`'s multicontig fixtures. Extended to 3 samples.
_SVAR1_MC_REF2 = "TTGGCCAATTGGCCAATTACGTACGTTTGGCCAATTGGCC"
_SVAR1_MC_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##contig=<ID=chr2,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\t0|1
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\t1|0
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0\t0|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\t1|1
chr2\t5\t.\tC\tT\t.\t.\t.\tGT\t1|0\t1|1\t0|1
chr2\t9\t.\tT\tTGG\t.\t.\t.\tGT\t0|1\t1|0\t1|1
chr2\t21\t.\tG\tA\t.\t.\t.\tGT\t1|1\t0|1\t1|0
"""


@dataclass(slots=True)
class Svar1MultiContigFixture:
    """Matched inputs for the Task 5 public `StreamingDataset` parity test:
    an unsorted, interleaved-contig bed (>=2 contigs, >=10 regions, 3
    samples) over a live `.svar` store + reference, plus the path to an
    independently-written `gvl.Dataset` directory over the same bed/store to
    compare against (opened lazily by the test via `gvl.Dataset.open`)."""

    svar_path: Path
    reference_path: Path
    contigs: list[str]
    bed: pl.DataFrame
    dataset_path: Path


@pytest.fixture(scope="module")
def svar1_multicontig_fixture(tmp_path_factory) -> Svar1MultiContigFixture:
    from genoray import SparseVar, VCF

    d = tmp_path_factory.mktemp("svar1_mc_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR1_STREAM_REF}\n>chr2\n{_SVAR1_MC_REF2}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR1_MC_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar_path = tmp_path_factory.mktemp("svar1_mc_store") / "store.svar"
    SparseVar.from_vcf(
        svar_path,
        VCF(bcf),
        max_mem="1g",
        samples=["S0", "S1", "S2"],
        overwrite=True,
    )

    # 12 regions (6 per contig, 20bp sliding windows), UNSORTED / interleaved
    # chrom order -- exercises the original-bed-row-order <-> sorted-storage
    # translation on both the streaming and written-dataset sides.
    starts = [0, 4, 8, 12, 16, 20]
    rows = []
    for i, s in enumerate(starts):
        # alternate contigs per row so the bed is genuinely interleaved, not
        # just "all of chr2 then all of chr1".
        c1, c2 = ("chr2", "chr1") if i % 2 == 0 else ("chr1", "chr2")
        rows.append({"chrom": c1, "chromStart": s, "chromEnd": s + 20})
        rows.append({"chrom": c2, "chromStart": s, "chromEnd": s + 20})
    bed = pl.DataFrame(rows)

    out = tmp_path_factory.mktemp("svar1_mc_ds") / "d1.gvl"
    gvl.write(out, bed, variants=SparseVar(svar_path), samples=None, overwrite=True)

    return Svar1MultiContigFixture(
        svar_path=svar_path,
        reference_path=ref,
        contigs=["chr1", "chr2"],
        bed=bed,
        dataset_path=out,
    )


@dataclass(slots=True)
class Svar1MixedNamingFixture:
    """Regression fixture for the `ref_c_idx` contig-naming-style bug: the `.svar`
    store/VCF uses UCSC-style contig names (``chr1``) while one of the two
    references provided uses Ensembl style (``1``, no prefix). `Reference.from_path`
    documents mixed UCSC/Ensembl naming as supported (it normalizes contig names to
    match the FASTA), so a correct `StreamingDataset` must handle this pairing.

    `reference_chr_path` names its single contig ``chr1`` (matches the store) and
    is used to build/open the comparison `gvl.Dataset`; `reference_ensembl_path`
    names it ``1`` and is the one actually fed to `StreamingDataset` under test --
    same underlying sequence, different naming style, so streamed output must
    still be byte-identical to the written-and-opened comparison dataset."""

    svar_path: Path
    reference_ensembl_path: Path
    reference_chr_path: Path
    contigs: list[str]
    bed: pl.DataFrame
    dataset_path: Path


@pytest.fixture(scope="module")
def svar1_mixed_naming_fixture(tmp_path_factory) -> Svar1MixedNamingFixture:
    from genoray import SparseVar, VCF

    d = tmp_path_factory.mktemp("svar1_mixed_naming_src")
    ref_chr = d / "ref_chr.fa"
    ref_chr.write_text(f">chr1\n{_SVAR1_STREAM_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref_chr)], check=True)

    ref_ensembl = d / "ref_ensembl.fa"
    ref_ensembl.write_text(f">1\n{_SVAR1_STREAM_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref_ensembl)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR1_STREAM_VCF)  # contig "chr1", UCSC-style, like the store
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar_path = tmp_path_factory.mktemp("svar1_mixed_naming_store") / "store.svar"
    SparseVar.from_vcf(
        svar_path, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})

    out = tmp_path_factory.mktemp("svar1_mixed_naming_ds") / "d1.gvl"
    gvl.write(out, bed, variants=SparseVar(svar_path), samples=None, overwrite=True)

    return Svar1MixedNamingFixture(
        svar_path=svar_path,
        reference_ensembl_path=ref_ensembl,
        reference_chr_path=ref_chr,
        contigs=["chr1"],
        bed=bed,
        dataset_path=out,
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
