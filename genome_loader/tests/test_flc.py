from itertools import product
from pathlib import Path

import numpy as np
import pytest
from genome_loader.gloader import FixedLengthConsensus
from genome_loader.utils import ohe_to_bytes
from pysam import FastaFile
from pytest_cases import fixture, parametrize, parametrize_with_cases


@fixture
def wdir():
    return Path(__file__).parent


@pytest.mark.skip
def ref_byte(wdir):
    raise NotImplementedError
    return wdir.joinpath("data", "grch38.20.21.h5")


@pytest.mark.skip
def ref_acgt(wdir):
    raise NotImplementedError
    return wdir.joinpath("data", "grch38.20.21.ohe.ACGT.h5")


def ref_acgtn(wdir):
    return wdir.joinpath("data", "grch38.20.21.ohe.ACGTN.h5")


@fixture
def vcf_file(wdir):
    return wdir.joinpath("data", "ccle_snp_wes.reduced.zarr")


@fixture
def bed_file(wdir):
    return wdir.joinpath("data", "test.bed")


@fixture
def out_file(wdir):
    return wdir.joinpath("data/test.bed.flc.zarr")


@fixture
def length():
    return np.uint32(5)


@parametrize_with_cases("ref_file", cases=".", prefix="ref_")
@parametrize("samples", [None, ["OCI-AML5"], ["OCI-AML5", "NCI-H660"]])
def test_flc_writer(ref_file, vcf_file, bed_file, out_file, length, samples):
    flc = FixedLengthConsensus.from_ref_vcf_bed(
        ref_file, vcf_file, bed_file, out_file, length, samples
    )


@fixture
@parametrize_with_cases("ref_file", cases=".", prefix="ref_")
@parametrize("samples", [["OCI-AML5", "NCI-H660"]])
def flc(ref_file, vcf_file, bed_file, out_file, length, samples):
    return FixedLengthConsensus.from_ref_vcf_bed(
        ref_file, vcf_file, bed_file, out_file, length, samples
    )


@parametrize("samples", [None, ["OCI-AML5"], ["OCI-AML5", "NCI-H660"]])
def test_flc_sel(flc: FixedLengthConsensus, length, samples: list[str], wdir):
    flc_seqs = ohe_to_bytes(flc.sel(samples=samples), flc.alphabet).astype("U")  # type: ignore
    chroms = flc.bed["chrom"].to_numpy()
    starts = flc.bed["start"].to_numpy()
    if samples is None:
        samples = flc.samples
    haps = [1, 2]
    for (s_idx, sample), (h_idx, hap) in product(enumerate(samples), enumerate(haps)):
        bcftools_consensus = wdir.joinpath(
            "data", "fasta", f"{sample}.20.21.h{hap}.fa.gz"
        )
        for i, (chrom, start) in enumerate(zip(chroms, starts)):
            with FastaFile(str(bcftools_consensus)) as f:
                bcf_cons = f.fetch(region=f"{chrom}:{start}-{start+length-np.uint(1)}")
            # flc_seqs: (regions length samples ploidy)
            assert bcf_cons == "".join(flc_seqs[i, :, s_idx, h_idx])
