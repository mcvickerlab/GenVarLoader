from dataclasses import asdict, astuple, dataclass
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest
from genome_loader.gloader import GenomeLoader
from genome_loader.utils import ohe_to_bytes
from numpy.typing import NDArray
from pysam import FastaFile
from pytest_cases import fixture, parametrize_with_cases


@dataclass
class GLSelArgs:
    chroms: NDArray[np.str_]
    starts: NDArray[np.uint64]
    length: np.uint64
    samples: Optional[Sequence[str]] = None
    sorted_chroms: bool = False

    def __post_init__(self):
        if self.sorted_chroms is None:
            self.sorted_chroms = False


def gl_sel_one_chrom_ref():
    return GLSelArgs(
        np.array(["20", "20"]),
        np.array([96319, 279175], dtype="u4"),
        np.uint64(5),
        None,
        False,
    )


def gl_sel_two_chrom_ref():
    return GLSelArgs(
        np.array(["21", "20", "20"]),
        np.array([10414881, 96319, 279175], dtype="u4"),
        np.uint64(5),
        None,
        False,
    )


def gl_sel_1_region_1_samp():
    return GLSelArgs(
        np.array(["20"]),
        np.array([96319], dtype="u4"),
        np.uint64(5),
        ["OCI-AML5"],
        False,
    )


def gl_sel_1_chrom_1_samp():
    return GLSelArgs(
        np.array(["20", "20"]),
        np.array([96319, 279175], dtype="u4"),
        np.uint64(5),
        ["OCI-AML5"],
        False,
    )


def gl_sel_1_chrom_2_samp():
    return GLSelArgs(
        np.array(["20", "20"]),
        np.array([96319, 279175], dtype="u4"),
        np.uint64(5),
        ["OCI-AML5", "NCI-H660"],
        False,
    )


def gl_sel_2_chrom_2_samp():
    return GLSelArgs(
        np.array(["21", "20", "20"]),
        np.array([10414881, 96319, 279175], dtype="u4"),
        np.uint64(5),
        ["OCI-AML5", "NCI-H660"],
        False,
    )


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
@parametrize_with_cases("ref_file", cases=".", prefix="ref_")
def gloader(ref_file, vcf_file):
    return GenomeLoader(ref_file, vcf_file)


@parametrize_with_cases("gl_sel_args", cases=".", prefix="gl_sel_")
def test_gloader_sel(gloader: GenomeLoader, gl_sel_args: GLSelArgs, wdir):
    chroms, starts, length, samples, sorted_chroms = astuple(gl_sel_args)
    ohe_cons = gloader.sel(**asdict(gl_sel_args))
    gl_seqs = ohe_to_bytes(ohe_cons, gloader.spec).astype("U")  # type: ignore
    haps = [1, 2]
    if samples is None:
        ref = wdir.joinpath("data", "fasta", "grch38.20.21.fa.gz")
        for i, (chrom, start) in enumerate(zip(chroms, starts)):
            with FastaFile(str(ref)) as f:
                ref_seq = f.fetch(region=f"{chrom}:{start}-{start+length-np.uint(1)}")
                # gl_seqs: (regions length)
                assert ref_seq == "".join(gl_seqs[i])
    else:
        for (s_idx, sample), (h_idx, hap) in product(
            enumerate(samples), enumerate(haps)
        ):
            bcftools_consensus = wdir.joinpath(
                "data", "fasta", f"{sample}.20.21.h{hap}.fa.gz"
            )
            for i, (chrom, start) in enumerate(zip(chroms, starts)):
                with FastaFile(str(bcftools_consensus)) as f:
                    bcf_cons = f.fetch(
                        region=f"{chrom}:{start}-{start+length-np.uint(1)}"
                    )
                # gl_seqs: (regions length samples ploidy)
                assert bcf_cons == "".join(gl_seqs[i, :, s_idx, h_idx])