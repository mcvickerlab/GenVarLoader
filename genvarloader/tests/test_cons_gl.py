from dataclasses import asdict, astuple, dataclass
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest
from icecream import ic
from numpy.typing import NDArray
from pysam import FastaFile
from pytest_cases import fixture, parametrize_with_cases

from genvarloader.gloader.consensus import (
    ConsensusGenomeLoader,
    _sel_bytes_helper,
    _sel_ohe_helper,
)
from genvarloader.utils import bytes_to_ohe, ohe_to_bytes


def test_sel_bytes_helper_acgtn():
    rel_starts = np.array([0], "i4")  # (r)
    length = np.uint32(3)
    offsets = np.array([0, 2], "u4")  # (r)
    genos = np.array([[[b"A"]]], dtype="|S1").view("i1")  # (v, s, p)
    var_pos = np.array([1], "i4")  # (v)
    ref_chrom = np.frombuffer(b"NNN", dtype="|S1").view("i1").copy()  # (l)
    ref_start = np.int32(0)
    out = _sel_bytes_helper(
        rel_starts, length, offsets, genos, var_pos, ref_chrom, ref_start
    )
    return out


def test_sel_ohe_helper_acgtn():
    rel_starts = np.array([0], "i4")  # (r)
    length = np.uint32(3)
    offsets = np.array([0, 2], "u4")  # (r)
    ohe_genos = bytes_to_ohe(
        np.array([[[b"A"]]], dtype="|S1"), np.frombuffer(b"ACGTN", dtype="|S1")
    )  # (v, s, p)
    var_pos = np.array([1], "i4")  # (v)
    ohe_ref_chrom = np.eye(5, dtype="u1")[[4, 4, 4]]  # (l, a)
    ref_start = np.int32(0)
    out = _sel_ohe_helper(
        rel_starts, length, offsets, ohe_genos, var_pos, ohe_ref_chrom, ref_start
    )
    return out


@dataclass
class ConsGLSelArgs:
    chroms: NDArray[np.str_]
    starts: NDArray[np.int32]
    length: np.uint64
    samples: Optional[Sequence[str]] = None
    ploid_idx: Optional[NDArray[np.uint32]] = None
    sorted_chroms: bool = False

    def __post_init__(self):
        if self.sorted_chroms is None:
            self.sorted_chroms = False


def cons_gl_sel_1_region_1_samp():
    return ConsGLSelArgs(
        np.array(["20"]),
        np.array([96319], dtype="i4"),
        np.uint64(5),
        ["OCI-AML5"],
        None,
        False,
    )


def cons_gl_sel_1_chrom_1_samp():
    return ConsGLSelArgs(
        np.array(["20", "20"]),
        np.array([96319, 279175], dtype="i4"),
        np.uint64(5),
        ["OCI-AML5"],
        None,
        False,
    )


def cons_gl_sel_1_chrom_2_samp():
    return ConsGLSelArgs(
        np.array(["20", "20"]),
        np.array([96319, 279175], dtype="i4"),
        np.uint64(5),
        ["OCI-AML5", "NCI-H660"],
        None,
        False,
    )


def cons_gl_sel_2_chrom_2_samp():
    return ConsGLSelArgs(
        np.array(["21", "20", "20"]),
        np.array([10414881, 96319, 279175], dtype="i4"),
        np.uint64(5),
        ["OCI-AML5", "NCI-H660"],
        None,
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
def cons_gloader(ref_file, vcf_file):
    return ConsensusGenomeLoader(ref_file, vcf_file)


@parametrize_with_cases("cons_gl_sel_args", cases=".", prefix="cons_gl_sel_")
def test_cons_gloader_sel(
    cons_gloader: ConsensusGenomeLoader, cons_gl_sel_args: ConsGLSelArgs, wdir
):
    chroms, starts, length, samples, ploid_idx, sorted_chroms = astuple(
        cons_gl_sel_args
    )
    ohe_cons = cons_gloader.sel(**asdict(cons_gl_sel_args))
    gl_seqs = ohe_to_bytes(ohe_cons, cons_gloader.spec).astype("U")  # type: ignore
    haps = [1, 2]
    for (s_idx, sample), (h_idx, hap) in product(enumerate(samples), enumerate(haps)):
        bcftools_consensus = wdir.joinpath(
            "data", "fasta", f"{sample}.20.21.h{hap}.fa.gz"
        )
        for i, (chrom, start) in enumerate(zip(chroms, starts)):
            with FastaFile(str(bcftools_consensus)) as f:
                bcf_cons = f.fetch(chrom, start, start + length)
            # gl_seqs: (regions samples ploidy length)
            assert bcf_cons == "".join(gl_seqs[i, s_idx, h_idx])
