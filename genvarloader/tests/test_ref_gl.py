from dataclasses import asdict, astuple, dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pysam import FastaFile
from pytest_cases import fixture, parametrize_with_cases

from genvarloader.gloader.reference import ReferenceGenomeLoader
from genvarloader.utils import ohe_to_bytes


@dataclass
class RefGLSelArgs:
    chroms: NDArray[np.str_]
    starts: NDArray[np.int32]
    length: np.uint64
    sorted_chroms: bool = False


def ref_gl_sel_one_chrom():
    return RefGLSelArgs(
        np.array(["20", "20"]),
        np.array([96319, 279175], dtype="i4"),
        np.uint64(5),
        False,
    )


def ref_gl_sel_two_chrom():
    return RefGLSelArgs(
        np.array(["21", "20", "20"]),
        np.array([10414881, 96319, 279175], dtype="i4"),
        np.uint64(5),
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
@parametrize_with_cases("ref_file", cases=".", prefix="ref_")
def ref_gloader(ref_file):
    return ReferenceGenomeLoader(ref_file)


@parametrize_with_cases("ref_gl_sel_args", cases=".", prefix="ref_gl_sel_")
def test_ref_gloader_sel(
    ref_gloader: ReferenceGenomeLoader, ref_gl_sel_args: RefGLSelArgs, wdir
):
    chroms, starts, length, sorted_chroms = astuple(ref_gl_sel_args)
    ohe_cons = ref_gloader.sel(**asdict(ref_gl_sel_args))
    gl_seqs = ohe_to_bytes(ohe_cons, gloader.spec).astype("U")  # type: ignore
    ref = wdir.joinpath("data", "fasta", "grch38.20.21.fa.gz")
    for i, (chrom, start) in enumerate(zip(chroms, starts)):
        with FastaFile(str(ref)) as f:
            ref_seq = f.fetch(chrom, start, start + length)
            # gl_seqs: (regions length)
            assert ref_seq == "".join(gl_seqs[i])
