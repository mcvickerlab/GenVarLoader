from pathlib import Path
from typing import Literal

import genvarloader as gvl
import numpy as np
from pytest_cases import fixture, parametrize_with_cases


def seqs_ref():
    return "reference"


def seqs_haps():
    return "haplotypes"


def seqs_annot():
    return "annotated"


@fixture(scope="session")
@parametrize_with_cases("seq_type", prefix="seqs_", cases=".")
def dataset(
    phased_vcf_gvl: Path,
    ref_fasta: Path,
    seq_type: Literal["reference", "haplotypes", "annotated"],
):
    ds = gvl.Dataset.open(phased_vcf_gvl, ref_fasta)
    return ds.with_seqs(seq_type)


def idx_scalar():
    return 0


def idx_neg_scalar():
    return -1


def idx_slice_none():
    return slice(None)


def idx_slice_start_none():
    return slice(1, None)


def idx_slice_none_stop():
    return slice(None, 2)


def idx_list():
    return [0, 1, 2]


def idx_array():
    return np.arange(3)


@parametrize_with_cases("idx", prefix="idx_", cases=".")
def test_ds_indexing(dataset, idx):
    dataset[idx]


@parametrize_with_cases("r_idx", prefix="idx_", cases=".")
@parametrize_with_cases("s_idx", prefix="idx_", cases=".")
def test_rs_indexing(dataset, r_idx, s_idx):
    dataset[r_idx, s_idx]
