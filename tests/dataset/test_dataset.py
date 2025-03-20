from pathlib import Path

import genvarloader as gvl
import numpy as np
from pytest_cases import fixture, parametrize_with_cases

data_dir = Path(__file__).resolve().parents[1] / "data"
ref = data_dir / "fasta" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.bgz"

def ds_phased():
    return gvl.Dataset.open(data_dir / "phased_dataset.gvl", ref)


def ds_unphased():
    return gvl.Dataset.open(data_dir / "unphased_dataset.gvl", ref)


def seqs_ref():
    return "reference"


def seqs_haps():
    return "haplotypes"


def seqs_annot():
    return "annotated"


def bool_false():
    return False


def bool_true():
    return True


@fixture(scope="session")
@parametrize_with_cases("ds", prefix="ds_", cases=".")
@parametrize_with_cases("seq_type", prefix="seqs_", cases=".")
@parametrize_with_cases("return_indices", prefix="bool_", cases=".")
def dataset(ds: gvl.Dataset, seq_type, return_indices: bool):
    return ds.with_seqs(seq_type).with_indices(return_indices)


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
