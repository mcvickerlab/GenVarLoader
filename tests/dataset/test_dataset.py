from pathlib import Path

import genvarloader as gvl
import numpy as np
from pytest_cases import parametrize_with_cases

data_dir = Path(__file__).resolve().parents[1] / "data"


def ds_phased():
    return gvl.Dataset.open(
        data_dir / "phased_dataset.gvl",
        data_dir / "fasta" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.bgz",
    )


def ds_unphased():
    return gvl.Dataset.open(
        data_dir / "unphased_dataset.gvl",
        data_dir / "fasta" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.bgz",
    )


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


@parametrize_with_cases("ds", prefix="ds_", cases=".")
@parametrize_with_cases("idx", prefix="idx_", cases=".")
def test_ds_indexing(ds, idx):
    ds[idx]


@parametrize_with_cases("ds", prefix="ds_", cases=".")
@parametrize_with_cases("r_idx", prefix="idx_", cases=".")
@parametrize_with_cases("s_idx", prefix="idx_", cases=".")
def test_rs_indexing(ds, r_idx, s_idx):
    ds[r_idx, s_idx]
