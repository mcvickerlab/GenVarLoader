import numpy as np
import pytest
from genome_loader.utils import bytes_to_ohe, ohe_to_bytes, read_bed2
from pytest_cases import fixture, parametrize, parametrize_with_cases


def alpha_acgt():
    return np.frombuffer(b"ACGT", dtype="S1")


def alpha_acgtn():
    return np.frombuffer(b"ACGTN", dtype="S1")


@fixture
@parametrize_with_cases("alphabet", cases=".", prefix="alpha_")
@parametrize(shape=[(), (4, 2)])
def byte_arr_and_alphabet(shape, alphabet):
    byte_arr = np.tile(alphabet, (*shape, 2))
    return byte_arr, alphabet


def test_bytes_to_ohe_and_back(byte_arr_and_alphabet):
    byte_arr, alphabet = byte_arr_and_alphabet
    assert (byte_arr == ohe_to_bytes(bytes_to_ohe(byte_arr, alphabet), alphabet)).all()


@fixture
def bed_file():
    return "test.bed"


def test_read_bed2(bed_file):
    bed = read_bed2(bed_file)
    assert (
        bed["chrom"].to_numpy() == np.array(["21", "20", "20"], dtype=np.dtype("U"))
    ).all()
    assert (
        bed["start"].to_numpy() == np.array([10414881, 96319, 279175], dtype=np.uint64)
    ).all()
