from typing import Tuple

import numpy as np
import polars as pl
import zarr
from pytest_cases import fixture, parametrize, parametrize_with_cases

from genvarloader.types import ALPHABETS, SequenceAlphabet
from genvarloader.utils import bytes_to_ohe, ohe_to_bytes


def alphabet_DNA():
    return ALPHABETS["DNA"]


def alphabet_RNA():
    return ALPHABETS["RNA"]


@fixture
@parametrize_with_cases("alphabet", cases=".", prefix="alphabet_")
@parametrize(shape=[(), (4, 2)])
def byte_arr_and_alphabet(shape: Tuple[int, ...], alphabet: SequenceAlphabet):
    byte_arr = np.tile(alphabet.array, (*shape, 2))
    return byte_arr, alphabet


def test_bytes_to_ohe_and_back(byte_arr_and_alphabet):
    byte_arr, alphabet = byte_arr_and_alphabet
    assert (byte_arr == ohe_to_bytes(bytes_to_ohe(byte_arr, alphabet), alphabet)).all()
