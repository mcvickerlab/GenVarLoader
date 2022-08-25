import numpy as np
import polars as pl
import pytest
import zarr
from pytest_cases import fixture, parametrize, parametrize_with_cases

from genome_loader.utils import (
    bytes_to_ohe,
    df_to_zarr,
    ohe_to_bytes,
    read_bed,
    zarr_to_df,
)


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
    return "data/test.bed"


def test_read_bed(bed_file):
    bed = read_bed(bed_file)
    pl.testing.assert_series_equal(
        bed["chrom"],
        pl.Series("chrom", np.array(["21", "20", "20"], dtype=np.dtype("U"))),
    )
    pl.testing.assert_series_equal(
        bed["start"],
        pl.Series("start", np.array([10414881, 96319, 279175], dtype=np.int32)),
    )


@fixture
def zarr_file():
    return "data/test.zarr"


def test_df_zarr_serialization(bed_file, zarr_file):
    df = read_bed(bed_file)
    z = zarr.open(zarr_file, "w")
    df_to_zarr(df, z.create_group("bed"))

    df_out = zarr_to_df(z["bed"])
    pl.testing.assert_frame_equal(df, df_out)
