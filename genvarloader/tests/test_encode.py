import sys
from pathlib import Path

import numpy as np
import pytest
from pytest_cases import fixture, parametrize, parametrize_with_cases

from genvarloader.encode_data import (
    encode_from_fasta,
    encode_sequence,
    parse_encode_spec,
)


@pytest.mark.encode_spec
@pytest.mark.parametrize(
    "spec_input, expected_spec",
    [
        (None, {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}),
        (["T", "G", "A", "C"], {"T": 0, "G": 1, "A": 2, "C": 3}),
        ("CGNAT", {"C": 0, "G": 1, "N": 2, "A": 3, "T": 4}),
    ],
)
def test_parse_encode_list(spec_input, expected_spec):
    assert parse_encode_spec(spec_input) == expected_spec


# Test parse encode
def acgtn_ohe():
    return np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )


def acgtn_ohe_no_n():
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    )


def acgtn_ohe_only_n():
    return np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ]
    )


def acgt_ohe_default_cols():
    return np.array(
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
    )


@pytest.mark.encode_sequence
@pytest.mark.parametrize(
    "in_seq, encoded_bases, encode_spec, ignore_case, out_ohe",
    [
        ("ACGTN", None, None, False, acgtn_ohe()),
        ("aCgTN", None, None, True, acgtn_ohe()),
        ("ACGTN", None, "ACGT", False, acgtn_ohe_no_n()),
        ("AcGtN", None, "ACGT", True, acgtn_ohe_no_n()),
        ("acgtN", None, "ACGTN", False, acgtn_ohe_only_n()),
        ("ACGT", None, None, True, acgt_ohe_default_cols()),
        (np.array(["A", "C", "G", "T", "N"]), None, None, False, acgtn_ohe()),
        (np.array(["a", "c", "g", "t", "N"]), None, "ACGT", True, acgtn_ohe_no_n()),
    ],
)
def test_encode_sequence_pandas(
    in_seq, encoded_bases, encode_spec, ignore_case, out_ohe
):
    assert np.array_equal(
        encode_sequence(
            in_seq,
            encoded_bases=encoded_bases,
            encode_spec=encode_spec,
            ignore_case=ignore_case,
            engine="pandas",
        ),
        out_ohe,
    )


# TODO WRITE TESTS FOR BOTH PANDAS AND POLAR ENCODING ENGINES/ as well as parity check
@pytest.mark.encode_sequence
@pytest.mark.parametrize(
    "in_seq, encoded_bases, encode_spec, ignore_case, out_ohe",
    [
        ("ACGTN", None, None, False, acgtn_ohe()),
        ("aCgTN", None, None, True, acgtn_ohe()),
        ("ACGTN", None, "ACGT", False, acgtn_ohe_no_n()),
        ("AcGtN", None, "ACGT", True, acgtn_ohe_no_n()),
        ("acgtN", None, "ACGTN", False, acgtn_ohe_only_n()),
        ("ACGT", None, None, True, acgt_ohe_default_cols()),
    ],
)
def test_encode_sequence_polars(
    in_seq, encoded_bases, encode_spec, ignore_case, out_ohe
):
    assert np.array_equal(
        encode_sequence(
            in_seq,
            encoded_bases=encoded_bases,
            encode_spec=encode_spec,
            ignore_case=ignore_case,
            engine="polars",
        ),
        out_ohe,
    )


# Test Encode from FASTA
def grch38_20_59997_60002():
    # NNNTG
    return np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ]
    )


def grch38_20_50000000_50000005():
    # TGACC
    return np.array(
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]
    )


@pytest.fixture(scope="module")
def encode_dict_grch38():
    data_path = Path(__file__).parent / "data"
    fasta_file = str(data_path / "fasta" / "grch38.20.21.fa.gz")
    encode_dict = encode_from_fasta(fasta_file)
    return encode_dict


@pytest.mark.encode_fasta
def test_encoded_fasta_keys(encode_dict_grch38):
    assert ("20" in encode_dict_grch38.keys()) and ("21" in encode_dict_grch38.keys())


@pytest.mark.encode_fasta
def test_encoded_fasta_shape(encode_dict_grch38):
    assert (encode_dict_grch38["20"].shape == (64444167, 5)) and (
        encode_dict_grch38["21"].shape == (46709983, 5)
    )


@pytest.mark.encode_fasta
@pytest.mark.parametrize(
    "chrom, start, stop, out_ohe",
    [
        ("20", 59997, 60002, grch38_20_59997_60002()),
        ("20", 50000000, 50000005, grch38_20_50000000_50000005()),
    ],
)
def test_encode_from_fasta(chrom, start, stop, out_ohe, encode_dict_grch38):
    assert np.array_equal(
        encode_dict_grch38[chrom][
            start:stop,
        ],
        out_ohe,
    )
