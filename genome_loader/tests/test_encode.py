import sys
from pathlib import Path

import numpy as np

import pytest
from pytest_cases import fixture, parametrize, parametrize_with_cases


from genome_loader.encode_data import parse_encode_list, encode_sequence, encode_from_fasta


# Test parse specs
@pytest.mark.encode_spec
@pytest.mark.parametrize("spec_input, expected_spec",
                         [(None, [b"A", b"C", b"G", b"T", b"N"]),
                          (["T", "G", "A", "C"], [b"T", b"G", b"A", b"C"]),
                          ("CGNAT", [b"C", b"G", b"N", b"A", b"T"])])
def test_parse_encode_list(spec_input, expected_spec):
    assert parse_encode_list(spec_input) == expected_spec


# Test parse encode
def acgtn_ohe():
    return np.array([
        [1, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0], 
        [0, 0, 1, 0, 0], 
        [0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 1]])

def acgtn_ohe_no_n():
    return np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 0]])

def acgtn_ohe_only_n():
    return np.array([
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1]])

@pytest.mark.encode_sequence
@pytest.mark.parametrize(
    "in_seq, out_ohe, encode_spec, ignore_case", [
        ("ACGTN", acgtn_ohe(), None, False),
        ("aCgTN", acgtn_ohe(), None, True), 
        ("ACGTN", acgtn_ohe_no_n(), "ACGT", False),
        ("AcGtN", acgtn_ohe_no_n(), "ACGT", True),
        ("acgtN", acgtn_ohe_only_n(), "ACGTN", False),
        (np.array(["A", "C", "G", "T", "N"]), acgtn_ohe(), None, False),
        (np.array(["a", "c", "g", "t", "N"]), acgtn_ohe_no_n(), "ACGT", True)
      ])
def test_encode_sequence(in_seq, out_ohe, encode_spec, ignore_case):
    assert np.array_equal(encode_sequence(in_seq, encode_spec=encode_spec, ignore_case=ignore_case), out_ohe)


# Test Encode from FASTA
# NEED TO CHANGE THIS TO OHE
def grch38_20_59997_60002():
    # NNNTG
    return np.array([
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 1, 0], 
        [0, 0, 1, 0, 0]])

def grch38_20_50000000_50000005():
    # TGACC
    return np.array([
        [0, 0, 0, 1, 0], 
        [0, 0, 1, 0, 0], 
        [1, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0], 
        [0, 1, 0, 0, 0]])

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
    assert (encode_dict_grch38["20"].shape == (64444167, 5)) and (encode_dict_grch38["21"].shape == (46709983, 5))

@pytest.mark.encode_fasta
@pytest.mark.parametrize(
    "chrom, start, stop, out_ohe", [
        ("20", 59997, 60002, grch38_20_59997_60002()),
        ("20", 50000000, 50000005, grch38_20_50000000_50000005())
    ])
def test_encode_from_fasta(chrom, start, stop, out_ohe, encode_dict_grch38):
    assert np.array_equal(encode_dict_grch38[chrom][start:stop,], out_ohe)

