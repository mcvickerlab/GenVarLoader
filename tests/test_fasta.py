from pathlib import Path

import numpy as np
import pytest
from pytest_cases import fixture

import genvarloader as gvl
from genvarloader.fasta import NoPadError


@fixture
def fasta_path():
    return (
        Path(__file__).parent
        / "data"
        / "fasta"
        / "Homo_sapiens.GRCh38.dna.toplevel.fa.gz"
    )


def test_pad_right(fasta_path):
    fasta = gvl.Fasta("ref", fasta_path, pad="N")
    end_of_contig_1 = 248956422
    contig = "1"
    start = end_of_contig_1 - 5
    end = start + 10
    seq = fasta.read(contig, start, end).to_numpy()

    assert len(seq) == 10
    np.testing.assert_equal(seq[5:], np.full(5, b"N", dtype="S1"))


def test_pad_left(fasta_path):
    fasta = gvl.Fasta("ref", fasta_path, pad="N")
    contig = "1"
    start = -5
    end = start + 10
    seq = fasta.read(contig, start, end).to_numpy()

    assert len(seq) == 10
    np.testing.assert_equal(seq[:5], np.full(5, b"N", dtype="S1"))


def test_no_pad(fasta_path):
    fasta = gvl.Fasta("ref", fasta_path)
    end_of_contig_1 = 248956422
    contig = "1"
    start = end_of_contig_1 - 5
    end = start + 10
    with pytest.raises(NoPadError):
        fasta.read(contig, start, end).to_numpy()

    contig = "1"
    start = -5
    end = start + 10
    with pytest.raises(NoPadError):
        fasta.read(contig, start, end).to_numpy()
