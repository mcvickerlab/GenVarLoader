from pathlib import Path

import numpy as np
import pytest
from genvarloader._fasta import Fasta, NoPadError
from pysam import FastaFile
from pytest import fixture


@fixture
def fasta_path():
    return (
        Path(__file__).parent
        / "data"
        / "fasta"
        / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.bgz"
    )


def test_pad_right(fasta_path):
    fasta = Fasta("ref", fasta_path, pad="N")
    contig = "1"
    with FastaFile(fasta_path) as f:
        end_of_contig = f.get_reference_length(contig)
        start = end_of_contig - 5
        end = start + 10
        desired = np.full(end - start, b"N", "S1")
        desired[:5] = np.frombuffer(
            f.fetch(contig, start, start + 5).encode("ascii"), "S1"
        )

    seq = fasta.read(contig, start, end)

    np.testing.assert_equal(seq, desired)


def test_pad_left(fasta_path):
    fasta = Fasta("ref", fasta_path, pad="N")
    contig = "1"
    start = -5
    end = start + 10
    seq = fasta.read(contig, start, end)
    with FastaFile(fasta_path) as f:
        desired = np.full(end - start, b"N", "S1")
        desired[5:] = np.frombuffer(f.fetch(contig, 0, end).encode("ascii"), "S1")

    np.testing.assert_equal(seq, desired)


def test_no_pad(fasta_path):
    fasta = Fasta("ref", fasta_path)
    end_of_contig_1 = 248956422
    contig = "1"
    start = end_of_contig_1 - 5
    end = start + 10
    with pytest.raises(NoPadError):
        fasta.read(contig, start, end)

    contig = "1"
    start = -5
    end = start + 10
    with pytest.raises(NoPadError):
        fasta.read(contig, start, end)
