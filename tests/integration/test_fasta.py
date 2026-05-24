import numpy as np
import pytest
from genvarloader._fasta import Fasta, NoPadError
from pysam import FastaFile


def test_pad_right(ref_fasta):
    fasta = Fasta("ref", ref_fasta, pad="N")
    contig = "chr1"
    with FastaFile(ref_fasta) as f:
        end_of_contig = f.get_reference_length(contig)
        start = end_of_contig - 5
        end = start + 10
        desired = np.full(end - start, b"N", "S1")
        desired[:5] = np.frombuffer(
            f.fetch(contig, start, start + 5).encode("ascii"), "S1"
        )

    seq = fasta.read(contig, start, end)

    np.testing.assert_equal(seq, desired)


def test_pad_left(ref_fasta):
    fasta = Fasta("ref", ref_fasta, pad="N")
    contig = "chr1"
    start = -5
    end = start + 10
    seq = fasta.read(contig, start, end)
    with FastaFile(ref_fasta) as f:
        desired = np.full(end - start, b"N", "S1")
        desired[5:] = np.frombuffer(f.fetch(contig, 0, end).encode("ascii"), "S1")

    np.testing.assert_equal(seq, desired)


def test_no_pad(ref_fasta):
    fasta = Fasta("ref", ref_fasta)
    end_of_contig_1 = 248956422
    contig = "chr1"
    start = end_of_contig_1 - 5
    end = start + 10
    with pytest.raises(NoPadError):
        fasta.read(contig, start, end)

    contig = "chr1"
    start = -5
    end = start + 10
    with pytest.raises(NoPadError):
        fasta.read(contig, start, end)
