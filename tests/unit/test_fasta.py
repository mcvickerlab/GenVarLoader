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
    with FastaFile(ref_fasta) as f:
        end_of_contig_1 = f.get_reference_length("chr1")
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


def test_fasta_missing_contig_raises(ref_fasta):
    """Reading a contig not in the FASTA raises ValueError."""
    fasta = Fasta("ref", ref_fasta, pad="N")
    with pytest.raises(ValueError, match="not found"):
        fasta.read("nonexistent_contig_zzz", 0, 100)


def test_fasta_reader_protocol_attrs(ref_fasta):
    """Fasta exposes the Reader protocol surface: name, dtype, contigs."""
    fasta = Fasta("ref", ref_fasta, pad="N")
    assert fasta.name == "ref"
    assert fasta.dtype == np.dtype("S1")
    assert isinstance(fasta.contigs, dict)
    assert "chr1" in fasta.contigs


def test_fasta_zero_length_range(ref_fasta):
    """start == end returns an empty array."""
    fasta = Fasta("ref", ref_fasta, pad="N")
    seq = fasta.read("chr1", 100, 100)
    assert len(seq) == 0
