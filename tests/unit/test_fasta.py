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


from pathlib import Path

import genvarloader._fasta_cache as fc


def test_fasta_cache_creates_gvlfa_dir(tmp_path, ref_fasta):
    import shutil

    local = tmp_path / Path(ref_fasta).name
    shutil.copy(ref_fasta, local)
    for ext in (".fai", ".gzi"):
        side = Path(str(ref_fasta) + ext)
        if side.exists():
            shutil.copy(side, tmp_path / side.name)

    fasta = Fasta("ref", local, pad="N", in_memory=True, cache=True)
    gvlfa = local.with_name(local.name + ".gvlfa")
    assert gvlfa.is_dir()
    assert (gvlfa / "sequence.bin").exists()
    # Sequence served from cache matches a direct pysam read.
    contig = next(iter(fasta.contigs))
    seq = fasta.read(contig, 0, 8)
    with FastaFile(str(local)) as f:
        expected = np.frombuffer(f.fetch(contig, 0, 8).encode("ascii").upper(), "S1")
    np.testing.assert_array_equal(seq, expected)


def test_fasta_accepts_gvlfa_directly_in_memory(tmp_path, ref_fasta):
    import shutil

    local = tmp_path / Path(ref_fasta).name
    shutil.copy(ref_fasta, local)
    for ext in (".fai", ".gzi"):
        side = Path(str(ref_fasta) + ext)
        if side.exists():
            shutil.copy(side, tmp_path / side.name)
    meta, _ = fc.ensure_cache(local)
    gvlfa = local.with_name(local.name + ".gvlfa")

    fasta = Fasta("ref", gvlfa, pad="N", in_memory=True)
    assert dict(fasta.contigs) == meta.contigs
    contig = next(iter(fasta.contigs))
    seq = fasta.read(contig, 0, 8)
    with FastaFile(str(local)) as f:
        expected = np.frombuffer(f.fetch(contig, 0, 8).encode("ascii").upper(), "S1")
    np.testing.assert_array_equal(seq, expected)


def test_fasta_gvlfa_missing_source_errors_on_ondemand_read(tmp_path):
    src = tmp_path / "tiny.fa"
    src.write_text(">chr1\nACGTACGT\n")
    FastaFile(str(src)).close()  # triggers .fai creation via pysam.faidx below
    import pysam

    pysam.faidx(str(src))
    fc.ensure_cache(src)
    gvlfa = src.with_name(src.name + ".gvlfa")
    src.unlink()
    (tmp_path / "tiny.fa.fai").unlink(missing_ok=True)

    with pytest.warns(UserWarning, match="Could not locate source FASTA"):
        fasta = Fasta("ref", gvlfa, pad="N", in_memory=False)
    with pytest.raises(FileNotFoundError, match="could not be located"):
        fasta.read("chr1", 0, 4)


def test_fasta_migrates_legacy_gvl(tmp_path):
    src = tmp_path / "tiny.fa"
    src.write_text(">chr1\nACGTACGT\n")
    import pysam

    pysam.faidx(str(src))
    # Stage bytes into a legacy flat .gvl next to the source.
    staging = tmp_path / "staging.gvlfa"
    fc.build(src, staging)
    import shutil

    legacy = src.with_name(src.name + ".gvl")
    shutil.move(str(staging / "sequence.bin"), str(legacy))
    shutil.rmtree(staging)

    fasta = Fasta("ref", src, pad="N", in_memory=True, cache=True)
    assert not legacy.exists()
    assert src.with_name(src.name + ".gvlfa").is_dir()
    np.testing.assert_array_equal(
        fasta.read("chr1", 0, 8), np.frombuffer(b"ACGTACGT", "S1")
    )
