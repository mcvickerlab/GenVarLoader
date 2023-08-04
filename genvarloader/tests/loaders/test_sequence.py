from pathlib import Path

import pandas as pd
import pytest_cases as pt
from pysam import FastaFile

import genvarloader
import genvarloader.loaders as gvl


@pt.fixture
def data_dir():
    return Path(genvarloader.__file__).parent / "tests" / "data"


@pt.fixture
def sequence(dat_dir: Path):
    return gvl.Sequence(dat_dir / "grch38.20.21.zarr")


def queries_multiple_contigs(sequence: gvl.Sequence):
    contigs = list(sequence.contig_lengths.keys())
    multiple_contigs = gvl.Queries(
        {
            "contig": contigs[:2],
            "start": [0, 0],
            "strand": ["+", "+"],
        }
    )
    return multiple_contigs


def queries_rev_comp(sequence: gvl.Sequence):
    contigs = list(sequence.contig_lengths.keys())
    rev_comp = gvl.Queries(
        {
            "contig": [contigs[0]],
            "start": [0],
            "strand": ["-"],
        }
    )
    return rev_comp


def queries_negative_start(sequence: gvl.Sequence):
    contigs = list(sequence.contig_lengths.keys())
    negative_start = gvl.Queries(
        {
            "contig": [contigs[0]],
            "start": [-1],
            "strand": ["+"],
        }
    )
    return negative_start


def queries_out_of_bounds_end(sequence: gvl.Sequence):
    contigs = list(sequence.contig_lengths.keys())
    longest_contig = max(sequence.contig_lengths.values())
    out_of_bounds_end = gvl.Queries(
        {
            "contig": [contigs[0]],
            "start": [longest_contig + 1],
            "strand": ["+"],
        }
    )
    return out_of_bounds_end


def queries_all(sequence: gvl.Sequence):
    contigs = list(sequence.contig_lengths.keys())
    longest_contig = max(sequence.contig_lengths.values())
    q_all = gvl.Queries(
        {
            "contig": contigs[:3],
            "start": [0, -1, longest_contig + 1],
            "strand": ["+", "-", "."],
        }
    )
    return q_all


@pt.parametrize_with_cases("queries", prefix="queries_")
@pt.parametrize_with_cases("length", [1, 600])
def test_sequence(
    sequence: gvl.Sequence,
    queries: pd.DataFrame,
    length: int,
    data_dir: Path,
    current_cases,
):
    queries_id, queries_fn, queries_params = current_cases["queries"]
    xfail = pt.matches_tag_query(queries_fn, has_tag="xfail")
    seqs = sequence.sel(queries, length, encoding="bytes").astype("U")
    ref_fasta = data_dir / "fasta" / "grch38.20.21.fa.gz"

    for seq, query in zip(seqs, queries.itertuples()):
        contig = query.contig
        start = query.start
        with FastaFile(str(ref_fasta)) as f:
            ref_seq = f.fetch(contig, start, start + length)
        assert ref_seq == "".join(seq)
