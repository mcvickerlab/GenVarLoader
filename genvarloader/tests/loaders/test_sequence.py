from pathlib import Path
from typing import Dict

import hypothesis.extra.pandas as st_pd
import hypothesis.strategies as st
from hypothesis import given
from pysam import FastaFile
from pytest_cases import fixture

import genvarloader
import genvarloader.loaders as gvl


@fixture
def wdir():
    return Path(genvarloader.__file__).parent / "tests"


@fixture
def sequence(wdir: Path):
    return gvl.Sequence(wdir / "data" / "grch38.20.21.zarr")


def strategy_sequence_queries(sequence: gvl.Sequence):
    longest_contig = max(sequence.contig_lengths.values())
    contig = st_pd.column(
        name="contig", elements=st.sampled_from(list(sequence.tstores.keys()))
    )
    start = st_pd.column(name="start", elements=st.integers(0, longest_contig + 1))
    strand = st_pd.column(name="strand", elements=st.sampled_from(["+", "-"]))
    df = st_pd.data_frames(columns=[contig, start, strand])
    return df.map(gvl.Queries)


def strategy_length():
    return st.integers(600, 1200).filter(lambda x: x % 2 == 0)


@given(queries=strategy_sequence_queries(sequence(wdir())))
def test_zarrsequence(sequence: gvl.Sequence, sel_args: Dict, wdir: Path):
    seqs = sequence.sel(**sel_args).astype("U")
    ref_fasta = wdir / "data" / "fasta" / "grch38.20.21.fa.gz"

    for seq, query in zip(seqs, sel_args["queries"].itertuples()):
        contig = query.contig
        start = query.start
        length = sel_args["length"]
        with FastaFile(str(ref_fasta)) as f:
            ref_seq = f.fetch(contig, start, start + length)
        assert ref_seq == "".join(seq)
