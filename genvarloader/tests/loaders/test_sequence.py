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
def data_dir():
    return Path(genvarloader.__file__).parent / "tests" / "data"


@fixture
def sequence(dat_dir: Path):
    return gvl.Sequence(dat_dir / "grch38.20.21.zarr")


def strategy_sequence_queries(sequence: gvl.Sequence):
    longest_contig = max(sequence.contig_lengths.values())
    contig = st_pd.column(
        name="contig", elements=st.sampled_from(list(sequence.tstores.keys()))  # type: ignore
    )
    start = st_pd.column(name="start", elements=st.integers(0, longest_contig + 1))  # type: ignore
    strand = st_pd.column(name="strand", elements=st.sampled_from(["+", "-"]))  # type: ignore
    df = st_pd.data_frames(columns=[contig, start, strand])
    return df.map(gvl.Queries)


@given(
    queries=strategy_sequence_queries(sequence(data_dir())),
    length=st.integers(600, 1200),
)
def test_sequence(
    sequence: gvl.Sequence, queries: gvl.Queries, length: int, data_dir: Path
):
    seqs = sequence.sel(queries, length, encoding="bytes").astype("U")
    ref_fasta = data_dir / "fasta" / "grch38.20.21.fa.gz"

    for seq, query in zip(seqs, queries.itertuples()):
        contig = query.contig
        start = query.start
        with FastaFile(str(ref_fasta)) as f:
            ref_seq = f.fetch(contig, start, start + length)
        assert ref_seq == "".join(seq)
