import re
from pathlib import Path
from typing import Dict

import hypothesis.extra.pandas as st_pd
import hypothesis.strategies as st
import numpy as np
import pandas as pd
from cyvcf2 import VCF
from hypothesis import given
from pytest_cases import fixture

import genvarloader
import genvarloader.loaders as gvl


@fixture
def data_dir():
    return Path(genvarloader.__file__).parent / "tests" / "data"


@fixture
def sequence(data_dir: Path):
    return gvl.Sequence(data_dir / "grch38.20.21.zarr")


@fixture
def variants(data_dir: Path):
    zarrs = [
        data_dir / "CDS-OJhAUD_cnn_filtered.zarr",
        data_dir / "CDS-oZPNvc_cnn_filtered.zarr",
    ]
    sample_ids = ["OCI-AML5", "NCI-H660"]
    return gvl.Variants.create(zarrs=zarrs, sample_ids=sample_ids)


@fixture
def varseq(sequence: gvl.Sequence, variants: gvl.Variants):
    return gvl.VarSequence(sequence, variants)


def strategy_varseq_query(varseq: gvl.VarSequence):
    longest_contig = max(varseq.sequence.contig_lengths.values())
    contig = st_pd.column(
        name="contig", elements=st.sampled_from(list(varseq.sequence.tstores.keys()))  # type: ignore
    )
    start = st_pd.column(name="start", elements=st.integers(0, longest_contig + 1))  # type: ignore
    strand = st_pd.column(name="strand", elements=st.sampled_from(["+", "-"]))  # type: ignore
    sample = st_pd.column(
        name="sample", elements=st.sampled_from(list(varseq.variants.samples))  # type: ignore
    )
    ploid_idx = st_pd.column(name="ploid_idx", elements=st.integers(0, 1))  # type: ignore
    df = st_pd.data_frames(columns=[contig, start, strand, sample, ploid_idx])
    return df.map(gvl.Queries)


@given(
    queries=strategy_varseq_query(varseq(sequence(data_dir()), variants(data_dir()))),
    length=st.integers(600, 1200),
)
def test_varseq(
    varseq: gvl.VarSequence, queries: pd.DataFrame, length: int, data_dir: Path
):
    raise NotImplementedError
