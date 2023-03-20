from pathlib import Path

import hypothesis.extra.pandas as st_pd
import hypothesis.strategies as st
import numpy as np
import pysam
from hypothesis import given
from pytest_cases import fixture

import genvarloader
import genvarloader.loaders as gvl


@fixture
def wdir():
    return Path(genvarloader.__file__).parent / "tests"


def strategy_coverage_query(coverage: gvl.Coverage):
    longest_contig = max(coverage.contig_lengths.values())
    contig = st_pd.column(
        name="contig", elements=st.sampled_from(list(coverage.tstores.keys()))
    )
    start = st_pd.column(name="start", elements=st.integers(0, longest_contig + 1))
    strand = st_pd.column(name="strand", elements=st.sampled_from(["+", "-"]))
    sample = st_pd.column(
        name="sample", elements=st.sampled_from(list(coverage.samples))
    )
    ploid_idx = st_pd.column(name="ploid_idx", elements=st.integers(0, 1))
    df = st_pd.data_frames(columns=[contig, start, strand, sample, ploid_idx])
    return df.map(gvl.Queries)


def strategy_length():
    return st.integers(600, 1200).filter(lambda x: x % 2 == 0)


@fixture
def coverage(wdir: Path):
    return gvl.Coverage(wdir / "data" / "coverage.zarr")


@given(queries=strategy_coverage_query(coverage(wdir())), length=strategy_length())
def test_coverage(
    coverage: gvl.Coverage, queries: gvl.Queries, length: int, wdir: Path
):
    # (n l)
    covs = coverage.sel(queries, length)
    with pysam.AlignmentFile(str(wdir / "data" / "coverage.bam")) as bam:
        for i, query in enumerate(queries.itertuples()):
            end = query.start + length
            bam_cov = np.stack(
                bam.count_coverage(query.contig, query.start, end, read_callback="all"),
                axis=1,
            ).sum(1)
            assert np.testing.assert_equal(covs[i], bam_cov)
