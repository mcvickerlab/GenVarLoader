from pathlib import Path

import hypothesis.extra.pandas as st_pd
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pysam
from hypothesis import given
from pytest_cases import fixture

import genvarloader
import genvarloader.loaders as gvl


@fixture
def data_dir():
    return Path(genvarloader.__file__).parent / "tests" / "data"


@fixture
def coverage(data_dir: Path):
    return gvl.Coverage(data_dir / "coverage.zarr")


def strategy_coverage_queries(coverage: gvl.Coverage):
    longest_contig = max(coverage.contig_lengths.values())
    contig = st_pd.column(
        name="contig", elements=st.sampled_from(list(coverage.tstores.keys()))  # type: ignore
    )
    start = st_pd.column(name="start", elements=st.integers(0, longest_contig + 1))  # type: ignore
    strand = st_pd.column(name="strand", elements=st.sampled_from(["+", "-"]))  # type: ignore
    sample = st_pd.column(
        name="sample", elements=st.sampled_from(list(coverage.samples))  # type: ignore
    )
    ploid_idx = st_pd.column(name="ploid_idx", elements=st.integers(0, 1))  # type: ignore
    df = st_pd.data_frames(columns=[contig, start, strand, sample, ploid_idx])
    return df.map(gvl.Queries)


@given(
    queries=strategy_coverage_queries(coverage(data_dir())),
    length=st.integers(600, 1200),
)
def test_coverage(
    coverage: gvl.Coverage, queries: pd.DataFrame, length: int, data_dir: Path
):
    for (sample, sample_queries) in queries.groupby("sample"):
        # (n l)
        covs = coverage.sel(sample_queries, length)  # type: ignore
        with pysam.AlignmentFile(str(data_dir / f"{sample}.bam")) as bam:
            for i, query in enumerate(sample_queries.itertuples()):
                end = query.start + length
                bam_cov = np.stack(
                    bam.count_coverage(
                        query.contig, query.start, end, read_callback="all"
                    ),
                    axis=1,
                ).sum(1)
                assert np.testing.assert_equal(covs[i], bam_cov)
