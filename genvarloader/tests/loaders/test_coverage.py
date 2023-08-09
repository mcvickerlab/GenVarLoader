from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest_cases as pt

import genvarloader
import genvarloader.loaders as gvl


@pt.fixture
def data_dir():
    return Path(genvarloader.__file__).parent / "tests" / "data"


@pt.fixture
def coverage(data_dir: Path):
    return gvl.Coverage(data_dir / "coverage.zarr")


def queries_has_coverage(data_dir: Path):
    queries = gvl.read_queries(data_dir / "bam" / "2k_peaks_20_21.tsv")
    # get 1 query from each contig-sample combination
    queries = queries.groupby(["contig", "sample"]).sample(1, random_state=0)
    return queries


def queries_no_coverage(data_dir: Path):
    raise NotImplementedError


@pt.case(tags="xfail")
def queries_nonexistent_sample():
    queries = gvl.Queries(
        {
            "contig": ["20"],
            "start": [188838],
            "sample": ["definitely_not_a_real_sample"],
        }
    )
    return queries


def queries_negative_start():
    queries = gvl.Queries({"contig": ["20"], "start": [-1], "sample": ["MCF10A"]})
    return queries


def queries_out_of_bounds_end(coverage: gvl.Coverage):
    longest_contig = max(coverage.contig_lengths.values())
    queries = gvl.Queries(
        {"contig": ["20"], "start": [longest_contig + 1], "sample": ["MCF10A"]}
    )
    return queries


@pt.parametrize_with_cases("queries", prefix="queries_")
@pt.parametrize_with_cases("length", [1, 600])
def test_coverage(
    coverage: gvl.Coverage,
    queries: pd.DataFrame,
    length: int,
    data_dir: Path,
    current_cases,
):
    queries_id, queries_fn, queries_params = current_cases["queries"]
    xfail = pt.matches_tag_query(queries_fn, has_tag="xfail")
    for sample, sample_queries in queries.groupby("sample"):
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
