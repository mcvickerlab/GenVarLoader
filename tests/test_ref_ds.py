from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
import pytest
import seqpro as sp
from pytest_cases import fixture, parametrize_with_cases

DDIR = Path(__file__).parent / "data"
REF = DDIR / "fasta" / "hg38.fa.bgz"


@fixture
def reference():
    return gvl.Reference.from_path(REF, in_memory=False)


@pytest.mark.xfail(strict=True, raises=ValueError)
def case_no_regions():
    regions = pl.DataFrame(
        schema={"chrom": pl.Utf8, "chromStart": pl.Int32, "chromEnd": pl.Int32}
    )
    desired = gvl.Ragged.empty(0, np.bytes_)
    return regions, desired


def case_ragged_regions():
    """Three regions with different lengths."""
    regions = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [0, 100],
            "chromEnd": [100, 150],
        }
    )
    data = sp.cast_seqs(b"N" * 150)
    lengths = (regions["chromEnd"] - regions["chromStart"]).to_numpy().astype(np.uint32)
    desired = gvl.Ragged.from_lengths(data, lengths)
    return regions, desired


@parametrize_with_cases("regions, desired", cases=".")
def test_getitem(
    reference: gvl.Reference, regions: pl.DataFrame, desired: gvl.Ragged[np.bytes_]
):
    ds = gvl.RefDataset(reference, regions, seed=0)
    actual = ds[:]

    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.lengths, desired.lengths)
