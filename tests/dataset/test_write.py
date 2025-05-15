from pathlib import Path

import awkward as ak
import genvarloader as gvl
import numpy as np
import polars as pl
import seqpro as sp
from genoray import PGEN, VCF, Reader
from genvarloader._dataset._genotypes import SparseGenotypes
from genvarloader._utils import _lengths_to_offsets
from polars.testing.asserts import assert_frame_equal
from pytest import fixture, mark
from pytest_cases import parametrize_with_cases

ddir = Path(__file__).parents[1] / "data"


def reader_vcf():
    vcf = VCF(ddir / "vcf" / "filtered_sample.vcf.gz")
    vcf._write_gvi_index()
    vcf._load_index()
    return vcf


def reader_pgen():
    index_path = (ddir / "pgen" / "filtered_sample.pvar.gvi")
    index_path.unlink()
    pgen = PGEN(ddir / "pgen" / "filtered_sample.pgen")
    return pgen


@fixture
def bed():
    return sp.bed.read_bedlike(ddir / "vcf" / "sample.bed")


@fixture
def ref():
    return ddir / "fasta" / "hg38.fa.bgz"


@mark.skip
@parametrize_with_cases("reader", cases=".", prefix="reader_")
def test_write(reader: Reader, bed: pl.DataFrame, ref: Path, tmp_path):
    out = tmp_path / "test.gvl"
    gvl.write(out, bed, reader)

    ds = gvl.Dataset.open(out, ref)
    assert ds.shape == (bed.height, reader.n_samples)
    assert_frame_equal(ds.regions, bed, categorical_as_str=True)

    var_idxs = np.memmap(out / "genotypes" / "variant_idxs.npy", dtype=np.int32)
    offsets = np.memmap(out / "genotypes" / "offsets.npy", dtype=np.int64)
    shape = (*ds.shape, reader.ploidy)
    actual = SparseGenotypes.from_offsets(var_idxs, shape, offsets).to_awkward()

    # *                 0,
    # * 2,3,   3,3,     1,
    # * 4,5,   4,5,4,5, 6,
    # * 7,8,   9,10,7,  7,7,9,10,
    # *        11,      11,11,
    # *        12,
    # * 13,14, 14,13,   14,14,
    # * 15,    16,
    var_idxs = (
        ak.flatten(
            ak.Array(
                [
                    [0],
                    [2, 3, 3, 3, 1],
                    [4, 5, 4, 5, 4, 5, 6],
                    [7, 8, 9, 10, 7, 7, 7, 9, 10],
                    [11, 11, 11],
                    [12],
                    [13, 14, 14, 13, 14, 14],
                    [15, 16],
                ]
            ),
            -1,
        )
        .to_numpy()
        .astype(np.int32)
    )
    # (r s p)
    lengths = np.array(
        [
            [[0, 0], [0, 0], [0, 1]],
            [[1, 1], [1, 1], [0, 1]],
            [[0, 2], [2, 2], [0, 1]],
            [[1, 1], [2, 1], [1, 3]],
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [0, 1], [0, 0]],
            [[1, 1], [1, 1], [1, 1]],
            [[0, 1], [0, 1], [0, 0]],
        ]
    )
    offsets = _lengths_to_offsets(lengths)
    shape = (8, 3, 2)
    desired = SparseGenotypes.from_offsets(var_idxs, shape, offsets).to_awkward()

    max_len = lengths.max()
    for len_ in range(1, max_len + 1):
        mask = ak.num(desired, -1) == len_
        assert ak.all(actual[mask][:, :len_] == desired[mask][:, :len_])  # type: ignore
