from pathlib import Path

import awkward as ak
import genvarloader as gvl
import numpy as np
import polars as pl
import seqpro as sp
from genoray import PGEN, VCF, Reader
from seqpro.rag import Ragged
from genvarloader._utils import lengths_to_offsets
from polars.testing.asserts import assert_frame_equal
from pytest import fixture, mark
from pytest_cases import parametrize_with_cases


def reader_vcf(vcf_dir):
    vcf = VCF(vcf_dir / "filtered_sample.vcf.gz")
    vcf._write_gvi_index()
    vcf._load_index()
    return vcf


def reader_pgen(pgen_dir):
    index_path = pgen_dir / "filtered_sample.pvar.gvi"
    index_path.unlink()
    pgen = PGEN(pgen_dir / "filtered_sample.pgen")
    return pgen


@fixture
def bed(vcf_dir: Path):
    return sp.bed.read(vcf_dir / "sample.bed")


@fixture
def ref(ref_fasta: Path):
    return ref_fasta


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
    actual = Ragged.from_offsets(var_idxs, (*shape, None), offsets).to_ak()

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
    offsets = lengths_to_offsets(lengths)
    shape = (8, 3, 2)
    desired = Ragged.from_offsets(var_idxs, (*shape, None), offsets).to_ak()

    max_len = lengths.max()
    for len_ in range(1, max_len + 1):
        mask = ak.num(desired, -1) == len_
        assert ak.all(actual[mask][:, :len_] == desired[mask][:, :len_])  # type: ignore


def test_write_errors_when_post_index_budget_too_small(
    tmp_path, monkeypatch, vcf_dir: Path, source_bed: Path
):
    """If max_mem minus the variant index leaves no room for even one
    variant chunk, gvl.write raises ValueError instead of silently
    blowing the budget."""
    import pytest

    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    vcf._write_gvi_index()
    vcf._load_index()

    bed = sp.bed.read(source_bed)

    # Force nbytes large enough that effective_max_mem < bytes_per_var.
    # bytes_per_var = n_samples * ploidy (VCF, Genos8 = 1 byte).
    # Set nbytes = max_mem so effective_max_mem == 0.
    max_mem = 4 * 1024 * 1024
    monkeypatch.setattr(type(vcf), "nbytes", property(lambda self: max_mem))

    out = tmp_path / "test.gvl"
    with pytest.raises(ValueError, match="max_mem"):
        gvl.write(out, bed, vcf, max_mem=max_mem)


def test_write_loads_lazy_vcf_index(tmp_path, vcf_dir: Path, source_bed: Path):
    """gvl.write should load the index itself when given a VCF constructed
    with with_gvi_index=False, and produce a valid dataset."""
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz", with_gvi_index=False)
    assert vcf._index is None

    bed = sp.bed.read(source_bed)
    out = tmp_path / "test.gvl"
    gvl.write(out, bed, vcf)

    assert (out / "metadata.json").exists()
    assert (out / "genotypes" / "variants.arrow").exists()


def test_write_loads_lazy_pgen_index(tmp_path, pgen_dir: Path, source_bed: Path):
    """gvl.write should load the index itself when given a PGEN constructed
    with load_index=False, and produce a valid dataset."""
    pgen = PGEN(pgen_dir / "filtered_source.pgen", load_index=False)
    assert pgen._index is None

    bed = sp.bed.read(source_bed)
    out = tmp_path / "test.gvl"
    gvl.write(out, bed, pgen)

    assert (out / "metadata.json").exists()
    assert (out / "genotypes" / "variants.arrow").exists()
