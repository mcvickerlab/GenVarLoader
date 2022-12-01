from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from genome_loader.load_data import load_vcf, load_vcf_polars


# Input files and preparsed vcf DF
@pytest.fixture()
def test_vcf_file():
    data_path = Path(__file__).parent / "data"
    vcf_file = str(data_path / "vcf" / "NA12878_chr20_21.vcf.gz")
    return vcf_file


@pytest.fixture(scope="module")
def test_vcf_df():
    data_path = Path(__file__).parent / "data"
    tsv_file = str(data_path / "vcf" / "NA12878_chr20_21.tsv")
    dtype_dict = {
        "start": np.uint32,
        "stop": np.uint32,
        "phase1": np.uint8,
        "phase2": np.uint8,
    }

    return pd.read_csv(tsv_file, sep="\t", dtype=dtype_dict)


@pytest.fixture(scope="module")
def test_vcf_df_polars():
    data_path = Path(__file__).parent / "data"
    tsv_file = str(data_path / "vcf" / "NA12878_chr20_21.tsv")

    dtype_dict = {
        "chrom": pl.Utf8,
        "start": pl.UInt32,
        "stop": pl.UInt32,
        "ref": pl.Utf8,
        "alt": pl.Utf8,
        "phase1": pl.UInt8,
        "phase2": pl.UInt8,
    }

    return pl.read_csv(tsv_file, sep="\t", dtypes=dtype_dict)


# Test Pandas based VCF Loader
@pytest.mark.parametrize(
    "chrom_inputs, expected_chroms",
    [(None, ["chr20", "chr21"]), (["chr20"], ["chr20"]), ("chr21", ["chr21"])],
)
def test_load_sample(chrom_inputs, expected_chroms, test_vcf_file, test_vcf_df):
    load_vcf_df = load_vcf(test_vcf_file, chrom_list=chrom_inputs, sample="NA12878")
    assert (
        test_vcf_df.loc[test_vcf_df["chrom"].isin(expected_chroms), :]
        .reset_index(drop=True)
        .equals(load_vcf_df)
    )


@pytest.mark.parametrize(
    "chrom_inputs, expected_chroms",
    [(None, ["chr20", "chr21"]), (["chr21"], ["chr21"]), ("chr20", ["chr20"])],
)
def test_load_no_sample(chrom_inputs, expected_chroms, test_vcf_file, test_vcf_df):
    load_vcf_df = load_vcf(test_vcf_file, chrom_list=chrom_inputs)
    assert (
        test_vcf_df.loc[
            test_vcf_df["chrom"].isin(expected_chroms),
            ["chrom", "start", "stop", "ref", "alt"],
        ]
        .reset_index(drop=True)
        .equals(load_vcf_df)
    )


# Test Polars based VCF Loader
@pytest.mark.parametrize(
    "chrom_inputs, expected_chroms",
    [(None, ["chr20", "chr21"]), (["chr21"], ["chr21"]), ("chr20", ["chr20"])],
)
def test_load_sample_polars(
    chrom_inputs, expected_chroms, test_vcf_file, test_vcf_df_polars
):
    load_vcf_df = load_vcf_polars(
        test_vcf_file, chrom_list=chrom_inputs, sample="NA12878"
    )
    assert test_vcf_df_polars.filter(
        pl.col("chrom").is_in(expected_chroms)
    ).frame_equal(load_vcf_df)


@pytest.mark.parametrize(
    "chrom_inputs, expected_chroms",
    [(None, ["chr20", "chr21"]), (["chr20"], ["chr20"]), ("chr21", ["chr21"])],
)
def test_load_no_sample_polars(
    chrom_inputs, expected_chroms, test_vcf_file, test_vcf_df_polars
):
    load_vcf_df = load_vcf_polars(test_vcf_file, chrom_list=chrom_inputs)
    assert (
        test_vcf_df_polars.filter(pl.col("chrom").is_in(expected_chroms))
        .select(["chrom", "start", "stop", "ref", "alt"])
        .frame_equal(load_vcf_df)
    )
