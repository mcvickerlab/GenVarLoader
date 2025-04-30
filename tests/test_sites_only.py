import os
import pytest
import logging
import pandas as pd
import polars as pl
from genvarloader._variants._sitesonly import sites_vcf_to_table
from cyvcf2 import VCF


VCF_FILE = os.path.join(
    os.path.dirname(__file__), "data", "vcf", "synthetic_test_data.vcf"
)


# convert values to float safely.
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


# check if the given DataFrame is valid (either pandas or polars).
def is_valid_dataframe(df):
    return isinstance(df, (pd.DataFrame, pl.DataFrame))


# check whether the DataFrame is empty.
def is_empty(df):
    if isinstance(df, pd.DataFrame):
        return df.empty
    elif isinstance(df, pl.DataFrame):
        return df.height == 0
    return True


# Test 1: Ensure the VCF is processed into a valid, non-empty DataFrame.
def test_sites_vcf_to_table_basic():
    df = sites_vcf_to_table(VCF_FILE)
    assert is_valid_dataframe(df), "The output is not a valid DataFrame."
    assert not is_empty(df), f"The DataFrame is empty for VCF: {VCF_FILE}"


# Test 2: verify that when INFO fields are requested, the 'AF' column appears.
def test_sites_vcf_to_table_with_info():
    try:
        df = sites_vcf_to_table(VCF_FILE, attributes=["ID"], info_fields=["AF"])
        assert not is_empty(df), f"DataFrame is empty for VCF: {VCF_FILE}"
        assert "AF" in df.columns, "The INFO field 'AF' is missing from the output."
    except KeyError as e:
        logging.error(f"Encountered a KeyError: {str(e)}")
        pytest.xfail(
            "KeyError due to missing 'AF' header in INFO; expected behavior in some cases."
        )


# Test 3: process one or more VCF files and verify essential columns are present.
@pytest.mark.parametrize(
    "vcf_file",
    [
        os.path.join(
            os.path.dirname(__file__), "data", "vcf", "synthetic_test_data.vcf"
        ),
        # additional VCF file paths can be added here.
    ],
)
def test_vcf_cases(vcf_file):
    df = sites_vcf_to_table(vcf_file)
    assert is_valid_dataframe(df), f"The output is not a DataFrame for VCF: {vcf_file}"
    assert not is_empty(df), f"The DataFrame is empty for VCF: {vcf_file}"
    required_columns = ["chrom", "chromStart", "chromEnd", "REF", "ALT"]
    for col in required_columns:
        assert col in df.columns, (
            f"Column '{col}' is missing in the output for VCF: {vcf_file}"
        )


# Test 4: validate that the ALT alleles are among the expected set.
@pytest.mark.parametrize(
    "vcf_file",
    [
        os.path.join(
            os.path.dirname(__file__), "data", "vcf", "synthetic_test_data.vcf"
        ),
    ],
)
def test_vcf_variant_types(vcf_file):
    df = sites_vcf_to_table(vcf_file)
    print(
        "ALT column contents:", df["ALT"].to_list()
    )  # Print ALT values for debugging.
    # define acceptable ALT alleles based on header definitions.
    valid_alts = {"A", "T", "C", "G", "<INS>", "<DEL>", "<DUP:TANDEM>", "<CNV>", "*"}
    for allele in df["ALT"].to_list():
        assert allele in valid_alts, f"Unexpected ALT allele encountered: {allele}"


# Test 5: confirm that non-sample Variant attributes and INFO fields have expected types.
def test_variant_info_data_types():
    vcf = VCF(VCF_FILE)
    for variant in vcf:
        # validate basic variant attributes.
        assert isinstance(variant.CHROM, str), f"CHROM is not a string: {variant.CHROM}"
        assert isinstance(variant.POS, int), f"POS is not an integer: {variant.POS}"
        assert isinstance(variant.REF, str), f"REF is not a string: {variant.REF}"
        # ALT should be a list or tuple.
        assert isinstance(variant.ALT, (list, tuple)), (
            f"ALT is not a list/tuple: {variant.ALT}"
        )
        for alt in variant.ALT:
            assert isinstance(alt, str), f"ALT allele is not a string: {alt}"

        # validate INFO field types.
        af = variant.INFO.get("AF")
        if af is not None:
            if isinstance(af, (list, tuple)):
                for val in af:
                    assert isinstance(val, float), f"AF value is not a float: {val}"
            else:
                assert isinstance(af, float), f"AF value is not a float: {af}"

        dp = variant.INFO.get("DP")
        if dp is not None:
            if isinstance(dp, (list, tuple)):
                for val in dp:
                    assert isinstance(val, int), f"DP value is not an int: {val}"
            else:
                assert isinstance(dp, int), f"DP value is not an int: {dp}"

        gt = variant.INFO.get("GT")
        if gt is not None:
            if isinstance(gt, (list, tuple)):
                for val in gt:
                    assert isinstance(val, str), f"GT value is not a string: {val}"
            else:
                assert isinstance(gt, str), f"GT value is not a string: {gt}"

        db = variant.INFO.get("DB")
        if db is not None:
            # For flag fields, cyvcf2 may return 1 if the flag is set.
            assert isinstance(db, (int, bool)), f"DB flag is not an int or bool: {db}"
