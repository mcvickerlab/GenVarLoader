from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from genome_loader.encode_data import encode_from_fasta
from genome_loader.get_encoded import get_encoded_haps, validate_snp_chroms
from genome_loader.load_data import load_vcf, load_vcf_polars


@pytest.fixture()
def vcf_file():
    data_path = Path(__file__).parent / "data"
    vcf_file = str(data_path / "vcf" / "NA12878_chr20_21.vcf.gz")
    return vcf_file


@pytest.fixture(scope="module")
def snp_df():
    data_path = Path(__file__).parent / "data"
    vcf_file = str(data_path / "vcf" / "NA12878_chr20_21.vcf.gz")
    return load_vcf(vcf_file, sample="NA12878")


@pytest.fixture(scope="module")
def snp_df_no_prefix(snp_df):
    no_prefix = snp_df.copy()
    no_prefix["chrom"] = no_prefix["chrom"].str.replace("chr", "", regex=False)
    return no_prefix


def acgtn_ohe():
    return np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )


def acgtn_ohe_no_n():
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    )


# If num cols in chroms don't match
@pytest.mark.exceptions
def test_exception_value_error(vcf_file):
    ohe_mixed_shape = {"20": acgtn_ohe(), "21": acgtn_ohe_no_n()}

    with pytest.raises(ValueError) as exc_info:
        get_encoded_haps(ohe_mixed_shape, vcf_file, "NA12878")


# If num cols and spec don't match
@pytest.mark.exceptions
@pytest.mark.parametrize(
    "ohe_array, test_spec", [(acgtn_ohe(), "ACGT"), (acgtn_ohe_no_n(), "ACGTN")]
)
def test_exception_index_error(ohe_array, test_spec, vcf_file):
    ohe_dict = {"20": ohe_array, "21": ohe_array}

    with pytest.raises(IndexError):
        get_encoded_haps(ohe_dict, vcf_file, "NA12878", encode_spec=test_spec)


# If no matches between vcf and encoding
@pytest.mark.exceptions
def test_exception_key_error(vcf_file):
    ohe_dict = {"20": acgtn_ohe(), "21": acgtn_ohe()}

    with pytest.raises(KeyError):
        get_encoded_haps(ohe_dict, vcf_file, "NA12878", chrom_list=["chr15"])


# TEST CHROMOSOME VALIDATION
@pytest.mark.validate_chroms
@pytest.mark.parametrize(
    "bam_chroms",
    [
        (["chr20", "chr21"]),
        (["chr20", "chr21", "chr22"]),
    ],
)
def test_validate_prefix(snp_df, bam_chroms):
    assert snp_df.equals(validate_snp_chroms(snp_df.copy(), bam_chroms))


# TEST CHROMOSOME VALIDATION
@pytest.mark.validate_chroms
@pytest.mark.parametrize(
    "bam_chroms",
    [
        (["20", "21"]),
        (["19", "20", "21"]),
    ],
)
def test_validate_noprefix(snp_df_no_prefix, bam_chroms):
    assert snp_df_no_prefix.equals(
        validate_snp_chroms(snp_df_no_prefix.copy(), bam_chroms)
    )


@pytest.mark.validate_chroms
@pytest.mark.parametrize(
    "bam_chroms", [(["chr20", "chr21"]), (["chr20", "chr21", "chr22"])]
)
def test_validate_add_prefix(snp_df, snp_df_no_prefix, bam_chroms):
    assert snp_df.equals(validate_snp_chroms(snp_df_no_prefix.copy(), bam_chroms))


@pytest.mark.validate_chroms
@pytest.mark.parametrize(
    "bam_chroms",
    [
        (["20", "21"]),
        (["19", "20", "21"]),
    ],
)
def test_validate_remove_prefix(snp_df, snp_df_no_prefix, bam_chroms):
    assert snp_df_no_prefix.equals(validate_snp_chroms(snp_df.copy(), bam_chroms))


@pytest.mark.validate_chroms
def test_validate_skip_chroms(snp_df):
    bam_chroms = [f"chr{i}" for i in range(1, 21)]  # chr1-20

    # Only chr20, no chr21
    snp_df_bam_filt = snp_df.loc[snp_df["chrom"].isin(bam_chroms), :]

    # Parsed with validator
    snp_df_validated = validate_snp_chroms(snp_df.copy(), bam_chroms)

    assert snp_df_bam_filt.equals(snp_df_validated)


# TEST HAPLOTYPE GENERATION
@pytest.fixture(scope="module")
def encode_dict_grch38():
    data_path = Path(__file__).parent / "data"
    fasta_file = str(data_path / "fasta" / "grch38.20.21.fa.gz")
    return encode_from_fasta(fasta_file, chrom_list=["20"])


@pytest.fixture(scope="module")
def encode_dict_grch38_no_n():
    data_path = Path(__file__).parent / "data"
    fasta_file = str(data_path / "fasta" / "grch38.20.21.fa.gz")
    return encode_from_fasta(fasta_file, chrom_list=["20"], encode_spec="ACGT")


@pytest.mark.encode_haps
def test_hap_encoding_sample(encode_dict_grch38, vcf_file, snp_df):

    # Create Haplotype Encoding to be tested
    hap1, hap2 = get_encoded_haps(
        encode_dict_grch38, vcf_file, "NA12878", chrom_list="chr20"
    )

    # Sample 5 random rows from chr20
    snp_df_chr20 = snp_df.loc[snp_df["chrom"] == "chr20", :]
    sample_rows = snp_df_chr20.sample(5)
    sample_locs = sample_rows["start"].to_numpy()

    allele_dict = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    sample_rows = sample_rows.replace(
        {"ref": allele_dict, "alt": allele_dict}
    )  # Replace data

    # Get positions for updated haps
    p1_cols = np.where(
        sample_rows["phase1"] == 1, sample_rows["alt"], sample_rows["ref"]
    )
    p2_cols = np.where(
        sample_rows["phase2"] == 1, sample_rows["alt"], sample_rows["ref"]
    )

    # Create new Haplotype 1
    hap1_seq = encode_dict_grch38["20"][sample_locs].copy()
    hap1_seq[:] = 0

    hap2_seq = hap1_seq.copy()  # Create new Haplotype 2

    # Update New values in haplotypes
    hap1_seq[[0, 1, 2, 3, 4], p1_cols] = 1
    hap2_seq[[0, 1, 2, 3, 4], p2_cols] = 1

    # Check manual against encoded_haps
    assert np.array_equal(hap1_seq, hap1["20"][sample_locs]) and np.array_equal(
        hap2_seq, hap2["20"][sample_locs]
    )


# TESTS FOR remove_ambiguous / N HANDLING
@pytest.fixture()
def vcf_with_ns():
    data_path = Path(__file__).parent / "data"
    return str(data_path / "vcf" / "NA12878_chr20_with_Ns.vcf.gz")


@pytest.fixture(scope="module")
def snp_df_with_ns():
    data_path = Path(__file__).parent / "data"
    n_vcf_file = str(data_path / "vcf" / "NA12878_chr20_with_Ns.vcf.gz")
    return load_vcf(n_vcf_file, sample="NA12878")


def n_vcf_starts():
    return np.array(
        [
            7373672,
            18616601,
            19084614,
            22196219,
            26102638,
            30668586,
            31394973,
            34136538,
            45760706,
            49323597,
            50225789,
            50706183,
            52906392,
            53363556,
            55707141,
        ]
    )


def n_vcf_og_seq():
    return np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )


def n_phase1_idx():
    return np.array([4, 0, 4, 2, 0, 1, 0, 3, 0, 3, 2, 0, 4, 1, 4])


def n_phase2_idx():
    return np.array([4, 0, 0, 1, 4, 0, 0, 3, 0, 3, 4, 3, 0, 0, 4])


@pytest.mark.encode_ambiguity
def test_hap_remove_ambiguity(encode_dict_grch38, vcf_with_ns):

    # Create Data to test
    test_dict1, test_dict2 = get_encoded_haps(
        encode_dict_grch38, vcf_with_ns, "NA12878", remove_ambiguity=True
    )

    test_hap1 = test_dict1["20"][n_vcf_starts()]
    test_hap2 = test_dict2["20"][n_vcf_starts()]

    # Manually create haps
    hap1 = n_vcf_og_seq()
    hap2 = n_vcf_og_seq()

    # Create idx data to change
    non_n_rows = np.arange(1, 14, 2)
    non_n_phase1_idx = n_phase1_idx()[1::2]
    non_n_phase2_idx = n_phase2_idx()[1::2]

    # Update hap1
    hap1[1::2] = 0
    hap1[non_n_rows, non_n_phase1_idx] = 1

    # Update hap2
    hap2[1::2] = 0
    hap2[non_n_rows, non_n_phase2_idx] = 1

    assert np.array_equal(hap1, test_hap1) and np.array_equal(hap2, test_hap2)


@pytest.mark.encode_ambiguity
def test_hap_keep_ambiguity(encode_dict_grch38, vcf_with_ns):

    # Create Data to test
    test_dict1, test_dict2 = get_encoded_haps(
        encode_dict_grch38, vcf_with_ns, "NA12878", remove_ambiguity=False
    )

    test_hap1 = test_dict1["20"][n_vcf_starts()]
    test_hap2 = test_dict2["20"][n_vcf_starts()]

    # Manually create haps
    hap1 = n_vcf_og_seq()
    hap2 = n_vcf_og_seq()

    # Update hap1
    hap1[:] = 0
    hap1[np.arange(0, 15), n_phase1_idx()] = 1

    # Update hap2
    hap2[:] = 0
    hap2[np.arange(0, 15), n_phase2_idx()] = 1

    assert np.array_equal(hap1, test_hap1) and np.array_equal(hap2, test_hap2)


@pytest.mark.encode_ambiguity
def test_hap_ambiguity_acgt_spec(encode_dict_grch38_no_n, vcf_with_ns):

    # Create Data to test
    test_dict1, test_dict2 = get_encoded_haps(
        encode_dict_grch38_no_n,
        vcf_with_ns,
        "NA12878",
        encode_spec="ACGT",
        remove_ambiguity=False,
    )

    test_hap1 = test_dict1["20"][n_vcf_starts()]
    test_hap2 = test_dict2["20"][n_vcf_starts()]

    # Manually create haps
    hap1 = n_vcf_og_seq()[:, :-1]
    hap2 = n_vcf_og_seq()[:, :-1]

    # Create idx data to change
    non_n_rows = np.arange(1, 14, 2)
    non_n_phase1_idx = n_phase1_idx()[1::2]
    non_n_phase2_idx = n_phase2_idx()[1::2]

    # Update hap1
    hap1[1::2] = 0
    hap1[non_n_rows, non_n_phase1_idx] = 1

    # Update hap2
    hap2[1::2] = 0
    hap2[non_n_rows, non_n_phase2_idx] = 1

    assert np.array_equal(hap1, test_hap1) and np.array_equal(hap2, test_hap2)
