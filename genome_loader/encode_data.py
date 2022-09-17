import timeit
from typing import List

import h5py
import numpy as np
import pandas as pd
import polars as pl
from pysam import FastaFile


def seq_to_onehot(seq_data, encoded_bases=None, out_spec=None):
    """HELPER CALLED BY encode_sequence()
    Converts sequence to onehot array

    :param seq_data: Sequence String
    :type seq_data: str
    :param encoded_bases: List of bases to encode,
        bases not found in list converted to 'N',
        default ["A", "C", "G", "T", "N"]
    :type encoded_bases: list of str, optional
    :param out_spec: THIS TAKES OUTPUT OF parse_encode_spec()!!!
        Output columns of OHE array repr w. base and col-order,
        default value of <encoded_bases> arg
    :type out_spec: dict of str, int, optional
    :return: onehot encoded data
    :rtype: np.ndarray
    """

    if encoded_bases is None:
        encoded_bases = ["A", "C", "G", "T", "N"]
    
    if out_spec is None:
        out_spec = encoded_bases
    else:
        out_spec = [base for base, idx in sorted(out_spec.items(), key=lambda x: x[1])]
    
    
    dummy_df = pl.Series(name="seq", values=seq_data, dtype=pl.Utf8).to_frame().lazy().with_column(
        pl.when(pl.col("seq").is_in(encoded_bases)).then(pl.col("seq")).otherwise("N").alias("seq")
    ).collect().to_dummies()

    # Process cols to rename and spec
    rename_cols = {base:base.rsplit("_")[-1] for base in dummy_df.columns}
    missing_cols = [base for base in out_spec if base not in rename_cols.values()]

    if missing_cols:
        dummy_df = dummy_df.with_columns(
            [pl.lit(0, dtype=pl.UInt8).alias(base) for base in missing_cols])
    
    return dummy_df.rename(rename_cols).select(out_spec).to_numpy()


def array_to_onehot(seq_array, encoded_bases=None, out_spec=None):
    """HELPER CALLED BY encode_sequence()
    Converts array data to onehot array

    :param seq_array: Sequence represented as an array of bytes
    :type seq_array: np.ndarray[byte]
    :param encoded_bases: List of bases to encode,
        bases not found in list converted to 'N',
        default ["A", "C", "G", "T", "N"]
    :type encoded_bases: list of str, optional
    :param out_spec: THIS TAKES OUTPUT OF parse_encode_spec()!!!
        Output columns of OHE array repr w. base and col-order,
        default value of <encoded_bases> arg
    :type out_spec: dict of str, int, optional
    :return: onehot encoded data
    :rtype: np.ndarray
    """

    if encoded_bases is None:
        encoded_bases = ["A", "C", "G", "T", "N"]
    
    encoded_bases = [base.encode() for base in encoded_bases] # encode to bytes
    
    if out_spec is None:
        out_spec = encoded_bases
    else:
        # Convert spec dict of base, idx to ordered byte list
        out_spec = [base.encode() for base, idx in sorted(out_spec.items(), key=lambda x: x[1])]

    seq_array[np.isin(seq_array, encoded_bases, invert=True)] = b"N"  # Convert ambiguous

    return pd.get_dummies(seq_array).reindex(columns=out_spec, fill_value=0).to_numpy()


def parse_encode_spec(encode_spec):
    """
    HELPER CALLED BY encode_sequences()
    """
    if not encode_spec:
        encode_spec = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    elif isinstance(encode_spec, (list, tuple, str)):
        encode_spec = {base: i for i, base in enumerate(encode_spec)}

    elif not isinstance(encode_spec, dict):
        raise TypeError("Please input as dict, list or string!")

    return encode_spec


def encode_sequence(seq_data, encoded_bases=None, encode_spec=None, ignore_case=True, engine=None):
    """Encodes sequence data into one-hot encoded format

    :param seq_data: Sequence data to encode
    :type seq_data: str or numpy char-array
    :param encoded_bases: List of bases to encode,
        bases not found in list converted to 'N',
        default ["A", "C", "G", "T", "N"]
    :type encoded_bases: list of str, optional
    :param encode_spec: Base order and shape of output columns.
        Bases in encode_spec DO NOT need to match encode_bases.
        Encoded matrix will always be shaped according to encode_spec, 
        regardless of whether or not base is found, defaults to 'ACGTN' (5 col matrix)
    :type encode_spec: dict with base, pos [str, int], str, or list of bases(str), optional
    :param ignore_case: Convert lowercase bases to upper, default True
    :type ignore_case: bool, optional
    :param engine: Encoding Engine, {pandas or polars} default None
    :type engine: str, optional
    :return: One-Hot encoded sequence
    :rtype: np.ndarray
    """

    # Process sequence input
    if isinstance(seq_data, str):  # sequence input as string
        if ignore_case:
            seq_data = seq_data.upper()
        
        if engine != "polars":
            seq_data = np.fromiter(seq_data, count=len(seq_data), dtype="|S1")

    elif isinstance(seq_data, np.ndarray):  # seq data is numpy array
        if seq_data.dtype != "|S1":
            seq_data = seq_data.astype("|S1")

        if ignore_case:
            # Much faster to convert upper as string
            seq_data = np.char.upper(seq_data)
        
        engine = "pandas" # array input not supported in polars

    else:
        raise TypeError("Please input as string or numpy array!")

    encode_spec = parse_encode_spec(encode_spec)

    if engine == "polars":
        ohe_sequence = seq_to_onehot(seq_data, encoded_bases=encoded_bases, out_spec=encode_spec)
    else:
        ohe_sequence = array_to_onehot(seq_data, encoded_bases=encoded_bases, out_spec=encode_spec)
    
    return ohe_sequence


def encode_from_fasta(in_fasta, chrom_list=None, encode_spec=None, ignore_case=True):
    """Create one-hot encoded data directly from fasta file

    :param in_fasta: Fasta file to encode
    :type in_fasta: str
    :param chrom_list: Chromosomes to encode, defaults to ALL chroms
    :type chrom_list: list of str, optional
    :param encode_spec: Bases and order to encode, defaults to 'ACGTN'
    :type encode_spec: str or list of bases(str), optional
    :param ignore_case: Convert lowercase bases to upper, default True
    :type ignore_case: bool, optional
    :return: Dictionary with keys: [chrom] and one-hot encoded data
    :rtype: dict of np.ndarray
    """

    onehot_dict = {}
    start_time = timeit.default_timer()
    print(
        f"Encoding w. Specs: {parse_encode_spec(encode_spec)}"
    )

    with FastaFile(in_fasta) as fasta:

        if not chrom_list:
            chrom_list = fasta.references

        for chrom in chrom_list:
            start_chrom = timeit.default_timer()

            # Update to ignore lowercase as string
            if ignore_case:
                fasta_seq = fasta.fetch(chrom).upper()
            else:
                fasta_seq = fasta.fetch(chrom)

            onehot_dict[chrom] = encode_sequence(
                fasta_seq, encode_spec=encode_spec, ignore_case=ignore_case, engine="polars"
            )

            print(
                f"Encoded {chrom} in {timeit.default_timer() - start_chrom:.2f} seconds!"
            )

    print(f"Finished in {timeit.default_timer() - start_time:.2f} seconds!")
    return onehot_dict


def encode_from_h5(in_h5, chrom_list=None, encode_spec=None):
    """Create one-hot encoded data from char-array encoded H5

    :param in_h5: Char-Encoded H5 created using 'writefasta' command
    :type in_h5: str
    :param chrom_list: Chromosomes to encode, defaults to ALL chroms
    :type chrom_list: list of str, optional
    :param encode_spec: Bases and order to encode, defaults to 'ACGTN'
    :type encode_spec: str or list of bases(str), optional
    :return: Dictionary with keys: [chrom] and one-hot encoded data
    :rtype: dict of np.ndarray
    """

    if not h5py.is_hdf5(in_h5):
        raise ValueError("File is not valid HDF5")

    onehot_dict = {}
    start_time = timeit.default_timer()
    print(
        f"Encoding w. Specs: {parse_encode_spec(encode_spec)}"
    )

    with h5py.File(in_h5, "r") as file:

        if not chrom_list:
            chrom_list = list(file.keys())

        for chrom in chrom_list:

            if f"{chrom}/sequence" not in file:
                print(f"{chrom} sequence not found!")
                continue

            start_chrom = timeit.default_timer()

            onehot_dict[chrom] = encode_sequence(
                file[chrom]["sequence"][:], encode_spec=encode_spec
            )

            print(
                f"Encoded {chrom} in {timeit.default_timer() - start_chrom:.2f} seconds!"
            )

        print(f"Finished in {timeit.default_timer() - start_time:.2f} seconds!")
        return onehot_dict
