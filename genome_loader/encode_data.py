import timeit

import h5py
import numpy as np
import pandas as pd

from pysam import FastaFile


def array_to_onehot(seq_array, base_list):
    """
    HELPER CALLED BY encode_sequence()
    """
    seq_array[np.isin(seq_array,
                      [b"A", b"C", b"G", b"T"], invert=True)] = b"N"  # Convert ambiguous

    return pd.get_dummies(seq_array).reindex(
        columns=base_list, fill_value=0).to_numpy()


def parse_encode_list(encode_spec):
    """
    HELPER CALLED BY encode_sequence()
    """
    if not encode_spec:
        encode_spec = [b"A", b"C", b"G", b"T", b"N"]

    elif isinstance(encode_spec, (list, tuple)):
        encode_spec = [base.encode() for base in encode_spec]

    elif isinstance(encode_spec, str):
        encode_spec = [base.encode() for base in list(encode_spec)]

    else:
        raise TypeError("Please input string or list of strings!")

    return encode_spec


def encode_sequence(seq_data, encode_spec=None):
    """Encodes sequence data into one-hot encoded format

    :param seq_data: Sequence data to encode
    :type seq_data: str or numpy char-array
    :param encode_spec: Bases and order to encode, defaults to 'ACGTN'
    :type encode_spec: str or list of bases(str), optional
    :return: One-Hot encoded sequence
    :rtype: np.ndarray
    """

    # Process sequence input
    if isinstance(seq_data, str):  # sequence input as string
        seq_data = np.fromiter(seq_data, count=len(seq_data), dtype="|S1")

    elif isinstance(seq_data, np.ndarray):  # seq data is numpy array
        if seq_data.dtype != "|S1":
            seq_data = seq_data.astype("|S1")

    else:
        raise TypeError("Please input as string or numpy array!")

    encode_spec = parse_encode_list(encode_spec)

    return array_to_onehot(seq_data, encode_spec)


def encode_from_fasta(in_fasta, chrom_list=None, encode_spec=None):
    """Create one-hot encoded data directly from fasta file

    :param in_fasta: Fasta file to encode
    :type in_fasta: str
    :param chrom_list: Chromosomes to encode, defaults to ALL chroms
    :type chrom_list: list of str, optional
    :param encode_spec: Bases and order to encode, defaults to 'ACGTN'
    :type encode_spec: str or list of bases(str), optional
    :return: Dictionary with keys: [chrom] and one-hot encoded data
    :rtype: dict of np.ndarray
    """

    onehot_dict = {}
    start_time = timeit.default_timer()
    print(
        f"Encoding w. Specs: {[base.decode() for base in parse_encode_list(encode_spec)]}")

    with FastaFile(in_fasta) as fasta:

        if not chrom_list:
            chrom_list = fasta.references

        for chrom in chrom_list:
            start_chrom = timeit.default_timer()

            seq_array = np.fromiter(fasta.fetch(chrom),
                                    count=fasta.get_reference_length(chrom),
                                    dtype="|S1")

            onehot_dict[chrom] = encode_sequence(seq_array, encode_spec)

            print(
                f"Encoded {chrom} in {timeit.default_timer() - start_chrom:.2f} seconds!")

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
        f"Encoding w. Specs: {[base.decode() for base in parse_encode_list(encode_spec)]}")

    with h5py.File(in_h5, "r") as file:

        if not chrom_list:
            chrom_list = list(file.keys())

        for chrom in chrom_list:

            if f"{chrom}/sequence" not in file:
                print(f"{chrom} sequence not found!")
                continue

            start_chrom = timeit.default_timer()

            onehot_dict[chrom] = encode_sequence(
                file[chrom]["sequence"][:], encode_spec)

            print(
                f"Encoded {chrom} in {timeit.default_timer() - start_chrom:.2f} seconds!")

        print(
            f"Finished in {timeit.default_timer() - start_time:.2f} seconds!")
        return onehot_dict
